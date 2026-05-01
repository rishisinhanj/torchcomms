#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
RDMA vs TCP GPU-tensor send bandwidth benchmark.

Two-rank (rank 0 = sender, rank 1 = receiver), two-node. Same warmup +
measured-loop shape for both transports so per-iter numbers are directly
comparable. Writes one JSONL record per (mode, size, repeat) to
results/run-<timestamp>/raw.jsonl. Post-process with pandas/awk.

Launch:
  torchrun --nnodes=2 --nproc-per-node=1 --node-rank=<0|1> \\
           --master-addr=<rank0> --master-port=29500 \\
           transport_bench.py [--mode rdma|tcp|all] [--sizes ...] [--repeats N]

Required env: MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE (all set by torchrun).
For TCP path: TCP_BIND_IP (this node's NIC IP), TCP_PEER_IP (rank 1 dials this).
For NIC pinning: HIP_VISIBLE_DEVICES, NCCL_IB_HCA (set in launcher).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pickle
import socket
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


SIZES = (4 << 10, 64 << 10, 1 << 20, 16 << 20, 64 << 20, 256 << 20)
WARMUP_S = 0.250
WARMUP_CAP = 200


def iters_for(nbytes: int) -> int:
    if nbytes <= 64 << 10:
        return 1000
    if nbytes <= 4 << 20:
        return 500
    return 200


def percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def barrier(store: dist.TCPStore, rank: int, tag: str) -> None:
    store.set(f"bar_{tag}_{rank}", b"1")
    store.get(f"bar_{tag}_{1 - rank}")


def make_record(mode: str, size: int, repeat: int, per_iter: list[float],
                warmup_iters: int, csum_ok: bool) -> dict:
    med = statistics.median(per_iter)
    return {
        "mode": mode,
        "size_bytes": size,
        "repeat": repeat,
        "iters": len(per_iter),
        "warmup_iters": warmup_iters,
        "median_iter_s": med,
        "p50_us": percentile(per_iter, 0.5) * 1e6,
        "p99_us": percentile(per_iter, 0.99) * 1e6,
        "bandwidth_gbytes_per_s": (size / med) / 1e9,
        "bandwidth_gbits_per_s": (size * 8 / med) / 1e9,
        "checksum_ok": csum_ok,
    }


# =========================================================================
# RDMA section
# =========================================================================


def setup_rdma(rank: int, store: dist.TCPStore, device: torch.device):
    from torchcomms._transport import RdmaTransport
    if not RdmaTransport.supported():
        raise RuntimeError("RdmaTransport.supported() is False")
    t = RdmaTransport(device)
    store.set(f"rdma_url_{rank}", t.bind())
    rc = t.connect(store.get(f"rdma_url_{1 - rank}"))
    if rc != 0 or not t.connected():
        raise RuntimeError(f"RdmaTransport.connect rc={rc}")
    return t


def run_rdma(rank: int, transport, store: dist.TCPStore,
             device: torch.device, size: int, repeat: int) -> dict | None:
    """RDMA WRITE bench. Per-iter transport.write() is sync-on-completion
    because the Python binding does .get() on the SemiFuture internally
    (see RdmaTransportPy.cpp:73-74)."""
    from torchcomms._transport import RdmaMemory

    ns = f"{size}_{repeat}"
    iters = iters_for(size)

    if rank == 0:
        local = torch.arange(size, dtype=torch.uint8, device=device) % 251
        local_mem = RdmaMemory(local)
    else:
        dest = torch.zeros(size, dtype=torch.uint8, device=device)
        dest_mem = RdmaMemory(dest)
        store.set(f"rdma_remote_{ns}", pickle.dumps(dest_mem.to_remote_buffer()))

    if rank == 0:
        remote = pickle.loads(store.get(f"rdma_remote_{ns}"))
        view = local_mem.to_view()

    # Warmup (time-bounded, iter-capped)
    warmup = 0
    t0 = time.perf_counter()
    while True:
        if rank == 0:
            transport.write(view, remote)
        warmup += 1
        if warmup >= WARMUP_CAP or (time.perf_counter() - t0) >= WARMUP_S:
            break
    barrier(store, rank, f"rdma_w_{ns}")

    # Measured (rank 0 times; rank 1 idles — one-sided WRITE)
    per_iter: list[float] = []
    for _ in range(iters):
        if rank == 0:
            ti = time.perf_counter()
            transport.write(view, remote)
            per_iter.append(time.perf_counter() - ti)
    barrier(store, rank, f"rdma_m_{ns}")

    # Checksum: rank 1 verifies, reports via store
    if rank == 1:
        ok = bool(torch.equal(dest.detach().cpu(),
                              torch.arange(size, dtype=torch.uint8) % 251))
        store.set(f"rdma_csum_{ns}", b"1" if ok else b"0")
        return None
    csum_ok = store.get(f"rdma_csum_{ns}") == b"1"
    return make_record("rdma", size, repeat, per_iter, warmup, csum_ok)


# =========================================================================
# TCP section
# =========================================================================


def setup_tcp(rank: int, store: dist.TCPStore, bind_ip: str | None,
              peer_ip: str | None, port: int) -> socket.socket:
    if rank == 0:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((bind_ip or "0.0.0.0", port))
        srv.listen(1)
        store.set("tcp_addr", f"{bind_ip or ''}|{port}".encode())
        sock, _ = srv.accept()
        srv.close()
    else:
        host_str, port_str = store.get("tcp_addr").decode().split("|")
        host = peer_ip or host_str
        if not host:
            raise RuntimeError("rank 1 needs TCP_PEER_IP (or rank 0 needs TCP_BIND_IP)")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if bind_ip:
            sock.bind((bind_ip, 0))
        sock.connect((host, int(port_str)))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 << 20)
    return sock


def recv_exact(sock: socket.socket, n: int, into: memoryview) -> None:
    got = 0
    while got < n:
        chunk = sock.recv_into(into[got:n], n - got)
        if chunk == 0:
            raise ConnectionError("peer closed mid-message")
        got += chunk


def run_tcp(rank: int, sock: socket.socket, store: dist.TCPStore,
            device: torch.device, size: int, repeat: int) -> dict | None:
    """TCP bench: bounce buffer cost (D<->H copy + cuda.sync + send/recv)
    is intentionally inside the timed region. That's what TCP for a GPU
    tensor actually costs."""
    ns = f"{size}_{repeat}"
    iters = iters_for(size)
    host_buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    if rank == 0:
        gpu = (torch.arange(size, dtype=torch.uint8, device=device) % 251).contiguous()
    else:
        gpu = torch.zeros(size, dtype=torch.uint8, device=device)

    def step() -> None:
        if rank == 0:
            host_buf.copy_(gpu)
            torch.cuda.synchronize(device)
            sock.sendall(memoryview(host_buf.numpy()))
        else:
            recv_exact(sock, size, memoryview(host_buf.numpy()))
            gpu.copy_(host_buf)
            torch.cuda.synchronize(device)

    # Warmup: TCP is two-sided so both ranks must run the SAME number of
    # iters or the receiver exits early, the sender's sendall blocks on
    # backpressure, and the warmup barrier deadlocks. Use a fixed small
    # iter count instead of the time-bounded loop the RDMA path uses.
    warmup = min(20, iters)
    for _ in range(warmup):
        step()
    barrier(store, rank, f"tcp_w_{ns}")

    # Measured
    per_iter: list[float] = []
    for _ in range(iters):
        ti = time.perf_counter()
        step()
        per_iter.append(time.perf_counter() - ti)
    barrier(store, rank, f"tcp_m_{ns}")

    # Checksum
    if rank == 1:
        ok = bool(torch.equal(gpu.detach().cpu(),
                              torch.arange(size, dtype=torch.uint8) % 251))
        store.set(f"tcp_csum_{ns}", b"1" if ok else b"0")
        return None
    csum_ok = store.get(f"tcp_csum_{ns}") == b"1"
    return make_record("tcp", size, repeat, per_iter, warmup, csum_ok)


# =========================================================================
# Main
# =========================================================================


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", nargs="+", choices=["rdma", "tcp", "all"], default=["all"])
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output-dir", type=Path, default=Path("./results"))
    p.add_argument("--device", default=os.environ.get("BENCH_DEVICE", "cuda:0"))
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise RuntimeError(f"two-rank only; got world_size={world}")
    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ["MASTER_PORT"])
    aux_port = master_port + 1
    tcp_port = master_port + 100

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    modes = ["rdma", "tcp"] if "all" in args.mode else list(args.mode)

    store = dist.TCPStore(master_addr, aux_port, world, is_master=(rank == 0),
                          timeout=_dt.timedelta(seconds=120))

    # Output dir (rank 0 mints, rank 1 reads via store)
    if rank == 0:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = args.output_dir / f"run-{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        store.set("run_dir", str(run_dir).encode())
        raw_fp = (run_dir / "raw.jsonl").open("a")
    else:
        run_dir = Path(store.get("run_dir").decode())
        raw_fp = None

    print(f"[rank {rank}] NIC pinning: NCCL_IB_HCA={os.environ.get('NCCL_IB_HCA','-')} "
          f"TCP_BIND_IP={os.environ.get('TCP_BIND_IP','-')} "
          f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES','-')} "
          f"-- verify these map to the same physical port", flush=True)

    for mode in modes:
        try:
            if mode == "rdma":
                tr = setup_rdma(rank, store, device)
                runner = lambda sz, rep: run_rdma(rank, tr, store, device, sz, rep)
            else:  # tcp
                sk = setup_tcp(rank, store, os.environ.get("TCP_BIND_IP"),
                               os.environ.get("TCP_PEER_IP"), tcp_port)
                runner = lambda sz, rep: run_tcp(rank, sk, store, device, sz, rep)
        except Exception as e:
            print(f"[rank {rank}] {mode} setup failed: {e!r}", flush=True)
            continue

        for size in args.sizes:
            for repeat in range(args.repeats):
                try:
                    rec = runner(size, repeat)
                except Exception as e:
                    print(f"[rank {rank}] {mode} size={size} rep={repeat} failed: {e!r}", flush=True)
                    continue
                if raw_fp and rec is not None:
                    raw_fp.write(json.dumps(rec) + "\n")
                    raw_fp.flush()
                    print(f"[rank 0] {mode:4s} size={size:>10d} rep={repeat} "
                          f"bw={rec['bandwidth_gbytes_per_s']:7.3f} GB/s "
                          f"p99={rec['p99_us']:7.1f}us "
                          f"csum={'OK' if rec['checksum_ok'] else 'BAD'}", flush=True)
        barrier(store, rank, f"done_{mode}")

    if raw_fp:
        raw_fp.close()
        print(f"[rank 0] wrote {run_dir/'raw.jsonl'}", flush=True)

    # Asymmetric goodbye: non-server sets, server gets last
    if rank != 0:
        store.set(f"bye_{rank}", b"bye")
    else:
        for peer in range(1, world):
            store.get(f"bye_{peer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
