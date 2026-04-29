#!/usr/bin/env python3
"""Two-node ROCm RDMA validation for torchcomms._transport.

Launch with torchrun using one process per node. torchrun/TCPStore are used only
for rendezvous and metadata exchange; tensor data moves through RdmaTransport.
"""

from __future__ import annotations

import argparse
import base64
import os
import pickle
import socket
import subprocess
import sys
from datetime import timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TORCHCOMMS_DIR = Path(os.environ.get("TORCHCOMMS_DIR", ROOT.parents[5])).resolve()
DEPS_PREFIX = Path(
    os.environ.get("DEPS_PREFIX", TORCHCOMMS_DIR / ".deps" / "transport" / "prefix")
)
ROCM_HOME = os.environ.get("ROCM_HOME", "/opt/rocm")


def configure_env() -> None:
    os.environ["DEPS_PREFIX"] = str(DEPS_PREFIX)
    os.environ["ROCM_HOME"] = ROCM_HOME
    os.environ["USE_SYSTEM_LIBS"] = "1"
    os.environ["CMAKE_PREFIX_PATH"] = str(DEPS_PREFIX)
    os.environ["PKG_CONFIG_PATH"] = str(DEPS_PREFIX / "lib" / "pkgconfig")
    os.environ["LD_LIBRARY_PATH"] = (
        f"{DEPS_PREFIX}/lib:{ROCM_HOME}/lib"
        f"{':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''}"
    )
    os.environ["PATH"] = (
        f"{Path(sys.executable).parent}:{DEPS_PREFIX}/bin:{os.environ.get('PATH', '')}"
    )


def reexec_with_env() -> None:
    """Restart once so LD_LIBRARY_PATH is active before importing torchcomms."""
    if os.environ.get("ROCM_TRANSPORT_ENV_READY") == "1":
        return
    configure_env()
    os.environ["ROCM_TRANSPORT_ENV_READY"] = "1"
    python = os.environ.get("PYTHON", sys.executable)
    os.execve(python, [python, *sys.argv], os.environ.copy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate cross-node RDMA write/read with torchcomms._transport."
    )
    parser.add_argument("--num-bytes", type=int, default=1 << 20)
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--store-timeout-seconds", type=int, default=300)
    parser.add_argument("--print-rdma-devices", action="store_true")
    return parser.parse_args()


def get_torchrun_env() -> tuple[int, int, str, int, int]:
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        master_addr = os.environ["MASTER_ADDR"]
        master_port = int(os.environ.get("RDMA_STORE_PORT", int(os.environ["MASTER_PORT"]) + 1))
    except KeyError as exc:
        raise SystemExit(
            f"Missing torchrun environment variable {exc.args[0]!r}. "
            "Launch this script with torchrun. RDMA_STORE_PORT is optional "
            "and defaults to MASTER_PORT + 1."
        ) from exc
    return rank, world_size, master_addr, master_port, local_rank


def store_bytes(store, key: str, value: bytes) -> None:
    store.set(key, base64.b64encode(value).decode("ascii"))


def load_bytes(store, key: str) -> bytes:
    return base64.b64decode(bytes(store.get(key)))


def print_environment(rank: int, local_rank: int, device_index: int) -> None:
    import torch
    import torchcomms._transport as transport

    print(f"[rank {rank}] host={socket.gethostname()}", flush=True)
    print(f"[rank {rank}] python={sys.executable}", flush=True)
    print(f"[rank {rank}] torch={torch.__version__}", flush=True)
    print(f"[rank {rank}] torch_hip={getattr(torch.version, 'hip', None)}", flush=True)
    print(f"[rank {rank}] cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"[rank {rank}] device_count={torch.cuda.device_count()}", flush=True)
    print(f"[rank {rank}] local_rank={local_rank} device_index={device_index}", flush=True)
    if torch.cuda.is_available():
        print(
            f"[rank {rank}] device_name={torch.cuda.get_device_name(device_index)}",
            flush=True,
        )
    print(f"[rank {rank}] transport_module={transport.__file__}", flush=True)
    print(f"[rank {rank}] rdma_supported={transport.RdmaTransport.supported()}", flush=True)


def print_rdma_devices(rank: int) -> None:
    print(f"[rank {rank}] == ibv_devices ==", flush=True)
    subprocess.run(["ibv_devices"], check=False)
    print(f"[rank {rank}] == rdma link ==", flush=True)
    subprocess.run(["rdma", "link"], check=False)


def make_pattern(num_bytes: int, rank: int, device: str):
    import torch

    return (torch.arange(num_bytes, dtype=torch.uint8, device=device) + (rank * 17)) % 251


def verify_tensor(actual, expected, label: str, rank: int) -> None:
    import torch

    if not torch.equal(actual, expected):
        mismatch = (actual != expected).nonzero()
        first = int(mismatch[0].item()) if mismatch.numel() else -1
        raise AssertionError(
            f"[rank {rank}] {label} mismatch at byte {first}: "
            f"expected={int(expected[first].item()) if first >= 0 else 'n/a'} "
            f"actual={int(actual[first].item()) if first >= 0 else 'n/a'}"
        )
    checksum = int(actual.to(torch.int64).sum().item())
    print(f"[rank {rank}] {label} verified checksum={checksum}", flush=True)


def two_phase_barrier(store, rank: int, peer: int, name: str) -> None:
    store.set(f"{name}_arrived_{rank}", "1")
    store.get(f"{name}_arrived_{peer}")
    store.set(f"{name}_ack_{rank}", "1")
    store.get(f"{name}_ack_{peer}")


def main() -> int:
    reexec_with_env()
    configure_env()
    args = parse_args()

    rank, world_size, master_addr, master_port, local_rank = get_torchrun_env()
    if world_size != 2:
        raise SystemExit(f"Expected WORLD_SIZE=2, got {world_size}")

    import torch
    from torch.distributed import TCPStore
    from torchcomms._transport import RdmaMemory, RdmaTransport

    device_index = args.device_index if args.device_index is not None else local_rank
    if not torch.cuda.is_available():
        raise SystemExit(f"[rank {rank}] torch.cuda.is_available() is false")
    if device_index >= torch.cuda.device_count():
        raise SystemExit(
            f"[rank {rank}] device_index {device_index} >= device_count {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"

    print_environment(rank, local_rank, device_index)
    if args.print_rdma_devices:
        print_rdma_devices(rank)

    if not RdmaTransport.supported():
        raise SystemExit(f"[rank {rank}] RdmaTransport is not supported")

    store = TCPStore(
        master_addr,
        master_port,
        world_size,
        rank == 0,
        timedelta(seconds=args.store_timeout_seconds),
    )
    peer = 1 - rank

    rdma = RdmaTransport(torch.device(device))
    local_bind = rdma.bind()
    store_bytes(store, f"bind_{rank}", local_bind)
    peer_bind = load_bytes(store, f"bind_{peer}")
    connect_result = rdma.connect(peer_bind)
    if connect_result != 0:
        raise RuntimeError(f"[rank {rank}] connect({peer}) returned {connect_result}")
    print(f"[rank {rank}] connected_to={peer}", flush=True)

    recv_tensor = torch.zeros(args.num_bytes, dtype=torch.uint8, device=device)
    send_tensor = make_pattern(args.num_bytes, rank, device)
    recv_mem = RdmaMemory(recv_tensor)
    send_mem = RdmaMemory(send_tensor)

    store_bytes(store, f"remote_buffer_{rank}", pickle.dumps(recv_mem.to_remote_buffer()))
    peer_remote_buffer = pickle.loads(load_bytes(store, f"remote_buffer_{peer}"))

    write_result = rdma.write(send_mem.to_view(), peer_remote_buffer)
    if write_result != 0:
        raise RuntimeError(f"[rank {rank}] write({peer}) returned {write_result}")
    two_phase_barrier(store, rank, peer, "write_done")
    torch.cuda.synchronize(device_index)

    expected_from_peer = make_pattern(args.num_bytes, peer, device)
    verify_tensor(recv_tensor, expected_from_peer, "peer_write", rank)

    read_tensor = torch.zeros(args.num_bytes, dtype=torch.uint8, device=device)
    read_mem = RdmaMemory(read_tensor)
    store_bytes(store, f"read_source_{rank}", pickle.dumps(send_mem.to_remote_buffer()))
    peer_read_source = pickle.loads(load_bytes(store, f"read_source_{peer}"))
    read_result = rdma.read(read_mem.to_mutable_view(), peer_read_source)
    if read_result != 0:
        raise RuntimeError(f"[rank {rank}] read({peer}) returned {read_result}")
    torch.cuda.synchronize(device_index)
    verify_tensor(read_tensor, expected_from_peer, "peer_read", rank)

    two_phase_barrier(store, rank, peer, "final_done")
    print(f"[rank {rank}] multinode RDMA transport validation PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
