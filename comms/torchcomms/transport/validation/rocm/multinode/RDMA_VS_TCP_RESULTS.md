# RDMA vs TCP single-tensor send — AMD MI355X + Mellanox RoCE

Bandwidth + latency comparison between `torchcomms._transport.RdmaTransport`
(one-sided RDMA WRITE) and a barebones host-bounce TCP send for the same
GPU tensor. Two ranks, **two physical nodes**, same GPU index, same fabric.

## Setup

| | |
|---|---|
| Branch | `rishisinhanj/rocm-multinode-transport-validation` @ `56220933` |
| Driver | `transport_bench.py` + `run_transport_bench.sh` (this directory) |
| Rank 0 | `mia1-p01-g05` (alias `mi355x-p01-g05`), GPU 4, AMD Instinct MI355X (gfx950) |
| Rank 1 | `mia1-p01-g06` (alias `mi355x-p01-g06`), GPU 4, AMD Instinct MI355X (gfx950) |
| Torch | `2.11.0+rocm7.1`, HIP `7.1.52802` |
| Python | `/home/rishi.sinha@amd.com/miniconda3/envs/torchcomms-rocm64/bin/python` (3.10) |
| RDMA fabric | Mellanox RoCE, NIC autopicked by `ctran` (DMABUF disabled → `ibv_reg_mr` slow path) |
| TCP path | tw-eth4 point-to-point: `10.101.5.141` (g05) ↔ `10.101.6.141` (g06) |
| Rendezvous | `MASTER_ADDR=10.24.112.181` (g05 intra-cluster IP), `MASTER_PORT=29500`, aux store `+1` |
| Repeats | 3 per `(mode, size)`; iters per repeat: 1000 ≤64 KiB, 500 ≤4 MiB, else 200 |
| Warmup | RDMA: time-bounded ≤250 ms / ≤200 iters; TCP: fixed 20 iters (lockstep) |
| Output | `results_full/run-20260501-055236/raw.jsonl` (36 records, all `checksum_ok=true`) |

## Results — medians of 3 repeats

| size | rdma GB/s | rdma p50 µs | rdma p99 µs | tcp GB/s | tcp p50 µs | tcp p99 µs | RDMA speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 KiB    |  0.27 |    15 |    17 | 0.16 |    26 |     31 |  **1.7×** |
| 64 KiB   |  3.59 |    18 |    20 | 2.82 |    23 |    141 |  **1.3×** |
| 1 MiB    | 24.01 |    44 |    45 | 4.03 |   260 |    428 |  **6.0×** |
| 16 MiB   | 36.52 |   459 |   463 | 2.32 | 7 224 |  8 051 | **15.7×** |
| 64 MiB   | 37.28 | 1 800 | 2 138 | 3.22 | 20 839 | 28 848 | **11.6×** |
| 256 MiB  | 37.44 | 7 170 | 8 295 | 4.16 | 64 552 | 110 206 | **9.0×** |

All 36 records (6 sizes × 3 repeats × 2 modes) verified byte-for-byte
on the receiver before being recorded.

## Interpretation

- **RDMA saturates the RoCE fabric at ~37 GB/s by 16 MiB** (~75% of 400 Gbps
  line rate — typical for a real one-sided WRITE; the gap to line rate is
  consistent with the documented ROCm slow path: `mlx5dv_reg_dmabuf_mr` is
  unresolved on this stack, so registration falls back to `ibv_reg_mr` and
  GPUDirect-via-DMABUF is disabled).
- **TCP plateaus at 3–4 GB/s** independent of message size — that's the cost
  of `host_buf.copy_(gpu) → cuda.synchronize → sock.sendall` (D→H copy + sync
  + kernel TCP send), which is what TCP for a GPU tensor actually costs.
  This is *not* a TCP-tuning problem; it's the bounce-buffer tax.
- **Latency advantage is sharper at mid sizes**: ~6× at 1 MiB, ~16× at 16 MiB
  on p50.
- The small-message regime (≤64 KiB) is the only place TCP is competitive,
  because per-iter syscall + control overhead dominates both paths.

## Confirmation that this is across-node

| evidence | observation |
|---|---|
| process tags | `mia1-p01-g05:2421278` (rank 0) and `mia1-p01-g06:3427551` (rank 1) |
| TCP wire | `10.101.5.141 ↔ 10.101.6.141` — a `/31` Ethernet point-to-point between the two boxes |
| sustained BW | ~37 GB/s on RDMA = ~75% of a 400 Gbps NIC, line-rate-bound (loopback would be much higher and CPU-copy bound) |
| TCP cap | ~3–4 GB/s ≈ ~32 Gbit, consistent with kernel TCP through the tw-eth4 NIC + bounce-buffer cost |

## How to reproduce

On any two MI355X nodes that share this WekaFS checkout:

```bash
# Rank 0 (sender):
ssh mi355x-p01-g05
cd /home/rishi.sinha@amd.com/comms_meta_test/torchcomms/comms/torchcomms/transport/validation/rocm/multinode
PYTHON=/home/rishi.sinha@amd.com/miniconda3/envs/torchcomms-rocm64/bin/python \
NODE_RANK=0 MASTER_ADDR=10.24.112.181 GPU_IDX=4 \
MODE=all REPEATS=3 \
TCP_BIND_IP=10.101.5.141 TCP_PEER_IP=10.101.6.141 \
OUTPUT_DIR=$PWD/results_full \
bash run_transport_bench.sh

# Rank 1 (receiver), same checkout via WekaFS:
ssh mi355x-p01-g06
cd /home/rishi.sinha@amd.com/comms_meta_test/torchcomms/comms/torchcomms/transport/validation/rocm/multinode
PYTHON=/home/rishi.sinha@amd.com/miniconda3/envs/torchcomms-rocm64/bin/python \
NODE_RANK=1 MASTER_ADDR=10.24.112.181 GPU_IDX=4 \
MODE=all REPEATS=3 \
TCP_BIND_IP=10.101.6.141 TCP_PEER_IP=10.101.5.141 \
OUTPUT_DIR=$PWD/results_full \
bash run_transport_bench.sh
```

For a quick smoke before the full sweep: `MODE=rdma REPEATS=1 SIZES="1048576"`.

## Caveats

- **DMABUF is disabled** on this ROCm/HIP build (`mlx5dv_reg_dmabuf_mr` is
  unresolved by `dlvsym` — see `IbverbxSymbols.cc:658` warning at startup).
  Registration goes through `ibv_reg_mr`. RDMA bandwidth is still
  fabric-bound here; if DMABUF is enabled later, expect numbers closer
  to the 400 Gbps cap.
- **PutSignal RMA primitives trap on AMD** (no `atomic_ref` for the signal
  kernel). The bench deliberately uses plain WRITE/READ only.
- **TCP warmup is fixed-iter**, not time-bounded, because the two sides
  must run lockstep — see commit `56220933` for the deadlock that fixes.
- **Rendezvous needs a routable hostname/IP**: the ssh alias
  `mi355x-p01-g05` doesn't resolve from inside the cluster, only from
  outside. Use the actual hostname (`mia1-p01-g05`) or its intra-cluster
  IP (`10.24.112.181`) for `MASTER_ADDR`.

## Branch history

| commit | message |
|---|---|
| `c1678aea` | Add RDMA vs TCP transport bandwidth bench |
| `d9b200cd` | Fix transport bench launcher for AMD/ROCm (activate deps, `rdmaN` naming, DMABUF env) |
| `56220933` | transport_bench: fix TCP warmup deadlock at large sizes |
