#!/usr/bin/env bash
# Launcher for transport_bench.py on a 2-node MI355X + Mellanox RoCE pair.
#
# Run on BOTH nodes in parallel, with NODE_RANK={0,1}. Rank 0 hosts the
# rendezvous and the auxiliary TCPStore; rank 1 dials in. Pin both ranks
# to the SAME GPU index and the SAME mlx5 device index so the comparison
# is symmetric across the fabric.
#
# Required environment:
#   NODE_RANK         0 on the rank-0 host, 1 on the rank-1 host
#   MASTER_ADDR       hostname/IP of the rank-0 node (reachable from rank 1)
#   GPU_IDX           which GPU index to use on each node (e.g. 4)
#   NIC_IDX           which mlx5 device index to bind (e.g. 4)
#
# Optional:
#   TCP_BIND_IP       local IP of the chosen NIC's tw-ethN interface
#   TCP_PEER_IP       peer's tw-ethN IP (rank 1 dials this for the TCP path)
#   MASTER_PORT       default 29500
#   RDMA_STORE_PORT   default MASTER_PORT+1
#   OUTPUT_DIR        default ./results
#   MODE              default "all" (rdma tcp); pass e.g. "rdma" or "tcp"
#   REPEATS           default 3

set -euo pipefail

: "${NODE_RANK:?must set NODE_RANK=0 or 1}"
: "${MASTER_ADDR:?must set MASTER_ADDR=<rank-0 hostname>}"
: "${GPU_IDX:?must set GPU_IDX (e.g. 4)}"
: "${NIC_IDX:?must set NIC_IDX (e.g. 4)}"

MASTER_PORT="${MASTER_PORT:-29500}"
RDMA_STORE_PORT="${RDMA_STORE_PORT:-$((MASTER_PORT+1))}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
MODE="${MODE:-all}"
REPEATS="${REPEATS:-3}"

export HIP_VISIBLE_DEVICES="${GPU_IDX}"
export NCCL_IB_HCA="=mlx5_${NIC_IDX}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export MASTER_ADDR MASTER_PORT RDMA_STORE_PORT

# Optional NUMA pinning if numactl is available; safe to no-op otherwise.
NUMA_PREFIX=()
if command -v numactl >/dev/null 2>&1 && [[ -n "${NUMA_NODE:-}" ]]; then
  NUMA_PREFIX=(numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -x
"${NUMA_PREFIX[@]}" torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/transport_bench.py" \
    --mode ${MODE} \
    --repeats "${REPEATS}" \
    --output-dir "${OUTPUT_DIR}" \
    ${TCP_BIND_IP:+--tcp-bind-ip "${TCP_BIND_IP}"} \
    ${TCP_PEER_IP:+--tcp-peer-ip "${TCP_PEER_IP}"}
