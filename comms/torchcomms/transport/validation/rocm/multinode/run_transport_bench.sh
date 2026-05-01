#!/usr/bin/env bash
# Launcher for transport_bench.py on a 2-node MI355X + Mellanox RoCE pair.
#
# Run on BOTH nodes in parallel, with NODE_RANK={0,1}. Rank 0 hosts the
# rendezvous and the auxiliary TCPStore; rank 1 dials in. Pin both ranks
# to the SAME GPU index and the SAME rdma device index so the comparison
# is symmetric across the fabric.
#
# Required environment:
#   NODE_RANK         0 on the rank-0 host, 1 on the rank-1 host
#   MASTER_ADDR       hostname/IP of the rank-0 node (reachable from rank 1)
#
# Optional:
#   GPU_IDX           GPU index to use on each node (default 0)
#   NIC_IDX           rdmaN device index to bind via NCCL_IB_HCA (default: unpinned, ctran auto-picks)
#   TCP_BIND_IP       local IP of the chosen NIC's tw-ethN interface
#   TCP_PEER_IP       peer's tw-ethN IP (rank 1 dials this for the TCP path)
#   MASTER_PORT       default 29500
#   RDMA_STORE_PORT   default MASTER_PORT+1
#   OUTPUT_DIR        default ./results
#   MODE              default "all" (rdma tcp); pass e.g. "rdma" or "tcp"
#   REPEATS           default 3
#   SIZES             optional space-separated byte sizes; defaults to bench's built-in sweep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TORCHCOMMS_DIR="${TORCHCOMMS_DIR:-$(cd "${SCRIPT_DIR}/../../../../../.." && pwd -P)}"
DEPS_ROOT="${DEPS_ROOT:-${TORCHCOMMS_DIR}/.deps/transport}"
PREFIX="${PREFIX:-${DEPS_ROOT}/prefix}"
ACTIVATE="${ACTIVATE:-${PREFIX}/activate_transport_deps.sh}"
LOG_DIR="${LOG_DIR:-${DEPS_ROOT}/logs}"
PYTHON="${PYTHON:-python3}"

: "${NODE_RANK:?must set NODE_RANK=0 or 1}"
: "${MASTER_ADDR:?must set MASTER_ADDR=<rank-0 hostname>}"

GPU_IDX="${GPU_IDX:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
RDMA_STORE_PORT="${RDMA_STORE_PORT:-$((MASTER_PORT + 1))}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
MODE="${MODE:-all}"
REPEATS="${REPEATS:-3}"

if [[ ! -f "$ACTIVATE" ]]; then
  printf 'Missing dependency activation file: %s\n' "$ACTIVATE" >&2
  printf 'Run single_node/build_transport_deps.sh first.\n' >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# shellcheck disable=SC1090
source "$ACTIVATE"

PYTHON_BIN_DIR="$(cd "$(dirname "$("$PYTHON" -c 'import sys; print(sys.executable)')")" && pwd -P)"
export PATH="${PYTHON_BIN_DIR}:${DEPS_PREFIX}/bin:${PATH}"
export CMAKE_PREFIX_PATH="${DEPS_PREFIX}"
export PKG_CONFIG_PATH="${DEPS_PREFIX}/lib/pkgconfig"
export LD_LIBRARY_PATH="${DEPS_PREFIX}/lib:${ROCM_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export USE_SYSTEM_LIBS=1
export RDMA_STORE_PORT
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_CTRAN_IB_DMABUF_ENABLE="${NCCL_CTRAN_IB_DMABUF_ENABLE:-1}"
export HIP_VISIBLE_DEVICES="${GPU_IDX}"

# Optional NIC pin. AMD MI355X uses rdma0..rdma7 (NOT mlx5_*). If unset,
# ctran auto-picks; if set, force the same rdmaN on both ranks for symmetry.
if [[ -n "${NIC_IDX:-}" ]]; then
  export NCCL_IB_HCA="=rdma${NIC_IDX}"
fi

# Optional NUMA pinning if numactl is available; safe to no-op otherwise.
NUMA_PREFIX=()
if command -v numactl >/dev/null 2>&1 && [[ -n "${NUMA_NODE:-}" ]]; then
  NUMA_PREFIX=(numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}")
fi

BENCH_ARGS=(
  "--mode" ${MODE}
  "--repeats" "${REPEATS}"
  "--output-dir" "${OUTPUT_DIR}"
  "--device" "cuda:0"
)
if [[ -n "${SIZES:-}" ]]; then
  # shellcheck disable=SC2206
  BENCH_ARGS+=("--sizes" ${SIZES})
fi

LOG_FILE="${LOG_FILE:-${LOG_DIR}/transport_bench_rank${NODE_RANK}_$(date -u +%Y%m%dT%H%M%SZ).log}"

{
  printf '== RDMA vs TCP transport bench ==\n'
  printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'node_rank=%s\n' "$NODE_RANK"
  printf 'master_addr=%s\n' "$MASTER_ADDR"
  printf 'master_port=%s\n' "$MASTER_PORT"
  printf 'rdma_store_port=%s\n' "$RDMA_STORE_PORT"
  printf 'gpu_idx=%s\n' "$GPU_IDX"
  printf 'nic_idx=%s\n' "${NIC_IDX:-<unset, ctran auto-pick>}"
  printf 'NCCL_IB_HCA=%s\n' "${NCCL_IB_HCA:-<unset>}"
  printf 'TCP_BIND_IP=%s\n' "${TCP_BIND_IP:-<unset>}"
  printf 'TCP_PEER_IP=%s\n' "${TCP_PEER_IP:-<unset>}"
  printf 'mode=%s repeats=%s sizes=%s\n' "$MODE" "$REPEATS" "${SIZES:-<default>}"
  printf 'NCCL_DEBUG=%s\n' "$NCCL_DEBUG"
  printf 'NCCL_CTRAN_IB_DMABUF_ENABLE=%s\n' "$NCCL_CTRAN_IB_DMABUF_ENABLE"
  printf 'python=%s\n' "$(command -v "$PYTHON" || printf '%s' "$PYTHON")"
  set -x
  "${NUMA_PREFIX[@]}" "$PYTHON" -m torch.distributed.run \
    --nnodes=2 \
    --nproc_per_node=1 \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/transport_bench.py" \
    "${BENCH_ARGS[@]}"
} 2>&1 | tee "$LOG_FILE"

printf 'Transport bench log: %s\n' "$LOG_FILE"
