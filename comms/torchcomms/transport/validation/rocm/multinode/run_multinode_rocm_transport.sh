#!/usr/bin/env bash
set -euo pipefail

# Launch the two-node torchcomms._transport RDMA validation harness.
# Run this script once on each node with NODE_RANK=0 on node0 and NODE_RANK=1 on node1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TORCHCOMMS_DIR="${TORCHCOMMS_DIR:-$(cd "${SCRIPT_DIR}/../../../../../.." && pwd -P)}"
DEPS_ROOT="${DEPS_ROOT:-${TORCHCOMMS_DIR}/.deps/transport}"
PREFIX="${PREFIX:-${DEPS_ROOT}/prefix}"
ACTIVATE="${ACTIVATE:-${PREFIX}/activate_transport_deps.sh}"
LOG_DIR="${LOG_DIR:-${DEPS_ROOT}/logs}"
PYTHON="${PYTHON:-python3}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
RDMA_STORE_PORT="${RDMA_STORE_PORT:-$((MASTER_PORT + 1))}"
NUM_BYTES="${NUM_BYTES:-1048576}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
PRINT_RDMA_DEVICES="${PRINT_RDMA_DEVICES:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
NCCL_CTRAN_IB_DMABUF_ENABLE="${NCCL_CTRAN_IB_DMABUF_ENABLE:-1}"

if [[ -z "${MASTER_ADDR:-}" ]]; then
  printf 'MASTER_ADDR is required, for example MASTER_ADDR=node0.example.com\n' >&2
  exit 1
fi

if [[ -z "${NODE_RANK:-}" ]]; then
  printf 'NODE_RANK is required: use 0 on the master node and 1 on the peer node\n' >&2
  exit 1
fi

if [[ ! -f "$ACTIVATE" ]]; then
  printf 'Missing dependency activation file: %s\n' "$ACTIVATE" >&2
  printf 'Run single_node/build_transport_deps.sh first.\n' >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
# shellcheck disable=SC1090
source "$ACTIVATE"

PYTHON_BIN_DIR="$(cd "$(dirname "$("$PYTHON" -c 'import sys; print(sys.executable)')")" && pwd -P)"
export PATH="${PYTHON_BIN_DIR}:${DEPS_PREFIX}/bin:${PATH}"
export CMAKE_PREFIX_PATH="${DEPS_PREFIX}"
export PKG_CONFIG_PATH="${DEPS_PREFIX}/lib/pkgconfig"
export LD_LIBRARY_PATH="${DEPS_PREFIX}/lib:${ROCM_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export USE_SYSTEM_LIBS=1
export RDMA_STORE_PORT
export NCCL_DEBUG
export NCCL_CTRAN_IB_DMABUF_ENABLE

LOG_FILE="${LOG_FILE:-${LOG_DIR}/multinode_rocm_transport_rank${NODE_RANK}_$(date -u +%Y%m%dT%H%M%SZ).log}"

ARGS=(
  "--num-bytes=${NUM_BYTES}"
  "--device-index=${DEVICE_INDEX}"
)

if [[ "$PRINT_RDMA_DEVICES" == "1" ]]; then
  ARGS+=("--print-rdma-devices")
fi

{
  printf '== Multinode ROCm RDMA transport test ==\n'
  printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'node_rank=%s\n' "$NODE_RANK"
  printf 'master_addr=%s\n' "$MASTER_ADDR"
  printf 'master_port=%s\n' "$MASTER_PORT"
  printf 'rdma_store_port=%s\n' "$RDMA_STORE_PORT"
  printf 'num_bytes=%s\n' "$NUM_BYTES"
  printf 'device_index=%s\n' "$DEVICE_INDEX"
  printf 'NCCL_DEBUG=%s\n' "$NCCL_DEBUG"
  printf 'NCCL_CTRAN_IB_DMABUF_ENABLE=%s\n' "$NCCL_CTRAN_IB_DMABUF_ENABLE"
  printf 'python=%s\n' "$(command -v "$PYTHON" || printf '%s' "$PYTHON")"
  "$PYTHON" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/multinode_rocm_transport_test.py" \
    "${ARGS[@]}"
} 2>&1 | tee "$LOG_FILE"

printf 'Multinode test log: %s\n' "$LOG_FILE"
