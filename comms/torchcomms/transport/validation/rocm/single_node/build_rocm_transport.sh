#!/usr/bin/env bash
set -euo pipefail

# Build only the torchcomms ROCm transport extension using the dependency
# prefix produced by build_transport_deps.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TORCHCOMMS_DIR="${TORCHCOMMS_DIR:-$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd -P)}"
DEPS_ROOT="${DEPS_ROOT:-${TORCHCOMMS_DIR}/.deps/transport}"
PREFIX="${PREFIX:-${DEPS_ROOT}/prefix}"
LOG_DIR="${LOG_DIR:-${DEPS_ROOT}/logs}"
ACTIVATE="${ACTIVATE:-${PREFIX}/activate_transport_deps.sh}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/build_rocm_transport_$(date -u +%Y%m%dT%H%M%SZ).log}"
PYTHON="${PYTHON:-python3}"

if [[ ! -f "$ACTIVATE" ]]; then
  printf 'Missing dependency activation file: %s\n' "$ACTIVATE" >&2
  printf 'Run ./build_transport_deps.sh first.\n' >&2
  exit 1
fi

if [[ ! -d "$TORCHCOMMS_DIR" ]]; then
  printf 'Missing torchcomms checkout: %s\n' "$TORCHCOMMS_DIR" >&2
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
export USE_TRANSPORT=1
export USE_NCCL=0
export USE_NCCLX=0
export USE_GLOO=0
export USE_RCCL=0
export USE_RCCLX=0
export USE_XCCL=0
export USE_TRITON=0
export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"

{
  printf '== ROCm transport build ==\n'
  printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'torchcomms_dir=%s\n' "$TORCHCOMMS_DIR"
  printf 'deps_prefix=%s\n' "$DEPS_PREFIX"
  printf 'rocm_home=%s\n' "$ROCM_HOME"
  printf 'python=%s\n' "$(command -v "$PYTHON" || printf '%s' "$PYTHON")"
  printf 'cmake=%s\n' "$(command -v cmake)"
  cmake --version | sed -n '1p'
  "$PYTHON" - <<'PY'
import sys
import torch

print(f"python_version={sys.version.split()[0]}")
print(f"torch_version={torch.__version__}")
print(f"torch_hip={getattr(torch.version, 'hip', None)}")
print(f"torch_cuda_available={torch.cuda.is_available()}")
print(f"torch_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
PY
  printf '\n== pip build ==\n'
  cd "$TORCHCOMMS_DIR"
  "$PYTHON" comms/utils/cvars/extractcvars.py
  "$PYTHON" -m pip install -v --no-build-isolation .
} 2>&1 | tee "$LOG_FILE"

printf 'Build log: %s\n' "$LOG_FILE"
