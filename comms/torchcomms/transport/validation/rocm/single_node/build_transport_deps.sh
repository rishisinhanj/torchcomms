#!/usr/bin/env bash
set -euo pipefail

# Reproducible-ish dependency bootstrap for the torchcomms ROCm transport build.
# Installs generated dependencies under .deps/transport by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TORCHCOMMS_DIR="${TORCHCOMMS_DIR:-$(cd "${SCRIPT_DIR}/../../../../../.." && pwd -P)}"
DEPS_ROOT="${DEPS_ROOT:-${TORCHCOMMS_DIR}/.deps/transport}"
PREFIX="${PREFIX:-${DEPS_ROOT}/prefix}"
SRC_DIR="${SRC_DIR:-${DEPS_ROOT}/src}"
BUILD_DIR="${BUILD_DIR:-${DEPS_ROOT}/build}"
LOG_DIR="${LOG_DIR:-${DEPS_ROOT}/logs}"
MANIFEST="${MANIFEST:-${DEPS_ROOT}/manifest.txt}"
DEFAULT_THREADS="$(nproc)"
if ((DEFAULT_THREADS > 32)); then
  DEFAULT_THREADS=32
fi
THREADS="${THREADS:-$DEFAULT_THREADS}"
ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
BUILD_THRIFT_DEPS="${BUILD_THRIFT_DEPS:-0}"

VERIFY_ONLY=0
CLEAN_BUILDS=0

usage() {
  printf 'Usage: %s [--verify-only] [--clean-builds]\n' "$0"
  printf '\n'
  printf 'Environment overrides: DEPS_ROOT PREFIX SRC_DIR BUILD_DIR LOG_DIR THREADS ROCM_HOME\n'
  printf 'Set BUILD_THRIFT_DEPS=1 to also build fizz, wangle, and fbthrift.\n'
}

while (($#)); do
  case "$1" in
    --verify-only)
      VERIFY_ONLY=1
      ;;
    --clean-builds)
      CLEAN_BUILDS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

require_command() {
  local command_name=$1
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "$command_name" >&2
    exit 1
  fi
}

require_command git
require_command cmake
require_command ninja
require_command make
require_command pkg-config
require_command python3

mkdir -p "$PREFIX" "$SRC_DIR" "$BUILD_DIR" "$LOG_DIR"

export PATH="$PREFIX/bin:${PATH}"
export CMAKE_PREFIX_PATH="$PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$PREFIX/lib:${ROCM_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

declare -A REPO=(
  [fmt]="https://github.com/fmtlib/fmt.git"
  [gflags]="https://github.com/gflags/gflags.git"
  [glog]="https://github.com/google/glog.git"
  [double-conversion]="https://github.com/google/double-conversion.git"
  [libevent]="https://github.com/libevent/libevent.git"
  [zstd]="https://github.com/facebook/zstd.git"
  [xxHash]="https://github.com/Cyan4973/xxHash.git"
  [libsodium]="https://github.com/jedisct1/libsodium.git"
  [boost]="https://github.com/boostorg/boost.git"
  [openssl]="https://github.com/openssl/openssl.git"
  [folly]="https://github.com/facebook/folly.git"
  [fizz]="https://github.com/facebookincubator/fizz.git"
  [wangle]="https://github.com/facebook/wangle.git"
  [fbthrift]="https://github.com/facebook/fbthrift.git"
)

declare -A REF=(
  [fmt]="10.2.1"
  [gflags]="v2.2.2"
  [glog]="v0.4.0"
  [double-conversion]="v3.3.1"
  [libevent]="release-2.1.12-stable"
  [zstd]="v1.5.6"
  [xxHash]="v0.8.0"
  [libsodium]="1.0.20-RELEASE"
  [boost]="boost-1.82.0"
  [openssl]="openssl-3.5.1"
  [folly]="v2024.05.20.00"
  [fizz]="v2024.05.20.00"
  [wangle]="v2024.05.20.00"
  [fbthrift]="v2024.05.20.00"
)

declare -A SHA=(
  [fmt]="e69e5f977d458f2650bb346dadf2ad30c5320281"
  [gflags]="e171aa2d15ed9eb17054558e0b3a6a413bb01067"
  [glog]="96a2f23dca4cc7180821ca5f32e526314395d26a"
  [double-conversion]="ae0dbfeb9744efd216c95b30555049d75d47116a"
  [libevent]="5df3037d10556bfcb675bc73e516978b75fc7bc7"
  [zstd]="794ea1b0afca0f020f4e57b6732332231fb23c70"
  [xxHash]="94e5f23e736f2bb67ebdf90727353e65344f9fc0"
  [libsodium]="9511c982fb1d046470a8b42aa36556cdb7da15de"
  [boost]="b6928ae5c92e21a04bbe17a558e6e066dbe632f6"
  [openssl]="aea7aaf2abb04789f5868cbabec406ea43aa84bf"
  [folly]="89ac8453fc7825e566b46a7ebb9e0348b1c09338"
  [fizz]="c5cb900f96355c9be0f41b77bcf7937a5909c9b3"
  [wangle]="beeb9478a0321190791d9816f7ffe9a37d02d429"
  [fbthrift]="8baf3aed99934494267647ee8f309f9d712d4325"
)

DEPS=(
  fmt
  gflags
  glog
  double-conversion
  libevent
  zstd
  xxHash
  libsodium
  boost
  openssl
  folly
)

if [[ "$BUILD_THRIFT_DEPS" == 1 ]]; then
  DEPS+=(fizz wangle fbthrift)
fi

write_activation() {
  cat > "$PREFIX/activate_transport_deps.sh" <<EOF
# Source this file before building or testing torchcomms transport.
export DEPS_PREFIX="${PREFIX}"
export ROCM_HOME="${ROCM_HOME}"
export USE_SYSTEM_LIBS=1
export CONDA_PREFIX="${PREFIX}"
export PATH="\${DEPS_PREFIX}/bin:\${PATH}"
export CMAKE_PREFIX_PATH="\${DEPS_PREFIX}\${CMAKE_PREFIX_PATH:+:\${CMAKE_PREFIX_PATH}}"
export PKG_CONFIG_PATH="\${DEPS_PREFIX}/lib/pkgconfig\${PKG_CONFIG_PATH:+:\${PKG_CONFIG_PATH}}"
export LD_LIBRARY_PATH="\${DEPS_PREFIX}/lib:\${ROCM_HOME}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOF
}

start_manifest() {
  {
    printf 'torchcomms transport dependency manifest\n'
    printf 'generated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'script=%s\n' "$0"
    printf 'deps_root=%s\n' "$DEPS_ROOT"
    printf 'prefix=%s\n' "$PREFIX"
    printf 'src_dir=%s\n' "$SRC_DIR"
    printf 'build_dir=%s\n' "$BUILD_DIR"
    printf 'threads=%s\n' "$THREADS"
    printf 'rocm_home=%s\n' "$ROCM_HOME"
    printf 'build_thrift_deps=%s\n' "$BUILD_THRIFT_DEPS"
    printf 'cc=%s\n' "${CC:-}"
    printf 'cxx=%s\n' "${CXX:-}"
    printf 'cmake=%s\n' "$(cmake --version | sed -n '1p')"
    printf 'ninja=%s\n' "$(ninja --version)"
    if command -v hipcc >/dev/null 2>&1; then
      printf 'hipcc=%s\n' "$(hipcc --version | sed -n '1p')"
    fi
    printf '\n[dependencies]\n'
  } > "$MANIFEST"
}

append_manifest_dep() {
  local name=$1 src=$2 actual_sha
  actual_sha="$(git -C "$src" rev-parse HEAD)"
  printf '%s repo=%s ref=%s expected_sha=%s actual_sha=%s\n' \
    "$name" "${REPO[$name]}" "${REF[$name]}" "${SHA[$name]}" "$actual_sha" >> "$MANIFEST"
}

checkout_dep() {
  local name=$1
  local src="${SRC_DIR}/${name}"

  if [[ -e "$src" && ! -d "$src/.git" ]]; then
    printf 'Path exists but is not a git checkout: %s\n' "$src" >&2
    exit 1
  fi

  if [[ ! -d "$src/.git" ]]; then
    git clone --filter=blob:none --no-checkout "${REPO[$name]}" "$src"
  else
    git -C "$src" remote set-url origin "${REPO[$name]}"
    git -C "$src" fetch origin --tags --prune
  fi

  git -C "$src" fetch origin "${SHA[$name]}" --depth 1
  git -C "$src" checkout --detach "${SHA[$name]}"

  local actual_sha
  actual_sha="$(git -C "$src" rev-parse HEAD)"
  if [[ "$actual_sha" != "${SHA[$name]}" ]]; then
    printf 'Dependency %s resolved to %s, expected %s\n' "$name" "$actual_sha" "${SHA[$name]}" >&2
    exit 1
  fi
}

build_cmake() {
  local name=$1 src=$2 suffix=$3
  shift 3
  local build_dir="${BUILD_DIR}/${name}-${suffix}"
  if [[ "$CLEAN_BUILDS" == 1 ]]; then
    rm -rf "$build_dir"
  fi
  cmake -S "$src" -B "$build_dir" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DCMAKE_MODULE_PATH="$PREFIX/lib/cmake" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_INCLUDEDIR=include \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    "$@"
  cmake --build "$build_dir" --target install -- -j "$THREADS"
}

build_configure() {
  local name=$1 dir=$2
  shift 2
  pushd "$dir" >/dev/null
  if [[ ! -x configure && -f autogen.sh ]]; then
    ./autogen.sh
  fi
  ./configure --prefix="$PREFIX" "$@"
  make -j "$THREADS"
  make install
  popd >/dev/null
}

build_boost() {
  local src="${SRC_DIR}/boost"
  git -C "$src" submodule update --init --recursive --depth 1
  pushd "$src" >/dev/null
  ./bootstrap.sh --prefix="$PREFIX" --libdir="$PREFIX/lib" --without-libraries=python
  ./b2 -q install -j "$THREADS"
  popd >/dev/null
}

build_openssl() {
  local src="${SRC_DIR}/openssl"
  pushd "$src" >/dev/null
  ./Configure no-shared --prefix="$PREFIX" --openssldir="$PREFIX" --libdir=lib linux-x86_64
  make -j "$THREADS"
  make install_sw
  popd >/dev/null
}

write_activation
start_manifest

for dep in "${DEPS[@]}"; do
  checkout_dep "$dep"
  append_manifest_dep "$dep" "${SRC_DIR}/${dep}"
done

if [[ "$VERIFY_ONLY" == 1 ]]; then
  printf 'Verified dependency checkouts and wrote %s\n' "$MANIFEST"
  printf 'Activation file: %s\n' "$PREFIX/activate_transport_deps.sh"
  exit 0
fi

build_cmake fmt "${SRC_DIR}/fmt" shared "-DFMT_TEST=OFF" "-DFMT_DOC=OFF"
build_cmake gflags "${SRC_DIR}/gflags" shared "-DBUILD_SHARED_LIBS=ON" "-DBUILD_STATIC_LIBS=OFF"
build_cmake glog "${SRC_DIR}/glog" shared "-DWITH_GFLAGS=ON"
build_cmake double-conversion "${SRC_DIR}/double-conversion" shared "-DBUILD_TESTING=OFF"

build_cmake libevent "${SRC_DIR}/libevent" shared \
  "-DEVENT__DISABLE_OPENSSL=ON" \
  "-DEVENT__LIBRARY_TYPE=SHARED" \
  "-DEVENT__DISABLE_BENCHMARK=ON" \
  "-DEVENT__DISABLE_TESTS=ON"

build_cmake zstd "${SRC_DIR}/zstd/build/cmake" shared \
  "-DZSTD_BUILD_PROGRAMS=OFF" \
  "-DZSTD_BUILD_CONTRIB=OFF" \
  "-DZSTD_BUILD_TESTS=OFF" \
  "-DZSTD_BUILD_STATIC=OFF" \
  "-DZSTD_BUILD_SHARED=ON"

build_cmake xxHash "${SRC_DIR}/xxHash/cmake_unofficial" shared "-DXXHASH_BUILD_XXHSUM=OFF"
build_configure libsodium "${SRC_DIR}/libsodium" --disable-pie
build_boost
build_openssl

build_cmake folly "${SRC_DIR}/folly" shared \
  "-DBUILD_SHARED_LIBS=ON" \
  "-DBUILD_TESTS=OFF" \
  "-DBUILD_EXAMPLES=OFF" \
  "-DFOLLY_HAVE_LIBGFLAGS=ON" \
  "-DFOLLY_HAVE_LIBGLOG=ON" \
  "-DFOLLY_USE_SYMBOLIZER=OFF"

if [[ "$BUILD_THRIFT_DEPS" == 1 ]]; then
  build_cmake fizz "${SRC_DIR}/fizz/fizz" shared "-DBUILD_TESTS=OFF" "-DBUILD_EXAMPLES=OFF"
  build_cmake wangle "${SRC_DIR}/wangle/wangle" shared "-DBUILD_TESTS=OFF"
  build_cmake fbthrift "${SRC_DIR}/fbthrift/thrift" shared \
    "-DBUILD_SHARED_LIBS=ON" \
    "-DBUILD_TESTS=OFF" \
    "-DBUILD_EXAMPLES=OFF"
fi

write_activation

printf 'Third-party transport deps installed under %s\n' "$PREFIX"
printf 'Manifest: %s\n' "$MANIFEST"
printf 'To reuse in new shells: source %s\n' "$PREFIX/activate_transport_deps.sh"
