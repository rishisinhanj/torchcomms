#!/usr/bin/env python3
"""Smoke-check torchcomms._transport, then run the transport test suite."""

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path



# Resolve paths relative to the torchcomms repository by default.
ROOT = Path(__file__).resolve().parent
TORCHCOMMS_DIR = Path(
    os.environ.get("TORCHCOMMS_DIR", ROOT.parents[5])
).resolve()
DEPS_PREFIX = Path(
    os.environ.get("DEPS_PREFIX", TORCHCOMMS_DIR / ".deps" / "transport" / "prefix")
)
ROCM_HOME = os.environ.get("ROCM_HOME", "/opt/rocm")
TEST_FILE = TORCHCOMMS_DIR / "comms" / "torchcomms" / "transport" / "tests" / "py" / "TransportTest.py"


def configure_env() -> None:
    activate = DEPS_PREFIX / "activate_transport_deps.sh"
    if not activate.exists():
        raise SystemExit(f"Missing {activate}. Run ./build_transport_deps.sh first.")

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
    """Restart once so LD_LIBRARY_PATH is visible before imports/dlopen."""
    if os.environ.get("ROCM_TRANSPORT_ENV_READY") == "1":
        return
    configure_env()
    os.environ["ROCM_TRANSPORT_ENV_READY"] = "1"
    python = os.environ.get("PYTHON", sys.executable)
    os.execve(python, [python, __file__], os.environ.copy())


def print_environment() -> None:
    import torch

    print("== Environment ==")
    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"torch_hip={getattr(torch.version, 'hip', None)}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device_count={torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"device_{idx}={torch.cuda.get_device_name(idx)}")

    print("\n== RDMA devices ==")
    subprocess.run(["ibv_devices"], check=False)
    subprocess.run(["rdma", "link"], check=False)


# this is where we actually start testing 
# create a small tensor and wrap it in RDMA memory
# Tensor registration works to create RDMAmemory obj 
def smoke_test_transport() -> None:
    import torch
    import torchcomms._transport as transport

    print("\n== torchcomms._transport smoke test ==")
    print(f"module={transport.__file__}")
    print(f"rdma_supported={transport.RdmaTransport.supported()}")
    tensor = torch.arange(16, dtype=torch.uint8, device="cuda:0")
    memory = transport.RdmaMemory(tensor)
    print(f"rdma_memory_view_size={memory.to_view().size()}")
    print(f"rdma_remote_buffer_type={type(memory.to_remote_buffer()).__name__}")


def run_transport_tests() -> int:
    print("\n== TransportTest.py ==")
    if not TEST_FILE.exists():
        raise SystemExit(f"Missing {TEST_FILE}")
    return subprocess.run([sys.executable, str(TEST_FILE)], cwd=TORCHCOMMS_DIR).returncode


def main() -> int:
    reexec_with_env()
    configure_env()
    print_environment()
    smoke_test_transport()
    return run_transport_tests()


if __name__ == "__main__":
    raise SystemExit(main())
