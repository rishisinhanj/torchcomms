# ROCm Transport Validation

This directory contains manual validation helpers for the ROCm-enabled
`torchcomms._transport` extension.

The goal is to validate the transport API directly:

- `RdmaTransport.supported()`
- `RdmaTransport(torch.device(...))`
- `RdmaMemory(tensor)`
- `bind()`
- `connect()`
- `write()`
- `read()`

These scripts are intentionally separate from normal CI. They are meant to make
ROCm transport bring-up reproducible on MI355X-style nodes.

## Layout

```text
validation/rocm/
  single_node/
    build_transport_deps.sh
    build_rocm_transport.sh
    test_rocm_transport.py
  multinode/
    multinode_rocm_transport_test.py
    run_multinode_rocm_transport.sh
```

## Prerequisites

Use a PyTorch ROCm environment that can see the target GPUs. On the MI355X test
nodes this used ROCm 7.1 PyTorch:

```bash
PYTHON=/home/rishi.sinha@amd.com/miniconda3/envs/torchcomms-rocm64/bin/python
```

The scripts default to building local native dependencies under:

```text
<torchcomms repo>/.deps/transport
```

Override paths with `DEPS_ROOT`, `PREFIX`, `TORCHCOMMS_DIR`, `ROCM_HOME`, and
`PYTHON` as needed.

## Build The Transport Extension

From the repository root:

```bash
PYTHON=/path/to/rocm/python \
  comms/torchcomms/transport/validation/rocm/single_node/build_transport_deps.sh

PYTHON=/path/to/rocm/python \
  comms/torchcomms/transport/validation/rocm/single_node/build_rocm_transport.sh
```

The build script enables only the transport extension:

```text
USE_TRANSPORT=1
USE_NCCL=0
USE_NCCLX=0
USE_GLOO=0
USE_RCCL=0
USE_RCCLX=0
USE_XCCL=0
USE_TRITON=0
```

The expected Python module is:

```python
from torchcomms._transport import RdmaMemory, RdmaTransport
```

## Single-Node Validation

Run:

```bash
PYTHON=/path/to/rocm/python \
  comms/torchcomms/transport/validation/rocm/single_node/test_rocm_transport.py
```

This script:

1. configures the local dependency prefix
2. re-execs so `LD_LIBRARY_PATH` is active before importing `torchcomms`
3. prints PyTorch/ROCm and RDMA device information
4. imports `torchcomms._transport`
5. checks `RdmaTransport.supported()`
6. wraps a CUDA/HIP tensor in `RdmaMemory`
7. runs the upstream single-node transport test suite:

```text
comms/torchcomms/transport/tests/py/TransportTest.py
```

This validates the local API and same-node transfer behavior. It does not prove
a cross-host RDMA transfer over the fabric.

## Multinode Validation

The multinode harness directly targets the CTRAN-backed RDMA transport path, not
an RCCLX collective.

```text
torchrun / TCPStore: process launch and metadata exchange only
RdmaTransport: actual cross-node data movement
```

The script exchanges:

- each rank's `RdmaTransport.bind()` bytes
- each rank's pickled `RdmaRemoteBuffer`

The tensor data path is:

```text
RdmaMemory
  -> RdmaTransport.connect(peer_bind)
  -> RdmaTransport.write(local_view, peer_remote_buffer)
  -> RdmaTransport.read(local_mutable_view, peer_remote_buffer)
```

Launch one process per node. On node 0:

```bash
cd <torchcomms repo>
export MASTER_ADDR=<node0 reachable hostname or IP>
export MASTER_PORT=29500
export RDMA_STORE_PORT=29501
export PYTHON=/path/to/rocm/python

DEVICE_INDEX=0 NODE_RANK=0 \
  comms/torchcomms/transport/validation/rocm/multinode/run_multinode_rocm_transport.sh
```

On node 1:

```bash
cd <torchcomms repo>
export MASTER_ADDR=<same node0 hostname or IP>
export MASTER_PORT=29500
export RDMA_STORE_PORT=29501
export PYTHON=/path/to/rocm/python

DEVICE_INDEX=0 NODE_RANK=1 \
  comms/torchcomms/transport/validation/rocm/multinode/run_multinode_rocm_transport.sh
```

Use a different `DEVICE_INDEX` if the default GPU is reserved.

## Observed Multinode Result

Observed on MI355X nodes:

```text
node0: mi355x-p02-g17, GPU 4
node1: mi355x-p01-g07, GPU 4
tensor: 1 MiB uint8
```

Connection setup:

```text
[rank 0] rdma_supported=True
[rank 1] rdma_supported=True
[rank 0] connected_to=1
[rank 1] connected_to=0
```

Data validation:

```text
[rank 0] peer_write verified checksum=128552960
[rank 0] peer_read verified checksum=128552960
[rank 1] peer_write verified checksum=128552960
[rank 1] peer_read verified checksum=128552960
```

The checksum is only a compact log summary. The pass condition is exact byte
comparison:

```python
torch.equal(actual, expected)
```

This validates cross-host `bind()`, `connect()`, `write()`, and `read()` for
the transport API.

## Known Harness Caveat

The validation harness force-exits after successful data verification by
default. This avoids an observed ROCm transport teardown hang after the RDMA
data path has already passed. Use `--no-force-exit` to debug normal destructor
teardown.
