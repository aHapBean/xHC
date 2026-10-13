#!/usr/bin/env bash
set -euo pipefail
python=${PYTHON_BIN:-python3}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LIBRARY_PATH="$CUDA_HOME/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$CUDA_HOME/include/cccl:$CUDA_HOME/include${CPATH:+:$CPATH}"
export MAX_JOBS=${MAX_JOBS:-8}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-10.0}
# The supplied EP=4 setup keeps each expert group within one NVLink domain.
export HYBRID_EP_MULTINODE=0
"$python" -m pip install --index-url https://pypi.org/simple 'pynvml==13.0.1'
"$python" -m pip install --no-build-isolation --no-deps \
  'git+https://github.com/deepseek-ai/DeepEP.git@eb9cee7de5a24193bf09500668d3a619d3d3f3fb'
"$python" -c 'from deep_ep import HybridEPBuffer; print("HybridEP import OK")'
