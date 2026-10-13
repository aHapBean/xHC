#!/usr/bin/env bash
# Run inside the supplied CUDA development image or an isolated Python environment.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${PYTHON_BIN:-python3}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDACXX="$CUDA_HOME/bin/nvcc"
export MAX_JOBS=${MAX_JOBS:-8}
export NVTE_FRAMEWORK=pytorch
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-10.0}
export NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS:-100}
"$python" -m pip install --index-url https://download.pytorch.org/whl/cu130 'torch==2.9.0'
"$python" -m pip install --index-url https://pypi.org/simple -c "$root/constraints-cu130.txt" 'setuptools==79.0.1' wheel ninja cmake pybind11 packaging 'numpy==1.26.4'
"$python" -m pip install --index-url https://pypi.org/simple --no-build-isolation -c "$root/constraints-cu130.txt" 'transformer-engine[pytorch,core_cu13]==2.10.0'
"$python" -m pip install --index-url https://pypi.org/simple -c "$root/constraints-cu130.txt" -r "$root/requirements.txt"
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 "$python" -m pip install --no-build-isolation --no-deps \
  'git+https://github.com/NVIDIA/apex.git@90055a9188f907e1fd78d9099aa45252efd62efb'
"$python" -m pip install --no-deps --no-build-isolation -e "$root/Megatron-LM"
