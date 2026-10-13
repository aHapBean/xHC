#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runtime=${XHC_RUNTIME_ENV:-$root/configs/runtime.env}
if [[ -n ${XHC_RUNTIME_ENV:-} || -f $runtime ]]; then source "$runtime"; fi
export EXP_NAME=${EXP_NAME:-2.5B_baseline}
exec "${PYTHON_BIN:-python3}" "$root/scripts/train.py" "$root/configs/2.5B_baseline.yaml" --backend reference "$@"
