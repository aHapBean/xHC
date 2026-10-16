#!/usr/bin/env bash
set -euo pipefail
# Reference training configuration.
# The training budget may differ from those used in the paper,
# as the token budgets are not disclosed in the paper.
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runtime=${XHC_RUNTIME_ENV:-$root/configs/runtime.env}
if [[ -n ${XHC_RUNTIME_ENV:-} || -f $runtime ]]; then source "$runtime"; fi
export EXP_NAME=${EXP_NAME:-10B_xHC}
exec "${PYTHON_BIN:-python3}" "$root/scripts/train.py" "$root/configs/10B_xHC.yaml" --backend reference "$@"
