"""Inspect the installed public training stack before launching an experiment."""
import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import torch
import transformer_engine.pytorch
import triton
import fused_weight_gradient_mlp_cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Megatron-LM'))
from megatron.core.fusions import fused_xhc

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dispatcher', choices=('alltoall', 'hybridep'), default='alltoall')
args = parser.parse_args()
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable; run the container with GPU access')
if args.dispatcher == 'hybridep':
    from deep_ep import HybridEPBuffer
report = {
    'python': sys.version,
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'triton': triton.__version__,
    'transformer_engine': importlib.metadata.version('transformer-engine'),
    'apex': importlib.metadata.version('apex'),
    'deep_ep': importlib.metadata.version('deep-ep') if args.dispatcher == 'hybridep' else None,
    'cuda_home': os.environ.get('CUDA_HOME'),
    'gpu': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    'fused_xhc_module': fused_xhc.__file__,
    'dispatcher': args.dispatcher,
}
x = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all().item()
torch.cuda.synchronize()
print(json.dumps(report, indent=2))
print('Imports and CUDA BF16 GEMM passed; this does not test distributed communication.')
