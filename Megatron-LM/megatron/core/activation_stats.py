"""Step-local activation moments, pooled over layers and data-parallel ranks.

The schedule opens a window around microbatch 0's original forward only.
Backward recomputation and evaluation cannot write into that window. No tensor
references or autograd hooks are retained. Kernels are imported only on use.
"""
from contextlib import contextmanager

import torch
import torch.distributed as dist

POINTS = (
    'attn_input', 'attn_output', 'post_attn_residual_output',
    'mlp_input', 'mlp_output', 'post_mlp_residual_output',
)
_active = None
_collector = None


class ActivationStats:
    def __init__(self, device):
        # Each row: count, mean, centered sum of squares, signed maximum.
        self.moments = torch.empty((len(POINTS), 4), dtype=torch.float64, device=device)
        self.scratch = torch.empty((3, 1024), dtype=torch.float32, device=device)
        self.seen = set()
        self.reset()

    def reset(self):
        self.moments.zero_()
        self.moments[:, 3].fill_(-float('inf'))
        self.seen.clear()

    def record(self, point, layer, tensor):
        key = (point, layer)
        if key in self.seen:
            return
        self.seen.add(key)
        if tensor.numel() == 0:
            return
        from megatron.core.fusions.fused_activation_stats import accumulate
        accumulate(tensor.detach(), self.scratch, self.moments, POINTS.index(point))

    def finish(self, writer, step, group=None, *, return_values=False):
        # One small collective for all 18 scalars. Gathering centered moments
        # avoids cancellation from reducing raw squared sums across ranks.
        world = dist.get_world_size(group) if dist.is_initialized() else 1
        gathered = torch.empty((world * len(POINTS), 4),
                               device=self.moments.device, dtype=self.moments.dtype)
        if world > 1:
            dist.all_gather_into_tensor(gathered, self.moments, group=group)
        else:
            gathered.copy_(self.moments)
        if writer is None and not return_values:
            return
        values = summarize(gathered.view(world, len(POINTS), 4).cpu())
        if writer is not None:
            for name, value in values.items():
                writer.add_scalar(name, value, step)
        if return_values:
            return values


def summarize(moments):
    """Merge centered moments with element weighting (CPU, float64)."""
    counts, means, m2, maxima = moments.unbind(-1)
    total = counts.sum(0)
    mean = (counts * means).sum(0) / total.clamp_min(1)
    variance = (m2 + counts * (means - mean).square()).sum(0) / total.clamp_min(1)
    std = variance.clamp_min(0).sqrt()
    maximum = maxima.amax(0)
    return {f'activation_{metric}/{point}': float(values[i])
            for metric, values in [('mean', mean), ('std', std), ('max', maximum)]
            for i, point in enumerate(POINTS) if total[i] > 0}


@contextmanager
def capture(config, microbatch, training):
    global _active, _collector
    if not getattr(config, 'log_activation_stats', False) or not training or microbatch != 0:
        yield
        return
    if (config.tensor_model_parallel_size != 1 or config.pipeline_model_parallel_size != 1
            or config.context_parallel_size != 1 or config.mtp_num_layers
            or config.cuda_graph_impl != 'none'):
        raise ValueError('Activation statistics currently require TP=PP=CP=1, no MTP/CUDA graphs')
    if _collector is None:
        _collector = ActivationStats(torch.cuda.current_device())
    _collector.reset()
    _active = _collector
    try:
        yield
    finally:
        _active = None


def record(point, layer, tensor):
    if _active is not None:
        _active.record(point, layer, tensor)


def write_step(writer, step, *, return_values=False):
    if _collector is not None:
        return _collector.finish(writer, step, return_values=return_values)
