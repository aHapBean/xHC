"""Fused xHC operators using the existing HyperConnection parameter layout.

The view contains references, not new Parameters. State-dict names, CPU
initialization, RNG order and optimizer grouping remain owned by HyperConnection.
"""
from types import SimpleNamespace

import torch

from .fused_xhc_kernels import (
    FusedPostFunction,
    FusedPostSplitR4Function,
    _TritonTemporalConvK4812Func,
    full_xhc_pre,
)


def _view(connection):
    cfg = connection.config
    return SimpleNamespace(
        config=SimpleNamespace(mhc_sinkhorn_iterations=cfg.sinkhorn_iterations,
                               mhc_xhc_keep_fixed_router_logits=True),
        num_fixed=cfg.fixed_streams,
        num_routed=cfg.active_streams - cfg.fixed_streams,
        active_n=cfg.active_streams,
        post_rank_for_proj=connection.rank,
        router=connection.router,
        proj_pre=connection.proj_pre,
        proj_post_res=connection.proj_post_res,
        alpha_pre=connection.alpha_pre,
        alpha_post=connection.alpha_post,
        alpha_res=connection.alpha_res,
        mhc_bias_pre=connection.bias_pre,
        mhc_bias_post=connection.bias_post,
        mhc_bias_res=connection.bias_res,
    )


def read(connection, hidden):
    from megatron.core.transformer.hyper_connections import ConnectionState

    cfg = connection.config
    if hidden.ndim != 4 or hidden.shape[-2:] != (cfg.streams, cfg.hidden_size):
        raise ValueError("Expected [sequence, batch, configured streams, hidden_size]")
    if not hidden.is_cuda or hidden.dtype != torch.bfloat16:
        raise ValueError("fused_xhc requires CUDA BF16 activations")
    for weight in (connection.router.weight, connection.proj_pre, connection.proj_post_res):
        if weight.dtype != torch.bfloat16:
            raise ValueError("fused_xhc requires BF16 projection parameters")
    pre = full_xhc_pre(_view(connection), hidden, hidden, return_residual_for_post=True)
    return pre[0], ConnectionState(pre[5], pre[1], pre[2], pre[3], pre[4])


def gram_schmidt(main, branches, precision):
    from megatron.core.transformer.hyper_connections import orthogonalize

    if (precision == "native" and main.dtype == torch.bfloat16
            and torch.__version__.startswith(("2.9.", "2.14."))
            and main.shape[-1] >= 128 and main.shape[-1] % 8 == 0
            and main.numel() // main.shape[-1] >= 16):
        from .fused_xhc_gs import NativeGS
        return NativeGS.apply(main, branches)
    return orthogonalize(main, branches, precision)


def write(connection, output, state):
    if not output.is_cuda or output.dtype != torch.bfloat16:
        raise ValueError("fused_xhc requires CUDA BF16 sublayer outputs")
    if connection.convs:
        parameters = [p for conv in connection.convs for p in (conv.weight, conv.bias)]
        branches = _TritonTemporalConvK4812Func.apply(output, *parameters)
        if connection.config.gram_schmidt:
            branches = gram_schmidt(output, branches, connection.config.gs_precision)
        return FusedPostSplitR4Function.apply(
            output, branches, state.post, state.route_weights, state.mixed,
            state.indices, state.residual, 2,
        )
    return FusedPostFunction.apply(
        output, state.post, state.route_weights, state.mixed,
        state.indices, state.residual, 1, 2,
    )
