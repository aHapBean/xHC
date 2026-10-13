"""Full dense-read xHC operators adapted from Sunan Zou's implementation.

Fused Triton kernels for projection, residual mixing and temporal convolution.
"""
import os
from contextlib import contextmanager
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
import triton
import triton.language as tl
HAS_TRITON = True


@contextmanager
def _mhc_nvtx_range(name: str):
    """Best-effort NVTX range for attributing generic cuBLAS kernels in nsys."""
    pushed = False
    try:
        enabled = os.environ.get("MHC_ENABLE_NVTX", "0").lower() in {"1", "true", "yes", "on"}
        nvtx = getattr(torch.cuda, "nvtx", None)
        if enabled and nvtx is not None and torch.cuda.is_available():
            nvtx.range_push(name)
            pushed = True
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


_XHC_BLOCK_C_COPY_CONFIGS = [
    triton.Config({"BLOCK_C": 256}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_C": 512}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_C": 1024}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_C": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 2048}, num_warps=1, num_stages=2),
]


_XHC_BLOCK_C_K4_COPY_CONFIGS = [
    triton.Config({"BLOCK_C": 128}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_C": 256}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_C": 512}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 512}, num_warps=4, num_stages=2),
]


@triton.jit
def xhc_fwd_proj_bf16_inline_matmul_kernel(
    hs_ptr,             # [M, D] bf16
    W_ptr,              # [D, OUT_DIM] bf16
    out_ptr,            # [M, OUT_DIM] fp32
    mean_ptr,           # [M] fp32
    mean_x2_ptr,        # [M] fp32
    M,
    D: tl.constexpr,
    OUT_DIM: tl.constexpr,
    stride_hs,
    PADDED_OUT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Projection forward using bf16 weights directly, matching cuBLAS precision."""
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    offs_out = tl.arange(0, PADDED_OUT)
    out_mask = offs_out < OUT_DIM

    sum_x = tl.zeros((BLOCK_T,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    h_comb = tl.zeros((BLOCK_T, PADDED_OUT), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        hs = tl.load(
            hs_ptr + offs_t[:, None] * stride_hs + d_offs[None, :],
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        hs_f32 = hs.to(tl.float32)
        sum_x += tl.sum(hs_f32, axis=1)
        sum_x2 += tl.sum(hs_f32 * hs_f32, axis=1)

        w = tl.load(
            W_ptr + d_offs[:, None] * OUT_DIM + offs_out[None, :],
            mask=d_mask[:, None] & out_mask[None, :],
            other=0.0,
        )
        h_comb += tl.dot(hs, w)

    inv_D = 1.0 / D
    tl.store(mean_ptr + offs_t, sum_x * inv_D, mask=t_mask)
    tl.store(mean_x2_ptr + offs_t, sum_x2 * inv_D, mask=t_mask)
    tl.store(
        out_ptr + offs_t[:, None] * OUT_DIM + offs_out[None, :],
        h_comb,
        mask=t_mask[:, None] & out_mask[None, :],
    )


@triton.jit
def xhc_fwd_proj_bf16_split_matmul_kernel(
    hs_ptr,             # [M, D] bf16
    W_ptr,              # [D, OUT_DIM] bf16
    partial_out_ptr,    # [NUM_D_SPLITS, M, OUT_DIM] fp32
    partial_sum_ptr,    # [NUM_D_SPLITS, M] fp32
    partial_sum2_ptr,   # [NUM_D_SPLITS, M] fp32
    M,
    D: tl.constexpr,
    OUT_DIM: tl.constexpr,
    stride_hs,
    PADDED_OUT: tl.constexpr,
    CHUNK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """D-split partial projection for the skinny full-proj workload.

    The original inline kernel has only cdiv(M, BLOCK_T) CTAs.  For the
    anchor [24576, 33792] x [33792, 32] shape that is 384 long CTAs, which
    under-fills GB200.  Splitting the D reduction produces more independent
    CTAs while multiplying bf16 activations by bf16 weights directly.
    """
    pid_m = tl.program_id(0)
    pid_split = tl.program_id(1)
    offs_t = pid_m * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    offs_out = tl.arange(0, PADDED_OUT)
    out_mask = offs_out < OUT_DIM

    split_start = pid_split * CHUNK_D
    sum_x = tl.zeros((BLOCK_T,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    h_comb = tl.zeros((BLOCK_T, PADDED_OUT), dtype=tl.float32)

    for local_start in range(0, CHUNK_D, BLOCK_D):
        d_offs = split_start + local_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        hs = tl.load(
            hs_ptr + offs_t[:, None] * stride_hs + d_offs[None, :],
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        hs_f32 = hs.to(tl.float32)
        sum_x += tl.sum(hs_f32, axis=1)
        sum_x2 += tl.sum(hs_f32 * hs_f32, axis=1)

        w = tl.load(
            W_ptr + d_offs[:, None] * OUT_DIM + offs_out[None, :],
            mask=d_mask[:, None] & out_mask[None, :],
            other=0.0,
        )
        h_comb += tl.dot(hs, w)

    partial_out_base = partial_out_ptr + pid_split * M * OUT_DIM
    tl.store(
        partial_out_base + offs_t[:, None] * OUT_DIM + offs_out[None, :],
        h_comb,
        mask=t_mask[:, None] & out_mask[None, :],
    )
    tl.store(partial_sum_ptr + pid_split * M + offs_t, sum_x, mask=t_mask)
    tl.store(partial_sum2_ptr + pid_split * M + offs_t, sum_x2, mask=t_mask)


@triton.jit
def xhc_fwd_proj_bf16_split_reduce_kernel(
    partial_out_ptr,    # [NUM_D_SPLITS, M, OUT_DIM] fp32
    partial_sum_ptr,    # [NUM_D_SPLITS, M] fp32
    partial_sum2_ptr,   # [NUM_D_SPLITS, M] fp32
    out_ptr,            # [M, OUT_DIM] fp32
    mean_ptr,           # [M] fp32
    mean_x2_ptr,        # [M] fp32
    M,
    D: tl.constexpr,
    OUT_DIM: tl.constexpr,
    PADDED_OUT: tl.constexpr,
    NUM_D_SPLITS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_t = pid_m * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    offs_out = tl.arange(0, PADDED_OUT)
    out_mask = offs_out < OUT_DIM

    sum_x = tl.zeros((BLOCK_T,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    h_comb = tl.zeros((BLOCK_T, PADDED_OUT), dtype=tl.float32)

    for split in tl.static_range(NUM_D_SPLITS):
        partial_out_base = partial_out_ptr + split * M * OUT_DIM
        h_comb += tl.load(
            partial_out_base + offs_t[:, None] * OUT_DIM + offs_out[None, :],
            mask=t_mask[:, None] & out_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        sum_x += tl.load(
            partial_sum_ptr + split * M + offs_t,
            mask=t_mask,
            other=0.0,
        ).to(tl.float32)
        sum_x2 += tl.load(
            partial_sum2_ptr + split * M + offs_t,
            mask=t_mask,
            other=0.0,
        ).to(tl.float32)

    inv_D = 1.0 / D
    tl.store(mean_ptr + offs_t, sum_x * inv_D, mask=t_mask)
    tl.store(mean_x2_ptr + offs_t, sum_x2 * inv_D, mask=t_mask)
    tl.store(
        out_ptr + offs_t[:, None] * OUT_DIM + offs_out[None, :],
        h_comb,
        mask=t_mask[:, None] & out_mask[None, :],
    )


@triton.jit
def xhc_sinkhorn_kernel(
    logits_ptr,     # [M, K, K] fp32 — h_res logits
    output_ptr,     # [M, K, K] fp32 — doubly stochastic matrix
    M,
    K: tl.constexpr,        # 4
    K_SQ: tl.constexpr,     # 16
    SINKHORN_ITERS: tl.constexpr,  # 20
    CLAMP_ROW_SUM: tl.constexpr,   # 1 = enable row_sum_clamp
):
    """
    Per-token Sinkhorn in register. [K,K]=[4,4] tensor fits in registers.
    1 kernel launch replaces 41+ PyTorch launches.
    Grid: (M,)
    """
    pid = tl.program_id(0)
    offs_r = tl.arange(0, K)
    offs_c = tl.arange(0, K)

    h_res_2d = tl.load(
        logits_ptr + pid * K_SQ + offs_r[:, None] * K + offs_c[None, :]
    ).to(tl.float32)

    for _ in range(SINKHORN_ITERS):
        row_max = tl.max(h_res_2d, axis=1)
        row_sum = tl.sum(tl.exp(h_res_2d - row_max[:, None]), axis=1)
        row_lse = tl.log(row_sum) + row_max
        h_res_2d = h_res_2d - row_lse[:, None]

        col_max = tl.max(h_res_2d, axis=0)
        col_sum = tl.sum(tl.exp(h_res_2d - col_max[None, :]), axis=0)
        col_lse = tl.log(col_sum) + col_max
        h_res_2d = h_res_2d - col_lse[None, :]

    h_res_2d = tl.exp(h_res_2d)

    if CLAMP_ROW_SUM:
        row_sum_final = tl.sum(h_res_2d, axis=1)
        clamp = tl.where(row_sum_final > 1.0, row_sum_final, 1.0)
        h_res_2d = h_res_2d / clamp[:, None]

    tl.store(
        output_ptr + pid * K_SQ + offs_r[:, None] * K + offs_c[None, :],
        h_res_2d,
    )


@triton.jit
def xhc_sinkhorn_bwd_kernel(
    logits_ptr,         # [M, K_SQ] fp32 — original logits (flattened [K,K])
    h_res_ptr,          # [M, K_SQ] fp32 — forward output (saved)
    grad_out_ptr,       # [M, K_SQ] fp32 — grad of output (after row_sum_clamp)
    grad_logits_ptr,    # [M, K_SQ] fp32 — output grad
    sk_buf_ptr,         # [M * SK_BUF_STRIDE] fp32 — scratchpad for intermediate states
    M,
    K: tl.constexpr,
    K_SQ: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    CLAMP_ROW_SUM: tl.constexpr,
    SK_BUF_STRIDE: tl.constexpr,  # = 2 * SINKHORN_ITERS * K_SQ
):
    """
    Sinkhorn backward via forward recompute + scratchpad (Expert pattern).
    Stores intermediate states during forward, then backward in reverse order.
    Grid: (M,)
    """
    pid = tl.program_id(0)
    offs_nsq = tl.arange(0, K_SQ)
    zero_nsq = tl.zeros((K_SQ,), dtype=tl.float32)

    # Load logits as flat [K_SQ]
    states = tl.load(logits_ptr + pid * K_SQ + offs_nsq).to(tl.float32)

    # --- Forward recompute: store intermediate states to scratchpad ---
    sk_base = sk_buf_ptr + pid * SK_BUF_STRIDE
    for iter_idx in range(SINKHORN_ITERS):
        # Row normalization
        for row in range(K):
            row_start = row * K
            row_mask = (offs_nsq >= row_start) & (offs_nsq < row_start + K)
            masked_vals = tl.where(row_mask, states, -1e30)
            row_max = tl.max(masked_vals)
            row_sum = tl.sum(tl.where(row_mask, tl.exp(states - row_max), zero_nsq))
            row_lse = tl.log(row_sum) + row_max
            states = tl.where(row_mask, states - row_lse, states)
        tl.store(sk_base + (2 * iter_idx) * K_SQ + offs_nsq, states)

        # Column normalization
        for col in range(K):
            col_mask = (offs_nsq % K) == col
            masked_vals = tl.where(col_mask, states, -1e30)
            col_max = tl.max(masked_vals)
            col_sum = tl.sum(tl.where(col_mask, tl.exp(states - col_max), zero_nsq))
            col_lse = tl.log(col_sum) + col_max
            states = tl.where(col_mask, states - col_lse, states)
        tl.store(sk_base + (2 * iter_idx + 1) * K_SQ + offs_nsq, states)

    # Reconstruct the *unclamped* matrix. The saved output has already
    # been divided by the row sum, so it cannot recover this derivative.
    h_res = tl.exp(states)
    grad_out = tl.load(grad_out_ptr + pid * K_SQ + offs_nsq).to(tl.float32)
    if CLAMP_ROW_SUM:
        for row in range(K):
            row_start = row * K
            row_mask = (offs_nsq >= row_start) & (offs_nsq < row_start + K)
            row_sum = tl.sum(tl.where(row_mask, h_res, zero_nsq))
            clamp_val = tl.maximum(row_sum, 1.0)
            row_dot = tl.sum(tl.where(row_mask, grad_out * h_res, zero_nsq))
            row_grad = grad_out / clamp_val
            # torch.clamp_min propagates the derivative at equality, too.
            row_grad -= tl.where(row_sum >= 1.0, row_dot / (clamp_val * clamp_val), 0.0)
            grad_out = tl.where(row_mask, row_grad, grad_out)

    # --- Backward through exp: grad_states = grad_out * exp(states_final) ---
    grad_sk = grad_out * tl.exp(states)

    # --- Backward through Sinkhorn iterations (reverse order) ---
    for i_inv in range(SINKHORN_ITERS):
        i = SINKHORN_ITERS - 1 - i_inv

        # Backward through column normalization: y = x - logsumexp(x, col)
        # grad_x = grad_y - softmax_col * sum(grad_y, col)
        col_probs = tl.exp(tl.load(sk_base + (2 * i + 1) * K_SQ + offs_nsq))
        for col in range(K):
            col_mask = (offs_nsq % K) == col
            col_grad_sum = tl.sum(tl.where(col_mask, grad_sk, zero_nsq))
            grad_sk = grad_sk - tl.where(col_mask, col_probs * col_grad_sum, zero_nsq)

        # Backward through row normalization: y = x - logsumexp(x, row)
        # grad_x = grad_y - softmax_row * sum(grad_y, row)
        row_probs = tl.exp(tl.load(sk_base + (2 * i) * K_SQ + offs_nsq))
        for row in range(K):
            row_start = row * K
            row_mask = (offs_nsq >= row_start) & (offs_nsq < row_start + K)
            row_grad_sum = tl.sum(tl.where(row_mask, grad_sk, zero_nsq))
            grad_sk = grad_sk - tl.where(row_mask, row_probs * row_grad_sum, zero_nsq)

    tl.store(grad_logits_ptr + pid * K_SQ + offs_nsq, grad_sk)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 128}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_C": 512}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_C": 1024}, num_warps=8, num_stages=3),
    ],
    key=["C", "N"],
)
@triton.jit
def xhc_pre_res_kernel(
    hs_ptr,             # [M, N, C] bf16 — hidden_states
    h_pre_ptr,          # [M, N] fp32 — h_pre weights
    h_res_ptr,          # [M, K, K] fp32 — h_res matrix
    topk_ptr,           # [M, K] int64 — topk indices
    out_weighted_ptr,   # [M, C] output — hidden_for_sub
    out_residual_ptr,   # [M, K, C] output — mixed_active
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    C,
    stride_hs_m,
    stride_hs_n,
    STORE_DTYPE: tl.constexpr,  # 0=fp32, 1=bf16, 2=fp16
    BLOCK_C: tl.constexpr,
):
    """
    Fused h_pre aggregation (dense read, N=16) + h_res mixing (sparse, K=4).
    1 pass over hidden_states, 2D grid (M, cdiv(C, BLOCK_C)).
    Adapted from Expert mhc_pre_res_kernel + topk sparse support.
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    h_pre_base = h_pre_ptr + pid_m * N
    h_res_base = h_res_ptr + pid_m * K * K
    hs_base = hs_ptr + pid_m * stride_hs_m

    # Load topk indices [K]
    tk0 = tl.load(topk_ptr + pid_m * K + 0)
    tk1 = tl.load(topk_ptr + pid_m * K + 1)
    tk2 = tl.load(topk_ptr + pid_m * K + 2)
    tk3 = tl.load(topk_ptr + pid_m * K + 3)

    weighted_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Accumulators for h_res mixing: mixed[k, c] = sum_j(h_res[k,j] * hs[topk[j], c])
    mixed0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    mixed1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    mixed2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    mixed3 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Load h_res [K, K] = [4, 4] scalars
    hr00 = tl.load(h_res_base + 0); hr01 = tl.load(h_res_base + 1)
    hr02 = tl.load(h_res_base + 2); hr03 = tl.load(h_res_base + 3)
    hr10 = tl.load(h_res_base + 4); hr11 = tl.load(h_res_base + 5)
    hr12 = tl.load(h_res_base + 6); hr13 = tl.load(h_res_base + 7)
    hr20 = tl.load(h_res_base + 8); hr21 = tl.load(h_res_base + 9)
    hr22 = tl.load(h_res_base + 10); hr23 = tl.load(h_res_base + 11)
    hr30 = tl.load(h_res_base + 12); hr31 = tl.load(h_res_base + 13)
    hr32 = tl.load(h_res_base + 14); hr33 = tl.load(h_res_base + 15)

    for n in tl.static_range(N):
        hs_block = tl.load(
            hs_base + n * stride_hs_n + c_offs,
            mask=c_mask, other=0.0,
        ).to(tl.float32)

        h_pre_n = tl.load(h_pre_base + n).to(tl.float32)
        weighted_sum += h_pre_n * hs_block

        # h_res mixing: if this stream n is topk[j], accumulate into mixed[k]
        # topk[j] → column j of h_res. Each row k gets h_res[k,j] * hs[n,:]
        if n == tk0:
            mixed0 += hr00 * hs_block; mixed1 += hr10 * hs_block
            mixed2 += hr20 * hs_block; mixed3 += hr30 * hs_block
        if n == tk1:
            mixed0 += hr01 * hs_block; mixed1 += hr11 * hs_block
            mixed2 += hr21 * hs_block; mixed3 += hr31 * hs_block
        if n == tk2:
            mixed0 += hr02 * hs_block; mixed1 += hr12 * hs_block
            mixed2 += hr22 * hs_block; mixed3 += hr32 * hs_block
        if n == tk3:
            mixed0 += hr03 * hs_block; mixed1 += hr13 * hs_block
            mixed2 += hr23 * hs_block; mixed3 += hr33 * hs_block

    # Store weighted sum
    out_w = out_weighted_ptr + pid_m * C
    if STORE_DTYPE == 1:
        tl.store(out_w + c_offs, weighted_sum.to(tl.bfloat16), mask=c_mask)
    elif STORE_DTYPE == 2:
        tl.store(out_w + c_offs, weighted_sum.to(tl.float16), mask=c_mask)
    else:
        tl.store(out_w + c_offs, weighted_sum, mask=c_mask)

    # Store mixed residuals
    out_r = out_residual_ptr + pid_m * K * C
    if STORE_DTYPE == 1:
        tl.store(out_r + 0 * C + c_offs, mixed0.to(tl.bfloat16), mask=c_mask)
        tl.store(out_r + 1 * C + c_offs, mixed1.to(tl.bfloat16), mask=c_mask)
        tl.store(out_r + 2 * C + c_offs, mixed2.to(tl.bfloat16), mask=c_mask)
        tl.store(out_r + 3 * C + c_offs, mixed3.to(tl.bfloat16), mask=c_mask)
    else:
        tl.store(out_r + 0 * C + c_offs, mixed0, mask=c_mask)
        tl.store(out_r + 1 * C + c_offs, mixed1, mask=c_mask)
        tl.store(out_r + 2 * C + c_offs, mixed2, mask=c_mask)
        tl.store(out_r + 3 * C + c_offs, mixed3, mask=c_mask)


@triton.jit
def xhc_pre_res_active_kernel(
    hs_ptr,             # [M, N, C]
    active_ptr,         # [M, K, C], gathered hidden_states[topk]
    topk_ptr,           # [M, K]
    h_pre_ptr,          # [M, N]
    h_res_ptr,          # [M, K, K]
    out_weighted_ptr,   # [M, C]
    out_residual_ptr,   # [M, K, C]
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    C,
    stride_hs_m,
    stride_hs_n,
    STORE_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """pre_res forward that reuses the active gather output for h_res mixing."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    hs_base = hs_ptr + pid_m * stride_hs_m
    h_pre_base = h_pre_ptr + pid_m * N
    h_res_base = h_res_ptr + pid_m * K * K

    weighted_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for n in tl.static_range(N):
        hs_block = tl.load(
            hs_base + n * stride_hs_n + c_offs,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
        h_pre_n = tl.load(h_pre_base + n).to(tl.float32)
        weighted_sum += h_pre_n * hs_block

    act_base = active_ptr + pid_m * K * C
    a0 = tl.load(act_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    a1 = tl.load(act_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    a2 = tl.load(act_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    a3 = tl.load(act_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    hr00 = tl.load(h_res_base + 0); hr01 = tl.load(h_res_base + 1)
    hr02 = tl.load(h_res_base + 2); hr03 = tl.load(h_res_base + 3)
    hr10 = tl.load(h_res_base + 4); hr11 = tl.load(h_res_base + 5)
    hr12 = tl.load(h_res_base + 6); hr13 = tl.load(h_res_base + 7)
    hr20 = tl.load(h_res_base + 8); hr21 = tl.load(h_res_base + 9)
    hr22 = tl.load(h_res_base + 10); hr23 = tl.load(h_res_base + 11)
    hr30 = tl.load(h_res_base + 12); hr31 = tl.load(h_res_base + 13)
    hr32 = tl.load(h_res_base + 14); hr33 = tl.load(h_res_base + 15)

    mixed0 = hr00 * a0 + hr01 * a1 + hr02 * a2 + hr03 * a3
    mixed1 = hr10 * a0 + hr11 * a1 + hr12 * a2 + hr13 * a3
    mixed2 = hr20 * a0 + hr21 * a1 + hr22 * a2 + hr23 * a3
    mixed3 = hr30 * a0 + hr31 * a1 + hr32 * a2 + hr33 * a3

    out_w = out_weighted_ptr + pid_m * C
    out_r = out_residual_ptr + pid_m * K * C
    if STORE_DTYPE == 1:
        tl.store(out_w + c_offs, weighted_sum.to(tl.bfloat16), mask=c_mask)
        tl.store(out_r + 0 * C + c_offs, mixed0.to(tl.bfloat16), mask=c_mask)
        tl.store(out_r + 1 * C + c_offs, mixed1.to(tl.bfloat16), mask=c_mask)
        tl.store(out_r + 2 * C + c_offs, mixed2.to(tl.bfloat16), mask=c_mask)
        tl.store(out_r + 3 * C + c_offs, mixed3.to(tl.bfloat16), mask=c_mask)
    elif STORE_DTYPE == 2:
        tl.store(out_w + c_offs, weighted_sum.to(tl.float16), mask=c_mask)
        tl.store(out_r + 0 * C + c_offs, mixed0.to(tl.float16), mask=c_mask)
        tl.store(out_r + 1 * C + c_offs, mixed1.to(tl.float16), mask=c_mask)
        tl.store(out_r + 2 * C + c_offs, mixed2.to(tl.float16), mask=c_mask)
        tl.store(out_r + 3 * C + c_offs, mixed3.to(tl.float16), mask=c_mask)
    else:
        tl.store(out_w + c_offs, weighted_sum, mask=c_mask)
        tl.store(out_r + 0 * C + c_offs, mixed0, mask=c_mask)
        tl.store(out_r + 1 * C + c_offs, mixed1, mask=c_mask)
        tl.store(out_r + 2 * C + c_offs, mixed2, mask=c_mask)
        tl.store(out_r + 3 * C + c_offs, mixed3, mask=c_mask)


@triton.jit
def xhc_pre_res_bwd_kernel(
    hs_ptr, h_pre_ptr, h_res_ptr, topk_ptr,
    grad_sub_ptr, grad_mixed_ptr,
    grad_residual_ptr,
    grad_hs_ptr,
    scratch_ghp_ptr,    # [M, NUM_C_TILES, N] fp32 — partial grad_h_pre
    scratch_ghr_ptr,    # [M, NUM_C_TILES, K*K] fp32 — partial grad_h_res
    M,
    N: tl.constexpr, K: tl.constexpr, C,
    stride_hs_m, stride_hs_n,
    NUM_C_TILES: tl.constexpr,
    HAS_GRAD_RESIDUAL: tl.constexpr,
    STORE_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Phase 1: 2D grid (M, cdiv(C, BLOCK_C)). Each program processes 1 token × 1 C-tile.
    Writes grad_hs directly; writes partial grad_h_pre/grad_h_res to scratch.
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    hs_base = hs_ptr + pid_m * stride_hs_m
    h_pre_base = h_pre_ptr + pid_m * N
    h_res_base = h_res_ptr + pid_m * K * K
    grad_sub_base = grad_sub_ptr + pid_m * C
    grad_mixed_base = grad_mixed_ptr + pid_m * K * C
    grad_residual_base = grad_residual_ptr + pid_m * N * C
    grad_hs_base = grad_hs_ptr + pid_m * N * C

    tk0 = tl.load(topk_ptr + pid_m * K + 0)
    tk1 = tl.load(topk_ptr + pid_m * K + 1)
    tk2 = tl.load(topk_ptr + pid_m * K + 2)
    tk3 = tl.load(topk_ptr + pid_m * K + 3)

    # h_pre is loaded inside the N loop; fused xHC runtime is fixed to N=16.

    # Load h_res [K*K=16] scalars
    hr00 = tl.load(h_res_base + 0).to(tl.float32); hr01 = tl.load(h_res_base + 1).to(tl.float32)
    hr02 = tl.load(h_res_base + 2).to(tl.float32); hr03 = tl.load(h_res_base + 3).to(tl.float32)
    hr10 = tl.load(h_res_base + 4).to(tl.float32); hr11 = tl.load(h_res_base + 5).to(tl.float32)
    hr12 = tl.load(h_res_base + 6).to(tl.float32); hr13 = tl.load(h_res_base + 7).to(tl.float32)
    hr20 = tl.load(h_res_base + 8).to(tl.float32); hr21 = tl.load(h_res_base + 9).to(tl.float32)
    hr22 = tl.load(h_res_base + 10).to(tl.float32); hr23 = tl.load(h_res_base + 11).to(tl.float32)
    hr30 = tl.load(h_res_base + 12).to(tl.float32); hr31 = tl.load(h_res_base + 13).to(tl.float32)
    hr32 = tl.load(h_res_base + 14).to(tl.float32); hr33 = tl.load(h_res_base + 15).to(tl.float32)

    grad_sub = tl.load(grad_sub_base + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gm0 = tl.load(grad_mixed_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gm1 = tl.load(grad_mixed_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gm2 = tl.load(grad_mixed_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gm3 = tl.load(grad_mixed_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    grad_topk0 = hr00 * gm0 + hr10 * gm1 + hr20 * gm2 + hr30 * gm3
    grad_topk1 = hr01 * gm0 + hr11 * gm1 + hr21 * gm2 + hr31 * gm3
    grad_topk2 = hr02 * gm0 + hr12 * gm1 + hr22 * gm2 + hr32 * gm3
    grad_topk3 = hr03 * gm0 + hr13 * gm1 + hr23 * gm2 + hr33 * gm3

    scratch_ghp_base = scratch_ghp_ptr + (pid_m * NUM_C_TILES + pid_c) * N
    scratch_ghr_base = scratch_ghr_ptr + (pid_m * NUM_C_TILES + pid_c) * (K * K)

    for n in tl.static_range(N):
        hs_n = tl.load(
            hs_base + n * stride_hs_n + c_offs, mask=c_mask, other=0.0
        ).to(tl.float32)

        h_pre_n = tl.load(h_pre_base + n).to(tl.float32)
        grad_hs_n = h_pre_n * grad_sub

        s0 = tl.where(tk0 == n, 1.0, 0.0)
        s1 = tl.where(tk1 == n, 1.0, 0.0)
        s2 = tl.where(tk2 == n, 1.0, 0.0)
        s3 = tl.where(tk3 == n, 1.0, 0.0)
        grad_hs_n += s0 * grad_topk0 + s1 * grad_topk1 + s2 * grad_topk2 + s3 * grad_topk3
        if HAS_GRAD_RESIDUAL:
            grad_hs_n += tl.load(
                grad_residual_base + n * C + c_offs,
                mask=c_mask,
                other=0.0,
            ).to(tl.float32)

        out_ptr = grad_hs_base + n * C + c_offs
        if STORE_DTYPE == 1:
            tl.store(out_ptr, grad_hs_n.to(tl.bfloat16), mask=c_mask)
        elif STORE_DTYPE == 2:
            tl.store(out_ptr, grad_hs_n.to(tl.float16), mask=c_mask)
        else:
            tl.store(out_ptr, grad_hs_n, mask=c_mask)

        tl.store(scratch_ghp_base + n, tl.sum(grad_sub * hs_n))

    # K=4 loop: partial grad_h_res
    for j in tl.static_range(K):
        if j == 0: tk_j = tk0
        elif j == 1: tk_j = tk1
        elif j == 2: tk_j = tk2
        else: tk_j = tk3

        hs_j = tl.load(
            hs_base + tk_j * stride_hs_n + c_offs, mask=c_mask, other=0.0
        ).to(tl.float32)

        tl.store(scratch_ghr_base + 0 * K + j, tl.sum(gm0 * hs_j))
        tl.store(scratch_ghr_base + 1 * K + j, tl.sum(gm1 * hs_j))
        tl.store(scratch_ghr_base + 2 * K + j, tl.sum(gm2 * hs_j))
        tl.store(scratch_ghr_base + 3 * K + j, tl.sum(gm3 * hs_j))


@triton.jit
def xhc_pre_res_bwd_reduce_kernel(
    scratch_ghp_ptr,    # [M, NUM_C_TILES, N] fp32
    scratch_ghr_ptr,    # [M, NUM_C_TILES, K*K] fp32
    grad_h_pre_ptr,     # [M, N] fp32 output
    grad_h_res_ptr,     # [M, K*K] fp32 output
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_C_TILES: tl.constexpr,
):
    """Phase 2: 1D grid (M,). Reduces partial sums → final grad_h_pre, grad_h_res."""
    pid = tl.program_id(0)
    K_SQ: tl.constexpr = K * K
    ghp_base = scratch_ghp_ptr + pid * NUM_C_TILES * N
    ghr_base = scratch_ghr_ptr + pid * NUM_C_TILES * K_SQ

    for n in tl.static_range(N):
        acc = 0.0
        for t in tl.static_range(NUM_C_TILES):
            acc += tl.load(ghp_base + t * N + n)
        tl.store(grad_h_pre_ptr + pid * N + n, acc)

    for idx in tl.static_range(K_SQ):
        acc = 0.0
        for t in tl.static_range(NUM_C_TILES):
            acc += tl.load(ghr_base + t * K_SQ + idx)
        tl.store(grad_h_res_ptr + pid * K_SQ + idx, acc)


@triton.jit
def xhc_pre_res_bwd_grad_hs_kernel(
    h_pre_ptr, h_res_ptr, topk_ptr,
    grad_sub_ptr, grad_mixed_ptr, grad_residual_ptr,
    grad_hs_ptr,
    M,
    N: tl.constexpr, K: tl.constexpr, C,
    HAS_GRAD_RESIDUAL: tl.constexpr,
    STORE_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    h_pre_base = h_pre_ptr + pid_m * N
    h_res_base = h_res_ptr + pid_m * K * K
    topk_base = topk_ptr + pid_m * K
    grad_sub_base = grad_sub_ptr + pid_m * C
    grad_mixed_base = grad_mixed_ptr + pid_m * K * C
    grad_residual_base = grad_residual_ptr + pid_m * N * C
    grad_hs_base = grad_hs_ptr + pid_m * N * C

    tk0 = tl.load(topk_base + 0)
    tk1 = tl.load(topk_base + 1)
    tk2 = tl.load(topk_base + 2)
    tk3 = tl.load(topk_base + 3)

    hr00 = tl.load(h_res_base + 0).to(tl.float32); hr01 = tl.load(h_res_base + 1).to(tl.float32)
    hr02 = tl.load(h_res_base + 2).to(tl.float32); hr03 = tl.load(h_res_base + 3).to(tl.float32)
    hr10 = tl.load(h_res_base + 4).to(tl.float32); hr11 = tl.load(h_res_base + 5).to(tl.float32)
    hr12 = tl.load(h_res_base + 6).to(tl.float32); hr13 = tl.load(h_res_base + 7).to(tl.float32)
    hr20 = tl.load(h_res_base + 8).to(tl.float32); hr21 = tl.load(h_res_base + 9).to(tl.float32)
    hr22 = tl.load(h_res_base + 10).to(tl.float32); hr23 = tl.load(h_res_base + 11).to(tl.float32)
    hr30 = tl.load(h_res_base + 12).to(tl.float32); hr31 = tl.load(h_res_base + 13).to(tl.float32)
    hr32 = tl.load(h_res_base + 14).to(tl.float32); hr33 = tl.load(h_res_base + 15).to(tl.float32)

    grad_sub = tl.load(grad_sub_base + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gm0 = tl.load(grad_mixed_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gm1 = tl.load(grad_mixed_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gm2 = tl.load(grad_mixed_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gm3 = tl.load(grad_mixed_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    grad_topk0 = hr00 * gm0 + hr10 * gm1 + hr20 * gm2 + hr30 * gm3
    grad_topk1 = hr01 * gm0 + hr11 * gm1 + hr21 * gm2 + hr31 * gm3
    grad_topk2 = hr02 * gm0 + hr12 * gm1 + hr22 * gm2 + hr32 * gm3
    grad_topk3 = hr03 * gm0 + hr13 * gm1 + hr23 * gm2 + hr33 * gm3

    for n in tl.static_range(N):
        h_pre_n = tl.load(h_pre_base + n).to(tl.float32)
        grad_hs_n = h_pre_n * grad_sub
        grad_hs_n += tl.where(tk0 == n, grad_topk0, 0.0)
        grad_hs_n += tl.where(tk1 == n, grad_topk1, 0.0)
        grad_hs_n += tl.where(tk2 == n, grad_topk2, 0.0)
        grad_hs_n += tl.where(tk3 == n, grad_topk3, 0.0)
        if HAS_GRAD_RESIDUAL:
            grad_hs_n += tl.load(
                grad_residual_base + n * C + c_offs,
                mask=c_mask,
                other=0.0,
            ).to(tl.float32)
        out_ptr = grad_hs_base + n * C + c_offs
        if STORE_DTYPE == 1:
            tl.store(out_ptr, grad_hs_n.to(tl.bfloat16), mask=c_mask)
        elif STORE_DTYPE == 2:
            tl.store(out_ptr, grad_hs_n.to(tl.float16), mask=c_mask)
        else:
            tl.store(out_ptr, grad_hs_n, mask=c_mask)


@triton.jit
def xhc_pre_res_bwd_ghp_kernel(
    hs_ptr, grad_sub_ptr, scratch_ghp_ptr,
    M,
    N: tl.constexpr, C,
    stride_hs_m, stride_hs_n,
    NUM_C_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    hs_base = hs_ptr + pid_m * stride_hs_m
    grad_sub = tl.load(grad_sub_ptr + pid_m * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    scratch_base = scratch_ghp_ptr + (pid_m * NUM_C_TILES + pid_c) * N
    for n in tl.static_range(N):
        hs_n = tl.load(hs_base + n * stride_hs_n + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(scratch_base + n, tl.sum(grad_sub * hs_n))


@triton.jit
def xhc_pre_res_bwd_ghr_active_kernel(
    active_ptr, grad_mixed_ptr, scratch_ghr_ptr,
    M,
    K: tl.constexpr, C,
    NUM_C_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    active_base = active_ptr + pid_m * K * C
    grad_mixed_base = grad_mixed_ptr + pid_m * K * C
    scratch_base = scratch_ghr_ptr + (pid_m * NUM_C_TILES + pid_c) * (K * K)

    gm0 = tl.load(grad_mixed_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gm1 = tl.load(grad_mixed_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gm2 = tl.load(grad_mixed_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    gm3 = tl.load(grad_mixed_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    a0 = tl.load(active_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    a1 = tl.load(active_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    a2 = tl.load(active_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    a3 = tl.load(active_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    tl.store(scratch_base + 0 * K + 0, tl.sum(gm0 * a0))
    tl.store(scratch_base + 0 * K + 1, tl.sum(gm0 * a1))
    tl.store(scratch_base + 0 * K + 2, tl.sum(gm0 * a2))
    tl.store(scratch_base + 0 * K + 3, tl.sum(gm0 * a3))
    tl.store(scratch_base + 1 * K + 0, tl.sum(gm1 * a0))
    tl.store(scratch_base + 1 * K + 1, tl.sum(gm1 * a1))
    tl.store(scratch_base + 1 * K + 2, tl.sum(gm1 * a2))
    tl.store(scratch_base + 1 * K + 3, tl.sum(gm1 * a3))
    tl.store(scratch_base + 2 * K + 0, tl.sum(gm2 * a0))
    tl.store(scratch_base + 2 * K + 1, tl.sum(gm2 * a1))
    tl.store(scratch_base + 2 * K + 2, tl.sum(gm2 * a2))
    tl.store(scratch_base + 2 * K + 3, tl.sum(gm2 * a3))
    tl.store(scratch_base + 3 * K + 0, tl.sum(gm3 * a0))
    tl.store(scratch_base + 3 * K + 1, tl.sum(gm3 * a1))
    tl.store(scratch_base + 3 * K + 2, tl.sum(gm3 * a2))
    tl.store(scratch_base + 3 * K + 3, tl.sum(gm3 * a3))


@triton.autotune(
    configs=_XHC_BLOCK_C_COPY_CONFIGS,
    key=["C"],
)
@triton.jit
def xhc_active_gather_kernel(
    hs_ptr,         # [M, N, C]
    topk_ptr,       # [M, K]
    active_ptr,     # [M, K, C]
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    C,
    BLOCK_C: tl.constexpr,
):
    """Gather active residual streams without PyTorch gather metadata overhead."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_c = tl.program_id(2)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    tk = tl.load(topk_ptr + pid_m * K + pid_k)
    val = tl.load(hs_ptr + (pid_m * N + tk) * C + c_offs, mask=c_mask, other=0.0)
    tl.store(active_ptr + (pid_m * K + pid_k) * C + c_offs, val, mask=c_mask)


@triton.autotune(
    configs=_XHC_BLOCK_C_K4_COPY_CONFIGS,
    key=["C"],
)
@triton.jit
def xhc_active_gather_k4_stats_kernel(
    hs_ptr,             # [M, N, C]
    topk_ptr,           # [M, 4]
    active_ptr,         # [M, 4, C]
    stats_ptr,          # [M, NUM_C_TILES] partial sumsq or [M, 1] mean_x2
    M,
    N: tl.constexpr,
    C,
    NUM_C_TILES: tl.constexpr,
    DIRECT_MEAN: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """K=4 gather that also accumulates per-row active sumsq partials."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    k_offs = tl.arange(0, 4)
    tk = tl.load(topk_ptr + pid_m * 4 + k_offs)
    vals = tl.load(
        hs_ptr + (pid_m * N + tk[:, None]) * C + c_offs[None, :],
        mask=c_mask[None, :],
        other=0.0,
    )
    tl.store(
        active_ptr + (pid_m * 4 + k_offs[:, None]) * C + c_offs[None, :],
        vals,
        mask=c_mask[None, :],
    )
    vals_f = vals.to(tl.float32)
    per_k = tl.sum(vals_f * vals_f, axis=1)
    sumsq = tl.sum(per_k, axis=0)
    if DIRECT_MEAN:
        tl.store(stats_ptr + pid_m, sumsq / (4 * C))
    else:
        tl.store(stats_ptr + pid_m * NUM_C_TILES + pid_c, sumsq)


@triton.jit
def xhc_active_gather_stats_reduce_kernel(
    partial_sumsq_ptr,  # [M, NUM_C_TILES]
    mean_x2_ptr,        # [M, 1] contiguous
    M,
    K: tl.constexpr,
    C,
    NUM_C_TILES: tl.constexpr,
    BLOCK_CT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_CT)
    mask = offs < NUM_C_TILES
    vals = tl.load(partial_sumsq_ptr + pid_m * NUM_C_TILES + offs, mask=mask, other=0.0)
    sumsq = tl.sum(vals, axis=0)
    tl.store(mean_x2_ptr + pid_m, sumsq / (K * C))


@triton.autotune(
    configs=_XHC_BLOCK_C_COPY_CONFIGS,
    key=["C", "N"],
)

@triton.jit
def xhc_active_gather_bwd_passthrough_kernel(
    grad_active_ptr,        # [M, K, C]
    topk_ptr,               # [M, K]
    grad_passthrough_ptr,   # [M, N, C], optional
    grad_hs_ptr,            # [M, N, C]
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    C,
    HAS_GRAD_PASSTHROUGH: tl.constexpr,
    STORE_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Add sparse active-gather gradients into a dense passthrough gradient."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_c = tl.program_id(2)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    tk0 = tl.load(topk_ptr + pid_m * K + 0)
    tk1 = tl.load(topk_ptr + pid_m * K + 1)
    tk2 = tl.load(topk_ptr + pid_m * K + 2)
    tk3 = tl.load(topk_ptr + pid_m * K + 3)

    if HAS_GRAD_PASSTHROUGH:
        val = tl.load(
            grad_passthrough_ptr + (pid_m * N + pid_n) * C + c_offs,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        val = tl.zeros((BLOCK_C,), dtype=tl.float32)

    ga_base = grad_active_ptr + pid_m * K * C
    is0 = tk0 == pid_n
    is1 = tk1 == pid_n
    is2 = tk2 == pid_n
    is3 = tk3 == pid_n
    val += tl.load(ga_base + 0 * C + c_offs, mask=c_mask & is0, other=0.0).to(tl.float32)
    val += tl.load(ga_base + 1 * C + c_offs, mask=c_mask & is1, other=0.0).to(tl.float32)
    val += tl.load(ga_base + 2 * C + c_offs, mask=c_mask & is2, other=0.0).to(tl.float32)
    val += tl.load(ga_base + 3 * C + c_offs, mask=c_mask & is3, other=0.0).to(tl.float32)

    out_ptr = grad_hs_ptr + (pid_m * N + pid_n) * C + c_offs
    if STORE_DTYPE == 1:
        tl.store(out_ptr, val.to(tl.bfloat16), mask=c_mask)
    elif STORE_DTYPE == 2:
        tl.store(out_ptr, val.to(tl.float16), mask=c_mask)
    else:
        tl.store(out_ptr, val, mask=c_mask)


@triton.jit
def xhc_active_gather_bwd_passthrough_inplace_kernel(
    grad_active_ptr,        # [M, K, C]
    topk_ptr,               # [M, K]
    grad_passthrough_ptr,   # [M, N, C], updated in-place
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    C,
    STORE_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Add active-gather gradients into an existing dense passthrough grad.

    The previous passthrough kernel copied every [M,N,C] element into a new
    output before adding the K active rows.  In the common training path the
    passthrough gradient is already the dense output we need, so this kernel
    only touches the K active rows and returns that buffer as grad_hidden.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_c = tl.program_id(2)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    tk = tl.load(topk_ptr + pid_m * K + pid_k)
    active = tl.load(
        grad_active_ptr + (pid_m * K + pid_k) * C + c_offs,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    out_ptr = grad_passthrough_ptr + (pid_m * N + tk) * C + c_offs
    base = tl.load(out_ptr, mask=c_mask, other=0.0).to(tl.float32)
    val = base + active
    if STORE_DTYPE == 1:
        tl.store(out_ptr, val.to(tl.bfloat16), mask=c_mask)
    elif STORE_DTYPE == 2:
        tl.store(out_ptr, val.to(tl.float16), mask=c_mask)
    else:
        tl.store(out_ptr, val, mask=c_mask)


@triton.jit
def xhc_active_gather_bwd_passthrough_inplace_k4_kernel(
    grad_active_ptr,        # [M, K, C]
    topk_ptr,               # [M, K]
    grad_passthrough_ptr,   # [M, N, C], updated in-place
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    C,
    STORE_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """K-wide variant for K=4: one program updates all active rows."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    k_offs = tl.arange(0, K)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    tk = tl.load(topk_ptr + pid_m * K + k_offs)
    active = tl.load(
        grad_active_ptr + (pid_m * K + k_offs[:, None]) * C + c_offs[None, :],
        mask=c_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    out_ptrs = grad_passthrough_ptr + (pid_m * N + tk[:, None]) * C + c_offs[None, :]
    base = tl.load(out_ptrs, mask=c_mask[None, :], other=0.0).to(tl.float32)
    val = base + active
    if STORE_DTYPE == 1:
        tl.store(out_ptrs, val.to(tl.bfloat16), mask=c_mask[None, :])
    elif STORE_DTYPE == 2:
        tl.store(out_ptrs, val.to(tl.float16), mask=c_mask[None, :])
    else:
        tl.store(out_ptrs, val, mask=c_mask[None, :])


@triton.jit
def xhc_post_stats_router_top2_fwd_routed_kernel(
    combined_ptr,      # [M, ROUTER_COLS + N] fp32: router logits, then h_pre raw
    mean_x_ptr,        # [M] fp32
    mean_x2_ptr,       # [M] fp32
    colsum_ptr,        # [ROUTED_N] fp32
    alpha_pre_ptr,     # [1] fp32
    bias_pre_ptr,      # [N] fp32
    routing_bias_ptr,  # [N] fp32, optional by USE_BIAS
    h_pre_ptr,         # [M, N] fp32 output
    topk_ptr,          # [M, 4] int64 output: fixed 0/1 + routed top2
    router_probs_ptr,  # [M, 2] fp32 output: routed probs only
    M,
    N: tl.constexpr,
    NUM_FIXED: tl.constexpr,
    ROUTER_COLS: tl.constexpr,
    ROUTER_OFFSET: tl.constexpr,
    ROUTED_N: tl.constexpr,
    PADDED_ROUTED_N: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    """Fused xHC routed-layout post-stats correction + fixed-2/routed-top2 router."""
    pid = tl.program_id(0)
    offs_n = tl.arange(0, N)
    offs_r = tl.arange(0, PADDED_ROUTED_N)
    r_mask = offs_r < ROUTED_N
    base = combined_ptr + pid * (ROUTER_COLS + N)

    router_raw = tl.load(base + ROUTER_OFFSET + offs_r, mask=r_mask, other=0.0).to(tl.float32)
    h_pre_raw = tl.load(base + ROUTER_COLS + offs_n).to(tl.float32)
    mean_x = tl.load(mean_x_ptr + pid).to(tl.float32)
    mean_x2 = tl.load(mean_x2_ptr + pid).to(tl.float32)

    var_raw = mean_x2 - mean_x * mean_x
    var_x = tl.where(var_raw >= 0.0, var_raw, 0.0)
    rstd = tl.rsqrt(var_x + 1.0e-5)
    inv_rms = tl.where(mean_x2 > 0.0, tl.rsqrt(mean_x2), 1.0)

    alpha_pre = tl.load(alpha_pre_ptr).to(tl.float32)
    bias_pre = tl.load(bias_pre_ptr + offs_n).to(tl.float32)
    h_pre = tl.sigmoid(h_pre_raw * inv_rms * alpha_pre + bias_pre)
    tl.store(h_pre_ptr + pid * N + offs_n, h_pre)

    colsum = tl.load(colsum_ptr + offs_r, mask=r_mask, other=0.0).to(tl.float32)
    routed_logits = (router_raw - mean_x * colsum) * rstd
    score = routed_logits
    if USE_BIAS:
        bias = tl.load(routing_bias_ptr + NUM_FIXED + offs_r, mask=r_mask, other=0.0).to(tl.float32)
        score = routed_logits + bias
    score = tl.where(r_mask, score, -float("inf"))

    max0 = tl.max(score, axis=0)
    idx0 = tl.min(tl.where(score == max0, offs_r, PADDED_ROUTED_N), axis=0)
    score1 = tl.where(offs_r == idx0, -float("inf"), score)
    max1 = tl.max(score1, axis=0)
    idx1 = tl.min(tl.where(score1 == max1, offs_r, PADDED_ROUTED_N), axis=0)

    raw0 = tl.load(base + ROUTER_OFFSET + idx0).to(tl.float32)
    raw1 = tl.load(base + ROUTER_OFFSET + idx1).to(tl.float32)
    col0 = tl.load(colsum_ptr + idx0).to(tl.float32)
    col1 = tl.load(colsum_ptr + idx1).to(tl.float32)
    prob0 = tl.sigmoid((raw0 - mean_x * col0) * rstd)
    prob1 = tl.sigmoid((raw1 - mean_x * col1) * rstd)

    out_i = topk_ptr + pid * 4
    out_p = router_probs_ptr + pid * 2
    tl.store(out_i + 0, 0)
    tl.store(out_i + 1, 1)
    tl.store(out_i + 2, idx0 + NUM_FIXED)
    tl.store(out_i + 3, idx1 + NUM_FIXED)
    tl.store(out_p + 0, prob0)
    tl.store(out_p + 1, prob1)


@triton.jit
def xhc_post_stats_router_top2_bwd_stage1_routed_kernel(
    combined_ptr,      # [M, ROUTER_COLS + N] fp32
    mean_x_ptr,        # [M] fp32
    mean_x2_ptr,       # [M] fp32
    colsum_ptr,        # [ROUTED_N] fp32
    alpha_pre_ptr,     # [1] fp32
    h_pre_ptr,         # [M, N] fp32 saved output
    topk_ptr,          # [M, 4] int64 saved output
    router_probs_ptr,  # [M, 2] fp32 saved output
    grad_h_pre_ptr,    # [M, N] fp32
    grad_probs_ptr,    # [M, 2] fp32
    grad_combined_ptr, # [M, ROUTER_COLS + N] fp32
    grad_mean_x_ptr,   # [M] fp32
    grad_mean_x2_ptr,  # [M] fp32
    partial_ptr,       # [N + ROUTED_N + 1, M] fp32
    M,
    N: tl.constexpr,
    NUM_FIXED: tl.constexpr,
    ROUTER_COLS: tl.constexpr,
    ROUTER_OFFSET: tl.constexpr,
    ROUTED_N: tl.constexpr,
    PADDED_ROUTED_N: tl.constexpr,
):
    """Fused xHC routed-layout post-stats/router backward stage1."""
    pid = tl.program_id(0)
    offs_n = tl.arange(0, N)
    offs_r = tl.arange(0, PADDED_ROUTED_N)
    r_mask = offs_r < ROUTED_N
    base = combined_ptr + pid * (ROUTER_COLS + N)

    router_raw = tl.load(base + ROUTER_OFFSET + offs_r, mask=r_mask, other=0.0).to(tl.float32)
    h_pre_raw = tl.load(base + ROUTER_COLS + offs_n).to(tl.float32)
    mean_x = tl.load(mean_x_ptr + pid).to(tl.float32)
    mean_x2 = tl.load(mean_x2_ptr + pid).to(tl.float32)
    colsum = tl.load(colsum_ptr + offs_r, mask=r_mask, other=0.0).to(tl.float32)
    alpha_pre = tl.load(alpha_pre_ptr).to(tl.float32)

    var_raw = mean_x2 - mean_x * mean_x
    var_x = tl.where(var_raw >= 0.0, var_raw, 0.0)
    rstd = tl.rsqrt(var_x + 1.0e-5)
    inv_rms = tl.where(mean_x2 > 0.0, tl.rsqrt(mean_x2), 1.0)

    h_pre = tl.load(h_pre_ptr + pid * N + offs_n).to(tl.float32)
    grad_h = tl.load(grad_h_pre_ptr + pid * N + offs_n).to(tl.float32)
    grad_z = grad_h * h_pre * (1.0 - h_pre)
    grad_h_pre_raw = grad_z * inv_rms * alpha_pre
    grad_bias = grad_z
    grad_alpha = tl.sum(grad_z * h_pre_raw * inv_rms)
    grad_inv_rms = tl.sum(grad_z * h_pre_raw * alpha_pre)
    denom_m2 = tl.maximum(mean_x2, 1.0e-30)
    grad_mean_x2_h = tl.where(
        mean_x2 > 0.0,
        grad_inv_rms * (-0.5) * tl.rsqrt(denom_m2 * denom_m2 * denom_m2),
        0.0,
    )

    tk0 = tl.load(topk_ptr + pid * 4 + 2) - NUM_FIXED
    tk1 = tl.load(topk_ptr + pid * 4 + 3) - NUM_FIXED
    p0 = tl.load(router_probs_ptr + pid * 2 + 0).to(tl.float32)
    p1 = tl.load(router_probs_ptr + pid * 2 + 1).to(tl.float32)
    gp0 = tl.load(grad_probs_ptr + pid * 2 + 0).to(tl.float32)
    gp1 = tl.load(grad_probs_ptr + pid * 2 + 1).to(tl.float32)
    g0 = gp0 * p0 * (1.0 - p0)
    g1 = gp1 * p1 * (1.0 - p1)
    grad_logits = tl.zeros((PADDED_ROUTED_N,), dtype=tl.float32)
    grad_logits = tl.where(offs_r == tk0, g0, grad_logits)
    grad_logits = tl.where(offs_r == tk1, g1, grad_logits)
    grad_logits = tl.where(r_mask, grad_logits, 0.0)

    centered = router_raw - mean_x * colsum
    grad_router_raw = grad_logits * rstd
    grad_colsum = grad_logits * (-mean_x) * rstd
    grad_mean_direct = tl.sum(grad_logits * (-colsum) * rstd)
    grad_rstd = tl.sum(grad_logits * centered)

    grad_var = tl.where(
        var_raw >= 0.0,
        grad_rstd * (-0.5) * tl.rsqrt((var_x + 1.0e-5) * (var_x + 1.0e-5) * (var_x + 1.0e-5)),
        0.0,
    )
    grad_mean_x = grad_mean_direct + grad_var * (-2.0 * mean_x)
    grad_mean_x2 = grad_mean_x2_h + grad_var

    tl.store(
        grad_combined_ptr + pid * (ROUTER_COLS + N) + ROUTER_OFFSET + offs_r,
        grad_router_raw,
        mask=r_mask,
    )
    tl.store(grad_combined_ptr + pid * (ROUTER_COLS + N) + ROUTER_COLS + offs_n, grad_h_pre_raw)
    tl.store(grad_mean_x_ptr + pid, grad_mean_x)
    tl.store(grad_mean_x2_ptr + pid, grad_mean_x2)

    tl.store(partial_ptr + offs_n * M + pid, grad_bias)
    tl.store(partial_ptr + (N + offs_r) * M + pid, grad_colsum, mask=r_mask)
    tl.store(partial_ptr + (N + ROUTED_N) * M + pid, grad_alpha)


@triton.jit
def xhc_post_stats_router_reduce_routed_kernel(
    partial_ptr,       # [N + ROUTED_N + 1, M] fp32
    grad_bias_ptr,     # [N] fp32
    grad_colsum_ptr,   # [ROUTED_N] fp32
    grad_alpha_ptr,    # [1] fp32
    M,
    N: tl.constexpr,
    ROUTED_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    mask = offs_m < M
    vals = tl.load(partial_ptr + rid * M + offs_m, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(vals, axis=0)
    tl.store(grad_bias_ptr + rid, acc, mask=rid < N)
    tl.store(grad_colsum_ptr + (rid - N), acc, mask=(rid >= N) & (rid < N + ROUTED_N))
    tl.store(grad_alpha_ptr, acc, mask=rid == N + ROUTED_N)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 16, "BLOCK_D": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 32, "BLOCK_D": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 16, "BLOCK_D": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 32, "BLOCK_D": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 16, "BLOCK_D": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_T": 32, "BLOCK_D": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_T": 16, "BLOCK_D": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_T": 32, "BLOCK_D": 256}, num_warps=8, num_stages=3),
    ],
    key=["D", "OUT_DIM"],
)
@triton.jit
def xhc_active_proj_bf16_kernel(
    active_ptr,         # [M, D] bf16 — flat_active (D = K*C)
    W_ptr,              # [D, OUT_DIM] bf16 — proj_post_res
    out_ptr,            # [M, OUT_DIM] fp32 — projected * inv_rms
    mean_x2_ptr,        # [M] fp32 — optional mean(x^2) for backward
    M,
    D: tl.constexpr,
    OUT_DIM: tl.constexpr,
    stride_act,
    PADDED_OUT: tl.constexpr,
    STORE_MEAN_X2: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """RMS norm + matmul on active streams in 1 pass. Reads active bf16 once.
    Grid: (cdiv(M, BLOCK_T),)"""
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    offs_out = tl.arange(0, PADDED_OUT)
    out_mask = offs_out < OUT_DIM

    sum_x2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_T, PADDED_OUT), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        x = tl.load(
            active_ptr + offs_t[:, None] * stride_act + d_offs[None, :],
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        x_f32 = x.to(tl.float32)
        sum_x2 += tl.sum(x_f32 * x_f32, axis=1)

        w = tl.load(
            W_ptr + d_offs[:, None] * OUT_DIM + offs_out[None, :],
            mask=d_mask[:, None] & out_mask[None, :],
            other=0.0,
        )

        acc += tl.dot(x, w)

    inv_rms = tl.where(sum_x2 > 0.0, tl.rsqrt(sum_x2 * (1.0 / D)), 1.0)
    acc = acc * inv_rms[:, None]

    if STORE_MEAN_X2:
        tl.store(mean_x2_ptr + offs_t, sum_x2 * (1.0 / D), mask=t_mask)

    tl.store(
        out_ptr + offs_t[:, None] * OUT_DIM + offs_out[None, :],
        acc,
        mask=t_mask[:, None] & out_mask[None, :],
    )


@triton.jit
def xhc_active_proj_bf16_inline_kernel(
    active_ptr,         # [M, D] bf16 — flat_active (D = K*C)
    W_ptr,              # [D, OUT_DIM] bf16 — proj_post_res
    out_ptr,            # [M, OUT_DIM] fp32 — projected * inv_rms
    mean_x2_ptr,        # [M] fp32 — optional mean(x^2) for backward
    M,
    D: tl.constexpr,
    OUT_DIM: tl.constexpr,
    stride_act,
    PADDED_OUT: tl.constexpr,
    STORE_MEAN_X2: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Active projection using bf16 activations and bf16 weights, with fp32 accumulation."""
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    offs_out = tl.arange(0, PADDED_OUT)
    out_mask = offs_out < OUT_DIM

    sum_x2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_T, PADDED_OUT), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        x = tl.load(
            active_ptr + offs_t[:, None] * stride_act + d_offs[None, :],
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        x_f32 = x.to(tl.float32)
        sum_x2 += tl.sum(x_f32 * x_f32, axis=1)

        w = tl.load(
            W_ptr + d_offs[:, None] * OUT_DIM + offs_out[None, :],
            mask=d_mask[:, None] & out_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x, w)

    mean_x2 = sum_x2 * (1.0 / D)
    inv_rms = tl.where(sum_x2 > 0.0, tl.rsqrt(mean_x2), 1.0)
    acc = acc * inv_rms[:, None]

    if STORE_MEAN_X2:
        tl.store(mean_x2_ptr + offs_t, mean_x2, mask=t_mask)

    tl.store(
        out_ptr + offs_t[:, None] * OUT_DIM + offs_out[None, :],
        acc,
        mask=t_mask[:, None] & out_mask[None, :],
    )


@triton.jit
def xhc_active_proj_bf16_with_stats_inline_kernel(
    active_ptr,         # [M, D] bf16 — flat_active (D = K*C)
    W_ptr,              # [D, OUT_DIM] bf16 — proj_post_res
    mean_x2_ptr,        # [M] or [M, 1] fp32 — active RMS stats
    out_ptr,            # [M, OUT_DIM] fp32 — projected * inv_rms
    M,
    D: tl.constexpr,
    OUT_DIM: tl.constexpr,
    stride_act,
    PADDED_OUT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Active projection using mean_x2 produced by active gather."""
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    offs_out = tl.arange(0, PADDED_OUT)
    out_mask = offs_out < OUT_DIM

    acc = tl.zeros((BLOCK_T, PADDED_OUT), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        x = tl.load(
            active_ptr + offs_t[:, None] * stride_act + d_offs[None, :],
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            W_ptr + d_offs[:, None] * OUT_DIM + offs_out[None, :],
            mask=d_mask[:, None] & out_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x, w)

    mean_x2 = tl.load(mean_x2_ptr + offs_t, mask=t_mask, other=0.0)
    inv_rms = tl.where(mean_x2 > 0.0, tl.rsqrt(mean_x2), 1.0)
    acc = acc * inv_rms[:, None]

    tl.store(
        out_ptr + offs_t[:, None] * OUT_DIM + offs_out[None, :],
        acc,
        mask=t_mask[:, None] & out_mask[None, :],
    )


@triton.jit
def xhc_active_proj_bf16_split_matmul_kernel(
    active_ptr,         # [M, D] bf16
    W_ptr,              # [D, OUT_DIM] bf16
    partial_out_ptr,    # [NUM_D_SPLITS, M, OUT_DIM] fp32 raw partial matmul
    partial_sum2_ptr,   # [NUM_D_SPLITS, M] fp32 partial sum(x^2), optional
    M,
    D: tl.constexpr,
    OUT_DIM: tl.constexpr,
    stride_act,
    PADDED_OUT: tl.constexpr,
    CHUNK_D: tl.constexpr,
    STORE_SUM_X2: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """D-split active projection partial matmul.

    The RMS scale is applied in the reduce kernel after summing all D
    chunks, so this kernel stores unnormalized partial matmul results.
    """
    pid_m = tl.program_id(0)
    pid_split = tl.program_id(1)
    offs_t = pid_m * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    offs_out = tl.arange(0, PADDED_OUT)
    out_mask = offs_out < OUT_DIM

    split_start = pid_split * CHUNK_D
    sum_x2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_T, PADDED_OUT), dtype=tl.float32)

    for local_start in range(0, CHUNK_D, BLOCK_D):
        d_offs = split_start + local_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        x = tl.load(
            active_ptr + offs_t[:, None] * stride_act + d_offs[None, :],
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if STORE_SUM_X2:
            x_f32 = x.to(tl.float32)
            sum_x2 += tl.sum(x_f32 * x_f32, axis=1)

        w = tl.load(
            W_ptr + d_offs[:, None] * OUT_DIM + offs_out[None, :],
            mask=d_mask[:, None] & out_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x, w)

    partial_out_base = partial_out_ptr + pid_split * M * OUT_DIM
    tl.store(
        partial_out_base + offs_t[:, None] * OUT_DIM + offs_out[None, :],
        acc,
        mask=t_mask[:, None] & out_mask[None, :],
    )
    if STORE_SUM_X2:
        tl.store(partial_sum2_ptr + pid_split * M + offs_t, sum_x2, mask=t_mask)


@triton.jit
def xhc_active_proj_bf16_split_reduce_kernel(
    partial_out_ptr,    # [NUM_D_SPLITS, M, OUT_DIM] fp32 raw partial matmul
    partial_sum2_ptr,   # [NUM_D_SPLITS, M] fp32 partial sum(x^2), optional
    mean_x2_ptr,        # [M] fp32 output or reused input
    out_ptr,            # [M, OUT_DIM] fp32 projected * inv_rms
    M,
    D: tl.constexpr,
    OUT_DIM: tl.constexpr,
    PADDED_OUT: tl.constexpr,
    NUM_D_SPLITS: tl.constexpr,
    HAS_PARTIAL_SUM_X2: tl.constexpr,
    STORE_MEAN_X2: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_t = pid_m * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    offs_out = tl.arange(0, PADDED_OUT)
    out_mask = offs_out < OUT_DIM

    sum_x2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_T, PADDED_OUT), dtype=tl.float32)

    for split in tl.static_range(NUM_D_SPLITS):
        partial_out_base = partial_out_ptr + split * M * OUT_DIM
        acc += tl.load(
            partial_out_base + offs_t[:, None] * OUT_DIM + offs_out[None, :],
            mask=t_mask[:, None] & out_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if HAS_PARTIAL_SUM_X2:
            sum_x2 += tl.load(
                partial_sum2_ptr + split * M + offs_t,
                mask=t_mask,
                other=0.0,
            ).to(tl.float32)

    if HAS_PARTIAL_SUM_X2:
        mean_x2 = sum_x2 * (1.0 / D)
        if STORE_MEAN_X2:
            tl.store(mean_x2_ptr + offs_t, mean_x2, mask=t_mask)
    else:
        mean_x2 = tl.load(mean_x2_ptr + offs_t, mask=t_mask, other=0.0)

    inv_rms = tl.where(mean_x2 > 0.0, tl.rsqrt(mean_x2), 1.0)
    acc = acc * inv_rms[:, None]
    tl.store(
        out_ptr + offs_t[:, None] * OUT_DIM + offs_out[None, :],
        acc,
        mask=t_mask[:, None] & out_mask[None, :],
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64, "BLOCK_T": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 64, "BLOCK_T": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 64, "BLOCK_T": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 64, "BLOCK_T": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_K": 128, "BLOCK_T": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 128, "BLOCK_T": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 256, "BLOCK_T": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 256, "BLOCK_T": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 128, "BLOCK_T": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128, "BLOCK_T": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 256, "BLOCK_T": 16}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_K": 256, "BLOCK_T": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "NC", "OUT_DIM"],
)
@triton.jit
def xhc_proj_bwd_merged_kernel(
    grad_combined_ptr,  # [M, OUT_DIM] fp32
    grad_mean_x_ptr,    # [M] fp32
    grad_mean_x2_ptr,   # [M] fp32
    W_ptr,              # [NC, OUT_DIM] bf16/fp32
    hs_ptr,             # [M, NC] bf16
    grad_hs_ptr,        # [M, NC] output (origin dtype)
    grad_pw_ptr,        # [NC, OUT_DIM] output fp32
    M,
    NC: tl.constexpr,
    OUT_DIM: tl.constexpr,
    PADDED_OUT: tl.constexpr,
    inv_D_val,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    STORE_DTYPE: tl.constexpr,
):
    """Merged proj backward: grad_hs + grad_W in 1 kernel, reads hs once.
    Grid: (cdiv(NC, BLOCK_K),). Inner loop over M."""
    pid = tl.program_id(0)
    offs_k = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < NC
    offs_j = tl.arange(0, PADDED_OUT)
    j_mask = offs_j < OUT_DIM

    pw = tl.load(
        W_ptr + offs_k[:, None] * OUT_DIM + offs_j[None, :],
        mask=k_mask[:, None] & j_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    wgrad_acc = tl.zeros((BLOCK_K, PADDED_OUT), dtype=tl.float32)

    for t_start in range(0, M, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        t_mask = offs_t < M

        hs = tl.load(
            hs_ptr + offs_t[:, None] * NC + offs_k[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        ghc = tl.load(
            grad_combined_ptr + offs_t[:, None] * OUT_DIM + offs_j[None, :],
            mask=t_mask[:, None] & j_mask[None, :],
            other=0.0,
        )

        wgrad_acc += tl.dot(tl.trans(hs), ghc, input_precision="tf32x3")

        dhs_tile = tl.dot(ghc, tl.trans(pw), input_precision="tf32x3")

        gmx = tl.load(grad_mean_x_ptr + offs_t, mask=t_mask, other=0.0)
        gmx2 = tl.load(grad_mean_x2_ptr + offs_t, mask=t_mask, other=0.0)
        dhs_tile = dhs_tile + gmx[:, None] * inv_D_val
        dhs_tile = dhs_tile + (gmx2 * 2.0 * inv_D_val)[:, None] * hs

        out_ptrs = grad_hs_ptr + offs_t[:, None] * NC + offs_k[None, :]
        out_mask_2d = t_mask[:, None] & k_mask[None, :]
        if STORE_DTYPE == 1:
            tl.store(out_ptrs, dhs_tile.to(tl.bfloat16), mask=out_mask_2d)
        elif STORE_DTYPE == 2:
            tl.store(out_ptrs, dhs_tile.to(tl.float16), mask=out_mask_2d)
        else:
            tl.store(out_ptrs, dhs_tile, mask=out_mask_2d)

    tl.store(
        grad_pw_ptr + offs_k[:, None] * OUT_DIM + offs_j[None, :],
        wgrad_acc,
        mask=k_mask[:, None] & j_mask[None, :],
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 128, "BLOCK_T": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 128, "BLOCK_T": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 128, "BLOCK_T": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 256, "BLOCK_T": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 256, "BLOCK_T": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 256, "BLOCK_T": 64}, num_warps=8, num_stages=2),
    ],
    key=["NC", "OUT_DIM"],
)
@triton.jit
def xhc_proj_bwd_wgrad_kernel(
    grad_combined_ptr,  # [M, OUT_DIM] fp32
    hs_ptr,             # [M, NC] bf16
    grad_pw_ptr,        # [NC, OUT_DIM] output fp32
    M,
    NC: tl.constexpr,
    OUT_DIM: tl.constexpr,
    PADDED_OUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Projection dW only. Grid: (cdiv(NC, BLOCK_K),)."""
    pid = tl.program_id(0)
    offs_k = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < NC
    offs_j = tl.arange(0, PADDED_OUT)
    j_mask = offs_j < OUT_DIM
    wgrad_acc = tl.zeros((BLOCK_K, PADDED_OUT), dtype=tl.float32)

    for t_start in range(0, M, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        t_mask = offs_t < M
        hs = tl.load(
            hs_ptr + offs_t[:, None] * NC + offs_k[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        ghc = tl.load(
            grad_combined_ptr + offs_t[:, None] * OUT_DIM + offs_j[None, :],
            mask=t_mask[:, None] & j_mask[None, :],
            other=0.0,
        )
        wgrad_acc += tl.dot(tl.trans(hs), ghc, input_precision="tf32x3")

    tl.store(
        grad_pw_ptr + offs_k[:, None] * OUT_DIM + offs_j[None, :],
        wgrad_acc,
        mask=k_mask[:, None] & j_mask[None, :],
    )


@triton.jit
def xhc_active_proj_bwd_wgrad_rms_kernel(
    grad_projected_ptr,  # [M, OUT_DIM] fp32
    mean_x2_ptr,         # [M] fp32
    hs_ptr,              # [M, NC] bf16
    grad_pw_ptr,         # [NC, OUT_DIM] output fp32
    M,
    NC: tl.constexpr,
    OUT_DIM: tl.constexpr,
    PADDED_OUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Active projection dW with RMS grad scaling inline."""
    pid = tl.program_id(0)
    offs_k = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < NC
    offs_j = tl.arange(0, PADDED_OUT)
    j_mask = offs_j < OUT_DIM
    wgrad_acc = tl.zeros((BLOCK_K, PADDED_OUT), dtype=tl.float32)

    for t_start in range(0, M, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        t_mask = offs_t < M
        hs = tl.load(
            hs_ptr + offs_t[:, None] * NC + offs_k[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gp = tl.load(
            grad_projected_ptr + offs_t[:, None] * OUT_DIM + offs_j[None, :],
            mask=t_mask[:, None] & j_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        mean_x2 = tl.load(mean_x2_ptr + offs_t, mask=t_mask, other=0.0)
        inv_rms = tl.where(mean_x2 > 0.0, tl.rsqrt(mean_x2), 1.0)
        grad_raw = gp * inv_rms[:, None]
        wgrad_acc += tl.dot(tl.trans(hs), grad_raw, input_precision="tf32x3")

    tl.store(
        grad_pw_ptr + offs_k[:, None] * OUT_DIM + offs_j[None, :],
        wgrad_acc,
        mask=k_mask[:, None] & j_mask[None, :],
    )


@triton.jit
def xhc_active_proj_bwd_dhs_rms_kernel(
    grad_projected_ptr,  # [M, OUT_DIM] fp32
    projected_ptr,       # [M, OUT_DIM] fp32, active projection output
    mean_x2_ptr,         # [M] fp32
    W_ptr,               # [NC, OUT_DIM] bf16
    hs_ptr,              # [M, NC] bf16
    grad_hs_ptr,         # [M, NC] output
    M,
    NC: tl.constexpr,
    OUT_DIM: tl.constexpr,
    PADDED_OUT: tl.constexpr,
    inv_D_val,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STORE_DTYPE: tl.constexpr,
):
    """Active projection dX with RMS backward correction inline."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_j = tl.arange(0, PADDED_OUT)
    m_mask = offs_m < M
    k_mask = offs_k < NC
    j_mask = offs_j < OUT_DIM

    gp = tl.load(
        grad_projected_ptr + offs_m[:, None] * OUT_DIM + offs_j[None, :],
        mask=m_mask[:, None] & j_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    projected = tl.load(
        projected_ptr + offs_m[:, None] * OUT_DIM + offs_j[None, :],
        mask=m_mask[:, None] & j_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    mean_x2 = tl.load(mean_x2_ptr + offs_m, mask=m_mask, other=0.0)
    inv_rms = tl.where(mean_x2 > 0.0, tl.rsqrt(mean_x2), 1.0)
    grad_raw = gp * inv_rms[:, None]

    w = tl.load(
        W_ptr + offs_k[:, None] * OUT_DIM + offs_j[None, :],
        mask=k_mask[:, None] & j_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    dhs = tl.dot(grad_raw, tl.trans(w), input_precision="tf32x3")

    hs = tl.load(
        hs_ptr + offs_m[:, None] * NC + offs_k[None, :],
        mask=m_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    grad_inv_rms = tl.sum(gp * projected, axis=1)
    denom = tl.maximum(mean_x2, 1.0e-30)
    dhs = dhs - (grad_inv_rms * inv_D_val / denom)[:, None] * hs

    out_ptrs = grad_hs_ptr + offs_m[:, None] * NC + offs_k[None, :]
    out_mask = m_mask[:, None] & k_mask[None, :]
    if STORE_DTYPE == 1:
        tl.store(out_ptrs, dhs.to(tl.bfloat16), mask=out_mask)
    elif STORE_DTYPE == 2:
        tl.store(out_ptrs, dhs.to(tl.float16), mask=out_mask)
    else:
        tl.store(out_ptrs, dhs, mask=out_mask)


@triton.jit
def xhc_post_res_activation_fwd_kernel(
    projected_ptr,
    alpha_post_ptr,
    bias_post_ptr,
    alpha_res_ptr,
    bias_res_ptr,
    h_post_ptr,
    h_res_logits_ptr,
    M,
    OUT_DIM: tl.constexpr,
    DIM_POST: tl.constexpr,
    DIM_RES: tl.constexpr,
    POST_RANK: tl.constexpr,
    ALPHA_POST_NUMEL: tl.constexpr,
    PADDED_POST: tl.constexpr,
    PADDED_RES: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_p = tl.arange(0, PADDED_POST)
    post_mask = offs_p < DIM_POST
    raw_p = tl.load(projected_ptr + pid * OUT_DIM + offs_p, mask=post_mask, other=0.0).to(tl.float32)
    if ALPHA_POST_NUMEL == 1:
        alpha_p = tl.load(alpha_post_ptr).to(tl.float32)
    elif ALPHA_POST_NUMEL == POST_RANK:
        alpha_p = tl.load(alpha_post_ptr + (offs_p % POST_RANK), mask=post_mask, other=0.0).to(tl.float32)
    else:
        alpha_p = tl.load(alpha_post_ptr + offs_p, mask=post_mask, other=0.0).to(tl.float32)
    bias_p = tl.load(bias_post_ptr + offs_p, mask=post_mask, other=0.0).to(tl.float32)
    h = tl.sigmoid(raw_p * alpha_p + bias_p) * 2.0
    tl.store(h_post_ptr + pid * DIM_POST + offs_p, h, mask=post_mask)

    offs_r = tl.arange(0, PADDED_RES)
    res_mask = offs_r < DIM_RES
    raw_r = tl.load(projected_ptr + pid * OUT_DIM + DIM_POST + offs_r, mask=res_mask, other=0.0).to(tl.float32)
    alpha_r = tl.load(alpha_res_ptr).to(tl.float32)
    bias_r = tl.load(bias_res_ptr + offs_r, mask=res_mask, other=0.0).to(tl.float32)
    tl.store(h_res_logits_ptr + pid * DIM_RES + offs_r, raw_r * alpha_r + bias_r, mask=res_mask)


@triton.jit
def xhc_post_res_activation_bwd_stage1_kernel(
    projected_ptr,
    h_post_ptr,
    grad_h_post_ptr,
    grad_h_res_ptr,
    alpha_post_ptr,
    alpha_res_ptr,
    grad_projected_ptr,
    partial_ptr,
    M,
    OUT_DIM: tl.constexpr,
    DIM_POST: tl.constexpr,
    DIM_RES: tl.constexpr,
    POST_RANK: tl.constexpr,
    ALPHA_POST_NUMEL: tl.constexpr,
    PADDED_POST: tl.constexpr,
    PADDED_RES: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_p = tl.arange(0, PADDED_POST)
    post_mask = offs_p < DIM_POST
    raw_p = tl.load(projected_ptr + pid * OUT_DIM + offs_p, mask=post_mask, other=0.0).to(tl.float32)
    h = tl.load(h_post_ptr + pid * DIM_POST + offs_p, mask=post_mask, other=0.0).to(tl.float32)
    grad_h = tl.load(grad_h_post_ptr + pid * DIM_POST + offs_p, mask=post_mask, other=0.0).to(tl.float32)
    if ALPHA_POST_NUMEL == 1:
        alpha_p = tl.load(alpha_post_ptr).to(tl.float32)
    elif ALPHA_POST_NUMEL == POST_RANK:
        alpha_p = tl.load(alpha_post_ptr + (offs_p % POST_RANK), mask=post_mask, other=0.0).to(tl.float32)
    else:
        alpha_p = tl.load(alpha_post_ptr + offs_p, mask=post_mask, other=0.0).to(tl.float32)
    grad_z = grad_h * h * (1.0 - 0.5 * h)
    grad_alpha_vals = grad_z * raw_p
    tl.store(grad_projected_ptr + pid * OUT_DIM + offs_p, grad_z * alpha_p, mask=post_mask)
    if ALPHA_POST_NUMEL == 1:
        tl.store(partial_ptr + pid, tl.sum(tl.where(post_mask, grad_alpha_vals, 0.0), axis=0))
    elif ALPHA_POST_NUMEL == POST_RANK:
        ranks = offs_p % POST_RANK
        for r in range(0, POST_RANK):
            vals = tl.where(post_mask & (ranks == r), grad_alpha_vals, 0.0)
            tl.store(partial_ptr + r * M + pid, tl.sum(vals, axis=0))
    else:
        tl.store(partial_ptr + offs_p * M + pid, grad_alpha_vals, mask=post_mask)
    tl.store(partial_ptr + (ALPHA_POST_NUMEL + offs_p) * M + pid, grad_z, mask=post_mask)

    offs_r = tl.arange(0, PADDED_RES)
    res_mask = offs_r < DIM_RES
    raw_r = tl.load(projected_ptr + pid * OUT_DIM + DIM_POST + offs_r, mask=res_mask, other=0.0).to(tl.float32)
    grad_r = tl.load(grad_h_res_ptr + pid * DIM_RES + offs_r, mask=res_mask, other=0.0).to(tl.float32)
    alpha_r = tl.load(alpha_res_ptr).to(tl.float32)
    tl.store(grad_projected_ptr + pid * OUT_DIM + DIM_POST + offs_r, grad_r * alpha_r, mask=res_mask)
    tl.store(
        partial_ptr + (ALPHA_POST_NUMEL + DIM_POST) * M + pid,
        tl.sum(tl.where(res_mask, grad_r * raw_r, 0.0), axis=0),
    )
    tl.store(
        partial_ptr + (ALPHA_POST_NUMEL + DIM_POST + 1 + offs_r) * M + pid,
        grad_r,
        mask=res_mask,
    )


@triton.jit
def xhc_post_res_activation_reduce_kernel(
    partial_ptr,
    grad_alpha_post_ptr,
    grad_bias_post_ptr,
    grad_alpha_res_ptr,
    grad_bias_res_ptr,
    M,
    DIM_POST: tl.constexpr,
    DIM_RES: tl.constexpr,
    ALPHA_POST_NUMEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    mask = offs_m < M
    vals = tl.load(partial_ptr + rid * M + offs_m, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(vals, axis=0)
    tl.store(grad_alpha_post_ptr + rid, acc, mask=rid < ALPHA_POST_NUMEL)
    tl.store(
        grad_bias_post_ptr + (rid - ALPHA_POST_NUMEL),
        acc,
        mask=(rid >= ALPHA_POST_NUMEL) & (rid < ALPHA_POST_NUMEL + DIM_POST),
    )
    tl.store(
        grad_alpha_res_ptr,
        acc,
        mask=rid == ALPHA_POST_NUMEL + DIM_POST,
    )
    tl.store(
        grad_bias_res_ptr + (rid - ALPHA_POST_NUMEL - DIM_POST - 1),
        acc,
        mask=(rid >= ALPHA_POST_NUMEL + DIM_POST + 1)
        & (rid < ALPHA_POST_NUMEL + DIM_POST + 1 + DIM_RES),
    )


@triton.jit
def xhc_proj_bwd_dhs_kernel(
    grad_combined_ptr,  # [M, OUT_DIM] fp32
    grad_mean_x_ptr,    # [M] fp32
    grad_mean_x2_ptr,   # [M] fp32
    W_ptr,              # [NC, OUT_DIM] bf16/fp32/bf16
    hs_ptr,             # [M, NC] bf16
    grad_passthrough_ptr,  # [M, NC] bf16/half/fp32 or dummy
    grad_hs_ptr,        # [M, NC] output
    M,
    NC: tl.constexpr,
    OUT_DIM: tl.constexpr,
    PADDED_OUT: tl.constexpr,
    inv_D_val,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_GRAD_MEAN_X: tl.constexpr,
    HAS_GRAD_PASSTHROUGH: tl.constexpr,
    STORE_DTYPE: tl.constexpr,
):
    """Projection dX only with 2D parallelism over token and hidden tiles."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_j = tl.arange(0, PADDED_OUT)
    m_mask = offs_m < M
    k_mask = offs_k < NC
    j_mask = offs_j < OUT_DIM

    ghc = tl.load(
        grad_combined_ptr + offs_m[:, None] * OUT_DIM + offs_j[None, :],
        mask=m_mask[:, None] & j_mask[None, :],
        other=0.0,
    )
    w = tl.load(
        W_ptr + offs_k[:, None] * OUT_DIM + offs_j[None, :],
        mask=k_mask[:, None] & j_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    dhs = tl.dot(ghc, tl.trans(w), input_precision="tf32x3")

    hs = tl.load(
        hs_ptr + offs_m[:, None] * NC + offs_k[None, :],
        mask=m_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    gmx2 = tl.load(grad_mean_x2_ptr + offs_m, mask=m_mask, other=0.0)
    if HAS_GRAD_MEAN_X:
        gmx = tl.load(grad_mean_x_ptr + offs_m, mask=m_mask, other=0.0)
        dhs = dhs + gmx[:, None] * inv_D_val
    dhs = dhs + (gmx2 * 2.0 * inv_D_val)[:, None] * hs
    if HAS_GRAD_PASSTHROUGH:
        if STORE_DTYPE == 1:
            dhs = dhs.to(tl.bfloat16).to(tl.float32)
        elif STORE_DTYPE == 2:
            dhs = dhs.to(tl.float16).to(tl.float32)
        passthrough = tl.load(
            grad_passthrough_ptr + offs_m[:, None] * NC + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        dhs = dhs + passthrough

    out_ptrs = grad_hs_ptr + offs_m[:, None] * NC + offs_k[None, :]
    out_mask = m_mask[:, None] & k_mask[None, :]
    if STORE_DTYPE == 1:
        tl.store(out_ptrs, dhs.to(tl.bfloat16), mask=out_mask)
    elif STORE_DTYPE == 2:
        tl.store(out_ptrs, dhs.to(tl.float16), mask=out_mask)
    else:
        tl.store(out_ptrs, dhs, mask=out_mask)


@triton.jit
def xhc_proj_bwd_dhs_bf16_fused_kernel(
    grad_combined_ptr,  # [M, OUT_DIM] fp32, rounded to bf16 for the matmul
    grad_mean_x_ptr,    # [M] fp32
    grad_mean_x2_ptr,   # [M] fp32
    W_ptr,              # [NC, OUT_DIM] bf16
    hs_ptr,             # [M, NC] bf16
    grad_passthrough_ptr,  # [M, NC] bf16/half/fp32 or dummy
    grad_hs_ptr,        # [M, NC] output
    M,
    NC: tl.constexpr,
    OUT_DIM: tl.constexpr,
    PADDED_OUT: tl.constexpr,
    inv_D_val,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_GRAD_MEAN_X: tl.constexpr,
    HAS_GRAD_PASSTHROUGH: tl.constexpr,
    STORE_DTYPE: tl.constexpr,
):
    """Projection dX + stats/pass-through correction with bf16 tensor-core dot."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_j = tl.arange(0, PADDED_OUT)
    m_mask = offs_m < M
    k_mask = offs_k < NC
    j_mask = offs_j < OUT_DIM

    ghc = tl.load(
        grad_combined_ptr + offs_m[:, None] * OUT_DIM + offs_j[None, :],
        mask=m_mask[:, None] & j_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    w = tl.load(
        W_ptr + offs_k[:, None] * OUT_DIM + offs_j[None, :],
        mask=k_mask[:, None] & j_mask[None, :],
        other=0.0,
    )
    dhs = tl.dot(ghc, tl.trans(w))
    if STORE_DTYPE == 1:
        dhs = dhs.to(tl.bfloat16).to(tl.float32)
    elif STORE_DTYPE == 2:
        dhs = dhs.to(tl.float16).to(tl.float32)

    hs = tl.load(
        hs_ptr + offs_m[:, None] * NC + offs_k[None, :],
        mask=m_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    gmx2 = tl.load(grad_mean_x2_ptr + offs_m, mask=m_mask, other=0.0)
    dhs = dhs + (gmx2 * 2.0 * inv_D_val)[:, None] * hs
    if HAS_GRAD_MEAN_X:
        gmx = tl.load(grad_mean_x_ptr + offs_m, mask=m_mask, other=0.0)
        dhs = dhs + gmx[:, None] * inv_D_val

    if HAS_GRAD_PASSTHROUGH:
        if STORE_DTYPE == 1:
            dhs = dhs.to(tl.bfloat16).to(tl.float32)
        elif STORE_DTYPE == 2:
            dhs = dhs.to(tl.float16).to(tl.float32)
        passthrough = tl.load(
            grad_passthrough_ptr + offs_m[:, None] * NC + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        dhs = dhs + passthrough

    out_ptrs = grad_hs_ptr + offs_m[:, None] * NC + offs_k[None, :]
    out_mask = m_mask[:, None] & k_mask[None, :]
    if STORE_DTYPE == 1:
        tl.store(out_ptrs, dhs.to(tl.bfloat16), mask=out_mask)
    elif STORE_DTYPE == 2:
        tl.store(out_ptrs, dhs.to(tl.float16), mask=out_mask)
    else:
        tl.store(out_ptrs, dhs, mask=out_mask)


@triton.jit
def xhc_active_proj_bwd_dhs_bf16_fused_kernel(
    grad_combined_ptr,  # [M, OUT_DIM] fp32, rounded to bf16 for the matmul
    grad_mean_x2_ptr,   # [M] fp32
    W_ptr,              # [NC, OUT_DIM] bf16
    hs_ptr,             # [M, NC] bf16
    grad_hs_ptr,        # [M, NC] output
    M,
    NC: tl.constexpr,
    OUT_DIM: tl.constexpr,
    PADDED_OUT: tl.constexpr,
    inv_D_val,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STORE_DTYPE: tl.constexpr,
):
    """Active projection dX + RMS correction with bf16 tensor-core dot."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_j = tl.arange(0, PADDED_OUT)
    m_mask = offs_m < M
    k_mask = offs_k < NC
    j_mask = offs_j < OUT_DIM

    ghc = tl.load(
        grad_combined_ptr + offs_m[:, None] * OUT_DIM + offs_j[None, :],
        mask=m_mask[:, None] & j_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    w = tl.load(
        W_ptr + offs_k[:, None] * OUT_DIM + offs_j[None, :],
        mask=k_mask[:, None] & j_mask[None, :],
        other=0.0,
    )
    dhs = tl.dot(ghc, tl.trans(w))
    if STORE_DTYPE == 1:
        dhs = dhs.to(tl.bfloat16).to(tl.float32)
    elif STORE_DTYPE == 2:
        dhs = dhs.to(tl.float16).to(tl.float32)

    hs = tl.load(
        hs_ptr + offs_m[:, None] * NC + offs_k[None, :],
        mask=m_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    gmx2 = tl.load(grad_mean_x2_ptr + offs_m, mask=m_mask, other=0.0)
    dhs = dhs + (gmx2 * 2.0 * inv_D_val)[:, None] * hs

    out_ptrs = grad_hs_ptr + offs_m[:, None] * NC + offs_k[None, :]
    out_mask = m_mask[:, None] & k_mask[None, :]
    if STORE_DTYPE == 1:
        tl.store(out_ptrs, dhs.to(tl.bfloat16), mask=out_mask)
    elif STORE_DTYPE == 2:
        tl.store(out_ptrs, dhs.to(tl.float16), mask=out_mask)
    else:
        tl.store(out_ptrs, dhs, mask=out_mask)


@triton.jit
def xhc_proj_bwd_dhs_correction_kernel(
    grad_hs_ptr,        # [M, NC], in-place bf16/half/fp32 base dX
    grad_mean_x_ptr,    # [M] fp32 or dummy
    grad_mean_x2_ptr,   # [M] fp32
    hs_ptr,             # [M, NC] bf16
    grad_passthrough_ptr,  # [M, NC] bf16/half/fp32 or dummy
    M,
    NC: tl.constexpr,
    inv_D_val,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_GRAD_MEAN_X: tl.constexpr,
    HAS_GRAD_PASSTHROUGH: tl.constexpr,
    STORE_DTYPE: tl.constexpr,
):
    """Apply projection dX reduction/pass-through corrections to a GEMM base."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    k_mask = offs_k < NC
    ptrs = grad_hs_ptr + offs_m[:, None] * NC + offs_k[None, :]
    mask = m_mask[:, None] & k_mask[None, :]

    dhs = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    hs = tl.load(
        hs_ptr + offs_m[:, None] * NC + offs_k[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gmx2 = tl.load(grad_mean_x2_ptr + offs_m, mask=m_mask, other=0.0)
    if HAS_GRAD_MEAN_X:
        gmx = tl.load(grad_mean_x_ptr + offs_m, mask=m_mask, other=0.0)
        dhs = dhs + gmx[:, None] * inv_D_val
    dhs = dhs + (gmx2 * 2.0 * inv_D_val)[:, None] * hs
    if HAS_GRAD_PASSTHROUGH:
        if STORE_DTYPE == 1:
            dhs = dhs.to(tl.bfloat16).to(tl.float32)
        elif STORE_DTYPE == 2:
            dhs = dhs.to(tl.float16).to(tl.float32)
        passthrough = tl.load(
            grad_passthrough_ptr + offs_m[:, None] * NC + offs_k[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dhs = dhs + passthrough

    if STORE_DTYPE == 1:
        tl.store(ptrs, dhs.to(tl.bfloat16), mask=mask)
    elif STORE_DTYPE == 2:
        tl.store(ptrs, dhs.to(tl.float16), mask=mask)
    else:
        tl.store(ptrs, dhs, mask=mask)


@triton.jit
def xhc_proj_bwd_dhs_correction_linear_kernel(
    grad_hs_ptr,        # [M, NC], in-place bf16/half/fp32 base dX
    grad_mean_x_ptr,    # [M] fp32 or dummy
    grad_mean_x2_ptr,   # [M] fp32
    hs_ptr,             # [M, NC] bf16
    grad_passthrough_ptr,  # [M, NC] bf16/half/fp32 or dummy
    TOTAL: tl.constexpr,
    NC: tl.constexpr,
    inv_D_val,
    HAS_GRAD_MEAN_X: tl.constexpr,
    HAS_GRAD_PASSTHROUGH: tl.constexpr,
    STORE_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Linearized projection correction; better memory scheduling for full D."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    offs_m = offs // NC

    dhs = tl.load(grad_hs_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    hs = tl.load(hs_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gmx2 = tl.load(grad_mean_x2_ptr + offs_m, mask=mask, other=0.0)
    dhs = dhs + gmx2 * (2.0 * inv_D_val) * hs
    if HAS_GRAD_MEAN_X:
        gmx = tl.load(grad_mean_x_ptr + offs_m, mask=mask, other=0.0)
        dhs = dhs + gmx * inv_D_val
    if HAS_GRAD_PASSTHROUGH:
        if STORE_DTYPE == 1:
            dhs = dhs.to(tl.bfloat16).to(tl.float32)
        elif STORE_DTYPE == 2:
            dhs = dhs.to(tl.float16).to(tl.float32)
        passthrough = tl.load(grad_passthrough_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dhs = dhs + passthrough

    if STORE_DTYPE == 1:
        tl.store(grad_hs_ptr + offs, dhs.to(tl.bfloat16), mask=mask)
    elif STORE_DTYPE == 2:
        tl.store(grad_hs_ptr + offs, dhs.to(tl.float16), mask=mask)
    else:
        tl.store(grad_hs_ptr + offs, dhs, mask=mask)


@triton.jit
def xhc_temporal_conv_k4812_fwd_kernel(
    x_ptr,      # [S, B, C]
    w0_ptr,     # [C, 1, 4]
    b0_ptr,     # [C] or dummy
    w1_ptr,     # [C, 1, 8]
    b1_ptr,
    w2_ptr,     # [C, 1, 12]
    b2_ptr,
    out_ptr,    # [S, B, 3, C]
    S: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = offs_t < S
    c_mask = offs_c < C

    acc0 = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

    for r in tl.static_range(4):
        t_in = offs_t + r - 3
        x = tl.load(
            x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w0_ptr + offs_c * 4 + r, mask=c_mask, other=0.0).to(tl.float32)
        acc0 += x * w[None, :]

    for r in tl.static_range(8):
        t_in = offs_t + r - 7
        x = tl.load(
            x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w1_ptr + offs_c * 8 + r, mask=c_mask, other=0.0).to(tl.float32)
        acc1 += x * w[None, :]

    for r in tl.static_range(12):
        t_in = offs_t + r - 11
        x = tl.load(
            x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w2_ptr + offs_c * 12 + r, mask=c_mask, other=0.0).to(tl.float32)
        acc2 += x * w[None, :]

    if HAS_BIAS:
        acc0 += tl.load(b0_ptr + offs_c, mask=c_mask, other=0.0)[None, :]
        acc1 += tl.load(b1_ptr + offs_c, mask=c_mask, other=0.0)[None, :]
        acc2 += tl.load(b2_ptr + offs_c, mask=c_mask, other=0.0)[None, :]

    out_base = ((offs_t[:, None] * B + pid_b) * 3) * C + offs_c[None, :]
    mask = t_mask[:, None] & c_mask[None, :]
    tl.store(out_ptr + out_base + 0 * C, acc0.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + out_base + 1 * C, acc1.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + out_base + 2 * C, acc2.to(tl.bfloat16), mask=mask)


@triton.jit
def xhc_temporal_conv_k4812_dx_kernel(
    go_ptr,     # [S, B, 3, C]
    w0_ptr,     # [C, 1, 4]
    w1_ptr,     # [C, 1, 8]
    w2_ptr,     # [C, 1, 12]
    gx_ptr,     # [S, B, C]
    S: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    stride_go_s: tl.constexpr,
    stride_go_b: tl.constexpr,
    stride_go_k: tl.constexpr,
    stride_go_c: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = offs_t < S
    c_mask = offs_c < C
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

    for r in tl.static_range(4):
        t_out = offs_t - r + 3
        g = tl.load(
            go_ptr + t_out[:, None] * stride_go_s + pid_b * stride_go_b + 0 * stride_go_k + offs_c[None, :] * stride_go_c,
            mask=(t_out[:, None] >= 0) & (t_out[:, None] < S) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w0_ptr + offs_c * 4 + r, mask=c_mask, other=0.0).to(tl.float32)
        acc += g * w[None, :]

    for r in tl.static_range(8):
        t_out = offs_t - r + 7
        g = tl.load(
            go_ptr + t_out[:, None] * stride_go_s + pid_b * stride_go_b + 1 * stride_go_k + offs_c[None, :] * stride_go_c,
            mask=(t_out[:, None] >= 0) & (t_out[:, None] < S) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w1_ptr + offs_c * 8 + r, mask=c_mask, other=0.0).to(tl.float32)
        acc += g * w[None, :]

    for r in tl.static_range(12):
        t_out = offs_t - r + 11
        g = tl.load(
            go_ptr + t_out[:, None] * stride_go_s + pid_b * stride_go_b + 2 * stride_go_k + offs_c[None, :] * stride_go_c,
            mask=(t_out[:, None] >= 0) & (t_out[:, None] < S) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w2_ptr + offs_c * 12 + r, mask=c_mask, other=0.0).to(tl.float32)
        acc += g * w[None, :]

    mask = t_mask[:, None] & c_mask[None, :]
    tl.store(gx_ptr + (offs_t[:, None] * B + pid_b) * C + offs_c[None, :], acc.to(tl.bfloat16), mask=mask)


@triton.jit
def xhc_temporal_conv_wgrad_kernel(
    x_ptr,      # [S, B, C]
    go_ptr,     # [S, B, 3, C]
    gw_ptr,     # [C, 1, K]
    gb_ptr,     # [C] or dummy
    S: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    stride_go_s: tl.constexpr,
    stride_go_b: tl.constexpr,
    stride_go_k: tl.constexpr,
    stride_go_c: tl.constexpr,
    BRANCH: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C

    bias_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc8 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc9 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for t_start in range(0, S, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        t_mask = offs_t < S
        for b in range(0, B):
            g = tl.load(
                go_ptr
                + offs_t[:, None] * stride_go_s
                + b * stride_go_b
                + BRANCH * stride_go_k
                + offs_c[None, :] * stride_go_c,
                mask=t_mask[:, None] & c_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            bias_acc += tl.sum(g, axis=0)

            t_in0 = offs_t - (K - 1)
            x0 = tl.load(
                x_ptr + (t_in0[:, None] * B + b) * C + offs_c[None, :],
                mask=(t_in0[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc0 += tl.sum(x0 * g, axis=0)

            t_in1 = offs_t + 1 - (K - 1)
            x1 = tl.load(
                x_ptr + (t_in1[:, None] * B + b) * C + offs_c[None, :],
                mask=(t_in1[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc1 += tl.sum(x1 * g, axis=0)

            t_in2 = offs_t + 2 - (K - 1)
            x2 = tl.load(
                x_ptr + (t_in2[:, None] * B + b) * C + offs_c[None, :],
                mask=(t_in2[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc2 += tl.sum(x2 * g, axis=0)

            t_in3 = offs_t + 3 - (K - 1)
            x3 = tl.load(
                x_ptr + (t_in3[:, None] * B + b) * C + offs_c[None, :],
                mask=(t_in3[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc3 += tl.sum(x3 * g, axis=0)

            if K >= 8:
                t_in4 = offs_t + 4 - (K - 1)
                x4 = tl.load(
                    x_ptr + (t_in4[:, None] * B + b) * C + offs_c[None, :],
                    mask=(t_in4[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc4 += tl.sum(x4 * g, axis=0)

                t_in5 = offs_t + 5 - (K - 1)
                x5 = tl.load(
                    x_ptr + (t_in5[:, None] * B + b) * C + offs_c[None, :],
                    mask=(t_in5[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc5 += tl.sum(x5 * g, axis=0)

                t_in6 = offs_t + 6 - (K - 1)
                x6 = tl.load(
                    x_ptr + (t_in6[:, None] * B + b) * C + offs_c[None, :],
                    mask=(t_in6[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc6 += tl.sum(x6 * g, axis=0)

                t_in7 = offs_t + 7 - (K - 1)
                x7 = tl.load(
                    x_ptr + (t_in7[:, None] * B + b) * C + offs_c[None, :],
                    mask=(t_in7[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc7 += tl.sum(x7 * g, axis=0)

            if K >= 12:
                t_in8 = offs_t + 8 - (K - 1)
                x8 = tl.load(
                    x_ptr + (t_in8[:, None] * B + b) * C + offs_c[None, :],
                    mask=(t_in8[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc8 += tl.sum(x8 * g, axis=0)

                t_in9 = offs_t + 9 - (K - 1)
                x9 = tl.load(
                    x_ptr + (t_in9[:, None] * B + b) * C + offs_c[None, :],
                    mask=(t_in9[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc9 += tl.sum(x9 * g, axis=0)

                t_in10 = offs_t + 10 - (K - 1)
                x10 = tl.load(
                    x_ptr + (t_in10[:, None] * B + b) * C + offs_c[None, :],
                    mask=(t_in10[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc10 += tl.sum(x10 * g, axis=0)

                t_in11 = offs_t + 11 - (K - 1)
                x11 = tl.load(
                    x_ptr + (t_in11[:, None] * B + b) * C + offs_c[None, :],
                    mask=(t_in11[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc11 += tl.sum(x11 * g, axis=0)

    tl.store(gw_ptr + offs_c * K + 0, acc0.to(tl.bfloat16), mask=c_mask)
    tl.store(gw_ptr + offs_c * K + 1, acc1.to(tl.bfloat16), mask=c_mask)
    tl.store(gw_ptr + offs_c * K + 2, acc2.to(tl.bfloat16), mask=c_mask)
    tl.store(gw_ptr + offs_c * K + 3, acc3.to(tl.bfloat16), mask=c_mask)
    if K >= 8:
        tl.store(gw_ptr + offs_c * K + 4, acc4.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 5, acc5.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 6, acc6.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 7, acc7.to(tl.bfloat16), mask=c_mask)
    if K >= 12:
        tl.store(gw_ptr + offs_c * K + 8, acc8.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 9, acc9.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 10, acc10.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 11, acc11.to(tl.bfloat16), mask=c_mask)
    if HAS_BIAS:
        tl.store(gb_ptr + offs_c, bias_acc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def xhc_temporal_conv_wgrad_partial_kernel(
    x_ptr,      # [S, B, C]
    go_ptr,     # [S, B, 3, C]
    partial_w_ptr,  # [NUM_TILES * B, C, K] fp32
    partial_b_ptr,  # [NUM_TILES * B, C] fp32 or dummy
    S: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    stride_go_s: tl.constexpr,
    stride_go_b: tl.constexpr,
    stride_go_k: tl.constexpr,
    stride_go_c: tl.constexpr,
    BRANCH: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c_mask = offs_c < C
    t_mask = offs_t < S

    bias_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc8 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc9 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    g = tl.load(
        go_ptr
        + offs_t[:, None] * stride_go_s
        + pid_b * stride_go_b
        + BRANCH * stride_go_k
        + offs_c[None, :] * stride_go_c,
        mask=t_mask[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    bias_acc += tl.sum(g, axis=0)

    t_in0 = offs_t - (K - 1)
    x0 = tl.load(
        x_ptr + (t_in0[:, None] * B + pid_b) * C + offs_c[None, :],
        mask=(t_in0[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc0 += tl.sum(x0 * g, axis=0)

    t_in1 = offs_t + 1 - (K - 1)
    x1 = tl.load(
        x_ptr + (t_in1[:, None] * B + pid_b) * C + offs_c[None, :],
        mask=(t_in1[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc1 += tl.sum(x1 * g, axis=0)

    t_in2 = offs_t + 2 - (K - 1)
    x2 = tl.load(
        x_ptr + (t_in2[:, None] * B + pid_b) * C + offs_c[None, :],
        mask=(t_in2[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc2 += tl.sum(x2 * g, axis=0)

    t_in3 = offs_t + 3 - (K - 1)
    x3 = tl.load(
        x_ptr + (t_in3[:, None] * B + pid_b) * C + offs_c[None, :],
        mask=(t_in3[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc3 += tl.sum(x3 * g, axis=0)

    if K >= 8:
        t_in4 = offs_t + 4 - (K - 1)
        x4 = tl.load(
            x_ptr + (t_in4[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in4[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc4 += tl.sum(x4 * g, axis=0)

        t_in5 = offs_t + 5 - (K - 1)
        x5 = tl.load(
            x_ptr + (t_in5[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in5[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc5 += tl.sum(x5 * g, axis=0)

        t_in6 = offs_t + 6 - (K - 1)
        x6 = tl.load(
            x_ptr + (t_in6[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in6[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc6 += tl.sum(x6 * g, axis=0)

        t_in7 = offs_t + 7 - (K - 1)
        x7 = tl.load(
            x_ptr + (t_in7[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in7[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc7 += tl.sum(x7 * g, axis=0)

    if K >= 12:
        t_in8 = offs_t + 8 - (K - 1)
        x8 = tl.load(
            x_ptr + (t_in8[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in8[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc8 += tl.sum(x8 * g, axis=0)

        t_in9 = offs_t + 9 - (K - 1)
        x9 = tl.load(
            x_ptr + (t_in9[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in9[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc9 += tl.sum(x9 * g, axis=0)

        t_in10 = offs_t + 10 - (K - 1)
        x10 = tl.load(
            x_ptr + (t_in10[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in10[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc10 += tl.sum(x10 * g, axis=0)

        t_in11 = offs_t + 11 - (K - 1)
        x11 = tl.load(
            x_ptr + (t_in11[:, None] * B + pid_b) * C + offs_c[None, :],
            mask=(t_in11[:, None] >= 0) & t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc11 += tl.sum(x11 * g, axis=0)

    partial_idx = pid_t * B + pid_b
    base = (partial_idx * C + offs_c) * K
    tl.store(partial_w_ptr + base + 0, acc0, mask=c_mask)
    tl.store(partial_w_ptr + base + 1, acc1, mask=c_mask)
    tl.store(partial_w_ptr + base + 2, acc2, mask=c_mask)
    tl.store(partial_w_ptr + base + 3, acc3, mask=c_mask)
    if K >= 8:
        tl.store(partial_w_ptr + base + 4, acc4, mask=c_mask)
        tl.store(partial_w_ptr + base + 5, acc5, mask=c_mask)
        tl.store(partial_w_ptr + base + 6, acc6, mask=c_mask)
        tl.store(partial_w_ptr + base + 7, acc7, mask=c_mask)
    if K >= 12:
        tl.store(partial_w_ptr + base + 8, acc8, mask=c_mask)
        tl.store(partial_w_ptr + base + 9, acc9, mask=c_mask)
        tl.store(partial_w_ptr + base + 10, acc10, mask=c_mask)
        tl.store(partial_w_ptr + base + 11, acc11, mask=c_mask)
    if HAS_BIAS:
        tl.store(partial_b_ptr + partial_idx * C + offs_c, bias_acc, mask=c_mask)


@triton.jit
def xhc_temporal_conv_wgrad_reduce_kernel(
    partial_w_ptr,  # [NUM_PARTIALS, C, K] fp32
    partial_b_ptr,  # [NUM_PARTIALS, C] fp32 or dummy
    gw_ptr,         # [C, 1, K]
    gb_ptr,         # [C] or dummy
    C: tl.constexpr,
    K: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C

    bacc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc8 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc9 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for p in tl.static_range(NUM_PARTIALS):
        base = (p * C + offs_c) * K
        acc0 += tl.load(partial_w_ptr + base + 0, mask=c_mask, other=0.0)
        acc1 += tl.load(partial_w_ptr + base + 1, mask=c_mask, other=0.0)
        acc2 += tl.load(partial_w_ptr + base + 2, mask=c_mask, other=0.0)
        acc3 += tl.load(partial_w_ptr + base + 3, mask=c_mask, other=0.0)
        if K >= 8:
            acc4 += tl.load(partial_w_ptr + base + 4, mask=c_mask, other=0.0)
            acc5 += tl.load(partial_w_ptr + base + 5, mask=c_mask, other=0.0)
            acc6 += tl.load(partial_w_ptr + base + 6, mask=c_mask, other=0.0)
            acc7 += tl.load(partial_w_ptr + base + 7, mask=c_mask, other=0.0)
        if K >= 12:
            acc8 += tl.load(partial_w_ptr + base + 8, mask=c_mask, other=0.0)
            acc9 += tl.load(partial_w_ptr + base + 9, mask=c_mask, other=0.0)
            acc10 += tl.load(partial_w_ptr + base + 10, mask=c_mask, other=0.0)
            acc11 += tl.load(partial_w_ptr + base + 11, mask=c_mask, other=0.0)
        if HAS_BIAS:
            bacc += tl.load(partial_b_ptr + p * C + offs_c, mask=c_mask, other=0.0)

    tl.store(gw_ptr + offs_c * K + 0, acc0.to(tl.bfloat16), mask=c_mask)
    tl.store(gw_ptr + offs_c * K + 1, acc1.to(tl.bfloat16), mask=c_mask)
    tl.store(gw_ptr + offs_c * K + 2, acc2.to(tl.bfloat16), mask=c_mask)
    tl.store(gw_ptr + offs_c * K + 3, acc3.to(tl.bfloat16), mask=c_mask)
    if K >= 8:
        tl.store(gw_ptr + offs_c * K + 4, acc4.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 5, acc5.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 6, acc6.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 7, acc7.to(tl.bfloat16), mask=c_mask)
    if K >= 12:
        tl.store(gw_ptr + offs_c * K + 8, acc8.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 9, acc9.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 10, acc10.to(tl.bfloat16), mask=c_mask)
        tl.store(gw_ptr + offs_c * K + 11, acc11.to(tl.bfloat16), mask=c_mask)
    if HAS_BIAS:
        tl.store(gb_ptr + offs_c, bacc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def xhc_temporal_conv_wgrad_k4812_partial_fused_kernel(
    x_ptr, go_ptr,
    pw0_ptr, pw1_ptr, pw2_ptr,
    pb0_ptr, pb1_ptr, pb2_ptr,
    S: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    stride_go_s: tl.constexpr,
    stride_go_b: tl.constexpr,
    stride_go_k: tl.constexpr,
    stride_go_c: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c_mask = offs_c < C
    t_mask = offs_t < S

    b0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    b1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    b2 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    a00 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a01 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a02 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a03 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    a10 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a11 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a12 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a13 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a14 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a15 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a16 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a17 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    a20 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a21 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a22 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a23 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a24 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a25 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a26 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a27 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a28 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a29 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a210 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a211 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    g0 = tl.load(
        go_ptr + offs_t[:, None] * stride_go_s + pid_b * stride_go_b + 0 * stride_go_k + offs_c[None, :] * stride_go_c,
        mask=t_mask[:, None] & c_mask[None, :], other=0.0,
    ).to(tl.float32)
    g1 = tl.load(
        go_ptr + offs_t[:, None] * stride_go_s + pid_b * stride_go_b + 1 * stride_go_k + offs_c[None, :] * stride_go_c,
        mask=t_mask[:, None] & c_mask[None, :], other=0.0,
    ).to(tl.float32)
    g2 = tl.load(
        go_ptr + offs_t[:, None] * stride_go_s + pid_b * stride_go_b + 2 * stride_go_k + offs_c[None, :] * stride_go_c,
        mask=t_mask[:, None] & c_mask[None, :], other=0.0,
    ).to(tl.float32)
    b0 += tl.sum(g0, axis=0)
    b1 += tl.sum(g1, axis=0)
    b2 += tl.sum(g2, axis=0)

    t_in = offs_t - 11
    x = tl.load(x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :], mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    a20 += tl.sum(x * g2, axis=0)
    t_in = offs_t - 10
    x = tl.load(x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :], mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    a21 += tl.sum(x * g2, axis=0)
    t_in = offs_t - 9
    x = tl.load(x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :], mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    a22 += tl.sum(x * g2, axis=0)
    t_in = offs_t - 8
    x = tl.load(x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :], mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    a23 += tl.sum(x * g2, axis=0)

    t_in = offs_t - 7
    x = tl.load(x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :], mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    a10 += tl.sum(x * g1, axis=0); a24 += tl.sum(x * g2, axis=0)
    t_in = offs_t - 6
    x = tl.load(x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :], mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    a11 += tl.sum(x * g1, axis=0); a25 += tl.sum(x * g2, axis=0)
    t_in = offs_t - 5
    x = tl.load(x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :], mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    a12 += tl.sum(x * g1, axis=0); a26 += tl.sum(x * g2, axis=0)
    t_in = offs_t - 4
    x = tl.load(x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :], mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    a13 += tl.sum(x * g1, axis=0); a27 += tl.sum(x * g2, axis=0)

    t_in = offs_t - 3
    x = tl.load(x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :], mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    a00 += tl.sum(x * g0, axis=0); a14 += tl.sum(x * g1, axis=0); a28 += tl.sum(x * g2, axis=0)
    t_in = offs_t - 2
    x = tl.load(x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :], mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    a01 += tl.sum(x * g0, axis=0); a15 += tl.sum(x * g1, axis=0); a29 += tl.sum(x * g2, axis=0)
    t_in = offs_t - 1
    x = tl.load(x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :], mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    a02 += tl.sum(x * g0, axis=0); a16 += tl.sum(x * g1, axis=0); a210 += tl.sum(x * g2, axis=0)
    t_in = offs_t
    x = tl.load(x_ptr + (t_in[:, None] * B + pid_b) * C + offs_c[None, :], mask=(t_in[:, None] >= 0) & t_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    a03 += tl.sum(x * g0, axis=0); a17 += tl.sum(x * g1, axis=0); a211 += tl.sum(x * g2, axis=0)

    partial_idx = pid_t * B + pid_b
    base0 = (partial_idx * C + offs_c) * 4
    base1 = (partial_idx * C + offs_c) * 8
    base2 = (partial_idx * C + offs_c) * 12
    tl.store(pw0_ptr + base0 + 0, a00, mask=c_mask); tl.store(pw0_ptr + base0 + 1, a01, mask=c_mask); tl.store(pw0_ptr + base0 + 2, a02, mask=c_mask); tl.store(pw0_ptr + base0 + 3, a03, mask=c_mask)
    tl.store(pw1_ptr + base1 + 0, a10, mask=c_mask); tl.store(pw1_ptr + base1 + 1, a11, mask=c_mask); tl.store(pw1_ptr + base1 + 2, a12, mask=c_mask); tl.store(pw1_ptr + base1 + 3, a13, mask=c_mask)
    tl.store(pw1_ptr + base1 + 4, a14, mask=c_mask); tl.store(pw1_ptr + base1 + 5, a15, mask=c_mask); tl.store(pw1_ptr + base1 + 6, a16, mask=c_mask); tl.store(pw1_ptr + base1 + 7, a17, mask=c_mask)
    tl.store(pw2_ptr + base2 + 0, a20, mask=c_mask); tl.store(pw2_ptr + base2 + 1, a21, mask=c_mask); tl.store(pw2_ptr + base2 + 2, a22, mask=c_mask); tl.store(pw2_ptr + base2 + 3, a23, mask=c_mask)
    tl.store(pw2_ptr + base2 + 4, a24, mask=c_mask); tl.store(pw2_ptr + base2 + 5, a25, mask=c_mask); tl.store(pw2_ptr + base2 + 6, a26, mask=c_mask); tl.store(pw2_ptr + base2 + 7, a27, mask=c_mask)
    tl.store(pw2_ptr + base2 + 8, a28, mask=c_mask); tl.store(pw2_ptr + base2 + 9, a29, mask=c_mask); tl.store(pw2_ptr + base2 + 10, a210, mask=c_mask); tl.store(pw2_ptr + base2 + 11, a211, mask=c_mask)
    if HAS_BIAS:
        tl.store(pb0_ptr + partial_idx * C + offs_c, b0, mask=c_mask)
        tl.store(pb1_ptr + partial_idx * C + offs_c, b1, mask=c_mask)
        tl.store(pb2_ptr + partial_idx * C + offs_c, b2, mask=c_mask)


@triton.jit
def xhc_temporal_conv_wgrad_k4812_reduce_fused_kernel(
    pw0_ptr, pw1_ptr, pw2_ptr,
    pb0_ptr, pb1_ptr, pb2_ptr,
    gw0_ptr, gw1_ptr, gw2_ptr,
    gb0_ptr, gb1_ptr, gb2_ptr,
    C: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C

    b0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    b1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    b2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a00 = tl.zeros((BLOCK_C,), dtype=tl.float32); a01 = tl.zeros((BLOCK_C,), dtype=tl.float32); a02 = tl.zeros((BLOCK_C,), dtype=tl.float32); a03 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a10 = tl.zeros((BLOCK_C,), dtype=tl.float32); a11 = tl.zeros((BLOCK_C,), dtype=tl.float32); a12 = tl.zeros((BLOCK_C,), dtype=tl.float32); a13 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a14 = tl.zeros((BLOCK_C,), dtype=tl.float32); a15 = tl.zeros((BLOCK_C,), dtype=tl.float32); a16 = tl.zeros((BLOCK_C,), dtype=tl.float32); a17 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a20 = tl.zeros((BLOCK_C,), dtype=tl.float32); a21 = tl.zeros((BLOCK_C,), dtype=tl.float32); a22 = tl.zeros((BLOCK_C,), dtype=tl.float32); a23 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a24 = tl.zeros((BLOCK_C,), dtype=tl.float32); a25 = tl.zeros((BLOCK_C,), dtype=tl.float32); a26 = tl.zeros((BLOCK_C,), dtype=tl.float32); a27 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    a28 = tl.zeros((BLOCK_C,), dtype=tl.float32); a29 = tl.zeros((BLOCK_C,), dtype=tl.float32); a210 = tl.zeros((BLOCK_C,), dtype=tl.float32); a211 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for p in tl.static_range(NUM_PARTIALS):
        base0 = (p * C + offs_c) * 4
        base1 = (p * C + offs_c) * 8
        base2 = (p * C + offs_c) * 12
        a00 += tl.load(pw0_ptr + base0 + 0, mask=c_mask, other=0.0); a01 += tl.load(pw0_ptr + base0 + 1, mask=c_mask, other=0.0); a02 += tl.load(pw0_ptr + base0 + 2, mask=c_mask, other=0.0); a03 += tl.load(pw0_ptr + base0 + 3, mask=c_mask, other=0.0)
        a10 += tl.load(pw1_ptr + base1 + 0, mask=c_mask, other=0.0); a11 += tl.load(pw1_ptr + base1 + 1, mask=c_mask, other=0.0); a12 += tl.load(pw1_ptr + base1 + 2, mask=c_mask, other=0.0); a13 += tl.load(pw1_ptr + base1 + 3, mask=c_mask, other=0.0)
        a14 += tl.load(pw1_ptr + base1 + 4, mask=c_mask, other=0.0); a15 += tl.load(pw1_ptr + base1 + 5, mask=c_mask, other=0.0); a16 += tl.load(pw1_ptr + base1 + 6, mask=c_mask, other=0.0); a17 += tl.load(pw1_ptr + base1 + 7, mask=c_mask, other=0.0)
        a20 += tl.load(pw2_ptr + base2 + 0, mask=c_mask, other=0.0); a21 += tl.load(pw2_ptr + base2 + 1, mask=c_mask, other=0.0); a22 += tl.load(pw2_ptr + base2 + 2, mask=c_mask, other=0.0); a23 += tl.load(pw2_ptr + base2 + 3, mask=c_mask, other=0.0)
        a24 += tl.load(pw2_ptr + base2 + 4, mask=c_mask, other=0.0); a25 += tl.load(pw2_ptr + base2 + 5, mask=c_mask, other=0.0); a26 += tl.load(pw2_ptr + base2 + 6, mask=c_mask, other=0.0); a27 += tl.load(pw2_ptr + base2 + 7, mask=c_mask, other=0.0)
        a28 += tl.load(pw2_ptr + base2 + 8, mask=c_mask, other=0.0); a29 += tl.load(pw2_ptr + base2 + 9, mask=c_mask, other=0.0); a210 += tl.load(pw2_ptr + base2 + 10, mask=c_mask, other=0.0); a211 += tl.load(pw2_ptr + base2 + 11, mask=c_mask, other=0.0)
        if HAS_BIAS:
            b0 += tl.load(pb0_ptr + p * C + offs_c, mask=c_mask, other=0.0)
            b1 += tl.load(pb1_ptr + p * C + offs_c, mask=c_mask, other=0.0)
            b2 += tl.load(pb2_ptr + p * C + offs_c, mask=c_mask, other=0.0)

    tl.store(gw0_ptr + offs_c * 4 + 0, a00.to(tl.bfloat16), mask=c_mask); tl.store(gw0_ptr + offs_c * 4 + 1, a01.to(tl.bfloat16), mask=c_mask); tl.store(gw0_ptr + offs_c * 4 + 2, a02.to(tl.bfloat16), mask=c_mask); tl.store(gw0_ptr + offs_c * 4 + 3, a03.to(tl.bfloat16), mask=c_mask)
    tl.store(gw1_ptr + offs_c * 8 + 0, a10.to(tl.bfloat16), mask=c_mask); tl.store(gw1_ptr + offs_c * 8 + 1, a11.to(tl.bfloat16), mask=c_mask); tl.store(gw1_ptr + offs_c * 8 + 2, a12.to(tl.bfloat16), mask=c_mask); tl.store(gw1_ptr + offs_c * 8 + 3, a13.to(tl.bfloat16), mask=c_mask)
    tl.store(gw1_ptr + offs_c * 8 + 4, a14.to(tl.bfloat16), mask=c_mask); tl.store(gw1_ptr + offs_c * 8 + 5, a15.to(tl.bfloat16), mask=c_mask); tl.store(gw1_ptr + offs_c * 8 + 6, a16.to(tl.bfloat16), mask=c_mask); tl.store(gw1_ptr + offs_c * 8 + 7, a17.to(tl.bfloat16), mask=c_mask)
    tl.store(gw2_ptr + offs_c * 12 + 0, a20.to(tl.bfloat16), mask=c_mask); tl.store(gw2_ptr + offs_c * 12 + 1, a21.to(tl.bfloat16), mask=c_mask); tl.store(gw2_ptr + offs_c * 12 + 2, a22.to(tl.bfloat16), mask=c_mask); tl.store(gw2_ptr + offs_c * 12 + 3, a23.to(tl.bfloat16), mask=c_mask)
    tl.store(gw2_ptr + offs_c * 12 + 4, a24.to(tl.bfloat16), mask=c_mask); tl.store(gw2_ptr + offs_c * 12 + 5, a25.to(tl.bfloat16), mask=c_mask); tl.store(gw2_ptr + offs_c * 12 + 6, a26.to(tl.bfloat16), mask=c_mask); tl.store(gw2_ptr + offs_c * 12 + 7, a27.to(tl.bfloat16), mask=c_mask)
    tl.store(gw2_ptr + offs_c * 12 + 8, a28.to(tl.bfloat16), mask=c_mask); tl.store(gw2_ptr + offs_c * 12 + 9, a29.to(tl.bfloat16), mask=c_mask); tl.store(gw2_ptr + offs_c * 12 + 10, a210.to(tl.bfloat16), mask=c_mask); tl.store(gw2_ptr + offs_c * 12 + 11, a211.to(tl.bfloat16), mask=c_mask)
    if HAS_BIAS:
        tl.store(gb0_ptr + offs_c, b0.to(tl.bfloat16), mask=c_mask)
        tl.store(gb1_ptr + offs_c, b1.to(tl.bfloat16), mask=c_mask)
        tl.store(gb2_ptr + offs_c, b2.to(tl.bfloat16), mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 256}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 512}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_C": 512}, num_warps=8, num_stages=3),
    ],
    key=["C", "R"],
)
@triton.jit
def xhc_post_fwd_slim_kernel(
    comb_ptr,       # [M, R, C] bf16
    hp_ptr,         # [M, K*R] fp32
    rw_ptr,         # [M, 2] fp32
    mixed_ptr,      # [M, K, C] bf16
    out_ptr,        # [M, K, C] bf16 output — updated values for K active positions
    M,
    K: tl.constexpr,
    C,
    R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Slim post forward: rank-lift + scale + BDA, output [M, K, C].
    No N=16 scatter loop. Caller clones residual and scatters K results.
    2D grid (M, cdiv(C, BLOCK_C)).
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    comb_base = comb_ptr + pid_m * R * C
    mixed_base = mixed_ptr + pid_m * K * C
    out_base = out_ptr + pid_m * K * C

    rw0 = 1.0
    rw1 = 1.0
    rw2 = tl.load(rw_ptr + pid_m * 2 + 0).to(tl.float32)
    rw3 = tl.load(rw_ptr + pid_m * 2 + 1).to(tl.float32)

    hp_base = hp_ptr + pid_m * K * R
    hp_00 = tl.load(hp_base + 0).to(tl.float32)
    hp_10 = tl.load(hp_base + 1 * R + 0).to(tl.float32)
    hp_20 = tl.load(hp_base + 2 * R + 0).to(tl.float32)
    hp_30 = tl.load(hp_base + 3 * R + 0).to(tl.float32)
    if R > 1:
        hp_01 = tl.load(hp_base + 0 * R + 1).to(tl.float32)
        hp_02 = tl.load(hp_base + 0 * R + 2).to(tl.float32)
        hp_03 = tl.load(hp_base + 0 * R + 3).to(tl.float32)
        hp_11 = tl.load(hp_base + 1 * R + 1).to(tl.float32)
        hp_12 = tl.load(hp_base + 1 * R + 2).to(tl.float32)
        hp_13 = tl.load(hp_base + 1 * R + 3).to(tl.float32)
        hp_21 = tl.load(hp_base + 2 * R + 1).to(tl.float32)
        hp_22 = tl.load(hp_base + 2 * R + 2).to(tl.float32)
        hp_23 = tl.load(hp_base + 2 * R + 3).to(tl.float32)
        hp_31 = tl.load(hp_base + 3 * R + 1).to(tl.float32)
        hp_32 = tl.load(hp_base + 3 * R + 2).to(tl.float32)
        hp_33 = tl.load(hp_base + 3 * R + 3).to(tl.float32)

    comb0 = tl.load(comb_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    if R > 1:
        comb1 = tl.load(comb_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        comb2 = tl.load(comb_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        comb3 = tl.load(comb_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    mx0 = tl.load(mixed_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    mx1 = tl.load(mixed_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    mx2 = tl.load(mixed_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    mx3 = tl.load(mixed_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    if R == 1:
        upd0 = (hp_00 * comb0) * rw0 + mx0
        upd1 = (hp_10 * comb0) * rw1 + mx1
        upd2 = (hp_20 * comb0) * rw2 + mx2
        upd3 = (hp_30 * comb0) * rw3 + mx3
    else:
        pa0 = hp_00*comb0 + hp_01*comb1 + hp_02*comb2 + hp_03*comb3
        pa1 = hp_10*comb0 + hp_11*comb1 + hp_12*comb2 + hp_13*comb3
        pa2 = hp_20*comb0 + hp_21*comb1 + hp_22*comb2 + hp_23*comb3
        pa3 = hp_30*comb0 + hp_31*comb1 + hp_32*comb2 + hp_33*comb3
        upd0 = pa0 * rw0 + mx0
        upd1 = pa1 * rw1 + mx1
        upd2 = pa2 * rw2 + mx2
        upd3 = pa3 * rw3 + mx3

    tl.store(out_base + 0 * C + c_offs, upd0.to(tl.bfloat16), mask=c_mask)
    tl.store(out_base + 1 * C + c_offs, upd1.to(tl.bfloat16), mask=c_mask)
    tl.store(out_base + 2 * C + c_offs, upd2.to(tl.bfloat16), mask=c_mask)
    tl.store(out_base + 3 * C + c_offs, upd3.to(tl.bfloat16), mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 128}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 512}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 512}, num_warps=4, num_stages=2),
    ],
    key=["C"],
)
@triton.jit
def xhc_post_fwd_slim_split_r4_kernel(
    main_ptr,       # [M, C] bf16
    conv_ptr,       # [M, 3, C] bf16
    hp_ptr,         # [M, 4*4] fp32
    rw_ptr,         # [M, 2] fp32
    mixed_ptr,      # [M, 4, C] bf16
    out_ptr,        # [M, 4, C] bf16
    M,
    C,
    BLOCK_C: tl.constexpr,
):
    """R=4 MLP post forward without materializing cat([main, conv])."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    main_base = main_ptr + pid_m * C
    conv_base = conv_ptr + pid_m * 3 * C
    mixed_base = mixed_ptr + pid_m * 4 * C
    out_base = out_ptr + pid_m * 4 * C

    comb0 = tl.load(main_base + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    comb1 = tl.load(conv_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    comb2 = tl.load(conv_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    comb3 = tl.load(conv_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    mx0 = tl.load(mixed_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    mx1 = tl.load(mixed_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    mx2 = tl.load(mixed_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    mx3 = tl.load(mixed_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    hp_base = hp_ptr + pid_m * 16
    hp_00 = tl.load(hp_base + 0).to(tl.float32)
    hp_01 = tl.load(hp_base + 1).to(tl.float32)
    hp_02 = tl.load(hp_base + 2).to(tl.float32)
    hp_03 = tl.load(hp_base + 3).to(tl.float32)
    hp_10 = tl.load(hp_base + 4).to(tl.float32)
    hp_11 = tl.load(hp_base + 5).to(tl.float32)
    hp_12 = tl.load(hp_base + 6).to(tl.float32)
    hp_13 = tl.load(hp_base + 7).to(tl.float32)
    hp_20 = tl.load(hp_base + 8).to(tl.float32)
    hp_21 = tl.load(hp_base + 9).to(tl.float32)
    hp_22 = tl.load(hp_base + 10).to(tl.float32)
    hp_23 = tl.load(hp_base + 11).to(tl.float32)
    hp_30 = tl.load(hp_base + 12).to(tl.float32)
    hp_31 = tl.load(hp_base + 13).to(tl.float32)
    hp_32 = tl.load(hp_base + 14).to(tl.float32)
    hp_33 = tl.load(hp_base + 15).to(tl.float32)

    rw0 = 1.0
    rw1 = 1.0
    rw2 = tl.load(rw_ptr + pid_m * 2 + 0).to(tl.float32)
    rw3 = tl.load(rw_ptr + pid_m * 2 + 1).to(tl.float32)

    pa0 = hp_00*comb0 + hp_01*comb1 + hp_02*comb2 + hp_03*comb3
    pa1 = hp_10*comb0 + hp_11*comb1 + hp_12*comb2 + hp_13*comb3
    pa2 = hp_20*comb0 + hp_21*comb1 + hp_22*comb2 + hp_23*comb3
    pa3 = hp_30*comb0 + hp_31*comb1 + hp_32*comb2 + hp_33*comb3

    tl.store(out_base + 0 * C + c_offs, (pa0 * rw0 + mx0).to(tl.bfloat16), mask=c_mask)
    tl.store(out_base + 1 * C + c_offs, (pa1 * rw1 + mx1).to(tl.bfloat16), mask=c_mask)
    tl.store(out_base + 2 * C + c_offs, (pa2 * rw2 + mx2).to(tl.bfloat16), mask=c_mask)
    tl.store(out_base + 3 * C + c_offs, (pa3 * rw3 + mx3).to(tl.bfloat16), mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 512}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 1024}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 2048}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 1024}, num_warps=4, num_stages=2),
    ],
    key=["C"],
)
@triton.jit
def xhc_post_update_scatter_kernel(
    updated_ptr,    # [M, K, C]
    topk_ptr,       # [M, K]
    out_ptr,        # [M, N, C], prefilled with residual
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    C,
    BLOCK_C: tl.constexpr,
):
    """Scatter only the K updated rows into an already-cloned residual tensor."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_c = tl.program_id(2)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    tk = tl.load(topk_ptr + pid_m * K + pid_k)
    val = tl.load(updated_ptr + (pid_m * K + pid_k) * C + c_offs, mask=c_mask, other=0.0)
    tl.store(out_ptr + (pid_m * N + tk) * C + c_offs, val, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 128}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 512}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=2),
    ],
    key=["C"],
)
@triton.jit
def xhc_post_fwd_update_r1_kernel(
    comb_ptr,       # [M, C] bf16
    hp_ptr,         # [M, 4] fp32
    rw_ptr,         # [M, 2] fp32
    mixed_ptr,      # [M, 4, C] bf16
    topk_ptr,       # [M, 4]
    out_ptr,        # [M, N, C], prefilled with residual
    M,
    N: tl.constexpr,
    C,
    BLOCK_C: tl.constexpr,
):
    """R=1 post forward rank-lift + K scatter into cloned residual."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    comb = tl.load(comb_ptr + pid_m * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    mixed_base = mixed_ptr + pid_m * 4 * C
    out_m_base = out_ptr + pid_m * N * C
    hp_base = hp_ptr + pid_m * 4
    topk_base = topk_ptr + pid_m * 4

    hp0 = tl.load(hp_base + 0).to(tl.float32)
    hp1 = tl.load(hp_base + 1).to(tl.float32)
    hp2 = tl.load(hp_base + 2).to(tl.float32)
    hp3 = tl.load(hp_base + 3).to(tl.float32)
    rw0 = 1.0
    rw1 = 1.0
    rw2 = tl.load(rw_ptr + pid_m * 2 + 0).to(tl.float32)
    rw3 = tl.load(rw_ptr + pid_m * 2 + 1).to(tl.float32)

    mx0 = tl.load(mixed_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    mx1 = tl.load(mixed_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    mx2 = tl.load(mixed_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    mx3 = tl.load(mixed_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    tk0 = tl.load(topk_base + 0)
    tk1 = tl.load(topk_base + 1)
    tk2 = tl.load(topk_base + 2)
    tk3 = tl.load(topk_base + 3)
    tl.store(out_m_base + tk0 * C + c_offs, (hp0 * comb * rw0 + mx0).to(tl.bfloat16), mask=c_mask)
    tl.store(out_m_base + tk1 * C + c_offs, (hp1 * comb * rw1 + mx1).to(tl.bfloat16), mask=c_mask)
    tl.store(out_m_base + tk2 * C + c_offs, (hp2 * comb * rw2 + mx2).to(tl.bfloat16), mask=c_mask)
    tl.store(out_m_base + tk3 * C + c_offs, (hp3 * comb * rw3 + mx3).to(tl.bfloat16), mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 128}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 512}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=2),
    ],
    key=["C"],
)
@triton.jit
def xhc_post_fwd_update_split_r4_kernel(
    main_ptr,       # [M, C] bf16
    conv_ptr,       # [M, 3, C] bf16
    hp_ptr,         # [M, 16] fp32
    rw_ptr,         # [M, 2] fp32
    mixed_ptr,      # [M, 4, C] bf16
    topk_ptr,       # [M, 4]
    out_ptr,        # [M, N, C], prefilled with residual
    M,
    N: tl.constexpr,
    C,
    BLOCK_C: tl.constexpr,
):
    """R=4 split MLP post forward rank-lift + K scatter into cloned residual."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    main_base = main_ptr + pid_m * C
    conv_base = conv_ptr + pid_m * 3 * C
    mixed_base = mixed_ptr + pid_m * 4 * C
    out_m_base = out_ptr + pid_m * N * C

    comb0 = tl.load(main_base + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    comb1 = tl.load(conv_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    comb2 = tl.load(conv_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    comb3 = tl.load(conv_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    hp_base = hp_ptr + pid_m * 16
    hp_00 = tl.load(hp_base + 0).to(tl.float32)
    hp_01 = tl.load(hp_base + 1).to(tl.float32)
    hp_02 = tl.load(hp_base + 2).to(tl.float32)
    hp_03 = tl.load(hp_base + 3).to(tl.float32)
    hp_10 = tl.load(hp_base + 4).to(tl.float32)
    hp_11 = tl.load(hp_base + 5).to(tl.float32)
    hp_12 = tl.load(hp_base + 6).to(tl.float32)
    hp_13 = tl.load(hp_base + 7).to(tl.float32)
    hp_20 = tl.load(hp_base + 8).to(tl.float32)
    hp_21 = tl.load(hp_base + 9).to(tl.float32)
    hp_22 = tl.load(hp_base + 10).to(tl.float32)
    hp_23 = tl.load(hp_base + 11).to(tl.float32)
    hp_30 = tl.load(hp_base + 12).to(tl.float32)
    hp_31 = tl.load(hp_base + 13).to(tl.float32)
    hp_32 = tl.load(hp_base + 14).to(tl.float32)
    hp_33 = tl.load(hp_base + 15).to(tl.float32)

    rw0 = 1.0
    rw1 = 1.0
    rw2 = tl.load(rw_ptr + pid_m * 2 + 0).to(tl.float32)
    rw3 = tl.load(rw_ptr + pid_m * 2 + 1).to(tl.float32)

    mx0 = tl.load(mixed_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    mx1 = tl.load(mixed_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    mx2 = tl.load(mixed_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    mx3 = tl.load(mixed_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    pa0 = hp_00*comb0 + hp_01*comb1 + hp_02*comb2 + hp_03*comb3
    pa1 = hp_10*comb0 + hp_11*comb1 + hp_12*comb2 + hp_13*comb3
    pa2 = hp_20*comb0 + hp_21*comb1 + hp_22*comb2 + hp_23*comb3
    pa3 = hp_30*comb0 + hp_31*comb1 + hp_32*comb2 + hp_33*comb3

    topk_base = topk_ptr + pid_m * 4
    tk0 = tl.load(topk_base + 0)
    tk1 = tl.load(topk_base + 1)
    tk2 = tl.load(topk_base + 2)
    tk3 = tl.load(topk_base + 3)
    tl.store(out_m_base + tk0 * C + c_offs, (pa0 * rw0 + mx0).to(tl.bfloat16), mask=c_mask)
    tl.store(out_m_base + tk1 * C + c_offs, (pa1 * rw1 + mx1).to(tl.bfloat16), mask=c_mask)
    tl.store(out_m_base + tk2 * C + c_offs, (pa2 * rw2 + mx2).to(tl.bfloat16), mask=c_mask)
    tl.store(out_m_base + tk3 * C + c_offs, (pa3 * rw3 + mx3).to(tl.bfloat16), mask=c_mask)


@triton.jit
def xhc_post_bwd_gather_zero_inplace_kernel(
    go_ptr,         # [M, N, C], updated in-place to grad_residual
    topk_ptr,       # [M, K]
    gu_ptr,         # [M, K, C]
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    C,
    STORE_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Gather active grad rows and zero them in-place in grad_residual."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_c = tl.program_id(2)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    tk = tl.load(topk_ptr + pid_m * K + pid_k)
    go_ptrs = go_ptr + (pid_m * N + tk) * C + c_offs
    go = tl.load(go_ptrs, mask=c_mask, other=0.0).to(tl.float32)
    gu_ptrs = gu_ptr + (pid_m * K + pid_k) * C + c_offs
    if STORE_DTYPE == 1:
        tl.store(gu_ptrs, go.to(tl.bfloat16), mask=c_mask)
        tl.store(go_ptrs, tl.zeros((BLOCK_C,), dtype=tl.float32).to(tl.bfloat16), mask=c_mask)
    elif STORE_DTYPE == 2:
        tl.store(gu_ptrs, go.to(tl.float16), mask=c_mask)
        tl.store(go_ptrs, tl.zeros((BLOCK_C,), dtype=tl.float32).to(tl.float16), mask=c_mask)
    else:
        tl.store(gu_ptrs, go, mask=c_mask)
        tl.store(go_ptrs, tl.zeros((BLOCK_C,), dtype=tl.float32), mask=c_mask)


@triton.jit
def xhc_post_bwd_gather_zero_inplace_k4_kernel(
    go_ptr,         # [M, N, C], updated in-place to grad_residual
    topk_ptr,       # [M, K]
    gu_ptr,         # [M, K, C]
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    C,
    STORE_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """K-wide variant for K=4: gather and zero all active rows per C tile."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    k_offs = tl.arange(0, K)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    tk = tl.load(topk_ptr + pid_m * K + k_offs)
    go_ptrs = go_ptr + (pid_m * N + tk[:, None]) * C + c_offs[None, :]
    go = tl.load(go_ptrs, mask=c_mask[None, :], other=0.0).to(tl.float32)
    gu_ptrs = gu_ptr + (pid_m * K + k_offs[:, None]) * C + c_offs[None, :]
    zeros = tl.zeros((K, BLOCK_C), dtype=tl.float32)
    if STORE_DTYPE == 1:
        tl.store(gu_ptrs, go.to(tl.bfloat16), mask=c_mask[None, :])
        tl.store(go_ptrs, zeros.to(tl.bfloat16), mask=c_mask[None, :])
    elif STORE_DTYPE == 2:
        tl.store(gu_ptrs, go.to(tl.float16), mask=c_mask[None, :])
        tl.store(go_ptrs, zeros.to(tl.float16), mask=c_mask[None, :])
    else:
        tl.store(gu_ptrs, go, mask=c_mask[None, :])
        tl.store(go_ptrs, zeros, mask=c_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 128}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 512}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_C": 512}, num_warps=8, num_stages=3),
    ],
    key=["C", "R"],
)
@triton.jit
def xhc_post_bwd_slim_kernel(
    gu_ptr,         # [M, K, C] bf16 — grad of updated (pre-gathered by caller)
    comb_ptr,       # [M, R, C] bf16
    hp_ptr,         # [M, K*R] fp32
    rw_ptr,         # [M, 2] fp32
    gc_ptr,         # [M, R, C] output — grad_combined
    ghp_ptr,        # [M, K*R] output fp32 — grad_h_post
    grw_ptr,        # [M, 2] output fp32 — grad_rw
    M,
    K: tl.constexpr,
    C,
    R: tl.constexpr,
    STORE_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Slim post backward on pre-gathered [M, K, C]. No N=16 scatter/gather.
    Caller handles: gather grad_updated from grad_output, scatter grad_residual.
    grad_mixed = grad_updated (pass-through, caller reuses gu directly).
    1D grid (M,).
    """
    pid = tl.program_id(0)

    gu_base = gu_ptr + pid * K * C
    comb_base = comb_ptr + pid * R * C
    gc_base = gc_ptr + pid * R * C

    rw0 = 1.0
    rw1 = 1.0
    rw2 = tl.load(rw_ptr + pid * 2 + 0).to(tl.float32)
    rw3 = tl.load(rw_ptr + pid * 2 + 1).to(tl.float32)

    hp_base = hp_ptr + pid * K * R

    ghp_00 = 0.0; ghp_01 = 0.0; ghp_02 = 0.0; ghp_03 = 0.0
    ghp_10 = 0.0; ghp_11 = 0.0; ghp_12 = 0.0; ghp_13 = 0.0
    ghp_20 = 0.0; ghp_21 = 0.0; ghp_22 = 0.0; ghp_23 = 0.0
    ghp_30 = 0.0; ghp_31 = 0.0; ghp_32 = 0.0; ghp_33 = 0.0
    grw_0 = 0.0; grw_1 = 0.0; grw_2 = 0.0; grw_3 = 0.0

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C

        gu0 = tl.load(gu_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gu1 = tl.load(gu_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gu2 = tl.load(gu_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gu3 = tl.load(gu_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        gpa0 = gu0 * rw0
        gpa1 = gu1 * rw1
        gpa2 = gu2 * rw2
        gpa3 = gu3 * rw3

        for r in tl.static_range(R):
            comb_r = tl.load(comb_base + r * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

            hp_0r = tl.load(hp_base + 0 * R + r).to(tl.float32)
            hp_1r = tl.load(hp_base + 1 * R + r).to(tl.float32)
            hp_2r = tl.load(hp_base + 2 * R + r).to(tl.float32)
            hp_3r = tl.load(hp_base + 3 * R + r).to(tl.float32)

            gc_r = gpa0 * hp_0r + gpa1 * hp_1r + gpa2 * hp_2r + gpa3 * hp_3r
            if STORE_DTYPE == 1:
                tl.store(gc_base + r * C + c_offs, gc_r.to(tl.bfloat16), mask=c_mask)
            elif STORE_DTYPE == 2:
                tl.store(gc_base + r * C + c_offs, gc_r.to(tl.float16), mask=c_mask)
            else:
                tl.store(gc_base + r * C + c_offs, gc_r, mask=c_mask)

            dot0 = tl.sum(gu0 * comb_r)
            dot1 = tl.sum(gu1 * comb_r)
            dot2 = tl.sum(gu2 * comb_r)
            dot3 = tl.sum(gu3 * comb_r)

            if r == 0:
                ghp_00 += rw0 * dot0; ghp_10 += rw1 * dot1
                ghp_20 += rw2 * dot2; ghp_30 += rw3 * dot3
            elif r == 1:
                ghp_01 += rw0 * dot0; ghp_11 += rw1 * dot1
                ghp_21 += rw2 * dot2; ghp_31 += rw3 * dot3
            elif r == 2:
                ghp_02 += rw0 * dot0; ghp_12 += rw1 * dot1
                ghp_22 += rw2 * dot2; ghp_32 += rw3 * dot3
            else:
                ghp_03 += rw0 * dot0; ghp_13 += rw1 * dot1
                ghp_23 += rw2 * dot2; ghp_33 += rw3 * dot3

            grw_0 += hp_0r * dot0
            grw_1 += hp_1r * dot1
            grw_2 += hp_2r * dot2
            grw_3 += hp_3r * dot3

    tl.store(ghp_ptr + pid * K * R + 0, ghp_00)
    if R > 1:
        tl.store(ghp_ptr + pid * K * R + 1, ghp_01)
        tl.store(ghp_ptr + pid * K * R + 2, ghp_02)
        tl.store(ghp_ptr + pid * K * R + 3, ghp_03)
    if R == 1:
        tl.store(ghp_ptr + pid * K * R + 1, ghp_10)
        tl.store(ghp_ptr + pid * K * R + 2, ghp_20)
        tl.store(ghp_ptr + pid * K * R + 3, ghp_30)
    else:
        tl.store(ghp_ptr + pid * K * R + 4, ghp_10)
        tl.store(ghp_ptr + pid * K * R + 5, ghp_11)
        tl.store(ghp_ptr + pid * K * R + 6, ghp_12)
        tl.store(ghp_ptr + pid * K * R + 7, ghp_13)
        tl.store(ghp_ptr + pid * K * R + 8, ghp_20)
        tl.store(ghp_ptr + pid * K * R + 9, ghp_21)
        tl.store(ghp_ptr + pid * K * R + 10, ghp_22)
        tl.store(ghp_ptr + pid * K * R + 11, ghp_23)
        tl.store(ghp_ptr + pid * K * R + 12, ghp_30)
        tl.store(ghp_ptr + pid * K * R + 13, ghp_31)
        tl.store(ghp_ptr + pid * K * R + 14, ghp_32)
        tl.store(ghp_ptr + pid * K * R + 15, ghp_33)

    tl.store(grw_ptr + pid * 2 + 0, grw_2)
    tl.store(grw_ptr + pid * 2 + 1, grw_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 128}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 512}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=2),
    ],
    key=["C"],
)
@triton.jit
def xhc_post_bwd_slim_split_r4_kernel(
    gu_ptr,         # [M, 4, C] bf16
    main_ptr,       # [M, C] bf16
    conv_ptr,       # [M, 3, C] bf16
    hp_ptr,         # [M, 16] fp32
    rw_ptr,         # [M, 2] fp32
    gm_ptr,         # [M, C] output
    gc_ptr,         # [M, 3, C] output
    ghp_ptr,        # [M, 16] fp32 output
    grw_ptr,        # [M, 2] fp32 output
    M,
    C,
    STORE_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """R=4 MLP post backward without materializing grad_combined."""
    pid = tl.program_id(0)

    gu_base = gu_ptr + pid * 4 * C
    main_base = main_ptr + pid * C
    conv_base = conv_ptr + pid * 3 * C
    gm_base = gm_ptr + pid * C
    gc_base = gc_ptr + pid * 3 * C
    hp_base = hp_ptr + pid * 16

    rw0 = 1.0
    rw1 = 1.0
    rw2 = tl.load(rw_ptr + pid * 2 + 0).to(tl.float32)
    rw3 = tl.load(rw_ptr + pid * 2 + 1).to(tl.float32)

    hp_00 = tl.load(hp_base + 0).to(tl.float32)
    hp_01 = tl.load(hp_base + 1).to(tl.float32)
    hp_02 = tl.load(hp_base + 2).to(tl.float32)
    hp_03 = tl.load(hp_base + 3).to(tl.float32)
    hp_10 = tl.load(hp_base + 4).to(tl.float32)
    hp_11 = tl.load(hp_base + 5).to(tl.float32)
    hp_12 = tl.load(hp_base + 6).to(tl.float32)
    hp_13 = tl.load(hp_base + 7).to(tl.float32)
    hp_20 = tl.load(hp_base + 8).to(tl.float32)
    hp_21 = tl.load(hp_base + 9).to(tl.float32)
    hp_22 = tl.load(hp_base + 10).to(tl.float32)
    hp_23 = tl.load(hp_base + 11).to(tl.float32)
    hp_30 = tl.load(hp_base + 12).to(tl.float32)
    hp_31 = tl.load(hp_base + 13).to(tl.float32)
    hp_32 = tl.load(hp_base + 14).to(tl.float32)
    hp_33 = tl.load(hp_base + 15).to(tl.float32)

    ghp_00 = 0.0; ghp_01 = 0.0; ghp_02 = 0.0; ghp_03 = 0.0
    ghp_10 = 0.0; ghp_11 = 0.0; ghp_12 = 0.0; ghp_13 = 0.0
    ghp_20 = 0.0; ghp_21 = 0.0; ghp_22 = 0.0; ghp_23 = 0.0
    ghp_30 = 0.0; ghp_31 = 0.0; ghp_32 = 0.0; ghp_33 = 0.0
    grw_0 = 0.0; grw_1 = 0.0; grw_2 = 0.0; grw_3 = 0.0

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C

        gu0 = tl.load(gu_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gu1 = tl.load(gu_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gu2 = tl.load(gu_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gu3 = tl.load(gu_base + 3 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        gpa0 = gu0 * rw0
        gpa1 = gu1 * rw1
        gpa2 = gu2 * rw2
        gpa3 = gu3 * rw3

        comb0 = tl.load(main_base + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        comb1 = tl.load(conv_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        comb2 = tl.load(conv_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        comb3 = tl.load(conv_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        gcomb0 = gpa0 * hp_00 + gpa1 * hp_10 + gpa2 * hp_20 + gpa3 * hp_30
        gcomb1 = gpa0 * hp_01 + gpa1 * hp_11 + gpa2 * hp_21 + gpa3 * hp_31
        gcomb2 = gpa0 * hp_02 + gpa1 * hp_12 + gpa2 * hp_22 + gpa3 * hp_32
        gcomb3 = gpa0 * hp_03 + gpa1 * hp_13 + gpa2 * hp_23 + gpa3 * hp_33

        if STORE_DTYPE == 1:
            tl.store(gm_base + c_offs, gcomb0.to(tl.bfloat16), mask=c_mask)
            tl.store(gc_base + 0 * C + c_offs, gcomb1.to(tl.bfloat16), mask=c_mask)
            tl.store(gc_base + 1 * C + c_offs, gcomb2.to(tl.bfloat16), mask=c_mask)
            tl.store(gc_base + 2 * C + c_offs, gcomb3.to(tl.bfloat16), mask=c_mask)
        elif STORE_DTYPE == 2:
            tl.store(gm_base + c_offs, gcomb0.to(tl.float16), mask=c_mask)
            tl.store(gc_base + 0 * C + c_offs, gcomb1.to(tl.float16), mask=c_mask)
            tl.store(gc_base + 1 * C + c_offs, gcomb2.to(tl.float16), mask=c_mask)
            tl.store(gc_base + 2 * C + c_offs, gcomb3.to(tl.float16), mask=c_mask)
        else:
            tl.store(gm_base + c_offs, gcomb0, mask=c_mask)
            tl.store(gc_base + 0 * C + c_offs, gcomb1, mask=c_mask)
            tl.store(gc_base + 1 * C + c_offs, gcomb2, mask=c_mask)
            tl.store(gc_base + 2 * C + c_offs, gcomb3, mask=c_mask)

        dot0 = tl.sum(gu0 * comb0)
        dot1 = tl.sum(gu1 * comb0)
        dot2 = tl.sum(gu2 * comb0)
        dot3 = tl.sum(gu3 * comb0)
        ghp_00 += rw0 * dot0; ghp_10 += rw1 * dot1
        ghp_20 += rw2 * dot2; ghp_30 += rw3 * dot3
        grw_0 += hp_00 * dot0; grw_1 += hp_10 * dot1
        grw_2 += hp_20 * dot2; grw_3 += hp_30 * dot3

        dot0 = tl.sum(gu0 * comb1)
        dot1 = tl.sum(gu1 * comb1)
        dot2 = tl.sum(gu2 * comb1)
        dot3 = tl.sum(gu3 * comb1)
        ghp_01 += rw0 * dot0; ghp_11 += rw1 * dot1
        ghp_21 += rw2 * dot2; ghp_31 += rw3 * dot3
        grw_0 += hp_01 * dot0; grw_1 += hp_11 * dot1
        grw_2 += hp_21 * dot2; grw_3 += hp_31 * dot3

        dot0 = tl.sum(gu0 * comb2)
        dot1 = tl.sum(gu1 * comb2)
        dot2 = tl.sum(gu2 * comb2)
        dot3 = tl.sum(gu3 * comb2)
        ghp_02 += rw0 * dot0; ghp_12 += rw1 * dot1
        ghp_22 += rw2 * dot2; ghp_32 += rw3 * dot3
        grw_0 += hp_02 * dot0; grw_1 += hp_12 * dot1
        grw_2 += hp_22 * dot2; grw_3 += hp_32 * dot3

        dot0 = tl.sum(gu0 * comb3)
        dot1 = tl.sum(gu1 * comb3)
        dot2 = tl.sum(gu2 * comb3)
        dot3 = tl.sum(gu3 * comb3)
        ghp_03 += rw0 * dot0; ghp_13 += rw1 * dot1
        ghp_23 += rw2 * dot2; ghp_33 += rw3 * dot3
        grw_0 += hp_03 * dot0; grw_1 += hp_13 * dot1
        grw_2 += hp_23 * dot2; grw_3 += hp_33 * dot3

    tl.store(ghp_ptr + pid * 16 + 0, ghp_00)
    tl.store(ghp_ptr + pid * 16 + 1, ghp_01)
    tl.store(ghp_ptr + pid * 16 + 2, ghp_02)
    tl.store(ghp_ptr + pid * 16 + 3, ghp_03)
    tl.store(ghp_ptr + pid * 16 + 4, ghp_10)
    tl.store(ghp_ptr + pid * 16 + 5, ghp_11)
    tl.store(ghp_ptr + pid * 16 + 6, ghp_12)
    tl.store(ghp_ptr + pid * 16 + 7, ghp_13)
    tl.store(ghp_ptr + pid * 16 + 8, ghp_20)
    tl.store(ghp_ptr + pid * 16 + 9, ghp_21)
    tl.store(ghp_ptr + pid * 16 + 10, ghp_22)
    tl.store(ghp_ptr + pid * 16 + 11, ghp_23)
    tl.store(ghp_ptr + pid * 16 + 12, ghp_30)
    tl.store(ghp_ptr + pid * 16 + 13, ghp_31)
    tl.store(ghp_ptr + pid * 16 + 14, ghp_32)
    tl.store(ghp_ptr + pid * 16 + 15, ghp_33)

    tl.store(grw_ptr + pid * 2 + 0, grw_2)
    tl.store(grw_ptr + pid * 2 + 1, grw_3)


@triton.jit
def xhc_post_bwd_gather_slim_r1_kernel(
    go_ptr,       # [M, N, C] in/out grad residual; active rows zeroed
    topk_ptr,     # [M, 4]
    comb_ptr,     # [M, C]
    hp_ptr,       # [M, 4] fp32
    rw_ptr,       # [M, 2] fp32
    gc_ptr,       # [M, C] output
    ghp_ptr,      # [M, 4] fp32 output
    grw_ptr,      # [M, 2] fp32 output
    gu_out_ptr,   # [M, 4, C] output grad_mixed passthrough
    M,
    N: tl.constexpr,
    C,
    STORE_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fuse K=4 post-bwd gather/zero with R=1 slim backward."""
    pid = tl.program_id(0)

    go_m_base = go_ptr + pid * N * C
    topk_base = topk_ptr + pid * 4
    comb_base = comb_ptr + pid * C
    gc_base = gc_ptr + pid * C
    gu_base = gu_out_ptr + pid * 4 * C
    hp_base = hp_ptr + pid * 4

    tk0 = tl.load(topk_base + 0)
    tk1 = tl.load(topk_base + 1)
    tk2 = tl.load(topk_base + 2)
    tk3 = tl.load(topk_base + 3)

    hp0 = tl.load(hp_base + 0).to(tl.float32)
    hp1 = tl.load(hp_base + 1).to(tl.float32)
    hp2 = tl.load(hp_base + 2).to(tl.float32)
    hp3 = tl.load(hp_base + 3).to(tl.float32)

    rw0 = 1.0
    rw1 = 1.0
    rw2 = tl.load(rw_ptr + pid * 2 + 0).to(tl.float32)
    rw3 = tl.load(rw_ptr + pid * 2 + 1).to(tl.float32)

    ghp0 = 0.0
    ghp1 = 0.0
    ghp2 = 0.0
    ghp3 = 0.0
    grw2 = 0.0
    grw3 = 0.0

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C

        go0_ptrs = go_m_base + tk0 * C + c_offs
        go1_ptrs = go_m_base + tk1 * C + c_offs
        go2_ptrs = go_m_base + tk2 * C + c_offs
        go3_ptrs = go_m_base + tk3 * C + c_offs
        gu0 = tl.load(go0_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        gu1 = tl.load(go1_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        gu2 = tl.load(go2_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        gu3 = tl.load(go3_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        zeros = tl.zeros((BLOCK_C,), dtype=tl.float32)

        if STORE_DTYPE == 1:
            tl.store(gu_base + 0 * C + c_offs, gu0.to(tl.bfloat16), mask=c_mask)
            tl.store(gu_base + 1 * C + c_offs, gu1.to(tl.bfloat16), mask=c_mask)
            tl.store(gu_base + 2 * C + c_offs, gu2.to(tl.bfloat16), mask=c_mask)
            tl.store(gu_base + 3 * C + c_offs, gu3.to(tl.bfloat16), mask=c_mask)
            tl.store(go0_ptrs, zeros.to(tl.bfloat16), mask=c_mask)
            tl.store(go1_ptrs, zeros.to(tl.bfloat16), mask=c_mask)
            tl.store(go2_ptrs, zeros.to(tl.bfloat16), mask=c_mask)
            tl.store(go3_ptrs, zeros.to(tl.bfloat16), mask=c_mask)
        elif STORE_DTYPE == 2:
            tl.store(gu_base + 0 * C + c_offs, gu0.to(tl.float16), mask=c_mask)
            tl.store(gu_base + 1 * C + c_offs, gu1.to(tl.float16), mask=c_mask)
            tl.store(gu_base + 2 * C + c_offs, gu2.to(tl.float16), mask=c_mask)
            tl.store(gu_base + 3 * C + c_offs, gu3.to(tl.float16), mask=c_mask)
            tl.store(go0_ptrs, zeros.to(tl.float16), mask=c_mask)
            tl.store(go1_ptrs, zeros.to(tl.float16), mask=c_mask)
            tl.store(go2_ptrs, zeros.to(tl.float16), mask=c_mask)
            tl.store(go3_ptrs, zeros.to(tl.float16), mask=c_mask)
        else:
            tl.store(gu_base + 0 * C + c_offs, gu0, mask=c_mask)
            tl.store(gu_base + 1 * C + c_offs, gu1, mask=c_mask)
            tl.store(gu_base + 2 * C + c_offs, gu2, mask=c_mask)
            tl.store(gu_base + 3 * C + c_offs, gu3, mask=c_mask)
            tl.store(go0_ptrs, zeros, mask=c_mask)
            tl.store(go1_ptrs, zeros, mask=c_mask)
            tl.store(go2_ptrs, zeros, mask=c_mask)
            tl.store(go3_ptrs, zeros, mask=c_mask)

        gpa0 = gu0 * rw0
        gpa1 = gu1 * rw1
        gpa2 = gu2 * rw2
        gpa3 = gu3 * rw3
        comb = tl.load(comb_base + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        gcomb = gpa0 * hp0 + gpa1 * hp1 + gpa2 * hp2 + gpa3 * hp3

        if STORE_DTYPE == 1:
            tl.store(gc_base + c_offs, gcomb.to(tl.bfloat16), mask=c_mask)
        elif STORE_DTYPE == 2:
            tl.store(gc_base + c_offs, gcomb.to(tl.float16), mask=c_mask)
        else:
            tl.store(gc_base + c_offs, gcomb, mask=c_mask)

        dot0 = tl.sum(gu0 * comb)
        dot1 = tl.sum(gu1 * comb)
        dot2 = tl.sum(gu2 * comb)
        dot3 = tl.sum(gu3 * comb)
        ghp0 += rw0 * dot0
        ghp1 += rw1 * dot1
        ghp2 += rw2 * dot2
        ghp3 += rw3 * dot3
        grw2 += hp2 * dot2
        grw3 += hp3 * dot3

    tl.store(ghp_ptr + pid * 4 + 0, ghp0)
    tl.store(ghp_ptr + pid * 4 + 1, ghp1)
    tl.store(ghp_ptr + pid * 4 + 2, ghp2)
    tl.store(ghp_ptr + pid * 4 + 3, ghp3)
    tl.store(grw_ptr + pid * 2 + 0, grw2)
    tl.store(grw_ptr + pid * 2 + 1, grw3)


@triton.jit
def xhc_post_bwd_gather_slim_split_r4_kernel(
    go_ptr,       # [M, N, C] in/out grad residual; active rows zeroed
    topk_ptr,     # [M, 4]
    main_ptr,     # [M, C]
    conv_ptr,     # [M, 3, C]
    hp_ptr,       # [M, 16] fp32
    rw_ptr,       # [M, 4] fp32
    gm_ptr,       # [M, C] output
    gc_ptr,       # [M, 3, C] output
    ghp_ptr,      # [M, 16] fp32 output
    grw_ptr,      # [M, 2] fp32 output
    gu_out_ptr,   # [M, 4, C] output grad_mixed passthrough
    M,
    N: tl.constexpr,
    C,
    STORE_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fuse K=4 post-bwd gather/zero with MLP split-R4 slim backward."""
    pid = tl.program_id(0)

    go_m_base = go_ptr + pid * N * C
    topk_base = topk_ptr + pid * 4
    main_base = main_ptr + pid * C
    conv_base = conv_ptr + pid * 3 * C
    gm_base = gm_ptr + pid * C
    gc_base = gc_ptr + pid * 3 * C
    gu_base = gu_out_ptr + pid * 4 * C
    hp_base = hp_ptr + pid * 16

    tk0 = tl.load(topk_base + 0)
    tk1 = tl.load(topk_base + 1)
    tk2 = tl.load(topk_base + 2)
    tk3 = tl.load(topk_base + 3)

    rw0 = 1.0
    rw1 = 1.0
    rw2 = tl.load(rw_ptr + pid * 2 + 0).to(tl.float32)
    rw3 = tl.load(rw_ptr + pid * 2 + 1).to(tl.float32)

    hp_00 = tl.load(hp_base + 0).to(tl.float32)
    hp_01 = tl.load(hp_base + 1).to(tl.float32)
    hp_02 = tl.load(hp_base + 2).to(tl.float32)
    hp_03 = tl.load(hp_base + 3).to(tl.float32)
    hp_10 = tl.load(hp_base + 4).to(tl.float32)
    hp_11 = tl.load(hp_base + 5).to(tl.float32)
    hp_12 = tl.load(hp_base + 6).to(tl.float32)
    hp_13 = tl.load(hp_base + 7).to(tl.float32)
    hp_20 = tl.load(hp_base + 8).to(tl.float32)
    hp_21 = tl.load(hp_base + 9).to(tl.float32)
    hp_22 = tl.load(hp_base + 10).to(tl.float32)
    hp_23 = tl.load(hp_base + 11).to(tl.float32)
    hp_30 = tl.load(hp_base + 12).to(tl.float32)
    hp_31 = tl.load(hp_base + 13).to(tl.float32)
    hp_32 = tl.load(hp_base + 14).to(tl.float32)
    hp_33 = tl.load(hp_base + 15).to(tl.float32)

    ghp_00 = 0.0; ghp_01 = 0.0; ghp_02 = 0.0; ghp_03 = 0.0
    ghp_10 = 0.0; ghp_11 = 0.0; ghp_12 = 0.0; ghp_13 = 0.0
    ghp_20 = 0.0; ghp_21 = 0.0; ghp_22 = 0.0; ghp_23 = 0.0
    ghp_30 = 0.0; ghp_31 = 0.0; ghp_32 = 0.0; ghp_33 = 0.0
    grw_0 = 0.0; grw_1 = 0.0; grw_2 = 0.0; grw_3 = 0.0

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C

        go0_ptrs = go_m_base + tk0 * C + c_offs
        go1_ptrs = go_m_base + tk1 * C + c_offs
        go2_ptrs = go_m_base + tk2 * C + c_offs
        go3_ptrs = go_m_base + tk3 * C + c_offs
        gu0 = tl.load(go0_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        gu1 = tl.load(go1_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        gu2 = tl.load(go2_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        gu3 = tl.load(go3_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        zeros = tl.zeros((BLOCK_C,), dtype=tl.float32)

        if STORE_DTYPE == 1:
            tl.store(gu_base + 0 * C + c_offs, gu0.to(tl.bfloat16), mask=c_mask)
            tl.store(gu_base + 1 * C + c_offs, gu1.to(tl.bfloat16), mask=c_mask)
            tl.store(gu_base + 2 * C + c_offs, gu2.to(tl.bfloat16), mask=c_mask)
            tl.store(gu_base + 3 * C + c_offs, gu3.to(tl.bfloat16), mask=c_mask)
            tl.store(go0_ptrs, zeros.to(tl.bfloat16), mask=c_mask)
            tl.store(go1_ptrs, zeros.to(tl.bfloat16), mask=c_mask)
            tl.store(go2_ptrs, zeros.to(tl.bfloat16), mask=c_mask)
            tl.store(go3_ptrs, zeros.to(tl.bfloat16), mask=c_mask)
        elif STORE_DTYPE == 2:
            tl.store(gu_base + 0 * C + c_offs, gu0.to(tl.float16), mask=c_mask)
            tl.store(gu_base + 1 * C + c_offs, gu1.to(tl.float16), mask=c_mask)
            tl.store(gu_base + 2 * C + c_offs, gu2.to(tl.float16), mask=c_mask)
            tl.store(gu_base + 3 * C + c_offs, gu3.to(tl.float16), mask=c_mask)
            tl.store(go0_ptrs, zeros.to(tl.float16), mask=c_mask)
            tl.store(go1_ptrs, zeros.to(tl.float16), mask=c_mask)
            tl.store(go2_ptrs, zeros.to(tl.float16), mask=c_mask)
            tl.store(go3_ptrs, zeros.to(tl.float16), mask=c_mask)
        else:
            tl.store(gu_base + 0 * C + c_offs, gu0, mask=c_mask)
            tl.store(gu_base + 1 * C + c_offs, gu1, mask=c_mask)
            tl.store(gu_base + 2 * C + c_offs, gu2, mask=c_mask)
            tl.store(gu_base + 3 * C + c_offs, gu3, mask=c_mask)
            tl.store(go0_ptrs, zeros, mask=c_mask)
            tl.store(go1_ptrs, zeros, mask=c_mask)
            tl.store(go2_ptrs, zeros, mask=c_mask)
            tl.store(go3_ptrs, zeros, mask=c_mask)

        gpa0 = gu0 * rw0
        gpa1 = gu1 * rw1
        gpa2 = gu2 * rw2
        gpa3 = gu3 * rw3

        comb0 = tl.load(main_base + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        comb1 = tl.load(conv_base + 0 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        comb2 = tl.load(conv_base + 1 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        comb3 = tl.load(conv_base + 2 * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        gcomb0 = gpa0 * hp_00 + gpa1 * hp_10 + gpa2 * hp_20 + gpa3 * hp_30
        gcomb1 = gpa0 * hp_01 + gpa1 * hp_11 + gpa2 * hp_21 + gpa3 * hp_31
        gcomb2 = gpa0 * hp_02 + gpa1 * hp_12 + gpa2 * hp_22 + gpa3 * hp_32
        gcomb3 = gpa0 * hp_03 + gpa1 * hp_13 + gpa2 * hp_23 + gpa3 * hp_33

        if STORE_DTYPE == 1:
            tl.store(gm_base + c_offs, gcomb0.to(tl.bfloat16), mask=c_mask)
            tl.store(gc_base + 0 * C + c_offs, gcomb1.to(tl.bfloat16), mask=c_mask)
            tl.store(gc_base + 1 * C + c_offs, gcomb2.to(tl.bfloat16), mask=c_mask)
            tl.store(gc_base + 2 * C + c_offs, gcomb3.to(tl.bfloat16), mask=c_mask)
        elif STORE_DTYPE == 2:
            tl.store(gm_base + c_offs, gcomb0.to(tl.float16), mask=c_mask)
            tl.store(gc_base + 0 * C + c_offs, gcomb1.to(tl.float16), mask=c_mask)
            tl.store(gc_base + 1 * C + c_offs, gcomb2.to(tl.float16), mask=c_mask)
            tl.store(gc_base + 2 * C + c_offs, gcomb3.to(tl.float16), mask=c_mask)
        else:
            tl.store(gm_base + c_offs, gcomb0, mask=c_mask)
            tl.store(gc_base + 0 * C + c_offs, gcomb1, mask=c_mask)
            tl.store(gc_base + 1 * C + c_offs, gcomb2, mask=c_mask)
            tl.store(gc_base + 2 * C + c_offs, gcomb3, mask=c_mask)

        dot0 = tl.sum(gu0 * comb0); dot1 = tl.sum(gu1 * comb0)
        dot2 = tl.sum(gu2 * comb0); dot3 = tl.sum(gu3 * comb0)
        ghp_00 += rw0 * dot0; ghp_10 += rw1 * dot1
        ghp_20 += rw2 * dot2; ghp_30 += rw3 * dot3
        grw_0 += hp_00 * dot0; grw_1 += hp_10 * dot1
        grw_2 += hp_20 * dot2; grw_3 += hp_30 * dot3

        dot0 = tl.sum(gu0 * comb1); dot1 = tl.sum(gu1 * comb1)
        dot2 = tl.sum(gu2 * comb1); dot3 = tl.sum(gu3 * comb1)
        ghp_01 += rw0 * dot0; ghp_11 += rw1 * dot1
        ghp_21 += rw2 * dot2; ghp_31 += rw3 * dot3
        grw_0 += hp_01 * dot0; grw_1 += hp_11 * dot1
        grw_2 += hp_21 * dot2; grw_3 += hp_31 * dot3

        dot0 = tl.sum(gu0 * comb2); dot1 = tl.sum(gu1 * comb2)
        dot2 = tl.sum(gu2 * comb2); dot3 = tl.sum(gu3 * comb2)
        ghp_02 += rw0 * dot0; ghp_12 += rw1 * dot1
        ghp_22 += rw2 * dot2; ghp_32 += rw3 * dot3
        grw_0 += hp_02 * dot0; grw_1 += hp_12 * dot1
        grw_2 += hp_22 * dot2; grw_3 += hp_32 * dot3

        dot0 = tl.sum(gu0 * comb3); dot1 = tl.sum(gu1 * comb3)
        dot2 = tl.sum(gu2 * comb3); dot3 = tl.sum(gu3 * comb3)
        ghp_03 += rw0 * dot0; ghp_13 += rw1 * dot1
        ghp_23 += rw2 * dot2; ghp_33 += rw3 * dot3
        grw_0 += hp_03 * dot0; grw_1 += hp_13 * dot1
        grw_2 += hp_23 * dot2; grw_3 += hp_33 * dot3

    tl.store(ghp_ptr + pid * 16 + 0, ghp_00)
    tl.store(ghp_ptr + pid * 16 + 1, ghp_01)
    tl.store(ghp_ptr + pid * 16 + 2, ghp_02)
    tl.store(ghp_ptr + pid * 16 + 3, ghp_03)
    tl.store(ghp_ptr + pid * 16 + 4, ghp_10)
    tl.store(ghp_ptr + pid * 16 + 5, ghp_11)
    tl.store(ghp_ptr + pid * 16 + 6, ghp_12)
    tl.store(ghp_ptr + pid * 16 + 7, ghp_13)
    tl.store(ghp_ptr + pid * 16 + 8, ghp_20)
    tl.store(ghp_ptr + pid * 16 + 9, ghp_21)
    tl.store(ghp_ptr + pid * 16 + 10, ghp_22)
    tl.store(ghp_ptr + pid * 16 + 11, ghp_23)
    tl.store(ghp_ptr + pid * 16 + 12, ghp_30)
    tl.store(ghp_ptr + pid * 16 + 13, ghp_31)
    tl.store(ghp_ptr + pid * 16 + 14, ghp_32)
    tl.store(ghp_ptr + pid * 16 + 15, ghp_33)
    tl.store(grw_ptr + pid * 2 + 0, grw_2)
    tl.store(grw_ptr + pid * 2 + 1, grw_3)


class _TritonTemporalConvK4812Func(torch.autograd.Function):
    """Specialized depthwise temporal conv for current MHC workload.

    Supports exactly three causal depthwise conv branches with kernel sizes
    4, 8, 12 and [S, B, C] bf16 input/output layout.
    """

    @staticmethod
    def forward(ctx, hidden_states, w0, b0, w1, b1, w2, b2):
        S, B, C = hidden_states.shape
        out = torch.empty(S, B, 3, C, device=hidden_states.device, dtype=hidden_states.dtype)
        has_bias = b0 is not None
        dummy = hidden_states.reshape(-1)
        b0_arg = b0 if b0 is not None else dummy
        b1_arg = b1 if b1 is not None else dummy
        b2_arg = b2 if b2 is not None else dummy
        block_t = 8
        block_c = 64
        grid = (triton.cdiv(S, block_t), B, triton.cdiv(C, block_c))
        xhc_temporal_conv_k4812_fwd_kernel[grid](hidden_states, w0, b0_arg, w1, b1_arg, w2, b2_arg, out, S=S, B=B, C=C, HAS_BIAS=has_bias, BLOCK_T=block_t, BLOCK_C=block_c, num_warps=2)
        ctx.save_for_backward(hidden_states, w0, w1, w2)
        ctx.has_bias = has_bias
        ctx.shape_info = (S, B, C)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, w0, w1, w2 = ctx.saved_tensors
        S, B, C = ctx.shape_info
        max_k = 12
        x_perm = None
        grad_y = None
        grad_raw = None

        def ensure_grad_raw():
            nonlocal x_perm, grad_y, grad_raw
            if x_perm is None:
                x_perm = hidden_states.permute(1, 2, 0)
            if grad_y is None:
                grad_y = grad_output.permute(1, 3, 2, 0).reshape(B, C * 3, S).contiguous()
            if grad_raw is None:
                grad_raw = F.pad(grad_y, (0, max_k - 1))
            return (x_perm, grad_y, grad_raw)
        use_triton_common = HAS_TRITON and grad_output.is_cuda and (grad_output.dtype == torch.bfloat16)
        if use_triton_common:
            grad_hidden = torch.empty_like(hidden_states)
            block_t = 8
            block_c = 64
            grid = (triton.cdiv(S, block_t), B, triton.cdiv(C, block_c))
            xhc_temporal_conv_k4812_dx_kernel[grid](grad_output, w0, w1, w2, grad_hidden, S=S, B=B, C=C, stride_go_s=grad_output.stride(0), stride_go_b=grad_output.stride(1), stride_go_k=grad_output.stride(2), stride_go_c=grad_output.stride(3), BLOCK_T=block_t, BLOCK_C=block_c, num_warps=2)
        else:
            x_perm, _, grad_raw = ensure_grad_raw()
            packed_weight = torch.stack([F.pad(w0, (max_k - w0.shape[-1], 0)), F.pad(w1, (max_k - w1.shape[-1], 0)), F.pad(w2, (max_k - w2.shape[-1], 0))], dim=1).reshape(C * 3, 1, max_k).contiguous()
            grad_hidden = torch.nn.grad.conv1d_input(x_perm.shape, packed_weight, grad_raw, padding=max_k - 1, groups=C).permute(2, 0, 1)
        if use_triton_common:
            use_split_wgrad = True
            use_fused_branch_wgrad = False
            block_t = _temporal_conv_k4812_wgrad_block_t()
            block_c_default = 16 if use_fused_branch_wgrad else 32 if use_split_wgrad else 16
            block_c = _temporal_conv_k4812_wgrad_block_c(block_c_default)
            grid = (triton.cdiv(C, block_c),)
            grad_w0 = torch.empty_like(w0)
            grad_w1 = torch.empty_like(w1)
            grad_w2 = torch.empty_like(w2)
            grad_b0 = torch.empty(C, device=grad_output.device, dtype=grad_output.dtype) if ctx.has_bias else None
            grad_b1 = torch.empty(C, device=grad_output.device, dtype=grad_output.dtype) if ctx.has_bias else None
            grad_b2 = torch.empty(C, device=grad_output.device, dtype=grad_output.dtype) if ctx.has_bias else None
            dummy = grad_w0
            if use_split_wgrad:
                num_t_tiles = triton.cdiv(S, block_t)
                num_partials = num_t_tiles * B
                if use_fused_branch_wgrad:
                    partial_w0 = torch.empty(num_partials, C, 4, device=grad_output.device, dtype=torch.float32)
                    partial_w1 = torch.empty(num_partials, C, 8, device=grad_output.device, dtype=torch.float32)
                    partial_w2 = torch.empty(num_partials, C, 12, device=grad_output.device, dtype=torch.float32)
                    partial_b0 = torch.empty(num_partials, C, device=grad_output.device, dtype=torch.float32) if ctx.has_bias else dummy
                    partial_b1 = torch.empty(num_partials, C, device=grad_output.device, dtype=torch.float32) if ctx.has_bias else dummy
                    partial_b2 = torch.empty(num_partials, C, device=grad_output.device, dtype=torch.float32) if ctx.has_bias else dummy
                    xhc_temporal_conv_wgrad_k4812_partial_fused_kernel[triton.cdiv(C, block_c), num_t_tiles, B](hidden_states, grad_output, partial_w0, partial_w1, partial_w2, partial_b0, partial_b1, partial_b2, S=S, B=B, C=C, stride_go_s=grad_output.stride(0), stride_go_b=grad_output.stride(1), stride_go_k=grad_output.stride(2), stride_go_c=grad_output.stride(3), HAS_BIAS=ctx.has_bias, BLOCK_T=block_t, BLOCK_C=block_c, num_warps=8)
                    xhc_temporal_conv_wgrad_k4812_reduce_fused_kernel[triton.cdiv(C, block_c),](partial_w0, partial_w1, partial_w2, partial_b0, partial_b1, partial_b2, grad_w0, grad_w1, grad_w2, grad_b0 if grad_b0 is not None else dummy, grad_b1 if grad_b1 is not None else dummy, grad_b2 if grad_b2 is not None else dummy, C=C, NUM_PARTIALS=num_partials, HAS_BIAS=ctx.has_bias, BLOCK_C=block_c, num_warps=1)
                else:

                    def run_split_wgrad(branch, k, grad_w, grad_b):
                        partial_w = torch.empty(num_partials, C, k, device=grad_output.device, dtype=torch.float32)
                        partial_b = torch.empty(num_partials, C, device=grad_output.device, dtype=torch.float32) if ctx.has_bias else dummy
                        xhc_temporal_conv_wgrad_partial_kernel[triton.cdiv(C, block_c), num_t_tiles, B](hidden_states, grad_output, partial_w, partial_b, S=S, B=B, C=C, stride_go_s=grad_output.stride(0), stride_go_b=grad_output.stride(1), stride_go_k=grad_output.stride(2), stride_go_c=grad_output.stride(3), BRANCH=branch, K=k, HAS_BIAS=ctx.has_bias, BLOCK_T=block_t, BLOCK_C=block_c, num_warps=8)
                        xhc_temporal_conv_wgrad_reduce_kernel[triton.cdiv(C, block_c),](partial_w, partial_b, grad_w, grad_b if grad_b is not None else dummy, C=C, K=k, NUM_PARTIALS=num_partials, HAS_BIAS=ctx.has_bias, BLOCK_C=block_c, num_warps=1)
                    run_split_wgrad(0, 4, grad_w0, grad_b0)
                    run_split_wgrad(1, 8, grad_w1, grad_b1)
                    run_split_wgrad(2, 12, grad_w2, grad_b2)
            else:
                xhc_temporal_conv_wgrad_kernel[grid](hidden_states, grad_output, grad_w0, grad_b0 if grad_b0 is not None else dummy, S=S, B=B, C=C, stride_go_s=grad_output.stride(0), stride_go_b=grad_output.stride(1), stride_go_k=grad_output.stride(2), stride_go_c=grad_output.stride(3), BRANCH=0, K=4, HAS_BIAS=ctx.has_bias, BLOCK_T=block_t, BLOCK_C=block_c, num_warps=8)
                xhc_temporal_conv_wgrad_kernel[grid](hidden_states, grad_output, grad_w1, grad_b1 if grad_b1 is not None else dummy, S=S, B=B, C=C, stride_go_s=grad_output.stride(0), stride_go_b=grad_output.stride(1), stride_go_k=grad_output.stride(2), stride_go_c=grad_output.stride(3), BRANCH=1, K=8, HAS_BIAS=ctx.has_bias, BLOCK_T=block_t, BLOCK_C=block_c, num_warps=8)
                xhc_temporal_conv_wgrad_kernel[grid](hidden_states, grad_output, grad_w2, grad_b2 if grad_b2 is not None else dummy, S=S, B=B, C=C, stride_go_s=grad_output.stride(0), stride_go_b=grad_output.stride(1), stride_go_k=grad_output.stride(2), stride_go_c=grad_output.stride(3), BRANCH=2, K=12, HAS_BIAS=ctx.has_bias, BLOCK_T=block_t, BLOCK_C=block_c, num_warps=8)
        else:
            x_perm, grad_y, grad_raw = ensure_grad_raw()
            grad_weight = torch.nn.grad.conv1d_weight(x_perm, (C * 3, 1, max_k), grad_raw, padding=max_k - 1, groups=C)
            gw = grad_weight.reshape(C, 3, 1, max_k)
            grad_w0 = gw[:, 0, :, max_k - w0.shape[-1]:].contiguous()
            grad_w1 = gw[:, 1, :, max_k - w1.shape[-1]:].contiguous()
            grad_w2 = gw[:, 2, :, max_k - w2.shape[-1]:].contiguous()
            if ctx.has_bias:
                gb = grad_y.sum(dim=(0, 2)).reshape(C, 3)
                grad_b0 = gb[:, 0].contiguous()
                grad_b1 = gb[:, 1].contiguous()
                grad_b2 = gb[:, 2].contiguous()
            else:
                grad_b0 = grad_b1 = grad_b2 = None
        return (grad_hidden, grad_w0.to(w0.dtype), grad_b0.to(w0.dtype) if grad_b0 is not None else None, grad_w1.to(w1.dtype), grad_b1.to(w1.dtype) if grad_b1 is not None else None, grad_w2.to(w2.dtype), grad_b2.to(w2.dtype) if grad_b2 is not None else None)


_HC_FUSED_OPT_LEVEL_RUNTIME = 2


def _set_hc_fused_opt_level(source) -> int:
    global _HC_FUSED_OPT_LEVEL_RUNTIME
    _HC_FUSED_OPT_LEVEL_RUNTIME = 2
    return 2


def _use_xhc_keep_fixed_router_logits(source=None) -> bool:
    cfg = getattr(source, "config", None)
    cfg_value = getattr(cfg, "mhc_xhc_keep_fixed_router_logits", None)
    return True if cfg_value is None else bool(cfg_value)


def _temporal_conv_k4812_wgrad_block_t(default: int = 1024) -> int:
    return default


def _temporal_conv_k4812_wgrad_block_c(default: int = 32) -> int:
    return default


def _bf16_active_proj_fwd_splits(D: int) -> int:
    return 2 if D >= 8192 else 1


def _bf16_active_proj_fwd_block_t(default: int = 64) -> int:
    return default


def _bf16_active_proj_fwd_block_d(default: int = 64) -> int:
    return default


def _mhc_log_inner_memory(mhc_module, tag: str) -> None:
    return


def _use_inline_active_proj_bwd_rms(source=None) -> bool:
    value = os.environ.get("MHC_USE_INLINE_ACTIVE_PROJ_BWD_RMS")
    if value is not None:
        return value.lower() in {"1", "true", "yes", "on"}
    return False


def _use_bf16_fused_active_proj_dhs(D: int, out_dim: int, weight_dtype: torch.dtype) -> bool:
    return weight_dtype == torch.bfloat16 and D >= 8192 and out_dim <= 64


def _bf16_fused_active_proj_dhs_block_m(default: int = 32) -> int:
    return default


def _bf16_fused_active_proj_dhs_block_k(default: int = 128) -> int:
    return default


def _use_bf16_full_proj_wgrad(D: int, out_dim: int) -> bool:
    return D >= 16384 and out_dim <= 64


def _use_bf16_cublas_proj_dhs(D: int, out_dim: int) -> bool:
    return D >= 16384 and out_dim <= 64


def _use_bf16_fused_proj_dhs(D: int, out_dim: int, weight_dtype: torch.dtype) -> bool:
    return weight_dtype == torch.bfloat16 and D >= 32768 and out_dim <= 64


def _bf16_fused_proj_dhs_block_m(default: int = 32) -> int:
    return default


def _bf16_fused_proj_dhs_block_k(default: int = 128) -> int:
    return default


def _use_linear_proj_dhs_correction(D: int) -> bool:
    return D >= 16384


def _use_overlap_split_proj_bwd(D: int, out_dim: int, weight_dtype: torch.dtype) -> bool:
    return weight_dtype == torch.float32 and D >= 16384 and out_dim <= 64


def _use_split_bf16_full_proj_fwd(D: int, out_dim: int, weight_dtype: torch.dtype) -> bool:
    return weight_dtype == torch.bfloat16 and D >= 32768 and out_dim <= 64


def _bf16_full_proj_fwd_splits(D: int) -> int:
    return 4 if D >= 32768 else 1


def _bf16_full_proj_fwd_block_t(default: int = 64) -> int:
    return default


def _bf16_full_proj_fwd_block_d(default: int = 64) -> int:
    return default


_PROJ_BWD_WGRAD_STREAMS = {}


def _get_proj_bwd_wgrad_stream(device) -> torch.cuda.Stream:
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _PROJ_BWD_WGRAD_STREAMS.get(device_index)
    if stream is None:
        with torch.cuda.device(device_index):
            stream = torch.cuda.Stream()
        _PROJ_BWD_WGRAD_STREAMS[device_index] = stream
    return stream


def _recompute_flat_active(hidden_states: Tensor, topk_indices: Tensor) -> Tensor:
    S, B, N, C = hidden_states.shape
    K = topk_indices.shape[2]
    M = S * B
    if HAS_TRITON and hidden_states.is_cuda:
        hs_flat = hidden_states.reshape(M, N, C).contiguous()
        topk_flat = topk_indices.reshape(M, K).contiguous()
        active = torch.empty(M, K, C, device=hidden_states.device, dtype=hidden_states.dtype)
        grid = lambda META: (M, K, triton.cdiv(C, META["BLOCK_C"]))
        xhc_active_gather_kernel[grid](
            hs_flat, topk_flat, active,
            M, N=N, K=K, C=C,
        )
        return active.reshape(M, K * C)

    idx = topk_indices.unsqueeze(-1).expand(-1, -1, -1, C)
    return torch.gather(hidden_states, 2, idx).reshape(M, K * C).contiguous()


def _scatter_flat_active_grad(
    grad_active: Tensor,
    topk_indices: Tensor,
    hidden_shape: Tuple[int, int, int, int],
    origin_dtype: torch.dtype,
) -> Tensor:
    S, B, N, C = hidden_shape
    K = topk_indices.shape[2]
    M = S * B
    if HAS_TRITON and grad_active.is_cuda:
        topk_flat = topk_indices.reshape(M, K).contiguous()
        grad_active_flat = grad_active.reshape(M, K, C).contiguous()
        grad_hidden = torch.empty(M, N, C, device=grad_active.device, dtype=origin_dtype)
        grad_hidden.zero_()
        grid = lambda META: (M, K, triton.cdiv(C, META["BLOCK_C"]))
        xhc_post_update_scatter_kernel[grid](
            grad_active_flat, topk_flat, grad_hidden,
            M, N=N, K=K, C=C,
        )
        return grad_hidden.reshape(S, B, N, C)

    idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, C)
    grad_hidden = torch.zeros(S, B, N, C, device=grad_active.device, dtype=origin_dtype)
    return grad_hidden.scatter_add(2, idx_exp, grad_active.reshape(S, B, K, C).to(origin_dtype))


def _split_pre_res_bwd_block_c(default: int = 512) -> int:
    return default


def _mhc_require_bf16_weight(weight: Tensor, name: str) -> Tensor:
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be torch.bfloat16 on the fused mHC path; got {weight.dtype}.")
    return weight


def _mhc_require_bf16_activation(tensor: Tensor, name: str) -> Tensor:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be torch.bfloat16 on the fused mHC path; got {tensor.dtype}.")
    return tensor


def _proj_bwd_merged(
    hs_flat: Tensor,
    W_for_bwd: Tensor,
    grad_combined: Tensor,
    grad_mean_x: Tensor,
    grad_mean_x2: Tensor,
):
    M, D = hs_flat.shape
    OUT_DIM = W_for_bwd.shape[1]
    origin_dtype = hs_flat.dtype

    if HAS_TRITON and hs_flat.is_cuda:
        # Autograd may pass broadcasted zero-stride gradients for reductions
        # such as out.sum().  The Triton kernel below assumes dense row-major
        # tensors, so materialize the gradient inputs before pointer arithmetic.
        hs_kernel = hs_flat.contiguous()
        W_kernel = W_for_bwd.contiguous()
        grad_combined_kernel = grad_combined.float().contiguous()
        padded_out = max(16, 1 << (OUT_DIM - 1).bit_length())
        store_dtype = 1 if origin_dtype == torch.bfloat16 else (2 if origin_dtype == torch.float16 else 0)
        inv_D = 1.0 / D
        gmx = (
            grad_mean_x.squeeze(-1).float().contiguous()
            if grad_mean_x is not None
            else torch.zeros(M, device=hs_flat.device, dtype=torch.float32)
        )
        gmx2 = (
            grad_mean_x2.squeeze(-1).float().contiguous()
            if grad_mean_x2 is not None
            else torch.zeros(M, device=hs_flat.device, dtype=torch.float32)
        )
        grad_hs = torch.empty(M, D, device=hs_flat.device, dtype=origin_dtype)
        grad_W = torch.empty(D, OUT_DIM, device=hs_flat.device, dtype=torch.float32)
        grid_merged = lambda META: (triton.cdiv(D, META["BLOCK_K"]),)
        xhc_proj_bwd_merged_kernel[grid_merged](
            grad_combined_kernel, gmx, gmx2,
            W_kernel, hs_kernel,
            grad_hs, grad_W,
            M, NC=D, OUT_DIM=OUT_DIM, PADDED_OUT=padded_out,
            inv_D_val=inv_D,
            STORE_DTYPE=store_dtype,
        )
        return grad_hs, grad_W

    hs_bf16 = _mhc_require_bf16_activation(hs_flat, "fused xHC projection backward input")
    W_bf16 = _mhc_require_bf16_weight(W_for_bwd, "fused xHC projection backward weight")
    grad_combined_bf16 = grad_combined.to(torch.bfloat16)
    hs_stats = hs_flat.float()
    grad_hs = torch.matmul(grad_combined_bf16, W_bf16.T).float()
    grad_W = torch.matmul(hs_bf16.T, grad_combined_bf16).float()
    inv_D = 1.0 / D
    if grad_mean_x is not None:
        grad_hs.add_(grad_mean_x, alpha=inv_D)
    if grad_mean_x2 is not None:
        grad_hs.addcmul_(grad_mean_x2.float() * (2.0 * inv_D), hs_stats)
    return grad_hs.to(origin_dtype), grad_W


def _use_split_proj_bwd(D: int, out_dim: int, weight_dtype: torch.dtype) -> bool:
    if D < 16384 or out_dim > 64:
        return False
    if weight_dtype == torch.bfloat16:
        return True
    return weight_dtype == torch.float32


def _proj_bwd_split(
    hs_flat: Tensor,
    W_for_bwd: Tensor,
    grad_combined: Tensor,
    grad_mean_x: Tensor,
    grad_mean_x2: Tensor,
    grad_passthrough: Tensor = None,
):
    M, D = hs_flat.shape
    OUT_DIM = W_for_bwd.shape[1]
    origin_dtype = hs_flat.dtype
    if not (HAS_TRITON and hs_flat.is_cuda):
        return _proj_bwd_merged(hs_flat, W_for_bwd, grad_combined, grad_mean_x, grad_mean_x2)

    hs_kernel = hs_flat.contiguous()
    W_kernel = W_for_bwd.contiguous()
    grad_combined_kernel = grad_combined.float().contiguous()
    has_grad_mean_x = grad_mean_x is not None
    gmx = grad_mean_x.squeeze(-1).float().contiguous() if has_grad_mean_x else grad_combined_kernel.reshape(-1)
    gmx2 = (
        grad_mean_x2.squeeze(-1).float().contiguous()
        if grad_mean_x2 is not None
        else torch.zeros(M, device=hs_flat.device, dtype=torch.float32)
    )
    has_grad_passthrough = grad_passthrough is not None
    if has_grad_passthrough:
        grad_passthrough_kernel = grad_passthrough.reshape(M, D).contiguous()
    else:
        grad_passthrough_kernel = hs_kernel

    padded_out = max(16, 1 << (OUT_DIM - 1).bit_length())
    store_dtype = 1 if origin_dtype == torch.bfloat16 else (2 if origin_dtype == torch.float16 else 0)
    grad_hs = torch.empty(M, D, device=hs_flat.device, dtype=origin_dtype)
    wgrad_stream = None

    if (
        _use_bf16_full_proj_wgrad(D, OUT_DIM)
        and hs_kernel.dtype == torch.bfloat16
        and W_kernel.dtype == torch.bfloat16
    ):
        with _mhc_nvtx_range("mhc.full_proj.bwd.wgrad_cublas"):
            grad_W = torch.matmul(hs_kernel.T, grad_combined_kernel.to(torch.bfloat16)).float()
    else:
        grad_W = torch.empty(D, OUT_DIM, device=hs_flat.device, dtype=torch.float32)

        def launch_wgrad():
            grid_w = lambda META: (triton.cdiv(D, META["BLOCK_K"]),)
            xhc_proj_bwd_wgrad_kernel[grid_w](
                grad_combined_kernel, hs_kernel, grad_W,
                M, NC=D, OUT_DIM=OUT_DIM, PADDED_OUT=padded_out,
            )

        if _use_overlap_split_proj_bwd(D, OUT_DIM, W_kernel.dtype):
            current_stream = torch.cuda.current_stream(device=hs_flat.device)
            wgrad_stream = _get_proj_bwd_wgrad_stream(hs_flat.device)
            wgrad_stream.wait_stream(current_stream)
            with torch.cuda.stream(wgrad_stream):
                launch_wgrad()
        else:
            launch_wgrad()

    if (
        _use_bf16_fused_proj_dhs(D, OUT_DIM, W_kernel.dtype)
        and origin_dtype == torch.bfloat16
        and W_kernel.dtype == torch.bfloat16
    ):
        block_m = _bf16_fused_proj_dhs_block_m()
        block_k = _bf16_fused_proj_dhs_block_k()
        grid_h = (triton.cdiv(M, block_m), triton.cdiv(D, block_k))
        with _mhc_nvtx_range("mhc.full_proj.bwd.dhs_bf16_fused"):
            xhc_proj_bwd_dhs_bf16_fused_kernel[grid_h](
                grad_combined_kernel, gmx, gmx2,
                W_kernel, hs_kernel, grad_passthrough_kernel, grad_hs,
                M, NC=D, OUT_DIM=OUT_DIM, PADDED_OUT=padded_out,
                inv_D_val=1.0 / D,
                BLOCK_M=block_m, BLOCK_K=block_k,
                HAS_GRAD_MEAN_X=has_grad_mean_x,
                HAS_GRAD_PASSTHROUGH=has_grad_passthrough,
                STORE_DTYPE=store_dtype,
                num_warps=4,
                num_stages=3,
            )
        if wgrad_stream is not None:
            torch.cuda.current_stream(device=hs_flat.device).wait_stream(wgrad_stream)
        return grad_hs, grad_W

    if _use_bf16_cublas_proj_dhs(D, OUT_DIM) and origin_dtype == torch.bfloat16 and W_kernel.dtype == torch.bfloat16:
        with _mhc_nvtx_range("mhc.full_proj.bwd.dhs_cublas"):
            grad_hs = torch.matmul(grad_combined_kernel.to(torch.bfloat16), W_kernel.T)
        if _use_linear_proj_dhs_correction(D):
            block = 2048
            with _mhc_nvtx_range("mhc.full_proj.bwd.dhs_correction_linear"):
                xhc_proj_bwd_dhs_correction_linear_kernel[(triton.cdiv(M * D, block),)](
                    grad_hs, gmx, gmx2, hs_kernel, grad_passthrough_kernel,
                    TOTAL=M * D, NC=D,
                    inv_D_val=1.0 / D,
                    HAS_GRAD_MEAN_X=has_grad_mean_x,
                    HAS_GRAD_PASSTHROUGH=has_grad_passthrough,
                    STORE_DTYPE=store_dtype,
                    BLOCK=block,
                    num_warps=4,
                    num_stages=3,
                )
        else:
            block_m = 128
            block_k = 128 if has_grad_passthrough else 256
            grid_h = (triton.cdiv(M, block_m), triton.cdiv(D, block_k))
            with _mhc_nvtx_range("mhc.full_proj.bwd.dhs_correction"):
                xhc_proj_bwd_dhs_correction_kernel[grid_h](
                    grad_hs, gmx, gmx2, hs_kernel, grad_passthrough_kernel,
                    M, NC=D,
                    inv_D_val=1.0 / D,
                    BLOCK_M=block_m, BLOCK_K=block_k,
                    HAS_GRAD_MEAN_X=has_grad_mean_x,
                    HAS_GRAD_PASSTHROUGH=has_grad_passthrough,
                    STORE_DTYPE=store_dtype,
                    num_warps=4,
                    num_stages=3,
                )
        if wgrad_stream is not None:
            torch.cuda.current_stream(device=hs_flat.device).wait_stream(wgrad_stream)
        return grad_hs, grad_W

    block_m = 64
    block_k = 64
    grid_h = (triton.cdiv(M, block_m), triton.cdiv(D, block_k))
    xhc_proj_bwd_dhs_kernel[grid_h](
        grad_combined_kernel, gmx, gmx2,
        W_kernel, hs_kernel, grad_passthrough_kernel, grad_hs,
        M, NC=D, OUT_DIM=OUT_DIM, PADDED_OUT=padded_out,
        inv_D_val=1.0 / D,
        BLOCK_M=block_m, BLOCK_K=block_k,
        HAS_GRAD_MEAN_X=has_grad_mean_x,
        HAS_GRAD_PASSTHROUGH=has_grad_passthrough,
        STORE_DTYPE=store_dtype,
        num_warps=4,
        num_stages=3,
    )
    if wgrad_stream is not None:
        torch.cuda.current_stream(device=hs_flat.device).wait_stream(wgrad_stream)
    return grad_hs, grad_W


def _active_proj_bwd_split(hs_flat: Tensor, W_for_bwd: Tensor, grad_combined: Tensor, grad_mean_x2: Tensor):
    M, D = hs_flat.shape
    OUT_DIM = W_for_bwd.shape[1]
    origin_dtype = hs_flat.dtype
    hs_kernel = hs_flat.contiguous()
    W_kernel = W_for_bwd.contiguous()
    grad_combined_kernel = grad_combined.float().contiguous()
    gmx2 = grad_mean_x2.squeeze(-1).float().contiguous()
    padded_out = max(16, 1 << (OUT_DIM - 1).bit_length())
    store_dtype = 1 if origin_dtype == torch.bfloat16 else 2 if origin_dtype == torch.float16 else 0
    if hs_kernel.dtype == torch.bfloat16 and W_kernel.dtype == torch.bfloat16:
        with _mhc_nvtx_range('mhc.active_proj.bwd.wgrad_cublas'):
            grad_W = torch.matmul(hs_kernel.T, grad_combined_kernel.to(torch.bfloat16)).float()
    else:
        grad_W = torch.empty(D, OUT_DIM, device=hs_flat.device, dtype=torch.float32)
        grid_w = lambda META: (triton.cdiv(D, META['BLOCK_K']),)
        xhc_proj_bwd_wgrad_kernel[grid_w](grad_combined_kernel, hs_kernel, grad_W, M, NC=D, OUT_DIM=OUT_DIM, PADDED_OUT=padded_out)
    if _use_bf16_fused_active_proj_dhs(D, OUT_DIM, W_kernel.dtype) and origin_dtype == torch.bfloat16 and (W_kernel.dtype == torch.bfloat16):
        grad_hs = torch.empty(M, D, device=hs_flat.device, dtype=origin_dtype)
        block_m = _bf16_fused_active_proj_dhs_block_m()
        block_k = _bf16_fused_active_proj_dhs_block_k(256 if OUT_DIM <= 20 else 128)
        grid_h = (triton.cdiv(M, block_m), triton.cdiv(D, block_k))
        with _mhc_nvtx_range('mhc.active_proj.bwd.dhs_bf16_fused'):
            xhc_active_proj_bwd_dhs_bf16_fused_kernel[grid_h](grad_combined_kernel, gmx2, W_kernel, hs_kernel, grad_hs, M, NC=D, OUT_DIM=OUT_DIM, PADDED_OUT=padded_out, inv_D_val=1.0 / D, BLOCK_M=block_m, BLOCK_K=block_k, STORE_DTYPE=store_dtype, num_warps=4, num_stages=3)
        return (grad_hs, grad_W)
    if origin_dtype == torch.bfloat16 and W_kernel.dtype == torch.bfloat16:
        with _mhc_nvtx_range('mhc.active_proj.bwd.dhs_cublas'):
            grad_hs = torch.matmul(grad_combined_kernel.to(torch.bfloat16), W_kernel.T)
        block_m = 64
        block_k = 256
        grid_h = (triton.cdiv(M, block_m), triton.cdiv(D, block_k))
        dummy = grad_combined_kernel.reshape(-1)
        with _mhc_nvtx_range('mhc.active_proj.bwd.dhs_correction'):
            xhc_proj_bwd_dhs_correction_kernel[grid_h](grad_hs, dummy, gmx2, hs_kernel, hs_kernel, M, NC=D, inv_D_val=1.0 / D, BLOCK_M=block_m, BLOCK_K=block_k, HAS_GRAD_MEAN_X=False, HAS_GRAD_PASSTHROUGH=False, STORE_DTYPE=store_dtype, num_warps=4, num_stages=3)
        return (grad_hs, grad_W)
    grad_hs = torch.empty(M, D, device=hs_flat.device, dtype=origin_dtype)
    block_m = 64
    block_k = 64
    grid_h = (triton.cdiv(M, block_m), triton.cdiv(D, block_k))
    dummy = grad_combined_kernel.reshape(-1)
    xhc_proj_bwd_dhs_kernel[grid_h](grad_combined_kernel, dummy, gmx2, W_kernel, hs_kernel, hs_kernel, grad_hs, M, NC=D, OUT_DIM=OUT_DIM, PADDED_OUT=padded_out, inv_D_val=1.0 / D, BLOCK_M=block_m, BLOCK_K=block_k, HAS_GRAD_MEAN_X=False, HAS_GRAD_PASSTHROUGH=False, STORE_DTYPE=store_dtype, num_warps=4, num_stages=3)
    return (grad_hs, grad_W)


def _active_proj_bwd_rms_torch(
    hs_flat: Tensor,
    W_for_bwd: Tensor,
    grad_projected: Tensor,
    projected: Tensor,
    mean_x2: Tensor,
):
    M, _ = hs_flat.shape
    mean_x2_2d = mean_x2.reshape(M, 1)
    inv_rms = torch.rsqrt(mean_x2_2d).nan_to_num(1.0, 1.0, 1.0)
    grad_projected_fp32 = grad_projected.float()
    grad_raw = grad_projected_fp32 * inv_rms
    grad_mean_x2 = (
        -0.5
        * (grad_projected_fp32 * projected.float()).sum(dim=-1, keepdim=True)
        / mean_x2_2d.clamp_min(1e-30)
    )
    return _proj_bwd_torch_bf16(hs_flat, W_for_bwd, grad_raw, None, grad_mean_x2)


def _active_proj_bwd_rms_split(
    hs_flat: Tensor,
    W_for_bwd: Tensor,
    grad_projected: Tensor,
    projected: Tensor,
    mean_x2: Tensor,
):
    M, D = hs_flat.shape
    OUT_DIM = W_for_bwd.shape[1]
    origin_dtype = hs_flat.dtype
    if not (
        HAS_TRITON
        and hs_flat.is_cuda
        and origin_dtype in (torch.bfloat16, torch.float16)
        and W_for_bwd.dtype in (torch.float32, torch.bfloat16)
        and OUT_DIM <= 64
    ):
        return _active_proj_bwd_rms_torch(hs_flat, W_for_bwd, grad_projected, projected, mean_x2)

    hs_kernel = hs_flat.contiguous()
    W_kernel = W_for_bwd.contiguous()
    grad_projected_kernel = (
        grad_projected.contiguous()
        if grad_projected.dtype == torch.float32
        else grad_projected.float().contiguous()
    )
    projected_kernel = projected.contiguous() if projected.dtype == torch.float32 else projected.float().contiguous()
    mean_x2_kernel = mean_x2.reshape(M).float().contiguous()
    padded_out = max(16, 1 << (OUT_DIM - 1).bit_length())
    store_dtype = 1 if origin_dtype == torch.bfloat16 else (2 if origin_dtype == torch.float16 else 0)

    grad_W = torch.empty(D, OUT_DIM, device=hs_flat.device, dtype=torch.float32)
    block_k = 128
    block_t = 64
    xhc_active_proj_bwd_wgrad_rms_kernel[(triton.cdiv(D, block_k),)](
        grad_projected_kernel, mean_x2_kernel, hs_kernel, grad_W,
        M, NC=D, OUT_DIM=OUT_DIM, PADDED_OUT=padded_out,
        BLOCK_K=block_k, BLOCK_T=block_t,
        num_warps=4, num_stages=2,
    )

    grad_hs = torch.empty(M, D, device=hs_flat.device, dtype=origin_dtype)
    block_m = 64
    block_k = 64
    grid_h = (triton.cdiv(M, block_m), triton.cdiv(D, block_k))
    xhc_active_proj_bwd_dhs_rms_kernel[grid_h](
        grad_projected_kernel, projected_kernel, mean_x2_kernel,
        W_kernel, hs_kernel, grad_hs,
        M, NC=D, OUT_DIM=OUT_DIM, PADDED_OUT=padded_out,
        inv_D_val=1.0 / D,
        BLOCK_M=block_m, BLOCK_K=block_k,
        STORE_DTYPE=store_dtype,
        num_warps=4, num_stages=3,
    )
    return grad_hs, grad_W


def _proj_bwd_torch_bf16(hs_flat: Tensor, W_for_bwd: Tensor, grad_combined: Tensor, grad_mean_x: Tensor, grad_mean_x2: Tensor):
    """Projection backward fallback using bf16 matmul operands."""
    M, D = hs_flat.shape
    if HAS_TRITON and hs_flat.is_cuda and (grad_mean_x is None) and (grad_mean_x2 is not None) and (hs_flat.dtype in (torch.bfloat16, torch.float16)):
        return _active_proj_bwd_split(hs_flat, W_for_bwd, grad_combined, grad_mean_x2)
    hs_bf16 = _mhc_require_bf16_activation(hs_flat, 'fused xHC projection backward input')
    W_bf16 = _mhc_require_bf16_weight(W_for_bwd, 'fused xHC projection backward weight')
    grad_combined_bf16 = grad_combined.to(torch.bfloat16)
    hs_stats = hs_flat.float()
    grad_hs = torch.matmul(grad_combined_bf16, W_bf16.T).float()
    grad_W = torch.matmul(hs_bf16.T, grad_combined_bf16).float()
    inv_D = 1.0 / D
    if grad_mean_x is not None:
        grad_hs.add_(grad_mean_x.float(), alpha=inv_D)
    if grad_mean_x2 is not None:
        grad_hs.addcmul_(grad_mean_x2.float() * (2.0 * inv_D), hs_stats)
    return (grad_hs.to(hs_flat.dtype), grad_W)


def _run_bf16_full_proj_fwd(
    hs_contig: Tensor,
    W_contig: Tensor,
    combined_raw: Tensor,
    mean_x: Tensor,
    mean_x2: Tensor,
    M: int,
    D: int,
    OUT_DIM: int,
    padded_out: int,
) -> None:
    use_split = _use_split_bf16_full_proj_fwd(D, OUT_DIM, W_contig.dtype)
    num_splits = _bf16_full_proj_fwd_splits(D) if use_split else 1
    split_active = use_split and num_splits > 1
    block_t = _bf16_full_proj_fwd_block_t(64)
    block_d = _bf16_full_proj_fwd_block_d(64)

    if split_active:
        chunk_d = triton.cdiv(triton.cdiv(D, block_d), num_splits) * block_d
        partial_out = torch.empty(
            (num_splits, M, OUT_DIM),
            device=hs_contig.device,
            dtype=torch.float32,
        )
        partial_sum = torch.empty((num_splits, M), device=hs_contig.device, dtype=torch.float32)
        partial_sum2 = torch.empty((num_splits, M), device=hs_contig.device, dtype=torch.float32)
        xhc_fwd_proj_bf16_split_matmul_kernel[(triton.cdiv(M, block_t), num_splits)](
            hs_contig, W_contig, partial_out, partial_sum, partial_sum2,
            M, D=D, OUT_DIM=OUT_DIM, stride_hs=hs_contig.stride(0),
            PADDED_OUT=padded_out,
            CHUNK_D=chunk_d,
            BLOCK_T=block_t,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=3,
        )
        xhc_fwd_proj_bf16_split_reduce_kernel[(triton.cdiv(M, block_t),)](
            partial_out, partial_sum, partial_sum2, combined_raw, mean_x, mean_x2,
            M, D=D, OUT_DIM=OUT_DIM,
            PADDED_OUT=padded_out,
            NUM_D_SPLITS=num_splits,
            BLOCK_T=block_t,
            num_warps=4,
            num_stages=3,
        )
        return

    xhc_fwd_proj_bf16_inline_matmul_kernel[(triton.cdiv(M, block_t),)](
        hs_contig, W_contig, combined_raw, mean_x, mean_x2,
        M, D=D, OUT_DIM=OUT_DIM, stride_hs=hs_contig.stride(0),
        PADDED_OUT=padded_out,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )


def _run_bf16_active_proj_fwd(active_contig: Tensor, W_contig: Tensor, projected: Tensor, mean_x2: Tensor, M: int, D: int, OUT_DIM: int, padded_out: int, *, store_mean_x2: bool, use_external_mean_x2: bool) -> None:
    use_split = False
    num_splits = _bf16_active_proj_fwd_splits(D) if use_split else 1
    split_active = use_split and num_splits > 1
    block_t = _bf16_active_proj_fwd_block_t(128 if split_active else 64)
    block_d = _bf16_active_proj_fwd_block_d(64 if split_active else 128)
    if split_active:
        chunk_d = triton.cdiv(triton.cdiv(D, block_d), num_splits) * block_d
        partial_out = torch.empty((num_splits, M, OUT_DIM), device=active_contig.device, dtype=torch.float32)
        has_partial_sum_x2 = not use_external_mean_x2
        if has_partial_sum_x2:
            partial_sum2 = torch.empty((num_splits, M), device=active_contig.device, dtype=torch.float32)
        else:
            partial_sum2 = torch.empty((1,), device=active_contig.device, dtype=torch.float32)
        xhc_active_proj_bf16_split_matmul_kernel[triton.cdiv(M, block_t), num_splits](active_contig, W_contig, partial_out, partial_sum2, M, D=D, OUT_DIM=OUT_DIM, stride_act=active_contig.stride(0), PADDED_OUT=padded_out, CHUNK_D=chunk_d, STORE_SUM_X2=has_partial_sum_x2, BLOCK_T=block_t, BLOCK_D=block_d, num_warps=4, num_stages=3)
        xhc_active_proj_bf16_split_reduce_kernel[triton.cdiv(M, block_t),](partial_out, partial_sum2, mean_x2, projected, M, D=D, OUT_DIM=OUT_DIM, PADDED_OUT=padded_out, NUM_D_SPLITS=num_splits, HAS_PARTIAL_SUM_X2=has_partial_sum_x2, STORE_MEAN_X2=store_mean_x2, BLOCK_T=block_t, num_warps=4, num_stages=3)
        return
    if use_external_mean_x2:
        xhc_active_proj_bf16_with_stats_inline_kernel[triton.cdiv(M, block_t),](active_contig, W_contig, mean_x2, projected, M, D=D, OUT_DIM=OUT_DIM, stride_act=active_contig.stride(0), PADDED_OUT=padded_out, BLOCK_T=block_t, BLOCK_D=block_d, num_warps=4, num_stages=3)
    else:
        xhc_active_proj_bf16_inline_kernel[triton.cdiv(M, block_t),](active_contig, W_contig, projected, mean_x2, M, D=D, OUT_DIM=OUT_DIM, stride_act=active_contig.stride(0), PADDED_OUT=padded_out, STORE_MEAN_X2=store_mean_x2, BLOCK_T=block_t, BLOCK_D=block_d, num_warps=4, num_stages=3)


class FusedProjPassthroughFunction(torch.autograd.Function):
    """Triton projection with a hidden-state passthrough for folded dX adds."""

    @staticmethod
    def forward(ctx, hidden_states_flat, W_combined):
        M, D = hidden_states_flat.shape
        OUT_DIM = W_combined.shape[1]
        device = hidden_states_flat.device
        padded_out = max(16, 1 << (OUT_DIM - 1).bit_length())

        mean_x = torch.empty(M, device=device, dtype=torch.float32)
        mean_x2 = torch.empty(M, device=device, dtype=torch.float32)
        combined_raw = torch.empty(M, OUT_DIM, device=device, dtype=torch.float32)

        hs_contig = _mhc_require_bf16_activation(hidden_states_flat, "fused xHC full projection input")
        W_bf16 = _mhc_require_bf16_weight(W_combined, "fused xHC full projection weight").contiguous()
        _run_bf16_full_proj_fwd(
            hs_contig, W_bf16, combined_raw, mean_x, mean_x2,
            M, D, OUT_DIM, padded_out,
        )

        ctx.save_for_backward(hidden_states_flat, W_bf16)
        return combined_raw, mean_x.unsqueeze(-1), mean_x2.unsqueeze(-1), hidden_states_flat

    @staticmethod
    def backward(ctx, grad_combined, grad_mean_x, grad_mean_x2, grad_passthrough):
        hs_flat, W_for_bwd = ctx.saved_tensors
        if _use_split_proj_bwd(hs_flat.shape[1], W_for_bwd.shape[1], W_for_bwd.dtype):
            return _proj_bwd_split(
                hs_flat,
                W_for_bwd,
                grad_combined,
                grad_mean_x,
                grad_mean_x2,
                grad_passthrough,
            )

        grad_hs, grad_W = _proj_bwd_merged(hs_flat, W_for_bwd, grad_combined, grad_mean_x, grad_mean_x2)
        if grad_passthrough is not None:
            grad_hs = grad_hs + grad_passthrough.reshape_as(grad_hs)
        return grad_hs, grad_W


class FusedActiveProjFunction(torch.autograd.Function):
    """Triton active_proj forward (RMS+matmul fused), PyTorch backward."""

    @staticmethod
    def forward(ctx, flat_active_bf16, proj_post_res):
        flat_active_bf16 = _mhc_require_bf16_activation(flat_active_bf16, 'fused xHC active projection input')
        M, D = flat_active_bf16.shape
        W_bf16 = _mhc_require_bf16_weight(proj_post_res, 'fused xHC active projection weight').contiguous()
        OUT_DIM = W_bf16.shape[1]
        padded_out = max(16, 1 << (OUT_DIM - 1).bit_length())
        projected = torch.empty(M, OUT_DIM, device=flat_active_bf16.device, dtype=torch.float32)
        mean_x2 = torch.empty(M, device=flat_active_bf16.device, dtype=torch.float32)
        if flat_active_bf16.dtype == torch.bfloat16 and OUT_DIM <= 64:
            _run_bf16_active_proj_fwd(flat_active_bf16, W_bf16, projected, mean_x2, M, D, OUT_DIM, padded_out, store_mean_x2=True, use_external_mean_x2=False)
        else:
            grid = lambda META: (triton.cdiv(M, META['BLOCK_T']),)
            xhc_active_proj_bf16_kernel[grid](flat_active_bf16, W_bf16, projected, mean_x2, M, D=D, OUT_DIM=OUT_DIM, stride_act=flat_active_bf16.stride(0), PADDED_OUT=padded_out, STORE_MEAN_X2=True)
        ctx.save_for_backward(flat_active_bf16, W_bf16, projected, mean_x2)
        return projected

    @staticmethod
    def backward(ctx, grad_projected):
        flat_active_bf16, W_bf16, projected, mean_x2 = ctx.saved_tensors
        if _use_inline_active_proj_bwd_rms():
            grad_active, grad_W = _active_proj_bwd_rms_split(flat_active_bf16, W_bf16, grad_projected, projected, mean_x2)
            return (grad_active, grad_W)
        mean_x2_2d = mean_x2.unsqueeze(-1)
        inv_rms = torch.rsqrt(mean_x2_2d).nan_to_num(1.0, 1.0, 1.0)
        grad_projected_fp32 = grad_projected.float()
        grad_raw = grad_projected_fp32 * inv_rms
        grad_mean_x2 = -0.5 * (grad_projected_fp32 * projected.float()).sum(dim=-1, keepdim=True) / mean_x2_2d.clamp_min(1e-30)
        grad_active, grad_W = _proj_bwd_torch_bf16(flat_active_bf16, W_bf16, grad_raw, None, grad_mean_x2)
        return (grad_active, grad_W)


class FusedActiveProjWithStatsFunction(torch.autograd.Function):
    """Active projection that reuses mean_x2 produced by active gather."""

    @staticmethod
    def forward(ctx, flat_active_bf16, proj_post_res, mean_x2):
        flat_active_bf16 = _mhc_require_bf16_activation(flat_active_bf16, 'fused xHC active projection input')
        M, D = flat_active_bf16.shape
        W_bf16 = _mhc_require_bf16_weight(proj_post_res, 'fused xHC active projection weight').contiguous()
        OUT_DIM = W_bf16.shape[1]
        padded_out = max(16, 1 << (OUT_DIM - 1).bit_length())
        mean_x2_flat = mean_x2.reshape(M).float()
        projected = torch.empty(M, OUT_DIM, device=flat_active_bf16.device, dtype=torch.float32)
        if flat_active_bf16.dtype == torch.bfloat16 and OUT_DIM <= 64:
            _run_bf16_active_proj_fwd(flat_active_bf16, W_bf16, projected, mean_x2_flat, M, D, OUT_DIM, padded_out, store_mean_x2=False, use_external_mean_x2=True)
        else:
            with _mhc_nvtx_range('mhc.active_proj.fwd.cublas'):
                raw = torch.mm(flat_active_bf16, W_bf16).float()
            inv_rms = torch.rsqrt(mean_x2_flat.unsqueeze(-1)).nan_to_num(1.0, 1.0, 1.0)
            projected = raw * inv_rms
        ctx.save_for_backward(flat_active_bf16, W_bf16, projected, mean_x2_flat)
        return projected

    @staticmethod
    def backward(ctx, grad_projected):
        flat_active_bf16, W_bf16, projected, mean_x2 = ctx.saved_tensors
        if _use_inline_active_proj_bwd_rms():
            grad_active, grad_W = _active_proj_bwd_rms_split(flat_active_bf16, W_bf16, grad_projected, projected, mean_x2)
            return (grad_active, grad_W, None)
        mean_x2_2d = mean_x2.unsqueeze(-1)
        inv_rms = torch.rsqrt(mean_x2_2d).nan_to_num(1.0, 1.0, 1.0)
        grad_projected_fp32 = grad_projected.float()
        grad_raw = grad_projected_fp32 * inv_rms
        grad_mean_x2 = -0.5 * (grad_projected_fp32 * projected.float()).sum(dim=-1, keepdim=True) / mean_x2_2d.clamp_min(1e-30)
        grad_active, grad_W = _proj_bwd_torch_bf16(flat_active_bf16, W_bf16, grad_raw, None, grad_mean_x2)
        return (grad_active, grad_W, None)


class FusedActiveProjRecomputeFunction(torch.autograd.Function):
    """Active projection that recomputes active gather in backward instead of saving it."""

    @staticmethod
    def forward(ctx, flat_active_bf16, hidden_states, topk_indices, proj_post_res):
        flat_active_bf16 = _mhc_require_bf16_activation(flat_active_bf16, 'fused xHC active projection input')
        M, D = flat_active_bf16.shape
        W_bf16 = _mhc_require_bf16_weight(proj_post_res, 'fused xHC active projection weight').contiguous()
        OUT_DIM = W_bf16.shape[1]
        padded_out = max(16, 1 << (OUT_DIM - 1).bit_length())
        projected = torch.empty(M, OUT_DIM, device=flat_active_bf16.device, dtype=torch.float32)
        mean_x2 = torch.empty(M, device=flat_active_bf16.device, dtype=torch.float32)
        if flat_active_bf16.dtype == torch.bfloat16 and OUT_DIM <= 64:
            _run_bf16_active_proj_fwd(flat_active_bf16, W_bf16, projected, mean_x2, M, D, OUT_DIM, padded_out, store_mean_x2=True, use_external_mean_x2=False)
        else:
            grid = lambda META: (triton.cdiv(M, META['BLOCK_T']),)
            xhc_active_proj_bf16_kernel[grid](flat_active_bf16, W_bf16, projected, mean_x2, M, D=D, OUT_DIM=OUT_DIM, stride_act=flat_active_bf16.stride(0), PADDED_OUT=padded_out, STORE_MEAN_X2=True)
        ctx.save_for_backward(hidden_states, topk_indices, W_bf16, projected, mean_x2)
        return projected

    @staticmethod
    def backward(ctx, grad_projected):
        hidden_states, topk_indices, W_bf16, projected, mean_x2 = ctx.saved_tensors
        flat_active_bf16 = _recompute_flat_active(hidden_states, topk_indices)
        if _use_inline_active_proj_bwd_rms():
            grad_active, grad_W = _active_proj_bwd_rms_split(flat_active_bf16, W_bf16, grad_projected, projected, mean_x2)
        else:
            mean_x2_2d = mean_x2.unsqueeze(-1)
            inv_rms = torch.rsqrt(mean_x2_2d).nan_to_num(1.0, 1.0, 1.0)
            grad_projected_fp32 = grad_projected.float()
            grad_raw = grad_projected_fp32 * inv_rms
            grad_mean_x2 = -0.5 * (grad_projected_fp32 * projected.float()).sum(dim=-1, keepdim=True) / mean_x2_2d.clamp_min(1e-30)
            grad_active, grad_W = _proj_bwd_torch_bf16(flat_active_bf16, W_bf16, grad_raw, None, grad_mean_x2)
        grad_hidden = _scatter_flat_active_grad(grad_active, topk_indices, hidden_states.shape, hidden_states.dtype)
        return (None, grad_hidden, None, grad_W)


class FusedActiveProjWithStatsRecomputeFunction(torch.autograd.Function):
    """Active projection with reused stats and recomputed active gather in backward."""

    @staticmethod
    def forward(ctx, flat_active_bf16, hidden_states, topk_indices, proj_post_res, mean_x2):
        flat_active_bf16 = _mhc_require_bf16_activation(flat_active_bf16, 'fused xHC active projection input')
        M, D = flat_active_bf16.shape
        W_bf16 = _mhc_require_bf16_weight(proj_post_res, 'fused xHC active projection weight').contiguous()
        OUT_DIM = W_bf16.shape[1]
        padded_out = max(16, 1 << (OUT_DIM - 1).bit_length())
        mean_x2_flat = mean_x2.reshape(M).float()
        projected = torch.empty(M, OUT_DIM, device=flat_active_bf16.device, dtype=torch.float32)
        if flat_active_bf16.dtype == torch.bfloat16 and OUT_DIM <= 64:
            _run_bf16_active_proj_fwd(flat_active_bf16, W_bf16, projected, mean_x2_flat, M, D, OUT_DIM, padded_out, store_mean_x2=False, use_external_mean_x2=True)
        else:
            with _mhc_nvtx_range('mhc.active_proj.fwd.cublas'):
                raw = torch.mm(flat_active_bf16, W_bf16).float()
            inv_rms = torch.rsqrt(mean_x2_flat.unsqueeze(-1)).nan_to_num(1.0, 1.0, 1.0)
            projected = raw * inv_rms
        ctx.save_for_backward(hidden_states, topk_indices, W_bf16, projected, mean_x2_flat)
        return projected

    @staticmethod
    def backward(ctx, grad_projected):
        hidden_states, topk_indices, W_bf16, projected, mean_x2 = ctx.saved_tensors
        flat_active_bf16 = _recompute_flat_active(hidden_states, topk_indices)
        if _use_inline_active_proj_bwd_rms():
            grad_active, grad_W = _active_proj_bwd_rms_split(flat_active_bf16, W_bf16, grad_projected, projected, mean_x2)
        else:
            mean_x2_2d = mean_x2.unsqueeze(-1)
            inv_rms = torch.rsqrt(mean_x2_2d).nan_to_num(1.0, 1.0, 1.0)
            grad_projected_fp32 = grad_projected.float()
            grad_raw = grad_projected_fp32 * inv_rms
            grad_mean_x2 = -0.5 * (grad_projected_fp32 * projected.float()).sum(dim=-1, keepdim=True) / mean_x2_2d.clamp_min(1e-30)
            grad_active, grad_W = _proj_bwd_torch_bf16(flat_active_bf16, W_bf16, grad_raw, None, grad_mean_x2)
        grad_hidden = _scatter_flat_active_grad(grad_active, topk_indices, hidden_states.shape, hidden_states.dtype)
        return (None, grad_hidden, None, grad_W, None)


class FusedActiveGatherPassthroughFunction(torch.autograd.Function):
    """Active gather with a dense passthrough output for fused gradient accumulation."""

    @staticmethod
    def forward(ctx, hidden_states, topk_indices):
        S, B, N, C = hidden_states.shape
        K = topk_indices.shape[2]
        M = S * B
        hs_flat = hidden_states.reshape(M, N, C).contiguous()
        topk_flat = topk_indices.reshape(M, K).contiguous()
        active = torch.empty(M, K, C, device=hidden_states.device, dtype=hidden_states.dtype)
        grid = lambda META: (M, K, triton.cdiv(C, META['BLOCK_C']))
        xhc_active_gather_kernel[grid](hs_flat, topk_flat, active, M, N=N, K=K, C=C)
        ctx.save_for_backward(topk_flat)
        ctx.shape_info = (S, B, N, C, K)
        ctx.origin_dtype = hidden_states.dtype
        return (active.reshape(S, B, K, C), hidden_states)

    @staticmethod
    def backward(ctx, grad_active, grad_passthrough):
        topk_flat, = ctx.saved_tensors
        S, B, N, C, K = ctx.shape_info
        M = S * B
        if grad_active is None:
            return (grad_passthrough, None)
        if not (HAS_TRITON and grad_active.is_cuda):
            idx_exp = topk_flat.reshape(S, B, K).unsqueeze(-1).expand(-1, -1, -1, C)
            base = torch.zeros(S, B, N, C, device=grad_active.device, dtype=ctx.origin_dtype) if grad_passthrough is None else grad_passthrough
            return (base.scatter_add(2, idx_exp, grad_active.to(ctx.origin_dtype)), None)
        grad_active_flat = grad_active.reshape(M, K, C).contiguous()
        has_grad_passthrough = grad_passthrough is not None
        store_dtype = 1 if ctx.origin_dtype == torch.bfloat16 else 2 if ctx.origin_dtype == torch.float16 else 0
        if has_grad_passthrough:
            grad_passthrough_flat = grad_passthrough.reshape(M, N, C).contiguous()
            if K == 4:
                grid = lambda META: (M, triton.cdiv(C, META['BLOCK_C']))
                xhc_active_gather_bwd_passthrough_inplace_k4_kernel[grid](grad_active_flat, topk_flat, grad_passthrough_flat, M, N=N, K=K, C=C, STORE_DTYPE=store_dtype, BLOCK_C=512, num_warps=1, num_stages=2)
            else:
                grid = lambda META: (M, K, triton.cdiv(C, META['BLOCK_C']))
                xhc_active_gather_bwd_passthrough_inplace_kernel[grid](grad_active_flat, topk_flat, grad_passthrough_flat, M, N=N, K=K, C=C, STORE_DTYPE=store_dtype, BLOCK_C=2048, num_warps=1, num_stages=2)
            return (grad_passthrough_flat.reshape(S, B, N, C), None)
        if has_grad_passthrough:
            grad_passthrough_flat = grad_passthrough.reshape(M, N, C).contiguous()
        else:
            grad_passthrough_flat = grad_active_flat
        grad_hidden = torch.empty(M, N, C, device=grad_active.device, dtype=ctx.origin_dtype)
        grid = lambda META: (M, N, triton.cdiv(C, META['BLOCK_C']))
        xhc_active_gather_bwd_passthrough_kernel[grid](grad_active_flat, topk_flat, grad_passthrough_flat, grad_hidden, M, N=N, K=K, C=C, HAS_GRAD_PASSTHROUGH=has_grad_passthrough, STORE_DTYPE=store_dtype)
        return (grad_hidden.reshape(S, B, N, C), None)


class FusedActiveGatherStatsPassthroughFunction(torch.autograd.Function):
    """Active gather that also produces mean_x2 for the active projection RMS."""

    @staticmethod
    def forward(ctx, hidden_states, topk_indices):
        S, B, N, C = hidden_states.shape
        K = topk_indices.shape[2]
        M = S * B
        hs_flat = hidden_states.reshape(M, N, C).contiguous()
        topk_flat = topk_indices.reshape(M, K).contiguous()
        active = torch.empty(M, K, C, device=hidden_states.device, dtype=hidden_states.dtype)
        mean_x2 = torch.empty(M, 1, device=hidden_states.device, dtype=torch.float32)
        if K == 4:
            block_c = 4096
            num_c_tiles = triton.cdiv(C, block_c)
            stats = mean_x2 if num_c_tiles == 1 else torch.empty(M, num_c_tiles, device=hidden_states.device, dtype=torch.float32)
            xhc_active_gather_k4_stats_kernel.fn[M, num_c_tiles](hs_flat, topk_flat, active, stats, M, N=N, C=C, NUM_C_TILES=num_c_tiles, DIRECT_MEAN=num_c_tiles == 1, BLOCK_C=block_c, num_warps=4, num_stages=2)
            if num_c_tiles != 1:
                block_ct = triton.next_power_of_2(num_c_tiles)
                xhc_active_gather_stats_reduce_kernel[M,](stats, mean_x2, M, K=K, C=C, NUM_C_TILES=num_c_tiles, BLOCK_CT=block_ct, num_warps=1)
        else:
            grid = lambda META: (M, K, triton.cdiv(C, META['BLOCK_C']))
            xhc_active_gather_kernel[grid](hs_flat, topk_flat, active, M, N=N, K=K, C=C)
            mean_x2 = active.reshape(M, K * C).float().square().mean(dim=-1, keepdim=True)
        ctx.save_for_backward(topk_flat)
        ctx.shape_info = (S, B, N, C, K)
        ctx.origin_dtype = hidden_states.dtype
        return (active.reshape(S, B, K, C), mean_x2, hidden_states)

    @staticmethod
    def backward(ctx, grad_active, _grad_mean_x2, grad_passthrough):
        topk_flat, = ctx.saved_tensors
        S, B, N, C, K = ctx.shape_info
        M = S * B
        if grad_active is None:
            return (grad_passthrough, None)
        if not (HAS_TRITON and grad_active.is_cuda):
            idx_exp = topk_flat.reshape(S, B, K).unsqueeze(-1).expand(-1, -1, -1, C)
            base = torch.zeros(S, B, N, C, device=grad_active.device, dtype=ctx.origin_dtype) if grad_passthrough is None else grad_passthrough
            return (base.scatter_add(2, idx_exp, grad_active.to(ctx.origin_dtype)), None)
        grad_active_flat = grad_active.reshape(M, K, C).contiguous()
        has_grad_passthrough = grad_passthrough is not None
        store_dtype = 1 if ctx.origin_dtype == torch.bfloat16 else 2 if ctx.origin_dtype == torch.float16 else 0
        if has_grad_passthrough:
            grad_passthrough_flat = grad_passthrough.reshape(M, N, C).contiguous()
            if K == 4:
                grid = lambda META: (M, triton.cdiv(C, META['BLOCK_C']))
                xhc_active_gather_bwd_passthrough_inplace_k4_kernel[grid](grad_active_flat, topk_flat, grad_passthrough_flat, M, N=N, K=K, C=C, STORE_DTYPE=store_dtype, BLOCK_C=512, num_warps=1, num_stages=2)
            else:
                grid = lambda META: (M, K, triton.cdiv(C, META['BLOCK_C']))
                xhc_active_gather_bwd_passthrough_inplace_kernel[grid](grad_active_flat, topk_flat, grad_passthrough_flat, M, N=N, K=K, C=C, STORE_DTYPE=store_dtype, BLOCK_C=2048, num_warps=1, num_stages=2)
            return (grad_passthrough_flat.reshape(S, B, N, C), None)
        if has_grad_passthrough:
            grad_passthrough_flat = grad_passthrough.reshape(M, N, C).contiguous()
        else:
            grad_passthrough_flat = grad_active_flat
        grad_hidden = torch.empty(M, N, C, device=grad_active.device, dtype=ctx.origin_dtype)
        grid = lambda META: (M, N, triton.cdiv(C, META['BLOCK_C']))
        xhc_active_gather_bwd_passthrough_kernel[grid](grad_active_flat, topk_flat, grad_passthrough_flat, grad_hidden, M, N=N, K=K, C=C, HAS_GRAD_PASSTHROUGH=has_grad_passthrough, STORE_DTYPE=store_dtype)
        return (grad_hidden.reshape(S, B, N, C), None)


class FusedRoutedPostStatsRouterTop2Function(torch.autograd.Function):
    """Fused fused xHC post-stats + fixed-2/routed-top2 router for compact/full router layouts."""

    @staticmethod
    def forward(
        ctx,
        combined_raw: torch.Tensor,
        mean_x: torch.Tensor,
        mean_x2: torch.Tensor,
        colsum_Wr_routed: torch.Tensor,
        alpha_pre: torch.Tensor,
        bias_pre: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        S: int,
        B: int,
        N: int,
        num_fixed: int,
        router_cols: int,
        router_offset: int,
        use_bias: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M = S * B
        routed_n = N - num_fixed
        router_cols = int(router_cols)
        router_offset = int(router_offset)
        if router_offset < 0 or router_offset + routed_n > router_cols:
            raise ValueError(
                f"invalid fused xHC router layout: router_cols={router_cols}, "
                f"router_offset={router_offset}, routed_n={routed_n}."
            )
        combined_flat = combined_raw.reshape(M, router_cols + N).contiguous()
        mean_x_flat = mean_x.reshape(M).contiguous()
        mean_x2_flat = mean_x2.reshape(M).contiguous()
        colsum = colsum_Wr_routed.contiguous()
        h_pre = torch.empty(M, N, device=combined_raw.device, dtype=torch.float32)
        topk = torch.empty(M, 4, device=combined_raw.device, dtype=torch.long)
        router_probs = torch.empty(M, 2, device=combined_raw.device, dtype=torch.float32)
        bias_arg = routing_bias if routing_bias is not None else bias_pre

        xhc_post_stats_router_top2_fwd_routed_kernel[(M,)](
            combined_flat,
            mean_x_flat,
            mean_x2_flat,
            colsum,
            alpha_pre.contiguous(),
            bias_pre.contiguous(),
            bias_arg,
            h_pre,
            topk,
            router_probs,
            M,
            N=N,
            NUM_FIXED=num_fixed,
            ROUTER_COLS=router_cols,
            ROUTER_OFFSET=router_offset,
            ROUTED_N=routed_n,
            PADDED_ROUTED_N=triton.next_power_of_2(routed_n),
            USE_BIAS=bool(use_bias),
            num_warps=1,
        )

        ctx.save_for_backward(
            combined_flat,
            mean_x_flat,
            mean_x2_flat,
            colsum,
            alpha_pre.contiguous(),
            h_pre,
            topk,
            router_probs,
        )
        ctx.input_shapes = (
            combined_raw.shape,
            mean_x.shape,
            mean_x2.shape,
            colsum_Wr_routed.shape,
            alpha_pre.shape,
            bias_pre.shape,
        )
        ctx.shape_info = (S, B, N, num_fixed, router_cols, router_offset, routed_n)
        return h_pre.reshape(S, B, N), topk.reshape(S, B, 4), router_probs.reshape(S, B, 2)

    @staticmethod
    def backward(ctx, grad_h_pre, grad_topk, grad_router_probs):
        del grad_topk
        combined_flat, mean_x_flat, mean_x2_flat, colsum, alpha, h_pre, topk, router_probs = ctx.saved_tensors
        combined_shape, mean_shape, mean2_shape, colsum_shape, alpha_shape, bias_shape = ctx.input_shapes
        S, B, N, num_fixed, router_cols, router_offset, routed_n = ctx.shape_info
        M = S * B

        if grad_h_pre is None:
            grad_h_flat = torch.zeros(M, N, device=combined_flat.device, dtype=torch.float32)
        else:
            grad_h_flat = grad_h_pre.reshape(M, N).contiguous().float()
        if grad_router_probs is None:
            grad_probs_flat = torch.zeros(M, 2, device=combined_flat.device, dtype=torch.float32)
        else:
            grad_probs_flat = grad_router_probs.reshape(M, 2).contiguous().float()

        if router_cols == routed_n and router_offset == 0:
            grad_combined = torch.empty_like(combined_flat)
        else:
            grad_combined = torch.zeros_like(combined_flat)
        grad_mean = torch.empty(M, device=combined_flat.device, dtype=torch.float32)
        grad_mean2 = torch.empty(M, device=combined_flat.device, dtype=torch.float32)
        partial = torch.empty(N + routed_n + 1, M, device=combined_flat.device, dtype=torch.float32)

        xhc_post_stats_router_top2_bwd_stage1_routed_kernel[(M,)](
            combined_flat,
            mean_x_flat,
            mean_x2_flat,
            colsum,
            alpha,
            h_pre,
            topk,
            router_probs,
            grad_h_flat,
            grad_probs_flat,
            grad_combined,
            grad_mean,
            grad_mean2,
            partial,
            M,
            N=N,
            NUM_FIXED=num_fixed,
            ROUTER_COLS=router_cols,
            ROUTER_OFFSET=router_offset,
            ROUTED_N=routed_n,
            PADDED_ROUTED_N=triton.next_power_of_2(routed_n),
            num_warps=1,
        )

        grad_bias = torch.empty(bias_shape, device=combined_flat.device, dtype=torch.float32)
        grad_colsum = torch.empty_like(colsum)
        grad_alpha = torch.empty_like(alpha)
        block_m = triton.next_power_of_2(M)
        xhc_post_stats_router_reduce_routed_kernel[(N + routed_n + 1,)](
            partial,
            grad_bias,
            grad_colsum,
            grad_alpha,
            M,
            N=N,
            ROUTED_N=routed_n,
            BLOCK_M=block_m,
            num_warps=8,
        )

        return (
            grad_combined.reshape(combined_shape),
            grad_mean.reshape(mean_shape),
            grad_mean2.reshape(mean2_shape),
            grad_colsum.reshape(colsum_shape),
            grad_alpha.reshape(alpha_shape),
            grad_bias.reshape(bias_shape),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FusedPreResFunction(torch.autograd.Function):
    """Triton pre_res forward + PyTorch backward."""

    @staticmethod
    def forward(ctx, hidden_states, h_pre, h_res, topk_indices, N, K):
        M = hidden_states.shape[0] * hidden_states.shape[1]
        S, B = hidden_states.shape[:2]
        C = hidden_states.shape[-1]
        origin_dtype = hidden_states.dtype

        hs_flat = hidden_states.reshape(M, N, C)
        h_pre_flat = h_pre.reshape(M, N)
        h_res_flat = h_res.reshape(M, K, K)
        topk_flat = topk_indices.reshape(M, K)

        store_dtype = 1 if origin_dtype == torch.bfloat16 else (2 if origin_dtype == torch.float16 else 0)
        hidden_for_sub = torch.empty(M, C, device=hidden_states.device, dtype=origin_dtype)
        mixed_active = torch.empty(M, K, C, device=hidden_states.device, dtype=origin_dtype)

        grid_pr = lambda META: (M, triton.cdiv(C, META["BLOCK_C"]))
        xhc_pre_res_kernel[grid_pr](
            hs_flat, h_pre_flat, h_res_flat, topk_flat,
            hidden_for_sub, mixed_active,
            M, N=N, K=K, C=C,
            stride_hs_m=hs_flat.stride(0), stride_hs_n=hs_flat.stride(1),
            STORE_DTYPE=store_dtype,
        )

        ctx.save_for_backward(hidden_states, h_pre, h_res, topk_indices)
        ctx.N = N
        ctx.K = K
        return hidden_for_sub.reshape(S, B, C), mixed_active.reshape(S, B, K, C), hidden_states

    @staticmethod
    def backward(ctx, grad_hidden_for_sub, grad_mixed_active, grad_residual_passthrough):
        hidden_states, h_pre, h_res, topk_indices = ctx.saved_tensors
        S, B, N, C = hidden_states.shape
        K = ctx.K
        M = S * B
        origin_dtype = hidden_states.dtype

        if HAS_TRITON and hidden_states.is_cuda:
            hs_flat = hidden_states.reshape(M, N, C).contiguous()
            h_pre_flat = h_pre.reshape(M, N).contiguous()
            h_res_flat = h_res.reshape(M, K, K).contiguous()
            topk_flat = topk_indices.reshape(M, K).contiguous()
            grad_sub_flat = grad_hidden_for_sub.reshape(M, C).contiguous()
            grad_mixed_flat = grad_mixed_active.reshape(M, K, C).contiguous()
            has_grad_residual = grad_residual_passthrough is not None
            if has_grad_residual:
                grad_residual_flat = grad_residual_passthrough.reshape(M, N, C).contiguous()
            else:
                grad_residual_flat = grad_mixed_flat

            store_dtype = 1 if origin_dtype == torch.bfloat16 else (2 if origin_dtype == torch.float16 else 0)
            grad_hs = torch.empty(M, N, C, device=hidden_states.device, dtype=origin_dtype)
            grad_h_pre = torch.empty(M, N, device=hidden_states.device, dtype=torch.float32)
            grad_h_res = torch.empty(M, K, K, device=hidden_states.device, dtype=torch.float32)

            # Phase B: 2D grid (M, NUM_C_TILES) + reduce — eliminates 1D serial C loop
            BWD_BLOCK_C = 256
            num_c_tiles = triton.cdiv(C, BWD_BLOCK_C)
            dev = hidden_states.device
            needed_ghp = M * num_c_tiles * N
            needed_ghr = M * num_c_tiles * K * K
            if not hasattr(FusedPreResFunction, '_scratch_ghp') or FusedPreResFunction._scratch_ghp.numel() < needed_ghp or FusedPreResFunction._scratch_ghp.device != dev:
                FusedPreResFunction._scratch_ghp = torch.empty(needed_ghp, device=dev, dtype=torch.float32)
            if not hasattr(FusedPreResFunction, '_scratch_ghr') or FusedPreResFunction._scratch_ghr.numel() < needed_ghr or FusedPreResFunction._scratch_ghr.device != dev:
                FusedPreResFunction._scratch_ghr = torch.empty(needed_ghr, device=dev, dtype=torch.float32)
            scratch_ghp = FusedPreResFunction._scratch_ghp[:needed_ghp]
            scratch_ghr = FusedPreResFunction._scratch_ghr[:needed_ghr]

            xhc_pre_res_bwd_kernel[(M, num_c_tiles)](
                hs_flat, h_pre_flat, h_res_flat, topk_flat,
                grad_sub_flat, grad_mixed_flat, grad_residual_flat,
                grad_hs, scratch_ghp, scratch_ghr,
                M, N=N, K=K, C=C,
                stride_hs_m=hs_flat.stride(0), stride_hs_n=hs_flat.stride(1),
                NUM_C_TILES=num_c_tiles,
                HAS_GRAD_RESIDUAL=has_grad_residual,
                STORE_DTYPE=store_dtype,
                BLOCK_C=BWD_BLOCK_C,
                num_warps=1, num_stages=2,
            )

            xhc_pre_res_bwd_reduce_kernel[(M,)](
                scratch_ghp, scratch_ghr,
                grad_h_pre, grad_h_res,
                M, N=N, K=K, NUM_C_TILES=num_c_tiles,
                num_warps=1,
            )

            grad_hs = grad_hs.reshape(S, B, N, C)
            grad_h_pre = grad_h_pre.reshape(S, B, N)
            grad_h_res = grad_h_res.reshape(S, B, K, K)
        else:
            tgt = torch.float32
            grad_hs = torch.einsum(
                "sbn,sbc->sbnc", h_pre.to(origin_dtype), grad_hidden_for_sub,
            )
            idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, C)
            grad_active_latent = torch.matmul(
                h_res.transpose(-2, -1).to(tgt), grad_mixed_active.to(tgt)
            ).to(origin_dtype)
            grad_hs_from_res = torch.zeros_like(hidden_states)
            grad_hs_from_res.scatter_add_(2, idx_exp, grad_active_latent)
            grad_hs = grad_hs + grad_hs_from_res
            if grad_residual_passthrough is not None:
                grad_hs = grad_hs + grad_residual_passthrough
            grad_h_pre = torch.einsum(
                "sbc,sbnc->sbn", grad_hidden_for_sub.to(tgt), hidden_states.to(tgt)
            )
            active_latent = torch.gather(hidden_states, 2, idx_exp)
            grad_h_res = torch.matmul(
                grad_mixed_active.to(tgt), active_latent.to(tgt).transpose(-2, -1)
            )

        return grad_hs, grad_h_pre, grad_h_res, None, None, None


class FusedPreResActiveFunction(torch.autograd.Function):
    """pre_res variant that reuses the active gather tensor for h_res mixing."""

    @staticmethod
    def forward(ctx, hidden_states, h_pre, h_res, active, topk_indices, N, K):
        M = hidden_states.shape[0] * hidden_states.shape[1]
        S, B = hidden_states.shape[:2]
        C = hidden_states.shape[-1]
        origin_dtype = hidden_states.dtype
        hs_flat = hidden_states.reshape(M, N, C)
        h_pre_flat = h_pre.reshape(M, N)
        h_res_flat = h_res.reshape(M, K, K)
        active_flat = active.reshape(M, K, C)
        store_dtype = 1 if origin_dtype == torch.bfloat16 else 2 if origin_dtype == torch.float16 else 0
        hidden_for_sub = torch.empty(M, C, device=hidden_states.device, dtype=origin_dtype)
        mixed_active = torch.empty(M, K, C, device=hidden_states.device, dtype=origin_dtype)
        grid_pr = lambda META: (M, triton.cdiv(C, META['BLOCK_C']))
        topk_flat = topk_indices.reshape(M, K).contiguous()
        xhc_pre_res_active_kernel[grid_pr](hs_flat, active_flat, topk_flat, h_pre_flat, h_res_flat, hidden_for_sub, mixed_active, M, N=N, K=K, C=C, stride_hs_m=hs_flat.stride(0), stride_hs_n=hs_flat.stride(1), STORE_DTYPE=store_dtype, BLOCK_C=512, num_warps=1, num_stages=2)
        ctx.recompute_active_saved = False
        if ctx.recompute_active_saved:
            ctx.save_for_backward(hidden_states, h_pre, h_res, topk_indices)
        else:
            ctx.save_for_backward(hidden_states, h_pre, h_res, active, topk_indices)
        ctx.N = N
        ctx.K = K
        return (hidden_for_sub.reshape(S, B, C), mixed_active.reshape(S, B, K, C), hidden_states)

    @staticmethod
    def backward(ctx, grad_hidden_for_sub, grad_mixed_active, grad_residual_passthrough):
        if ctx.recompute_active_saved:
            hidden_states, h_pre, h_res, topk_indices = ctx.saved_tensors
            active = None
        else:
            hidden_states, h_pre, h_res, active, topk_indices = ctx.saved_tensors
        S, B, N, C = hidden_states.shape
        K = ctx.K
        M = S * B
        origin_dtype = hidden_states.dtype
        if HAS_TRITON and hidden_states.is_cuda:
            hs_flat = hidden_states.reshape(M, N, C).contiguous()
            h_pre_flat = h_pre.reshape(M, N).contiguous()
            h_res_flat = h_res.reshape(M, K, K).contiguous()
            topk_flat = topk_indices.reshape(M, K).contiguous()
            grad_sub_flat = grad_hidden_for_sub.reshape(M, C).contiguous()
            grad_mixed_flat = grad_mixed_active.reshape(M, K, C).contiguous()
            has_grad_residual = grad_residual_passthrough is not None
            if has_grad_residual:
                grad_residual_flat = grad_residual_passthrough.reshape(M, N, C).contiguous()
            else:
                grad_residual_flat = grad_mixed_flat
            store_dtype = 1 if origin_dtype == torch.bfloat16 else 2 if origin_dtype == torch.float16 else 0
            grad_hs = torch.empty(M, N, C, device=hidden_states.device, dtype=origin_dtype)
            grad_h_pre = torch.empty(M, N, device=hidden_states.device, dtype=torch.float32)
            grad_h_res = torch.empty(M, K, K, device=hidden_states.device, dtype=torch.float32)
            use_split_pre_res_bwd = False
            bwd_block_c = _split_pre_res_bwd_block_c() if use_split_pre_res_bwd else 256
            num_c_tiles = triton.cdiv(C, bwd_block_c)
            dev = hidden_states.device
            needed_ghp = M * num_c_tiles * N
            needed_ghr = M * num_c_tiles * K * K
            if not hasattr(FusedPreResActiveFunction, '_scratch_ghp') or FusedPreResActiveFunction._scratch_ghp.numel() < needed_ghp or FusedPreResActiveFunction._scratch_ghp.device != dev:
                FusedPreResActiveFunction._scratch_ghp = torch.empty(needed_ghp, device=dev, dtype=torch.float32)
            if not hasattr(FusedPreResActiveFunction, '_scratch_ghr') or FusedPreResActiveFunction._scratch_ghr.numel() < needed_ghr or FusedPreResActiveFunction._scratch_ghr.device != dev:
                FusedPreResActiveFunction._scratch_ghr = torch.empty(needed_ghr, device=dev, dtype=torch.float32)
            scratch_ghp = FusedPreResActiveFunction._scratch_ghp[:needed_ghp]
            scratch_ghr = FusedPreResActiveFunction._scratch_ghr[:needed_ghr]
            if use_split_pre_res_bwd:
                active_flat = active.reshape(M, K, C).contiguous()
                xhc_pre_res_bwd_grad_hs_kernel[M, num_c_tiles](h_pre_flat, h_res_flat, topk_flat, grad_sub_flat, grad_mixed_flat, grad_residual_flat, grad_hs, M, N=N, K=K, C=C, HAS_GRAD_RESIDUAL=has_grad_residual, STORE_DTYPE=store_dtype, BLOCK_C=bwd_block_c, num_warps=1, num_stages=2)
                xhc_pre_res_bwd_ghp_kernel[M, num_c_tiles](hs_flat, grad_sub_flat, scratch_ghp, M, N=N, C=C, stride_hs_m=hs_flat.stride(0), stride_hs_n=hs_flat.stride(1), NUM_C_TILES=num_c_tiles, BLOCK_C=bwd_block_c, num_warps=1, num_stages=2)
                xhc_pre_res_bwd_ghr_active_kernel[M, num_c_tiles](active_flat, grad_mixed_flat, scratch_ghr, M, K=K, C=C, NUM_C_TILES=num_c_tiles, BLOCK_C=bwd_block_c, num_warps=1, num_stages=2)
            else:
                xhc_pre_res_bwd_kernel[M, num_c_tiles](hs_flat, h_pre_flat, h_res_flat, topk_flat, grad_sub_flat, grad_mixed_flat, grad_residual_flat, grad_hs, scratch_ghp, scratch_ghr, M, N=N, K=K, C=C, stride_hs_m=hs_flat.stride(0), stride_hs_n=hs_flat.stride(1), NUM_C_TILES=num_c_tiles, HAS_GRAD_RESIDUAL=has_grad_residual, STORE_DTYPE=store_dtype, BLOCK_C=bwd_block_c, num_warps=1, num_stages=2)
            xhc_pre_res_bwd_reduce_kernel[M,](scratch_ghp, scratch_ghr, grad_h_pre, grad_h_res, M, N=N, K=K, NUM_C_TILES=num_c_tiles, num_warps=1)
            grad_hs = grad_hs.reshape(S, B, N, C)
            grad_active_out = None
            return (grad_hs, grad_h_pre.reshape(S, B, N), grad_h_res.reshape(S, B, K, K), grad_active_out, None, None, None)
        tgt = torch.float32
        grad_hs = grad_hidden_for_sub.unsqueeze(2).to(tgt) * h_pre.unsqueeze(-1).to(tgt)
        if grad_residual_passthrough is not None:
            grad_hs = grad_hs + grad_residual_passthrough.to(tgt)
        if active is None:
            active = _recompute_flat_active(hidden_states, topk_indices).reshape(S, B, K, C)
        grad_active = torch.matmul(h_res.transpose(-2, -1).to(tgt), grad_mixed_active.to(tgt))
        grad_h_pre = (grad_hidden_for_sub.unsqueeze(2).to(tgt) * hidden_states.to(tgt)).sum(-1)
        grad_h_res = torch.matmul(grad_mixed_active.to(tgt), active.to(tgt).transpose(-2, -1))
        if ctx.recompute_active_saved:
            grad_hs = grad_hs + _scatter_flat_active_grad(grad_active, topk_indices, (S, B, N, C), grad_hs.dtype).to(tgt)
        return (grad_hs.to(origin_dtype), grad_h_pre, grad_h_res, None if ctx.recompute_active_saved else grad_active.to(origin_dtype), None, None, None)


class _TritonSinkhornFunc(torch.autograd.Function):
    """Triton Sinkhorn forward + backward (scratchpad pattern from Expert)."""

    @staticmethod
    def forward(ctx, logits, K, num_iters):
        M_flat = logits.numel() // (K * K)
        K_SQ = K * K
        device = logits.device
        num_iters = int(num_iters)
        if num_iters <= 0:
            raise ValueError(f"Sinkhorn iteration count must be positive, got {num_iters}.")

        h_res_flat = logits.reshape(M_flat, K_SQ).float()
        h_res_out = torch.empty_like(h_res_flat)

        xhc_sinkhorn_kernel[(M_flat,)](
            h_res_flat.reshape(M_flat, K, K), h_res_out.reshape(M_flat, K, K), M_flat,
            K=K, K_SQ=K_SQ, SINKHORN_ITERS=num_iters, CLAMP_ROW_SUM=1,
            num_warps=1,
        )

        ctx.save_for_backward(h_res_flat, h_res_out)  # save logits + output
        ctx.K = K
        ctx.M_flat = M_flat
        ctx.num_iters = num_iters
        ctx.original_shape = logits.shape
        return h_res_out.reshape(M_flat, K, K).reshape(logits.shape)

    @staticmethod
    def backward(ctx, grad_output):
        logits_flat, h_res_out = ctx.saved_tensors
        K = ctx.K
        M_flat = ctx.M_flat
        num_iters = ctx.num_iters
        K_SQ = K * K
        device = logits_flat.device

        grad_out_flat = grad_output.reshape(M_flat, K_SQ).contiguous().float()
        grad_logits = torch.empty(M_flat, K_SQ, device=device, dtype=torch.float32)

        SK_BUF_STRIDE = 2 * num_iters * K_SQ
        needed = M_flat * SK_BUF_STRIDE
        # Reuse scratchpad across calls to avoid repeated 10MB allocation
        if not hasattr(_TritonSinkhornFunc, '_sk_buf') or _TritonSinkhornFunc._sk_buf.numel() < needed or _TritonSinkhornFunc._sk_buf.device != device:
            _TritonSinkhornFunc._sk_buf = torch.empty(needed, device=device, dtype=torch.float32)
        sk_buf = _TritonSinkhornFunc._sk_buf[:needed]

        xhc_sinkhorn_bwd_kernel[(M_flat,)](
            logits_flat, h_res_out, grad_out_flat, grad_logits, sk_buf,
            M_flat, K=K, K_SQ=K_SQ, SINKHORN_ITERS=num_iters, CLAMP_ROW_SUM=1,
            SK_BUF_STRIDE=SK_BUF_STRIDE,
            num_warps=1,
        )

        return grad_logits.reshape(ctx.original_shape), None, None


def _triton_sinkhorn_with_grad(logits: Tensor, K: int, num_iters: int = 20) -> Tensor:
    return _TritonSinkhornFunc.apply(logits, K, int(num_iters))


class FusedPostFunction(torch.autograd.Function):
    """Triton post backward: scatter_bwd + addcmul_bwd + matmul/einsum_bwd fused."""

    @staticmethod
    def forward(ctx, comb_or_out, h_post, router_probs, mixed_active, topk_indices, residual, post_rank, opt_level):
        S, B, C = (residual.shape[0], residual.shape[1], residual.shape[-1])
        K = topk_indices.shape[2]
        N = residual.shape[2]
        origin_dtype = residual.dtype
        R = post_rank
        M = S * B
        if HAS_TRITON and residual.is_cuda:
            store_dtype = 1 if origin_dtype == torch.bfloat16 else 2 if origin_dtype == torch.float16 else 0
            if R > 1:
                comb_flat = comb_or_out.reshape(M, R, C)
                hp_flat = h_post.float().reshape(M, K * R)
            else:
                comb_flat = comb_or_out.reshape(M, 1, C)
                hp_flat = h_post.float().reshape(M, K)
            if router_probs.shape[-1] != K - 2:
                raise RuntimeError(f'fused xHC post expects only dynamic router_probs with last dim {K - 2}, got {router_probs.shape[-1]}.')
            rw_flat = router_probs.float().reshape(M, K - 2)
            mixed_flat = mixed_active.reshape(M, K, C)
            topk_flat = topk_indices.reshape(M, K)
            residual_flat = residual.reshape(M, N, C).contiguous()
            fuse_update_scatter = R == 1 and K == 4
            if fuse_update_scatter:
                out_flat = residual_flat.clone()
                grid_update = lambda META: (M, triton.cdiv(C, META['BLOCK_C']))
                xhc_post_fwd_update_r1_kernel[grid_update](comb_flat, hp_flat, rw_flat, mixed_flat, topk_flat, out_flat, M, N=N, C=C)
            else:
                updated_flat = torch.empty(M, K, C, device=residual.device, dtype=residual.dtype)
                grid = lambda META: (M, triton.cdiv(C, META['BLOCK_C']))
                xhc_post_fwd_slim_kernel[grid](comb_flat, hp_flat, rw_flat, mixed_flat, updated_flat, M, K=K, C=C, R=R)
            if not fuse_update_scatter:
                out_flat = residual_flat.clone()
                grid_update = lambda META: (M, K, triton.cdiv(C, META['BLOCK_C']))
                xhc_post_update_scatter_kernel[grid_update](updated_flat, topk_flat, out_flat, M, N=N, K=K, C=C)
            hidden_states = out_flat.reshape(S, B, N, C)
        else:
            compute_dtype = torch.bfloat16
            if R > 1:
                post_add = torch.matmul(h_post.to(compute_dtype), comb_or_out.to(compute_dtype))
            else:
                post_add = torch.einsum('sbn,sbc->sbnc', h_post.to(compute_dtype), comb_or_out.to(compute_dtype))
            if router_probs.shape[-1] != K - 2:
                raise RuntimeError(f'fused xHC post expects only dynamic router_probs with last dim {K - 2}, got {router_probs.shape[-1]}.')
            fixed = torch.ones(*router_probs.shape[:-1], K - router_probs.shape[-1], device=router_probs.device, dtype=router_probs.dtype)
            rw_full = torch.cat([fixed, router_probs], dim=-1)
            rw = rw_full.unsqueeze(-1).to(compute_dtype)
            updated = torch.addcmul(mixed_active.to(compute_dtype), post_add, rw).to(origin_dtype)
            idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, C)
            hidden_states = residual.scatter(2, idx_exp, updated)
        ctx.save_for_backward(comb_or_out, h_post, router_probs, topk_indices)
        ctx.R = R
        ctx.opt_level = int(opt_level)
        ctx.shape_info = (S, B, C, K, N)
        ctx.origin_dtype = origin_dtype
        return hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        comb_or_out, h_post, router_probs, topk_indices = ctx.saved_tensors
        S, B, C, K, N = ctx.shape_info
        R = ctx.R
        M = S * B
        origin_dtype = ctx.origin_dtype
        if HAS_TRITON and grad_output.is_cuda:
            store_dtype = 1 if origin_dtype == torch.bfloat16 else 2 if origin_dtype == torch.float16 else 0
            go_flat = grad_output.reshape(M, N, C).clone(memory_format=torch.contiguous_format)
            topk_flat = topk_indices.reshape(M, K).contiguous()
            if R > 1:
                comb_flat = comb_or_out.reshape(M, R, C).contiguous()
                hp_flat = h_post.float().reshape(M, K * R)
            else:
                comb_flat = comb_or_out.reshape(M, 1, C).contiguous()
                hp_flat = h_post.float().reshape(M, K)
            if router_probs.shape[-1] != K - 2:
                raise RuntimeError(f'fused xHC post expects only dynamic router_probs with last dim {K - 2}, got {router_probs.shape[-1]}.')
            rw_flat = router_probs.float().reshape(M, K - 2)
            fuse_gather_slim_r1 = R == 1 and K == 4
            grad_comb = torch.empty(M, R, C, device=grad_output.device, dtype=origin_dtype)
            grad_hp = torch.empty(M, K * R, device=grad_output.device, dtype=torch.float32)
            grad_rw = torch.empty(M, K - 2, device=grad_output.device, dtype=torch.float32)
            gu_flat = torch.empty(M, K, C, device=grad_output.device, dtype=origin_dtype)
            if fuse_gather_slim_r1:
                xhc_post_bwd_gather_slim_r1_kernel[M,](go_flat, topk_flat, comb_flat, hp_flat, rw_flat, grad_comb, grad_hp, grad_rw, gu_flat, M, N=N, C=C, STORE_DTYPE=store_dtype, BLOCK_C=1024, num_warps=1, num_stages=2)
                grad_residual = go_flat
            else:
                if K == 4:
                    grid_gather = lambda META: (M, triton.cdiv(C, META['BLOCK_C']))
                    xhc_post_bwd_gather_zero_inplace_k4_kernel[grid_gather](go_flat, topk_flat, gu_flat, M, N=N, K=K, C=C, STORE_DTYPE=store_dtype, BLOCK_C=512, num_warps=1, num_stages=2)
                else:
                    grid_gather = lambda META: (M, K, triton.cdiv(C, META['BLOCK_C']))
                    xhc_post_bwd_gather_zero_inplace_kernel[grid_gather](go_flat, topk_flat, gu_flat, M, N=N, K=K, C=C, STORE_DTYPE=store_dtype, BLOCK_C=2048, num_warps=1, num_stages=2)
                grad_residual = go_flat
                xhc_post_bwd_slim_kernel[M,](gu_flat, comb_flat, hp_flat, rw_flat, grad_comb, grad_hp, grad_rw, M, K=K, C=C, R=R, STORE_DTYPE=store_dtype)
            grad_comb_out = grad_comb.reshape(S, B, R, C) if R > 1 else grad_comb.reshape(S, B, C)
            grad_hp_out = grad_hp.reshape(S, B, K, R) if R > 1 else grad_hp.reshape(S, B, K)
            grad_rw_out = grad_rw.reshape(S, B, K - 2)
            grad_mixed_out = gu_flat.reshape(S, B, K, C)
            grad_residual_out = grad_residual.reshape(S, B, N, C)
        else:
            tgt = torch.float32
            idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, C)
            grad_updated = torch.gather(grad_output, 2, idx_exp)
            grad_residual_out = grad_output.clone()
            grad_residual_out.scatter_(2, idx_exp, torch.zeros_like(grad_updated))
            grad_mixed_out = grad_updated
            if router_probs.shape[-1] != K - 2:
                raise RuntimeError(f'fused xHC post expects only dynamic router_probs with last dim {K - 2}, got {router_probs.shape[-1]}.')
            fixed = torch.ones(*router_probs.shape[:-1], K - router_probs.shape[-1], device=router_probs.device, dtype=router_probs.dtype)
            rw_full = torch.cat([fixed, router_probs], dim=-1)
            rw = rw_full.unsqueeze(-1).to(tgt)
            grad_post_add = grad_updated.to(tgt) * rw
            if R > 1:
                grad_comb_out = torch.matmul(h_post.transpose(-2, -1).to(tgt), grad_post_add).to(origin_dtype)
                grad_hp_out = torch.matmul(grad_post_add, comb_or_out.to(tgt).transpose(-2, -1))
            else:
                grad_comb_out = torch.einsum('sbnc,sbn->sbc', grad_post_add, h_post.to(tgt)).to(origin_dtype)
                grad_hp_out = torch.einsum('sbnc,sbc->sbn', grad_post_add, comb_or_out.to(tgt))
            recompute_dtype = torch.bfloat16
            if R > 1:
                post_add = torch.matmul(h_post.to(recompute_dtype), comb_or_out.to(recompute_dtype)).to(tgt)
            else:
                post_add = torch.einsum('sbn,sbc->sbnc', h_post.to(recompute_dtype), comb_or_out.to(recompute_dtype)).to(tgt)
            grad_rw_full = (grad_updated.to(tgt) * post_add).sum(-1)
            grad_rw_out = grad_rw_full[..., K - 2:]
        return (grad_comb_out, grad_hp_out, grad_rw_out, grad_mixed_out, None, grad_residual_out, None, None)


class FusedPostSplitR4Function(torch.autograd.Function):
    """MLP post path for R=4 that avoids cat([main_out, conv_out])."""

    @staticmethod
    def forward(ctx, main_out, conv_out, h_post, router_probs, mixed_active, topk_indices, residual, opt_level):
        S, B, C = (residual.shape[0], residual.shape[1], residual.shape[-1])
        K = topk_indices.shape[2]
        N = residual.shape[2]
        origin_dtype = residual.dtype
        M = S * B
        if HAS_TRITON and residual.is_cuda and (K == 4):
            store_dtype = 1 if origin_dtype == torch.bfloat16 else 2 if origin_dtype == torch.float16 else 0
            main_flat = main_out.reshape(M, C)
            conv_flat = conv_out.reshape(M, 3, C)
            hp_flat = h_post.float().reshape(M, 16)
            if router_probs.shape[-1] != 2:
                raise RuntimeError(f'fused xHC split-R4 post expects only dynamic router_probs with last dim 2, got {router_probs.shape[-1]}.')
            rw_flat = router_probs.float().reshape(M, 2)
            mixed_flat = mixed_active.reshape(M, 4, C)
            topk_flat = topk_indices.reshape(M, 4)
            residual_flat = residual.reshape(M, N, C).contiguous()
            fuse_update_scatter = True
            if fuse_update_scatter:
                out_flat = residual_flat.clone()
                grid_update = lambda META: (M, triton.cdiv(C, META['BLOCK_C']))
                xhc_post_fwd_update_split_r4_kernel[grid_update](main_flat, conv_flat, hp_flat, rw_flat, mixed_flat, topk_flat, out_flat, M, N=N, C=C)
            else:
                updated_flat = torch.empty(M, 4, C, device=residual.device, dtype=residual.dtype)
                grid = lambda META: (M, triton.cdiv(C, META['BLOCK_C']))
                xhc_post_fwd_slim_split_r4_kernel[grid](main_flat, conv_flat, hp_flat, rw_flat, mixed_flat, updated_flat, M, C=C)
            if not fuse_update_scatter:
                out_flat = residual_flat.clone()
                grid_update = lambda META: (M, K, triton.cdiv(C, META['BLOCK_C']))
                xhc_post_update_scatter_kernel[grid_update](updated_flat, topk_flat, out_flat, M, N=N, K=K, C=C)
            hidden_states = out_flat.reshape(S, B, N, C)
        else:
            compute_dtype = torch.bfloat16
            combined = torch.cat([main_out.unsqueeze(2), conv_out], dim=2)
            post_add = torch.matmul(h_post.to(compute_dtype), combined.to(compute_dtype))
            if router_probs.shape[-1] != 2:
                raise RuntimeError(f'fused xHC split-R4 post expects only dynamic router_probs with last dim 2, got {router_probs.shape[-1]}.')
            fixed = torch.ones(*router_probs.shape[:-1], 2, device=router_probs.device, dtype=router_probs.dtype)
            rw_full = torch.cat([fixed, router_probs], dim=-1)
            rw = rw_full.unsqueeze(-1).to(compute_dtype)
            updated = torch.addcmul(mixed_active.to(compute_dtype), post_add, rw).to(origin_dtype)
            idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, C)
            hidden_states = residual.scatter(2, idx_exp, updated)
        ctx.save_for_backward(main_out, conv_out, h_post, router_probs, topk_indices)
        ctx.opt_level = int(opt_level)
        ctx.shape_info = (S, B, C, K, N)
        ctx.origin_dtype = origin_dtype
        return hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        main_out, conv_out, h_post, router_probs, topk_indices = ctx.saved_tensors
        S, B, C, K, N = ctx.shape_info
        M = S * B
        origin_dtype = ctx.origin_dtype
        if HAS_TRITON and grad_output.is_cuda and (K == 4):
            store_dtype = 1 if origin_dtype == torch.bfloat16 else 2 if origin_dtype == torch.float16 else 0
            go_flat = grad_output.reshape(M, N, C).clone(memory_format=torch.contiguous_format)
            topk_flat = topk_indices.reshape(M, 4).contiguous()
            main_flat = main_out.reshape(M, C)
            conv_flat = conv_out.reshape(M, 3, C)
            hp_flat = h_post.float().reshape(M, 16)
            if router_probs.shape[-1] != 2:
                raise RuntimeError(f'fused xHC split-R4 post expects only dynamic router_probs with last dim 2, got {router_probs.shape[-1]}.')
            rw_flat = router_probs.float().reshape(M, 2)
            grad_main = torch.empty(M, C, device=grad_output.device, dtype=origin_dtype)
            grad_conv = torch.empty(M, 3, C, device=grad_output.device, dtype=origin_dtype)
            grad_hp = torch.empty(M, 16, device=grad_output.device, dtype=torch.float32)
            grad_rw = torch.empty(M, K - 2, device=grad_output.device, dtype=torch.float32)
            fuse_gather_slim = True
            if fuse_gather_slim:
                gu_flat = torch.empty(M, 4, C, device=grad_output.device, dtype=origin_dtype)
                xhc_post_bwd_gather_slim_split_r4_kernel[M,](go_flat, topk_flat, main_flat, conv_flat, hp_flat, rw_flat, grad_main, grad_conv, grad_hp, grad_rw, gu_flat, M, N=N, C=C, STORE_DTYPE=store_dtype, BLOCK_C=256, num_warps=1, num_stages=2)
                grad_residual = go_flat
            else:
                gu_flat = torch.empty(M, 4, C, device=grad_output.device, dtype=origin_dtype)
                grid_gather = lambda META: (M, triton.cdiv(C, META['BLOCK_C']))
                xhc_post_bwd_gather_zero_inplace_k4_kernel[grid_gather](go_flat, topk_flat, gu_flat, M, N=N, K=K, C=C, STORE_DTYPE=store_dtype, BLOCK_C=512, num_warps=1, num_stages=2)
                grad_residual = go_flat
                xhc_post_bwd_slim_split_r4_kernel[M,](gu_flat, main_flat, conv_flat, hp_flat, rw_flat, grad_main, grad_conv, grad_hp, grad_rw, M, C=C, STORE_DTYPE=store_dtype)
            grad_main_out = grad_main.reshape(S, B, C)
            grad_conv_out = grad_conv.reshape(S, B, 3, C)
            grad_hp_out = grad_hp.reshape(S, B, 4, 4)
            grad_rw_out = grad_rw.reshape(S, B, K - 2)
            grad_mixed_out = gu_flat.reshape(S, B, 4, C)
            grad_residual_out = grad_residual.reshape(S, B, N, C)
        else:
            combined = torch.cat([main_out.unsqueeze(2), conv_out], dim=2)
            idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, C)
            grad_updated = torch.gather(grad_output, 2, idx_exp)
            grad_residual_out = grad_output.clone()
            grad_residual_out.scatter_(2, idx_exp, torch.zeros_like(grad_updated))
            grad_mixed_out = grad_updated
            if router_probs.shape[-1] != 2:
                raise RuntimeError(f'fused xHC split-R4 post expects only dynamic router_probs with last dim 2, got {router_probs.shape[-1]}.')
            fixed = torch.ones(*router_probs.shape[:-1], 2, device=router_probs.device, dtype=router_probs.dtype)
            rw_full = torch.cat([fixed, router_probs], dim=-1)
            grad_post_add = grad_updated.float() * rw_full.unsqueeze(-1).float()
            grad_comb = torch.matmul(h_post.transpose(-2, -1).float(), grad_post_add).to(origin_dtype)
            recompute_dtype = torch.bfloat16
            post_add = torch.matmul(h_post.to(recompute_dtype), combined.to(recompute_dtype)).to(torch.float32)
            grad_hp_out = torch.matmul(grad_post_add.to(recompute_dtype), combined.to(recompute_dtype).transpose(-2, -1)).float()
            grad_rw_full = (grad_updated.float() * post_add).sum(-1)
            grad_rw_out = grad_rw_full[..., 2:]
            grad_main_out = grad_comb[:, :, 0]
            grad_conv_out = grad_comb[:, :, 1:]
        return (grad_main_out, grad_conv_out, grad_hp_out, grad_rw_out, grad_mixed_out, None, grad_residual_out, None)


def _sinkhorn_knopp_log(logits: Tensor, num_iters: int = 20) -> Tensor:
    for _ in range(num_iters):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return torch.exp(logits)


def _compiled_post_stats_routed(
    mean_x, mean_x2, combined_raw,
    colsum_Wr_routed, alpha_pre, bias_pre,
    S, B, N, num_fixed, router_cols, router_offset,
):
    """Stats correction for fused xHC layouts with routed router logits and full h_pre."""
    routed_n = N - num_fixed
    var_x = (mean_x2 - mean_x.pow(2)).clamp_min(0)
    rstd = torch.rsqrt(var_x + 1e-5)
    inv_rms = torch.rsqrt(mean_x2).nan_to_num(1.0, 1.0, 1.0)

    combined_raw = combined_raw.reshape(S, B, router_cols + N)
    router_raw = combined_raw[..., router_offset : router_offset + routed_n]
    h_pre_raw = combined_raw[..., router_cols : router_cols + N]

    mean_x_sb = mean_x.reshape(S, B, 1)
    rstd_sb = rstd.reshape(S, B, 1)
    inv_rms_sb = inv_rms.reshape(S, B, 1)
    router_logits = (router_raw - mean_x_sb * colsum_Wr_routed) * rstd_sb
    h_pre_raw = h_pre_raw * inv_rms_sb
    h_pre = torch.sigmoid(h_pre_raw * alpha_pre + bias_pre)
    return router_logits, h_pre


class _FusedPostResActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        projected,
        alpha_post,
        bias_post,
        alpha_res,
        bias_res,
        S: int,
        B: int,
        active_n: int,
        post_rank: int,
        dim_post: int,
        dim_res: int,
    ):
        M = int(S) * int(B)
        active_n = int(active_n)
        post_rank = int(post_rank)
        dim_post = int(dim_post)
        dim_res = int(dim_res)
        out_dim = dim_post + dim_res
        alpha_numel = int(alpha_post.numel())
        projected_flat = projected.reshape(M, out_dim).contiguous()
        alpha_p = alpha_post.contiguous().reshape(-1)
        bias_p = bias_post.contiguous().reshape(-1)
        alpha_r = alpha_res.contiguous().reshape(-1)
        bias_r = bias_res.contiguous().reshape(-1)
        h_post = torch.empty((M, dim_post), device=projected.device, dtype=torch.float32)
        h_res_logits = torch.empty((M, dim_res), device=projected.device, dtype=torch.float32)
        padded_post = max(16, 1 << (dim_post - 1).bit_length())
        padded_res = max(16, 1 << (dim_res - 1).bit_length())
        xhc_post_res_activation_fwd_kernel[(M,)](
            projected_flat,
            alpha_p,
            bias_p,
            alpha_r,
            bias_r,
            h_post,
            h_res_logits,
            M,
            OUT_DIM=out_dim,
            DIM_POST=dim_post,
            DIM_RES=dim_res,
            POST_RANK=post_rank,
            ALPHA_POST_NUMEL=alpha_numel,
            PADDED_POST=padded_post,
            PADDED_RES=padded_res,
            num_warps=1,
        )
        ctx.save_for_backward(projected_flat, h_post, alpha_p, alpha_r)
        ctx.shape_info = (
            projected.shape,
            alpha_post.shape,
            bias_post.shape,
            alpha_res.shape,
            bias_res.shape,
            M,
            out_dim,
            dim_post,
            dim_res,
            post_rank,
            alpha_numel,
            int(S),
            int(B),
            active_n,
        )
        h_post_out = (
            h_post.reshape(int(S), int(B), active_n, post_rank)
            if post_rank > 1
            else h_post.reshape(int(S), int(B), active_n)
        )
        h_res_out = h_res_logits.reshape(int(S), int(B), active_n, active_n)
        return h_post_out, h_res_out

    @staticmethod
    def backward(ctx, grad_h_post, grad_h_res_logits):
        projected, h_post, alpha_p, alpha_r = ctx.saved_tensors
        (
            projected_shape,
            alpha_post_shape,
            bias_post_shape,
            alpha_res_shape,
            bias_res_shape,
            M,
            out_dim,
            dim_post,
            dim_res,
            post_rank,
            alpha_numel,
            _S,
            _B,
            _active_n,
        ) = ctx.shape_info
        if grad_h_post is None:
            grad_post = torch.zeros((M, dim_post), device=projected.device, dtype=torch.float32)
        else:
            grad_post = grad_h_post.reshape(M, dim_post).contiguous().float()
        if grad_h_res_logits is None:
            grad_res = torch.zeros((M, dim_res), device=projected.device, dtype=torch.float32)
        else:
            grad_res = grad_h_res_logits.reshape(M, dim_res).contiguous().float()
        grad_projected = torch.empty_like(projected)
        grad_alpha_post = torch.empty((alpha_numel,), device=projected.device, dtype=torch.float32)
        grad_bias_post = torch.empty((dim_post,), device=projected.device, dtype=torch.float32)
        grad_alpha_res = torch.empty((1,), device=projected.device, dtype=torch.float32)
        grad_bias_res = torch.empty((dim_res,), device=projected.device, dtype=torch.float32)
        partial_rows = alpha_numel + dim_post + 1 + dim_res
        partial = torch.empty((partial_rows, M), device=projected.device, dtype=torch.float32)
        padded_post = max(16, 1 << (dim_post - 1).bit_length())
        padded_res = max(16, 1 << (dim_res - 1).bit_length())
        xhc_post_res_activation_bwd_stage1_kernel[(M,)](
            projected,
            h_post,
            grad_post,
            grad_res,
            alpha_p,
            alpha_r,
            grad_projected,
            partial,
            M,
            OUT_DIM=out_dim,
            DIM_POST=dim_post,
            DIM_RES=dim_res,
            POST_RANK=post_rank,
            ALPHA_POST_NUMEL=alpha_numel,
            PADDED_POST=padded_post,
            PADDED_RES=padded_res,
            num_warps=1,
        )
        block_m = max(1, 1 << (M - 1).bit_length())
        xhc_post_res_activation_reduce_kernel[(partial_rows,)](
            partial,
            grad_alpha_post,
            grad_bias_post,
            grad_alpha_res,
            grad_bias_res,
            M,
            DIM_POST=dim_post,
            DIM_RES=dim_res,
            ALPHA_POST_NUMEL=alpha_numel,
            BLOCK_M=block_m,
            num_warps=8,
        )
        return (
            grad_projected.reshape(projected_shape),
            grad_alpha_post.reshape(alpha_post_shape),
            grad_bias_post.reshape(bias_post_shape),
            grad_alpha_res.reshape(alpha_res_shape),
            grad_bias_res.reshape(bias_res_shape),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _can_use_xhc_post_res_activation(projected: Tensor, alpha_post: Tensor, bias_post: Tensor, alpha_res: Tensor, bias_res: Tensor, post_rank: int, dim_post: int, dim_res: int) -> bool:
    return HAS_TRITON and projected.is_cuda and (projected.dtype == torch.float32) and (int(dim_res) > 0) and (int(alpha_post.numel()) in (1, int(post_rank), int(dim_post))) and (int(bias_post.numel()) == int(dim_post)) and (int(alpha_res.numel()) == 1) and (int(bias_res.numel()) == int(dim_res))


def _compiled_post_proj_sigmoid(
    projected, alpha_post, bias_post, alpha_res, bias_res,
    S, B, active_n, post_rank, dim_post, dim_res,
):
    """Fuse split + sigmoid + scale for h_post and h_res after projection."""
    if _can_use_xhc_post_res_activation(
        projected, alpha_post, bias_post, alpha_res, bias_res, post_rank, dim_post, dim_res
    ):
        with _mhc_nvtx_range("xhc.fused.post_res_activation"):
            return _FusedPostResActivationFunction.apply(
                projected,
                alpha_post,
                bias_post,
                alpha_res,
                bias_res,
                int(S),
                int(B),
                int(active_n),
                int(post_rank),
                int(dim_post),
                int(dim_res),
            )

    projected = projected.reshape(S, B, -1)
    h_post_raw, h_res_raw = projected.split([dim_post, dim_res], dim=-1)
    if post_rank > 1:
        h_post_raw = h_post_raw.reshape(S, B, active_n, post_rank)
        if bias_post.dim() == 1:
            bias_post = bias_post.view(active_n, post_rank)
    h_post = torch.sigmoid(h_post_raw * alpha_post + bias_post) * 2.0
    h_res_logits = (h_res_raw * alpha_res + bias_res).reshape(S, B, active_n, active_n)
    return h_post, h_res_logits


@torch.compile(dynamic=False, disable=not torch.cuda.is_available())
def _compiled_active_proj(
    flat_active_bf16, proj_post_res,
    alpha_post, bias_post, alpha_res, bias_res,
    S, B, active_n, post_rank, dim_post, dim_res,
):
    """Active stream RMS + projection + h_post sigmoid + h_res logits — fuses ~8 kernels."""
    inv_rms_act = torch.rsqrt(
        flat_active_bf16.float().pow(2).mean(-1, keepdim=True)
    ).nan_to_num(1.0, 1.0, 1.0)
    flat_active_bf16 = _mhc_require_bf16_activation(flat_active_bf16, "fused xHC compiled active projection input")
    W_bf16 = _mhc_require_bf16_weight(proj_post_res, "fused xHC compiled active projection weight")
    projected = torch.matmul(flat_active_bf16, W_bf16).float() * inv_rms_act
    projected = projected.reshape(S, B, -1)

    h_post_raw, h_res_raw = projected.split([dim_post, dim_res], dim=-1)
    if post_rank > 1:
        h_post_raw = h_post_raw.reshape(S, B, active_n, post_rank)
        if bias_post.dim() == 1:
            bias_post = bias_post.view(active_n, post_rank)
    h_post = torch.sigmoid(h_post_raw * alpha_post + bias_post) * 2.0
    h_res_logits = (h_res_raw * alpha_res + bias_res).reshape(S, B, active_n, active_n)
    return h_post, h_res_logits


def _fused_mhc_xhc_pre_forward_impl(mhc_module, hidden_states: Tensor, residual: Tensor, return_residual_for_post: bool=False) -> Tuple[Tensor, ...]:
    """
    PyTorch optimized pre-forward for anchor config.

    Optimizations:
      1. Merged matmul: flat @ [W_router; W_pre] → 1 cuBLAS instead of 2
      2. LN post-correction: (x@W - μ·colsum(W)) * rstd — no LN output tensor
      3. bf16 einsum: h_pre aggregation without .to(fp32) cast

    Returns: (hidden_for_submodule, mixed_active_latent, h_post, topk_indices, router_probs)
    where fused xHC router_probs stores only the two dynamic routed weights; the two fixed
    stream weights are implicit ones in the fused xHC post path.
    and, when requested by the fused transformer path, a residual passthrough
    whose gradient is folded into the pre_res backward kernel.
    """
    S, B, N, C = hidden_states.shape
    if hidden_states.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise TypeError(f'fused xHC is a bf16 activation path; got hidden_states={hidden_states.dtype}, residual={residual.dtype}.')
    origin_dtype = hidden_states.dtype
    M = S * B
    D = N * C
    use_triton = HAS_TRITON and hidden_states.is_cuda
    opt_level = _set_hc_fused_opt_level(mhc_module)
    _mhc_log_inner_memory(mhc_module, 'pre_start')
    if N != 16:
        raise ValueError(f'fused xHC is fixed to 16 streams, got N={N}.')
    num_fixed = mhc_module.num_fixed
    num_routed = mhc_module.num_routed
    W_router_T_full = mhc_module.router.weight.T
    keep_fixed_router_logits = _use_xhc_keep_fixed_router_logits(mhc_module)
    if keep_fixed_router_logits:
        W_router_T = W_router_T_full
        router_offset = num_fixed
    else:
        W_router_T = W_router_T_full[:, num_fixed:]
        router_offset = 0
    router_cols = W_router_T.shape[1]
    routed_n = N - num_fixed
    OUT_DIM = router_cols + mhc_module.proj_pre.shape[1]
    use_cublas_full_proj = False
    use_cublas_active_proj = False
    flat_bf16 = hidden_states.reshape(M, D)
    hidden_states_for_sparse = hidden_states
    if use_triton:
        W_combined = torch.cat([W_router_T, mhc_module.proj_pre], dim=1)
        combined_raw, mean_x, mean_x2, flat_for_sparse = FusedProjPassthroughFunction.apply(flat_bf16, W_combined)
        hidden_states_for_sparse = flat_for_sparse.reshape(S, B, N, C)
    else:
        W_combined = torch.cat([W_router_T, mhc_module.proj_pre], dim=1)
        flat_stats = flat_bf16.float()
        mean_x = flat_stats.mean(dim=-1, keepdim=True)
        mean_x2 = flat_stats.pow(2).mean(dim=-1, keepdim=True)
        combined_raw = torch.matmul(_mhc_require_bf16_activation(flat_bf16, 'fused xHC full projection input'), _mhc_require_bf16_weight(W_combined, 'fused xHC full projection weight')).float()
    _mhc_log_inner_memory(mhc_module, 'pre_after_full_proj')
    colsum_Wr = W_router_T.float().sum(dim=0)[router_offset:router_offset + routed_n].contiguous()
    use_routing_bias = hasattr(mhc_module, 'routing_bias') and (not getattr(mhc_module.config, 'debug_routing_bias', False))
    use_post_stats_router_top2 = use_triton and N == 16 and (num_fixed == 2) and (num_routed == 2)
    if use_post_stats_router_top2:
        h_pre, topk_indices, router_probs = FusedRoutedPostStatsRouterTop2Function.apply(combined_raw, mean_x, mean_x2, colsum_Wr, mhc_module.alpha_pre, mhc_module.mhc_bias_pre, mhc_module.routing_bias if use_routing_bias else None, S, B, N, num_fixed, router_cols, router_offset, use_routing_bias)
        _mhc_log_inner_memory(mhc_module, 'pre_after_hpre_router')
    else:
        router_logits, h_pre = _compiled_post_stats_routed(mean_x, mean_x2, combined_raw, colsum_Wr, mhc_module.alpha_pre, mhc_module.mhc_bias_pre, S, B, N, num_fixed, router_cols, router_offset)
        _mhc_log_inner_memory(mhc_module, 'pre_after_hpre')
        fixed_idx = torch.arange(num_fixed, device=hidden_states.device, dtype=torch.long).view(1, 1, -1).expand(S, B, -1)
        if use_routing_bias:
            routed_logits_for_topk = router_logits + mhc_module.routing_bias[num_fixed:]
        else:
            routed_logits_for_topk = router_logits
        _, routed_idx_raw = torch.topk(routed_logits_for_topk, num_routed, dim=-1)
        routed_idx = routed_idx_raw + num_fixed
        pure_scores = torch.gather(router_logits, -1, routed_idx_raw)
        dynamic_w = torch.sigmoid(pure_scores)
        topk_indices = torch.cat([fixed_idx, routed_idx], dim=-1)
        router_probs = dynamic_w
        _mhc_log_inner_memory(mhc_module, 'pre_after_router')
    idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, C)
    active_mean_x2 = None
    if use_triton and mhc_module.active_n == 4:
        active, active_mean_x2, hidden_states_for_pre_res = FusedActiveGatherStatsPassthroughFunction.apply(hidden_states_for_sparse, topk_indices)
    elif use_triton:
        active, hidden_states_for_pre_res = FusedActiveGatherPassthroughFunction.apply(hidden_states_for_sparse, topk_indices)
    else:
        active = torch.gather(hidden_states_for_sparse, 2, idx_exp)
        hidden_states_for_pre_res = hidden_states_for_sparse
    _mhc_log_inner_memory(mhc_module, 'pre_after_active_gather')
    post_rank = mhc_module.post_rank_for_proj
    dim_post = mhc_module.active_n * post_rank
    dim_res = mhc_module.active_n ** 2
    flat_active_bf16 = active.reshape(S * B, mhc_module.active_n * C)
    recompute_active_saved = False
    if use_triton:
        if active_mean_x2 is not None:
            if recompute_active_saved:
                projected = FusedActiveProjWithStatsRecomputeFunction.apply(flat_active_bf16, hidden_states_for_sparse, topk_indices, mhc_module.proj_post_res, active_mean_x2.detach())
            else:
                projected = FusedActiveProjWithStatsFunction.apply(flat_active_bf16, mhc_module.proj_post_res, active_mean_x2.detach())
        elif recompute_active_saved:
            projected = FusedActiveProjRecomputeFunction.apply(flat_active_bf16, hidden_states_for_sparse, topk_indices, mhc_module.proj_post_res)
        else:
            projected = FusedActiveProjFunction.apply(flat_active_bf16, mhc_module.proj_post_res)
        h_post, h_res_logits = _compiled_post_proj_sigmoid(projected, mhc_module.alpha_post, mhc_module.mhc_bias_post, mhc_module.alpha_res, mhc_module.mhc_bias_res, S, B, mhc_module.active_n, post_rank, dim_post, dim_res)
    else:
        h_post, h_res_logits = _compiled_active_proj(flat_active_bf16, mhc_module.proj_post_res, mhc_module.alpha_post, mhc_module.mhc_bias_post, mhc_module.alpha_res, mhc_module.mhc_bias_res, S, B, mhc_module.active_n, post_rank, dim_post, dim_res)
    _mhc_log_inner_memory(mhc_module, 'pre_after_active_proj')
    if use_triton:
        sinkhorn_iters = int(getattr(getattr(mhc_module, 'config', None), 'mhc_sinkhorn_iterations', 20))
        h_res = _triton_sinkhorn_with_grad(h_res_logits, mhc_module.active_n, sinkhorn_iters)
    else:
        h_res = _sinkhorn_knopp_log(h_res_logits, num_iters=20)
        h_res = h_res / h_res.sum(dim=-1, keepdim=True).clamp(min=1.0)
    _mhc_log_inner_memory(mhc_module, 'pre_after_sinkhorn')
    K = mhc_module.active_n
    if use_triton and N == 16 and (K == 4):
        hidden_for_sub, mixed_active, residual_for_post = FusedPreResActiveFunction.apply(hidden_states_for_pre_res, h_pre, h_res, active, topk_indices, N, K)
    elif use_triton:
        hidden_for_sub, mixed_active, residual_for_post = FusedPreResFunction.apply(hidden_states_for_pre_res, h_pre, h_res, topk_indices, N, K)
    else:
        hidden_for_sub = torch.einsum('sbn,sbnc->sbc', h_pre.to(origin_dtype), hidden_states)
        active_latent = torch.gather(residual, 2, idx_exp)
        mixed_active = torch.matmul(h_res.to(torch.bfloat16), _mhc_require_bf16_activation(active_latent, 'fused xHC pre-res active latent')).to(origin_dtype)
        residual_for_post = residual
    _mhc_log_inner_memory(mhc_module, 'pre_after_pre_res')
    if return_residual_for_post:
        return (hidden_for_sub, mixed_active, h_post, topk_indices, router_probs, residual_for_post, h_pre)
    return (hidden_for_sub, mixed_active, h_post, topk_indices, router_probs)


full_xhc_pre = _fused_mhc_xhc_pre_forward_impl
