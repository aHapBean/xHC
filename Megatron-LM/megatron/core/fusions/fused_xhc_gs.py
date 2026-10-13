"""BF16 detached-basis Gram-Schmidt preserving the reference rounding order.

Constant chunk selection uses register splits on the eight-accumulator path;
the original summation order is retained.
"""
import torch
import triton
import triton.language as tl

# Native-dtype GS preserves the explicit rounding boundaries in HyperConnection.
# It replaces the imported GS path, whose FP32 intermediates change native BF16 GS.
@triton.jit
def _native_bf16_sum(value, VEC: tl.constexpr, C: tl.constexpr):
    # Match the BF16 vectorized sum: four (2.9) or eight (2.14) accumulators per lane,
    # combined in order, followed by a 32-lane reduction. A flat tree can move
    # values across a BF16 rounding boundary even when its FP32 error is tiny.
    BLOCK: tl.constexpr = value.shape[0]
    WIDTH: tl.constexpr = 32 * VEC
    positions = tl.arange(0, WIDTH)
    accumulators = tl.full((WIDTH,), 0, tl.float32)
    CHUNKS: tl.constexpr = (C + WIDTH - 1) // WIDTH
    for chunk in tl.static_range(CHUNKS):
        if BLOCK >= WIDTH and VEC == 8:
            piece = value.reshape((BLOCK // WIDTH, WIDTH)).trans()
            # Select a constant row by splitting registers, without a gather
            # layout conversion. Accumulation order remains chunk 0, 1, ... .
            for level in tl.static_range(BLOCK // WIDTH):
                if (1 << level) < BLOCK // WIDTH:
                    even, odd = tl.split(piece.reshape((WIDTH, (BLOCK // WIDTH) // (2 << level), 2)))
                    if (chunk >> level) & 1:
                        piece = odd
                    else:
                        piece = even
            accumulators += piece.reshape((WIDTH,))
        elif BLOCK >= WIDTH:
            matrix = value.reshape((BLOCK // WIDTH, WIDTH))
            indices = tl.full((1, WIDTH), chunk, tl.int32)
            accumulators += tl.gather(matrix, indices, 0).reshape((WIDTH,))
        else:
            index = positions + chunk * WIDTH
            part = tl.gather(value, tl.minimum(index, BLOCK - 1), 0)
            accumulators += tl.where(index < BLOCK, part, 0)
    vectors = accumulators.reshape((32, VEC))
    lanes = tl.full((32,), 0, tl.float32)
    for slot in tl.static_range(VEC):
        lanes += tl.gather(vectors, tl.full((32, 1), slot, tl.int32), 1).reshape((32,))
    lane_ids = tl.arange(0, 32)
    for shift in tl.static_range(5):
        offset = (1 << shift) if VEC == 4 else (16 >> shift)
        others = tl.gather(lanes, (lane_ids + offset) % 32, 0)
        lanes = lanes + tl.where(lane_ids + offset < 32, others, lanes)
    return tl.sum(tl.where(lane_ids == 0, lanes, 0), 0)


@triton.jit
def _native_denominator(base, DTYPE: tl.constexpr, VEC: tl.constexpr, C: tl.constexpr):
    square = (base * base).to(DTYPE).to(tl.float32)
    denominator = _native_bf16_sum(square, VEC, C).to(DTYPE).to(tl.float32)
    return tl.maximum(denominator, 1.e-12).to(DTYPE).to(tl.float32)

@triton.jit
def _native_remove(value, base, denominator, DTYPE: tl.constexpr, VEC: tl.constexpr, C: tl.constexpr):
    product = (value * base).to(DTYPE).to(tl.float32)
    numerator = _native_bf16_sum(product, VEC, C).to(DTYPE).to(tl.float32)
    coefficient = (numerator / denominator).to(DTYPE).to(tl.float32)
    delta = (coefficient * base).to(DTYPE).to(tl.float32)
    return (value - delta).to(DTYPE).to(tl.float32)

@triton.jit
def _native_remove_backward(grad, base, denominator, DTYPE: tl.constexpr, VEC: tl.constexpr, C: tl.constexpr):
    dcoef = _native_bf16_sum((-grad * base).to(DTYPE).to(tl.float32), VEC, C).to(DTYPE).to(tl.float32)
    dnum = (dcoef / denominator).to(DTYPE).to(tl.float32)
    extra = (dnum * base).to(DTYPE).to(tl.float32)
    return (grad + extra).to(DTYPE).to(tl.float32)

@triton.jit
def _native_gs_forward(Main, Branches, Out, Denoms, C: tl.constexpr, BLOCK: tl.constexpr, VEC: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < C
    dtype: tl.constexpr = Main.dtype.element_ty
    main = tl.load(Main + row*C + col, mask, 0).to(tl.float32)
    d0 = _native_denominator(main, dtype, VEC, C)
    b0 = tl.load(Branches + row*3*C + col, mask, 0).to(tl.float32)
    u0 = _native_remove(b0, main, d0, dtype, VEC, C)
    d1 = _native_denominator(u0, dtype, VEC, C)
    b1 = tl.load(Branches + row*3*C + C + col, mask, 0).to(tl.float32)
    u1 = _native_remove(_native_remove(b1, main, d0, dtype, VEC, C), u0, d1, dtype, VEC, C)
    d2 = _native_denominator(u1, dtype, VEC, C)
    b2 = tl.load(Branches + row*3*C + 2*C + col, mask, 0).to(tl.float32)
    u2 = _native_remove(_native_remove(_native_remove(b2, main, d0, dtype, VEC, C), u0, d1, dtype, VEC, C), u1, d2, dtype, VEC, C)
    tl.store(Out + row*3*C + col, u0, mask)
    tl.store(Out + row*3*C + C + col, u1, mask)
    tl.store(Out + row*3*C + 2*C + col, u2, mask)
    tl.store(Denoms + row*3 + 0, d0)
    tl.store(Denoms + row*3 + 1, d1)
    tl.store(Denoms + row*3 + 2, d2)


@triton.jit
def _native_gs_backward(Main, Bases, Denoms, Grad, Out, C: tl.constexpr, BLOCK: tl.constexpr, VEC: tl.constexpr):
    row, branch = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, BLOCK)
    mask = col < C
    dtype: tl.constexpr = Main.dtype.element_ty
    g = tl.load(Grad + (row * 3 + branch) * C + col, mask, 0).to(tl.float32)
    if branch == 2:
        base = tl.load(Bases + (row * 3 + 1) * C + col, mask, 0).to(tl.float32)
        g = _native_remove_backward(g, base, tl.load(Denoms + row*3 + 2), dtype, VEC, C)
    if branch >= 1:
        base = tl.load(Bases + row * 3 * C + col, mask, 0).to(tl.float32)
        g = _native_remove_backward(g, base, tl.load(Denoms + row*3 + 1), dtype, VEC, C)
    main = tl.load(Main + row * C + col, mask, 0).to(tl.float32)
    g = _native_remove_backward(g, main, tl.load(Denoms + row*3), dtype, VEC, C)
    tl.store(Out + (row * 3 + branch) * C + col, g, mask)


class NativeGS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, main, branches):
        if main.dtype != torch.bfloat16 or branches.dtype != torch.bfloat16:
            raise ValueError("NativeGS kernel supports BF16; use native GS for other dtypes")
        assert branches.shape == (*main.shape[:2], 3, main.shape[-1])
        main, branches = main.contiguous(), branches.contiguous()
        result = torch.empty_like(branches)
        c = main.shape[-1]
        denoms = torch.empty((main.numel() // c, 3), device=main.device, dtype=torch.float32)
        # CUDA Reduce.cuh changed vector width and warp reduction order.
        ctx.vector_width = 4 if torch.__version__.startswith("2.9.") else 8
        ctx.warps = 1 if ctx.vector_width == 8 else 4
        _native_gs_forward[(main.numel() // c,)](main, branches, result, denoms, c, triton.next_power_of_2(c), ctx.vector_width, enable_fp_fusion=False, num_warps=ctx.warps)
        ctx.save_for_backward(main, result, denoms)
        return result

    @staticmethod
    def backward(ctx, grad):
        main, bases, denoms = ctx.saved_tensors
        result = torch.empty_like(bases)
        c = main.shape[-1]
        _native_gs_backward[(main.numel() // c, 3)](main, bases, denoms, grad.contiguous(), result, c, triton.next_power_of_2(c), ctx.vector_width, enable_fp_fusion=False, num_warps=ctx.warps)
        return None, result
