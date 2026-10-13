"""Bounded-scratch, strided activation reduction with centered moments."""
import torch
import triton
import triton.language as tl


@triton.jit
def _partials(X, P, N: tl.constexpr, SHAPE: tl.constexpr, STRIDE: tl.constexpr,
              GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    count = tl.full((), 0., tl.float32)
    mean = tl.full((), 0., tl.float32)
    m2 = tl.full((), 0., tl.float32)
    maximum = tl.full((), -float('inf'), tl.float32)
    for start in range(group * BLOCK, N, GROUPS * BLOCK):
        linear = start + lanes
        remaining = linear
        offset = tl.full((BLOCK,), 0, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            offset += (remaining % SHAPE[axis]).to(tl.int64) * STRIDE[axis]
            remaining = remaining // SHAPE[axis]
        valid = linear < N
        x = tl.load(X + offset, valid, other=0).to(tl.float32)
        n = tl.sum(valid.to(tl.float32), 0)
        local_mean = tl.sum(x, 0) / n
        centered = x - local_mean
        local_m2 = tl.sum(tl.where(valid, centered * centered, 0.), 0)
        delta = local_mean - mean
        total = count + n
        m2 += local_m2 + delta * delta * count * n / total
        mean += delta * n / total
        count = total
        maximum = tl.maximum(maximum, tl.max(tl.where(valid, x, -float('inf')), 0))
    tl.store(P + group, mean)
    tl.store(P + 1024 + group, m2)
    tl.store(P + 2048 + group, maximum)


@triton.jit
def _merge(P, M, N: tl.constexpr, ROW: tl.constexpr, GROUPS: tl.constexpr,
           BLOCK: tl.constexpr, REDUCE: tl.constexpr):
    i = tl.arange(0, REDUCE)
    # Each program owns a strided series of blocks, plus possibly one tail.
    cycles = N // (GROUPS * BLOCK)
    remainder = N % (GROUPS * BLOCK)
    counts = (cycles * BLOCK + tl.minimum(tl.maximum(remainder - i * BLOCK, 0), BLOCK))
    counts = tl.where(i < GROUPS, counts, 0).to(tl.float64)
    means = tl.load(P + i, i < GROUPS, 0).to(tl.float64)
    m2s = tl.load(P + 1024 + i, i < GROUPS, 0).to(tl.float64)
    maxima = tl.load(P + 2048 + i, i < GROUPS, -float('inf')).to(tl.float64)
    mean = tl.sum(means * counts, 0) / N
    centered = means - mean
    m2 = tl.sum(m2s + counts * centered * centered, 0)
    dst = M + ROW * 4
    old_n = tl.load(dst)
    old_mean = tl.load(dst + 1)
    old_m2 = tl.load(dst + 2)
    old_max = tl.load(dst + 3)
    total = old_n + N
    delta = mean - old_mean
    tl.store(dst, total)
    tl.store(dst + 1, old_mean + delta * N / total)
    tl.store(dst + 2, old_m2 + m2 + delta * delta * old_n * N / total)
    tl.store(dst + 3, tl.maximum(old_max, tl.max(maxima, 0)))


def accumulate(tensor, scratch, moments, row):
    if not tensor.is_cuda or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('Activation statistics require CUDA FP16/BF16/FP32 tensors')
    n = tensor.numel()
    if tensor.is_contiguous():
        shape, strides = (n,), (1,)
    else:
        shape, strides = tuple(tensor.shape), tuple(tensor.stride())
    block = 4096
    groups = min(triton.cdiv(n, block), 1024)
    _partials[(groups,)](tensor, scratch, n, shape, strides, groups, block)
    _merge[(1,)](scratch, moments, n, row, groups, block, triton.next_power_of_2(groups))
