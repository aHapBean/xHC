"""mHC and standard xHC, using sequence-major [sequence, batch, stream, hidden].

The module owns only the connection. Attention/MLP and their normalization
remain in the backbone. Fused xHC fusion is optional; no cross-sublayer sharing.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ConnectionConfig:
    hidden_size: int
    streams: int = 16
    active_streams: int = 4
    fixed_streams: int = 2
    sparse: bool = True
    kernels: tuple[int, ...] = ()
    sinkhorn_iterations: int = 20
    row_sum_clamp: bool = True
    init_std: float = 0.006
    router_init_std: float = 0.02
    alpha_init: float = 0.01
    conv_bias: bool = True
    conv_init: str = "zero"
    gram_schmidt: bool = True
    # Both layouts occur in training recipes. Their weight-decay groups differ.
    one_dim_post: bool = False
    # "native" matches the fast GS training path; "fp32" is the reference path.
    gs_precision: str = "native"
    backend: str = "reference"

    def __post_init__(self):
        if self.backend not in ("reference", "fused_xhc"):
            raise ValueError("HC backend must be reference or fused_xhc")
        if self.backend == "fused_xhc" and (
            not self.sparse or self.streams != 16 or self.active_streams != 4
            or self.fixed_streams != 2 or self.kernels not in ((), (4, 8, 12))
            or not self.row_sum_clamp
        ):
            raise ValueError("fused_xhc requires full xHC: N16/K4/fixed2, row clamp, MLP kernels 4/8/12")
        if self.hidden_size < 1 or not 1 <= self.active_streams <= self.streams:
            raise ValueError("Require hidden_size > 0 and 1 <= active_streams <= streams")
        if not 0 <= self.fixed_streams <= self.active_streams:
            raise ValueError("fixed_streams must be in [0, active_streams]")
        if not self.sparse and (self.streams != self.active_streams or self.kernels):
            raise ValueError("Dense mHC uses all streams and no temporal branches")
        if any(k < 1 for k in self.kernels) or self.sinkhorn_iterations < 1:
            raise ValueError("Kernel sizes and Sinkhorn iterations must be positive")
        if self.gs_precision not in ("native", "fp32"):
            raise ValueError("gs_precision must be native or fp32")
        if self.conv_init != "zero":
            raise ValueError("Only zero convolution initialization is supported")

    @classmethod
    def mhc(cls, hidden_size: int, **kwargs):
        return cls(hidden_size, streams=4, active_streams=4, fixed_streams=0,
                   sparse=False, row_sum_clamp=False, **kwargs)


def connection_pair(hidden_size: int, kind: str, settings: dict | None = None):
    """Resolve the recipe section into attention/MLP configs; conv is MLP-only."""
    if kind == "mhc":
        options = dict(streams=4, active_streams=4, fixed_streams=0,
                       sparse=False, row_sum_clamp=False, gram_schmidt=False,
                       mlp_kernels=[])
    elif kind == "xhc":
        options = dict(sparse=True, mlp_kernels=[4, 8, 12])
    else:
        raise ValueError("kind must be mhc or xhc")
    options.update(settings or {})
    if "kernels" in options or "hidden_size" in options:
        raise ValueError("Use mlp_kernels; hidden_size comes from the backbone")
    if options['sparse'] != (kind == "xhc"):
        raise ValueError("mHC requires sparse=false; xHC requires sparse=true")
    kernels = tuple(options.pop("mlp_kernels"))
    return (ConnectionConfig(hidden_size, kernels=(), **options),
            ConnectionConfig(hidden_size, kernels=kernels, **options))


@dataclass
class ConnectionState:
    residual: Tensor
    mixed: Tensor
    post: Tensor
    indices: Tensor | None
    route_weights: Tensor | None


def expand_streams(x: Tensor, streams: int) -> Tensor:
    return x.unsqueeze(-2).expand(*x.shape[:-1], streams, x.shape[-1])


def reduce_streams(x: Tensor) -> Tensor:
    return x.sum(dim=-2)


def sinkhorn(logits: Tensor, iterations: int) -> Tensor:
    for _ in range(iterations):
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        logits = logits - logits.logsumexp(dim=-2, keepdim=True)
    return logits.exp()


def orthogonalize(main: Tensor, branches: Tensor, precision: str) -> Tensor:
    """Sequential GS with stopped gradients through every basis vector."""
    dtype = branches.dtype
    basis = [main.detach().float() if precision == "fp32" else main.detach()]
    result = []
    for branch in branches.unbind(dim=2):
        projected = branch.float() if precision == "fp32" else branch
        for base in basis:
            denominator = (base * base).sum(-1, keepdim=True).clamp_min(1e-12)
            projected = projected - (projected * base).sum(-1, keepdim=True) / denominator * base
        result.append(projected)
        basis.append(projected.detach())
    return torch.stack(result, dim=2).to(dtype)


class HyperConnection(nn.Module):
    def __init__(self, config: ConnectionConfig):
        super().__init__()
        self.config = config
        n, k, c = config.streams, config.active_streams, config.hidden_size
        self.rank = 1 + len(config.kernels)
        post_size = k * self.rank
        if config.sparse:
            self.router = nn.Linear(n * c, n, bias=False)
            self.proj_pre = nn.Parameter(torch.empty(n * c, n))
            self.proj_post_res = nn.Parameter(torch.empty(k * c, post_size + k * k))
        else:
            self.proj = nn.Parameter(torch.empty(n * c, n + post_size + n * n))
        self.alpha_pre = nn.Parameter(torch.full((1,), config.alpha_init))
        self.bias_pre = nn.Parameter(torch.zeros(n))
        alpha_shape = (self.rank,) if config.one_dim_post or self.rank == 1 else (1, self.rank)
        bias_shape = (k * self.rank,) if config.one_dim_post or self.rank == 1 else (k, self.rank)
        self.alpha_post = nn.Parameter(torch.full(alpha_shape, config.alpha_init))
        self.bias_post = nn.Parameter(torch.zeros(bias_shape))
        self.alpha_res = nn.Parameter(torch.full((1,), config.alpha_init))
        self.bias_res = nn.Parameter(torch.zeros(k * k))
        self.convs = nn.ModuleList([
            nn.Conv1d(c, c, kernel, padding=kernel - 1, groups=c, bias=config.conv_bias)
            for kernel in config.kernels
        ])
        self.reset_parameters()

    def reset_parameters(self):
        cfg = self.config
        if cfg.sparse:
            nn.init.normal_(self.router.weight, std=cfg.router_init_std)
            nn.init.normal_(self.proj_pre, std=cfg.init_std)
            nn.init.normal_(self.proj_post_res, std=cfg.init_std)
        else:
            nn.init.normal_(self.proj, std=cfg.init_std)
        for name in ("alpha_pre", "alpha_post", "alpha_res"):
            nn.init.constant_(getattr(self, name), cfg.alpha_init)
        for name in ("bias_pre", "bias_post", "bias_res"):
            nn.init.zeros_(getattr(self, name))
        for conv in self.convs:
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    @staticmethod
    def _project(x: Tensor, weight: Tensor) -> Tensor:
        # The experimental mapping uses no additive RMS epsilon. For exactly
        # zero input it replaces the non-finite reciprocal RMS with one.
        flat = x.flatten(-2).float()
        inv = torch.rsqrt(flat.square().mean(-1, keepdim=True)).nan_to_num(
            nan=1.0, posinf=1.0, neginf=1.0
        )
        return (flat @ weight.float()) * inv

    def read(self, x: Tensor) -> tuple[Tensor, ConnectionState]:
        """Produce a single-stream sublayer input and a per-call write context."""
        cfg = self.config
        if cfg.backend == "fused_xhc":
            from megatron.core.fusions.fused_xhc import read
            return read(self, x)
        if x.ndim != 4 or x.shape[-2:] != (cfg.streams, cfg.hidden_size):
            raise ValueError("Expected [sequence, batch, configured streams, hidden_size]")
        k = cfg.active_streams
        indices = route_weights = None
        # Disable autocast here: small mapping reductions are explicitly FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            if cfg.sparse:
                flat = x.flatten(-2)
                # Match the router's native-dtype LayerNorm + linear path.
                logits = F.linear(F.layer_norm(flat, (flat.shape[-1],)), self.router.weight)
                count = k - cfg.fixed_streams
                scores, routed = logits[..., cfg.fixed_streams:].topk(count, dim=-1)
                fixed = torch.arange(cfg.fixed_streams, device=x.device).expand(*x.shape[:2], -1)
                indices = torch.cat((fixed, routed + cfg.fixed_streams), dim=-1)
                route_weights = torch.cat((torch.ones_like(fixed, dtype=x.dtype), scores.sigmoid()), dim=-1)
                active = x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, cfg.hidden_size))
                pre_raw = self._project(x, self.proj_pre)
                post_raw, res_raw = self._project(active, self.proj_post_res).split((k * self.rank, k * k), -1)
            else:
                active = x
                pre_raw, post_raw, res_raw = self._project(x, self.proj).split((k, k, k * k), -1)
            pre = torch.sigmoid(pre_raw * self.alpha_pre.float() + self.bias_pre.float())
            if self.rank > 1:
                post_raw = post_raw.unflatten(-1, (k, self.rank))
            bias_post = self.bias_post.float()
            if self.rank > 1:
                bias_post = bias_post.view(k, self.rank)
            post = 2 * torch.sigmoid(post_raw * self.alpha_post.float() + bias_post)
            res_logits = (res_raw * self.alpha_res.float() + self.bias_res.float()).unflatten(-1, (k, k))
            mixing = sinkhorn(res_logits, cfg.sinkhorn_iterations)
            if cfg.row_sum_clamp:
                mixing = mixing / mixing.sum(-1, keepdim=True).clamp_min(1)
            sublayer_input = torch.einsum("sbn,sbnc->sbc", pre, x.float()).to(x.dtype)
            mixed = (mixing @ active.float()).to(x.dtype)
        return sublayer_input, ConnectionState(x, mixed, post, indices, route_weights)

    def features(self, output: Tensor) -> Tensor:
        if not self.convs:
            return output.unsqueeze(2)
        sequence = output.shape[0]
        conv_input = output.permute(1, 2, 0)
        branches = torch.stack([
            conv(conv_input)[..., :sequence].permute(2, 0, 1) for conv in self.convs
        ], dim=2)
        if self.config.gram_schmidt:
            branches = orthogonalize(output, branches, self.config.gs_precision)
        return torch.cat((output.unsqueeze(2), branches), dim=2)

    def write(self, output: Tensor, state: ConnectionState) -> Tensor:
        if self.config.backend == "fused_xhc":
            from megatron.core.fusions.fused_xhc import write
            return write(self, output, state)
        features = self.features(output)
        with torch.autocast(device_type=output.device.type, enabled=False):
            post = state.post.unsqueeze(-1) if self.rank == 1 else state.post
            # Cast each branch contribution BEFORE the branch sum, as in training.
            addition = (post.unsqueeze(-1) * features.float().unsqueeze(2)).to(output.dtype).sum(3)
            if state.route_weights is not None:
                addition = (addition.float() * state.route_weights.float().unsqueeze(-1)).to(output.dtype)
            updated = state.mixed + addition
            if state.indices is None:
                return updated
            index = state.indices.unsqueeze(-1).expand_as(updated)
            return state.residual.scatter(2, index, updated)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        sublayer_input, state = self.read(x)
        return self.write(sublayer(sublayer_input), state)
