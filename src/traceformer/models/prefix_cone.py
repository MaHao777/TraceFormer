"""精确锥内全量池化：前缀和共享求和，六摘要进行中心注意力。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from traceformer.config import ConeWindowConfig

from .cone_geometry import euclidean_rows
from .cone_transformer import ConeAttentionOutput, ConeTransformerBlock, ConeTransformerOutput


@dataclass(frozen=True)
class PrefixSample:
    indices: torch.Tensor
    coords: torch.Tensor
    flat_indices: torch.Tensor
    shape: tuple[int, int, int]
    corner_cache: dict[tuple, torch.Tensor] | None = field(default=None, compare=False, repr=False)


def _rectangle_indices(
    sample: PrefixSample, window: ConeWindowConfig, direction: int, start: int, end: int,
    fixed_radius: bool = False,
) -> torch.Tensor:
    """生成四角索引；越过时间边界时四角均置零，使求和贡献为零。"""
    key = (window.spatial_radius, window.time_horizon, direction, start, end, fixed_radius)
    if sample.corner_cache is not None and key in sample.corner_cache:
        return sample.corner_cache[key]
    times, height, width = sample.shape
    xyz = sample.coords[start:end]
    distance = torch.arange(1, window.time_horizon + 1, device=xyz.device)
    # 固定窗口消融仅替换截面，时间边界和求和路径与正式锥一致。
    radius = (torch.full_like(distance, window.spatial_radius) if fixed_radius
              else window.spatial_radius * distance // window.time_horizon)
    time = xyz[:, 2:3] + direction * distance
    valid = (time >= 0) & (time < times)
    time = time.clamp(0, times - 1)
    x0 = (xyz[:, 0:1] - radius).clamp(0, width)
    x1 = (xyz[:, 0:1] + radius + 1).clamp(0, width)
    y0 = (xyz[:, 1:2] - radius).clamp(0, height)
    y1 = (xyz[:, 1:2] + radius + 1).clamp(0, height)
    base = time * (height + 1) * (width + 1)
    corners = torch.stack(
        (
            base + y1 * (width + 1) + x1,
            base + y0 * (width + 1) + x1,
            base + y1 * (width + 1) + x0,
            base + y0 * (width + 1) + x0,
        ),
        dim=-1,
    )
    corners = torch.where(valid[..., None], corners, 0)
    if sample.corner_cache is not None:
        sample.corner_cache[key] = corners
    return corners


def _euclidean_indices(sample, window, direction, start, end, fixed_radius=False):
    """圆盘拆为互不重叠的单行矩形，复用前缀和及其伴随。"""
    if fixed_radius:
        raise ValueError("Euclidean cones do not support fixed-radius boxes")
    key = ("euclidean", window.spatial_radius, window.time_horizon, direction, start, end)
    if sample.corner_cache is not None and key in sample.corner_cache:
        return sample.corner_cache[key]
    times, height, width = sample.shape
    xyz = sample.coords[start:end]
    rows = xyz.new_tensor(euclidean_rows(window.spatial_radius, window.time_horizon))
    distance, dy, half_width = rows.unbind(-1)
    time = xyz[:, 2:3] + direction * distance
    y = xyz[:, 1:2] + dy
    valid = (time >= 0) & (time < times) & (y >= 0) & (y < height)
    base = time.clamp(0, times - 1) * (height + 1) * (width + 1)
    y0, y1 = y.clamp(0, height), (y + 1).clamp(0, height)
    x0 = (xyz[:, 0:1] - half_width).clamp(0, width)
    x1 = (xyz[:, 0:1] + half_width + 1).clamp(0, width)
    corners = torch.stack((base + y1 * (width + 1) + x1,
                           base + y0 * (width + 1) + x1,
                           base + y1 * (width + 1) + x0,
                           base + y0 * (width + 1) + x0), dim=-1)
    corners = torch.where(valid[..., None], corners, 0)
    if sample.corner_cache is not None:
        sample.corner_cache[key] = corners
    return corners


class _ConeSums(torch.autograd.Function):
    """固定几何下的线性区域求和；反向只分配一张前缀表梯度。"""

    @staticmethod
    def forward(ctx, features, sample, windows, chunk_size, fixed_radius, directions, metric):
        times, height, width = sample.shape
        grid = features.new_zeros((times * height * width, features.shape[-1]))
        grid.index_add_(0, sample.flat_indices, features)
        integral = F.pad(
            grid.reshape(times, height, width, -1).cumsum(1).cumsum(2), (0, 0, 1, 0, 1, 0)
        ).flatten(0, 2)
        occupied = torch.zeros(times * height * width, device=features.device, dtype=torch.int64)
        occupied.index_add_(0, sample.flat_indices, torch.ones_like(sample.flat_indices))
        occupied = F.pad(
            occupied.reshape(times, height, width).cumsum(1).cumsum(2), (1, 0, 1, 0)
        ).flatten()
        total = features.new_zeros((len(features), len(directions) * len(windows), features.shape[-1]))
        counts = torch.zeros(total.shape[:2], dtype=torch.int64, device=features.device)
        signs = features.new_tensor([1, -1, -1, 1])
        for start in range(0, len(features), chunk_size):
            end = min(start + chunk_size, len(features))
            for index, window in enumerate(windows):
                for direction_index, direction in enumerate(directions):
                    indexer = _euclidean_indices if metric == "euclidean" else _rectangle_indices
                    corners = indexer(
                        sample, window, direction, start, end, fixed_radius
                    )
                    slot = len(directions) * index + direction_index
                    # 顺序相减减少峰值临时张量，不保留逐邻居特征。
                    rectangle = (
                        integral[corners[..., 0]]
                        - integral[corners[..., 1]]
                        - integral[corners[..., 2]]
                        + integral[corners[..., 3]]
                    )
                    total[start:end, slot] = rectangle.sum(1)
                    counts[start:end, slot] = (occupied[corners] * signs.long()).sum((1, 2))
        ctx.sample, ctx.windows, ctx.chunk_size = sample, windows, chunk_size
        ctx.fixed_radius = fixed_radius
        ctx.directions = directions
        ctx.metric = metric
        ctx.feature_dim = features.shape[-1]
        ctx.mark_non_differentiable(counts)
        return total, counts

    @staticmethod
    def backward(ctx, grad_total, _grad_counts):
        sample = ctx.sample
        times, height, width = sample.shape
        grad_integral = grad_total.new_zeros((times * (height + 1) * (width + 1), ctx.feature_dim))
        for start in range(0, len(sample.indices), ctx.chunk_size):
            end = min(start + ctx.chunk_size, len(sample.indices))
            for index, window in enumerate(ctx.windows):
                for direction_index, direction in enumerate(ctx.directions):
                    indexer = _euclidean_indices if ctx.metric == "euclidean" else _rectangle_indices
                    corners = indexer(
                        sample, window, direction, start, end, ctx.fixed_radius
                    )
                    grad = grad_total[start:end, len(ctx.directions) * index + direction_index]
                    grad = (
                        grad[:, None]
                        .expand(-1, corners.shape[1], -1)
                        .reshape(-1, ctx.feature_dim)
                    )
                    for corner, sign in enumerate((1, -1, -1, 1)):
                        grad_integral.index_add_(0, corners[..., corner].reshape(-1), grad * sign)
        # 前缀和的伴随是逆向累加，再抽取输入 token 所在网格。
        grad_grid = grad_integral.reshape(times, height + 1, width + 1, -1)
        grad_grid = grad_grid.flip((1, 2)).cumsum(1).cumsum(2).flip((1, 2))
        grad_grid = grad_grid[:, 1:, 1:].reshape(-1, ctx.feature_dim)
        return grad_grid[sample.flat_indices], None, None, None, None, None, None


def prepare_prefix_geometry(coords: torch.Tensor, batches: torch.Tensor) -> list[PrefixSample]:
    """每次 encoder 前向只分组一次；不同样本不会共享前缀和。"""
    samples = []
    for batch in batches.unique():
        indices = torch.where(batches == batch)[0]
        xyz = coords[indices].long()
        # 平移后压缩网格范围，坐标差和锥边界保持不变。
        xyz = xyz - xyz.amin(dim=0)
        width, height, times = (xyz.amax(dim=0) + 1).tolist()
        flat = (xyz[:, 2] * height + xyz[:, 1]) * width + xyz[:, 0]
        # 整数几何仅在本次无梯度前向内复用；训练不保留索引缓存。
        cache = {} if not torch.is_grad_enabled() else None
        samples.append(PrefixSample(indices, xyz, flat, (times, height, width), cache))
    return samples


def prefix_cone_pool(
    values: torch.Tensor,
    weights: torch.Tensor,
    geometry: list[PrefixSample],
    windows: tuple[ConeWindowConfig, ...],
    chunk_size: int | None = None,
    *,
    fixed_radius: bool = False,
    directions: tuple[int, ...] = (-1, 1),
    metric: str = "chebyshev",
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 [N, 6, D] 摘要和 [N, 6] 成员数；不创建邻居列表。"""
    # 单向消融只计算过去区域；默认双向运算顺序保持不变。
    if directions not in {(-1, 1), (-1,)}:
        raise ValueError("directions must be bidirectional or past-only")
    if metric not in {"chebyshev", "euclidean"} or (metric == "euclidean" and fixed_radius):
        raise ValueError("unsupported cone metric or fixed-radius Euclidean cone")
    if chunk_size is None:
        # CUDA推理减少小分块发射；训练和CPU保留原临时内存规模。
        chunk_size = 16384 if values.is_cuda and not torch.is_grad_enabled() else 1024
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    summaries = values.new_zeros(
        (len(values), len(directions) * len(windows), values.shape[-1]), dtype=torch.float32
    )
    counts = torch.zeros(summaries.shape[:2], dtype=torch.int64, device=values.device)
    for sample in geometry:
        selected = sample.indices
        w = weights[selected].float().reshape(-1, 1)
        features = torch.cat((values[selected].float() * w, w), dim=-1)
        total, count = _ConeSums.apply(features, sample, windows, chunk_size, fixed_radius, directions, metric)
        mean = total[..., :-1] / total[..., -1:].clamp_min(1e-8)
        summaries[selected] = torch.where(count[..., None] > 0, mean, 0.0)
        counts[selected] = count
    return summaries, counts


class PrefixConeAttention(nn.Module):
    def __init__(
        self, token_dim: int, num_heads: int, windows: tuple[ConeWindowConfig, ...], dropout: float,
        *, directions: tuple[int, ...] = (-1, 1), learned_pool_weights: bool = True,
    ) -> None:
        super().__init__()
        self.windows = windows
        self.fixed_radius = False
        self.directions = directions
        self.pool_value = nn.Linear(token_dim, token_dim)
        self.pool_weight = nn.Linear(token_dim, 1) if learned_pool_weights else None
        self.shape_embedding = nn.Parameter(torch.empty(len(windows), token_dim))
        self.direction_embedding = nn.Parameter(torch.empty(len(directions), token_dim))
        nn.init.normal_(self.shape_embedding, std=0.02)
        nn.init.normal_(self.direction_embedding, std=0.02)
        self.center_attention = nn.MultiheadAttention(
            token_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, tokens, token_coords, token_batch_indices, *, geometry=None):
        if len(tokens) == 0:
            return ConeAttentionOutput(tokens)
        if geometry is None:
            geometry = prepare_prefix_geometry(token_coords, token_batch_indices)
        values = self.pool_value(tokens)
        # 无门控时每个成员权重严格为1，得到锥内算术均值。
        weights = (torch.sigmoid(self.pool_weight(tokens).float()) + 1e-4
                   if self.pool_weight is not None else tokens.new_ones((len(tokens), 1)))
        summaries, counts = prefix_cone_pool(
            values, weights, geometry, self.windows, fixed_radius=self.fixed_radius,
            directions=self.directions,
        )
        embeddings = (self.shape_embedding[:, None] + self.direction_embedding[None]).flatten(0, 1)
        summaries = summaries.to(tokens.dtype) + embeddings
        available = counts > 0
        has_neighbors = available.any(dim=1)
        # 全空时开放一个占位槽避免 softmax NaN，投影后再显式归零。
        mask = ~available
        mask = mask.clone()
        mask[~has_neighbors, 0] = False
        output, _ = self.center_attention(
            tokens[:, None], summaries, summaries, key_padding_mask=mask, need_weights=False
        )
        output = torch.where(has_neighbors[:, None], output[:, 0], 0.0)
        return ConeAttentionOutput(output)


class PrefixConeEncoder(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        windows: tuple[ConeWindowConfig, ...],
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConeTransformerBlock(
                    token_dim=token_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention=PrefixConeAttention(token_dim, num_heads, windows, dropout),
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(token_dim)

    def forward(self, tokens, token_coords, token_batch_indices):
        geometry = prepare_prefix_geometry(token_coords, token_batch_indices)
        for layer in self.layers:
            tokens = layer(tokens, token_coords, token_batch_indices, geometry=geometry)
        return ConeTransformerOutput(self.final_norm(tokens))
