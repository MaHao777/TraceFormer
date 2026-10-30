"""注意力消融的固定窗口与逐 Token 等容量替代，不进入正式配置。"""

from __future__ import annotations

import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from traceformer.config import ConeWindowConfig
from traceformer.models.cone_transformer import ConeAttentionOutput, ConeTransformerOutput
from traceformer.models.prefix_cone import PrefixConeEncoder, prepare_prefix_geometry


class SlidingLocalAttention(nn.Module):
    """A0：完整滑动窗口内逐 Token 注意力，分块重算控制训练内存。"""

    def __init__(self, original: nn.Module, chunk_size: int = 128) -> None:
        super().__init__()
        self.windows = original.windows
        self.fixed_radius = True
        self.chunk_size = chunk_size
        self.cpu_backend = os.environ.get("TRACEFORMER_LOCAL_CPU_BACKEND", "torch")
        if self.cpu_backend not in {"torch", "fused"}:
            raise ValueError("unknown local CPU attention backend")
        self.center_attention = original.center_attention
        self.shape_embedding = original.shape_embedding
        self.direction_embedding = original.direction_embedding
        window = self.windows[0]
        offsets = [(x, y, direction * distance, side)
                   for side, direction in enumerate((-1, 1))
                   for distance in range(1, window.time_horizon + 1)
                   for y in range(-window.spatial_radius, window.spatial_radius + 1)
                   for x in range(-window.spatial_radius, window.spatial_radius + 1)]
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)

    def _attend_chunk(self, q, k, v, key_embedding, value_embedding, coords, lookup,
                      shape, start, end):
        times, height, width = shape
        xyz = coords[start:end, None] + self.offsets[None, :, :3]
        inside = ((xyz >= 0) & (xyz < xyz.new_tensor([width, height, times]))).all(-1)
        flat = ((xyz[..., 2].clamp(0, times - 1) * height
                 + xyz[..., 1].clamp(0, height - 1)) * width
                + xyz[..., 0].clamp(0, width - 1))
        neighbors = lookup[flat]
        valid = inside & (neighbors >= 0)
        has_neighbors = valid.any(-1)
        neighbors = neighbors.clamp_min(0)
        side = self.offsets[:, 3]
        # 方向/尺度编码在投影后相加，与先加编码再线性映射严格等价。
        keys = k[neighbors] + key_embedding[side]
        values = v[neighbors] + value_embedding[side]
        heads = self.center_attention.num_heads
        head_dim = q.shape[-1] // heads
        query = q[start:end].reshape(-1, heads, 1, head_dim)
        keys = keys.reshape(len(query), -1, heads, head_dim).transpose(1, 2)
        values = values.reshape(len(query), -1, heads, head_dim).transpose(1, 2)
        # 全空行只开放一个占位，最后归零，避免无效 softmax 或偏置泄漏。
        valid = valid.clone()
        valid[~has_neighbors, 0] = True
        output = F.scaled_dot_product_attention(
            query, keys, values, attn_mask=valid[:, None, None],
            dropout_p=self.center_attention.dropout if self.training else 0.0,
        ).reshape(len(query), -1)
        return torch.where(has_neighbors[:, None], output, 0.0), has_neighbors

    def forward(self, tokens, token_coords, token_batch_indices, *, geometry=None):
        if not len(tokens):
            return ConeAttentionOutput(tokens)
        if geometry is None:
            geometry = prepare_prefix_geometry(token_coords, token_batch_indices)
        attention = self.center_attention
        q, k, v = F.linear(tokens, attention.in_proj_weight, attention.in_proj_bias).chunk(3, -1)
        embeddings = self.shape_embedding + self.direction_embedding
        key_weight, value_weight = attention.in_proj_weight.chunk(3)[1:]
        key_embedding = F.linear(embeddings, key_weight)
        value_embedding = F.linear(embeddings, value_weight)
        features = torch.zeros_like(tokens)
        available = torch.zeros(len(tokens), dtype=torch.bool, device=tokens.device)
        for sample in geometry:
            indices = sample.indices
            lookup = torch.full((sample.shape[0] * sample.shape[1] * sample.shape[2],), -1,
                                dtype=torch.long, device=tokens.device)
            lookup[sample.flat_indices] = torch.arange(len(indices), device=tokens.device)
            arguments = (q[indices], k[indices], v[indices], key_embedding, value_embedding,
                         sample.coords, lookup, sample.shape)
            if self.cpu_backend == "fused" and tokens.device.type == "cpu":
                from traceformer.experiments.local_cpu import local_cpu_attention
                result, mask = local_cpu_attention(
                    *arguments, self.offsets, attention.num_heads,
                    attention.dropout if self.training else 0.0, self.chunk_size,
                )
                features[indices], available[indices] = result, mask
                continue
            chunks, masks = [], []
            for start in range(0, len(indices), self.chunk_size):
                args = (*arguments, start, min(start + self.chunk_size, len(indices)))
                # 不保留 O(M*K*C) 邻居特征；反向逐块重算并恢复该块 dropout RNG。
                result = (checkpoint(self._attend_chunk, *args, use_reentrant=False)
                          if torch.is_grad_enabled() else self._attend_chunk(*args))
                chunks.append(result[0])
                masks.append(result[1])
            features[indices] = torch.cat(chunks)
            available[indices] = torch.cat(masks)
        projected = attention.out_proj(features)
        # 空邻域在输出投影后仍严格归零。
        return ConeAttentionOutput(torch.where(available[:, None], projected, 0.0))


def matched_box_windows(windows: tuple[ConeWindowConfig, ...]) -> tuple[ConeWindowConfig, ...]:
    """选择离散截面积最接近锥平均截面积的整数半径，平局取较小值。"""
    result = []
    for window in windows:
        volume = sum(
            (2 * (window.spatial_radius * d // window.time_horizon) + 1) ** 2
            for d in range(1, window.time_horizon + 1)
        )
        radius = min(range(window.spatial_radius + 1), key=lambda r: abs(
            window.time_horizon * (2 * r + 1) ** 2 - volume
        ))
        result.append(ConeWindowConfig(radius, window.time_horizon))
    return tuple(result)


class PointwiseInteraction(nn.Module):
    def __init__(self, token_dim: int, target_parameters: int, dropout: float) -> None:
        super().__init__()
        # 两层线性层共有 (2D+1)H+D 个参数，选择最接近原注意力容量的宽度。
        hidden = max(1, round((target_parameters - token_dim) / (2 * token_dim + 1)))
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, token_dim),
        )

    def forward(self, tokens, token_coords, token_batch_indices, *, geometry=None):
        return ConeAttentionOutput(self.mlp(tokens))


class PointwiseEncoder(nn.Module):
    def __init__(self, original: PrefixConeEncoder, dropout: float) -> None:
        super().__init__()
        self.layers = original.layers
        self.final_norm = original.final_norm
        # 替代分支初始化不推进训练 RNG；公共权重与其他组完全一致。
        with torch.random.fork_rng(devices=[]):
            for layer in self.layers:
                dim = layer.attention_norm.normalized_shape[0]
                count = sum(p.numel() for p in layer.attention.parameters())
                layer.attention = PointwiseInteraction(dim, count, dropout)

    def forward(self, tokens, token_coords, token_batch_indices):
        # 逐 Token 网络不创建空间网格，也不读取其他 Token 特征。
        for layer in self.layers:
            tokens = layer(tokens, token_coords, token_batch_indices)
        return ConeTransformerOutput(self.final_norm(tokens))


def neighbor_count_summary(counts: torch.Tensor, windows: tuple[ConeWindowConfig, ...]) -> list:
    """按尺度和方向汇总真实占用数，不用理论网格体积代替实际邻居。"""
    result = []
    for index, window in enumerate(windows):
        for side, direction in enumerate(("past", "future")):
            values = counts[:, 2 * index + side].float()
            quantiles = torch.quantile(values, values.new_tensor([0.5, 0.95]))
            result.append({
                "scale": index, "direction": direction,
                "spatial_radius": window.spatial_radius, "time_horizon": window.time_horizon,
                "tokens": len(values), "mean": values.mean().item(),
                "p50": quantiles[0].item(), "p95": quantiles[1].item(),
                "empty_fraction": (values == 0).float().mean().item(),
            })
    return result
