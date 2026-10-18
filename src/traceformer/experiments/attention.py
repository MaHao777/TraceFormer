"""注意力消融的固定窗口与逐 Token 等容量替代，不进入正式配置。"""

from __future__ import annotations

import torch
from torch import nn

from traceformer.config import ConeWindowConfig
from traceformer.models.cone_transformer import ConeAttentionOutput, ConeTransformerOutput
from traceformer.models.prefix_cone import PrefixConeEncoder


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
