"""Residual transformer blocks built from bidirectional sparse cone attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from huav.config import ConeHeadConfig

from .cone_attention import AttentionStats, BidirectionalConeAttention


@dataclass(frozen=True)
class ConeTransformerOutput:
    token_features: torch.Tensor
    past_attention_stats: list[AttentionStats] | None
    future_attention_stats: list[AttentionStats] | None


class ConeTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int,
        cone_heads: tuple[ConeHeadConfig, ...],
        max_neighbors_per_cone: int,
        layer_time_scale: float,
        mlp_ratio: float,
        dropout: float,
        causal_mode: bool,
        direction_dropout: float,
        debug_attention_stats: bool,
        token_importance_enabled: bool,
        token_importance_topk_ratio: float,
        token_importance_density_weight: float,
        token_importance_attention_weight: float,
        token_importance_track_weight: float,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(round(token_dim * mlp_ratio)))
        self.attention_norm = nn.LayerNorm(token_dim)
        self.attention = BidirectionalConeAttention(
            token_dim=token_dim,
            num_heads=num_heads,
            cone_heads=cone_heads,
            max_neighbors_per_cone=max_neighbors_per_cone,
            layer_time_scale=layer_time_scale,
            dropout=dropout,
            causal_mode=causal_mode,
            direction_dropout=direction_dropout,
            debug_attention_stats=debug_attention_stats,
            token_importance_enabled=token_importance_enabled,
            token_importance_topk_ratio=token_importance_topk_ratio,
            token_importance_density_weight=token_importance_density_weight,
            token_importance_attention_weight=token_importance_attention_weight,
            token_importance_track_weight=token_importance_track_weight,
        )
        self.feed_forward_norm = nn.LayerNorm(token_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        token_coords: torch.Tensor,
        token_batch_indices: torch.Tensor,
        token_event_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, AttentionStats | None, AttentionStats | None]:
        attention = self.attention(
            self.attention_norm(tokens),
            token_coords,
            token_batch_indices,
            token_event_counts,
        )
        tokens = tokens + self.residual_dropout(attention.features)
        tokens = tokens + self.residual_dropout(
            self.feed_forward(self.feed_forward_norm(tokens))
        )
        return (
            tokens,
            attention.past_attention_stats,
            attention.future_attention_stats,
        )


class ConeTransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int,
        cone_heads: tuple[ConeHeadConfig, ...],
        layer_time_scales: tuple[float, ...],
        max_neighbors_per_cone: int,
        mlp_ratio: float,
        dropout: float,
        causal_mode: bool,
        direction_dropout: float,
        debug_attention_stats: bool,
        token_importance_enabled: bool = True,
        token_importance_topk_ratio: float = 0.75,
        token_importance_density_weight: float = 0.3,
        token_importance_attention_weight: float = 0.3,
        token_importance_track_weight: float = 0.4,
    ) -> None:
        super().__init__()
        if not layer_time_scales:
            raise ValueError("layer_time_scales must not be empty")
        self.debug_attention_stats = bool(debug_attention_stats)
        self.layers = nn.ModuleList(
            [
                ConeTransformerBlock(
                    token_dim=token_dim,
                    num_heads=num_heads,
                    cone_heads=cone_heads,
                    max_neighbors_per_cone=max_neighbors_per_cone,
                    layer_time_scale=scale,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    causal_mode=causal_mode,
                    direction_dropout=direction_dropout,
                    debug_attention_stats=debug_attention_stats,
                    token_importance_enabled=token_importance_enabled,
                    token_importance_topk_ratio=token_importance_topk_ratio,
                    token_importance_density_weight=(
                        token_importance_density_weight
                    ),
                    token_importance_attention_weight=(
                        token_importance_attention_weight
                    ),
                    token_importance_track_weight=token_importance_track_weight,
                )
                for scale in layer_time_scales
            ]
        )
        self.final_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        token_coords: torch.Tensor,
        token_batch_indices: torch.Tensor,
        token_event_counts: torch.Tensor,
    ) -> ConeTransformerOutput:
        past_stats: list[AttentionStats] = []
        future_stats: list[AttentionStats] = []
        for layer in self.layers:
            tokens, layer_past, layer_future = layer(
                tokens, token_coords, token_batch_indices, token_event_counts
            )
            if self.debug_attention_stats:
                if layer_past is None or layer_future is None:
                    raise RuntimeError("debug attention statistics were not returned")
                past_stats.append(layer_past)
                future_stats.append(layer_future)
        return ConeTransformerOutput(
            token_features=self.final_norm(tokens),
            past_attention_stats=past_stats if self.debug_attention_stats else None,
            future_attention_stats=future_stats if self.debug_attention_stats else None,
        )


__all__ = [
    "ConeTransformerBlock",
    "ConeTransformerEncoder",
    "ConeTransformerOutput",
]
