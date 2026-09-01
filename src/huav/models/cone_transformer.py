"""Residual transformer blocks built from bidirectional sparse cone attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from huav.config import ConeHeadConfig

from .cone_attention import BidirectionalConeAttention


@dataclass(frozen=True)
class ConeTransformerOutput:
    token_features: torch.Tensor


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
        direction_dropout: float,
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
            direction_dropout=direction_dropout,
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
    ) -> torch.Tensor:
        attention = self.attention(
            self.attention_norm(tokens),
            token_coords,
            token_batch_indices,
        )
        tokens = tokens + self.residual_dropout(attention.features)
        tokens = tokens + self.residual_dropout(
            self.feed_forward(self.feed_forward_norm(tokens))
        )
        return tokens


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
        direction_dropout: float,
    ) -> None:
        super().__init__()
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
                    direction_dropout=direction_dropout,
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
    ) -> ConeTransformerOutput:
        for layer in self.layers:
            tokens = layer(tokens, token_coords, token_batch_indices)
        return ConeTransformerOutput(token_features=self.final_norm(tokens))


__all__ = [
    "ConeTransformerBlock",
    "ConeTransformerEncoder",
    "ConeTransformerOutput",
]
