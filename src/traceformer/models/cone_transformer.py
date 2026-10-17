"""Residual transformer blocks built from bidirectional sparse cone attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ConeAttentionOutput:
    features: torch.Tensor


@dataclass(frozen=True)
class ConeTransformerOutput:
    token_features: torch.Tensor


class ConeTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        mlp_ratio: float,
        dropout: float,
        attention: nn.Module,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(round(token_dim * mlp_ratio)))
        self.attention_norm = nn.LayerNorm(token_dim)
        self.attention = attention
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
        *,
        geometry: Any = None,
    ) -> torch.Tensor:
        attention = self.attention(
            self.attention_norm(tokens),
            token_coords,
            token_batch_indices,
            geometry=geometry,
        )
        tokens = tokens + self.residual_dropout(attention.features)
        tokens = tokens + self.residual_dropout(self.feed_forward(self.feed_forward_norm(tokens)))
        return tokens


__all__ = [
    "ConeTransformerBlock",
    "ConeAttentionOutput",
    "ConeTransformerOutput",
]
