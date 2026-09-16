"""Lightweight decoder from updated parent tokens to original-event logits."""

from __future__ import annotations

import torch
from torch import nn


class EventDecoder(nn.Module):
    event_feature_dim = 6

    def __init__(self, *, token_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(8, token_dim // 2)
        self.token_dim = int(token_dim)
        self.classifier = nn.Sequential(
            nn.Linear(
                self.token_dim + self.event_feature_dim,
                self.token_dim,
            ),
            nn.LayerNorm(self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        token_features: torch.Tensor,
        event_token_indices: torch.Tensor,
        event_local_features: torch.Tensor,
    ) -> torch.Tensor:
        if len(event_token_indices) == 0:
            return token_features.new_empty((0,))
        parent_tokens = token_features[event_token_indices]
        decoder_input = torch.cat(
            (
                parent_tokens,
                event_local_features,
            ),
            dim=1,
        )
        return self.classifier(decoder_input).squeeze(-1)


__all__ = ["EventDecoder"]
