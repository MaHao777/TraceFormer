"""Relative motion encoding for sparse cone-attention edges."""

from __future__ import annotations

import torch
from torch import nn


class RelativePositionEncoder(nn.Module):
    """Convert six geometric/motion components to an attention-score bias."""

    feature_dim = 6

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def build_features(
        center_coords: torch.Tensor,
        neighbor_coords: torch.Tensor,
    ) -> torch.Tensor:
        delta = neighbor_coords - center_coords
        delta_x, delta_y, delta_time = delta.unbind(dim=-1)
        spatial_distance = torch.maximum(delta_x.abs(), delta_y.abs())
        absolute_time = delta_time.abs()
        apparent_speed = spatial_distance / absolute_time.clamp_min(1.0)
        return torch.stack(
            (
                delta_x,
                delta_y,
                delta_time,
                spatial_distance,
                absolute_time,
                apparent_speed,
            ),
            dim=-1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError("relative position features must end with dimension 6")
        return self.network(features).squeeze(-1)


__all__ = ["RelativePositionEncoder"]
