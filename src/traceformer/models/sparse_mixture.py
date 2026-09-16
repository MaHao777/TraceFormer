"""Token-aligned residual mixture of full and high-threshold event views."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SparseMixtureOutput:
    token_features: torch.Tensor
    gate: torch.Tensor
    sparse_present: torch.Tensor


class SparseTokenMixture(nn.Module):
    """Align sparse tokens to full tokens and apply a conservative residual."""

    statistic_dim = 4

    def __init__(
        self,
        *,
        token_dim: int,
        num_x_blocks: int,
        num_y_blocks: int,
        num_time_blocks: int,
        initial_router_bias: float,
        initial_residual_scale: float,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_x_blocks = int(num_x_blocks)
        self.num_y_blocks = int(num_y_blocks)
        self.num_time_blocks = int(num_time_blocks)
        input_dim = self.token_dim * 3 + self.statistic_dim
        self.adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.router = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1),
        )
        nn.init.constant_(self.router[-1].bias, float(initial_router_bias))
        self.residual_scale = nn.Parameter(
            torch.tensor(float(initial_residual_scale))
        )

    def _keys(
        self,
        coordinates: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("token coordinates must have shape [T,3]")
        if batch_indices.ndim != 1 or len(batch_indices) != len(coordinates):
            raise ValueError("token batch indices must have shape [T]")
        x, y, time = coordinates.long().unbind(dim=1)
        if len(coordinates) and (
            bool(torch.any(x < 0))
            or bool(torch.any(x >= self.num_x_blocks))
            or bool(torch.any(y < 0))
            or bool(torch.any(y >= self.num_y_blocks))
            or bool(torch.any(time < 0))
            or bool(torch.any(time >= self.num_time_blocks))
            or bool(torch.any(batch_indices < 0))
        ):
            raise ValueError("token coordinates are outside mixture bounds")
        cells_per_sample = (
            self.num_x_blocks * self.num_y_blocks * self.num_time_blocks
        )
        spatial = (
            (time * self.num_y_blocks + y) * self.num_x_blocks + x
        )
        return batch_indices.long() * cells_per_sample + spatial

    def forward(
        self,
        full_features: torch.Tensor,
        full_coordinates: torch.Tensor,
        full_batch_indices: torch.Tensor,
        full_statistics: torch.Tensor,
        sparse_features: torch.Tensor,
        sparse_coordinates: torch.Tensor,
        sparse_batch_indices: torch.Tensor,
        sparse_statistics: torch.Tensor,
        *,
        mixture_weight: float,
    ) -> SparseMixtureOutput:
        full_keys = self._keys(full_coordinates, full_batch_indices)
        sparse_keys = self._keys(sparse_coordinates, sparse_batch_indices)
        if len(full_keys) > 1 and bool(torch.any(full_keys[1:] <= full_keys[:-1])):
            raise ValueError("full token keys must be strictly increasing")

        sparse_present = torch.zeros(
            len(full_features), dtype=torch.bool, device=full_features.device
        )
        aligned_sparse = torch.zeros_like(full_features)
        aligned_sparse_statistics = full_statistics.new_zeros(
            (len(full_features), sparse_statistics.shape[1])
        )
        if len(sparse_features):
            positions = torch.searchsorted(full_keys, sparse_keys)
            safe_positions = positions.clamp(max=max(0, len(full_keys) - 1))
            if len(full_keys) == 0 or bool(
                torch.any(positions >= len(full_keys))
            ) or bool(torch.any(full_keys[safe_positions] != sparse_keys)):
                raise ValueError("sparse token is absent from the full view")
            if len(positions) > 1 and len(torch.unique(positions)) != len(positions):
                raise ValueError("sparse token keys must be unique")
            aligned_sparse = aligned_sparse.index_copy(
                0, positions, sparse_features
            )
            aligned_sparse_statistics = aligned_sparse_statistics.index_copy(
                0, positions, sparse_statistics
            )
            sparse_present[positions] = True

        full_log_count = full_statistics[:, :1]
        sparse_log_count = aligned_sparse_statistics[:, :1]
        full_count = torch.expm1(full_log_count).clamp_min(0)
        sparse_count = torch.expm1(sparse_log_count).clamp_min(0)
        retention = sparse_count / full_count.clamp_min(1)
        presence_feature = sparse_present.to(full_features.dtype)[:, None]
        mixture_input = torch.cat(
            (
                full_features,
                aligned_sparse,
                full_features - aligned_sparse,
                full_log_count,
                sparse_log_count,
                retention,
                presence_feature,
            ),
            dim=1,
        )
        gate = torch.sigmoid(self.router(mixture_input)).squeeze(-1)
        gate = (
            gate
            * sparse_present.to(gate.dtype)
            * float(mixture_weight)
        )
        residual = self.adapter(mixture_input)
        token_features = (
            full_features
            + self.residual_scale * gate[:, None] * residual
        )
        return SparseMixtureOutput(
            token_features=token_features,
            gate=gate,
            sparse_present=sparse_present,
        )


__all__ = ["SparseMixtureOutput", "SparseTokenMixture"]
