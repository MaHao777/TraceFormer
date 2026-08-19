"""Multi-head center attention over one fixed local spatiotemporal cone."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from huav.config import LocalConeConfig

from .cone_neighbor_builder import ConeNeighborBuilder
from .relative_position import RelativePositionEncoder

AttentionStats = dict[str, torch.Tensor]


@dataclass(frozen=True)
class LocalConeAttentionOutput:
    features: torch.Tensor
    past_attention_stats: AttentionStats | None
    future_attention_stats: AttentionStats | None


class LocalConeAttention(nn.Module):
    """Update every center token from the same fixed ``t-1/t/t+1`` geometry."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int,
        local_cone: LocalConeConfig,
        dropout: float = 0.1,
        causal_mode: bool = False,
        direction_dropout: float = 0.0,
        debug_attention_stats: bool = False,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or num_heads <= 0 or token_dim % num_heads:
            raise ValueError("token_dim must be positive and divisible by num_heads")
        if not 0 <= dropout < 1 or not 0 <= direction_dropout <= 1:
            raise ValueError("dropout values are outside their valid ranges")
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.causal_mode = bool(causal_mode)
        self.direction_dropout = float(direction_dropout)
        self.debug_attention_stats = bool(debug_attention_stats)
        self.neighbor_builder = ConeNeighborBuilder(
            base_radius=local_cone.base_radius,
            slope=local_cone.slope,
            max_neighbors_per_time_slice=(
                local_cone.max_neighbors_per_time_slice
            ),
            causal_mode=causal_mode,
        )

        self.q_proj = nn.Linear(self.token_dim, self.token_dim)
        self.k_proj = nn.Linear(self.token_dim, self.token_dim)
        self.v_proj = nn.Linear(self.token_dim, self.token_dim)
        self.relative_position = RelativePositionEncoder(
            hidden_dim=max(8, self.head_dim * 2)
        )
        self.out_proj = nn.Linear(self.token_dim, self.token_dim)
        self.attention_dropout = nn.Dropout(dropout)

    def _reshape_heads(self, features: torch.Tensor) -> torch.Tensor:
        return features.reshape(len(features), self.num_heads, self.head_dim)

    def _apply_direction_dropout(
        self,
        mask: torch.Tensor,
        time_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or self.direction_dropout <= 0 or self.causal_mode:
            return mask
        draw = float(torch.rand((), device=mask.device).item())
        if draw < self.direction_dropout * 0.5:
            return mask & (time_offsets <= 0)
        if draw < self.direction_dropout:
            return mask & (time_offsets >= 0)
        return mask

    def _stats(
        self,
        counts: torch.Tensor,
        attention_mass: torch.Tensor,
    ) -> AttentionStats:
        return {
            "neighbor_counts": counts.detach(),
            "edge_count": counts.sum().detach(),
            "mean_neighbors": (
                counts.float().mean().detach()
                if counts.numel()
                else counts.new_zeros((), dtype=torch.float32)
            ),
            "max_neighbors": (
                counts.max().detach() if counts.numel() else counts.new_zeros(())
            ),
            "gate_weights": attention_mass.detach(),
        }

    def forward(
        self,
        tokens: torch.Tensor,
        token_coords: torch.Tensor,
        token_batch_indices: torch.Tensor,
    ) -> LocalConeAttentionOutput:
        if tokens.ndim != 2 or tokens.shape[1] != self.token_dim:
            raise ValueError(f"tokens must have shape [M,{self.token_dim}]")
        if len(tokens) != len(token_coords):
            raise ValueError("tokens and token_coords must have equal length")
        if len(tokens) == 0:
            empty_counts = torch.empty(
                (0, self.num_heads), dtype=torch.long, device=tokens.device
            )
            empty_mass = torch.empty((0,), dtype=tokens.dtype, device=tokens.device)
            stats = (
                self._stats(empty_counts, empty_mass)
                if self.debug_attention_stats
                else None
            )
            return LocalConeAttentionOutput(tokens, stats, stats)

        neighbor_indices = self.neighbor_builder(
            token_coords, token_batch_indices
        ).indices
        valid_mask = neighbor_indices >= 0
        safe_indices = neighbor_indices.clamp_min(0)
        neighbor_coords = token_coords[safe_indices].to(dtype=tokens.dtype)
        center_coords = token_coords.to(dtype=tokens.dtype)[:, None, :]
        time_offsets = neighbor_coords[..., 2] - center_coords[..., 2]
        valid_mask = self._apply_direction_dropout(valid_mask, time_offsets)

        query = self._reshape_heads(self.q_proj(tokens))
        key = self._reshape_heads(self.k_proj(tokens))
        value = self._reshape_heads(self.v_proj(tokens))
        neighbor_keys = key[safe_indices]
        neighbor_values = value[safe_indices]
        relative_features = RelativePositionEncoder.build_features(
            center_coords, neighbor_coords
        )
        relative_bias = self.relative_position(relative_features)
        scores = (
            torch.einsum("mhd,mshd->mhs", query, neighbor_keys)
            / math.sqrt(self.head_dim)
            + relative_bias[:, None, :]
        )
        head_mask = valid_mask[:, None, :]
        scores = scores.masked_fill(~head_mask, -torch.inf)
        has_neighbors = head_mask.any(dim=-1, keepdim=True)
        scores = torch.where(has_neighbors, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(head_mask, weights, torch.zeros_like(weights))
        weights = self.attention_dropout(weights)
        attended = torch.einsum("mhs,mshd->mhd", weights, neighbor_values)
        features = self.out_proj(attended.reshape(len(tokens), self.token_dim))

        past_mask = valid_mask & (time_offsets < 0)
        future_mask = valid_mask & (time_offsets > 0)
        past_counts = past_mask.sum(dim=-1, keepdim=True).expand(
            -1, self.num_heads
        )
        future_counts = future_mask.sum(dim=-1, keepdim=True).expand(
            -1, self.num_heads
        )
        past_mass = (
            weights * past_mask[:, None, :].to(dtype=weights.dtype)
        ).sum(dim=-1).mean(dim=1)
        future_mass = (
            weights * future_mask[:, None, :].to(dtype=weights.dtype)
        ).sum(dim=-1).mean(dim=1)
        past_stats = (
            self._stats(past_counts, past_mass)
            if self.debug_attention_stats
            else None
        )
        future_stats = (
            self._stats(future_counts, future_mass)
            if self.debug_attention_stats
            else None
        )
        return LocalConeAttentionOutput(features, past_stats, future_stats)


__all__ = [
    "AttentionStats",
    "LocalConeAttention",
    "LocalConeAttentionOutput",
]
