"""Bidirectional multi-head center attention over fixed sparse cones."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from traceformer.config import ConeHeadConfig

from .cone_neighbor_builder import ConeNeighborBuilder
from .relative_position import RelativePositionEncoder


@dataclass(frozen=True)
class ConeAttentionOutput:
    features: torch.Tensor


class BidirectionalConeAttention(nn.Module):
    """Use center queries and separate past/future neighbor softmax operations."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int,
        cone_heads: tuple[ConeHeadConfig, ...],
        max_neighbors_per_cone: int,
        layer_time_scale: float = 1.0,
        dropout: float = 0.1,
        direction_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.direction_dropout = float(direction_dropout)
        self.neighbor_builder = ConeNeighborBuilder(
            cone_heads=cone_heads,
            max_neighbors_per_cone=max_neighbors_per_cone,
            layer_time_scale=layer_time_scale,
        )

        self.query_projection = nn.Linear(self.token_dim, self.token_dim)
        self.key_projection = nn.Linear(self.token_dim, self.token_dim)
        self.value_projection = nn.Linear(self.token_dim, self.token_dim)
        self.relative_position = RelativePositionEncoder(
            hidden_dim=max(8, self.head_dim * 2)
        )
        self.direction_embeddings = nn.Parameter(
            torch.empty(2, self.num_heads, self.head_dim)
        )
        nn.init.normal_(self.direction_embeddings, std=0.02)
        self.direction_gate = nn.Linear(self.token_dim, 2)
        self.output_projection = nn.Linear(self.token_dim, self.token_dim)
        self.attention_dropout = nn.Dropout(dropout)

    def _reshape_heads(self, features: torch.Tensor) -> torch.Tensor:
        return features.reshape(len(features), self.num_heads, self.head_dim)

    def _attend_direction(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_coords: torch.Tensor,
        neighbor_indices: torch.Tensor,
        direction_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = neighbor_indices >= 0
        safe_indices = neighbor_indices.clamp_min(0)
        head_indices = torch.arange(
            self.num_heads, device=query.device
        ).view(1, self.num_heads, 1)
        neighbor_keys = key[safe_indices, head_indices]
        neighbor_values = value[safe_indices, head_indices]
        neighbor_keys = neighbor_keys + self.direction_embeddings[
            direction_index
        ].view(1, self.num_heads, 1, self.head_dim)

        neighbor_coords = token_coords[safe_indices].to(dtype=query.dtype)
        center_coords = token_coords.to(dtype=query.dtype)[:, None, None, :]
        relative_features = RelativePositionEncoder.build_features(
            center_coords, neighbor_coords
        )
        relative_bias = self.relative_position(relative_features)
        scores = (
            (query[:, :, None, :] * neighbor_keys).sum(dim=-1)
            / math.sqrt(self.head_dim)
            + relative_bias
        )
        scores = scores.masked_fill(~mask, -torch.inf)
        has_neighbors = mask.any(dim=-1, keepdim=True)
        scores = torch.where(has_neighbors, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights = self.attention_dropout(weights)
        aggregate = (weights[..., None] * neighbor_values).sum(dim=-2)
        return aggregate, mask.sum(dim=-1)

    def _apply_direction_dropout(
        self,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or self.direction_dropout <= 0:
            return availability
        draw = float(torch.rand((), device=availability.device).item())
        if draw < self.direction_dropout * 0.5:
            availability = availability.clone()
            availability[:, 1] = False
        elif draw < self.direction_dropout:
            availability = availability.clone()
            availability[:, 0] = False
        return availability

    def forward(
        self,
        tokens: torch.Tensor,
        token_coords: torch.Tensor,
        token_batch_indices: torch.Tensor,
    ) -> ConeAttentionOutput:
        if len(tokens) == 0:
            return ConeAttentionOutput(tokens)

        neighborhoods = self.neighbor_builder(token_coords, token_batch_indices)
        query = self._reshape_heads(self.query_projection(tokens))
        key = self._reshape_heads(self.key_projection(tokens))
        value = self._reshape_heads(self.value_projection(tokens))
        past, past_counts = self._attend_direction(
            query=query,
            key=key,
            value=value,
            token_coords=token_coords,
            neighbor_indices=neighborhoods.past_indices,
            direction_index=0,
        )
        future, future_counts = self._attend_direction(
            query=query,
            key=key,
            value=value,
            token_coords=token_coords,
            neighbor_indices=neighborhoods.future_indices,
            direction_index=1,
        )
        past_flat = past.reshape(len(tokens), self.token_dim)
        future_flat = future.reshape(len(tokens), self.token_dim)
        availability = torch.stack(
            (
                past_counts.sum(dim=1) > 0,
                future_counts.sum(dim=1) > 0,
            ),
            dim=1,
        )
        availability = self._apply_direction_dropout(availability)
        raw_gates = torch.softmax(self.direction_gate(tokens), dim=-1)
        gates = raw_gates * availability.to(raw_gates.dtype)
        gate_total = gates.sum(dim=-1, keepdim=True)
        gates = torch.where(
            gate_total > 0,
            gates / gate_total.clamp_min(torch.finfo(gates.dtype).eps),
            torch.zeros_like(gates),
        )
        fused = gates[:, :1] * past_flat + gates[:, 1:] * future_flat
        features = self.output_projection(fused)
        return ConeAttentionOutput(features)


__all__ = [
    "BidirectionalConeAttention",
    "ConeAttentionOutput",
]
