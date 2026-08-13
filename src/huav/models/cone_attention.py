"""Bidirectional multi-head center attention over fixed sparse cones."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from huav.config import ConeHeadConfig

from .cone_neighbor_builder import ConeNeighborBuilder
from .relative_position import RelativePositionEncoder
from .token_importance import TokenImportance, TokenImportanceScores

AttentionStats = dict[str, Any]


@dataclass(frozen=True)
class ConeAttentionOutput:
    features: torch.Tensor
    past_attention_stats: AttentionStats | None
    future_attention_stats: AttentionStats | None


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
        causal_mode: bool = False,
        direction_dropout: float = 0.0,
        debug_attention_stats: bool = False,
        token_importance_enabled: bool = True,
        token_importance_topk_ratio: float = 0.5,
        token_importance_density_weight: float = 0.3,
        token_importance_attention_weight: float = 0.3,
        token_importance_track_weight: float = 0.4,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or num_heads <= 0 or token_dim % num_heads:
            raise ValueError("token_dim must be positive and divisible by num_heads")
        if len(cone_heads) != num_heads:
            raise ValueError("cone_heads must contain one configuration per head")
        if not 0 <= dropout < 1 or not 0 <= direction_dropout <= 1:
            raise ValueError("dropout values are outside their valid ranges")
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.causal_mode = bool(causal_mode)
        self.direction_dropout = float(direction_dropout)
        self.debug_attention_stats = bool(debug_attention_stats)
        self.token_importance = (
            TokenImportance(
                topk_ratio=token_importance_topk_ratio,
                density_weight=token_importance_density_weight,
                attention_weight=token_importance_attention_weight,
                track_weight=token_importance_track_weight,
            )
            if token_importance_enabled
            else None
        )
        self.neighbor_builder = ConeNeighborBuilder(
            cone_heads=cone_heads,
            max_neighbors_per_cone=max_neighbors_per_cone,
            layer_time_scale=layer_time_scale,
            causal_mode=causal_mode,
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

    def _direction_weights(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
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
        return weights, mask.sum(dim=-1)

    def _aggregate_direction(
        self,
        *,
        value: torch.Tensor,
        neighbor_indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        safe_indices = neighbor_indices.clamp_min(0)
        head_indices = torch.arange(
            self.num_heads, device=value.device
        ).view(1, self.num_heads, 1)
        neighbor_values = value[safe_indices, head_indices]
        weights = self.attention_dropout(weights)
        aggregate = (weights[..., None] * neighbor_values).sum(dim=-2)
        return aggregate

    def _apply_direction_dropout(
        self,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or self.direction_dropout <= 0 or self.causal_mode:
            return availability
        draw = float(torch.rand((), device=availability.device).item())
        if draw < self.direction_dropout * 0.5:
            availability = availability.clone()
            availability[:, 1] = False
        elif draw < self.direction_dropout:
            availability = availability.clone()
            availability[:, 0] = False
        return availability

    def _stats(
        self,
        counts: torch.Tensor,
        gate_weights: torch.Tensor,
        *,
        original_counts: torch.Tensor | None = None,
        importance: TokenImportanceScores | None = None,
    ) -> AttentionStats:
        stats = {
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
            "gate_weights": gate_weights.detach(),
        }
        if original_counts is not None:
            stats["original_neighbor_counts"] = original_counts.detach()
            stats["original_edge_count"] = original_counts.sum().detach()
        if importance is not None:
            stats["importance"] = {
                "density": importance.density.detach(),
                "attention": importance.attention.detach(),
                "track": importance.track.detach(),
                "final": importance.final.detach(),
            }
        return stats

    def forward(
        self,
        tokens: torch.Tensor,
        token_coords: torch.Tensor,
        token_batch_indices: torch.Tensor,
        token_event_counts: torch.Tensor | None = None,
    ) -> ConeAttentionOutput:
        if tokens.ndim != 2 or tokens.shape[1] != self.token_dim:
            raise ValueError(f"tokens must have shape [M,{self.token_dim}]")
        if len(tokens) != len(token_coords):
            raise ValueError("tokens and token_coords must have equal length")
        if token_event_counts is None:
            token_event_counts = torch.ones(
                len(tokens), dtype=torch.long, device=tokens.device
            )
        if token_event_counts.shape != (len(tokens),):
            raise ValueError("token_event_counts must have shape [M]")
        if len(tokens) == 0:
            empty_counts = torch.empty(
                (0, self.num_heads), dtype=torch.long, device=tokens.device
            )
            empty_gates = torch.empty((0,), dtype=tokens.dtype, device=tokens.device)
            stats = (
                self._stats(empty_counts, empty_gates)
                if self.debug_attention_stats
                else None
            )
            return ConeAttentionOutput(tokens, stats, stats)

        neighborhoods = self.neighbor_builder(token_coords, token_batch_indices)
        query = self._reshape_heads(self.query_projection(tokens))
        key = self._reshape_heads(self.key_projection(tokens))
        value = self._reshape_heads(self.value_projection(tokens))
        full_past_weights, original_past_counts = self._direction_weights(
            query=query,
            key=key,
            token_coords=token_coords,
            neighbor_indices=neighborhoods.past_indices,
            direction_index=0,
        )
        full_future_weights, original_future_counts = self._direction_weights(
            query=query,
            key=key,
            token_coords=token_coords,
            neighbor_indices=neighborhoods.future_indices,
            direction_index=1,
        )
        importance: TokenImportanceScores | None = None
        past_indices = neighborhoods.past_indices
        future_indices = neighborhoods.future_indices
        if self.token_importance is not None:
            importance = self.token_importance(
                tokens=tokens,
                event_counts=token_event_counts,
                token_coords=token_coords,
                token_batch_indices=token_batch_indices,
                past_indices=past_indices,
                future_indices=future_indices,
                past_attention_weights=full_past_weights,
                future_attention_weights=full_future_weights,
            )
            past_indices = self.token_importance.select_topk(
                past_indices, importance.final
            )
            future_indices = self.token_importance.select_topk(
                future_indices, importance.final
            )
            past_weights, past_counts = self._direction_weights(
                query=query,
                key=key,
                token_coords=token_coords,
                neighbor_indices=past_indices,
                direction_index=0,
            )
            future_weights, future_counts = self._direction_weights(
                query=query,
                key=key,
                token_coords=token_coords,
                neighbor_indices=future_indices,
                direction_index=1,
            )
        else:
            past_weights, past_counts = full_past_weights, original_past_counts
            future_weights, future_counts = (
                full_future_weights,
                original_future_counts,
            )
        past = self._aggregate_direction(
            value=value,
            neighbor_indices=past_indices,
            weights=past_weights,
        )
        future = self._aggregate_direction(
            value=value,
            neighbor_indices=future_indices,
            weights=future_weights,
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
        if self.causal_mode:
            availability[:, 1] = False
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
        past_stats = (
            self._stats(
                past_counts,
                gates[:, 0],
                original_counts=original_past_counts,
                importance=importance,
            )
            if self.debug_attention_stats
            else None
        )
        future_stats = (
            self._stats(
                future_counts,
                gates[:, 1],
                original_counts=original_future_counts,
                importance=importance,
            )
            if self.debug_attention_stats
            else None
        )
        return ConeAttentionOutput(features, past_stats, future_stats)


__all__ = [
    "AttentionStats",
    "BidirectionalConeAttention",
    "ConeAttentionOutput",
]
