"""Per-sample token importance scores and per-cone K/V selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TokenImportanceScores:
    """Normalized importance components and their weighted combination."""

    density: torch.Tensor
    attention: torch.Tensor
    track: torch.Tensor
    final: torch.Tensor


class TokenImportance(nn.Module):
    """Rank K/V candidates without removing any token from the Query stream."""

    def __init__(
        self,
        *,
        topk_ratio: float = 0.5,
        density_weight: float = 0.3,
        attention_weight: float = 0.3,
        track_weight: float = 0.4,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0 < topk_ratio <= 1:
            raise ValueError("topk_ratio must be in (0,1]")
        weights = (density_weight, attention_weight, track_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("importance weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one importance weight must be positive")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.topk_ratio = float(topk_ratio)
        weight_total = float(sum(weights))
        self.density_weight = float(density_weight) / weight_total
        self.attention_weight = float(attention_weight) / weight_total
        self.track_weight = float(track_weight) / weight_total
        self.epsilon = float(epsilon)

    @staticmethod
    def _validate_neighborhoods(
        *,
        token_count: int,
        past_indices: torch.Tensor,
        future_indices: torch.Tensor,
        past_attention_weights: torch.Tensor,
        future_attention_weights: torch.Tensor,
    ) -> None:
        if past_indices.ndim != 3 or past_indices.shape[0] != token_count:
            raise ValueError("past_indices must have shape [M,H,K]")
        if future_indices.shape != past_indices.shape:
            raise ValueError("future_indices must match past_indices")
        if past_attention_weights.shape != past_indices.shape:
            raise ValueError("past_attention_weights must match past_indices")
        if future_attention_weights.shape != future_indices.shape:
            raise ValueError("future_attention_weights must match future_indices")

    @staticmethod
    def _sample_minmax(
        values: torch.Tensor,
        token_batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        normalized = torch.zeros_like(values)
        for batch_index in torch.unique(token_batch_indices):
            mask = token_batch_indices == batch_index
            sample = values[mask]
            minimum = sample.min()
            span = sample.max() - minimum
            scaled = (sample - minimum) / span.clamp_min(
                torch.finfo(values.dtype).eps
            )
            normalized[mask] = torch.where(span > 0, scaled, torch.zeros_like(scaled))
        return normalized

    def _relative_density(
        self,
        event_counts: torch.Tensor,
        past_indices: torch.Tensor,
        future_indices: torch.Tensor,
    ) -> torch.Tensor:
        token_count = len(event_counts)
        joined = torch.cat(
            (past_indices.flatten(start_dim=1), future_indices.flatten(start_dim=1)),
            dim=1,
        )
        sorted_indices = joined.sort(dim=1, stable=True).values
        valid = sorted_indices >= 0
        unique = valid.clone()
        if sorted_indices.shape[1] > 1:
            unique[:, 1:] &= sorted_indices[:, 1:] != sorted_indices[:, :-1]
        safe_indices = sorted_indices.clamp_min(0)
        neighbor_counts = event_counts[safe_indices].to(dtype=torch.float32)
        unique_float = unique.to(neighbor_counts.dtype)
        neighbor_total = (neighbor_counts * unique_float).sum(dim=1)
        neighborhood_size = unique_float.sum(dim=1)
        neighbor_mean = neighbor_total / neighborhood_size.clamp_min(1)
        current_counts = event_counts.to(dtype=torch.float32)
        relative = torch.log1p(
            current_counts / (neighbor_mean + self.epsilon)
        )
        return torch.where(
            neighborhood_size > 0,
            relative,
            torch.zeros(token_count, dtype=relative.dtype, device=relative.device),
        )

    @staticmethod
    def _incoming_attention(
        token_count: int,
        directions: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> torch.Tensor:
        first_weights = directions[0][1]
        totals = torch.zeros(
            token_count, dtype=first_weights.dtype, device=first_weights.device
        )
        counts = torch.zeros_like(totals)
        for indices, weights in directions:
            flat_indices = indices.reshape(-1)
            flat_weights = weights.reshape(-1)
            valid = flat_indices >= 0
            totals.scatter_add_(0, flat_indices[valid], flat_weights[valid])
            counts.scatter_add_(
                0,
                flat_indices[valid],
                torch.ones_like(flat_weights[valid]),
            )
        return totals / counts.clamp_min(1)

    @staticmethod
    def _direction_track_score(
        tokens: torch.Tensor,
        token_coords: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_indices = neighbor_indices.flatten(start_dim=1)
        valid = flat_indices >= 0
        safe_indices = flat_indices.clamp_min(0)
        candidate_features = tokens[safe_indices]
        cosine = F.cosine_similarity(
            tokens[:, None, :], candidate_features, dim=-1, eps=1e-8
        )
        feature_match = ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)
        candidate_xy = token_coords[safe_indices, :2].to(dtype=tokens.dtype)
        current_xy = token_coords[:, None, :2].to(dtype=tokens.dtype)
        spatial_distance = (candidate_xy - current_xy).abs().amax(dim=-1)
        spatial_match = 1.0 / (1.0 + spatial_distance)
        match = 0.5 * feature_match + 0.5 * spatial_match
        match = match.masked_fill(~valid, -torch.inf)
        available = valid.any(dim=1)
        best = match.max(dim=1).values
        return torch.where(available, best, torch.zeros_like(best)), available

    def _track_continuity(
        self,
        tokens: torch.Tensor,
        token_coords: torch.Tensor,
        past_indices: torch.Tensor,
        future_indices: torch.Tensor,
    ) -> torch.Tensor:
        past, past_available = self._direction_track_score(
            tokens, token_coords, past_indices
        )
        future, future_available = self._direction_track_score(
            tokens, token_coords, future_indices
        )
        available_count = past_available.to(tokens.dtype) + future_available.to(
            tokens.dtype
        )
        return (past + future) / available_count.clamp_min(1)

    def forward(
        self,
        *,
        tokens: torch.Tensor,
        event_counts: torch.Tensor,
        token_coords: torch.Tensor,
        token_batch_indices: torch.Tensor,
        past_indices: torch.Tensor,
        future_indices: torch.Tensor,
        past_attention_weights: torch.Tensor,
        future_attention_weights: torch.Tensor,
    ) -> TokenImportanceScores:
        token_count = len(tokens)
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [M,D]")
        if event_counts.shape != (token_count,):
            raise ValueError("event_counts must have shape [M]")
        if token_coords.shape != (token_count, 3):
            raise ValueError("token_coords must have shape [M,3]")
        if token_batch_indices.shape != (token_count,):
            raise ValueError("token_batch_indices must have shape [M]")
        self._validate_neighborhoods(
            token_count=token_count,
            past_indices=past_indices,
            future_indices=future_indices,
            past_attention_weights=past_attention_weights,
            future_attention_weights=future_attention_weights,
        )
        if token_count == 0:
            empty = tokens.new_empty((0,))
            return TokenImportanceScores(empty, empty, empty, empty)

        density = self._sample_minmax(
            self._relative_density(event_counts, past_indices, future_indices).to(
                dtype=tokens.dtype
            ),
            token_batch_indices,
        )
        attention = self._sample_minmax(
            self._incoming_attention(
                token_count,
                (
                    (past_indices, past_attention_weights),
                    (future_indices, future_attention_weights),
                ),
            ),
            token_batch_indices,
        )
        track = self._sample_minmax(
            self._track_continuity(
                tokens, token_coords, past_indices, future_indices
            ),
            token_batch_indices,
        )
        final = (
            self.density_weight * density
            + self.attention_weight * attention
            + self.track_weight * track
        )
        return TokenImportanceScores(density, attention, track, final)

    def select_topk(
        self,
        neighbor_indices: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        """Mask all but the highest-importance fraction in each original cone."""
        if neighbor_indices.ndim != 3:
            raise ValueError("neighbor_indices must have shape [M,H,K]")
        if importance.ndim != 1:
            raise ValueError("importance must have shape [M]")
        if neighbor_indices.numel() == 0:
            return neighbor_indices.clone()
        valid = neighbor_indices >= 0
        safe_indices = neighbor_indices.clamp_min(0)
        candidate_scores = importance[safe_indices].masked_fill(~valid, -torch.inf)
        order = torch.argsort(
            candidate_scores, dim=-1, descending=True, stable=True
        )
        ranks = torch.empty_like(order)
        rank_values = torch.arange(
            neighbor_indices.shape[-1], device=neighbor_indices.device
        ).view(1, 1, -1)
        ranks.scatter_(-1, order, rank_values.expand_as(order))
        valid_counts = valid.sum(dim=-1)
        keep_counts = torch.ceil(valid_counts * self.topk_ratio).to(torch.long)
        keep = valid & (ranks < keep_counts[..., None])
        return torch.where(keep, neighbor_indices, -torch.ones_like(neighbor_indices))


__all__ = ["TokenImportance", "TokenImportanceScores"]
