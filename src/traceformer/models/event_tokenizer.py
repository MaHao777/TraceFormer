"""Sparse event-to-block tokenization without a dense voxel volume."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TokenizedEvents:
    tokens: torch.Tensor
    token_coords: torch.Tensor
    token_batch_indices: torch.Tensor
    event_token_indices: torch.Tensor
    event_local_features: torch.Tensor
    event_xy: torch.Tensor
    normalized_time: torch.Tensor
    event_batch_indices: torch.Tensor
    token_statistics: torch.Tensor
    event_counts: torch.Tensor


class EventTokenizer(nn.Module):
    """Map occupied regular spacetime blocks to learned tokens."""

    event_feature_dim = 6
    statistic_dim = 7

    def __init__(
        self,
        *,
        width: int,
        height: int,
        spatial_block_size_x: int,
        spatial_block_size_y: int,
        num_time_blocks: int,
        event_embed_dim: int,
        token_dim: int,
        num_pool_queries: int = 4,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.spatial_block_size_x = int(spatial_block_size_x)
        self.spatial_block_size_y = int(spatial_block_size_y)
        self.num_time_blocks = int(num_time_blocks)
        self.event_embed_dim = int(event_embed_dim)
        self.token_dim = int(token_dim)
        self.num_pool_queries = int(num_pool_queries)
        self.num_x_blocks = math.ceil(self.width / self.spatial_block_size_x)
        self.num_y_blocks = math.ceil(self.height / self.spatial_block_size_y)

        self.event_encoder = nn.Sequential(
            nn.Linear(self.event_feature_dim, self.event_embed_dim),
            nn.LayerNorm(self.event_embed_dim),
            nn.GELU(),
            nn.Linear(self.event_embed_dim, self.event_embed_dim),
        )
        self.pool_queries = nn.Parameter(
            torch.empty(self.num_pool_queries, self.event_embed_dim)
        )
        nn.init.normal_(
            self.pool_queries,
            mean=0.0,
            std=1.0 / math.sqrt(self.event_embed_dim),
        )
        self.key_projection = nn.Linear(
            self.event_embed_dim,
            self.event_embed_dim,
            bias=False,
        )
        self.value_projection = nn.Linear(
            self.event_embed_dim,
            self.event_embed_dim,
            bias=False,
        )
        self.query_fusion = nn.Sequential(
            nn.Linear(
                self.num_pool_queries * self.event_embed_dim,
                self.event_embed_dim,
            ),
            nn.LayerNorm(self.event_embed_dim),
            nn.GELU(),
            nn.Linear(self.event_embed_dim, self.event_embed_dim),
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.event_embed_dim),
            nn.GELU(),
        )
        projection_input = self.event_embed_dim * 2 + self.statistic_dim
        self.token_projection = nn.Sequential(
            nn.Linear(projection_input, self.token_dim),
            nn.LayerNorm(self.token_dim),
        )

    def _attention_pool(
        self,
        encoded_events: torch.Tensor,
        inverse: torch.Tensor,
        *,
        token_count: int,
    ) -> torch.Tensor:
        """Pool variable-length event groups without padding."""
        keys = self.key_projection(encoded_events)
        values = self.value_projection(encoded_events)
        value_indices = inverse[:, None].expand(-1, self.event_embed_dim)
        scale = 1.0 / math.sqrt(self.event_embed_dim)
        pooled_queries = []
        for query in self.pool_queries:
            logits = torch.sum(keys * query, dim=1) * scale
            group_maximum = torch.full(
                (token_count,),
                -torch.inf,
                dtype=logits.dtype,
                device=logits.device,
            )
            group_maximum.scatter_reduce_(
                0,
                inverse,
                logits,
                reduce="amax",
                include_self=True,
            )
            unnormalized_weights = torch.exp(logits - group_maximum[inverse])
            normalizers = torch.zeros(
                token_count,
                dtype=logits.dtype,
                device=logits.device,
            )
            normalizers.scatter_add_(0, inverse, unnormalized_weights)
            weights = unnormalized_weights / normalizers[inverse].clamp_min(
                torch.finfo(logits.dtype).tiny
            )
            weighted_values = values * weights[:, None]
            pooled = weighted_values.new_zeros(
                (token_count, self.event_embed_dim)
            )
            pooled.scatter_add_(0, value_indices, weighted_values)
            pooled_queries.append(pooled)
        return torch.stack(pooled_queries, dim=1)

    def _split_input(
        self,
        events: torch.Tensor,
        batch_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if events.ndim != 2 or events.shape[1] != 4:
            raise ValueError("events must have shape [N,4]")
        if not events.is_floating_point():
            raise TypeError("events must have a floating dtype")
        parsed_batches = (
            torch.zeros(len(events), dtype=torch.long, device=events.device)
            if batch_indices is None
            else batch_indices.long()
        )
        if parsed_batches.ndim != 1 or len(parsed_batches) != len(events):
            raise ValueError("batch_indices must have shape [N]")
        if len(parsed_batches) and torch.any(parsed_batches < 0):
            raise ValueError("batch indices must be non-negative")
        return events, parsed_batches

    def _empty(self, events: torch.Tensor) -> TokenizedEvents:
        device = events.device
        dtype = events.dtype
        return TokenizedEvents(
            tokens=torch.empty((0, self.token_dim), dtype=dtype, device=device),
            token_coords=torch.empty((0, 3), dtype=torch.long, device=device),
            token_batch_indices=torch.empty((0,), dtype=torch.long, device=device),
            event_token_indices=torch.empty((0,), dtype=torch.long, device=device),
            event_local_features=torch.empty(
                (0, self.event_feature_dim), dtype=dtype, device=device
            ),
            event_xy=torch.empty((0, 2), dtype=events.dtype, device=events.device),
            normalized_time=torch.empty(0, dtype=events.dtype, device=events.device),
            event_batch_indices=torch.empty(
                0, dtype=torch.long, device=events.device
            ),
            token_statistics=torch.empty(
                (0, self.statistic_dim), dtype=dtype, device=device
            ),
            event_counts=torch.empty((0,), dtype=torch.long, device=device),
        )

    def forward(
        self,
        events: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
    ) -> TokenizedEvents:
        event_features, parsed_batches = self._split_input(events, batch_indices)
        if len(events) == 0:
            return self._empty(events)

        x, y, normalized_time, polarity = event_features.unbind(dim=1)
        x_index = torch.div(
            x.long(), self.spatial_block_size_x, rounding_mode="floor"
        )
        y_index = torch.div(
            y.long(), self.spatial_block_size_y, rounding_mode="floor"
        )
        scaled_time = normalized_time * self.num_time_blocks
        time_index = torch.clamp(scaled_time.long(), max=self.num_time_blocks - 1)
        sample_count = int(parsed_batches.max().item()) + 1
        sample_event_counts = torch.bincount(
            parsed_batches, minlength=sample_count
        ).to(events.dtype)
        positive_indicator = (polarity > 0).to(events.dtype)
        sample_polarity_sum = torch.zeros(
            sample_count, dtype=events.dtype, device=events.device
        )
        sample_polarity_sum.scatter_add_(0, parsed_batches, polarity)
        sample_polarity_mean = sample_polarity_sum / sample_event_counts.clamp_min(1)
        sample_positive_sum = torch.zeros_like(sample_polarity_sum)
        sample_positive_sum.scatter_add_(0, parsed_batches, positive_indicator)
        sample_positive_fraction = (
            sample_positive_sum / sample_event_counts.clamp_min(1)
        )
        event_sample_polarity_mean = sample_polarity_mean[parsed_batches]
        local_features = torch.stack(
            (
                (x - x_index * self.spatial_block_size_x)
                / self.spatial_block_size_x,
                (y - y_index * self.spatial_block_size_y)
                / self.spatial_block_size_y,
                scaled_time - time_index,
                polarity,
                polarity - event_sample_polarity_mean,
                event_sample_polarity_mean,
            ),
            dim=1,
        )

        cells_per_sample = (
            self.num_x_blocks * self.num_y_blocks * self.num_time_blocks
        )
        spatial_key = (
            (time_index * self.num_y_blocks + y_index) * self.num_x_blocks
            + x_index
        )
        composite_key = parsed_batches * cells_per_sample + spatial_key
        unique_keys, inverse, counts = torch.unique(
            composite_key,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        token_count = len(unique_keys)
        token_batches = torch.div(
            unique_keys, cells_per_sample, rounding_mode="floor"
        )
        encoded_events = self.event_encoder(local_features)
        attention_output = self._attention_pool(
            encoded_events,
            inverse,
            token_count=token_count,
        )
        attention_feature = self.query_fusion(
            attention_output.flatten(start_dim=1)
        )

        float_counts = counts.to(events.dtype)
        polarity_sum = torch.zeros(token_count, dtype=events.dtype, device=events.device)
        polarity_sum.scatter_add_(0, inverse, positive_indicator)
        positive_fraction = polarity_sum / float_counts
        local_time_sum = torch.zeros_like(polarity_sum)
        local_time_sum.scatter_add_(0, inverse, local_features[:, 2])
        local_time_mean = local_time_sum / float_counts
        local_time_square_sum = torch.zeros_like(polarity_sum)
        local_time_square_sum.scatter_add_(0, inverse, local_features[:, 2].square())
        local_time_variance = (
            local_time_square_sum / float_counts - local_time_mean.square()
        ).clamp_min(0)
        density = float_counts / (float_counts + 8.0)
        token_sample_positive_fraction = sample_positive_fraction[token_batches]
        polarity_contrast = positive_fraction - token_sample_positive_fraction
        epsilon = torch.finfo(events.dtype).eps
        bounded_polarity = positive_fraction.clamp(epsilon, 1.0 - epsilon)
        polarity_entropy = -(
            bounded_polarity * torch.log(bounded_polarity)
            + (1.0 - bounded_polarity) * torch.log(1.0 - bounded_polarity)
        )
        statistics = torch.stack(
            (
                torch.log1p(float_counts),
                positive_fraction,
                local_time_mean,
                local_time_variance,
                density,
                polarity_contrast,
                polarity_entropy,
            ),
            dim=1,
        )
        token_spatial_keys = unique_keys.remainder(cells_per_sample)
        token_time = torch.div(
            token_spatial_keys,
            self.num_x_blocks * self.num_y_blocks,
            rounding_mode="floor",
        )
        within_time = token_spatial_keys.remainder(
            self.num_x_blocks * self.num_y_blocks
        )
        token_y = torch.div(within_time, self.num_x_blocks, rounding_mode="floor")
        token_x = within_time.remainder(self.num_x_blocks)
        token_coords = torch.stack((token_x, token_y, token_time), dim=1)
        position = torch.stack(
            (
                token_x.to(events.dtype) / max(1, self.num_x_blocks - 1),
                token_y.to(events.dtype) / max(1, self.num_y_blocks - 1),
                token_time.to(events.dtype) / max(1, self.num_time_blocks - 1),
            ),
            dim=1,
        )
        encoded_position = self.position_encoder(position)
        tokens = self.token_projection(
            torch.cat(
                (attention_feature, statistics, encoded_position),
                dim=1,
            )
        )
        return TokenizedEvents(
            tokens=tokens,
            token_coords=token_coords,
            token_batch_indices=token_batches,
            event_token_indices=inverse,
            event_local_features=local_features,
            event_xy=torch.stack((x, y), dim=1),
            normalized_time=normalized_time,
            event_batch_indices=parsed_batches,
            token_statistics=statistics,
            event_counts=counts[inverse],
        )


__all__ = ["EventTokenizer", "TokenizedEvents"]
