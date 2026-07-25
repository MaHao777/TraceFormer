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
    token_statistics: torch.Tensor
    event_counts: torch.Tensor


class EventTokenizer(nn.Module):
    """Map occupied regular spacetime blocks to learned tokens."""

    statistic_dim = 6

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
    ) -> None:
        super().__init__()
        values = (
            width,
            height,
            spatial_block_size_x,
            spatial_block_size_y,
            num_time_blocks,
            event_embed_dim,
            token_dim,
        )
        if any(value <= 0 for value in values):
            raise ValueError("tokenizer dimensions must be positive")
        self.width = int(width)
        self.height = int(height)
        self.spatial_block_size_x = int(spatial_block_size_x)
        self.spatial_block_size_y = int(spatial_block_size_y)
        self.num_time_blocks = int(num_time_blocks)
        self.event_embed_dim = int(event_embed_dim)
        self.token_dim = int(token_dim)
        self.num_x_blocks = math.ceil(self.width / self.spatial_block_size_x)
        self.num_y_blocks = math.ceil(self.height / self.spatial_block_size_y)

        self.event_encoder = nn.Sequential(
            nn.Linear(4, self.event_embed_dim),
            nn.GELU(),
            nn.Linear(self.event_embed_dim, self.event_embed_dim),
            nn.GELU(),
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.event_embed_dim),
            nn.GELU(),
        )
        projection_input = self.event_embed_dim * 3 + self.statistic_dim
        self.token_projection = nn.Sequential(
            nn.Linear(projection_input, self.token_dim),
            nn.LayerNorm(self.token_dim),
        )

    def _split_input(
        self,
        events: torch.Tensor,
        batch_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if events.ndim != 2 or events.shape[1] not in (4, 5):
            raise ValueError("events must have shape [N,4] or [N,5]")
        if not events.is_floating_point():
            raise TypeError("events must have a floating dtype")
        if events.shape[1] == 5:
            if batch_indices is not None:
                raise ValueError("batch_indices must be omitted for [N,5] events")
            parsed_batches = events[:, 0].long()
            event_features = events[:, 1:]
        else:
            event_features = events
            parsed_batches = (
                torch.zeros(len(events), dtype=torch.long, device=events.device)
                if batch_indices is None
                else batch_indices.long()
            )
        if parsed_batches.ndim != 1 or len(parsed_batches) != len(events):
            raise ValueError("batch_indices must have shape [N]")
        if len(parsed_batches) and torch.any(parsed_batches < 0):
            raise ValueError("batch indices must be non-negative")
        return event_features, parsed_batches

    def _empty(self, events: torch.Tensor) -> TokenizedEvents:
        device = events.device
        dtype = events.dtype
        return TokenizedEvents(
            tokens=torch.empty((0, self.token_dim), dtype=dtype, device=device),
            token_coords=torch.empty((0, 3), dtype=torch.long, device=device),
            token_batch_indices=torch.empty((0,), dtype=torch.long, device=device),
            event_token_indices=torch.empty((0,), dtype=torch.long, device=device),
            event_local_features=torch.empty((0, 4), dtype=dtype, device=device),
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
        if (
            torch.any(x < 0)
            or torch.any(x >= self.width)
            or torch.any(y < 0)
            or torch.any(y >= self.height)
            or torch.any(normalized_time < 0)
            or torch.any(normalized_time > 1)
        ):
            raise ValueError("event coordinates are outside configured bounds")

        x_index = torch.div(
            x.long(), self.spatial_block_size_x, rounding_mode="floor"
        )
        y_index = torch.div(
            y.long(), self.spatial_block_size_y, rounding_mode="floor"
        )
        scaled_time = normalized_time * self.num_time_blocks
        time_index = torch.clamp(scaled_time.long(), max=self.num_time_blocks - 1)
        local_features = torch.stack(
            (
                (x - x_index * self.spatial_block_size_x)
                / self.spatial_block_size_x,
                (y - y_index * self.spatial_block_size_y)
                / self.spatial_block_size_y,
                scaled_time - time_index,
                polarity,
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
        encoded_events = self.event_encoder(local_features)
        expanded_inverse = inverse[:, None].expand(-1, self.event_embed_dim)
        mean_pool = torch.zeros(
            (token_count, self.event_embed_dim),
            dtype=encoded_events.dtype,
            device=encoded_events.device,
        )
        mean_pool.scatter_add_(0, expanded_inverse, encoded_events)
        mean_pool = mean_pool / counts[:, None].to(mean_pool.dtype)
        max_pool = torch.full_like(mean_pool, -torch.inf)
        max_pool.scatter_reduce_(
            0,
            expanded_inverse,
            encoded_events,
            reduce="amax",
            include_self=True,
        )

        float_counts = counts.to(events.dtype)
        polarity_sum = torch.zeros(token_count, dtype=events.dtype, device=events.device)
        polarity_sum.scatter_add_(0, inverse, polarity)
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
        statistics = torch.stack(
            (
                torch.log1p(float_counts),
                positive_fraction,
                1.0 - positive_fraction,
                local_time_mean,
                local_time_variance,
                density,
            ),
            dim=1,
        )

        token_batches = torch.div(
            unique_keys, cells_per_sample, rounding_mode="floor"
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
            torch.cat((mean_pool, max_pool, statistics, encoded_position), dim=1)
        )
        return TokenizedEvents(
            tokens=tokens,
            token_coords=token_coords,
            token_batch_indices=token_batches,
            event_token_indices=inverse,
            event_local_features=local_features,
            token_statistics=statistics,
            event_counts=counts[inverse],
        )


__all__ = ["EventTokenizer", "TokenizedEvents"]
