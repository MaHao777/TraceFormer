"""Sparse adaptive threshold calibration for event-level logits."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AdaptiveThresholdResult:
    """Calibrated event logits and sparse grouping diagnostics."""

    final_logits: torch.Tensor
    base_event_logits: torch.Tensor
    global_event_offsets: torch.Tensor
    local_event_offsets: torch.Tensor
    event_group_indices: torch.Tensor
    group_coords: torch.Tensor
    group_batch_indices: torch.Tensor
    group_descriptors: torch.Tensor
    neighbor_edge_indices: torch.Tensor


class AdaptiveThresholdCalibrator(nn.Module):
    """Predict bounded global and occupied-cell logit threshold offsets."""

    # Learned calibration only needs an ordered, bounded view of pathological
    # descriptor magnitudes; raw summaries remain available in the result.
    projection_input_bound = 1.0e4
    allowed_modes = frozenset(
        {"none", "global", "local", "local_neighbor", "hierarchical"}
    )
    statistic_dim = 7
    summary_dim = 5
    coordinate_dim = 3

    def __init__(
        self,
        *,
        token_dim: int,
        width: int,
        height: int,
        spatial_block_size_x: int,
        spatial_block_size_y: int,
        threshold_time_bins: int,
        mode: str,
        hidden_dim: int = 32,
        max_global_offset: float = 0.5,
        max_local_offset: float = 0.75,
    ) -> None:
        super().__init__()
        dimensions = (
            token_dim,
            width,
            height,
            spatial_block_size_x,
            spatial_block_size_y,
            threshold_time_bins,
            hidden_dim,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions):
            raise ValueError("calibrator dimensions must be positive integers")
        if mode not in self.allowed_modes:
            raise ValueError(
                f"mode must be one of {sorted(self.allowed_modes)}, got {mode!r}"
            )
        if not math.isfinite(max_global_offset) or max_global_offset < 0:
            raise ValueError("max_global_offset must be finite and non-negative")
        if not math.isfinite(max_local_offset) or max_local_offset < 0:
            raise ValueError("max_local_offset must be finite and non-negative")

        self.token_dim = token_dim
        self.width = width
        self.height = height
        self.spatial_block_size_x = spatial_block_size_x
        self.spatial_block_size_y = spatial_block_size_y
        self.threshold_time_bins = threshold_time_bins
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.max_global_offset = float(max_global_offset)
        self.max_local_offset = float(max_local_offset)
        self.num_x_blocks = math.ceil(width / spatial_block_size_x)
        self.num_y_blocks = math.ceil(height / spatial_block_size_y)
        self.descriptor_dim = (
            token_dim + self.statistic_dim + self.summary_dim + self.coordinate_dim
        )

        self.descriptor_projection = nn.Sequential(
            nn.Linear(self.descriptor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.local_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.neighbor_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.neighbor_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.neighbor_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relative_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)

    def _validate_inputs(
        self,
        events: torch.Tensor,
        encoded_parent_features: torch.Tensor,
        event_token_statistics: torch.Tensor,
        base_logits: torch.Tensor,
        event_batch_indices: torch.Tensor,
    ) -> None:
        if events.ndim != 2 or events.shape[1] != 4:
            raise ValueError("events must have shape [N,4]")
        if not events.is_floating_point():
            raise TypeError("events must have a floating dtype")
        event_count = len(events)
        expected_shapes = (
            (
                encoded_parent_features,
                (event_count, self.token_dim),
                "encoded_parent_features",
            ),
            (
                event_token_statistics,
                (event_count, self.statistic_dim),
                "event_token_statistics",
            ),
            (base_logits, (event_count,), "base_logits"),
            (event_batch_indices, (event_count,), "event_batch_indices"),
        )
        for tensor, expected, name in expected_shapes:
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape {list(expected)}")
            if tensor.device != events.device:
                raise ValueError("all calibrator inputs must be on the same device")
        for tensor, name in (
            (encoded_parent_features, "encoded_parent_features"),
            (event_token_statistics, "event_token_statistics"),
            (base_logits, "base_logits"),
        ):
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must have a floating dtype")
            if tensor.dtype != events.dtype:
                raise TypeError("all floating calibrator inputs must have the same dtype")
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if event_batch_indices.dtype not in integer_dtypes:
            raise TypeError("event_batch_indices must have an integer dtype")
        if event_count == 0:
            return
        if not all(
            torch.isfinite(tensor).all()
            for tensor in (
                events,
                encoded_parent_features,
                event_token_statistics,
                base_logits,
            )
        ):
            raise ValueError("floating calibrator inputs must be finite")
        x, y, normalized_time, _ = events.unbind(dim=1)
        if (
            torch.any(x < 0)
            or torch.any(x >= self.width)
            or torch.any(y < 0)
            or torch.any(y >= self.height)
            or torch.any(normalized_time < 0)
            or torch.any(normalized_time > 1)
        ):
            raise ValueError("event coordinates are outside configured bounds")
        if torch.any(event_batch_indices < 0):
            raise ValueError("event batch indices must be non-negative")

    def _empty_result(
        self,
        events: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> AdaptiveThresholdResult:
        device = events.device
        dtype = events.dtype
        empty_events = torch.empty((0,), dtype=dtype, device=device)
        return AdaptiveThresholdResult(
            final_logits=base_logits,
            base_event_logits=base_logits,
            global_event_offsets=empty_events,
            local_event_offsets=empty_events.clone(),
            event_group_indices=torch.empty((0,), dtype=torch.long, device=device),
            group_coords=torch.empty((0, 3), dtype=torch.long, device=device),
            group_batch_indices=torch.empty((0,), dtype=torch.long, device=device),
            group_descriptors=torch.empty(
                (0, self.descriptor_dim), dtype=dtype, device=device
            ),
            neighbor_edge_indices=torch.empty(
                (2, 0), dtype=torch.long, device=device
            ),
        )

    @staticmethod
    def _group_sum(
        values: torch.Tensor,
        inverse: torch.Tensor,
        group_count: int,
    ) -> torch.Tensor:
        output = values.new_zeros((group_count, values.shape[1]))
        indices = inverse[:, None].expand(-1, values.shape[1])
        output.scatter_add_(0, indices, values)
        return output

    def _group_events(
        self,
        events: torch.Tensor,
        event_batch_indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x, y, normalized_time, _ = events.detach().unbind(dim=1)
        x_indices = torch.floor(x / self.spatial_block_size_x).long()
        y_indices = torch.floor(y / self.spatial_block_size_y).long()
        time_indices = torch.clamp(
            torch.floor(normalized_time * self.threshold_time_bins).long(),
            max=self.threshold_time_bins - 1,
        )
        groups_per_batch = (
            self.threshold_time_bins * self.num_y_blocks * self.num_x_blocks
        )
        spatial_keys = (
            (time_indices * self.num_y_blocks + y_indices) * self.num_x_blocks
            + x_indices
        )
        event_keys = event_batch_indices.long() * groups_per_batch + spatial_keys
        group_keys, inverse, counts = torch.unique(
            event_keys,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        group_batches = torch.div(
            group_keys, groups_per_batch, rounding_mode="floor"
        )
        group_spatial_keys = group_keys.remainder(groups_per_batch)
        group_time = torch.div(
            group_spatial_keys,
            self.num_y_blocks * self.num_x_blocks,
            rounding_mode="floor",
        )
        within_time = group_spatial_keys.remainder(
            self.num_y_blocks * self.num_x_blocks
        )
        group_y = torch.div(
            within_time, self.num_x_blocks, rounding_mode="floor"
        )
        group_x = within_time.remainder(self.num_x_blocks)
        group_coords = torch.stack((group_x, group_y, group_time), dim=1)
        return group_keys, inverse, counts, group_coords, group_batches

    def _build_descriptors(
        self,
        events: torch.Tensor,
        encoded_parent_features: torch.Tensor,
        event_token_statistics: torch.Tensor,
        base_logits: torch.Tensor,
        inverse: torch.Tensor,
        counts: torch.Tensor,
        group_coords: torch.Tensor,
    ) -> torch.Tensor:
        group_count = len(counts)
        dtype = events.dtype
        detached_features = encoded_parent_features.detach()
        detached_statistics = event_token_statistics.detach()
        detached_logits = base_logits.detach()
        float_counts = counts.to(dtype)
        feature_mean = self._group_sum(
            detached_features, inverse, group_count
        ) / float_counts[:, None]
        statistic_mean = self._group_sum(
            detached_statistics, inverse, group_count
        ) / float_counts[:, None]

        positive = (events.detach()[:, 3] > 0).to(dtype)
        positive_sum = positive.new_zeros(group_count)
        positive_sum.scatter_add_(0, inverse, positive)
        positive_fraction = positive_sum / float_counts

        logit_scale = detached_logits.new_zeros(group_count)
        logit_scale.scatter_reduce_(
            0,
            inverse,
            detached_logits.abs(),
            reduce="amax",
            include_self=True,
        )
        safe_logit_scale = torch.where(
            logit_scale > 0,
            logit_scale,
            torch.ones_like(logit_scale),
        )
        normalized_logits = detached_logits / safe_logit_scale[inverse]
        normalized_logit_sum = detached_logits.new_zeros(group_count)
        normalized_logit_sum.scatter_add_(0, inverse, normalized_logits)
        normalized_logit_mean = normalized_logit_sum / float_counts
        logit_mean = normalized_logit_mean * logit_scale
        logit_max = torch.full(
            (group_count,),
            -torch.inf,
            dtype=dtype,
            device=events.device,
        )
        logit_max.scatter_reduce_(
            0, inverse, detached_logits, reduce="amax", include_self=True
        )
        centered_normalized_logits = (
            normalized_logits - normalized_logit_mean[inverse]
        )
        normalized_centered_square_sum = detached_logits.new_zeros(group_count)
        normalized_centered_square_sum.scatter_add_(
            0,
            inverse,
            centered_normalized_logits.square(),
        )
        normalized_logit_variance = (
            normalized_centered_square_sum / float_counts
        ).clamp_min(0)
        logit_std = torch.sqrt(normalized_logit_variance) * logit_scale

        normalized_coords = torch.stack(
            (
                group_coords[:, 0].to(dtype) / max(1, self.num_x_blocks - 1),
                group_coords[:, 1].to(dtype) / max(1, self.num_y_blocks - 1),
                group_coords[:, 2].to(dtype)
                / max(1, self.threshold_time_bins - 1),
            ),
            dim=1,
        )
        summaries = torch.stack(
            (
                torch.log1p(float_counts),
                positive_fraction,
                logit_mean,
                logit_max,
                logit_std,
            ),
            dim=1,
        )
        return torch.cat(
            (feature_mean, statistic_mean, summaries, normalized_coords),
            dim=1,
        )

    def _neighbor_edges(
        self,
        group_keys: torch.Tensor,
        group_coords: torch.Tensor,
        group_batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        group_count = len(group_keys)
        device = group_keys.device
        all_groups = torch.arange(group_count, dtype=torch.long, device=device)
        targets: list[torch.Tensor] = []
        sources: list[torch.Tensor] = []
        groups_per_batch = (
            self.threshold_time_bins * self.num_y_blocks * self.num_x_blocks
        )
        for delta_time in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if delta_x == delta_y == delta_time == 0:
                        continue
                    candidate_x = group_coords[:, 0] + delta_x
                    candidate_y = group_coords[:, 1] + delta_y
                    candidate_time = group_coords[:, 2] + delta_time
                    valid = (
                        (candidate_x >= 0)
                        & (candidate_x < self.num_x_blocks)
                        & (candidate_y >= 0)
                        & (candidate_y < self.num_y_blocks)
                        & (candidate_time >= 0)
                        & (candidate_time < self.threshold_time_bins)
                    )
                    valid_targets = all_groups[valid]
                    if len(valid_targets) == 0:
                        continue
                    candidate_spatial_keys = (
                        (
                            candidate_time[valid] * self.num_y_blocks
                            + candidate_y[valid]
                        )
                        * self.num_x_blocks
                        + candidate_x[valid]
                    )
                    candidate_keys = (
                        group_batch_indices[valid] * groups_per_batch
                        + candidate_spatial_keys
                    )
                    positions = torch.searchsorted(group_keys, candidate_keys)
                    safe_positions = positions.clamp(max=group_count - 1)
                    found = (positions < group_count) & (
                        group_keys[safe_positions] == candidate_keys
                    )
                    if torch.any(found):
                        targets.append(valid_targets[found])
                        sources.append(positions[found])
        if not targets:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.stack((torch.cat(targets), torch.cat(sources)), dim=0)

    def _neighbor_context(
        self,
        group_embeddings: torch.Tensor,
        group_coords: torch.Tensor,
        edge_indices: torch.Tensor,
    ) -> torch.Tensor:
        if edge_indices.shape[1] == 0:
            return torch.zeros_like(group_embeddings)
        targets, sources = edge_indices
        relative_coords = (
            group_coords[sources] - group_coords[targets]
        ).to(group_embeddings.dtype)
        relative_features = self.relative_encoder(relative_coords)
        queries = self.neighbor_query(group_embeddings[targets])
        keys = self.neighbor_key(group_embeddings[sources]) + relative_features
        scores = torch.sum(queries * keys, dim=1) / math.sqrt(self.hidden_dim)

        group_count = len(group_embeddings)
        maximum = torch.full(
            (group_count,),
            -torch.inf,
            dtype=scores.dtype,
            device=scores.device,
        )
        maximum.scatter_reduce_(
            0, targets, scores, reduce="amax", include_self=True
        )
        unnormalized = torch.exp(scores - maximum[targets])
        normalizers = scores.new_zeros(group_count)
        normalizers.scatter_add_(0, targets, unnormalized)
        weights = unnormalized / normalizers[targets].clamp_min(
            torch.finfo(scores.dtype).tiny
        )
        values = self.neighbor_value(group_embeddings[sources]) + relative_features
        weighted_values = values * weights[:, None]
        context = torch.zeros_like(group_embeddings)
        context.scatter_add_(
            0,
            targets[:, None].expand(-1, self.hidden_dim),
            weighted_values,
        )
        return context

    def _global_offsets(
        self,
        group_embeddings: torch.Tensor,
        group_batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        unique_batches, group_to_batch = torch.unique(
            group_batch_indices, sorted=True, return_inverse=True
        )
        batch_count = len(unique_batches)
        counts = torch.bincount(group_to_batch, minlength=batch_count).to(
            group_embeddings.dtype
        )
        means = self._group_sum(
            group_embeddings, group_to_batch, batch_count
        ) / counts[:, None]
        maxima = torch.full(
            (batch_count, self.hidden_dim),
            -torch.inf,
            dtype=group_embeddings.dtype,
            device=group_embeddings.device,
        )
        maxima.scatter_reduce_(
            0,
            group_to_batch[:, None].expand(-1, self.hidden_dim),
            group_embeddings,
            reduce="amax",
            include_self=True,
        )
        batch_offsets = self.max_global_offset * torch.tanh(
            self.global_head(torch.cat((means, maxima), dim=1)).squeeze(1)
        )
        return batch_offsets[group_to_batch]

    def forward(
        self,
        events: torch.Tensor,
        encoded_parent_features: torch.Tensor,
        event_token_statistics: torch.Tensor,
        base_logits: torch.Tensor,
        event_batch_indices: torch.Tensor,
    ) -> AdaptiveThresholdResult:
        self._validate_inputs(
            events,
            encoded_parent_features,
            event_token_statistics,
            base_logits,
            event_batch_indices,
        )
        if len(events) == 0:
            return self._empty_result(events, base_logits)

        (
            group_keys,
            inverse,
            counts,
            group_coords,
            group_batches,
        ) = self._group_events(events, event_batch_indices)
        descriptors = self._build_descriptors(
            events,
            encoded_parent_features,
            event_token_statistics,
            base_logits,
            inverse,
            counts,
            group_coords,
        )
        projection_descriptors = descriptors.clamp(
            min=-self.projection_input_bound,
            max=self.projection_input_bound,
        )
        group_embeddings = self.descriptor_projection(projection_descriptors)
        if not torch.isfinite(group_embeddings).all():
            raise FloatingPointError(
                "adaptive threshold descriptor projection produced "
                "a non-finite learned embedding"
            )
        neighbor_edges = self._neighbor_edges(
            group_keys, group_coords, group_batches
        )
        zero_group_offsets = base_logits.new_zeros(len(group_coords))

        if self.mode in {"global", "hierarchical"}:
            global_group_offsets = self._global_offsets(
                group_embeddings, group_batches
            )
        else:
            global_group_offsets = zero_group_offsets

        if self.mode in {"local", "local_neighbor", "hierarchical"}:
            local_input = group_embeddings
            if self.mode in {"local_neighbor", "hierarchical"}:
                local_input = local_input + self._neighbor_context(
                    group_embeddings, group_coords, neighbor_edges
                )
            local_group_offsets = self.max_local_offset * torch.tanh(
                self.local_head(local_input).squeeze(1)
            )
        else:
            local_group_offsets = zero_group_offsets

        global_event_offsets = global_group_offsets[inverse]
        local_event_offsets = local_group_offsets[inverse]
        if self.mode == "none":
            final_logits = base_logits
        else:
            final_logits = (
                base_logits - global_event_offsets - local_event_offsets
            )
        return AdaptiveThresholdResult(
            final_logits=final_logits,
            base_event_logits=base_logits,
            global_event_offsets=global_event_offsets,
            local_event_offsets=local_event_offsets,
            event_group_indices=inverse,
            group_coords=group_coords,
            group_batch_indices=group_batches,
            group_descriptors=descriptors,
            neighbor_edge_indices=neighbor_edges,
        )


__all__ = ["AdaptiveThresholdCalibrator", "AdaptiveThresholdResult"]
