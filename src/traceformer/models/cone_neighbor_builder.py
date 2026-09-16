"""Sparse, batch-isolated spatiotemporal cone neighborhood construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree  # type: ignore[import]

from traceformer.config import ConeHeadConfig


@dataclass(frozen=True)
class ConeNeighborhoods:
    past_indices: torch.Tensor
    future_indices: torch.Tensor


class ConeNeighborBuilder:
    """Build capped per-head neighborhoods without an all-pairs distance matrix."""

    def __init__(
        self,
        *,
        cone_heads: tuple[ConeHeadConfig, ...],
        max_neighbors_per_cone: int,
        layer_time_scale: float = 1.0,
    ) -> None:
        self.cone_heads = tuple(cone_heads)
        self.max_neighbors_per_cone = int(max_neighbors_per_cone)
        self.layer_time_scale = float(layer_time_scale)

    def _query_direction(
        self,
        *,
        coordinates: np.ndarray,
        batch_indices: np.ndarray,
        head: ConeHeadConfig,
        direction: int,
        output: np.ndarray,
        head_index: int,
    ) -> None:
        horizon = max(
            1,
            int(math.ceil(head.max_time_distance * self.layer_time_scale)),
        )
        for batch_index in np.unique(batch_indices):
            sample_indices = np.flatnonzero(batch_indices == batch_index)
            sample_times = coordinates[sample_indices, 2]
            by_time = {
                int(time): sample_indices[sample_times == time]
                for time in np.unique(sample_times)
            }
            trees = {
                time: cKDTree(coordinates[indices, :2].astype(np.float64, copy=False))
                for time, indices in by_time.items()
            }
            for source_time, source_indices in by_time.items():
                candidate_indices: list[np.ndarray] = []
                candidate_scores: list[np.ndarray] = []
                source_xy = coordinates[source_indices, :2].astype(
                    np.float64, copy=False
                )
                for time_distance in range(1, horizon + 1):
                    target_time = source_time + direction * time_distance
                    target_indices = by_time.get(target_time)
                    if target_indices is None:
                        continue
                    radius = head.base_radius + head.slope * time_distance
                    query_count = min(
                        self.max_neighbors_per_cone, len(target_indices)
                    )
                    ranks = np.arange(1, query_count + 1)
                    distances, local_indices = trees[target_time].query(
                        source_xy,
                        k=ranks,
                        p=np.inf,
                        distance_upper_bound=radius + 1e-9,
                        workers=1,
                    )
                    distances = np.asarray(distances)
                    local_indices = np.asarray(local_indices)
                    if distances.ndim == 1:
                        distances = distances[:, None]
                        local_indices = local_indices[:, None]
                    valid = np.isfinite(distances) & (
                        local_indices < len(target_indices)
                    )
                    global_indices = np.full(local_indices.shape, -1, dtype=np.int64)
                    global_indices[valid] = target_indices[local_indices[valid]]
                    scores = np.full(distances.shape, np.inf, dtype=np.float64)
                    rank_priority = np.broadcast_to(
                        ranks[None, :], distances.shape
                    )
                    scores[valid] = (
                        rank_priority[valid] * 1_000_000.0
                        + time_distance * 1_000.0
                        + distances[valid]
                    )
                    candidate_indices.append(global_indices)
                    candidate_scores.append(scores)

                if not candidate_indices:
                    continue
                joined_indices = np.concatenate(candidate_indices, axis=1)
                joined_scores = np.concatenate(candidate_scores, axis=1)
                order = np.argsort(joined_scores, axis=1, kind="stable")
                keep = order[:, : self.max_neighbors_per_cone]
                selected = np.take_along_axis(joined_indices, keep, axis=1)
                output[
                    source_indices,
                    head_index,
                    : selected.shape[1],
                ] = selected

    def __call__(
        self,
        token_coords: torch.Tensor,
        token_batch_indices: torch.Tensor,
    ) -> ConeNeighborhoods:
        token_count = len(token_coords)
        shape = (
            token_count,
            len(self.cone_heads),
            self.max_neighbors_per_cone,
        )
        past = np.full(shape, -1, dtype=np.int64)
        future = np.full(shape, -1, dtype=np.int64)
        if token_count:
            coordinates = token_coords.detach().cpu().numpy().astype(
                np.int64, copy=False
            )
            batch_indices = token_batch_indices.detach().cpu().numpy().astype(
                np.int64, copy=False
            )
            for head_index, head in enumerate(self.cone_heads):
                self._query_direction(
                    coordinates=coordinates,
                    batch_indices=batch_indices,
                    head=head,
                    direction=-1,
                    output=past,
                    head_index=head_index,
                )
                self._query_direction(
                    coordinates=coordinates,
                    batch_indices=batch_indices,
                    head=head,
                    direction=1,
                    output=future,
                    head_index=head_index,
                )
        device = token_coords.device
        return ConeNeighborhoods(
            past_indices=torch.from_numpy(past).to(device=device),
            future_indices=torch.from_numpy(future).to(device=device),
        )


__all__ = ["ConeNeighborBuilder", "ConeNeighborhoods"]
