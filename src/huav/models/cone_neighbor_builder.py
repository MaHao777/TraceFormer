"""Fixed local, batch-isolated spatiotemporal cone neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree  # type: ignore[import]


@dataclass(frozen=True)
class ConeNeighborhoods:
    """Padded neighbors ordered by the ``t-1``, ``t``, ``t+1`` slices."""

    indices: torch.Tensor


class ConeNeighborBuilder:
    """Build one fixed local quadrangular-pyramid neighborhood per token."""

    time_offsets = (-1, 0, 1)

    def __init__(
        self,
        *,
        base_radius: float,
        slope: float,
        max_neighbors_per_time_slice: int,
        causal_mode: bool = False,
    ) -> None:
        if base_radius < 0 or slope < 0:
            raise ValueError("local cone radii must be non-negative")
        if max_neighbors_per_time_slice <= 0:
            raise ValueError("max_neighbors_per_time_slice must be positive")
        self.base_radius = float(base_radius)
        self.slope = float(slope)
        self.max_neighbors_per_time_slice = int(max_neighbors_per_time_slice)
        self.causal_mode = bool(causal_mode)

    def _query_time_slice(
        self,
        *,
        coordinates: np.ndarray,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        time_offset: int,
        output: np.ndarray,
    ) -> None:
        source_xy = coordinates[source_indices, :2].astype(np.float64, copy=False)
        target_xy = coordinates[target_indices, :2].astype(np.float64, copy=False)
        tree = cKDTree(target_xy)
        query_count = min(self.max_neighbors_per_time_slice, len(target_indices))
        ranks = np.arange(1, query_count + 1)
        radius = self.base_radius + self.slope * abs(time_offset)
        distances, local_indices = tree.query(
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
        valid = np.isfinite(distances) & (local_indices < len(target_indices))
        global_indices = np.full(local_indices.shape, -1, dtype=np.int64)
        global_indices[valid] = target_indices[local_indices[valid]]
        slice_index = self.time_offsets.index(time_offset)
        start = slice_index * self.max_neighbors_per_time_slice
        output[source_indices, start : start + query_count] = global_indices

    def __call__(
        self,
        token_coords: torch.Tensor,
        token_batch_indices: torch.Tensor,
    ) -> ConeNeighborhoods:
        if token_coords.ndim != 2 or token_coords.shape[1] != 3:
            raise ValueError("token_coords must have shape [M,3]")
        if (
            token_batch_indices.ndim != 1
            or len(token_batch_indices) != len(token_coords)
        ):
            raise ValueError("token_batch_indices must have shape [M]")
        token_count = len(token_coords)
        output = np.full(
            (token_count, len(self.time_offsets) * self.max_neighbors_per_time_slice),
            -1,
            dtype=np.int64,
        )
        if token_count:
            coordinates = token_coords.detach().cpu().numpy().astype(
                np.int64, copy=False
            )
            batch_indices = token_batch_indices.detach().cpu().numpy().astype(
                np.int64, copy=False
            )
            for batch_index in np.unique(batch_indices):
                sample_indices = np.flatnonzero(batch_indices == batch_index)
                sample_times = coordinates[sample_indices, 2]
                by_time = {
                    int(time): sample_indices[sample_times == time]
                    for time in np.unique(sample_times)
                }
                for source_time, source_indices in by_time.items():
                    for time_offset in self.time_offsets:
                        if self.causal_mode and time_offset > 0:
                            continue
                        target_indices = by_time.get(source_time + time_offset)
                        if target_indices is None:
                            continue
                        self._query_time_slice(
                            coordinates=coordinates,
                            source_indices=source_indices,
                            target_indices=target_indices,
                            time_offset=time_offset,
                            output=output,
                        )
        return ConeNeighborhoods(
            indices=torch.from_numpy(output).to(device=token_coords.device)
        )


__all__ = ["ConeNeighborBuilder", "ConeNeighborhoods"]
