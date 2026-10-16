from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from traceformer.config import ConeHeadConfig
from traceformer.models.cone_neighbor_builder import ConeNeighborBuilder, prepare_cone_geometry
from traceformer.models.cone_transformer import ConeTransformerEncoder


def test_past_and_future_cones_obey_time_direction_and_chebyshev_radius() -> None:
    coords = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 1],
            [3, 0, 2],
            [6, 0, 1],
        ],
        dtype=torch.long,
    )
    batches = torch.zeros(4, dtype=torch.long)
    builder = ConeNeighborBuilder(
        cone_heads=(ConeHeadConfig(1.0, 1.0, 1),),
        max_neighbors_per_cone=4,
    )

    neighbors = builder(coords, batches)

    assert set(neighbors.past_indices[2, 0].tolist()) == {1, -1}
    assert set(neighbors.future_indices[0, 0].tolist()) == {1, -1}
    valid_past = neighbors.past_indices >= 0
    valid_future = neighbors.future_indices >= 0
    center_times = coords[:, 2, None, None]
    gathered_past_times = coords[neighbors.past_indices.clamp_min(0), 2]
    gathered_future_times = coords[neighbors.future_indices.clamp_min(0), 2]
    assert torch.all(gathered_past_times[valid_past] < center_times.expand_as(valid_past)[valid_past])
    assert torch.all(
        gathered_future_times[valid_future]
        > center_times.expand_as(valid_future)[valid_future]
    )


def test_different_cone_heads_produce_different_neighborhoods() -> None:
    coords = torch.tensor([[0, 0, 0], [1, 0, 1], [3, 0, 2]], dtype=torch.long)
    batches = torch.zeros(3, dtype=torch.long)
    builder = ConeNeighborBuilder(
        cone_heads=(
            ConeHeadConfig(0.0, 1.0, 2),
            ConeHeadConfig(0.0, 2.0, 2),
        ),
        max_neighbors_per_cone=4,
    )

    neighbors = builder(coords, batches)

    assert 1 not in neighbors.past_indices[2, 0].tolist()
    assert 1 in neighbors.past_indices[2, 1].tolist()
    assert 0 in neighbors.past_indices[2, 1].tolist()


def test_tokens_from_different_batches_are_never_neighbors() -> None:
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.long)
    batches = torch.tensor([0, 0, 1], dtype=torch.long)
    builder = ConeNeighborBuilder(
        cone_heads=(ConeHeadConfig(1.0, 1.0, 2),),
        max_neighbors_per_cone=4,
    )

    neighbors = builder(coords, batches)

    assert 0 in neighbors.past_indices[1, 0].tolist()
    assert 2 not in neighbors.past_indices[1, 0].tolist()
    assert torch.all(neighbors.past_indices[2] == -1)
    assert torch.all(neighbors.future_indices[2] == -1)


def test_neighbor_cap_keeps_temporally_then_spatially_nearest_tokens() -> None:
    coords = torch.tensor(
        [[0, 0, 0], [3, 0, 1], [1, 0, 1], [0, 0, 2]],
        dtype=torch.long,
    )
    batches = torch.zeros(4, dtype=torch.long)
    builder = ConeNeighborBuilder(
        cone_heads=(ConeHeadConfig(4.0, 0.0, 2),),
        max_neighbors_per_cone=1,
    )

    neighbors = builder(coords, batches)

    assert neighbors.past_indices[3, 0, 0].item() == 2


def test_neighbor_cap_preserves_available_longer_time_lags() -> None:
    coords = torch.tensor(
        [
            [0, 0, 1],
            [0, 0, 2],
            [1, 0, 2],
            [2, 0, 2],
            [0, 0, 3],
        ],
        dtype=torch.long,
    )
    batches = torch.zeros(len(coords), dtype=torch.long)
    builder = ConeNeighborBuilder(
        cone_heads=(ConeHeadConfig(3.0, 0.0, 2),),
        max_neighbors_per_cone=2,
    )

    neighbors = builder(coords, batches)

    assert set(neighbors.past_indices[4, 0].tolist()) == {0, 1}


def test_empty_tokens_return_padded_empty_neighborhoods() -> None:
    builder = ConeNeighborBuilder(
        cone_heads=(ConeHeadConfig(1.0, 1.0, 2),),
        max_neighbors_per_cone=2,
    )

    neighbors = builder(torch.empty((0, 3), dtype=torch.long), torch.empty(0, dtype=torch.long))

    assert neighbors.past_indices.shape == (0, 1, 2)
    assert neighbors.future_indices.shape == (0, 1, 2)


def _original_neighbors(coords, batches, heads, scale, cap, direction):
    """保留修改前每个 head/方向重新建树的查询逻辑，作为逐位对照。"""
    output = np.full((len(coords), len(heads), cap), -1, dtype=np.int64)
    for head_index, head in enumerate(heads):
        horizon = max(1, math.ceil(head.max_time_distance * scale))
        for batch_index in np.unique(batches):
            indices = np.flatnonzero(batches == batch_index)
            times = coords[indices, 2]
            by_time = {int(t): indices[times == t] for t in np.unique(times)}
            trees = {t: cKDTree(coords[rows, :2].astype(np.float64))
                     for t, rows in by_time.items()}
            for source_time, rows in by_time.items():
                candidates, priorities = [], []
                for lag in range(1, horizon + 1):
                    target = by_time.get(source_time + direction * lag)
                    if target is None:
                        continue
                    ranks = np.arange(1, min(cap, len(target)) + 1)
                    distances, local = trees[source_time + direction * lag].query(
                        coords[rows, :2].astype(np.float64), k=ranks, p=np.inf,
                        distance_upper_bound=head.base_radius + head.slope * lag + 1e-9,
                        workers=1,
                    )
                    distances, local = np.asarray(distances), np.asarray(local)
                    if distances.ndim == 1:
                        distances, local = distances[:, None], local[:, None]
                    valid = np.isfinite(distances) & (local < len(target))
                    selected = np.full(local.shape, -1, dtype=np.int64)
                    selected[valid] = target[local[valid]]
                    scores = np.full(distances.shape, np.inf)
                    rank_priority = np.broadcast_to(ranks[None, :], distances.shape)
                    scores[valid] = rank_priority[valid] * 1e6 + lag * 1e3 + distances[valid]
                    candidates.append(selected)
                    priorities.append(scores)
                if candidates:
                    joined, scores = np.concatenate(candidates, 1), np.concatenate(priorities, 1)
                    order = np.argsort(scores, axis=1, kind="stable")[:, :cap]
                    kept = np.take_along_axis(joined, order, axis=1)
                    output[rows, head_index, :kept.shape[1]] = kept
    return torch.from_numpy(output)


@pytest.mark.parametrize("scale", [0.5, 0.75, 1.0, 1.5])
@pytest.mark.parametrize("cap", [1, 3, 8])
@pytest.mark.parametrize("empty", [False, True])
def test_shared_geometry_matches_original_indices(scale, cap, empty):
    # 多 batch、时间片缺失、等距离点和半径边界均保留原始输入顺序。
    coords = torch.tensor([
        [0, 0, 0], [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1],
        [2, 0, 3], [0, 0, 4], [8, 0, 4], [0, 0, 0], [1, 0, 3], [2, 0, 4],
    ], dtype=torch.long)
    batches = torch.tensor([0] * 8 + [2] * 3)
    if empty:
        coords, batches = coords[:0], batches[:0]
    heads = (ConeHeadConfig(1.0, 0.0, 4), ConeHeadConfig(0.0, 1.0, 3))
    builder = ConeNeighborBuilder(cone_heads=heads, max_neighbors_per_cone=cap,
                                  layer_time_scale=scale)
    geometry = prepare_cone_geometry(coords, batches)
    shared, independent = builder(coords, batches, geometry=geometry), builder(coords, batches)
    for name, direction in (("past_indices", -1), ("future_indices", 1)):
        expected = _original_neighbors(coords.numpy(), batches.numpy(), heads, scale, cap, direction)
        assert torch.equal(getattr(shared, name), expected)
        assert torch.equal(getattr(independent, name), expected)


def test_encoder_builds_each_time_tree_once_per_view(monkeypatch):
    from traceformer.models import cone_neighbor_builder

    builds = []

    def count_tree(xy):
        builds.append(xy.copy())
        return cKDTree(xy)

    monkeypatch.setattr(cone_neighbor_builder, "cKDTree", count_tree)
    encoder = ConeTransformerEncoder(
        token_dim=8, num_heads=2,
        cone_heads=(ConeHeadConfig(1, 1, 3), ConeHeadConfig(2, 2, 3)),
        layer_time_scales=(0.5, 0.75, 1.0), max_neighbors_per_cone=3,
        mlp_ratio=2, dropout=0, direction_dropout=0,
    ).eval()
    coords = torch.tensor([[0, 0, 0], [1, 0, 1], [2, 0, 3], [0, 0, 0]])
    batches = torch.tensor([0, 0, 0, 1])
    with torch.no_grad():
        encoder(torch.randn(4, 8), coords, batches)
        assert len(builds) == 4
        encoder(torch.randn(4, 8), coords, batches)
    # 第二次前向重新准备，缓存不跨事件视图或前向保留。
    assert len(builds) == 8
