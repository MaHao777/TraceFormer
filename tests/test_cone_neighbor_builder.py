from __future__ import annotations

import torch

from huav.config import ConeHeadConfig
from huav.models.cone_neighbor_builder import ConeNeighborBuilder


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


def test_causal_mode_never_constructs_future_neighbors() -> None:
    coords = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.long)
    batches = torch.zeros(2, dtype=torch.long)
    builder = ConeNeighborBuilder(
        cone_heads=(ConeHeadConfig(1.0, 1.0, 2),),
        max_neighbors_per_cone=2,
        causal_mode=True,
    )

    neighbors = builder(coords, batches)

    assert torch.all(neighbors.future_indices == -1)
    assert neighbors.past_indices[1, 0, 0].item() == 0


def test_empty_tokens_return_padded_empty_neighborhoods() -> None:
    builder = ConeNeighborBuilder(
        cone_heads=(ConeHeadConfig(1.0, 1.0, 2),),
        max_neighbors_per_cone=2,
    )

    neighbors = builder(torch.empty((0, 3), dtype=torch.long), torch.empty(0, dtype=torch.long))

    assert neighbors.past_indices.shape == (0, 1, 2)
    assert neighbors.future_indices.shape == (0, 1, 2)
