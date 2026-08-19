from __future__ import annotations

import torch

from huav.models.cone_neighbor_builder import ConeNeighborBuilder


def _builder(
    *,
    base_radius: float = 1.0,
    slope: float = 1.0,
    cap: int = 4,
    causal_mode: bool = False,
) -> ConeNeighborBuilder:
    return ConeNeighborBuilder(
        base_radius=base_radius,
        slope=slope,
        max_neighbors_per_time_slice=cap,
        causal_mode=causal_mode,
    )


def test_local_cone_uses_only_t_minus_one_t_and_t_plus_one() -> None:
    coords = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
        dtype=torch.long,
    )

    indices = _builder()(coords, torch.zeros(4, dtype=torch.long)).indices

    for center_index, selected in enumerate(indices):
        valid = selected[selected >= 0]
        offsets = coords[valid, 2] - coords[center_index, 2]
        assert set(offsets.tolist()) <= {-1, 0, 1}
    assert 0 not in indices[2].tolist()
    assert 3 in indices[2].tolist()


def test_quadrangular_pyramid_radius_expands_with_absolute_time_offset() -> None:
    coords = torch.tensor(
        [
            [0, 0, 1],
            [1, 1, 1],
            [2, 0, 1],
            [2, 0, 0],
            [3, 0, 0],
            [2, 2, 0],
        ],
        dtype=torch.long,
    )

    selected = _builder(base_radius=1.0, slope=1.0, cap=8)(
        coords, torch.zeros(len(coords), dtype=torch.long)
    ).indices[0]

    assert 1 in selected.tolist()
    assert 2 not in selected.tolist()
    assert 3 in selected.tolist()
    assert 4 not in selected.tolist()
    assert 5 in selected.tolist()


def test_neighbor_cap_is_applied_independently_to_each_time_slice() -> None:
    coords = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 2],
            [1, 0, 2],
        ],
        dtype=torch.long,
    )

    selected = _builder(base_radius=4.0, slope=0.0, cap=1)(
        coords, torch.zeros(len(coords), dtype=torch.long)
    ).indices[2]
    valid = selected[selected >= 0]
    selected_times = coords[valid, 2]

    assert torch.bincount(selected_times, minlength=3).tolist() == [1, 1, 1]


def test_tokens_from_different_batches_are_never_neighbors() -> None:
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.long)
    batches = torch.tensor([0, 0, 1], dtype=torch.long)

    indices = _builder()(coords, batches).indices

    assert 0 in indices[1].tolist()
    assert 2 not in indices[1].tolist()
    assert set(indices[2].tolist()) <= {2, -1}


def test_causal_mode_keeps_present_and_past_but_suppresses_future() -> None:
    coords = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.long)

    indices = _builder(causal_mode=True)(
        coords, torch.zeros(2, dtype=torch.long)
    ).indices

    assert 1 not in indices[0].tolist()
    assert 0 in indices[0].tolist()
    assert 0 in indices[1].tolist()
    assert 1 in indices[1].tolist()


def test_builder_preserves_source_coordinates_and_returns_fixed_shape() -> None:
    coords = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.long)
    original = coords.clone()

    indices = _builder(cap=2)(coords, torch.zeros(2, dtype=torch.long)).indices

    torch.testing.assert_close(coords, original)
    assert indices.shape == (2, 6)


def test_empty_tokens_return_padded_empty_neighborhoods() -> None:
    neighborhoods = _builder(cap=2)(
        torch.empty((0, 3), dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )

    assert neighborhoods.indices.shape == (0, 6)
