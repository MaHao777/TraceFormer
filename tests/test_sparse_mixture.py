from __future__ import annotations

import torch

from traceformer.models.sparse_mixture import SparseTokenMixture


def _module() -> SparseTokenMixture:
    return SparseTokenMixture(
        token_dim=4,
        num_x_blocks=4,
        num_y_blocks=3,
        num_time_blocks=2,
        initial_router_bias=-2.0,
        initial_residual_scale=0.01,
    )


def test_zero_mixture_weight_reproduces_full_tokens_exactly() -> None:
    module = _module()
    full = torch.randn(3, 4)
    sparse = torch.randn(1, 4)
    full_coords = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
    sparse_coords = torch.tensor([[1, 0, 0]])
    full_batches = torch.zeros(3, dtype=torch.long)
    sparse_batches = torch.zeros(1, dtype=torch.long)
    full_statistics = torch.zeros(3, 7)
    sparse_statistics = torch.zeros(1, 7)

    output = module(
        full,
        full_coords,
        full_batches,
        full_statistics,
        sparse,
        sparse_coords,
        sparse_batches,
        sparse_statistics,
        mixture_weight=0.0,
    )

    assert torch.equal(output.token_features, full)
    assert torch.equal(output.gate, torch.zeros(3))
    assert output.sparse_present.tolist() == [False, True, False]


def test_sparse_tokens_align_by_batch_and_spacetime_coordinate() -> None:
    module = _module()
    full = torch.zeros(4, 4, requires_grad=True)
    sparse = torch.ones(2, 4, requires_grad=True)
    full_coords = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]]
    )
    sparse_coords = torch.tensor([[1, 0, 0], [0, 0, 0]])
    full_batches = torch.tensor([0, 0, 1, 1])
    sparse_batches = torch.tensor([0, 1])
    full_statistics = torch.zeros(4, 7)
    sparse_statistics = torch.zeros(2, 7)

    output = module(
        full,
        full_coords,
        full_batches,
        full_statistics,
        sparse,
        sparse_coords,
        sparse_batches,
        sparse_statistics,
        mixture_weight=1.0,
    )

    assert output.sparse_present.tolist() == [False, True, True, False]
    assert torch.equal(output.gate[[0, 3]], torch.zeros(2))
    assert bool(torch.all(output.gate[[1, 2]] > 0))
    output.token_features.square().sum().backward()
    assert full.grad is not None and torch.isfinite(full.grad).all()
    assert sparse.grad is not None and torch.isfinite(sparse.grad).all()


def test_rejects_sparse_coordinates_absent_from_full_view() -> None:
    module = _module()

    try:
        module(
            torch.zeros(1, 4),
            torch.tensor([[0, 0, 0]]),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, 7),
            torch.zeros(1, 4),
            torch.tensor([[1, 0, 0]]),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, 7),
            mixture_weight=1.0,
        )
    except ValueError as error:
        assert "absent from the full view" in str(error)
    else:
        raise AssertionError("expected sparse/full alignment failure")
