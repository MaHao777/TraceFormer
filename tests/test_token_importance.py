from __future__ import annotations

import torch

from huav.models.token_importance import TokenImportance


def _importance(*, topk_ratio: float = 0.5) -> TokenImportance:
    return TokenImportance(
        topk_ratio=topk_ratio,
        density_weight=0.3,
        attention_weight=0.3,
        track_weight=0.4,
    )


def test_default_topk_ratio_retains_three_quarters() -> None:
    assert TokenImportance().topk_ratio == 0.75


def test_scores_are_sample_normalized_and_combined_with_positive_weights() -> None:
    tokens = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    event_counts = torch.tensor([2, 8, 4, 3, 9])
    coords = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [2, 0, 2], [0, 0, 0], [0, 0, 1]]
    )
    batches = torch.tensor([0, 0, 0, 1, 1])
    past = torch.tensor(
        [[[-1, -1]], [[0, -1]], [[1, -1]], [[-1, -1]], [[3, -1]]]
    )
    future = torch.tensor(
        [[[1, -1]], [[2, -1]], [[-1, -1]], [[4, -1]], [[-1, -1]]]
    )
    past_weights = torch.tensor(
        [[[0.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.0]], [[1.0, 0.0]]]
    )
    future_weights = torch.tensor(
        [[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.0]]]
    )

    scores = _importance()(
        tokens=tokens,
        event_counts=event_counts,
        token_coords=coords,
        token_batch_indices=batches,
        past_indices=past,
        future_indices=future,
        past_attention_weights=past_weights,
        future_attention_weights=future_weights,
    )

    for component in (scores.density, scores.attention, scores.track, scores.final):
        assert torch.isfinite(component).all()
        assert torch.all(component >= 0)
        assert torch.all(component <= 1)
    torch.testing.assert_close(
        scores.final,
        0.3 * scores.density + 0.3 * scores.attention + 0.4 * scores.track,
    )
    for sample_index in (0, 1):
        sample_mask = batches == sample_index
        for component in (scores.density, scores.attention, scores.track):
            sample = component[sample_mask]
            assert sample.min().item() == 0.0
            assert sample.max().item() in {0.0, 1.0}


def test_topk_selection_is_per_query_head_and_keeps_highest_importance() -> None:
    neighborhoods = torch.tensor(
        [
            [[1, 2, 3, -1], [0, 2, -1, -1]],
            [[0, -1, -1, -1], [-1, -1, -1, -1]],
        ]
    )
    importance = torch.tensor([0.1, 0.9, 0.4, 0.8])

    selected = _importance(topk_ratio=0.5).select_topk(
        neighborhoods, importance
    )

    assert selected[0, 0].tolist() == [1, -1, 3, -1]
    assert selected[0, 1].tolist() == [-1, 2, -1, -1]
    assert selected[1, 0].tolist() == [0, -1, -1, -1]
    assert selected[1, 1].tolist() == [-1, -1, -1, -1]


def test_empty_tokens_return_empty_scores_and_selection() -> None:
    module = _importance()
    scores = module(
        tokens=torch.empty((0, 4)),
        event_counts=torch.empty((0,), dtype=torch.long),
        token_coords=torch.empty((0, 3), dtype=torch.long),
        token_batch_indices=torch.empty((0,), dtype=torch.long),
        past_indices=torch.empty((0, 2, 3), dtype=torch.long),
        future_indices=torch.empty((0, 2, 3), dtype=torch.long),
        past_attention_weights=torch.empty((0, 2, 3)),
        future_attention_weights=torch.empty((0, 2, 3)),
    )

    assert scores.final.shape == (0,)
    assert module.select_topk(
        torch.empty((0, 2, 3), dtype=torch.long), scores.final
    ).shape == (0, 2, 3)
