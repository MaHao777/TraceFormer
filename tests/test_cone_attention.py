from __future__ import annotations

import torch

from huav.config import ConeHeadConfig
from huav.models.cone_attention import BidirectionalConeAttention
from huav.models.relative_position import RelativePositionEncoder


def _attention(*, causal_mode: bool = False, debug: bool = True) -> BidirectionalConeAttention:
    return BidirectionalConeAttention(
        token_dim=8,
        num_heads=2,
        cone_heads=(
            ConeHeadConfig(1.0, 1.0, 2),
            ConeHeadConfig(2.0, 2.0, 2),
        ),
        max_neighbors_per_cone=3,
        dropout=0.0,
        causal_mode=causal_mode,
        debug_attention_stats=debug,
        token_importance_topk_ratio=0.5,
    )


def test_relative_position_features_include_signed_time_and_apparent_speed() -> None:
    centers = torch.tensor([[[[1.0, 2.0, 5.0]]]])
    neighbors = torch.tensor([[[[3.0, 1.0, 2.0]]]])

    features = RelativePositionEncoder.build_features(centers, neighbors)

    torch.testing.assert_close(
        features[0, 0, 0],
        torch.tensor([2.0, -1.0, -3.0, 2.0, 3.0, 2.0 / 3.0]),
    )


def test_empty_past_and_future_cones_return_finite_features() -> None:
    attention = _attention()
    tokens = torch.randn(1, 8)
    coords = torch.tensor([[0, 0, 0]], dtype=torch.long)
    batches = torch.zeros(1, dtype=torch.long)

    output = attention(tokens, coords, batches)

    assert output.features.shape == tokens.shape
    assert torch.isfinite(output.features).all()
    assert output.past_attention_stats["edge_count"].item() == 0
    assert output.future_attention_stats["edge_count"].item() == 0


def test_debug_stats_show_separate_past_future_counts_and_normalized_gates() -> None:
    attention = _attention()
    tokens = torch.randn(3, 8)
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=torch.long)
    batches = torch.zeros(3, dtype=torch.long)

    output = attention(tokens, coords, batches)
    center_gate_sum = (
        output.past_attention_stats["gate_weights"][1]
        + output.future_attention_stats["gate_weights"][1]
    )

    assert output.past_attention_stats["neighbor_counts"].shape == (3, 2)
    assert output.future_attention_stats["neighbor_counts"].shape == (3, 2)
    torch.testing.assert_close(center_gate_sum, torch.tensor(1.0))
    assert output.past_attention_stats["edge_count"].item() > 0
    assert output.future_attention_stats["edge_count"].item() > 0
    assert torch.all(
        output.past_attention_stats["neighbor_counts"]
        <= output.past_attention_stats["original_neighbor_counts"]
    )
    assert torch.all(
        output.future_attention_stats["neighbor_counts"]
        <= output.future_attention_stats["original_neighbor_counts"]
    )
    assert output.past_attention_stats["importance"]["final"].shape == (3,)


def test_all_tokens_remain_queries_when_only_half_of_each_cone_is_key_value() -> None:
    attention = _attention()
    tokens = torch.randn(5, 8, requires_grad=True)
    coords = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]],
        dtype=torch.long,
    )
    batches = torch.zeros(5, dtype=torch.long)
    event_counts = torch.tensor([1, 2, 3, 4, 5])

    output = attention(tokens, coords, batches, event_counts)
    output.features.sum().backward()

    assert output.features.shape == tokens.shape
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert output.past_attention_stats["neighbor_counts"].shape[0] == len(tokens)
    assert output.future_attention_stats["neighbor_counts"].shape[0] == len(tokens)


def test_causal_attention_suppresses_future_branch() -> None:
    attention = _attention(causal_mode=True)
    tokens = torch.randn(3, 8)
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=torch.long)
    batches = torch.zeros(3, dtype=torch.long)

    output = attention(tokens, coords, batches)

    assert output.future_attention_stats["edge_count"].item() == 0
    assert torch.all(output.future_attention_stats["gate_weights"] == 0)
    assert output.past_attention_stats["gate_weights"][2].item() == 1.0


def test_attention_uses_one_shared_key_and_value_projection_and_backpropagates() -> None:
    attention = _attention()
    assert hasattr(attention, "key_projection")
    assert hasattr(attention, "value_projection")
    assert not hasattr(attention, "past_key_projection")
    assert not hasattr(attention, "future_key_projection")
    tokens = torch.randn(3, 8, requires_grad=True)
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 2]], dtype=torch.long)
    batches = torch.zeros(3, dtype=torch.long)

    attention(tokens, coords, batches).features.square().mean().backward()

    assert torch.isfinite(tokens.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in attention.parameters()
    )
