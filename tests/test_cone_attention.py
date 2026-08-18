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


def test_attention_keeps_the_local_cap_from_every_target_time_slice() -> None:
    attention = BidirectionalConeAttention(
        token_dim=4,
        num_heads=1,
        cone_heads=(ConeHeadConfig(4.0, 0.0, 3),),
        max_neighbors_per_cone=2,
        dropout=0.0,
        debug_attention_stats=True,
    )
    coords = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [2, 0, 1],
            [0, 0, 2],
            [1, 0, 2],
            [2, 0, 2],
            [0, 0, 3],
        ],
        dtype=torch.long,
    )
    tokens = torch.randn(len(coords), 4, requires_grad=True)

    output = attention(tokens, coords, torch.zeros(len(coords), dtype=torch.long))

    assert output.past_attention_stats["neighbor_counts"][9, 0].item() == 6
    output.features.square().mean().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


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
