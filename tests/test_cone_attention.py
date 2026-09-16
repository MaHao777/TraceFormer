from __future__ import annotations

import torch

from traceformer.config import ConeHeadConfig
from traceformer.models.cone_attention import BidirectionalConeAttention
from traceformer.models.relative_position import RelativePositionEncoder


def _attention() -> BidirectionalConeAttention:
    return BidirectionalConeAttention(
        token_dim=8,
        num_heads=2,
        cone_heads=(
            ConeHeadConfig(1.0, 1.0, 2),
            ConeHeadConfig(2.0, 2.0, 2),
        ),
        max_neighbors_per_cone=3,
        dropout=0.0,
        direction_dropout=0.0,
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

    output = attention(
        tokens,
        torch.tensor([[0, 0, 0]], dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
    )

    assert output.features.shape == tokens.shape
    assert torch.isfinite(output.features).all()


def test_attention_uses_shared_bidirectional_key_value_and_backpropagates() -> None:
    attention = _attention()
    tokens = torch.randn(3, 8, requires_grad=True)
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 2]], dtype=torch.long)

    attention(tokens, coords, torch.zeros(3, dtype=torch.long)).features.square().mean().backward()

    assert hasattr(attention, "key_projection")
    assert hasattr(attention, "value_projection")
    assert not hasattr(attention, "past_key_projection")
    assert not hasattr(attention, "future_key_projection")
    assert torch.isfinite(tokens.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in attention.parameters()
    )
