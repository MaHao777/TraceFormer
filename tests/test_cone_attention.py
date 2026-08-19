from __future__ import annotations

import torch

from huav.config import LocalConeConfig
from huav.models.cone_attention import LocalConeAttention
from huav.models.relative_position import RelativePositionEncoder


def _attention(
    *,
    causal_mode: bool = False,
    debug: bool = True,
) -> LocalConeAttention:
    return LocalConeAttention(
        token_dim=8,
        num_heads=2,
        local_cone=LocalConeConfig(
            base_radius=1.0,
            slope=1.0,
            max_neighbors_per_time_slice=3,
        ),
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


def test_empty_local_cones_return_finite_features() -> None:
    attention = _attention()
    tokens = torch.empty((0, 8))
    coords = torch.empty((0, 3), dtype=torch.long)
    batches = torch.empty(0, dtype=torch.long)

    output = attention(tokens, coords, batches)

    assert output.features.shape == tokens.shape
    assert torch.isfinite(output.features).all()
    assert output.past_attention_stats is not None
    assert output.future_attention_stats is not None
    assert output.past_attention_stats["edge_count"].item() == 0
    assert output.future_attention_stats["edge_count"].item() == 0


def test_attention_uses_one_shared_local_neighborhood_for_all_heads() -> None:
    attention = _attention()
    tokens = torch.randn(3, 8)
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 2]], dtype=torch.long)

    output = attention(tokens, coords, torch.zeros(3, dtype=torch.long))

    assert output.past_attention_stats is not None
    assert output.future_attention_stats is not None
    past_counts = output.past_attention_stats["neighbor_counts"]
    future_counts = output.future_attention_stats["neighbor_counts"]
    torch.testing.assert_close(past_counts[:, 0], past_counts[:, 1])
    torch.testing.assert_close(future_counts[:, 0], future_counts[:, 1])
    assert not hasattr(attention.neighbor_builder, "cone_heads")


def test_q_k_v_are_recomputed_from_the_same_current_input() -> None:
    attention = _attention(debug=False)
    projection_inputs: dict[str, torch.Tensor] = {}
    handles = []
    for name in ("q_proj", "k_proj", "v_proj"):
        projection = getattr(attention, name)
        handles.append(
            projection.register_forward_pre_hook(
                lambda _module, args, name=name: projection_inputs.setdefault(
                    name, args[0].detach().clone()
                )
            )
        )
    tokens = torch.randn(3, 8)
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 2]], dtype=torch.long)

    attention(tokens, coords, torch.zeros(3, dtype=torch.long))

    for handle in handles:
        handle.remove()
    torch.testing.assert_close(projection_inputs["q_proj"], tokens)
    torch.testing.assert_close(projection_inputs["k_proj"], tokens)
    torch.testing.assert_close(projection_inputs["v_proj"], tokens)


def test_attention_is_synchronous_and_backpropagates_without_mutating_tokens() -> None:
    attention = _attention()
    tokens = torch.randn(4, 8, requires_grad=True)
    original = tokens.detach().clone()
    coords = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 2]],
        dtype=torch.long,
    )

    output = attention(tokens, coords, torch.zeros(4, dtype=torch.long))
    output.features.square().mean().backward()

    torch.testing.assert_close(tokens.detach(), original)
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in attention.parameters()
    )


def test_causal_attention_suppresses_future_neighbors() -> None:
    attention = _attention(causal_mode=True)
    tokens = torch.randn(3, 8)
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=torch.long)

    output = attention(tokens, coords, torch.zeros(3, dtype=torch.long))

    assert output.future_attention_stats is not None
    assert output.future_attention_stats["edge_count"].item() == 0
    assert torch.all(output.future_attention_stats["gate_weights"] == 0)
