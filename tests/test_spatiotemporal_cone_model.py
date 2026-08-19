from __future__ import annotations

import torch

from huav.config import LocalConeConfig
from huav.models import SpatioTemporalConeDetector


def _model(
    *,
    num_cone_blocks: int = 3,
    causal_mode: bool = False,
    debug: bool = True,
) -> SpatioTemporalConeDetector:
    return SpatioTemporalConeDetector(
        width=16,
        height=12,
        spatial_block_size_x=4,
        spatial_block_size_y=4,
        num_time_blocks=4,
        token_dim=16,
        event_embed_dim=8,
        num_cone_blocks=num_cone_blocks,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        local_cone=LocalConeConfig(
            base_radius=1.0,
            slope=1.0,
            max_neighbors_per_time_slice=3,
        ),
        causal_mode=causal_mode,
        direction_dropout=0.0,
        debug_attention_stats=debug,
    )


def _events() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 1.0, 2.0, 0.10, 1.0],
            [0.0, 2.0, 2.0, 0.40, 0.0],
            [0.0, 4.0, 3.0, 0.70, 1.0],
            [1.0, 1.0, 2.0, 0.10, 1.0],
            [1.0, 2.0, 2.0, 0.40, 0.0],
        ]
    )


def test_forward_returns_unchanged_event_token_and_attention_shapes() -> None:
    output = _model()(_events())

    assert {
        "event_logits",
        "token_features",
        "token_coords",
        "token_batch_indices",
        "event_token_indices",
        "past_attention_stats",
        "future_attention_stats",
    } == set(output)
    assert output["event_logits"].shape == (5,)
    assert output["event_token_indices"].shape == (5,)
    assert output["token_features"].shape[1] == 16
    assert len(output["past_attention_stats"]) == 3
    assert len(output["future_attention_stats"]) == 3


def test_requested_block_depths_preserve_forward_shapes() -> None:
    for depth in (1, 3, 5, 7):
        output = _model(num_cone_blocks=depth)(_events())

        assert output["event_logits"].shape == (5,)
        assert output["event_token_indices"].shape == (5,)
        assert output["token_features"].shape[1] == 16
        assert len(output["past_attention_stats"]) == depth


def test_every_block_uses_the_previous_blocks_complete_updated_tokens() -> None:
    model = _model(num_cone_blocks=3, debug=False)
    block_inputs: list[torch.Tensor] = []
    block_outputs: list[torch.Tensor] = []
    handles = []
    for block in model.encoder.blocks:
        handles.append(
            block.register_forward_pre_hook(
                lambda _module, args: block_inputs.append(args[0].detach().clone())
            )
        )
        handles.append(
            block.register_forward_hook(
                lambda _module, _args, output: block_outputs.append(
                    output[0].detach().clone()
                )
            )
        )

    model(_events())

    for handle in handles:
        handle.remove()
    assert len(block_inputs) == len(block_outputs) == 3
    torch.testing.assert_close(block_inputs[1], block_outputs[0])
    torch.testing.assert_close(block_inputs[2], block_outputs[1])
    assert not torch.equal(block_inputs[0], block_inputs[1])


def test_blocks_share_geometry_but_not_attention_ffn_norm_or_scale_parameters() -> None:
    model = _model(num_cone_blocks=3)
    first, second, third = model.encoder.blocks

    assert first.attention.neighbor_builder.base_radius == 1.0
    assert second.attention.neighbor_builder.base_radius == 1.0
    assert third.attention.neighbor_builder.base_radius == 1.0
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        assert getattr(first.attention, name) is not getattr(second.attention, name)
    assert first.ffn is not second.ffn
    assert first.attention_norm is not second.attention_norm
    assert first.ffn_norm is not second.ffn_norm
    assert first.attn_scale is not second.attn_scale
    torch.testing.assert_close(first.attn_scale, torch.ones(1))


def test_batch_column_is_preserved_and_never_merged_in_tokenization() -> None:
    output = _model()(_events())

    assert set(output["token_batch_indices"].tolist()) == {0, 1}
    first_batch_tokens = output["event_token_indices"][:3]
    second_batch_tokens = output["event_token_indices"][3:]
    assert not set(first_batch_tokens.tolist()) & set(second_batch_tokens.tolist())


def test_four_column_events_and_explicit_batch_indices_are_supported() -> None:
    events = _events()
    output = _model()(events[:, 1:], batch_indices=events[:, 0].long())

    assert output["event_logits"].shape == (5,)
    assert set(output["token_batch_indices"].tolist()) == {0, 1}


def test_empty_event_sample_returns_empty_finite_outputs() -> None:
    output = _model()(torch.empty((0, 5)))

    assert output["event_logits"].shape == (0,)
    assert output["token_features"].shape == (0, 16)
    assert output["token_coords"].shape == (0, 3)
    assert len(output["past_attention_stats"]) == 3


def test_causal_mode_has_no_future_attention_edges() -> None:
    output = _model(causal_mode=True)(_events())

    assert all(
        block["edge_count"].item() == 0
        for block in output["future_attention_stats"]
    )


def test_end_to_end_event_logits_backpropagate_through_the_detector() -> None:
    model = _model()

    model(_events())["event_logits"].square().mean().backward()

    parameters_with_grad = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert parameters_with_grad
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in parameters_with_grad
    )
