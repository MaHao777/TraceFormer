from __future__ import annotations

import torch

from huav.config import LocalConeConfig
from huav.models import SpatioTemporalConeDetector
from huav.models.attention_residual import FullAttentionResidual


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
    for depth in (1, 6, 7):
        output = _model(num_cone_blocks=depth)(_events())

        assert output["event_logits"].shape == (5,)
        assert output["event_token_indices"].shape == (5,)
        assert output["token_features"].shape[1] == 16
        assert len(output["past_attention_stats"]) == depth


def test_six_blocks_attend_over_every_preceding_sublayer_increment() -> None:
    model = _model(num_cone_blocks=6, debug=False)
    attention_source_counts: list[int] = []
    ffn_source_counts: list[int] = []
    output_source_counts: list[int] = []
    handles = []
    for block in model.encoder.blocks:
        handles.append(
            block.attention_residual.register_forward_pre_hook(
                lambda _module, args: attention_source_counts.append(len(args[0]))
            )
        )
        handles.append(
            block.ffn_residual.register_forward_pre_hook(
                lambda _module, args: ffn_source_counts.append(len(args[0]))
            )
        )
    handles.append(
        model.encoder.output_residual.register_forward_pre_hook(
            lambda _module, args: output_source_counts.append(len(args[0]))
        )
    )

    model(_events())

    for handle in handles:
        handle.remove()
    assert attention_source_counts == [1, 3, 5, 7, 9, 11]
    assert ffn_source_counts == [2, 4, 6, 8, 10, 12]
    assert output_source_counts == [13]


def test_blocks_share_geometry_but_not_attention_ffn_norm_scale_or_depth_mixers() -> None:
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
    assert first.attention_residual is not second.attention_residual
    assert first.ffn_residual is not second.ffn_residual
    assert first.attention_residual.query is not second.attention_residual.query
    assert first.ffn_residual.key_norm is not second.ffn_residual.key_norm
    torch.testing.assert_close(first.attn_scale, torch.ones(1))


def test_six_block_attention_residual_parameter_count_and_initialization() -> None:
    model = _model(num_cone_blocks=6)
    mixers = [
        module
        for module in model.encoder.modules()
        if isinstance(module, FullAttentionResidual)
    ]

    assert len(mixers) == 13
    assert sum(
        parameter.numel()
        for mixer in mixers
        for parameter in mixer.parameters()
    ) == 13 * 2 * 16
    assert all(torch.count_nonzero(mixer.query).item() == 0 for mixer in mixers)


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


def test_bfloat16_autocast_keeps_depth_history_in_one_residual_dtype() -> None:
    model = _model(num_cone_blocks=3, debug=False)
    source_dtypes: list[set[torch.dtype]] = []
    handles = [
        mixer.register_forward_pre_hook(
            lambda _module, args: source_dtypes.append(
                {source.dtype for source in args[0]}
            )
        )
        for mixer in (
            model.encoder.blocks[0].attention_residual,
            model.encoder.blocks[0].ffn_residual,
            model.encoder.blocks[1].attention_residual,
            model.encoder.blocks[1].ffn_residual,
            model.encoder.blocks[2].attention_residual,
            model.encoder.blocks[2].ffn_residual,
            model.encoder.output_residual,
        )
    ]

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = model(_events())["event_logits"].square().mean()
    loss.backward()

    for handle in handles:
        handle.remove()
    assert len(source_dtypes) == 7
    assert all(len(dtypes) == 1 for dtypes in source_dtypes)
    assert all(dtypes == source_dtypes[0] for dtypes in source_dtypes)
