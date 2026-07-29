from __future__ import annotations

import torch

from huav.config import ConeHeadConfig
from huav.models import SpatioTemporalConeDetector


def _model(
    *,
    causal_mode: bool = False,
    debug: bool = True,
    adaptive_threshold_mode: str = "none",
) -> SpatioTemporalConeDetector:
    return SpatioTemporalConeDetector(
        width=16,
        height=12,
        spatial_block_size_x=4,
        spatial_block_size_y=4,
        num_time_blocks=4,
        token_dim=16,
        event_embed_dim=8,
        num_layers=3,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        cone_heads=(
            ConeHeadConfig(1.0, 0.5, 2),
            ConeHeadConfig(1.0, 1.0, 2),
            ConeHeadConfig(2.0, 1.5, 3),
            ConeHeadConfig(2.0, 2.0, 3),
        ),
        layer_time_scales=(0.5, 0.75, 1.0),
        max_neighbors_per_cone=3,
        causal_mode=causal_mode,
        direction_dropout=0.0,
        debug_attention_stats=debug,
        adaptive_threshold_mode=adaptive_threshold_mode,
        threshold_hidden_dim=8,
        threshold_time_bins=4,
        threshold_max_global_offset=0.5,
        threshold_max_local_offset=0.75,
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


def test_forward_returns_required_event_token_and_attention_outputs() -> None:
    output = _model()(_events())

    assert {
        "event_logits",
        "base_event_logits",
        "global_threshold_offsets",
        "local_threshold_offsets",
        "threshold_group_indices",
        "threshold_group_coords",
        "threshold_group_batch_indices",
        "threshold_neighbor_edges",
        "threshold_group_descriptors",
        "token_features",
        "token_coords",
        "token_batch_indices",
        "event_token_indices",
        "past_attention_stats",
        "future_attention_stats",
    } == set(output)
    assert output["event_logits"].shape == (5,)
    assert output["event_logits"] is output["base_event_logits"]
    assert output["global_threshold_offsets"].shape == (5,)
    assert output["local_threshold_offsets"].shape == (5,)
    assert output["threshold_group_indices"].shape == (5,)
    assert output["threshold_group_coords"].shape[1] == 3
    assert output["threshold_group_batch_indices"].ndim == 1
    assert output["threshold_neighbor_edges"].shape == (2, 0)
    assert output["threshold_group_descriptors"].shape[0] == len(
        output["threshold_group_coords"]
    )
    assert output["event_token_indices"].shape == (5,)
    assert output["token_features"].shape[1] == 16
    assert len(output["past_attention_stats"]) == 3
    assert len(output["future_attention_stats"]) == 3


def test_batch_column_is_preserved_and_never_merged_in_tokenization() -> None:
    output = _model()(_events())

    assert set(output["token_batch_indices"].tolist()) == {0, 1}
    first_batch_tokens = output["event_token_indices"][:3]
    second_batch_tokens = output["event_token_indices"][3:]
    assert not set(first_batch_tokens.tolist()) & set(second_batch_tokens.tolist())
    first_threshold_groups = output["threshold_group_indices"][:3]
    second_threshold_groups = output["threshold_group_indices"][3:]
    assert not set(first_threshold_groups.tolist()) & set(
        second_threshold_groups.tolist()
    )
    assert set(output["threshold_group_batch_indices"].tolist()) == {0, 1}


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
    assert output["base_event_logits"].shape == (0,)
    assert output["global_threshold_offsets"].shape == (0,)
    assert output["local_threshold_offsets"].shape == (0,)
    assert output["threshold_group_indices"].shape == (0,)
    assert output["threshold_group_coords"].shape == (0, 3)
    assert output["threshold_group_batch_indices"].shape == (0,)
    assert output["threshold_neighbor_edges"].shape == (2, 0)
    assert output["threshold_group_descriptors"].shape == (0, 31)
    assert len(output["past_attention_stats"]) == 3


def test_causal_mode_has_no_future_attention_edges() -> None:
    output = _model(causal_mode=True)(_events())

    assert all(
        layer["edge_count"].item() == 0
        for layer in output["future_attention_stats"]
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


def test_hierarchical_calibration_shares_offsets_and_preserves_base_gradient_path() -> None:
    model = _model(adaptive_threshold_mode="hierarchical")
    output = model(_events())

    group_indices = output["threshold_group_indices"]
    for group_index in torch.unique(group_indices):
        mask = group_indices == group_index
        assert torch.unique(output["local_threshold_offsets"][mask]).numel() == 1
        assert torch.unique(output["global_threshold_offsets"][mask]).numel() == 1

    output["event_logits"].sum().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.calibrator.parameters()
    )
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.decoder.parameters()
    )
