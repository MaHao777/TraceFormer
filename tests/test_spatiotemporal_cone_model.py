from __future__ import annotations

import torch

from huav.config import ConeHeadConfig
from huav.models import SpatioTemporalConeDetector


def _model(
    *,
    causal_mode: bool = False,
    debug: bool = True,
    sparse_mixture: bool = False,
    frame_objectness: bool = False,
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
        sparse_mixture_enabled=sparse_mixture,
        sparse_mixture_router_bias=-2.0,
        sparse_mixture_residual_scale=0.01,
        frame_objectness_enabled=frame_objectness,
        frame_objectness_frame_ms=50.0,
        duration_ms=100.0,
        frame_objectness_block_size_x=4,
        frame_objectness_block_size_y=4,
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


def test_freeze_full_path_keeps_only_sparse_mixture_trainable_and_full_eval() -> None:
    model = _model(sparse_mixture=True)
    model.freeze_full_path()
    model.train()

    for module in (model.tokenizer, model.encoder, model.decoder):
        assert not module.training
        assert not any(parameter.requires_grad for parameter in module.parameters())
    assert model.sparse_mixture is not None
    assert model.sparse_mixture.training
    assert all(
        parameter.requires_grad for parameter in model.sparse_mixture.parameters()
    )


def test_forward_returns_required_event_token_and_attention_outputs() -> None:
    output = _model()(_events())

    assert {
        "event_logits",
        "base_event_logits",
        "frame_objectness_cell_logits",
        "frame_objectness_event_cell_indices",
        "frame_objectness_event_logits",
        "full_event_logits",
        "sparse_event_logits",
        "sparse_source_indices",
        "mixture_gate",
        "sparse_token_present",
        "token_features",
        "token_coords",
        "token_batch_indices",
        "event_token_indices",
        "past_attention_stats",
        "future_attention_stats",
    } == set(output)
    assert output["event_logits"].shape == (5,)
    assert torch.equal(output["event_logits"], output["full_event_logits"])
    assert torch.equal(output["event_logits"], output["base_event_logits"])
    assert torch.equal(output["frame_objectness_event_logits"], torch.zeros(5))
    assert output["sparse_event_logits"].shape == (0,)
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


def test_paired_sparse_view_preserves_full_event_order_and_trains_router() -> None:
    model = _model(sparse_mixture=True)
    events = _events()
    source_indices = torch.tensor([0, 1, 3, 4])
    sparse_events = events[source_indices]

    output = model(
        events,
        sparse_events=sparse_events,
        sparse_source_indices=source_indices,
        mixture_weight=1.0,
    )

    assert output["event_logits"].shape == (len(events),)
    assert output["full_event_logits"].shape == (len(events),)
    assert output["sparse_event_logits"].shape == (len(source_indices),)
    assert torch.equal(output["sparse_source_indices"], source_indices)
    assert output["mixture_gate"].ndim == 1
    assert bool(output["sparse_token_present"].any())
    output["event_logits"].square().mean().backward()
    mixture_parameters = list(model.sparse_mixture.parameters())
    assert mixture_parameters
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in mixture_parameters
    )


def test_frame_objectness_zero_initialization_preserves_base_logits() -> None:
    output = _model(frame_objectness=True)(_events())

    torch.testing.assert_close(output["event_logits"], output["base_event_logits"])
    assert output["frame_objectness_cell_logits"].ndim == 1
    assert output["frame_objectness_event_cell_indices"].shape == (5,)
    assert torch.equal(output["frame_objectness_event_logits"], torch.zeros(5))


def test_freeze_base_for_objectness_keeps_only_new_head_trainable() -> None:
    model = _model(sparse_mixture=True, frame_objectness=True)
    model.freeze_base_for_objectness()
    model.train()

    for module in (model.tokenizer, model.encoder, model.decoder, model.sparse_mixture):
        assert module is not None
        assert not module.training
        assert not any(parameter.requires_grad for parameter in module.parameters())
    assert model.frame_objectness_head is not None
    assert model.frame_objectness_head.training
    assert all(
        parameter.requires_grad
        for parameter in model.frame_objectness_head.parameters()
    )
