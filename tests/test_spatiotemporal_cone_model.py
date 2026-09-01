from __future__ import annotations

import torch

from huav.config import ConeHeadConfig, RefinementScaleConfig
from huav.models import SpatioTemporalConeDetector


def _model() -> SpatioTemporalConeDetector:
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
        direction_dropout=0.0,
        sparse_mixture_router_bias=-2.0,
        sparse_mixture_residual_scale=0.01,
        frame_refinement_context_dim=4,
        frame_refinement_scales=(
            RefinementScaleConfig(2, 2, 10.0),
            RefinementScaleConfig(8, 8, 50.0),
        ),
        duration_ms=100.0,
    )


def _events() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 0.10, 1.0],
            [2.0, 2.0, 0.40, 0.0],
            [4.0, 3.0, 0.70, 1.0],
            [1.0, 2.0, 0.10, 1.0],
            [2.0, 2.0, 0.40, 0.0],
        ]
    )


def _batch_indices() -> torch.Tensor:
    return torch.tensor([0, 0, 0, 1, 1])


def test_fixed_model_always_contains_sparse_mixture_and_refinement() -> None:
    model = _model()

    assert model.sparse_mixture is not None
    assert model.event_refinement_head is not None


def test_training_stage_full_keeps_every_module_trainable() -> None:
    model = _model()
    model.configure_training_stage("full")
    model.train()

    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(module.training for module in model.children())


def test_training_stage_sparse_mixture_opens_only_adapter() -> None:
    model = _model()
    model.configure_training_stage("sparse_mixture")
    model.train()

    assert model.sparse_mixture.training
    assert all(parameter.requires_grad for parameter in model.sparse_mixture.parameters())
    for module in (
        model.tokenizer,
        model.encoder,
        model.decoder,
        model.event_refinement_head,
    ):
        assert not module.training
        assert not any(parameter.requires_grad for parameter in module.parameters())


def test_training_stage_refinement_opens_only_new_head() -> None:
    model = _model()
    model.configure_training_stage("refinement")
    model.train()

    assert model.event_refinement_head.training
    assert all(
        parameter.requires_grad for parameter in model.event_refinement_head.parameters()
    )
    for module in (model.tokenizer, model.encoder, model.decoder, model.sparse_mixture):
        assert not module.training
        assert not any(parameter.requires_grad for parameter in module.parameters())


def test_unknown_training_stage_is_rejected() -> None:
    model = _model()

    try:
        model.configure_training_stage("disabled")
    except ValueError as error:
        assert "unsupported training stage" in str(error)
    else:
        raise AssertionError("unknown training stage was accepted")


def test_forward_returns_only_fixed_model_training_contract() -> None:
    output = _model()(_events(), _batch_indices())

    assert set(output) == {
        "event_logits",
        "base_event_logits",
        "event_refinement_cell_logits",
        "event_refinement_event_cell_indices",
        "event_refinement_event_logits",
        "full_event_logits",
        "sparse_event_logits",
        "sparse_source_indices",
        "mixture_gate",
        "sparse_token_present",
        "token_features",
        "token_coords",
        "token_batch_indices",
        "event_token_indices",
    }
    assert output["event_logits"].shape == (5,)
    torch.testing.assert_close(output["event_logits"], output["base_event_logits"])
    assert output["event_refinement_cell_logits"].ndim == 1
    assert output["event_refinement_event_cell_indices"].shape == (5,)
    assert torch.equal(output["event_refinement_event_logits"], torch.zeros(5))
    assert output["sparse_event_logits"].shape == (0,)


def test_paired_sparse_view_preserves_event_order_and_trains_router() -> None:
    model = _model()
    events = _events()
    source_indices = torch.tensor([0, 1, 3, 4])

    output = model(
        events,
        _batch_indices(),
        sparse_events=events[source_indices],
        sparse_batch_indices=_batch_indices()[source_indices],
        sparse_source_indices=source_indices,
        mixture_weight=1.0,
    )

    assert output["event_logits"].shape == (len(events),)
    assert output["sparse_event_logits"].shape == (len(source_indices),)
    assert torch.equal(output["sparse_source_indices"], source_indices)
    assert bool(output["sparse_token_present"].any())
    output["event_logits"].square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.sparse_mixture.parameters()
    )


def test_explicit_batch_indices_stay_isolated() -> None:
    events = _events()
    explicit = _model()(events, batch_indices=_batch_indices())

    assert set(explicit["token_batch_indices"].tolist()) == {0, 1}
    first = explicit["event_token_indices"][:3]
    second = explicit["event_token_indices"][3:]
    assert not set(first.tolist()) & set(second.tolist())


def test_empty_event_sample_returns_empty_finite_outputs() -> None:
    output = _model()(torch.empty((0, 4)))

    assert output["event_logits"].shape == (0,)
    assert output["token_features"].shape == (0, 16)
    assert output["token_coords"].shape == (0, 3)


def test_end_to_end_event_logits_backpropagate_through_fixed_model() -> None:
    model = _model()
    model.configure_training_stage("full")

    model(_events(), _batch_indices())["event_logits"].square().mean().backward()

    parameters_with_grad = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert parameters_with_grad
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters_with_grad)
