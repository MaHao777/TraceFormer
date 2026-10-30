from __future__ import annotations

import torch

from traceformer.config import RefinementScaleConfig
from traceformer.models import TraceFormer


def _model() -> TraceFormer:
    return TraceFormer(
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


def test_fixed_model_has_no_sparse_parameters_and_retains_refinement() -> None:
    model = _model()

    assert not hasattr(model, "sparse_mixture")
    assert not any("sparse_mixture" in key for key in model.state_dict())
    assert model.event_refinement_head is not None


def test_training_stage_full_keeps_every_module_trainable() -> None:
    model = _model()
    model.configure_training_stage("full")
    model.train()

    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(module.training for module in model.children())




def test_training_stage_refinement_opens_only_new_head() -> None:
    model = _model()
    model.configure_training_stage("refinement")
    model.train()

    assert model.event_refinement_head.training
    assert all(
        parameter.requires_grad for parameter in model.event_refinement_head.parameters()
    )
    for module in (model.tokenizer, model.encoder, model.decoder):
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
