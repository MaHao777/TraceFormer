from __future__ import annotations

import torch

from huav.models import DualViewEventClassifier


def _model() -> DualViewEventClassifier:
    return DualViewEventClassifier(
        width=8,
        height=6,
        time_bins=(4, 2),
        feature_channels=8,
        attention_heads=2,
        window_size=(2, 4),
        cross_time_radius=1,
        dropout=0.0,
    )


def test_forward_returns_all_required_event_and_view_outputs() -> None:
    events = torch.tensor(
        [
            [1.0, 2.0, 0.00, 1.0],
            [2.0, 3.0, 0.25, 0.0],
            [6.0, 4.0, 1.00, 1.0],
        ],
        dtype=torch.float32,
    )
    output = _model()(events, torch.zeros(3, dtype=torch.long), batch_size=1)

    assert set(output) == {
        "event_logits",
        "xt_logits",
        "yt_logits",
        "xt_event_logits",
        "yt_event_logits",
        "event_features",
    }
    assert output["event_logits"].shape == (3,)
    assert output["xt_event_logits"].shape == (3,)
    assert output["yt_event_logits"].shape == (3,)
    assert output["event_features"].shape == (3, 34)
    assert output["xt_logits"].shape == (1, 1, 4, 8)
    assert output["yt_logits"].shape == (1, 1, 4, 6)


def test_encoders_are_independent_and_gradients_reach_both_views() -> None:
    model = _model()
    assert next(model.xt_encoder.parameters()) is not next(model.yt_encoder.parameters())

    events = torch.tensor(
        [[1.0, 2.0, 0.0, 1.0], [6.0, 4.0, 1.0, 0.0]], dtype=torch.float32
    )
    output = model(events, torch.zeros(2, dtype=torch.long), batch_size=1)
    output["event_logits"].sum().backward()

    assert any(parameter.grad is not None for parameter in model.xt_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.yt_encoder.parameters())


def test_query_indices_limit_training_outputs_without_changing_projection_maps() -> None:
    model = _model()
    events = torch.tensor(
        [
            [1.0, 2.0, 0.0, 1.0],
            [2.0, 3.0, 0.5, 0.0],
            [6.0, 4.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    output = model(
        events,
        torch.zeros(3, dtype=torch.long),
        batch_size=1,
        query_indices=torch.tensor([0, 2]),
    )
    assert output["event_logits"].shape == (2,)
    assert output["xt_logits"].shape == (1, 1, 4, 8)


def test_temporal_query_uses_projection_bin_centers() -> None:
    model = _model()
    feature = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
    # For T=4, the four projection-cell centers are .125, .375, .625 and .875.
    events = torch.tensor(
        [[0.0, 0.0, 0.125, 1.0], [0.0, 0.0, 0.375, 1.0]],
        dtype=torch.float32,
    )
    sampled = model._query(
        feature,
        events,
        torch.zeros(2, dtype=torch.long),
        coordinate_column=0,
        spatial_size=1,
    )
    assert torch.allclose(sampled[:, 0], torch.tensor([0.0, 1.0]))
