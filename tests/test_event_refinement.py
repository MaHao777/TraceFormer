from __future__ import annotations

import torch

from huav.config import RefinementScaleConfig
from huav.models.event_refinement import MultiScaleEventRefinementHead


def _head() -> MultiScaleEventRefinementHead:
    return MultiScaleEventRefinementHead(
        token_dim=4,
        event_feature_dim=6,
        context_dim=4,
        width=16,
        height=16,
        duration_ms=100.0,
        scales=(
            RefinementScaleConfig(2, 2, 10.0),
            RefinementScaleConfig(8, 8, 50.0),
        ),
        dropout=0.0,
    )


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "token_features": torch.randn(2, 4),
        "event_token_indices": torch.tensor([0, 0, 1, 0]),
        "event_local_features": torch.randn(4, 6),
        "event_xy": torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [9.0, 1.0], [1.0, 1.0]]
        ),
        "normalized_time": torch.tensor([0.10, 0.20, 0.20, 0.60]),
        "event_batch_indices": torch.zeros(4, dtype=torch.long),
        "base_event_logits": torch.randn(4),
    }


def test_refinement_groups_primary_cells_and_starts_as_identity() -> None:
    output = _head()(**_inputs())

    indices = output.event_cell_indices.tolist()
    assert indices[0] == indices[1]
    assert indices[2] != indices[0]
    assert indices[3] not in {indices[0], indices[2]}
    assert output.cell_logits.shape == (3,)
    assert output.event_logits.shape == (4,)
    assert torch.equal(output.cell_logits, torch.zeros(3))
    assert torch.equal(output.event_logits, torch.zeros(4))


def test_refinement_uses_previous_and_following_cells_without_crossing_batches() -> None:
    head = _head()
    captured: list[torch.Tensor] = []
    hook = head.temporal_fusers[-1][0].register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach().clone())
    )
    inputs = _inputs()
    inputs["event_batch_indices"] = torch.tensor([0, 0, 0, 1])
    inputs["normalized_time"] = torch.tensor([0.10, 0.60, 0.20, 0.60])
    head(**inputs)
    hook.remove()

    temporal = captured[0]
    context_dim = head.context_dim
    previous = temporal[:, :context_dim]
    following = temporal[:, 2 * context_dim :]
    assert bool((previous.abs().sum(dim=1) > 0).any())
    assert bool((following.abs().sum(dim=1) > 0).any())
    batch_one_row = temporal[-1]
    assert torch.equal(batch_one_row[:context_dim], torch.zeros(context_dim))


def test_refinement_can_assign_different_residuals_inside_one_primary_cell() -> None:
    head = _head()
    with torch.no_grad():
        head.event_classifier[-1].weight.fill_(1.0)
    inputs = _inputs()
    inputs["event_xy"][:2] = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    inputs["normalized_time"][:2] = torch.tensor([0.10, 0.20])
    inputs["event_local_features"][1] += 4.0

    output = head(**inputs)

    assert output.event_cell_indices[0] == output.event_cell_indices[1]
    assert output.event_logits[0] != output.event_logits[1]


def test_refinement_empty_input_is_safe() -> None:
    output = _head()(
        token_features=torch.empty((0, 4)),
        event_token_indices=torch.empty(0, dtype=torch.long),
        event_local_features=torch.empty((0, 6)),
        event_xy=torch.empty((0, 2)),
        normalized_time=torch.empty(0),
        event_batch_indices=torch.empty(0, dtype=torch.long),
        base_event_logits=torch.empty(0),
    )

    assert output.cell_logits.shape == (0,)
    assert output.event_cell_indices.shape == (0,)
    assert output.event_logits.shape == (0,)


def test_refinement_cell_and_event_paths_are_differentiable() -> None:
    head = _head()
    output = head(**_inputs())

    (output.cell_logits.sum() + output.event_logits.sum()).backward()

    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )
