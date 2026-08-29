from __future__ import annotations

import torch

from huav.models.frame_objectness import FrameCellObjectnessHead


def test_frame_objectness_groups_events_by_50ms_and_spatial_cell() -> None:
    head = FrameCellObjectnessHead(
        token_dim=4,
        event_feature_dim=6,
        width=16,
        height=16,
        spatial_block_size_x=8,
        spatial_block_size_y=8,
        duration_ms=100.0,
        frame_ms=50.0,
        dropout=0.0,
    )
    output = head(
        torch.randn(2, 4),
        torch.tensor([0, 0, 1, 0]),
        torch.randn(4, 6),
        torch.tensor([[1.0, 1.0], [2.0, 2.0], [9.0, 1.0], [1.0, 1.0]]),
        torch.tensor([0.10, 0.20, 0.20, 0.60]),
        torch.zeros(4, dtype=torch.long),
    )

    indices = output.event_cell_indices.tolist()
    assert indices[0] == indices[1]
    assert indices[2] != indices[0]
    assert indices[3] not in {indices[0], indices[2]}
    assert output.cell_logits.shape == (3,)
    assert output.event_logits.shape == (4,)


def test_frame_objectness_zero_initialization_preserves_event_logits() -> None:
    head = FrameCellObjectnessHead(
        token_dim=4,
        event_feature_dim=6,
        width=16,
        height=16,
        spatial_block_size_x=8,
        spatial_block_size_y=8,
        duration_ms=100.0,
        frame_ms=50.0,
        dropout=0.0,
    )

    output = head(
        torch.randn(1, 4),
        torch.zeros(2, dtype=torch.long),
        torch.randn(2, 6),
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        torch.tensor([0.10, 0.20]),
        torch.zeros(2, dtype=torch.long),
    )

    assert torch.equal(output.cell_logits, torch.zeros(1))
    assert torch.equal(output.event_logits, torch.zeros(2))


def test_frame_objectness_empty_input_is_safe() -> None:
    head = FrameCellObjectnessHead(
        token_dim=4,
        event_feature_dim=6,
        width=16,
        height=16,
        spatial_block_size_x=8,
        spatial_block_size_y=8,
        duration_ms=100.0,
        frame_ms=50.0,
        dropout=0.0,
    )

    output = head(
        torch.empty((0, 4)),
        torch.empty(0, dtype=torch.long),
        torch.empty((0, 6)),
        torch.empty((0, 2)),
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
    )

    assert output.cell_logits.shape == (0,)
    assert output.event_cell_indices.shape == (0,)
    assert output.event_logits.shape == (0,)
