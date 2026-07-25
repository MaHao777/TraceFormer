from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huav.data import EventBatch, EventSequence, EventWindow
from huav.inference import validate_prediction_logits
from huav.models import ModelOutput


def raises(exc_type: type[Exception], fn: object) -> None:
    try:
        fn()  # type: ignore[operator]
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def sequence(**changes: object) -> EventSequence:
    values: dict[str, object] = {
        "x": np.array([1, 2, 3], dtype=np.int16),
        "y": np.array([4, 5, 6], dtype=np.int16),
        "t": np.array([0.0, 0.01, 0.02], dtype=np.float64),
        "p": np.array([0, 1, 0], dtype=np.uint8),
        "width": 346,
        "height": 260,
        "label": np.array([0, 1, 0], dtype=np.uint8),
        "object_id": np.array([9, 9, 10], dtype=np.int64),
        "source_name": "train_001.npz",
    }
    values.update(changes)
    return EventSequence(**values)  # type: ignore[arg-type]


def test_event_sequence_keeps_labels_out_of_model_inputs() -> None:
    item = sequence()
    assert item.length == 3
    assert set(item.inputs) == {"x", "y", "t", "p"}
    assert item.inputs["x"].tolist() == [1, 2, 3]
    assert item.source_stem == "train_001"


def test_event_sequence_rejects_shape_dtype_value_time_and_resolution_errors() -> None:
    raises(ValueError, lambda: sequence(y=np.array([[1, 2, 3]], dtype=np.int16)))
    raises(TypeError, lambda: sequence(x=np.array([1.0, 2.0, 3.0], dtype=np.float32)))
    raises(ValueError, lambda: sequence(p=np.array([0, 2, 1], dtype=np.uint8)))
    raises(ValueError, lambda: sequence(t=np.array([0.0, 0.02, 0.01], dtype=np.float64)))
    raises(ValueError, lambda: sequence(x=np.array([1, 346, 3], dtype=np.int16)))
    raises(ValueError, lambda: sequence(width=0))
    raises(ValueError, lambda: sequence(label=np.array([0, 1], dtype=np.uint8)))


def test_event_sequence_rejects_non_integer_metadata_and_complex_timestamps_even_when_empty() -> None:
    empty = dict(
        x=np.array([], dtype=np.int16), y=np.array([], dtype=np.int16),
        t=np.array([], dtype=np.float64), p=np.array([], dtype=np.uint8),
    )
    raises(ValueError, lambda: EventSequence(**empty, width=True, height=260))
    raises(ValueError, lambda: EventSequence(**empty, width=346.0, height=260))
    raises(TypeError, lambda: sequence(t=np.array([0.0 + 1.0j, 0.01 + 0j, 0.02 + 0j])))


def test_event_sequence_rejects_descending_unsigned_timestamps_and_accepts_ascending() -> None:
    raises(ValueError, lambda: sequence(t=np.array([2, 1, 0], dtype=np.uint32)))
    item = sequence(t=np.array([0, 1, 2], dtype=np.uint64))
    assert item.t.tolist() == [0, 1, 2]


def test_event_window_validates_volume_mapping_and_optional_labels() -> None:
    window = EventWindow(
        volume=torch.zeros((4, 6, 7), dtype=torch.float32),
        event_indices=torch.tensor([0, 2], dtype=torch.int64),
        voxel_indices=torch.tensor([[0, 4, 1], [1, 5, 2]], dtype=torch.int64),
        time_bins=2,
        height=6,
        width=7,
        sequence_length=3,
        labels=torch.tensor([0, 1], dtype=torch.uint8),
    )
    assert window.event_indices.tolist() == [0, 2]
    raises(ValueError, lambda: EventWindow(torch.zeros((3, 6, 7)), torch.tensor([0]), torch.tensor([[0, 1, 1]]), 2, 6, 7))
    raises(TypeError, lambda: EventWindow(torch.zeros((4, 6, 7), dtype=torch.float64), torch.tensor([0]), torch.tensor([[0, 1, 1]]), 2, 6, 7))
    raises(ValueError, lambda: EventWindow(torch.zeros((4, 6, 7)), torch.tensor([3]), torch.tensor([[0, 1, 1]]), 2, 6, 7, sequence_length=3))
    raises(ValueError, lambda: EventWindow(torch.zeros((4, 6, 7)), torch.tensor([0]), torch.tensor([[2, 1, 1]]), 2, 6, 7))


def test_event_window_rejects_bad_scalar_metadata_and_duplicate_event_indices_when_empty_or_populated() -> None:
    empty_indices = torch.empty((0,), dtype=torch.int64)
    empty_voxels = torch.empty((0, 3), dtype=torch.int64)
    raises(ValueError, lambda: EventWindow(torch.zeros((4, 6, 7)), empty_indices, empty_voxels, True, 6, 7))
    raises(ValueError, lambda: EventWindow(torch.zeros((4, 6, 7)), empty_indices, empty_voxels, 2, 6, 7, sequence_length=-1))
    raises(ValueError, lambda: EventWindow(torch.zeros((4, 6, 7)), torch.tensor([0, 0]), torch.tensor([[0, 1, 1], [1, 1, 1]]), 2, 6, 7))


def test_event_batch_preserves_source_event_mapping_and_rejects_bad_alignment() -> None:
    batch = EventBatch(
        volumes=torch.zeros((2, 4, 6, 7), dtype=torch.float32),
        batch_indices=torch.tensor([0, 1, 1], dtype=torch.int64),
        event_indices=torch.tensor([2, 0, 1], dtype=torch.int64),
        voxel_indices=torch.tensor([[0, 4, 1], [1, 5, 2], [1, 5, 3]], dtype=torch.int64),
        time_bins=2,
        height=6,
        width=7,
        sequence_length=3,
    )
    assert batch.event_indices.tolist() == [2, 0, 1]
    raises(ValueError, lambda: EventBatch(torch.zeros((2, 4, 6, 7)), torch.tensor([2]), torch.tensor([0]), torch.tensor([[0, 0, 0]]), 2, 6, 7, 3))
    raises(ValueError, lambda: EventBatch(torch.zeros((2, 4, 6, 7)), torch.tensor([0]), torch.tensor([3]), torch.tensor([[0, 0, 0]]), 2, 6, 7, 3))


def test_event_batch_rejects_bad_scalar_metadata_and_duplicate_pairs_but_allows_overlap() -> None:
    empty = torch.empty((0,), dtype=torch.int64)
    empty_voxels = torch.empty((0, 3), dtype=torch.int64)
    raises(ValueError, lambda: EventBatch(torch.zeros((1, 4, 6, 7)), empty, empty, empty_voxels, 2.0, 6, 7, 0))
    raises(ValueError, lambda: EventBatch(torch.zeros((1, 4, 6, 7)), empty, empty, empty_voxels, 2, 6, 7, -1))
    raises(ValueError, lambda: EventBatch(torch.zeros((2, 4, 6, 7)), torch.tensor([0, 0]), torch.tensor([1, 1]), torch.tensor([[0, 1, 1], [1, 1, 1]]), 2, 6, 7, 2))
    allowed = EventBatch(
        torch.zeros((2, 4, 6, 7)), torch.tensor([0, 1]), torch.tensor([1, 1]),
        torch.tensor([[0, 1, 1], [1, 1, 1]]), 2, 6, 7, 2,
    )
    assert allowed.event_indices.tolist() == [1, 1]


def test_model_output_requires_float_logits_with_batched_bin_shape() -> None:
    output = ModelOutput(torch.zeros((2, 2, 6, 7), dtype=torch.float32))
    assert output.voxel_logits.shape == (2, 2, 6, 7)
    raises(ValueError, lambda: ModelOutput(torch.zeros((2, 6, 7), dtype=torch.float32)))
    raises(TypeError, lambda: ModelOutput(torch.zeros((2, 2, 6, 7), dtype=torch.int64)))
    raises(ValueError, lambda: ModelOutput(torch.zeros((0, 2, 6, 7), dtype=torch.float32)))


def test_prediction_logits_are_one_per_original_event_in_original_order() -> None:
    item = sequence()
    logits = validate_prediction_logits(item, torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32))
    assert torch.allclose(logits, torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32))
    raises(ValueError, lambda: validate_prediction_logits(item, torch.tensor([0.1, 0.2], dtype=torch.float32)))
    raises(TypeError, lambda: validate_prediction_logits(item, torch.tensor([0, 1, 0], dtype=torch.int64)))


def test_prediction_logits_reject_nonfinite_values_and_accept_empty_sequence() -> None:
    item = sequence()
    raises(ValueError, lambda: validate_prediction_logits(item, torch.tensor([0.0, float("nan"), 0.0])))
    raises(ValueError, lambda: validate_prediction_logits(item, torch.tensor([0.0, float("inf"), 0.0])))
    empty = EventSequence(
        x=np.array([], dtype=np.int16), y=np.array([], dtype=np.int16),
        t=np.array([], dtype=np.float64), p=np.array([], dtype=np.uint8), width=346, height=260,
    )
    assert validate_prediction_logits(empty, torch.empty((0,), dtype=torch.float32)).numel() == 0


def test_prediction_logits_cuda_nonfinite_validation_when_available() -> None:
    if not torch.cuda.is_available():
        return
    item = sequence()
    raises(ValueError, lambda: validate_prediction_logits(item, torch.tensor([0.0, float("nan"), 0.0], device="cuda")))
