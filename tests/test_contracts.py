from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huav.data import EventSequence
from huav.inference import validate_prediction_logits


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
