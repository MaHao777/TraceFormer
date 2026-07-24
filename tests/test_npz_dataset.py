from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from huav.data import NPZEventDataset


def _write_npz(path: Path) -> None:
    dtype = [
        ("x", "<i2"),
        ("y", "<i2"),
        ("t", "<f8"),
        ("p", "i1"),
        ("label", "i1"),
        ("name", "i1"),
    ]
    events = np.array(
        [
            (1, 2, 0.0, 1, 0, 0),
            (3, 4, 4000.0, 0, 1, 2),
            (5, 6, 8000.0, 1, 1, 2),
        ],
        dtype=dtype,
    )
    np.savez(path, ev=events)


def test_npz_dataset_preserves_order_and_normalizes_eight_second_time(tmp_path: Path) -> None:
    _write_npz(tmp_path / "sample.npz")
    item = NPZEventDataset(tmp_path, width=8, height=7, duration_ms=8000)[0]

    assert item.events.shape == (3, 4)
    assert item.events[:, :2].tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert item.events[:, 2].tolist() == [0.0, 0.5, 1.0]
    assert item.labels.tolist() == [0.0, 1.0, 1.0]
    assert item.object_ids.tolist() == [0, 2, 2]
    assert item.timestamps_ms.dtype == torch.float64
    assert item.source_path.name == "sample.npz"


def test_npz_dataset_rejects_out_of_range_coordinates_or_timestamps(tmp_path: Path) -> None:
    _write_npz(tmp_path / "sample.npz")
    with np.load(tmp_path / "sample.npz") as loaded:
        events = loaded["ev"].copy()
    events["x"][0] = 8
    np.savez(tmp_path / "sample.npz", ev=events)

    dataset = NPZEventDataset(tmp_path, width=8, height=7, duration_ms=8000)
    with pytest.raises(ValueError, match="coordinates"):
        dataset[0]


def test_npz_dataset_rejects_noninteger_coordinate_fields(tmp_path: Path) -> None:
    dtype = [
        ("x", "<f4"),
        ("y", "<i2"),
        ("t", "<f8"),
        ("p", "i1"),
        ("label", "i1"),
        ("name", "i1"),
    ]
    np.savez(
        tmp_path / "sample.npz",
        ev=np.array([(1.5, 2, 0.0, 1, 0, 0)], dtype=dtype),
    )
    with pytest.raises(TypeError, match="integer"):
        NPZEventDataset(tmp_path, width=8, height=7)[0]
