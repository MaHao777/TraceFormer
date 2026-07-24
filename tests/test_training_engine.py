from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from huav.config import (
    DatasetConfig,
    HUAVConfig,
    ModelConfig,
    TrainingConfig,
)
from huav.training.engine import evaluate_checkpoint, run_training


def _write_sequence(path: Path, offset: int) -> None:
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
            (1 + offset, 1, 0.0, 1, 1, 1),
            (2 + offset, 2, 2000.0, 0, 0, 0),
            (3 + offset, 2, 4000.0, 1, 1, 1),
            (4 + offset, 3, 6000.0, 0, 0, 0),
        ],
        dtype=dtype,
    )
    np.savez(path, ev=events)


def _config(data_root: Path, output_root: Path) -> HUAVConfig:
    return replace(
        HUAVConfig.default(),
        resolution=replace(HUAVConfig.default().resolution, width=8, height=6),
        dataset=DatasetConfig(
            root=Path(os.path.relpath(data_root, Path.cwd())), duration_ms=8000
        ),
        model=ModelConfig(
            time_bins=(4, 2),
            feature_channels=8,
            attention_heads=2,
            window_time=2,
            window_space=4,
            cross_time_radius=1,
            dropout=0.0,
        ),
        training=TrainingConfig(
            device="cpu",
            epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            seed=7,
            background_ratio=2,
            num_threads=1,
            output_root=output_root,
            auxiliary_warmup_epochs=0,
            auxiliary_ramp_epochs=1,
            target_sigma=1.0,
            gradient_clip_norm=1.0,
            max_train_files=None,
            max_val_files=None,
        ),
    ).validate()


def test_cpu_training_writes_reproducible_metrics_and_checkpoint() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        train_root = data_root / "train"
        val_root = data_root / "val"
        train_root.mkdir(parents=True)
        val_root.mkdir(parents=True)
        _write_sequence(train_root / "train_000.npz", 0)
        _write_sequence(val_root / "val_000.npz", 1)
        output_root = temporary_root / "outputs"

        summary = run_training(
            _config(data_root, output_root), output_dir=output_root / "run"
        )
        run_dir = Path(summary["run_dir"])
        assert summary["device"] == "cpu"
        assert (run_dir / "best.pt").is_file()
        assert (run_dir / "history.json").is_file()
        assert (run_dir / "metrics.json").is_file()
        assert (run_dir / "metrics.md").is_file()
        manifest = json.loads(
            (run_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        assert all(len(item["sha256"]) == 64 for item in manifest["files"])

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["event"]["event_count"] == 4
        assert {"pd", "fa", "iou", "acc"}.issubset(metrics["event"])
        checkpoint = __import__("torch").load(
            run_dir / "best.pt", map_location="cpu", weights_only=True
        )
        assert checkpoint["decision_threshold"] == metrics["threshold"]
        changed_config = replace(
            _config(data_root, output_root),
            inference=replace(
                _config(data_root, output_root).inference, sigmoid_threshold=0.99
            ),
            evaluation=replace(
                _config(data_root, output_root).evaluation,
                optimize_threshold=False,
            ),
        )
        reproduced = evaluate_checkpoint(changed_config, run_dir / "best.pt", device="cpu")
        assert reproduced["threshold"] == metrics["threshold"]
        assert reproduced["event"] == metrics["event"]
        incompatible_config = replace(
            _config(data_root, output_root),
            model=replace(_config(data_root, output_root).model, cross_time_radius=2),
        )
        with pytest.raises(ValueError, match="model configuration"):
            evaluate_checkpoint(incompatible_config, run_dir / "best.pt", device="cpu")
        with pytest.raises(FileExistsError, match="not empty"):
            run_training(_config(data_root, output_root), output_dir=run_dir)
