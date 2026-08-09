from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from huav.config import (
    ConeHeadConfig,
    DatasetConfig,
    HUAVConfig,
    ModelConfig,
    TrainingConfig,
)
from huav.data import LoadedEventSequence
from huav.models import SpatioTemporalConeDetector
from huav.training.engine import (
    _amp_is_enabled,
    _autocast_dtype,
    _dense_sequence_loss_weight,
    _grad_scaler_is_enabled,
    _select_competition_threshold,
    evaluate_checkpoint,
    resume_training,
    run_training,
)
from huav.training.losses import EventFocalDiceLoss


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


def _write_imbalanced_sequence(path: Path) -> None:
    dtype = [
        ("x", "<i2"),
        ("y", "<i2"),
        ("t", "<f8"),
        ("p", "i1"),
        ("label", "i1"),
        ("name", "i1"),
    ]
    rows = []
    for index in range(12):
        is_target = index in (0, 11)
        rows.append(
            (
                index % 8,
                index % 6,
                index * 600.0,
                index % 2,
                int(is_target),
                int(is_target),
            )
        )
    np.savez(path, ev=np.array(rows, dtype=dtype))


def _config(data_root: Path, output_root: Path) -> HUAVConfig:
    return replace(
        HUAVConfig.default(),
        resolution=replace(HUAVConfig.default().resolution, width=8, height=6),
        dataset=DatasetConfig(
            root=Path(os.path.relpath(data_root, Path.cwd())), duration_ms=8000
        ),
        model=ModelConfig(
            spatial_block_size_x=4,
            spatial_block_size_y=3,
            num_time_blocks=4,
            token_dim=8,
            event_embed_dim=4,
            num_layers=1,
            num_heads=2,
            mlp_ratio=2.0,
            dropout=0.0,
            cone_heads=(
                ConeHeadConfig(1.0, 1.0, 2),
                ConeHeadConfig(2.0, 2.0, 2),
            ),
            layer_time_scales=(1.0,),
            max_neighbors_per_cone=2,
            causal_mode=False,
            direction_dropout=0.0,
            debug_attention_stats=False,
        ),
        training=TrainingConfig(
            device="cpu",
            epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            seed=7,
            background_ratio=2,
            max_background_without_positive=4,
            num_threads=1,
            output_root=output_root,
            gradient_clip_norm=1.0,
            mixed_precision=False,
            max_train_files=None,
            max_val_files=None,
        ),
        loss=replace(
            HUAVConfig.default().loss,
            lambda_false_alarm=0.05,
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
        assert (run_dir / "training_curves.png").read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        )
        assert (run_dir / "metrics.json").is_file()
        assert (run_dir / "metrics.md").is_file()
        manifest = json.loads(
            (run_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        assert all(len(item["sha256"]) == 64 for item in manifest["files"])

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["event"]["event_count"] == 4
        assert {"pd", "fa", "iou", "acc"}.issubset(metrics["event"])
        assert {
            "iou",
            "seg_acc",
            "pd",
            "fa",
            "score_fa",
            "score",
            "counts",
        }.issubset(metrics["competition"])
        assert summary["best_score"] == metrics["competition"]["score"]
        history = json.loads(
            (run_dir / "history.json").read_text(encoding="utf-8")
        )
        assert history[0]["train_loss"]["false_alarm"] >= 0.0
        assert set(history[0]["validation_loss"]) == {
            "total",
            "focal",
            "dice",
            "object_frame",
            "false_alarm",
            "stc_weight_mean",
        }
        assert all(
            value >= 0.0 for value in history[0]["validation_loss"].values()
        )
        assert "loss" not in history[0]["validation"]
        assert "loss" not in metrics
        checkpoint = __import__("torch").load(
            run_dir / "best.pt", map_location="cpu", weights_only=True
        )
        assert checkpoint["schema_version"] == 6
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
            model=replace(
                _config(data_root, output_root).model,
                max_neighbors_per_cone=3,
            ),
        )
        with pytest.raises(ValueError, match="model configuration"):
            evaluate_checkpoint(incompatible_config, run_dir / "best.pt", device="cpu")
        unsupported_checkpoint = run_dir / "unsupported-schema.pt"
        checkpoint["schema_version"] = 999
        torch.save(checkpoint, unsupported_checkpoint)
        with pytest.raises(ValueError, match="schema version"):
            evaluate_checkpoint(
                _config(data_root, output_root),
                unsupported_checkpoint,
                device="cpu",
            )
        with pytest.raises(FileExistsError, match="not empty"):
            run_training(_config(data_root, output_root), output_dir=run_dir)


def test_training_curve_failure_warns_without_aborting_training(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        train_root = data_root / "train"
        val_root = data_root / "val"
        train_root.mkdir(parents=True)
        val_root.mkdir(parents=True)
        _write_sequence(train_root / "train_000.npz", 0)
        _write_sequence(val_root / "val_000.npz", 1)

        def fail_render(*_args, **_kwargs):
            raise OSError("plot filesystem unavailable")

        monkeypatch.setattr(
            "huav.training.engine.render_training_curves", fail_render
        )
        summary = run_training(
            _config(data_root, temporary_root / "outputs"),
            output_dir=temporary_root / "outputs" / "run",
        )

        assert (Path(summary["run_dir"]) / "best.pt").is_file()
        assert "warning: training curves were not updated" in capsys.readouterr().err


def test_training_records_note_and_success_in_experiment() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        (data_root / "train").mkdir(parents=True)
        (data_root / "val").mkdir()
        _write_sequence(data_root / "train" / "train_000.npz", 0)
        _write_sequence(data_root / "val" / "val_000.npz", 1)
        output_root = temporary_root / "outputs"

        summary = run_training(
            _config(data_root, output_root),
            output_dir=output_root / "group" / "exact",
            note="记录新的损失实验",
            config_source="configs/test.yaml",
        )

        experiment = (Path(summary["run_dir"]) / "experiment.md").read_text(
            encoding="utf-8"
        )
        assert "记录新的损失实验" in experiment
        assert "成功" in experiment
        assert "configs/test.yaml" in experiment
        assert "训练/验证文件数：1/1" in experiment
        assert "训练阶段：full" in experiment
        assert (output_root / "group" / "INDEX.md").is_file()
        assert not (output_root / "INDEX.md").exists()


def test_limited_training_is_recorded_as_smoke() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        (data_root / "train").mkdir(parents=True)
        (data_root / "val").mkdir()
        _write_sequence(data_root / "train" / "train_000.npz", 0)
        _write_sequence(data_root / "val" / "val_000.npz", 1)
        output_root = temporary_root / "outputs"
        config = _config(data_root, output_root)
        config = replace(
            config,
            training=replace(config.training, max_train_files=1),
        )

        summary = run_training(config, output_dir=output_root / "smoke")

        experiment = (Path(summary["run_dir"]) / "experiment.md").read_text(
            encoding="utf-8"
        )
        assert "训练阶段：smoke" in experiment


def test_training_failure_is_recorded_in_experiment() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "missing-data"
        output_root = temporary_root / "outputs"
        run_dir = output_root / "failed"

        with pytest.raises((FileNotFoundError, ValueError)):
            run_training(
                _config(data_root, output_root),
                output_dir=run_dir,
                note=None,
            )

        experiment = (run_dir / "experiment.md").read_text(encoding="utf-8")
        assert "失败" in experiment
        assert "待补充" in experiment


def test_invalid_direct_device_creates_no_output_path() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        config = _config(temporary_root / "data", temporary_root / "outputs")

        with pytest.raises(ValueError, match="device"):
            run_training(config, device="../escape")

        assert not (temporary_root / "outputs").exists()
        assert not (temporary_root / "escape").exists()


def test_training_primary_exception_survives_bookkeeping_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        config = _config(temporary_root / "missing", temporary_root / "outputs")
        monkeypatch.setattr(
            "huav.training.engine.record_run_finished",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bookkeeping failed")),
        )

        with pytest.raises((FileNotFoundError, ValueError)) as caught:
            run_training(config, output_dir=temporary_root / "outputs" / "failed")

        assert any("bookkeeping failed" in note for note in getattr(caught.value, "__notes__", ()))


def test_training_success_reports_bookkeeping_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        config = _config(temporary_root / "data", temporary_root / "outputs")
        monkeypatch.setattr(
            "huav.training.engine._run_training_impl",
            lambda *args, **kwargs: {"run_dir": str(temporary_root / "outputs" / "run")},
        )
        monkeypatch.setattr(
            "huav.training.engine.record_run_finished",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bookkeeping failed")),
        )

        with pytest.raises(RuntimeError, match="core operation succeeded.*artifact update failed"):
            run_training(config, output_dir=temporary_root / "outputs" / "run")


def test_resumed_training_records_validation_loss() -> None:
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
        config = _config(data_root, output_root)
        initial = run_training(config, output_dir=output_root / "initial")

        resumed = resume_training(
            config,
            Path(initial["run_dir"]) / "best.pt",
            additional_epochs=1,
            note="续训验证",
        )

        resumed_dir = Path(resumed["run_dir"])
        assert resumed_dir.name.startswith("cpu_")
        assert "resume" not in resumed_dir.name
        experiment = (resumed_dir / "experiment.md").read_text(encoding="utf-8")
        assert "续训验证" in experiment
        assert "训练类型：续训" in experiment
        assert "续训源 epoch：1" in experiment
        assert not (resumed_dir / "resumed_from.txt").exists()
        history = json.loads((resumed_dir / "history.json").read_text(encoding="utf-8"))
        assert history[0]["epoch"] == 2
        assert history[0]["resumed_epoch"] == 1
        assert set(history[0]["validation_loss"]) == {
            "total",
            "focal",
            "dice",
            "object_frame",
            "false_alarm",
            "stc_weight_mean",
        }
        assert all(
            value >= 0.0 for value in history[0]["validation_loss"].values()
        )
        assert "loss" not in history[0]["validation"]
        assert (resumed_dir / "training_curves.png").read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        )


def test_failed_resume_keeps_readable_checkpoint_source_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        checkpoint = temporary_root / "source.pt"
        torch.save({"epoch": 7}, checkpoint)
        run_dir = temporary_root / "outputs" / "failed-resume"
        config = _config(temporary_root / "data", temporary_root / "outputs")
        monkeypatch.setattr(
            "huav.training.engine._resume_training_impl",
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("later failure")),
        )

        with pytest.raises(ValueError, match="later failure"):
            resume_training(config, checkpoint, output_dir=run_dir)

        experiment = (run_dir / "experiment.md").read_text(encoding="utf-8")
        assert "续训源 epoch：7" in experiment


def test_amp_is_enabled_only_when_requested_on_cuda() -> None:
    assert _amp_is_enabled(True, torch.device("cuda"))
    assert not _amp_is_enabled(False, torch.device("cuda"))
    assert not _amp_is_enabled(True, torch.device("cpu"))


def test_blackwell_bfloat16_amp_does_not_use_gradient_scaling() -> None:
    assert _autocast_dtype("float16") == torch.float16
    assert _autocast_dtype("bfloat16") == torch.bfloat16
    assert _grad_scaler_is_enabled(True, torch.float16)
    assert not _grad_scaler_is_enabled(True, torch.bfloat16)
    assert not _grad_scaler_is_enabled(False, torch.float16)


def test_threshold_selection_prefers_official_score_over_event_iou() -> None:
    high_confidence_count = 100
    probabilities = torch.tensor(
        [0.99] * high_confidence_count + [0.60, 0.60],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [1.0] * high_confidence_count + [1.0, 0.0],
        dtype=torch.float32,
    )
    sequence = LoadedEventSequence(
        events=torch.column_stack(
            (
                torch.arange(len(labels), dtype=torch.float32),
                torch.zeros(len(labels)),
                torch.zeros(len(labels)),
                torch.ones(len(labels)),
            )
        ),
        labels=labels,
        object_ids=torch.tensor(
            [1] * high_confidence_count + [2, 0],
            dtype=torch.long,
        ),
        timestamps_ms=torch.tensor(
            [1.0] * high_confidence_count + [51.0, 51.0],
            dtype=torch.float64,
        ),
        source_path=Path("synthetic.npz"),
    )
    config = replace(
        HUAVConfig.default(),
        resolution=replace(
            HUAVConfig.default().resolution,
            width=1000,
            height=1000,
        ),
    )

    metrics = _select_competition_threshold(
        [(torch.logit(probabilities), sequence)],
        config,
        steps=2,
        fallback=0.5,
    )

    assert metrics["threshold"] == pytest.approx(0.5)
    assert metrics["competition"]["pd"] == pytest.approx(1.0)


def test_dense_balanced_polarity_sequences_receive_configured_loss_weight() -> None:
    config = replace(
        HUAVConfig.default(),
        training=replace(
            HUAVConfig.default().training,
            dense_sequence_event_threshold=5,
            dense_sequence_background_polarity_max=0.6,
            dense_sequence_weight=3.0,
        ),
    )
    events = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.2, 1.0],
            [0.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.4, 1.0],
            [0.0, 0.0, 0.5, 0.0],
        ]
    )
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    weight = _dense_sequence_loss_weight(events, labels, config)

    assert weight == pytest.approx(3.0)


def test_training_forwards_all_events_before_sampling_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_event_counts: list[int] = []
    loss_event_counts: list[int] = []
    original_model_forward = SpatioTemporalConeDetector.forward
    original_loss_forward = EventFocalDiceLoss.forward

    def record_model_forward(
        self: SpatioTemporalConeDetector,
        events: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
    ) -> dict[str, object]:
        model_event_counts.append(len(events))
        return original_model_forward(self, events, batch_indices)

    def record_loss_forward(
        self: EventFocalDiceLoss,
        logits: torch.Tensor,
        targets: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        loss_event_counts.append(len(logits))
        return original_loss_forward(
            self,
            logits,
            targets,
            batch_indices,
            **kwargs,
        )

    monkeypatch.setattr(SpatioTemporalConeDetector, "forward", record_model_forward)
    monkeypatch.setattr(EventFocalDiceLoss, "forward", record_loss_forward)
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        train_root = data_root / "train"
        val_root = data_root / "val"
        train_root.mkdir(parents=True)
        val_root.mkdir(parents=True)
        _write_imbalanced_sequence(train_root / "train_000.npz")
        _write_sequence(val_root / "val_000.npz", 0)
        config = _config(data_root, temporary_root / "outputs")
        config = replace(
            config,
            training=replace(config.training, background_ratio=1),
        )

        run_training(config, output_dir=temporary_root / "outputs" / "run")

    assert model_event_counts == [12, 4]
    assert loss_event_counts == [4, 4]
