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
    CHECKPOINT_SCHEMA_VERSION,
    _amp_is_enabled,
    _autocast_dtype,
    _configure_training_stage,
    _dense_sequence_loss_weight,
    _grad_scaler_is_enabled,
    _make_optimizer,
    _make_scheduler,
    _select_competition_threshold,
    build_model,
    evaluate_checkpoint,
    generate_submission_package,
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


def _adaptive_config(
    data_root: Path,
    output_root: Path,
    *,
    stage: str = "calibrator",
) -> HUAVConfig:
    base = _config(data_root, output_root)
    return replace(
        base,
        model=replace(
            base.model,
            adaptive_threshold_mode="hierarchical",
        ),
        training=replace(
            base.training,
            training_stage=stage,
            detector_learning_rate=(2e-4 if stage == "joint" else None),
        ),
        loss=replace(
            base.loss,
            lambda_threshold_magnitude=0.1,
            lambda_threshold_center=0.1,
            lambda_threshold_smoothness=0.1,
        ),
    ).validate()


@pytest.mark.parametrize(
    ("stage", "mode", "detector_lr"),
    (
        ("full", "none", None),
        ("full", "hierarchical", None),
        ("calibrator", "hierarchical", None),
        ("joint", "hierarchical", 1e-4),
    ),
)
def test_training_stage_configuration_accepts_valid_combinations(
    stage: str,
    mode: str,
    detector_lr: float | None,
) -> None:
    config = replace(
        HUAVConfig.default(),
        model=replace(
            HUAVConfig.default().model,
            adaptive_threshold_mode=mode,
        ),
        training=replace(
            HUAVConfig.default().training,
            training_stage=stage,
            detector_learning_rate=detector_lr,
        ),
    )

    assert config.validate() is config


@pytest.mark.parametrize(
    ("stage", "mode", "detector_lr", "message"),
    (
        ("unknown", "none", None, "training_stage"),
        ("calibrator", "none", None, "adaptive"),
        ("joint", "none", 1e-4, "adaptive"),
        ("joint", "hierarchical", None, "detector_learning_rate"),
        ("joint", "hierarchical", 0.0, "detector_learning_rate"),
        ("full", "none", 1e-4, "detector_learning_rate"),
        ("calibrator", "hierarchical", 1e-4, "detector_learning_rate"),
    ),
)
def test_training_stage_configuration_rejects_invalid_combinations(
    stage: str,
    mode: str,
    detector_lr: float | None,
    message: str,
) -> None:
    config = replace(
        HUAVConfig.default(),
        model=replace(
            HUAVConfig.default().model,
            adaptive_threshold_mode=mode,
        ),
        training=replace(
            HUAVConfig.default().training,
            training_stage=stage,
            detector_learning_rate=detector_lr,
        ),
    )

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_calibrator_stage_freezes_detector_and_optimizer_contains_only_head() -> None:
    config = replace(
        HUAVConfig.default(),
        model=replace(
            HUAVConfig.default().model,
            adaptive_threshold_mode="hierarchical",
        ),
        training=replace(
            HUAVConfig.default().training,
            training_stage="calibrator",
        ),
    ).validate()
    model = __import__("huav.training.engine", fromlist=["build_model"]).build_model(
        config
    )

    trainable = _configure_training_stage(model, config.training.training_stage)
    optimizer = _make_optimizer(model, config)

    assert not model.tokenizer.training
    assert not model.encoder.training
    assert not model.decoder.training
    assert model.calibrator is not None and model.calibrator.training
    assert all(not parameter.requires_grad for parameter in model.tokenizer.parameters())
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.decoder.parameters())
    assert all(parameter.requires_grad for parameter in model.calibrator.parameters())
    assert trainable == sum(
        parameter.numel() for parameter in model.calibrator.parameters()
    )
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        config.training.learning_rate
    )


def test_joint_stage_builds_detector_and_calibrator_optimizer_groups() -> None:
    config = replace(
        HUAVConfig.default(),
        model=replace(
            HUAVConfig.default().model,
            adaptive_threshold_mode="hierarchical",
        ),
        training=replace(
            HUAVConfig.default().training,
            training_stage="joint",
            detector_learning_rate=2e-4,
            learning_rate=7e-4,
        ),
    ).validate()
    model = __import__("huav.training.engine", fromlist=["build_model"]).build_model(
        config
    )

    _configure_training_stage(model, config.training.training_stage)
    optimizer = _make_optimizer(model, config)

    assert all(parameter.requires_grad for parameter in model.parameters())
    assert [group["name"] for group in optimizer.param_groups] == [
        "detector",
        "calibrator",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [2e-4, 7e-4]
    )


def test_joint_stage_optimizer_updates_detector_and_calibrator() -> None:
    config = replace(
        HUAVConfig.default(),
        resolution=replace(HUAVConfig.default().resolution, width=8, height=6),
        model=replace(
            _config(Path("data"), Path("outputs")).model,
            adaptive_threshold_mode="hierarchical",
        ),
        training=replace(
            HUAVConfig.default().training,
            training_stage="joint",
            detector_learning_rate=2e-4,
            learning_rate=7e-4,
        ),
    ).validate()
    model = build_model(config)
    _configure_training_stage(model, "joint")
    optimizer = _make_optimizer(model, config)
    detector_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("calibrator.")
    }
    calibrator_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("calibrator.")
    }
    events = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0],
            [5.0, 1.0, 0.01, 0.0],
            [2.0, 2.0, 0.1, 1.0],
            [6.0, 2.0, 0.11, 0.0],
        ]
    )
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

    output = model(events)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["event_logits"], labels
    )
    loss.backward()
    optimizer.step()

    assert any(
        not torch.equal(detector_before[name], parameter)
        for name, parameter in model.named_parameters()
        if name in detector_before
    )
    assert any(
        not torch.equal(calibrator_before[name], parameter)
        for name, parameter in model.named_parameters()
        if name in calibrator_before
    )


def test_joint_scheduler_preserves_group_ratio_and_reaches_minimum_ratio() -> None:
    config = replace(
        HUAVConfig.default(),
        model=replace(
            HUAVConfig.default().model,
            adaptive_threshold_mode="hierarchical",
        ),
        training=replace(
            HUAVConfig.default().training,
            training_stage="joint",
            detector_learning_rate=2e-4,
            learning_rate=8e-4,
        ),
    ).validate()
    model = build_model(config)
    _configure_training_stage(model, "joint")
    optimizer = _make_optimizer(model, config)
    scheduler = _make_scheduler(optimizer, epochs=4, minimum_ratio=0.1)

    for _ in range(4):
        optimizer.step()
        scheduler.step()

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [2e-5, 8e-5]
    )
    assert (
        optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"]
    ) == pytest.approx(4.0)


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
            "threshold_magnitude",
            "threshold_center",
            "threshold_smoothness",
        }
        assert all(
            value >= 0.0 for value in history[0]["validation_loss"].values()
        )
        assert "loss" not in history[0]["validation"]
        assert "loss" not in metrics
        checkpoint = __import__("torch").load(
            run_dir / "best.pt", map_location="cpu", weights_only=True
        )
        assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert checkpoint["training_stage"] == "full"
        assert checkpoint["trainable_parameter_count"] == summary[
            "trainable_parameter_count"
        ]
        assert checkpoint["decision_threshold"] == metrics["threshold"]
        for index, invalid_threshold in enumerate((None, float("nan"), -0.1, 1.1)):
            invalid_checkpoint = dict(checkpoint)
            if invalid_threshold is None:
                invalid_checkpoint.pop("decision_threshold")
            else:
                invalid_checkpoint["decision_threshold"] = invalid_threshold
            invalid_path = run_dir / f"invalid-threshold-{index}.pt"
            torch.save(invalid_checkpoint, invalid_path)
            with pytest.raises(ValueError, match="decision_threshold"):
                evaluate_checkpoint(
                    _config(data_root, output_root),
                    invalid_path,
                    device="cpu",
                    reoptimize_threshold=True,
                )
            with pytest.raises(ValueError, match="decision_threshold"):
                generate_submission_package(
                    _config(data_root, output_root),
                    invalid_path,
                    output_root / f"invalid-submission-{index}",
                )
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
        assert reproduced == metrics
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
        checkpoint["schema_version"] = 3
        torch.save(checkpoint, unsupported_checkpoint)
        with pytest.raises(ValueError, match="schema version"):
            evaluate_checkpoint(
                _config(data_root, output_root),
                unsupported_checkpoint,
                device="cpu",
            )
        with pytest.raises(FileExistsError, match="not empty"):
            run_training(_config(data_root, output_root), output_dir=run_dir)


def test_resume_rejects_non_joint_training_stage() -> None:
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

        for stage in ("full", "calibrator"):
            stage_config = (
                config
                if stage == "full"
                else _adaptive_config(data_root, output_root)
            )
            with pytest.raises(ValueError, match="joint training stage"):
                rejected_dir = output_root / f"resume-{stage}"
                resume_training(
                    stage_config,
                    Path(initial["run_dir"]) / "best.pt",
                    additional_epochs=1,
                    output_dir=rejected_dir,
                )
            assert not rejected_dir.exists()


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
    calibration_argument_sets: list[set[str]] = []
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
        calibration_argument_sets.append(
            {
                name
                for name in (
                    "global_threshold_offsets",
                    "local_threshold_offsets",
                    "threshold_group_indices",
                    "threshold_group_batch_indices",
                    "threshold_neighbor_edges",
                )
                if name in kwargs
            }
        )
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
    assert calibration_argument_sets == [
        {
            "global_threshold_offsets",
            "local_threshold_offsets",
            "threshold_group_indices",
            "threshold_group_batch_indices",
            "threshold_neighbor_edges",
        },
        {
            "global_threshold_offsets",
            "local_threshold_offsets",
            "threshold_group_indices",
            "threshold_group_batch_indices",
            "threshold_neighbor_edges",
        },
    ]


def test_schema4_detector_initialization_records_provenance_and_trains_calibrator() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        train_root = data_root / "train"
        val_root = data_root / "val"
        train_root.mkdir(parents=True)
        val_root.mkdir(parents=True)
        _write_sequence(train_root / "train_000.npz", 0)
        _write_sequence(val_root / "val_000.npz", 1)
        _write_sequence(val_root / "val_001.npz", 0)
        output_root = temporary_root / "outputs"

        control = run_training(
            _config(data_root, output_root),
            output_dir=output_root / "control",
        )
        control_checkpoint = Path(control["run_dir"]) / "best.pt"
        legacy_checkpoint = output_root / "legacy-schema4.pt"
        legacy = torch.load(
            control_checkpoint, map_location="cpu", weights_only=True
        )
        legacy["schema_version"] = 4
        for name in (
            "adaptive_threshold_mode",
            "threshold_hidden_dim",
            "threshold_time_bins",
            "threshold_max_global_offset",
            "threshold_max_local_offset",
        ):
            legacy["model_config"].pop(name)
        torch.save(legacy, legacy_checkpoint)

        adaptive = _adaptive_config(data_root, output_root)
        trained = run_training(
            adaptive,
            output_dir=output_root / "adaptive",
            initialize_detector=legacy_checkpoint,
        )
        run_dir = Path(trained["run_dir"])
        checkpoint = torch.load(
            run_dir / "best.pt", map_location="cpu", weights_only=True
        )
        history = json.loads(
            (run_dir / "history.json").read_text(encoding="utf-8")
        )

        assert trained["training_stage"] == "calibrator"
        assert trained["trainable_parameter_count"] < trained["parameter_count"]
        assert checkpoint["schema_version"] == 5
        assert checkpoint["source_checkpoint"] == str(legacy_checkpoint.resolve())
        assert checkpoint["source_checkpoint_sha256"] == __import__(
            "hashlib"
        ).sha256(legacy_checkpoint.read_bytes()).hexdigest()
        assert (run_dir / "initialized_from.txt").read_text(
            encoding="utf-8"
        ).startswith(f"checkpoint: {legacy_checkpoint.resolve()}\nsha256: ")
        assert set(history[0]["train_loss"]).issuperset(
            {
                "threshold_magnitude",
                "threshold_center",
                "threshold_smoothness",
            }
        )
        assert history[0]["learning_rates"] == {
            "calibrator": pytest.approx(adaptive.training.learning_rate)
        }
        assert history[0]["validation_loss"]["threshold_magnitude"] > 0.0
        assert all(
            torch.equal(checkpoint["model_state"][name], value)
            for name, value in legacy["model_state"].items()
        )
        assert any(
            bool(value.abs().max() > 0)
            for name, value in checkpoint["model_state"].items()
            if name.startswith("calibrator.")
        )
        assert "calibration" in history[0]["validation"]
        assert set(history[0]["validation"]["calibration"]) == {
            "global",
            "local",
            "per_sequence_global_mean",
        }
        first = evaluate_checkpoint(adaptive, run_dir / "best.pt")
        second = evaluate_checkpoint(adaptive, run_dir / "best.pt")
        assert first["calibration"] == second["calibration"]
        assert first == json.loads(
            (run_dir / "metrics.json").read_text(encoding="utf-8")
        )
        submission = generate_submission_package(
            adaptive,
            run_dir / "best.pt",
            output_root / "submission",
        )
        assert submission.is_file()

        resumed = resume_training(
            _adaptive_config(data_root, output_root, stage="joint"),
            run_dir / "best.pt",
            additional_epochs=1,
            output_dir=output_root / "resumed",
        )
        resumed_dir = Path(resumed["run_dir"])
        assert resumed_dir.joinpath("best.pt").is_file()
        resumed_checkpoint = torch.load(
            resumed_dir / "best.pt", map_location="cpu", weights_only=True
        )
        assert resumed_checkpoint["source_checkpoint"] == checkpoint[
            "source_checkpoint"
        ]
        assert resumed_checkpoint["source_checkpoint_sha256"] == checkpoint[
            "source_checkpoint_sha256"
        ]
        assert resumed_checkpoint["resumed_from_checkpoint"] == str(
            (run_dir / "best.pt").resolve()
        )
        assert resumed_checkpoint["resumed_from_sha256"] == __import__(
            "hashlib"
        ).sha256((run_dir / "best.pt").read_bytes()).hexdigest()
        assert resumed["resumed_from_checkpoint"] == str(
            (run_dir / "best.pt").resolve()
        )
        assert resumed["resumed_from_sha256"] == resumed_checkpoint[
            "resumed_from_sha256"
        ]
        resumed_history = json.loads(
            resumed_dir.joinpath("history.json").read_text(encoding="utf-8")
        )
        assert resumed_history[0]["learning_rate"] == pytest.approx(
            adaptive.training.learning_rate
        )
        assert set(resumed_history[0]["learning_rates"]) == {
            "detector",
            "calibrator",
        }
        for invalid_epochs in (0, -1):
            with pytest.raises(ValueError, match="positive"):
                resume_training(
                    _adaptive_config(data_root, output_root, stage="joint"),
                    run_dir / "best.pt",
                    additional_epochs=invalid_epochs,
                    output_dir=output_root / f"invalid-{invalid_epochs}",
                )

        schema5_initialized = run_training(
            adaptive,
            output_dir=output_root / "adaptive-schema5-source",
            initialize_detector=control_checkpoint,
        )
        assert Path(schema5_initialized["run_dir"]).joinpath("best.pt").is_file()

        with pytest.raises(ValueError, match="non-adaptive"):
            run_training(
                adaptive,
                output_dir=output_root / "adaptive-source-rejected",
                initialize_detector=run_dir / "best.pt",
            )

        markdown = (run_dir / "metrics.md").read_text(encoding="utf-8")
        first_file = markdown.index("| val_000.npz |")
        second_file = markdown.index("| val_001.npz |")
        assert first_file < second_file

        _write_sequence(val_root / "val_000.npz", 3)
        mismatch_output = output_root / "resume-manifest-mismatch"
        with pytest.raises(ValueError, match="validation dataset manifest"):
            resume_training(
                _adaptive_config(data_root, output_root, stage="joint"),
                run_dir / "best.pt",
                additional_epochs=1,
                output_dir=mismatch_output,
            )
        assert not mismatch_output.exists()


def test_schema4_none_checkpoint_supports_control_evaluation_and_reoptimization() -> None:
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
        base_config = _config(data_root, output_root)
        config = replace(
            base_config,
            evaluation=replace(
                base_config.evaluation,
                optimize_threshold=False,
            ),
        )
        summary = run_training(config, output_dir=output_root / "control")
        checkpoint_path = Path(summary["run_dir"]) / "best.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        checkpoint["schema_version"] = 4
        checkpoint["decision_threshold"] = 0.99
        for name in (
            "adaptive_threshold_mode",
            "threshold_hidden_dim",
            "threshold_time_bins",
            "threshold_max_global_offset",
            "threshold_max_local_offset",
        ):
            checkpoint["model_config"].pop(name)
        torch.save(checkpoint, checkpoint_path)

        stored = evaluate_checkpoint(config, checkpoint_path)
        optimized = evaluate_checkpoint(
            config,
            checkpoint_path,
            reoptimize_threshold=True,
        )

        assert stored["threshold"] == pytest.approx(0.99)
        assert optimized["threshold"] != pytest.approx(0.99)
        assert optimized["threshold"] != pytest.approx(
            config.inference.sigmoid_threshold
        )
        assert stored["calibration"]["global"]["max_abs"] == 0.0
        assert optimized["calibration"]["local"]["p95_abs"] == 0.0
        submission = generate_submission_package(
            config,
            checkpoint_path,
            output_root / "legacy-submission",
        )
        assert submission.is_file()


def test_schema5_metadata_is_strict_for_evaluate_predict_and_resume() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        (data_root / "train").mkdir(parents=True)
        (data_root / "val").mkdir(parents=True)
        _write_sequence(data_root / "train" / "train_000.npz", 0)
        _write_sequence(data_root / "val" / "val_000.npz", 1)
        output_root = temporary_root / "outputs"
        control = run_training(
            _config(data_root, output_root),
            output_dir=output_root / "control",
        )
        adaptive = _adaptive_config(data_root, output_root)
        trained = run_training(
            adaptive,
            output_dir=output_root / "adaptive",
            initialize_detector=Path(control["run_dir"]) / "best.pt",
        )
        checkpoint_path = Path(trained["run_dir"]) / "best.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        variants: list[tuple[str, dict[str, object]]] = []

        def variant(name: str, **updates: object) -> None:
            payload = dict(checkpoint)
            payload.update(updates)
            variants.append((name, payload))

        missing_stage = dict(checkpoint)
        missing_stage.pop("training_stage")
        variants.append(("missing-stage", missing_stage))
        variant("invalid-stage", training_stage="invalid")
        missing_count = dict(checkpoint)
        missing_count.pop("trainable_parameter_count")
        variants.append(("missing-count", missing_count))
        variant("boolean-count", trainable_parameter_count=True)
        variant("zero-count", trainable_parameter_count=0)
        variant(
            "wrong-count",
            trainable_parameter_count=checkpoint[
                "trainable_parameter_count"
            ]
            + 1,
        )
        variant("invalid-model-config", model_config=[])
        mode_none = dict(checkpoint)
        mode_none["model_config"] = dict(checkpoint["model_config"])
        mode_none["model_config"]["adaptive_threshold_mode"] = "none"
        variants.append(("calibrator-none-mode", mode_none))
        full_with_provenance = dict(checkpoint)
        full_with_provenance["training_stage"] = "full"
        full_with_provenance["trainable_parameter_count"] = sum(
            value.numel() for value in checkpoint["model_state"].values()
        )
        variants.append(("full-with-provenance", full_with_provenance))
        missing_source = dict(checkpoint)
        missing_source.pop("source_checkpoint")
        variants.append(("missing-source", missing_source))
        missing_hash = dict(checkpoint)
        missing_hash.pop("source_checkpoint_sha256")
        variants.append(("missing-hash", missing_hash))
        variant("empty-source", source_checkpoint=" ")
        variant("short-hash", source_checkpoint_sha256="abc")
        missing_provenance = dict(checkpoint)
        missing_provenance.pop("source_checkpoint")
        missing_provenance.pop("source_checkpoint_sha256")
        variants.append(("calibrator-without-provenance", missing_provenance))
        missing_state = dict(checkpoint)
        missing_state.pop("model_state")
        variants.append(("missing-state", missing_state))
        variant("non-mapping-state", model_state=[])
        invalid_state_value = dict(checkpoint)
        invalid_state_value["model_state"] = dict(checkpoint["model_state"])
        invalid_state_value["model_state"]["calibrator.invalid"] = "not-a-tensor"
        variants.append(("non-tensor-state", invalid_state_value))

        joint = _adaptive_config(data_root, output_root, stage="joint")
        for name, payload in variants:
            invalid_path = output_root / f"{name}.pt"
            torch.save(payload, invalid_path)
            with pytest.raises(ValueError, match="checkpoint"):
                evaluate_checkpoint(adaptive, invalid_path)
            with pytest.raises(ValueError, match="checkpoint"):
                generate_submission_package(
                    adaptive,
                    invalid_path,
                    output_root / f"submission-{name}",
                )
            resume_dir = output_root / f"resume-{name}"
            with pytest.raises(ValueError, match="checkpoint"):
                resume_training(
                    joint,
                    invalid_path,
                    additional_epochs=1,
                    output_dir=resume_dir,
                )
            assert not resume_dir.exists()


def test_checkpoint_load_paths_reject_non_mapping_top_level_without_artifacts() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        (data_root / "train").mkdir(parents=True)
        (data_root / "val").mkdir(parents=True)
        _write_sequence(data_root / "train" / "train_000.npz", 0)
        _write_sequence(data_root / "val" / "val_000.npz", 1)
        checkpoint_path = temporary_root / "not-mapping.pt"
        torch.save([], checkpoint_path)
        full = _config(data_root, temporary_root / "outputs")
        calibrator = _adaptive_config(
            data_root, temporary_root / "outputs"
        )
        joint = _adaptive_config(
            data_root, temporary_root / "outputs", stage="joint"
        )

        with pytest.raises(ValueError, match="checkpoint.*mapping"):
            evaluate_checkpoint(full, checkpoint_path)
        with pytest.raises(ValueError, match="checkpoint.*mapping"):
            generate_submission_package(
                full,
                checkpoint_path,
                temporary_root / "submission",
            )
        init_output = temporary_root / "init-output"
        with pytest.raises(ValueError, match="checkpoint.*mapping"):
            run_training(
                calibrator,
                output_dir=init_output,
                initialize_detector=checkpoint_path,
            )
        assert not init_output.exists()
        resume_output = temporary_root / "resume-output"
        with pytest.raises(ValueError, match="checkpoint.*mapping"):
            resume_training(
                joint,
                checkpoint_path,
                additional_epochs=1,
                output_dir=resume_output,
            )
        assert not resume_output.exists()


def test_run_training_rejects_fresh_joint_stage() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        (data_root / "train").mkdir(parents=True)
        (data_root / "val").mkdir(parents=True)
        _write_sequence(data_root / "train" / "train_000.npz", 0)
        _write_sequence(data_root / "val" / "val_000.npz", 1)
        config = _adaptive_config(
            data_root,
            temporary_root / "outputs",
            stage="joint",
        )

        with pytest.raises(ValueError, match="run_training.*joint"):
            output_dir = temporary_root / "outputs" / "joint"
            run_training(
                config,
                output_dir=output_dir,
            )
        assert not output_dir.exists()


def test_calibrator_training_requires_detector_initialization() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        (data_root / "train").mkdir(parents=True)
        (data_root / "val").mkdir(parents=True)
        _write_sequence(data_root / "train" / "train_000.npz", 0)
        _write_sequence(data_root / "val" / "val_000.npz", 1)
        config = _adaptive_config(data_root, temporary_root / "outputs")

        with pytest.raises(ValueError, match="initialize_detector"):
            run_training(config, output_dir=temporary_root / "outputs" / "run")


@pytest.mark.parametrize("stage", ("full", "joint"))
def test_detector_initialization_is_restricted_to_calibrator_stage(
    stage: str,
) -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        (data_root / "train").mkdir(parents=True)
        (data_root / "val").mkdir(parents=True)
        _write_sequence(data_root / "train" / "train_000.npz", 0)
        _write_sequence(data_root / "val" / "val_000.npz", 1)
        config = _adaptive_config(
            data_root,
            temporary_root / "outputs",
            stage=stage,
        )

        message = (
            "calibrator training stage"
            if stage == "full"
            else "run_training.*joint"
        )
        with pytest.raises(ValueError, match=message):
            run_training(
                config,
                output_dir=temporary_root / "outputs" / stage,
                initialize_detector=temporary_root / "detector.pt",
            )


def test_detector_initialization_rejects_schema3() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        (data_root / "train").mkdir(parents=True)
        (data_root / "val").mkdir(parents=True)
        _write_sequence(data_root / "train" / "train_000.npz", 0)
        _write_sequence(data_root / "val" / "val_000.npz", 1)
        config = _adaptive_config(data_root, temporary_root / "outputs")
        checkpoint = temporary_root / "schema3.pt"
        torch.save({"schema_version": 3}, checkpoint)
        output_dir = temporary_root / "outputs" / "run"

        with pytest.raises(ValueError, match="schema version"):
            run_training(
                config,
                output_dir=output_dir,
                initialize_detector=checkpoint,
            )
        assert not output_dir.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("config", "model configuration"),
        ("shape", "shape mismatch"),
        ("missing", "missing="),
        ("unexpected", "unexpected="),
        ("calibrator_state", "non-adaptive source state"),
        ("missing_state", "model_state"),
        ("non_mapping_state", "model_state.*mapping"),
        ("non_tensor_state", "string names to tensors"),
    ),
)
def test_detector_initialization_rejects_incompatible_detector_state(
    mutation: str,
    message: str,
) -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        (data_root / "train").mkdir(parents=True)
        (data_root / "val").mkdir(parents=True)
        _write_sequence(data_root / "train" / "train_000.npz", 0)
        _write_sequence(data_root / "val" / "val_000.npz", 1)
        output_root = temporary_root / "outputs"
        control = run_training(
            _config(data_root, output_root),
            output_dir=output_root / "control",
        )
        source_path = Path(control["run_dir"]) / "best.pt"
        checkpoint = torch.load(
            source_path, map_location="cpu", weights_only=True
        )
        detector_key = next(iter(checkpoint["model_state"]))
        if mutation == "config":
            checkpoint["model_config"]["max_neighbors_per_cone"] += 1
        elif mutation == "shape":
            value = checkpoint["model_state"][detector_key]
            checkpoint["model_state"][detector_key] = value.reshape(-1)[:1]
        elif mutation == "missing":
            checkpoint["model_state"].pop(detector_key)
        elif mutation == "calibrator_state":
            checkpoint["model_state"]["calibrator.unexpected"] = torch.zeros(1)
        elif mutation == "missing_state":
            checkpoint.pop("model_state")
        elif mutation == "non_mapping_state":
            checkpoint["model_state"] = []
        elif mutation == "non_tensor_state":
            checkpoint["model_state"]["invalid"] = "not-a-tensor"
        else:
            checkpoint["model_state"]["detector.unexpected"] = torch.zeros(1)
        if mutation not in {
            "config",
            "missing_state",
            "non_mapping_state",
            "non_tensor_state",
        }:
            checkpoint["trainable_parameter_count"] = sum(
                value.numel()
                for value in checkpoint["model_state"].values()
            )
        incompatible = output_root / f"{mutation}.pt"
        torch.save(checkpoint, incompatible)

        output_dir = output_root / f"adaptive-{mutation}"
        with pytest.raises(ValueError, match=message):
            run_training(
                _adaptive_config(data_root, output_root),
                output_dir=output_dir,
                initialize_detector=incompatible,
            )
        assert not output_dir.exists()
