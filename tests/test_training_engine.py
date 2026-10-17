from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from traceformer.config import (
    DatasetConfig,
    FrameRefinementConfig,
    ModelConfig,
    RefinementScaleConfig,
    TraceFormerConfig,
    TrainingConfig,
)
from traceformer.data import LoadedEventSequence
from traceformer.models import TraceFormer
from traceformer.training.engine import (
    _amp_is_enabled,
    _autocast_dtype,
    _dense_sequence_loss_weight,
    _grad_scaler_is_enabled,
    _restore_rng_state,
    _select_competition_threshold,
    _serializable,
    _sparse_mixture_weight,
    build_model,
    evaluate_checkpoint,
    run_training,
)
from traceformer.training.losses import EventFocalDiceLoss


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


def _write_sparse_mixture_sequence(path: Path) -> None:
    dtype = [
        ("x", "<i2"),
        ("y", "<i2"),
        ("t", "<f8"),
        ("p", "i1"),
        ("label", "i1"),
        ("name", "i1"),
    ]
    rows = [
        (1, 1, 100.0, 1, 1, 1),
        (1, 1, 200.0, 1, 1, 1),
        (1, 1, 300.0, 0, 0, 0),
        (1, 1, 400.0, 0, 0, 0),
    ]
    np.savez(path, ev=np.array(rows, dtype=dtype))


def _config(data_root: Path, output_root: Path) -> TraceFormerConfig:
    return replace(
        TraceFormerConfig.default(),
        resolution=replace(TraceFormerConfig.default().resolution, width=8, height=6),
        dataset=DatasetConfig(
            root=Path(os.path.relpath(data_root, Path.cwd())), duration_ms=8000
        ),
        model=ModelConfig(
            spatial_block_size_x=4,
            spatial_block_size_y=3,
            num_time_blocks=4,
            token_dim=8,
            event_embed_dim=4,
            num_heads=2,

            mlp_ratio=2.0,
            dropout=0.0,
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
            TraceFormerConfig.default().loss,
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
            "event_refinement",
        }
        assert all(
            value >= 0.0 for value in history[0]["validation_loss"].values()
        )
        assert "loss" not in history[0]["validation"]
        assert "loss" not in metrics
        checkpoint = __import__("torch").load(
            run_dir / "best.pt", map_location="cpu", weights_only=True
        )
        assert checkpoint["schema_version"] == 7
        assert "enabled" not in checkpoint["sparse_mixture_config"]
        assert "enabled" not in checkpoint["frame_refinement_config"]
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
        legacy_metadata_checkpoint = run_dir / "legacy-metadata.pt"
        checkpoint["model_config"]["causal_mode"] = False
        checkpoint["model_config"]["debug_attention_stats"] = False
        torch.save(checkpoint, legacy_metadata_checkpoint)
        with pytest.raises(ValueError, match="model configuration"):
            evaluate_checkpoint(changed_config, legacy_metadata_checkpoint, device="cpu")
        checkpoint["model_config"].pop("causal_mode")
        checkpoint["model_config"].pop("debug_attention_stats")
        incompatible_config = replace(
            _config(data_root, output_root),
            model=replace(
                _config(data_root, output_root).model,
                mlp_ratio=3.0,
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


def test_refinement_initialization_loads_matching_base_and_trains_only_new_head() -> None:
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
        source_config = _config(data_root, output_root)
        source_model = build_model(source_config)
        source_path = temporary_root / "base.pt"
        torch.save(
            {
                "schema_version": 7,
                "model_state": {
                    name: value
                    for name, value in source_model.state_dict().items()
                    if not name.startswith("event_refinement_head.")
                },
                "model_config": _serializable(asdict(source_config.model)),
                "sparse_mixture_config": _serializable(
                    asdict(source_config.sparse_mixture)
                ),
                "resolution": _serializable(asdict(source_config.resolution)),
                "epoch": 7,
            },
            source_path,
        )
        target_config = replace(
            source_config,
            training=replace(source_config.training, stage="refinement"),
            frame_refinement=FrameRefinementConfig(
                context_dim=4,
                scales=(
                    RefinementScaleConfig(2, 2, 10.0),
                    RefinementScaleConfig(8, 8, 50.0),
                ),
                loss_weight=0.2,
            ),
        ).validate()

        summary = run_training(
            target_config,
            output_dir=output_root / "refinement",
            initial_refinement_checkpoint_path=source_path,
        )

        run_dir = Path(summary["run_dir"])
        checkpoint = torch.load(
            run_dir / "best.pt", map_location="cpu", weights_only=True
        )
        trained_model = build_model(target_config)
        assert trained_model.event_refinement_head is not None
        expected_trainable = sum(
            parameter.numel()
            for parameter in trained_model.event_refinement_head.parameters()
        )
        assert summary["trainable_parameter_count"] == expected_trainable
        assert summary["full_initialization"]["kind"] == "event_refinement"
        assert summary["full_initialization"]["epoch"] == 7
        assert checkpoint["schema_version"] == 7
        assert "enabled" not in checkpoint["frame_refinement_config"]
        for name, tensor in source_model.state_dict().items():
            if name.startswith("event_refinement_head."):
                continue
            assert torch.equal(checkpoint["model_state"][name], tensor)
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        assert history[0]["train_loss"]["event_refinement"] > 0.0


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
            "traceformer.training.engine.render_training_curves", fail_render
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
            "traceformer.training.engine.record_run_finished",
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
            "traceformer.training.engine._run_training_impl",
            lambda *args, **kwargs: {"run_dir": str(temporary_root / "outputs" / "run")},
        )
        monkeypatch.setattr(
            "traceformer.training.engine.record_run_finished",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bookkeeping failed")),
        )

        with pytest.raises(RuntimeError, match="core operation succeeded.*artifact update failed"):
            run_training(config, output_dir=temporary_root / "outputs" / "run")


def test_exact_training_state_branch_matches_uninterrupted_cpu_run() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        (data_root / "train").mkdir(parents=True)
        (data_root / "val").mkdir()
        _write_sequence(data_root / "train" / "train_000.npz", 0)
        _write_sequence(data_root / "val" / "val_000.npz", 1)
        output_root = temporary_root / "outputs"
        base = _config(data_root, output_root)
        direct_config = replace(
            base,
            training=replace(
                base.training,
                epochs=2,
                lr_schedule="constant",
                save_training_state_interval=1,
            ),
        ).validate()
        prefix_config = replace(
            direct_config,
            training=replace(direct_config.training, epochs=1),
        ).validate()

        direct = run_training(direct_config, output_dir=output_root / "direct")
        prefix = run_training(prefix_config, output_dir=output_root / "prefix")
        branched = run_training(
            direct_config,
            output_dir=output_root / "branched",
            resume_state_path=Path(prefix["run_dir"]) / "training_state.pt",
        )

        direct_dir = Path(direct["run_dir"])
        branched_dir = Path(branched["run_dir"])
        direct_history = json.loads(
            (direct_dir / "history.json").read_text(encoding="utf-8")
        )
        branched_history = json.loads(
            (branched_dir / "history.json").read_text(encoding="utf-8")
        )
        for history in (direct_history, branched_history):
            for record in history:
                record["optimization"].pop("elapsed_seconds")
        assert branched_history == direct_history
        assert direct_history[0]["learning_rate_schedule"] == "constant"
        assert direct_history[0]["learning_phase"] == "fast"
        assert direct_history[0]["optimization"]["nonfinite_steps"] == 0
        direct_checkpoint = torch.load(
            direct_dir / "best.pt", map_location="cpu", weights_only=True
        )
        branched_checkpoint = torch.load(
            branched_dir / "best.pt", map_location="cpu", weights_only=True
        )
        for name, tensor in direct_checkpoint["model_state"].items():
            assert torch.equal(tensor, branched_checkpoint["model_state"][name])
        assert (branched_dir / "training_states" / "epoch_002.pt").is_file()


def test_restore_rng_state_keeps_serialized_generator_states_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_states: list[torch.Tensor] = []
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda states: cuda_states.extend(states),
    )
    generator_state = torch.Generator().manual_seed(37).get_state()
    state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": [generator_state.clone()],
        "order_generator_state": generator_state.clone(),
        "sampling_generator_state": generator_state.clone(),
    }

    _restore_rng_state(
        state,
        torch_device=torch.device("cuda"),
        order_generator=torch.Generator(),
        sampling_generator=torch.Generator(),
    )

    assert len(cuda_states) == 1
    assert cuda_states[0].device.type == "cpu"
    assert cuda_states[0].dtype == torch.uint8


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
        TraceFormerConfig.default(),
        resolution=replace(
            TraceFormerConfig.default().resolution,
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
        TraceFormerConfig.default(),
        training=replace(
            TraceFormerConfig.default().training,
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


def test_sparse_mixture_schedule_holds_router_then_ramps() -> None:
    base = TraceFormerConfig.default()
    config = replace(
        base,
        sparse_mixture=replace(
            base.sparse_mixture,
            warmup_epochs=2,
            ramp_epochs=2,
        ),
    ).validate()

    assert [_sparse_mixture_weight(config, epoch) for epoch in range(1, 6)] == [
        0.0,
        0.0,
        pytest.approx(0.5),
        1.0,
        1.0,
    ]


def test_training_records_sparse_mixture_losses_and_paired_inference() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        temporary_root = Path(directory)
        data_root = temporary_root / "data"
        train_root = data_root / "train"
        val_root = data_root / "val"
        train_root.mkdir(parents=True)
        val_root.mkdir(parents=True)
        _write_sparse_mixture_sequence(train_root / "train_000.npz")
        _write_sparse_mixture_sequence(val_root / "val_000.npz")
        output_root = temporary_root / "outputs"
        config = _config(data_root, output_root)
        config = replace(
            config,
            sparse_mixture=replace(
                config.sparse_mixture,
                min_events=1,
                max_polarity_fraction=1.0,
                warmup_epochs=0,
                ramp_epochs=1,
            ),
        ).validate()

        summary = run_training(config, output_dir=output_root / "run")
        run_dir = Path(summary["run_dir"])
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        mixture = json.loads(
            (run_dir / "sparse_mixture.json").read_text(encoding="utf-8")
        )
        reproduced = evaluate_checkpoint(config, run_dir / "best.pt")

    train_loss = history[0]["train_loss"]
    assert train_loss["full_aux_total"] > 0.0
    assert train_loss["sparse_total"] > 0.0
    assert train_loss["background_consistency"] > 0.0
    assert train_loss["sparse_false_peak"] > 0.0
    assert train_loss["gate_regularization"] > 0.0
    assert history[0]["sparse_mixture"]["augmented_files"] == 1
    assert mixture[0]["ratio_usage"] == {"1.5": 1}
    assert reproduced["event"]["event_count"] == 4


def test_training_forwards_all_events_before_sampling_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_event_counts: list[int] = []
    loss_event_counts: list[int] = []
    original_model_forward = TraceFormer.forward
    original_loss_forward = EventFocalDiceLoss.forward

    def record_model_forward(
        self: TraceFormer,
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

    monkeypatch.setattr(TraceFormer, "forward", record_model_forward)
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
