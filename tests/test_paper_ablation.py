from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from test_training_engine import _config, _write_sparse_mixture_sequence

from traceformer.experiments.ablation import AblationSpec, NoRefinement, NoSparse
from traceformer.experiments.suite import job_config, stage_jobs
from traceformer.training.engine import (
    _dataset,
    _forward_with_sparse_mixture,
    _initialize_ablation_stage,
    _validate_checkpoint_contract,
    build_model,
    evaluate_checkpoint,
    run_training,
)


@pytest.fixture
def small_config(tmp_path):
    for split in ("train", "val"):
        folder = tmp_path / "data" / split
        folder.mkdir(parents=True)
        _write_sparse_mixture_sequence(folder / f"{split}_000.npz")
    config = _config(Path.cwd() / "data", tmp_path / "out")
    return replace(config, dataset=replace(config.dataset, root=tmp_path / "data"), sparse_mixture=replace(
        config.sparse_mixture, min_events=1, max_polarity_fraction=1.0,
        warmup_epochs=0, ramp_epochs=1,
    ))


def test_dag_has_eight_results_and_fourteen_stages():
    jobs = stage_jobs()
    seen = set()
    assert len(jobs) == 14
    assert {job.result_id for job in jobs if job.result_id} == {f"A{i}" for i in range(8)}
    for job in jobs:
        assert job.parent is None or job.parent in seen
        job.spec.validate_stage(job.stage)
        seen.add(job.name)


def test_full_ablation_matches_production_exactly(small_config):
    torch.manual_seed(37)
    production = build_model(small_config).eval()
    torch.manual_seed(37)
    experiment = build_model(small_config, ablation=AblationSpec()).eval()
    events = _dataset(small_config, "train")[0].events
    for name, value in production.state_dict().items():
        assert torch.equal(value, experiment.state_dict()[name])
    with torch.no_grad():
        for sparse in (None, events[:2]):
            kwargs = {} if sparse is None else {
                "sparse_events": sparse, "sparse_source_indices": torch.tensor([0, 1]),
            }
            left, right = production(events, **kwargs), experiment(events, **kwargs)
            assert all(torch.equal(left[key], right[key]) for key in left)


@pytest.mark.parametrize("backbone", ["cone", "window", "past", "no_position"])
def test_base_removes_modules_and_preserves_event_universe(small_config, backbone):
    model = build_model(small_config, ablation=AblationSpec(backbone, False, "none"))
    assert isinstance(model.sparse_mixture, NoSparse)
    assert isinstance(model.event_refinement_head, NoRefinement)
    sequence = _dataset(small_config, "train")[0]
    output, view = _forward_with_sparse_mixture(
        model, sequence, sequence.events, torch.zeros(sequence.length, dtype=torch.long),
        small_config, mixture_weight=1.0,
    )
    assert view is None
    assert output["event_logits"].shape == (sequence.length,)
    assert torch.equal(output["event_logits"], output["full_event_logits"])
    output["event_logits"].sum().backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert model(torch.empty(0, 4))["event_logits"].shape == (0,)


def test_same_scale_matches_parameter_count_and_supervision(small_config):
    full = build_model(small_config, ablation=AblationSpec())
    same = build_model(small_config, ablation=AblationSpec(refinement="same_scale"))
    assert sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in same.parameters())
    assert same.event_refinement_head.scales == (small_config.frame_refinement.scales[-1],) * 3


def test_window_preserves_horizon_cap_and_mean_area(small_config):
    model = build_model(small_config, ablation=AblationSpec("window", False, "none"))
    import math
    for layer in model.encoder.layers:
        builder = layer.attention.neighbor_builder
        assert builder.max_neighbors_per_cone == small_config.model.max_neighbors_per_cone
        for original, window in zip(small_config.model.cone_heads, builder.cone_heads, strict=True):
            h = math.ceil(original.max_time_distance * builder.layer_time_scale)
            assert window.max_time_distance == original.max_time_distance
            assert window.slope == 0
            assert window.base_radius**2 == pytest.approx(
                sum((original.base_radius + original.slope*d)**2 for d in range(1, h+1)) / h
            )


def test_past_attention_ignores_future_and_other_batches(small_config):
    model = build_model(small_config, ablation=AblationSpec("past", False, "none"))
    attention = model.encoder.layers[-1].attention.eval()
    tokens = torch.randn(4, small_config.model.token_dim)
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 0]])
    batches = torch.tensor([0, 0, 0, 1])
    expected = attention(tokens, coords, batches).features
    changed = tokens.clone()
    changed[2:] += 100
    actual = attention(changed, coords, batches).features
    assert torch.equal(expected[1], actual[1])
    assert not hasattr(attention, "direction_gate")
    assert attention.direction_embeddings.shape[0] == 1


def test_all_stages_train_reload_and_reject_wrong_contract(small_config, tmp_path):
    # 微型数据覆盖整个 DAG，验证初始化、冻结、checkpoint 和独立复评。
    paths = {}
    for job in stage_jobs():
        config = job_config(small_config, job, 37, tmp_path)
        config = replace(config, training=replace(config.training, epochs=1, num_threads=1))
        directory = tmp_path / job.name
        args = {}
        if job.parent:
            key = "initial_refinement_checkpoint_path" if job.stage == "refinement" else "initial_full_checkpoint_path"
            args[key] = paths[job.parent]
        run_training(config, output_dir=directory, ablation=job.spec, **args)
        checkpoint_path = directory / "best.pt"
        paths[job.name] = checkpoint_path
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        with pytest.raises(ValueError, match="ablation"):
            _validate_checkpoint_contract(checkpoint, config)
        assert evaluate_checkpoint(config, checkpoint_path, ablation=job.spec) == json.loads(
            (directory / "metrics.json").read_text(encoding="utf-8")
        )
        if job.parent:
            upstream = torch.load(paths[job.parent], weights_only=True)
            for name, value in upstream["model_state"].items():
                assert torch.equal(value, checkpoint["model_state"][name]), name
            assert checkpoint["full_initialization"]["sha256"]
            wrong = replace(config, training=replace(config.training, seed=73))
            with pytest.raises(ValueError, match="seed"):
                _initialize_ablation_stage(
                    build_model(config, ablation=job.spec), wrong, job.spec, paths[job.parent],
                    manifest_digest=checkpoint["dataset_manifest_sha256"], device=torch.device("cpu"),
                )
        assert checkpoint["ablation"] == asdict(job.spec)
        assert checkpoint["training_seed"] == 37


def test_ablation_resume_recovers_identical_training(small_config, tmp_path):
    job = stage_jobs()[0]
    config = job_config(small_config, job, 37, tmp_path)
    config = replace(config, training=replace(config.training, epochs=2, save_training_state_interval=1))
    original = tmp_path / "original"
    resumed = tmp_path / "resumed"
    run_training(config, output_dir=original, ablation=job.spec)
    run_training(config, output_dir=resumed, ablation=job.spec,
                 resume_state_path=original / "training_states" / "epoch_001.pt")
    left = torch.load(original / "training_state.pt", weights_only=False)
    right = torch.load(resumed / "training_state.pt", weights_only=False)
    assert all(torch.equal(value, right["model_state"][key]) for key, value in left["model_state"].items())
    assert left["best_metrics"] == right["best_metrics"]


def test_independent_report_and_saved_predictions(small_config, tmp_path):
    from traceformer.experiments.report import evaluate_ablation

    job = stage_jobs()[0]
    config = job_config(small_config, job, 37, tmp_path)
    config = replace(config, training=replace(config.training, epochs=1, num_threads=1))
    directory = tmp_path / "report"
    run_training(config, output_dir=directory, ablation=job.spec)
    report = evaluate_ablation(config, directory / "best.pt", job.spec, save_predictions=True)
    assert report["metrics"] == json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert report["fixed_threshold"]["threshold"] == 0.5
    assert report["groups"]["sparse_eligible"]["validation_files"] == 1
    assert report["per_file"]["val_000"]["refinement_count_delta"] == {"tp": 0, "fp": 0, "fn": 0}
    saved = torch.load(directory / "predictions" / "val_000.pt", weights_only=True)
    assert saved["event_logits"].shape == (4,)
