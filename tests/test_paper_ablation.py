from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_training_engine import _config, _write_sparse_mixture_sequence

from traceformer.experiments.ablation import AblationSpec, NoRefinement
from traceformer.experiments.suite import job_config, stage_jobs
from traceformer.training.engine import (
    _dataset,
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
    return replace(config, dataset=replace(config.dataset, root=tmp_path / "data"))


def test_dag_has_three_current_structure_stages():
    jobs = stage_jobs()
    seen = set()
    assert len(jobs) == 3
    assert {job.result_id for job in jobs if job.result_id} == {"A0", "A2", "A7"}
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
        left, right = production(events), experiment(events)
        assert all(torch.equal(left[key], right[key]) for key in left)


@pytest.mark.parametrize("backbone", ["cone"])
def test_base_removes_modules_and_preserves_event_universe(small_config, backbone):
    model = build_model(small_config, ablation=AblationSpec(backbone, False, "none"))
    assert not hasattr(model, "sparse_mixture")
    assert isinstance(model.event_refinement_head, NoRefinement)
    sequence = _dataset(small_config, "train")[0]
    output = model(sequence.events, torch.zeros(sequence.length, dtype=torch.long))
    assert output["event_logits"].shape == (sequence.length,)
    assert torch.equal(output["event_logits"], output["base_event_logits"])
    output["event_logits"].sum().backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert model(torch.empty(0, 4))["event_logits"].shape == (0,)


def test_same_scale_matches_parameter_count_and_supervision(small_config):
    full = build_model(small_config, ablation=AblationSpec())
    same = build_model(small_config, ablation=AblationSpec(refinement="same_scale"))
    assert sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in same.parameters())
    assert same.event_refinement_head.scales == (small_config.frame_refinement.scales[-1],) * 3


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
        assert checkpoint["ablation"] == job.spec.to_dict()
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
    assert report["groups"]["ordinary"]["validation_files"] == 1
    assert report["per_file"]["val_000"]["refinement_count_delta"] == {"tp": 0, "fp": 0, "fn": 0}
    saved = torch.load(directory / "predictions" / "val_000.pt", weights_only=True)
    assert saved["event_logits"].shape == (4,)


def test_simple_suite_has_no_sparse_stage(small_config, tmp_path):
    from traceformer.experiments.suite import prepare_suite, simple_stage_jobs

    jobs = simple_stage_jobs()
    assert [(job.name, job.result_id) for job in jobs] == [
        ("B", "E1"), ("BR", "E3"),
    ]
    assert jobs[-1].parent == "B"
    assert job_config(small_config, jobs[0], 37, tmp_path).training.epochs == 18
    assert job_config(small_config, jobs[1], 37, tmp_path).training.epochs == 24
    manifest = prepare_suite(small_config, tmp_path / "simple", [37], definitions=jobs)
    assert len(manifest["jobs"]) == 2
    assert all(job["seed"] == 37 for job in manifest["jobs"])


def test_simple_summary_uses_base_and_refinement_results(small_config, tmp_path):
    from traceformer.experiments.suite import prepare_suite, simple_stage_jobs, summarize
    from traceformer.training.engine import _write_json

    root = tmp_path / "summary_simple"
    manifest = prepare_suite(small_config, root, [37], definitions=simple_stage_jobs())
    manifest["comparisons"] = [["E3", "E1"]]
    _write_json(root / "suite.json", manifest)
    for job in manifest["jobs"]:
        folder = root / job["id"]
        folder.mkdir(parents=True)
        score = .8 + int(job["result_id"][1]) * .01
        report = {
            "metrics": {"competition": {key: score for key in ("score", "iou", "seg_acc", "pd", "fa")}, "threshold": .5},
            "fixed_threshold": {"competition": {"score": score}}, "groups": {},
            "parameters": 100, "mean_seconds_per_sequence": 1., "process_peak_rss_bytes": 1000,
        }
        _write_json(folder / "report.json", report)
        _write_json(folder / "done.json", {"report": str(folder / "report.json")})
    summarize(root)
    assert "完成 2/2" in (root / "results.md").read_text(encoding="utf-8")
    aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    assert [row["comparison"] for row in aggregate["paired"]] == ["E3-E1"]
    assert all(row["score"] == pytest.approx(.02) for row in aggregate["paired"])


def test_simple_reuse_rejects_changed_source_config(small_config, tmp_path):
    from traceformer.experiments.suite import prepare_simple_suite, prepare_suite

    source = tmp_path / "prior"
    manifest = prepare_suite(small_config, source, [37])
    with Path(manifest["jobs"][0]["config"]).open("a", encoding="utf-8") as stream:
        stream.write("\n# changed after source run\n")
    destination = tmp_path / "simple_rejected"
    with pytest.raises(ValueError, match="configuration changed"):
        prepare_simple_suite(source, destination)
    assert not destination.exists()
