"""用隔离的小产物核验调度、Full验收和26到6的固定回退规则。"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

from traceformer.experiments.component_suite import audit_baseline, read_json
from traceformer.training.engine import (
    _dataset,
    _dataset_manifest,
    _file_sha256,
    _manifest_identity_digest,
    _serializable,
    _write_json,
    build_model,
)


@pytest.fixture
def runner():
    scripts = Path(__file__).resolve().parents[1] / "archive/experiments/scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("component_queue_test", scripts / "run_component_core.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(scripts))


@pytest.fixture
def accepted_artifacts(tmp_path):
    from test_training_engine import _config, _write_sequence

    root, data = tmp_path / "baseline", tmp_path / "data"
    training = root / "training"
    training.mkdir(parents=True)
    for split, count in (("train", 99), ("val", 24)):
        (data / split).mkdir(parents=True)
        for index in range(count):
            _write_sequence(data / split / f"{split}_{index:03}.npz", 0)
    config = _config(data, root)
    config = replace(config, dataset=replace(config.dataset, root=data),
                     training=replace(config.training, seed=37, epochs=24, num_threads=8,
                                      learning_rate=0.0007, weight_decay=0.0002))
    serialized = _serializable(asdict(config))
    (training / "config.yaml").write_text(yaml.safe_dump(serialized), encoding="utf-8")
    manifest = _dataset_manifest(_dataset(config, "train"), _dataset(config, "val"))
    checkpoint = {"schema_version": 8, "model_state": build_model(config).state_dict(),
                  "ablation": None, "model_config": serialized["model"],
                  "frame_refinement_config": serialized["frame_refinement"],
                  "resolution": serialized["resolution"], "duration_ms": config.dataset.duration_ms,
                  "decision_threshold": 0.5, "dataset_manifest_sha256": _manifest_identity_digest(manifest)}
    torch.save(checkpoint, training / "best.pt")
    for name in ("metrics.json", "reproduced_metrics.json"):
        _write_json(training / name, {"competition": {"score": 0.5}})
    _write_json(training / "history.json", [{"epoch": i} for i in range(1, 25)])
    _write_json(root / "status.json", {"status": "complete"})
    _write_json(root / "paper_verification.json", {"unchanged": True})
    (root / "test").mkdir()
    report = {"competition": {"score": 0.5}, "checkpoint_sha256": _file_sha256(training / "best.pt"),
              "threshold": 0.5, "parameters": sum(p.numel() for p in build_model(config).parameters()),
              "sequences": 24, "ablation": None, "validation_identity": "fixture",
              "files": {f"test_{i:03}.npz": {"events": 4, "source_sha256": "fixture",
                        "prediction_sha256": "fixture", "seconds": 1} for i in range(24)},
              "per_sequence": {f"test_{i:03}.npz": {} for i in range(24)}}
    _write_json(root / "test/report.json", report)
    for record in report["files"].values():
        record["seconds"] = 0.1
    _write_json(root / "test/independent_report.json", report)
    return root, config


def test_baseline_accepts_rescored_timing_difference_and_records_reuse(accepted_artifacts):
    root, _ = accepted_artifacts
    result = audit_baseline(root)
    assert result["checkpoint"].endswith("best.pt")
    assert result["sha256"] == _file_sha256(root / "training/best.pt")
    json.dumps(result)  # 可直接写入suite，不留下Path等非JSON类型。


@pytest.mark.parametrize("change", ["threshold", "paper", "data", "schema"])
def test_baseline_refuses_changed_evidence(accepted_artifacts, change):
    root, config = accepted_artifacts
    if change == "threshold":
        path = root / "test/independent_report.json"
        report = read_json(path)
        report["threshold"] = 0.9
        _write_json(path, report)
    elif change == "paper":
        _write_json(root / "paper_verification.json", {"unchanged": False})
    elif change == "data":
        path = Path(config.dataset.root) / "train/train_000.npz"
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        path = root / "training/best.pt"
        saved = torch.load(path, weights_only=True)
        saved["schema_version"] = 7
        torch.save(saved, path)
    with pytest.raises((ValueError, RuntimeError)):
        audit_baseline(root)


def test_prepare_creates_only_ten_random_start_jobs(runner, accepted_artifacts, tmp_path):
    baseline, _ = accepted_artifacts
    full = audit_baseline(baseline)
    root = tmp_path / "suite"
    root.mkdir()
    test_root = tmp_path / "EV-UAV-dataset/test"
    test_root.mkdir(parents=True)
    for index in range(24):
        (test_root / f"test_{index:03}.npz").write_bytes(b"fixture")
    report = read_json(full["test_report"])
    for name, row in report["files"].items():
        row["source_sha256"] = _file_sha256(test_root / name)
    _write_json(Path(full["test_report"]), report)

    def snapshot(target):
        (target / "source").mkdir()
        _write_json(target / "source_hashes.json", {})

    def execute(target, phase, command, environment, capture=None):
        capture.parent.mkdir(parents=True, exist_ok=True)
        for index in range(24):
            torch.save({"event_logits": torch.zeros(4), "event_token_indices": torch.zeros(4).long(),
                        "token_coords": torch.zeros(1, 3).long()}, capture.parent / f"val_{index:03}.pt")
        _write_json(capture, {"files": 24})

    with patch.object(runner, "source_snapshot", side_effect=snapshot), \
         patch.object(runner, "execute", side_effect=execute), patch.object(runner, "REPO", tmp_path / "repo"):
        suite, _ = runner.prepare(root, baseline, full)
    assert len(suite["jobs"]) == 10
    assert {job["name"] for job in suite["jobs"]} == {"M1", "M2", "M3", "M4", "M5",
                                                       "C1", "C2", "C3", "C4", "C5"}
    assert all(job["parent"] is None and job["stage"] == "full" for job in suite["jobs"])
    assert all("sparse" not in job["spec"] for job in suite["jobs"])
    assert read_json(root / "full_equivalence.json")["exact"] is True


def test_resource_selection_freezes_six_neighbors_before_training(runner, tmp_path):
    root = tmp_path / "suite"
    root.mkdir()
    suite = {"source_snapshot": str(tmp_path / "source"), "jobs": [
        {"name": name, "config": "fixture", "spec": {"version": 2, "variant": name}}
        for name in ("M1", "M2", "M3", "M4", "M5", "C1", "C2", "C3", "C4", "C5")]}

    def execute(target, phase, command, environment, capture=None):
        variant = command[command.index("--gate") + 1]
        neighbors = int(command[command.index("--neighbors") + 1])
        sequence = command[command.index("--sequence") + 1]
        seconds = 21 if variant == "C1" and neighbors == 26 else 10
        _write_json(capture, {"sequence": sequence, "median_seconds": seconds,
                             "finite_gradients": True, "memory_passed": True})

    with patch.object(runner, "execute", side_effect=execute):
        runner.gates(root, suite, {})
    assert read_json(root / "resource_gate.json")["neighbors"] == 6
    assert next(job for job in read_json(root / "suite.json")["jobs"]
                if job["name"] == "C1")["spec"]["local_neighbors"] == 6


def test_both_local_budgets_fail_without_cancelling_other_variants(runner, tmp_path):
    root = tmp_path / "suite"
    root.mkdir()
    suite = {"source_snapshot": str(tmp_path / "source"), "jobs": [
        {"name": name, "config": "fixture", "spec": {"version": 2, "variant": name}}
        for name in ("M1", "M2", "M3", "M4", "M5", "C1", "C2", "C3", "C4", "C5")]}

    def execute(target, phase, command, environment, capture=None):
        variant = command[command.index("--gate") + 1]
        sequence = command[command.index("--sequence") + 1]
        _write_json(capture, {"sequence": sequence, "median_seconds": 21 if variant == "C1" else 10,
                             "finite_gradients": True, "memory_passed": True})

    with patch.object(runner, "execute", side_effect=execute):
        runner.gates(root, suite, {})
    result = read_json(root / "resource_gate.json")
    assert result["passed"] is False and result["runnable_passed"] is True
    assert set(result["skipped_variants"]) == {"C1"}
    assert (root / "preflight/C5_26_train_094.json").exists()


def test_frozen_checks_verify_the_snapshot_independently_of_workspace(runner, tmp_path):
    root, source = tmp_path / "run", tmp_path / "snapshot"
    root.mkdir()
    (source / "src").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "src/model.py").write_text("value = 1\n", encoding="utf-8")
    identity = runner.source_identity(source)
    _write_json(root / "implementation_checks.json", {"passed": True, "source_sha256": identity})
    runner.verify_checks(root, source)
    (source / "src/model.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        runner.verify_checks(root, source)
