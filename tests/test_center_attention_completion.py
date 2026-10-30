"""用微型训练和24个微型测试文件验证自动计分、计时与科研图表闭环。"""

from __future__ import annotations

import importlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import test_paper_ablation

from traceformer.config import ResolutionConfig
from traceformer.experiments.report import evaluate_ablation
from traceformer.experiments.suite import center_attention_stage_jobs
from traceformer.training.engine import (
    _dataset,
    _dataset_manifest,
    _file_sha256,
    _manifest_identity_digest,
    _write_json,
    run_training,
)

small_config = test_paper_ablation.small_config


def test_finalization_scores_masks_and_runs_three_benchmark_passes(small_config, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path.cwd() / "archive/experiments/scripts"))
    finalizer = importlib.import_module("finish_center_attention")
    config = replace(small_config, resolution=ResolutionConfig(), training=replace(
        small_config.training, stage="full", epochs=1, num_threads=1, seed=37,
    ))
    test = config.dataset.root / "test"
    test.mkdir()
    for index in range(24):
        shutil.copy2(config.dataset.root / "val/val_000.npz", test / f"test_{index:03d}.npz")
    root = tmp_path / "suite"
    jobs = []
    for job in center_attention_stage_jobs():
        folder = root / job.result_id
        run_training(config, output_dir=folder, ablation=job.spec)
        evaluate_ablation(config, folder / "best.pt", job.spec, save_predictions=True)
        jobs.append({"id": job.result_id, "result_id": job.result_id, "seed": 37,
                     "spec": job.spec.to_dict(), "config": str(folder / "config.yaml"),
                     "config_sha256": _file_sha256(folder / "config.yaml")})
        _write_json(folder / "done.json", {"checkpoint": str(folder / "best.pt"),
                    "sha256": _file_sha256(folder / "best.pt"), "report": str(folder / "ablation_report.json")})
    _write_json(root / "suite.json", {
        "protocol": "center_attention_four_variants", "jobs": jobs,
        "test_dataset_root": str(config.dataset.root), "source_snapshot": str(Path.cwd()),
        "dataset_manifest_sha256": _manifest_identity_digest(_dataset_manifest(
            _dataset(config, "train"), _dataset(config, "val"))),
        "comparisons": [["A3", "A0"], ["A1", "A0"], ["A2", "A1"], ["A3", "A2"]],
    })
    _write_json(root / "status.json", {"status": "complete"})
    _write_json(root / "source_hashes.json", {})
    finalizer.finish(root)
    report = json.loads((root / "final_report.json").read_text(encoding="utf-8"))
    assert report["validation_raw_audit"] is True
    assert len(report["test"]["results"]) == 4
    assert len(report["test"]["comparisons"]) == 4
    for variant, benchmark in report["benchmarks"].items():
        assert benchmark["passes"] == 3 and benchmark["sequences"] == 24
        assert all(len(times) == 3 for times in benchmark["per_sequence_seconds"].values())
        assert (root / "curves" / f"{variant}.json").exists()
    assert (root / "figures/structure.pdf").exists()
    assert (root / "figures/tradeoff.pdf").exists()


def test_finalization_rejects_changed_test_data(small_config, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path.cwd() / "archive/experiments/scripts"))
    finalizer = importlib.import_module("finish_center_attention")
    test = small_config.dataset.root / "test"
    test.mkdir()
    shutil.copy2(small_config.dataset.root / "val/val_000.npz", test / "test_000.npz")
    root = tmp_path / "suite"
    root.mkdir()
    _write_json(root / "suite.json", {"protocol": "center_attention_four_variants",
                "test_dataset_root": str(small_config.dataset.root),
                "test_dataset_manifest_sha256": "changed"})
    _write_json(root / "status.json", {"status": "complete"})
    with pytest.raises(ValueError, match="test content changed"):
        finalizer.finish(root)
