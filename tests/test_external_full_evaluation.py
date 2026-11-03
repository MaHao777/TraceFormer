"""核对全量外部模型批跑、基准指标和回传身份，不模拟神经网络性能。"""

import ast
import importlib
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "remote/external_qualitative/tools"
SNAPSHOT = Path(__file__).resolve().parents[1] / "remote_upload/external_qualitative_windows_20261101"


@pytest.fixture
def helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    return tuple(importlib.import_module(name) for name in
                 ("event_io", "benchmark_metrics", "analyze_results", "run_full"))


def events(time, x, labels, identities):
    raw = np.zeros(len(time), dtype=[("x", "i2"), ("y", "i2"), ("t", "f8"),
                                     ("p", "i1"), ("label", "i1"), ("name", "i1")])
    raw["t"], raw["x"], raw["label"], raw["name"] = time, x, labels, identities
    return raw


def test_counts_use_global_ratios_and_reference_frame_boundaries(helpers):
    _, metrics, _, _ = helpers
    raw = events([0, 1, 2, 3, 4, 5, 50, 100, 101],
                 [0, 1, 2, 10, 11, 30, 2, 2, 1],
                 [1, 1, 1, 0, 0, 0, 1, 1, 1], [1, 1, 2, 0, 0, 0, 2, 2, 1])
    result = metrics.measure(raw, np.array([0, 1, 0, 1, 1, 1, 1, 1, 1]))
    assert result["counts"] == dict(tp=4, fp=3, fn=2, tn=0, detected_objects=2,
                                    objects=3, false_components=2, frames=2)
    assert result["iou"] == 4 / 9
    assert result["seg_acc"] == 4 / 6
    assert result["pd"] == 2 / 3
    assert result["fa"] == 2 / (2 * 346 * 260)


def reference_object_counts(model, raw, prediction):
    # 直接执行固定官方源码中的评估类，作为独立oracle；只调用CPU的Pd/Fa路径。
    import torch

    cv2 = pytest.importorskip("cv2")
    path = SNAPSHOT / "models" / model / "utils/eval.py"
    if not path.is_file():
        pytest.skip("Official snapshot not present")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = dict(torch=torch, np=np, cv2=cv2, pd=SimpleNamespace(DataFrame=lambda: None))
    exec(compile(tree, str(path), "exec"), namespace)
    evaluator = namespace["evalute"](SimpleNamespace(roc=True, pd_detT=50, correct_thresh=0.0001))
    locations = torch.tensor(np.column_stack((np.zeros(len(raw)), raw["x"], raw["y"],
                                              raw["t"].astype(np.int64))), dtype=torch.float32)
    with np.errstate(over="ignore"):
        evaluator.roc_update(locations[:, 3], torch.tensor(prediction, dtype=torch.float32),
                             raw["name"], torch.tensor(raw["label"], dtype=torch.float32), locations)
    return dict(objects=int(evaluator.obj_num), detected_objects=int(evaluator.correct_num),
                false_components=int(evaluator.false_num), frames=int(evaluator.frame_num))


def test_matches_official_reference_including_uint8_wrap(helpers):
    _, metrics, _, _ = helpers
    raw = events([0, 1] + [10] * 256 + [51, 101],
                 [1, 1] + [10] * 256 + [12, 1],
                 [1, 1] + [0] * 256 + [0, 1], [1, 1] + [0] * 256 + [0, 1])
    prediction = np.ones(len(raw), dtype=np.uint8)
    result = metrics.measure(raw, prediction)["counts"]
    expected = reference_object_counts("ev_spsegnet", raw, prediction)
    assert {key: result[key] for key in expected} == expected
    assert result["false_components"] == 1


def test_pact_legacy_fa_differs_due_to_per_event_accumulation(helpers):
    _, metrics, _, _ = helpers
    raw = events([0, 1, 2, 3, 101], [1, 10, 11, 30, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1])
    prediction = np.ones(len(raw), dtype=np.uint8)
    benchmark = metrics.measure(raw, prediction)["counts"]["false_components"]
    legacy = reference_object_counts("pact", raw, prediction)["false_components"]
    assert benchmark == 2 and legacy == 4


def make_split(io_helper, root, split, names, models):
    records = []
    for index, name in enumerate(names):
        raw = events([0, 1, 51, 101], [1, 1, 10, 1], [1, 1, 0, 1], [1, 1, 0, 1])
        if index:
            raw["label"] = raw["name"] = 1
        record = dict(file=name, sha256=f"source-{index}", events=4)
        records.append(record)
        io_helper.save_input(root, Path(name), record, raw, raw["label"])
        for model in models:
            probability = np.array([0.95, 0.1, 0.2, 0.95]) if index == 0 else np.zeros(4)
            io_helper.save_prediction(root / model, name, probability, record)
    for model in models:
        io_helper.write_json(root / model / "report.json", dict(
            status="complete", model=model, split=split, threshold=0.9,
            backend="torch-reference", checkpoint_sha256="test-checkpoint", sequences=records))
    return records


def test_analyzer_recounts_micro_iou_and_flags_probability_suppression(helpers, tmp_path):
    io_helper, _, analyzer, _ = helpers
    records = make_split(io_helper, tmp_path, "test", ["test_000.npz", "test_001.npz"], ["pact"])
    totals, rows = analyzer.analyze_split(tmp_path, "test", records)
    assert len(totals) == 1 and len(rows) == 2
    assert totals[0]["iou"] == 2 / 7
    assert totals[0]["iou_mean_sequence"] == 1 / 3
    assert totals[0]["zero_positive_sequences"] == ["test_001.npz"]
    assert totals[0]["counts"]["tp"] == 2
    assert rows[1]["positives_at_05_diagnostic"] == 0
    assert totals[0]["auc_mean_sequence"] == np.mean([r["auc"] for r in rows if r["auc"] is not None])


def test_full_return_archive_checks_nested_members_and_marks_partial(helpers, tmp_path):
    io_helper, _, analyzer, _ = helpers
    root = tmp_path / "results"
    records = {split: make_split(io_helper, root / split, split, [f"{split}_000.npz"], ["pact"])
               for split in ("val", "test")}
    io_helper.write_json(root / "dataset_inventory.json", dict(splits=records))
    archive_path = tmp_path / "return.zip"
    summary = analyzer.collect_full(root, archive_path)
    assert summary["status"] == "partial" and summary["prediction_files"] == 2
    with zipfile.ZipFile(archive_path) as archive:
        sums = archive.read("external_full_results/SHA256SUMS").decode().splitlines()
        import hashlib

        for line in sums:
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read("external_full_results/" + name)).hexdigest() == digest
        with zipfile.ZipFile(io.BytesIO(archive.read("external_full_results/test.zip"))) as split_archive:
            assert "external_qualitative_results/pact/predictions/test_000.npz" in split_archive.namelist()
        metrics = json.loads(archive.read("external_full_results/analysis/metrics.json"))
        assert all("paper_difference" not in row for row in metrics["summary"])
    with pytest.raises(FileExistsError):
        analyzer.collect_full(root, archive_path)


def test_completed_job_reuse_requires_matching_identity_and_keeps_failed_attempts(helpers, tmp_path):
    io_helper, _, _, runner = helpers
    records = [dict(file="test_000.npz", sha256="source", events=4)]
    folder = tmp_path / "test/pact"
    io_helper.write_json(folder / "report.json", dict(status="failed"))
    assert not runner.reuse_or_preserve(folder, records, "test", "weight")
    assert (tmp_path / "test/_failed_attempts/pact_001/report.json").is_file()
    io_helper.write_json(folder / "report.json", dict(status="complete", split="test", threshold=0.9,
                         backend="torch-reference", checkpoint_sha256="weight", sequences=records))
    assert runner.reuse_or_preserve(folder, records, "test", "weight")
    with pytest.raises(ValueError, match="different identities"):
        runner.reuse_or_preserve(folder, records, "test", "different")


def test_all_144_predictions_receive_test_only_paper_deltas(helpers, tmp_path):
    io_helper, _, analyzer, _ = helpers
    root = tmp_path / "results"
    records = {split: make_split(io_helper, root / split, split,
                                [f"{split}_{i:03d}.npz" for i in range(24)], analyzer.MODELS)
               for split in ("val", "test")}
    io_helper.write_json(root / "dataset_inventory.json", dict(splits=records))
    summary = analyzer.collect_full(root, tmp_path / "full.zip")
    assert summary["status"] == "complete" and summary["prediction_files"] == 144
    metrics = json.loads((root / "analysis/metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["summary"]) == 6 and len(metrics["per_sequence"]) == 144
    assert all(("paper_difference" in row) == (row["split"] == "test") for row in metrics["summary"])
    assert "paper_delta_iou_percentage_points" in (root / "analysis/metrics_summary.csv").read_text()


def test_runner_launches_six_full_split_jobs_then_reuses_complete_jobs(helpers, tmp_path, monkeypatch):
    io_helper, _, analyzer, runner = helpers
    predictor = importlib.import_module("predict")
    fake_bundle = tmp_path / "bundle"
    fake_bundle.mkdir()
    io_helper.write_json(fake_bundle / "qualitative_selection.json", {})
    for relative in predictor.CHECKPOINTS.values():
        path = fake_bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    records = {split: [dict(file=f"{split}_000.npz", sha256="input", events=4)]
               for split in ("val", "test")}
    monkeypatch.setattr(runner, "BUNDLE", fake_bundle)
    monkeypatch.setattr(runner, "inventory", lambda *args: dict(splits=records))
    calls = []

    def fake_command(arguments, log):
        # 只测试进程调度与恢复，不用桩输出宣称GPU推理或论文复现。
        if "--model" not in arguments:
            return 0
        assert "--sequences" not in arguments
        model = arguments[arguments.index("--model") + 1]
        split = arguments[arguments.index("--split") + 1]
        calls.append((split, model))
        folder = Path(arguments[arguments.index("--output-root") + 1]) / model
        if len(calls) == 2:
            io_helper.write_json(folder / "report.json", dict(status="failed"))
            return 1
        io_helper.write_json(folder / "report.json", dict(status="complete", split=split, model=model,
                             backend="torch-reference", threshold=0.9, sequences=records[split],
                             checkpoint_sha256=io_helper.sha256(fake_bundle / predictor.CHECKPOINTS[model])))
        return 0

    monkeypatch.setattr(runner, "command", fake_command)
    monkeypatch.setattr(analyzer, "collect_full", lambda *args: {"status": "complete"})
    root = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["run_full", "--data-root", str(tmp_path), "--output-root", str(root)])
    with pytest.raises(SystemExit, match="Partial run"):
        runner.main()
    assert len(calls) == 6 and set(calls) == {(s, m) for s in records for m in runner.MODELS}
    runner.main()
    assert len(calls) == 7 and calls[-1] == ("val", "pact")
    status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert sum(j["status"] == "reused_complete" for j in status["jobs"]) == 5
