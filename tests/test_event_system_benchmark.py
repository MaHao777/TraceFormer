"""新计时入口的默认 GPU 协议、标注隔离和测试前置验收。"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from traceformer.postprocessing.contracts import EventCache, sha256_file
from traceformer.postprocessing.submission import write_prediction_npz


def benchmark_module():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_event_system.py"
    spec = importlib.util.spec_from_file_location("event_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_benchmark_uses_gpu_and_three_passes():
    module = benchmark_module()
    args = module.build_parser().parse_args([
        "--config", "config.yaml", "--checkpoint", "best.pt", "--event-release-dir", "release",
        "--dataset-root", "data", "--output", "timing.json", "--validation-report", "validation.json",
    ])
    assert (args.device, args.prediction_backend, args.feature_backend) == ("cuda", "cuda", "cuda")
    assert args.repeats == 3 and args.split == "val"
    assert args.prediction_threads == 8


def test_benchmark_repeats_and_refuses_unverified_test(tmp_path, monkeypatch):
    module = benchmark_module()
    for name in ("config.yaml", "best.pt", "release_report.json"):
        (tmp_path / name).write_bytes(b"fixture")
    metrics = {"aggregate": {"score": 0.92, "fa": 5e-6}, "files": {"val_023.npz": {"pd": 0.9}}}
    baseline = {"aggregate": {"score": 0.9, "fa": 8e-6}, "files": {"val_023.npz": {"pd": 0.8}}}
    validation = {"event_release": metrics, "original": baseline, "protocol": {
        "event_release_report_sha256": sha256_file(tmp_path / "release_report.json")}}
    (tmp_path / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(module, "load_config", lambda _: SimpleNamespace(dataset=SimpleNamespace(duration_ms=8000)))
    monkeypatch.setattr(module, "load_inference_model", lambda *a, **k: (model, {"decision_threshold": 0.5}))
    monkeypatch.setattr(module, "load_event_release", lambda *a, **k: ({}, {"event_contract": {}}))
    monkeypatch.setattr(module, "predict_event_scores", lambda *a, **k: {})
    monkeypatch.setattr(module, "event_branches", lambda cache, *a: (None, None, cache.prediction, cache.prediction))
    calls = []

    def fake_infer(model, config, sequence, threshold, *, source):
        # 真实标注为1，计时入口送入推理的标注必须为零。
        assert not sequence.labels.any() and not sequence.object_ids.any()
        calls.append(sequence.source_path.name)
        return EventCache(sequence.source_path.name, source["locations"], source["raw_features"],
                          np.ones(2, np.float32), np.zeros((1, 64), np.float32),
                          np.zeros(2, np.int64), np.ones(2, bool))

    monkeypatch.setattr(module, "infer_event_cache", fake_infer)
    reference = tmp_path / "reference"
    data = tmp_path / "data/val"
    data.mkdir(parents=True)
    for index in range(24):
        name = f"val_{index:03d}.npz"
        raw = np.zeros(2, dtype=[("x", "i4"), ("y", "i4"), ("t", "f8"),
                               ("p", "i4"), ("label", "i4"), ("name", "i4")])
        raw["label"], raw["name"] = 1, 1
        np.savez(data / name, ev=raw, ev_loc=np.zeros((2, 3)), evs_norm=np.zeros((2, 6)))
        write_prediction_npz(reference / name, np.ones(2, bool))
    args = module.build_parser().parse_args([
        "--config", str(tmp_path / "config.yaml"), "--checkpoint", str(tmp_path / "best.pt"),
        "--event-release-dir", str(tmp_path), "--dataset-root", str(tmp_path / "data"),
        "--output", str(tmp_path / "timing.json"), "--validation-report", str(tmp_path / "validation.json"),
        "--device", "cpu", "--prediction-backend", "cpu", "--feature-backend", "cpu",
        "--reference-dir", str(reference), "--repeats", "2",
    ])
    module.run(args)
    report = json.loads(args.output.read_text(encoding="utf-8"))
    assert report["reference_masks_exact"] and report["repeat_masks_exact"]
    assert report["identity"]["torch_threads"] == args.num_threads
    assert report["environment"]["cpu_tree_threads_requested"] == 8
    assert len(calls) == 1 + 24 * 2
    args.split = "test"
    with pytest.raises(ValueError, match="validation-evidence"):
        module.run(args)
    # 单网络计时不得创建后处理缓存、加载树或执行轨迹。
    args.split, args.mode, args.repeats = "val", "network", 1
    args.output = tmp_path / "network.json"
    monkeypatch.setattr(module, "infer_event_output", lambda *a, **k: {"event_logits": torch.ones(2)})
    monkeypatch.setattr(module, "infer_event_cache", lambda *a, **k: pytest.fail("network created cache"))
    monkeypatch.setattr(module, "predict_event_scores", lambda *a, **k: pytest.fail("network used event trees"))
    module.run(args)
    report = json.loads(args.output.read_text(encoding="utf-8"))
    assert report["mode"] == "network" and not report["active_event_models"]
    # 诊断授权只允许评测候选，不修改原验收结论或正式导出限制。
    validation["event_release"]["aggregate"]["fa"] = 7e-6
    (tmp_path / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
    args.output = tmp_path / "diagnostic.json"
    with pytest.raises(ValueError, match="local validation"):
        module.run(args)
    args.diagnostic = True
    module.run(args)
    assert not json.loads(args.output.read_text())["local_validation_gate"]["passed"]
    args.export_track2 = True
    with pytest.raises(ValueError, match="non-diagnostic"):
        module.run(args)
