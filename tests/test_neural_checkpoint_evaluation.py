"""独立测试测试集预测只使用验证阈值，且保存结果可以再次计分。"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from traceformer.config import TraceFormerConfig
from traceformer.training.engine import _dataset, _dataset_manifest, _manifest_identity_digest


def test_neural_test_prediction_is_annotation_isolated_and_rescorable(tmp_path, monkeypatch):
    from dataclasses import replace

    source = Path(__file__).parents[1] / "scripts/evaluate_neural_checkpoint.py"
    spec = importlib.util.spec_from_file_location("evaluate_neural_checkpoint", source)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    dtype = [("x", "i2"), ("y", "i2"), ("t", "f8"), ("p", "i1"), ("label", "i1"), ("name", "i1")]
    events = np.array([(1, 1, 0.0, 1, 1, 1), (2, 2, 2.0, 0, 0, 0)], dtype=dtype)
    for split in ("val", "test"):
        (tmp_path / split).mkdir()
        np.savez(tmp_path / split / f"{split}_000.npz", ev=events)
    base = TraceFormerConfig.default()
    config = replace(base, dataset=replace(base.dataset, root=tmp_path))
    monkeypatch.setattr(evaluator, "load_config", lambda _: config)
    checkpoint = {
        "decision_threshold": 0.7,
        "validation_manifest_sha256": _manifest_identity_digest(
            _dataset_manifest(_dataset(config, "val"))
        ),
    }
    monkeypatch.setattr(
        evaluator, "load_inference_model", lambda *a, **k: (torch.nn.Linear(1, 1), checkpoint)
    )
    calls = []

    def infer(model, sequence, events, batches, config, **kwargs):
        assert sequence.labels.eq(0).all() and sequence.object_ids.eq(0).all()
        calls.append(True)
        return {"event_logits": torch.logit(torch.tensor([0.8, 0.6]))}, None

    monkeypatch.setattr(evaluator, "forward_with_sparse_mixture", infer)
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"checkpoint test identity")
    args = SimpleNamespace(
        config=None,
        checkpoint=checkpoint_path,
        dataset_root=tmp_path,
        output=tmp_path / "result",
        split="test",
        device="cpu",
        num_threads=1,
        rescore_only=False,
    )
    evaluator.evaluate(args)
    original = json.loads((args.output / "report.json").read_text())
    assert original["threshold"] == 0.7
    assert original["test_threshold_selection"] is False
    with np.load(args.output / "predictions/test_000.npz") as saved:
        assert saved["prediction"].tolist() == [True, False]
    args.rescore_only = True
    evaluator.evaluate(args)
    assert len(calls) == 1
    independent = json.loads((args.output / "independent_report.json").read_text())
    assert independent["competition"] == original["competition"]
    np.savez(args.output / "predictions/test_000.npz", prediction=np.array([True, True]))
    with pytest.raises(ValueError, match="saved mask differs"):
        evaluator.evaluate(args)
