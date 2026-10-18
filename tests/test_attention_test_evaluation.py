"""测试消融权重的显式加载、测试标签隔离及固定验证阈值复评。"""

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest
import test_paper_ablation
import torch
from test_neural_checkpoint_evaluation import load_evaluator

from traceformer.experiments.suite import attention_stage_jobs
from traceformer.training.engine import run_training

small_config = test_paper_ablation.small_config


@pytest.mark.parametrize("job", attention_stage_jobs(), ids=lambda job: job.name)
def test_ablation_test_uses_saved_threshold_and_rejects_wrong_model(small_config, tmp_path, job, monkeypatch):
    config = replace(small_config, training=replace(
        small_config.training, stage="full", epochs=1, num_threads=1,
    ))
    run = tmp_path / job.name
    run_training(config, output_dir=run, ablation=job.spec)
    root = config.dataset.root
    (root / "test").mkdir()
    shutil.copy2(root / "val/val_000.npz", root / "test/test_000.npz")
    evaluator = load_evaluator()
    original_forward = evaluator.forward_with_sparse_mixture
    calls = []

    def infer(model, sequence, *args, **kwargs):
        # 原始测试标签只用于后续评分，不能传给模型或子流生成器。
        assert sequence.labels.eq(0).all() and sequence.object_ids.eq(0).all()
        assert model.ablation_spec == job.spec.to_dict()
        calls.append(True)
        return original_forward(model, sequence, *args, **kwargs)

    monkeypatch.setattr(evaluator, "forward_with_sparse_mixture", infer)
    args = SimpleNamespace(
        config=run / "config.yaml", checkpoint=run / "best.pt", ablation=run / "ablation.json",
        dataset_root=root, split="test", output=tmp_path / "test_result", device="cpu",
        num_threads=1, rescore_only=False,
    )
    evaluator.evaluate(args)
    report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    assert report["threshold"] == checkpoint["decision_threshold"]
    assert report["ablation"] == job.spec.to_dict() and report["test_threshold_selection"] is False
    args.rescore_only = True
    evaluator.evaluate(args)
    rescored = json.loads((args.output / "independent_report.json").read_text(encoding="utf-8"))
    assert rescored["competition"] == report["competition"] and len(calls) == 1
    # 窗口与锥的权重形状相同，仍必须拒绝错误几何契约。
    wrong = next(other.spec for other in attention_stage_jobs() if other != job)
    args.ablation = tmp_path / "wrong.json"
    args.ablation.write_text(json.dumps(wrong.to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="ablation"):
        evaluator.evaluate(args)
