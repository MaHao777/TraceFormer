"""用解析的小轨迹检查几何诊断和统一评分的边界行为。"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from traceformer.config import TraceFormerConfig


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probability_scoring_keeps_exact_half_threshold_and_original_order():
    evaluator = script("evaluate_external_baseline")
    sequence = SimpleNamespace(
        labels=torch.tensor([1.0, 0.0, 1.0]),
        object_ids=torch.tensor([1, 0, 1]),
        timestamps_ms=torch.tensor([10.0, 20.0, 60.0]),
        events=torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )
    probability = torch.tensor([0.5, 0.50000006, 0.49999997])
    result = evaluator.probability_metrics([(probability, sequence)], TraceFormerConfig(), 0.5)
    assert (result["event"]["tp"], result["event"]["fp"], result["event"]["fn"]) == (1, 1, 1)
    assert result["threshold"] == 0.5
