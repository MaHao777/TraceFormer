"""用解析的小轨迹检查几何诊断和统一评分的边界行为。"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree

from traceformer.config import ConeHeadConfig, TraceFormerConfig
from traceformer.models.cone_neighbor_builder import ConeNeighborBuilder


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cone_support_counts_displaced_target_and_excludes_near_clutter():
    diagnostic = script("analyze_cone_support")
    coords = np.array([[0, 0, 0], [2, 0, 1], [4, 0, 3], [3, 0, 1]])
    head = ConeHeadConfig(base_radius=1, slope=1, max_time_distance=3)
    builder = ConeNeighborBuilder(cone_heads=(head,), max_neighbors_per_cone=8)
    neighbors = builder(torch.from_numpy(coords), torch.zeros(4, dtype=torch.long))
    selected = torch.cat((neighbors.past_indices, neighbors.future_indices), dim=2)[:, 0].numpy()
    trees = {int(t): cKDTree(coords[coords[:, 2] == t, :2]) for t in np.unique(coords[:, 2])}
    rows = diagnostic.describe(coords, selected, head, 3, {1: np.array([0, 1, 2])},
                               {0: {1}}, trees)
    assert sum(row["possible_same"] for row in rows) == 2
    assert sum(row["eligible_same"] for row in rows) == 2
    assert sum(row["selected_same"] for row in rows) == 2
    assert rows[0]["eligible_all"] == 1  # 同一近时刻的背景 x=3 被排除。
    assert rows[2]["selected_same"] == 1  # 远时刻位移 x=4 仍可关联。


def test_probability_scoring_keeps_exact_half_threshold_and_original_order():
    evaluator = script("evaluate_external_baseline")
    sequence = SimpleNamespace(
        labels=torch.tensor([1.0, 0.0, 1.0]), object_ids=torch.tensor([1, 0, 1]),
        timestamps_ms=torch.tensor([10.0, 20.0, 60.0]),
        events=torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )
    probability = torch.tensor([0.5, 0.50000006, 0.49999997])
    result = evaluator.probability_metrics([(probability, sequence)], TraceFormerConfig(), 0.5)
    assert (result["event"]["tp"], result["event"]["fp"], result["event"]["fn"]) == (1, 1, 1)
    assert result["threshold"] == 0.5
