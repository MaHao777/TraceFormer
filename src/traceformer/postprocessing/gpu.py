"""保留发布特征和树模型数值规则的可选 GPU 后端。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from traceformer.gpu.runtime import launch, require_cuda, upload

from .features import CONTEXT_DIM, EVENT_FEATURE_DIM, SCALES


@dataclass(frozen=True)
class PackedForest:
    arrays: tuple[np.ndarray, ...]
    sigmoid: float
    feature_count: int


def pack_forest(model: dict) -> PackedForest:
    """导出 double 阈值和叶值；不改变分裂、叶值或树顺序。"""
    objective = model.get("objective", "")
    if not objective.startswith("binary") or model.get("num_tree_per_iteration", 1) != 1:
        raise ValueError("CUDA event trees require a binary objective")
    if model.get("average_output", False):
        raise ValueError("CUDA event trees do not support average_output")
    if model.get("linear_tree", False):
        raise ValueError("CUDA event trees do not support linear leaves")
    sigmoid = 1.0
    for field in objective.split():
        if field.startswith("sigmoid:"):
            sigmoid = float(field.split(":", 1)[1])
    if not np.isfinite(sigmoid) or sigmoid <= 0:
        raise ValueError("binary sigmoid must be finite and positive")
    roots, columns, children, thresholds, leaves, missing, defaults = [], [], [], [], [], [], []

    def visit(node):
        if "leaf_const" in node:
            raise ValueError("CUDA event trees do not support linear leaves")
        index = len(columns)
        columns.append(-1)
        children.append([-1, -1])
        thresholds.append(0.0)
        leaves.append(float(node.get("leaf_value", 0)))
        missing.append(0)
        defaults.append(0)
        if "split_feature" in node:
            if node["decision_type"] != "<=":
                raise ValueError("CUDA event trees require numerical <= splits")
            columns[index] = int(node["split_feature"])
            thresholds[index] = float(node["threshold"])
            missing[index] = {"None": 0, "Zero": 1, "NaN": 2}[node["missing_type"]]
            defaults[index] = int(node["default_left"])
            children[index] = [visit(node["left_child"]), visit(node["right_child"])]
        return index

    for tree in model["tree_info"]:
        roots.append(visit(tree["tree_structure"]))
    return PackedForest((np.asarray(roots, np.int64), np.asarray(columns, np.int32),
        np.asarray(children, np.int32).reshape(-1, 2), np.asarray(thresholds, np.float64),
        np.asarray(leaves, np.float64), np.asarray(missing, np.uint8), np.asarray(defaults, np.uint8)),
        sigmoid, int(model["max_feature_idx"]) + 1)


class CudaEventModel:
    """只读加载现有 LightGBM；CPU 句柄保留用于审计，不参与 GPU 预测。"""

    def __init__(self, booster, *, device: str | torch.device = "cuda"):
        self.device, _ = require_cuda(device)
        self.cpu_model = booster
        packed = pack_forest(booster.dump_model())
        self.arrays = tuple(upload(array, self.device) for array in packed.arrays)
        self.sigmoid, self.feature_count = packed.sigmoid, packed.feature_count

    def dump_model(self):
        return self.cpu_model.dump_model()

    def num_trees(self):
        return len(self.arrays[0])

    def predict_tensor(self, features: np.ndarray | torch.Tensor) -> torch.Tensor:
        from traceformer.gpu.kernels import forest_predict

        if isinstance(features, np.ndarray):
            dtype = np.float64 if features.dtype == np.float64 else np.float32
            features = upload(features.astype(dtype, copy=False), self.device)
        if features.device != self.arrays[0].device or features.dtype not in {torch.float32, torch.float64}:
            raise ValueError("CUDA tree features must be float32/float64 on the tree device")
        if features.ndim != 2 or features.shape[1] != self.feature_count:
            raise ValueError("CUDA tree feature count differs")
        output = torch.empty(len(features), dtype=torch.float64, device=features.device)
        launch(forest_predict, len(features), features.device, features, *self.arrays, self.sigmoid, output)
        return output

    def predict(self, features, *, num_threads: int = 1):
        return self.predict_tensor(features).cpu().numpy()


class CudaEventFeatures:
    """每个序列只构造一次上下文，分块展开 64 维父 token。"""

    def __init__(self, cache, *, device: str | torch.device = "cuda"):
        self.device, _ = require_cuda(device)
        # 先复用缓存契约检查，避免无效映射触发 CUDA 越界。
        cache.validate(allow_labels=cache.labels is not None)
        locations = np.asarray(cache.locations, dtype=np.int64)
        probability = np.asarray(cache.base_probability, dtype=np.float32)
        if not len(locations):
            raise ValueError("GPU context requires non-empty locations")
        if len(locations) > 2**24:
            raise ValueError("GPU context supports at most 2**24 events to preserve float32 count semantics")
        if (locations[:, :2] < 0).any() or (locations[:, 0] >= 346).any() or (locations[:, 1] >= 260).any() or (locations[:, 2] < 0).any():
            raise ValueError("GPU feature locations must be inside the 346x260 sensor with nonnegative time")
        self.locations = upload(locations, self.device)
        self.probability = upload(probability, self.device)
        self.tokens = upload(cache.token_features.astype(np.float32, copy=False), self.device)
        self.indices = upload(cache.event_token_indices.astype(np.int64, copy=False), self.device)
        self.raw = upload(cache.raw_features.astype(np.float32, copy=False), self.device)
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        frames = locations[:, 2] // 50
        frame_max = max(int(locations[:, 2].max() // 50), 1)
        # 非线性函数沿用 NumPy 的舍入，GPU 负责分组、统计及特征展开。
        self.base = upload(np.column_stack((clipped, np.sqrt(clipped),
            np.log(clipped / (1 - clipped)))).astype(np.float32), self.device)
        self.normalized = upload(np.column_stack((locations[:, 0] / 346.0,
            locations[:, 1] / 260.0, frames / frame_max)).astype(np.float32), self.device)
        integers = np.arange(len(locations) + 1, dtype=np.int64)
        self.log_integer = upload(np.log1p(integers).astype(np.float32), self.device)
        self.log_float = upload(np.log1p(integers.astype(np.float32)), self.device)
        self.context = self._context(int(locations[:, 2].max()))

    def _context(self, maximum_time):
        from traceformer.gpu.kernels import (
            group_statistics,
            histogram,
            integer_histogram,
            trajectory_context,
        )

        device, locations, probability = self.locations.device, self.locations, self.probability
        count = len(locations)
        context = torch.empty((count, CONTEXT_DIM), dtype=torch.float32, device=device)
        for index, scale in enumerate(SCALES):
            sx, sy, st = scale
            cells = locations // torch.tensor(scale, dtype=torch.int64, device=device)
            ny, nt = (260 + sy - 1) // sy, maximum_time // st + 1
            codes = (cells[:, 0] * ny + cells[:, 1]) * nt + cells[:, 2]
            _, inverse, counts = torch.unique(codes, sorted=True, return_inverse=True, return_counts=True)
            order = torch.argsort(inverse, stable=True)
            offsets = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0)))
            means = torch.empty(len(counts), dtype=torch.float64, device=device)
            maxima = torch.empty(len(counts), dtype=torch.float32, device=device)
            launch(group_statistics, len(counts), device, order, offsets, probability, means, maxima)
            context[:, index * 3] = self.log_integer[counts[inverse]]
            context[:, index * 3 + 1] = means[inverse].float()
            context[:, index * 3 + 2] = maxima[inverse]
        frame_count = maximum_time // 50 + 1
        frame_ids = locations[:, 2] // 50
        for index, size in enumerate((2, 4, 8)):
            nx, ny = (346 + size - 1) // size, (260 + size - 1) // size
            flat = (frame_ids * ny + locations[:, 1] // size) * nx + locations[:, 0] // size
            counts = torch.zeros(frame_count * ny * nx, dtype=torch.int64, device=device)
            maxima = torch.zeros(len(counts), dtype=torch.float32, device=device)
            launch(histogram, count, device, flat, probability, counts, maxima)
            launch(trajectory_context, count, device, locations, size, nx, ny, frame_count,
                   counts, maxima, self.log_float, context, len(SCALES) * 3 + index * 6)
        pixels = locations[:, 1] * 346 + locations[:, 0]
        pixel_counts = torch.zeros(346 * 260, dtype=torch.int64, device=device)
        active_counts = torch.zeros_like(pixel_counts)
        active_pixels = torch.unique(pixels * frame_count + frame_ids) // frame_count
        launch(integer_histogram, count, device, pixels, pixel_counts)
        launch(integer_histogram, len(active_pixels), device, active_pixels, active_counts)
        context[:, -2] = self.log_integer[pixel_counts[pixels]]
        context[:, -1] = self.log_float[active_counts[pixels]]
        return context

    def rows(self, start: int, stop: int) -> torch.Tensor:
        result = torch.cat((self.base[start:stop], self.tokens[self.indices[start:stop]],
            self.raw[start:stop], self.normalized[start:stop], self.context[start:stop]), dim=1)
        if result.shape[1] != EVENT_FEATURE_DIM:
            raise RuntimeError("GPU event feature schema differs")
        return result
