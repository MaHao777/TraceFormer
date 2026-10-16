"""沿用 SciPy 树布局，将查询和限额合并放到 GPU。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from traceformer.gpu.runtime import launch, require_cuda, upload


def pack_geometry(geometry) -> tuple[np.ndarray, ...]:
    """保留原树叶序；不重新排序坐标或等距离点。"""
    times, offsets, roots, sizes, boxes, nodes, splits, permutation = [], [0], [], [], [], [], [], []

    def visit(node, point_offset):
        index = len(nodes)
        nodes.append([node.split_dim, -1, -1, node.start_idx + point_offset, node.end_idx + point_offset])
        splits.append(node.split)
        if node.split_dim >= 0:
            nodes[index][1] = visit(node.lesser, point_offset)
            nodes[index][2] = visit(node.greater, point_offset)
        return index

    for sample in geometry.samples:
        for time in sorted(sample.by_time):
            tree = sample.trees[time]
            times.append(time)
            point_offset = len(permutation)
            permutation.extend(sample.by_time[time][tree.indices])
            roots.append(visit(tree.tree, point_offset))
            sizes.append(tree.n)
            boxes.append([*tree.mins, *tree.maxes])
        offsets.append(len(times))
    sample_ids = np.searchsorted(np.unique(geometry.batch_indices), geometry.batch_indices)
    return (
        geometry.coordinates, sample_ids.astype(np.int64), np.asarray(times, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64), np.asarray(roots, dtype=np.int64),
        np.asarray(sizes, dtype=np.int64), np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        np.asarray(nodes, dtype=np.int64).reshape(-1, 5), np.asarray(splits, dtype=np.float64),
        np.asarray(permutation, dtype=np.int64),
    )


@dataclass(frozen=True)
class CudaConeGeometry:
    tensors: tuple[torch.Tensor, ...]
    device: torch.device


def upload_geometry(geometry, device: torch.device) -> CudaConeGeometry:
    require_cuda(device)
    return CudaConeGeometry(tuple(upload(array, device) for array in pack_geometry(geometry)), device)


def query_geometry(builder, geometry: CudaConeGeometry):
    from traceformer.gpu.kernels import cone_kernel

    device = geometry.device
    count = len(geometry.tensors[0])
    heads = builder.cone_heads
    limit = builder.max_neighbors_per_cone
    output = torch.full((count, len(heads), 2, limit), -1, dtype=torch.long, device=device)
    overflow = torch.zeros(count * len(heads) * 2, dtype=torch.uint8, device=device)
    radii = torch.tensor([head.base_radius for head in heads], dtype=torch.float64, device=device)
    slopes = torch.tensor([head.slope for head in heads], dtype=torch.float64, device=device)
    horizons = torch.tensor([max(1, math.ceil(head.max_time_distance * builder.layer_time_scale))
                             for head in heads], dtype=torch.int64, device=device)
    launch(cone_kernel(limit), len(overflow), device, *geometry.tensors, radii, slopes, horizons,
           output, overflow)
    if bool(overflow.any().item()):
        raise RuntimeError("CUDA KD query queue exceeded 128 entries; use --neighbor-backend cpu")
    return output[:, :, 0].contiguous(), output[:, :, 1].contiguous()


class _ArchivedGpuBuilder:
    """归档 Attention 的薄 adapter；本次 encoder 内共享，上下文退出即清除。"""

    def __init__(self, original):
        from .cone_neighbor_builder import ConeNeighborBuilder

        self.original = original
        self.builder = ConeNeighborBuilder(cone_heads=original.cone_heads,
            max_neighbors_per_cone=original.max_neighbors_per_cone,
            layer_time_scale=original.layer_time_scale, backend="cuda")
        self.causal_mode = getattr(original, "causal_mode", False)
        self.geometry = None

    def __call__(self, coordinates, batches):
        result = self.builder(coordinates, batches, geometry=self.geometry)
        if self.causal_mode:
            from .cone_neighbor_builder import ConeNeighborhoods

            return ConeNeighborhoods(result.past_indices, torch.full_like(result.future_indices, -1))
        return result


def configure_neighbor_backend(model: Any, backend: str = "cpu") -> None:
    """后端是运行时选择，不写入模型配置、权重或历史源码。"""
    from .cone_neighbor_builder import ConeNeighborBuilder, prepare_cone_geometry

    if backend not in {"cpu", "cuda"}:
        raise ValueError("neighbor backend must be cpu or cuda")
    if backend == "cuda":
        require_cuda(next(model.parameters()).device)
    builders = []
    encoder = getattr(model, "encoder", None)
    for layer in getattr(encoder, "layers", ()):
        attention = layer.attention
        original = attention.neighbor_builder
        if isinstance(original, ConeNeighborBuilder):
            original.backend = backend
        elif isinstance(original, _ArchivedGpuBuilder):
            if backend == "cpu":
                attention.neighbor_builder = original.original
            else:
                builders.append(original)
        elif backend == "cuda":
            if not all(hasattr(original, name) for name in ("cone_heads", "max_neighbors_per_cone", "layer_time_scale")):
                raise ValueError("CUDA neighbor backend requires cone neighborhoods")
            attention.neighbor_builder = _ArchivedGpuBuilder(original)
            builders.append(attention.neighbor_builder)
    if backend == "cpu" and hasattr(encoder, "_gpu_original_forward"):
        encoder.forward = encoder._gpu_original_forward
        del encoder._gpu_original_forward
    if not builders:
        return
    original_forward = getattr(encoder, "_gpu_original_forward", encoder.forward)
    encoder._gpu_original_forward = original_forward

    def forward(tokens, coordinates, batches):
        geometry = prepare_cone_geometry(coordinates, batches, backend="cuda")
        for builder in builders:
            builder.geometry = geometry
        try:
            return original_forward(tokens, coordinates, batches)
        finally:
            for builder in builders:
                builder.geometry = None

    encoder.forward = forward
