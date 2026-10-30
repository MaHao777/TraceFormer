"""固定子格上的轻量空间卷积池化，保留事件与父 Token 的原映射。"""

from __future__ import annotations

import copy
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from traceformer.models.event_tokenizer import EventTokenizer

_GRID_SIZE = 4
_CELLS = _GRID_SIZE**2


def _token_chunk_size() -> int:
    value = os.environ.get("TRACEFORMER_SPATIAL_TOKEN_CHUNK", "4096")
    if not value.isdecimal() or int(value) < 1:
        raise ValueError("TRACEFORMER_SPATIAL_TOKEN_CHUNK must be a positive integer")
    return int(value)


def _cell_grid(values, cell_indices, token_indices, token_count):
    # 只为当前父 Token 分块分配子格；空格均值为零，不建立全序列体素。
    groups = token_indices * _CELLS + cell_indices
    counts = torch.bincount(groups, minlength=token_count * _CELLS)
    sums = values.new_zeros((token_count * _CELLS, values.shape[1]))
    sums.index_add_(0, groups, values)
    means = sums / counts.clamp_min(1)[:, None].to(values.dtype)
    grid = means.reshape(token_count, _GRID_SIZE, _GRID_SIZE, -1).permute(0, 3, 1, 2)
    return grid, counts.reshape(token_count, 1, _GRID_SIZE, _GRID_SIZE)


def weighted_grid_mean(grid: torch.Tensor, cell_counts: torch.Tensor) -> torch.Tensor:
    # 按原事件数回读，恒等子格处理即可严格恢复 M1 的事件均值定义。
    weights = cell_counts.to(grid.dtype)
    return (grid * weights).sum(dim=(-2, -1)) / weights.sum(dim=(-2, -1)).clamp_min(1)


class SpatialConvTokenizer(EventTokenizer):
    @classmethod
    def from_mean(cls, original: EventTokenizer, kernels: int) -> SpatialConvTokenizer:
        if type(kernels) is not int or kernels not in (1, 2):
            raise ValueError("spatial pooling kernels must be 1 or 2")
        if any(hasattr(original, name) for name in
               ("pool_queries", "key_projection", "value_projection", "query_fusion")):
            raise ValueError("spatial pooling requires an existing mean tokenizer")
        _token_chunk_size()
        result = copy.deepcopy(original)
        result.__class__ = cls
        parameter = next(original.parameters())
        device = parameter.device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] \
            if device.type == "cuda" else []
        # 新卷积独立初始化，公共参数与外部 CPU/CUDA 随机数状态均保持不变。
        with torch.random.fork_rng(devices=devices):
            result.spatial_kernels = nn.ModuleList(
                nn.Conv2d(original.event_embed_dim, original.event_embed_dim, 3,
                          padding=1, groups=original.event_embed_dim, bias=False,
                          device=device, dtype=parameter.dtype)
                for _ in range(kernels)
            )
        result.spatial_kernels.train(original.training)
        return result

    def _pool_chunk(self, values, cell_indices, token_indices, token_count):
        grid, cell_counts = _cell_grid(values, cell_indices, token_indices, token_count)
        response = F.gelu(self.spatial_kernels[0](grid))
        for kernel in self.spatial_kernels[1:]:
            response = response + F.gelu(kernel(grid))
        return weighted_grid_mean(response / len(self.spatial_kernels), cell_counts)

    def _encode_block_features(self, local_features, inverse, counts, statistics):
        token_count, chunk_size = len(counts), _token_chunk_size()
        if not token_count:
            return local_features.new_empty((0, self.event_embed_dim))
        # 事件只排序一次，随后用连续区间读取，避免每块重扫全部事件。
        order = torch.argsort(inverse, stable=True)
        ordered_local = local_features.index_select(0, order)
        encoded = self.event_encoder(ordered_local)
        token_indices = inverse.index_select(0, order)
        cells = torch.floor(ordered_local[:, :2] * _GRID_SIZE).long().clamp(0, _GRID_SIZE - 1)
        cell_indices = cells[:, 1] * _GRID_SIZE + cells[:, 0]
        token_edges = list(range(0, token_count, chunk_size)) + [token_count]
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        edge_indices = torch.tensor(token_edges, device=counts.device)
        event_edges = offsets.index_select(0, edge_indices).cpu().tolist()
        chunks = []
        for index, start in enumerate(token_edges[:-1]):
            end = token_edges[index + 1]
            first, last = event_edges[index:index + 2]
            arguments = (encoded[first:last], cell_indices[first:last],
                         token_indices[first:last] - start, end - start)
            if self.training and torch.is_grad_enabled():
                # 反向时重建小网格和卷积激活，不保存全序列的子格特征。
                pooled = checkpoint(self._pool_chunk, *arguments, use_reentrant=False,
                                    preserve_rng_state=False)
            else:
                pooled = self._pool_chunk(*arguments)
            chunks.append(pooled)
        return torch.cat(chunks)


__all__ = ["SpatialConvTokenizer", "weighted_grid_mean"]
