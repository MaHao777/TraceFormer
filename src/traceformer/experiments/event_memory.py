"""同像素严格时间记忆及独立神经解码实验，不改变正式模型入口。"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

try:
    from numba import njit
except ImportError:
    njit = None


def _strict_pixel_gaps(pixels, timestamps, polarity, order, pixel_count):
    gaps = np.full((len(order), 4), np.inf, dtype=np.float64)
    previous = np.full((pixel_count, 2), -np.inf, dtype=np.float64)
    first = 0
    while first < len(order):
        timestamp = timestamps[order[first]]
        stop = first + 1
        while stop < len(order) and timestamps[order[stop]] == timestamp:
            stop += 1
        # 同时间组先查询，再统一更新；自身和同时间事件均不会泄入记忆。
        for position in range(first, stop):
            index = order[position]
            pixel, sign = pixels[index], polarity[index]
            gaps[index, 0] = timestamp - previous[pixel, sign]
            gaps[index, 2] = timestamp - previous[pixel, 1 - sign]
        for position in range(first, stop):
            index = order[position]
            previous[pixels[index], polarity[index]] = timestamp
        first = stop

    following = np.full((pixel_count, 2), np.inf, dtype=np.float64)
    stop = len(order)
    while stop > 0:
        timestamp = timestamps[order[stop - 1]]
        first = stop - 1
        while first > 0 and timestamps[order[first - 1]] == timestamp:
            first -= 1
        for position in range(first, stop):
            index = order[position]
            pixel, sign = pixels[index], polarity[index]
            gaps[index, 1] = following[pixel, sign] - timestamp
            gaps[index, 3] = following[pixel, 1 - sign] - timestamp
        for position in range(first, stop):
            index = order[position]
            following[pixels[index], polarity[index]] = timestamp
        stop = first
    return gaps


if njit is not None:
    # 可选编译只加速确定性遍历；没有Numba时保留相同NumPy接口和数学定义。
    _strict_pixel_gaps = njit(cache=True)(_strict_pixel_gaps)


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


class SamePixelEventMemory:
    """单个序列的同/异极性×严格过去/未来间隔，输出顺序与事件行一致。"""

    decay_ms = (1.0, 10.0, 50.0)
    feature_dim = 12
    gap_names = ("same_past", "same_future", "opposite_past", "opposite_future")

    def __init__(self, *, width: int = 346, height: int = 260) -> None:
        if (not isinstance(width, (int, np.integer)) or isinstance(width, (bool, np.bool_))
                or not isinstance(height, (int, np.integer))
                or isinstance(height, (bool, np.bool_)) or width <= 0 or height <= 0):
            raise ValueError("width and height must be positive integers")
        self.width, self.height = int(width), int(height)

    def gaps(self, xy, timestamps_ms, polarity) -> np.ndarray:
        """接收NumPy或Torch原始字段，返回float64[N,4]；缺失间隔为inf。"""
        xy, timestamps, polarity = _numpy(xy), _numpy(timestamps_ms), _numpy(polarity)
        if timestamps.dtype != np.float64:
            raise TypeError("timestamps_ms must be float64 to retain original millisecond precision")
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("coordinates must have shape [N,2]")
        count = len(xy)
        if timestamps.shape != (count,) or polarity.shape != (count,):
            raise ValueError("timestamps and polarity must align with coordinates")
        if (not np.issubdtype(xy.dtype, np.number) or np.iscomplexobj(xy)
                or not np.all(np.isfinite(xy)) or np.any(xy != np.floor(xy))):
            raise ValueError("coordinates must contain finite integer pixels")
        if (np.any(xy < 0) or np.any(xy[:, 0] >= self.width)
                or np.any(xy[:, 1] >= self.height)):
            raise ValueError("coordinates are outside the configured sensor")
        if not np.all(np.isfinite(timestamps)) or np.any(timestamps < 0):
            raise ValueError("timestamps must be finite nonnegative milliseconds")
        if not np.all(np.isin(polarity, (0, 1))):
            raise ValueError("polarity must contain only zero or one")
        if count == 0:
            return np.empty((0, 4), dtype=np.float64)
        # 原始坐标可能为int16/uint16，必须先扩宽再计算展平索引。
        xy = xy.astype(np.int64, copy=False)
        pixels = xy[:, 1] * self.width + xy[:, 0]
        # 正常原始流已经有序；无序输入稳定排序后仍写回原始事件行。
        order = (np.arange(count, dtype=np.int64) if np.all(timestamps[1:] >= timestamps[:-1])
                 else np.argsort(timestamps, kind="stable"))
        return _strict_pixel_gaps(
            pixels, timestamps, polarity.astype(np.int64), order, self.width * self.height
        )

    def __call__(self, xy, timestamps_ms, polarity) -> np.ndarray:
        """返回float32[N,12]，每个间隔依次使用1/10/50ms指数基底。"""
        gaps = self.gaps(xy, timestamps_ms, polarity)
        # exp(-inf)=0，使缺失记忆不伪造任何邻居支持。
        features = np.exp(-gaps[:, :, None] / np.asarray(self.decay_ms))
        return features.reshape(len(gaps), self.feature_dim).astype(np.float32)


def _validate_memory_rows(memory, event_token_indices):
    if memory.ndim != 2 or not memory.is_floating_point():
        raise ValueError("memory must be a floating [N,D] tensor")
    if (event_token_indices.ndim != 1 or event_token_indices.dtype != torch.long
            or len(event_token_indices) != len(memory)):
        raise ValueError("event_token_indices must be int64 and align with memory")
    if event_token_indices.device != memory.device:
        raise ValueError("memory and token indices must share a device")
    if len(event_token_indices) and bool(torch.any(event_token_indices < 0)):
        raise ValueError("event token indices must be nonnegative")


def shuffle_memory_within_tokens(memory, event_token_indices, *, generator: torch.Generator):
    """按Token内联合行置换记忆，独立generator不推进训练的全局RNG。"""
    _validate_memory_rows(memory, event_token_indices)
    count = len(memory)
    if count == 0:
        return memory.clone()
    random_order = torch.randperm(count, generator=generator, device=generator.device).to(memory.device)
    shuffled_order = random_order[
        torch.argsort(event_token_indices[random_order], stable=True)
    ]
    original_order = torch.argsort(event_token_indices, stable=True)
    source_rows = torch.empty_like(event_token_indices)
    source_rows[original_order] = shuffled_order
    return memory[source_rows]


def mean_memory_within_tokens(memory, event_token_indices):
    """用同Token均值替换事件记忆，保留相同形状和可微聚合。"""
    _validate_memory_rows(memory, event_token_indices)
    if len(memory) == 0:
        return memory.clone()
    _, inverse = torch.unique(event_token_indices, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse).to(memory.dtype)
    sums = memory.new_zeros((len(counts), memory.shape[1]))
    sums.index_add_(0, inverse, memory)
    return (sums / counts[:, None])[inverse]


def _parent_events(token_features, event_token_indices, event_local_features, event_memory):
    _validate_memory_rows(event_memory, event_token_indices)
    if token_features.ndim != 2 or token_features.shape[1] != 64:
        raise ValueError("token_features must have shape [T,64]")
    count = len(event_token_indices)
    if event_local_features.shape != (count, 6) or event_memory.shape != (count, 12):
        raise ValueError("local features and memory must have shapes [N,6] and [N,12]")
    if (token_features.device != event_memory.device
            or event_local_features.device != event_memory.device):
        raise ValueError("all decoder inputs must share a device")
    if (token_features.dtype != event_memory.dtype
            or event_local_features.dtype != token_features.dtype):
        raise ValueError("all decoder feature tensors must share a floating dtype")
    if count and bool(torch.any(event_token_indices >= len(token_features))):
        raise ValueError("event token index is outside token_features")
    return token_features[event_token_indices]


def _classifier(input_dim, hidden_dim, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        nn.Dropout(dropout), nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1),
    )


class MemoryDecoder(nn.Module):
    """6758参数：用相同可学习层对照加性融合与Token条件化乘性融合。"""

    def __init__(self, *, token_dim: int = 64, dropout: float = 0.1,
                 fusion: str = "gated") -> None:
        super().__init__()
        if token_dim != 64:
            raise ValueError("matched memory decoder requires token_dim=64")
        if fusion not in {"additive", "gated"}:
            raise ValueError("fusion must be additive or gated")
        self.fusion = fusion
        self.token_dim = token_dim
        self.memory_encoder = nn.Sequential(nn.Linear(12, 16), nn.GELU())
        self.conditioner = nn.Sequential(nn.Linear(64, 16), nn.Sigmoid())
        self.classifier = _classifier(86, 45, dropout)

    def forward(self, token_features, event_token_indices, event_local_features, event_memory):
        parent = _parent_events(
            token_features, event_token_indices, event_local_features, event_memory
        )
        if len(parent) == 0:
            return token_features.new_empty((0,))
        memory = self.memory_encoder(event_memory)
        condition = self.conditioner(parent)
        # 两组使用完全相同的层和初值，只替换融合关系。
        fused = memory * condition if self.fusion == "gated" else memory + condition
        return self.classifier(torch.cat((parent, event_local_features, fused), dim=1)).squeeze(-1)


class MemoryConcatDecoder(nn.Module):
    """6734参数的强基线：将同一12维记忆直接拼入原解码输入。"""

    def __init__(self, *, token_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        if token_dim != 64:
            raise ValueError("matched memory decoder requires token_dim=64")
        self.token_dim = token_dim
        self.classifier = _classifier(82, 57, dropout)

    def forward(self, token_features, event_token_indices, event_local_features, event_memory):
        parent = _parent_events(
            token_features, event_token_indices, event_local_features, event_memory
        )
        if len(parent) == 0:
            return token_features.new_empty((0,))
        return self.classifier(
            torch.cat((parent, event_local_features, event_memory), dim=1)
        ).squeeze(-1)


__all__ = [
    "MemoryConcatDecoder", "MemoryDecoder", "SamePixelEventMemory",
    "mean_memory_within_tokens", "shuffle_memory_within_tokens",
]
