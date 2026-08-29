"""事件上下文和LightGBM输入特征。"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .contracts import EventCache

SCALES = (
    (1, 1, 50),
    (2, 2, 10),
    (3, 3, 20),
    (4, 4, 50),
    (6, 6, 100),
    (10, 10, 200),
)
CONTEXT_DIM = len(SCALES) * 3 + 3 * 6 + 2
EVENT_FEATURE_DIM = 3 + 64 + 4 + 3 + CONTEXT_DIM


def _grouped_features(
    locations: np.ndarray, probability: np.ndarray, scale: tuple[int, int, int]
) -> np.ndarray:
    cells = locations // np.asarray(scale, dtype=np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=probability)
    maxima = np.zeros(len(counts), dtype=np.float32)
    np.maximum.at(maxima, inverse, probability)
    return np.column_stack(
        (
            np.log1p(counts[inverse]),
            sums[inverse] / counts[inverse],
            maxima[inverse],
        )
    ).astype(np.float32)


def _trajectory_features(
    locations: np.ndarray, probability: np.ndarray, cell_size: int
) -> np.ndarray:
    x_cells = (346 + cell_size - 1) // cell_size
    y_cells = (260 + cell_size - 1) // cell_size
    frames = (locations[:, 2] // 50).astype(np.int64)
    x_cell = locations[:, 0] // cell_size
    y_cell = locations[:, 1] // cell_size
    frame_count = int(frames.max()) + 1
    flat = (frames * y_cells + y_cell) * x_cells + x_cell
    shape = (frame_count, y_cells, x_cells)
    counts = np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape)
    counts = counts.astype(np.float32)
    maxima = np.zeros(int(np.prod(shape)), dtype=np.float32)
    np.maximum.at(maxima, flat, probability)
    maxima = maxima.reshape(shape)

    spatial_counts = ndimage.convolve(counts, np.ones((1, 3, 3), np.float32), mode="constant")
    spatial_maxima = ndimage.maximum_filter(maxima, size=(1, 3, 3), mode="constant")
    previous_counts = np.zeros_like(spatial_counts)
    next_counts = np.zeros_like(spatial_counts)
    previous_maxima = np.zeros_like(spatial_maxima)
    next_maxima = np.zeros_like(spatial_maxima)
    previous_counts[1:] = spatial_counts[:-1]
    next_counts[:-1] = spatial_counts[1:]
    previous_maxima[1:] = spatial_maxima[:-1]
    next_maxima[:-1] = spatial_maxima[1:]
    index = (frames, y_cell, x_cell)
    return np.column_stack(
        (
            np.log1p(spatial_counts[index]),
            np.log1p(previous_counts[index]),
            np.log1p(next_counts[index]),
            spatial_maxima[index],
            previous_maxima[index],
            next_maxima[index],
        )
    ).astype(np.float32)


def make_context(locations: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """仅由无标签事件和当前模型概率构造时空上下文。"""

    locations = np.asarray(locations, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float32)
    if locations.shape != (len(probability), 3) or not len(locations):
        raise ValueError("locations and probability must align and be non-empty")
    parts = [_grouped_features(locations, probability, scale) for scale in SCALES]
    parts.extend(_trajectory_features(locations, probability, cell_size) for cell_size in (2, 4, 8))
    pixels = locations[:, 1] * 346 + locations[:, 0]
    frames = locations[:, 2] // 50
    _, pixel_inverse, pixel_counts = np.unique(pixels, return_inverse=True, return_counts=True)
    pairs = np.unique(np.column_stack((pixels, frames)), axis=0)
    active_pixels, active_counts = np.unique(pairs[:, 0], return_counts=True)
    lookup = np.zeros(260 * 346, dtype=np.float32)
    lookup[active_pixels] = active_counts
    parts.append(
        np.column_stack(
            (
                np.log1p(pixel_counts[pixel_inverse]),
                np.log1p(lookup[pixels]),
            )
        )
    )
    result = np.concatenate(parts, axis=1).astype(np.float32)
    if result.shape[1] != CONTEXT_DIM:
        raise RuntimeError("unexpected context feature count")
    return result


def build_event_features(
    cache: EventCache,
    *,
    indices: np.ndarray | None = None,
    context: np.ndarray | None = None,
) -> np.ndarray:
    """按需展开父token，避免把64维表示重复写入磁盘。"""

    rows = (
        np.arange(len(cache.locations), dtype=np.int64)
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )
    if context is None:
        context = make_context(cache.locations, cache.base_probability)
    if context.shape != (len(cache.locations), CONTEXT_DIM):
        raise ValueError("context does not align with the cache")
    probability = np.clip(cache.base_probability[rows], 1e-6, 1 - 1e-6)
    locations = cache.locations[rows]
    frames = locations[:, 2] // 50
    frame_count = max(int(cache.locations[:, 2].max() // 50), 1)
    features = np.column_stack(
        (
            probability,
            np.sqrt(probability),
            np.log(probability / (1 - probability)),
            cache.token_features[cache.event_token_indices[rows]],
            cache.raw_features[rows],
            locations[:, 0] / 346.0,
            locations[:, 1] / 260.0,
            frames / frame_count,
            context[rows],
        )
    ).astype(np.float32)
    if features.shape[1] != EVENT_FEATURE_DIM:
        raise RuntimeError("unexpected event feature count")
    return features


def event_feature_names() -> tuple[str, ...]:
    names = ["base", "sqrt_base", "base_logit"]
    names.extend(f"token_{index}" for index in range(64))
    names.extend(f"raw_{index}" for index in range(4))
    names.extend(("x", "y", "time"))
    names.extend(f"context_{index}" for index in range(CONTEXT_DIM))
    return tuple(names)
