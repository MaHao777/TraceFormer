"""CUDA 与 CUDA simulator 共用的查询、树预测和整数统计内核。"""

from __future__ import annotations

import math

from numba import cuda, float64


@cuda.jit
def forest_predict(features, roots, columns, children, thresholds, leaves, missing, defaults,
                   sigmoid, output):
    row = cuda.grid(1)
    if row >= features.shape[0]:
        return
    score = 0.0
    # 每个事件按发布模型的树顺序累计 double，不引入并行归约误差。
    for tree in range(roots.size):
        node = roots[tree]
        while columns[node] >= 0:
            value = float64(features[row, columns[node]])
            kind = missing[node]
            if math.isnan(value) and kind != 2:
                value = 0.0
            absent = (kind == 1 and abs(value) <= 1e-35) or (kind == 2 and math.isnan(value))
            go_left = defaults[node] != 0 if absent else value <= thresholds[node]
            node = children[node, 0 if go_left else 1]
        score += leaves[node]
    exponent = -sigmoid * score
    output[row] = 0.0 if exponent > 709.782712893384 else 1.0 / (1.0 + math.exp(exponent))


@cuda.jit
def group_statistics(order, offsets, probability, means, maxima):
    group = cuda.grid(1)
    if group >= means.size:
        return
    total, maximum = 0.0, 0.0
    for index in range(offsets[group], offsets[group + 1]):
        value = float64(probability[order[index]])
        total += value
        maximum = max(maximum, value)
    means[group] = total / (offsets[group + 1] - offsets[group])
    maxima[group] = maximum


@cuda.jit
def histogram(indices, probability, counts, maxima):
    index = cuda.grid(1)
    if index < indices.size:
        cell = indices[index]
        cuda.atomic.add(counts, cell, 1)
        cuda.atomic.max(maxima, cell, probability[index])


@cuda.jit
def integer_histogram(indices, counts):
    index = cuda.grid(1)
    if index < indices.size:
        cuda.atomic.add(counts, indices[index], 1)


@cuda.jit
def trajectory_context(locations, cell_size, nx, ny, frames, counts, maxima, log_counts,
                       output, column):
    row = cuda.grid(1)
    if row >= locations.shape[0]:
        return
    x, y, time = locations[row, 0] // cell_size, locations[row, 1] // cell_size, locations[row, 2] // 50
    for phase in range(3):
        frame = time + (0 if phase == 0 else (-1 if phase == 1 else 1))
        total, maximum = 0, 0.0
        if 0 <= frame < frames:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < nx and 0 <= yy < ny:
                        cell = (frame * ny + yy) * nx + xx
                        total += counts[cell]
                        maximum = max(maximum, maxima[cell])
        output[row, column + phase] = log_counts[total]
        output[row, column + 3 + phase] = maximum
