"""CUDA 与 CUDA simulator 共用的查询、树预测和整数统计内核。"""

from __future__ import annotations

import math
from functools import lru_cache

from numba import cuda, float64, int64


@cuda.jit(device=True)
def _push(keys, values, size, key, value):
    position = size
    while position > 0:
        parent = (position - 1) // 2
        if not key < keys[parent]:
            break
        keys[position], values[position] = keys[parent], values[parent]
        position = parent
    keys[position], values[position] = key, value
    return size + 1


@cuda.jit(device=True)
def _pop(keys, values, size):
    # 相等优先级不交换，两个子节点相等时取左侧，与原 KD 查询一致。
    key, value = keys[0], values[0]
    size -= 1
    if size:
        last_key, last_value = keys[size], values[size]
        position = 0
        while position * 2 + 1 < size:
            child = position * 2 + 1
            if child + 1 < size and keys[child] > keys[child + 1]:
                child += 1
            if not last_key > keys[child]:
                break
            keys[position], values[position] = keys[child], values[child]
            position = child
        keys[position], values[position] = last_key, last_value
    return key, value, size


@lru_cache(maxsize=16)
def cone_kernel(limit: int):
    """只缓存编译后的代码；不缓存任何前向输入或几何结果。"""

    @cuda.jit
    def query(points, sample_ids, times, offsets, roots, sizes, boxes, nodes, splits,
              permutation, radii, slopes, horizons, output, overflow):
        job = cuda.grid(1)
        heads = radii.size
        if job >= points.shape[0] * heads * 2:
            return
        direction = job % 2
        head = (job // 2) % heads
        source = job // (2 * heads)
        sample = sample_ids[source]
        x, y, time = float64(points[source, 0]), float64(points[source, 1]), points[source, 2]
        queue_keys = cuda.local.array(128, float64)
        queue_values = cuda.local.array(128, int64)
        neighbor_keys = cuda.local.array(limit, float64)
        neighbor_values = cuda.local.array(limit, int64)
        selected_scores = cuda.local.array(limit, float64)
        selected_indices = cuda.local.array(limit, int64)
        rank_distances = cuda.local.array(limit, float64)
        rank_indices = cuda.local.array(limit, int64)
        for index in range(limit):
            selected_scores[index], selected_indices[index] = math.inf, -1
        for delta in range(1, horizons[head] + 1):
            target_time = time + (1 if direction else -1) * delta
            left, right = offsets[sample], offsets[sample + 1]
            end = right
            while left < right:
                middle = (left + right) // 2
                if times[middle] < target_time:
                    left = middle + 1
                else:
                    right = middle
            if left == end or times[left] != target_time:
                continue
            tree = left
            wanted = min(limit, sizes[tree])
            bound = radii[head] + slopes[head] * delta + 1e-9
            node = roots[tree]
            lower = max(0.0, boxes[tree, 0] - x, x - boxes[tree, 2],
                        boxes[tree, 1] - y, y - boxes[tree, 3])
            queued, found = 0, 0
            while True:
                dimension = nodes[node, 0]
                if dimension < 0:
                    for position in range(nodes[node, 3], nodes[node, 4]):
                        target = permutation[position]
                        distance = max(abs(float64(points[target, 0]) - x), abs(float64(points[target, 1]) - y))
                        if distance < bound:
                            if found == wanted:
                                _, _, found = _pop(neighbor_keys, neighbor_values, found)
                            found = _push(neighbor_keys, neighbor_values, found, -distance, target)
                            if found == wanted:
                                bound = -neighbor_keys[0]
                    if not queued:
                        break
                    lower, node, queued = _pop(queue_keys, queue_values, queued)
                else:
                    if lower > bound:
                        break
                    coordinate = x if dimension == 0 else y
                    split = splits[node]
                    if coordinate < split:
                        near, far = nodes[node, 1], nodes[node, 2]
                    else:
                        near, far = nodes[node, 2], nodes[node, 1]
                    far_lower = max(lower, abs(coordinate - split))
                    if far_lower <= bound:
                        if queued == 128:
                            overflow[job] = 1
                            return
                        queued = _push(queue_keys, queue_values, queued, far_lower, far)
                    node = near
            total = found
            for rank in range(total - 1, -1, -1):
                key, target, found = _pop(neighbor_keys, neighbor_values, found)
                rank_distances[rank], rank_indices[rank] = -key, target
            for rank in range(total):
                score = (rank + 1) * 1_000_000.0 + delta * 1_000.0 + rank_distances[rank]
                position = limit - 1
                if score >= selected_scores[position]:
                    continue
                while position > 0 and score < selected_scores[position - 1]:
                    selected_scores[position] = selected_scores[position - 1]
                    selected_indices[position] = selected_indices[position - 1]
                    position -= 1
                selected_scores[position], selected_indices[position] = score, rank_indices[rank]
        for index in range(limit):
            output[source, head, direction, index] = selected_indices[index]

    return query


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
