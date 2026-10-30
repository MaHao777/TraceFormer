"""CPU完整局部注意力：融合邻居访问与梯度累加，避免逐查询复制K/V。"""

from __future__ import annotations

import numpy as np
import torch
from numba import config, get_num_threads, njit, prange, set_num_threads
from torch.nn import functional as F


@njit(cache=True)
def _neighbors(center, lookup, shape, offsets):
    times, height, width = shape
    indices = np.empty(len(offsets), dtype=np.int64)
    for slot in range(len(offsets)):
        x, y, t = center[0] + offsets[slot, 0], center[1] + offsets[slot, 1], center[2] + offsets[slot, 2]
        indices[slot] = (lookup[(t * height + y) * width + x]
                         if 0 <= x < width and 0 <= y < height and 0 <= t < times else -1)
    return indices


@njit(cache=True, parallel=True, fastmath={"reassoc", "contract"})
def _forward(q, k, v, ek, ev, coords, lookup, shape, offsets, mask, dropout):
    count, heads, dimension = q.shape
    scale = np.float32(1 / np.sqrt(dimension))
    keep_scale = np.float32(1 / (1 - dropout))
    output = np.zeros(q.shape, dtype=np.float32)
    maximum = np.zeros((count, heads), dtype=np.float32)
    inverse = np.zeros((count, heads), dtype=np.float32)
    for row in prange(count):
        neighbors = _neighbors(coords[row], lookup, shape, offsets)
        for head in range(heads):
            scores = np.empty(len(offsets), dtype=np.float32)
            peak = np.float32(-np.inf)
            for slot in range(len(offsets)):
                neighbor = neighbors[slot]
                score = np.float32(-np.inf)
                if neighbor >= 0:
                    score = np.float32(0)
                    side = offsets[slot, 3]
                    for channel in range(dimension):
                        score += q[row, head, channel] * (k[neighbor, head, channel] + ek[side, head, channel])
                    score *= scale
                scores[slot] = score
                peak = max(peak, score)
            if peak == np.float32(-np.inf):
                continue
            total = np.float32(0)
            for slot in range(len(offsets)):
                scores[slot] = np.exp(scores[slot] - peak)
                total += scores[slot]
            maximum[row, head], inverse[row, head] = peak, np.float32(1) / total
            for slot in range(len(offsets)):
                neighbor = neighbors[slot]
                if neighbor < 0:
                    continue
                probability = scores[slot] / total
                if dropout > 0:
                    probability *= mask[row, head, slot] * keep_scale
                side = offsets[slot, 3]
                for channel in range(dimension):
                    output[row, head, channel] += probability * (v[neighbor, head, channel] + ev[side, head, channel])
    return output, maximum, inverse


@njit(cache=True, parallel=True, fastmath={"reassoc", "contract"})
def _backward(q, k, v, ek, ev, output, maximum, inverse, coords, lookup, shape,
              offsets, mask, dropout, grad_output, parts):
    count, heads, dimension = q.shape
    scale = np.float32(1 / np.sqrt(dimension))
    keep_scale = np.float32(1 / (1 - dropout))
    grad_q = np.zeros(q.shape, dtype=np.float32)
    # 每分区仅分配一次全局梯度；避免每个128查询块都创建完整N×C梯度。
    grad_k = np.zeros((parts, count, heads, dimension), dtype=np.float32)
    grad_v = np.zeros((parts, count, heads, dimension), dtype=np.float32)
    grad_ek = np.zeros((parts, 2, heads, dimension), dtype=np.float32)
    grad_ev = np.zeros((parts, 2, heads, dimension), dtype=np.float32)
    for part in prange(parts):
        for row in range(count * part // parts, count * (part + 1) // parts):
            neighbors = _neighbors(coords[row], lookup, shape, offsets)
            for head in range(heads):
                if inverse[row, head] == 0:
                    continue
                delta = np.float32(0)
                for channel in range(dimension):
                    delta += grad_output[row, head, channel] * output[row, head, channel]
                for slot in range(len(offsets)):
                    neighbor = neighbors[slot]
                    if neighbor < 0:
                        continue
                    side = offsets[slot, 3]
                    score, value_gradient = np.float32(0), np.float32(0)
                    for channel in range(dimension):
                        score += q[row, head, channel] * (k[neighbor, head, channel] + ek[side, head, channel])
                        value_gradient += grad_output[row, head, channel] * (v[neighbor, head, channel] + ev[side, head, channel])
                    probability = np.exp(score * scale - maximum[row, head]) * inverse[row, head]
                    drop_scale = mask[row, head, slot] * keep_scale if dropout > 0 else np.float32(1)
                    score_gradient = probability * (value_gradient * drop_scale - delta) * scale
                    value_scale = probability * drop_scale
                    for channel in range(dimension):
                        grad_q[row, head, channel] += score_gradient * (k[neighbor, head, channel] + ek[side, head, channel])
                        key_gradient = score_gradient * q[row, head, channel]
                        value_result = value_scale * grad_output[row, head, channel]
                        grad_k[part, neighbor, head, channel] += key_gradient
                        grad_v[part, neighbor, head, channel] += value_result
                        grad_ek[part, side, head, channel] += key_gradient
                        grad_ev[part, side, head, channel] += value_result
    return grad_q, grad_k, grad_v, grad_ek, grad_ev


def _numpy(tensor):
    return tensor.detach().contiguous().numpy()


class _LocalCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ek, ev, coords, lookup, shape, offsets, heads, dropout, chunk):
        set_num_threads(min(torch.get_num_threads(), config.NUMBA_NUM_THREADS))
        count, channels = q.shape
        head_dim = channels // heads
        mask = (torch.empty((count, heads, len(offsets)), dtype=torch.uint8)
                if dropout else torch.empty((0, 0, 0), dtype=torch.uint8))
        if dropout:
            # 沿用原128查询分块和Torch dropout的RNG调用顺序。
            for start in range(0, count, chunk):
                end = min(start + chunk, count)
                keep = F.dropout(torch.ones((end - start, heads, 1, len(offsets))), dropout, True)
                mask[start:end] = keep[:, :, 0].ne(0).to(torch.uint8)
        arrays = [_numpy(t.reshape(-1, heads, head_dim)) for t in (q, k, v, ek, ev)]
        result, maximum, inverse = _forward(*arrays, _numpy(coords), _numpy(lookup), shape,
                                             _numpy(offsets), _numpy(mask), dropout)
        output = torch.from_numpy(result)
        ctx.save_for_backward(q, k, v, ek, ev, output, torch.from_numpy(maximum),
                              torch.from_numpy(inverse), coords, lookup, offsets, mask)
        ctx.shape, ctx.heads, ctx.dropout = shape, heads, dropout
        return output.reshape(count, channels), torch.from_numpy(inverse.any(axis=1))

    @staticmethod
    def backward(ctx, grad_output, _grad_available):
        q, k, v, ek, ev, output, maximum, inverse, coords, lookup, offsets, mask = ctx.saved_tensors
        set_num_threads(min(torch.get_num_threads(), config.NUMBA_NUM_THREADS))
        heads, dimension = ctx.heads, q.shape[-1] // ctx.heads
        arrays = [_numpy(t.reshape(-1, heads, dimension)) for t in (q, k, v, ek, ev)]
        gradients = _backward(*arrays, _numpy(output), _numpy(maximum), _numpy(inverse),
                              _numpy(coords), _numpy(lookup), ctx.shape, _numpy(offsets),
                              _numpy(mask), ctx.dropout, _numpy(grad_output.reshape(-1, heads, dimension)),
                              min(get_num_threads(), len(q)))
        result = [torch.from_numpy(gradients[0]).reshape_as(q)]
        for original, gradient in zip((k, v, ek, ev), gradients[1:], strict=True):
            result.append(torch.from_numpy(gradient).sum(0).reshape_as(original))
        return (*result, None, None, None, None, None, None, None)


def local_cpu_attention(q, k, v, ek, ev, coords, lookup, shape, offsets, heads, dropout, chunk):
    """保持同一邻域、投影、softmax和dropout，仅替换CPU执行方式。"""
    return _LocalCPU.apply(q, k, v, ek, ev, coords, lookup, shape, offsets, heads, dropout, chunk)
