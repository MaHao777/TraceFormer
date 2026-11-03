"""固定小过去锥的门控局部注意力，不构建锥摘要或全局注意力矩阵。"""

from __future__ import annotations

import os
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from traceformer.config import ConeWindowConfig
from traceformer.models.cone_geometry import euclidean_offsets
from traceformer.models.cone_transformer import ConeAttentionOutput, ConeTransformerOutput
from traceformer.models.prefix_cone import prepare_prefix_geometry


class PastLocalAttention(nn.Module):
    """过去两块内的完整小锥；源 Token 门控作为注意力概率先验。"""

    def __init__(self, original: nn.Module, chunk_size: int | None = None, *,
                 metric: str = "chebyshev", windows: tuple[ConeWindowConfig, ...] | None = None) -> None:
        super().__init__()
        size = os.environ.get("TRACEFORMER_LOCAL_QUERY_CHUNK", "256") if chunk_size is None else chunk_size
        if isinstance(size, str) and size.isdecimal():
            size = int(size)
        if type(size) is not int or size <= 0:
            raise ValueError("local query chunk_size must be a positive integer")
        if original.pool_weight is None:
            raise ValueError("past local attention requires a learned source gate")
        self.chunk_size = size
        self.attention = original.center_attention
        self.pool_weight = original.pool_weight
        self.windows = (ConeWindowConfig(2, 2),) if windows is None else windows
        if not self.windows or any(w.spatial_radius < 0 or w.time_horizon < 1 for w in self.windows):
            raise ValueError("local cone windows must be nonempty with valid radii/horizons")
        self.directions = (-1,)
        self.fixed_radius = False
        if metric not in {"chebyshev", "euclidean"}:
            raise ValueError("unsupported local cone metric")
        self.metric = metric
        # 三锥并集先去重；原单小锥保持距离、行、列的枚举顺序。
        offsets = set()
        for window in self.windows:
            radius, horizon = window.spatial_radius, window.time_horizon
            if metric == "euclidean":
                offsets.update(euclidean_offsets(radius, horizon))
            else:
                offsets.update((x, y, -d) for d in range(1, horizon + 1)
                               for y in range(-(radius*d//horizon), radius*d//horizon + 1)
                               for x in range(-(radius*d//horizon), radius*d//horizon + 1))
        offsets = sorted(offsets, key=lambda xyz: (-xyz[2], xyz[1], xyz[0]))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)

    def _build_neighbors(self, sample) -> torch.Tensor:
        times, height, width = sample.shape
        lookup = torch.full((times * height * width,), -1, dtype=torch.long,
                            device=sample.coords.device)
        lookup[sample.flat_indices] = torch.arange(len(sample.indices), device=lookup.device)
        chunks = []
        for start in range(0, len(sample.indices), self.chunk_size):
            xyz = sample.coords[start:start + self.chunk_size, None] + self.offsets[None]
            inside = ((xyz >= 0) & (xyz < xyz.new_tensor((width, height, times)))).all(-1)
            flat = ((xyz[..., 2].clamp(0, times - 1) * height
                     + xyz[..., 1].clamp(0, height - 1)) * width
                    + xyz[..., 0].clamp(0, width - 1))
            neighbors = lookup[flat]
            chunks.append(torch.where(inside, neighbors, -1))
        return torch.cat(chunks)

    def _neighbor_indices(self, sample) -> torch.Tensor:
        # 本次前向的三个层共享整数索引；不缓存任何依赖网络权重的特征。
        key = ("past_local_neighbors", tuple((w.spatial_radius, w.time_horizon) for w in self.windows), self.metric)
        cache = sample.corner_cache
        if cache is not None and key in cache:
            return cache[key]
        neighbors = self._build_neighbors(sample)
        if cache is not None:
            cache[key] = neighbors
        return neighbors

    def _attend_chunk(self, q, k, v, log_gate, neighbors, start, end):
        selected = neighbors[start:end]
        valid = selected >= 0
        available = valid.any(-1)
        selected = selected.clamp_min(0)
        valid = valid.clone()
        valid[~available, 0] = True
        heads = self.attention.num_heads
        head_dim = q.shape[-1] // heads
        query = q[start:end].reshape(-1, heads, 1, head_dim)
        keys = k[selected].reshape(end - start, -1, heads, head_dim).transpose(1, 2)
        values = v[selected].reshape(end - start, -1, heads, head_dim).transpose(1, 2)
        # log(g) 加到 logits 后，softmax 自动归一化门控与内容相似度的乘积。
        bias = log_gate[selected].masked_fill(~valid, float("-inf"))
        result = F.scaled_dot_product_attention(
            query, keys, values, attn_mask=bias[:, None, None],
            dropout_p=self.attention.dropout if self.training else 0.0,
        ).reshape(end - start, -1)
        return torch.where(available[:, None], result, 0.0), available

    def forward(self, tokens, token_coords, token_batch_indices, *, geometry=None):
        if not len(tokens):
            return ConeAttentionOutput(tokens)
        if geometry is None:
            geometry = prepare_prefix_geometry(token_coords, token_batch_indices)
        # 每层仅投影一次全部 Q/K/V；逐块取邻居，避免保存 [N, K, D] 特征。
        attention = self.attention
        q, k, v = F.linear(tokens, attention.in_proj_weight, attention.in_proj_bias).chunk(3, -1)
        log_gate = (torch.sigmoid(self.pool_weight(tokens).float()) + 1e-4).log()
        log_gate = log_gate.flatten().to(q.dtype)
        features = torch.zeros_like(tokens)
        available = torch.zeros(len(tokens), dtype=torch.bool, device=tokens.device)
        for sample in geometry:
            indices = sample.indices
            neighbors = self._neighbor_indices(sample)
            args = (q[indices], k[indices], v[indices], log_gate[indices], neighbors)
            chunks, masks = [], []
            for start in range(0, len(indices), self.chunk_size):
                arguments = (*args, start, min(start + self.chunk_size, len(indices)))
                # 反向按块重算，并恢复该块 dropout 随机状态。
                result = (checkpoint(self._attend_chunk, *arguments, use_reentrant=False,
                                     preserve_rng_state=True)
                          if torch.is_grad_enabled() else self._attend_chunk(*arguments))
                chunks.append(result[0])
                masks.append(result[1])
            features[indices], available[indices] = torch.cat(chunks), torch.cat(masks)
        output = attention.out_proj(features)
        # 空邻域在输出投影后仍严格归零，防止输出偏置泄漏。
        return ConeAttentionOutput(torch.where(available[:, None], output, 0.0))


class PastLocalEncoder(nn.Module):
    """沿用原残差层，在本次前向内为训练和推理共享小锥邻居索引。"""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.layers = original.layers
        self.final_norm = original.final_norm

    def forward(self, tokens, token_coords, token_batch_indices):
        if not len(tokens):
            return ConeTransformerOutput(self.final_norm(tokens))
        geometry = prepare_prefix_geometry(token_coords, token_batch_indices)
        geometry = [sample if sample.corner_cache is not None else replace(sample, corner_cache={})
                    for sample in geometry]
        for layer in self.layers:
            tokens = layer(tokens, token_coords, token_batch_indices, geometry=geometry)
        return ConeTransformerOutput(self.final_norm(tokens))
