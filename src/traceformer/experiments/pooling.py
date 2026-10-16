"""仅用于消融的统计编码与均值池化，不改变事件映射和解码器输入。"""

from __future__ import annotations

import copy

import torch
from torch import nn

from traceformer.models.event_tokenizer import EventTokenizer


class PoolingTokenizer(EventTokenizer):
    @classmethod
    def from_attention(cls, source: EventTokenizer, mode: str) -> PoolingTokenizer:
        if mode not in {"statistics", "mean"}:
            raise ValueError("unsupported pooling replacement")
        # 复制公共权重，新增层不消耗外部 RNG，保证配对主干初始化。
        result = copy.deepcopy(source)
        result.__class__ = cls
        result.pooling_mode = mode
        dim = source.event_embed_dim
        # 均值分支使用实际参与计算的宽 MLP，使其参数量接近四查询注意力。
        hidden = dim if mode == "statistics" else 3 * dim
        with torch.random.fork_rng(devices=[]):
            result.block_encoder = nn.Sequential(
                nn.Linear(source.statistic_dim if mode == "statistics" else dim, hidden),
                nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, dim),
            )
        # 删除不参与当前方案的参数，参数统计只包含实际网络。
        del result.pool_queries, result.key_projection, result.query_fusion
        if mode == "statistics":
            del result.event_encoder, result.value_projection
        return result

    def _encode_block_features(self, local_features, inverse, counts, statistics):
        if self.pooling_mode == "statistics":
            return self.block_encoder(statistics)
        values = self.value_projection(self.event_encoder(local_features))
        pooled = values.new_zeros((len(counts), self.event_embed_dim))
        pooled.index_add_(0, inverse, values)
        pooled = pooled / counts[:, None].to(values.dtype)
        return self.block_encoder(pooled)
