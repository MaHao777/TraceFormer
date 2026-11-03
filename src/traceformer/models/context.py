"""论文过去摘要交互与两个结构消融的公共模块。"""
from __future__ import annotations

import torch
from torch import nn

from traceformer.models.cone_transformer import ConeAttentionOutput, ConeTransformerOutput
from traceformer.models.event_tokenizer import EventTokenizer
from traceformer.models.prefix_cone import prefix_cone_pool, prepare_prefix_geometry


class MeanTokenizer(EventTokenizer):
    def _encode_block_features(self, local_features, inverse, counts, statistics):
        # M1直接平均事件编码，没有查询投影或额外补偿MLP。
        encoded = self.event_encoder(local_features)
        result = encoded.new_zeros((len(counts), self.event_embed_dim))
        result.index_add_(0, inverse, encoded)
        return result / counts[:, None].to(encoded.dtype)


class FFNBlock(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.norm = original.feed_forward_norm
        self.ffn = original.feed_forward
        self.dropout = original.residual_dropout

    def forward(self, tokens):
        return tokens + self.dropout(self.ffn(self.norm(tokens)))


class FFNEncoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.layers = nn.ModuleList(FFNBlock(layer) for layer in original.layers)
        self.final_norm = original.final_norm

    def forward(self, tokens, token_coords, token_batch_indices):
        # M3完全不构建邻域或读取其他Token。
        for layer in self.layers:
            tokens = layer(tokens)
        return ConeTransformerOutput(self.final_norm(tokens))


class SummaryAttention(nn.Module):
    def __init__(self, original, *, past_only=False, mean=False):
        super().__init__()
        self.windows = original.windows
        self.fixed_radius = original.fixed_radius
        self.directions = (-1,) if past_only else (-1, 1)
        self.pool_value, self.pool_weight = original.pool_value, original.pool_weight
        self.shape_embedding = original.shape_embedding
        self.direction_embedding = (nn.Parameter(original.direction_embedding[:1].detach().clone())
                                    if past_only else original.direction_embedding)
        self.mean = mean
        if mean:
            self.projection = nn.Linear(original.pool_value.out_features,
                                        original.pool_value.out_features)
        else:
            self.center_attention = original.center_attention

    def forward(self, tokens, token_coords, token_batch_indices, *, geometry=None):
        if not len(tokens):
            return ConeAttentionOutput(tokens)
        if geometry is None:
            geometry = prepare_prefix_geometry(token_coords, token_batch_indices)
        summaries, counts = prefix_cone_pool(
            self.pool_value(tokens), torch.sigmoid(self.pool_weight(tokens).float()) + 1e-4,
            geometry, self.windows, fixed_radius=self.fixed_radius, directions=self.directions,
            metric=getattr(self, "metric", "chebyshev"),
        )
        embeddings = (self.shape_embedding[:, None] + self.direction_embedding[None]).flatten(0, 1)
        summaries = summaries.to(tokens.dtype) + embeddings
        available = counts > 0
        valid = available.any(1)
        if self.mean:
            pooled = (summaries * available[..., None]).sum(1)
            pooled = pooled / available.sum(1, keepdim=True).clamp_min(1)
            output = self.projection(pooled)
        else:
            mask = ~available
            mask = mask.clone()
            mask[~valid, 0] = False
            # CUDA带Dropout的高效注意力限制批量维度；按独立中心分批，不截断任何邻居。
            chunk_size = 32768 if tokens.is_cuda else len(tokens)
            chunks = []
            for start in range(0, len(tokens), chunk_size):
                end = min(start + chunk_size, len(tokens))
                context = summaries[start:end]
                chunk, _ = self.center_attention(
                    tokens[start:end, None], context, context,
                    key_padding_mask=mask[start:end], need_weights=False,
                )
                chunks.append(chunk[:, 0])
            output = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        return ConeAttentionOutput(torch.where(valid[:, None], output, 0.0))
