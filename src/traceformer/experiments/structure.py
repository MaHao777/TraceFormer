"""统计均值主导的单Query分块编码与固定过去锥六层编码器。"""

from __future__ import annotations

import torch
from torch import nn

from traceformer.models.cone_transformer import ConeTransformerBlock
from traceformer.models.event_tokenizer import EventTokenizer
from traceformer.models.prefix_cone import PrefixConeAttention


class MeanStatisticsQueryTokenizer(EventTokenizer):
    """统计量只注入一次，均值主支路与单Query固定按3:1融合。"""

    query_fraction = 0.25

    def _encode_block_features(self, local_features, inverse, counts, statistics):
        encoded = self.event_encoder(local_features)
        mean = encoded.new_zeros((len(counts), self.event_embed_dim))
        mean.index_add_(0, inverse, encoded)
        mean = mean / counts[:, None].to(encoded.dtype)
        primary = mean + self.statistics_projection(statistics)
        # Query读取该时空块全部事件；不截断、不采样。
        query = self._attention_pool(encoded, inverse, token_count=len(counts))[:, 0]
        return (1.0 - self.query_fraction) * primary + self.query_fraction * query

    def _project_token(self, block_features, statistics, encoded_position):
        return self.token_projection(torch.cat((block_features, encoded_position), dim=1))


def apply_structure_improvements(model, config, spec):
    # 全部变体先构建同一个Full，替换不推进训练RNG，公共层初值完全配对。
    if config.model.num_layers != 3 or config.model.num_pool_queries != 4:
        raise ValueError("structure experiment requires the original 3-layer/4-query template")
    with torch.random.fork_rng(devices=[]):
        shared_rng = torch.get_rng_state()
        if spec.variant[0] == "1":
            tokenizer = model.tokenizer
            tokenizer.__class__ = MeanStatisticsQueryTokenizer
            dim, stats = tokenizer.event_embed_dim, tokenizer.statistic_dim
            tokenizer.num_pool_queries = 1
            tokenizer.pool_queries = nn.Parameter(tokenizer.pool_queries[:1].detach().clone())
            tokenizer.statistics_projection = nn.Linear(stats, dim, bias=False)
            del tokenizer.query_fusion
            original = tokenizer.token_projection[0]
            projection = nn.Linear(2 * dim, original.out_features)
            columns = list(range(dim)) + list(range(dim + stats, 2 * dim + stats))
            with torch.no_grad():
                projection.weight.copy_(original.weight[:, columns])
                projection.bias.copy_(original.bias)
            tokenizer.token_projection[0] = projection
        if spec.variant[1] == "1":
            # 01和11的新增三层初值也必须相同。
            torch.set_rng_state(shared_rng)
            for layer in model.encoder.layers:
                attention = layer.attention
                attention.directions = (-1,)
                attention.pool_weight = None
                attention.direction_embedding = nn.Parameter(
                    attention.direction_embedding[:1].detach().clone())
            # 新增三层仍使用原三尺度，不随深度扩张几何窗口。
            for _ in range(3):
                model.encoder.layers.append(ConeTransformerBlock(
                    token_dim=config.model.token_dim, mlp_ratio=config.model.mlp_ratio,
                    dropout=config.model.dropout,
                    attention=PrefixConeAttention(
                        config.model.token_dim, config.model.num_heads,
                        tuple(config.model.cone_windows), config.model.dropout,
                        directions=(-1,), learned_pool_weights=False,
                    ),
                ))
            model.num_layers = 6
    model.ablation_spec = spec.to_dict()
    return model


def describe_structure(model):
    """保存实际结构，避免把共同模板的3层/4Query误读为替换后模型。"""
    return {
        "queries": model.tokenizer.num_pool_queries,
        "embedding": type(model.tokenizer).__name__,
        "query_fraction": getattr(model.tokenizer, "query_fraction", 1.0),
        "layers": len(model.encoder.layers),
        "directions": list(model.encoder.layers[0].attention.directions),
        "learned_pool_weights": model.encoder.layers[0].attention.pool_weight is not None,
        "windows_per_layer": [[[w.spatial_radius, w.time_horizon] for w in layer.attention.windows]
                              for layer in model.encoder.layers],
        "parameters": sum(p.numel() for p in model.parameters()),
    }
