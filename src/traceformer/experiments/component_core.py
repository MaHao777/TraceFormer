"""无额外视图的十组单因素结构替换，共用正式网络与训练循环。"""

from __future__ import annotations

import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from traceformer.experiments.attention import matched_box_windows
from traceformer.models.cone_transformer import ConeAttentionOutput, ConeTransformerOutput
from traceformer.models.event_tokenizer import EventTokenizer
from traceformer.models.prefix_cone import prefix_cone_pool, prepare_prefix_geometry


class SelectedLinear(nn.Module):
    """真正删除指定输入列的参数，而非用零值保留无效权重。"""

    def __init__(self, source: nn.Linear, columns: list[int]) -> None:
        super().__init__()
        self.register_buffer("columns", torch.tensor(columns), persistent=False)
        self.linear = nn.Linear(len(columns), source.out_features, bias=source.bias is not None)
        with torch.no_grad():
            self.linear.weight.copy_(source.weight[:, columns])
            if source.bias is not None:
                self.linear.bias.copy_(source.bias)

    def forward(self, value):
        return self.linear(value.index_select(-1, self.columns))


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


class GridLocalAttention(nn.Module):
    """C1：中心查询与26/6邻接，QKV只投影一次，分块重算控制内存。"""

    def __init__(self, original, neighbors=26, chunk_size=256):
        super().__init__()
        if neighbors not in {6, 26}:
            raise ValueError("neighbors must be 6 or 26")
        self.center_attention = original.center_attention
        self.chunk_size = chunk_size
        offsets = [p for p in itertools.product((-1, 0, 1), repeat=3)
                   if any(p) and (neighbors == 26 or sum(abs(v) for v in p) == 1)]
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)

    def _chunk(self, q, k, v, coords, lookup, shape, start, end):
        times, height, width = shape
        xyz = coords[start:end, None] + self.offsets[None]
        inside = ((xyz >= 0) & (xyz < xyz.new_tensor((width, height, times)))).all(-1)
        flat = ((xyz[..., 2].clamp(0, times - 1) * height
                 + xyz[..., 1].clamp(0, height - 1)) * width
                + xyz[..., 0].clamp(0, width - 1))
        neighbors = lookup[flat]
        mask = inside & (neighbors >= 0)
        valid = mask.any(1)
        neighbors = neighbors.clamp_min(0)
        mask = mask.clone()
        mask[~valid, 0] = True
        heads = self.center_attention.num_heads
        dim = q.shape[-1] // heads
        keys = k[neighbors].reshape(end - start, -1, heads, dim).transpose(1, 2)
        values = v[neighbors].reshape(end - start, -1, heads, dim).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            q[start:end].reshape(-1, heads, 1, dim), keys, values,
            attn_mask=mask[:, None, None],
            dropout_p=self.center_attention.dropout if self.training else 0.0,
        ).reshape(end - start, -1)
        return torch.where(valid[:, None], output, 0.0), valid

    def forward(self, tokens, token_coords, token_batch_indices, *, geometry=None):
        if not len(tokens):
            return ConeAttentionOutput(tokens)
        if geometry is None:
            geometry = prepare_prefix_geometry(token_coords, token_batch_indices)
        attention = self.center_attention
        q, k, v = F.linear(tokens, attention.in_proj_weight, attention.in_proj_bias).chunk(3, -1)
        features, available = torch.zeros_like(tokens), torch.zeros(len(tokens), dtype=torch.bool,
                                                                   device=tokens.device)
        for sample in geometry:
            indices = sample.indices
            lookup = torch.full((sample.shape[0] * sample.shape[1] * sample.shape[2],), -1,
                                dtype=torch.long, device=tokens.device)
            lookup[sample.flat_indices] = torch.arange(len(indices), device=tokens.device)
            args = (q[indices], k[indices], v[indices], sample.coords, lookup, sample.shape)
            chunks, masks = [], []
            for start in range(0, len(indices), self.chunk_size):
                arguments = (*args, start, min(start + self.chunk_size, len(indices)))
                result = (checkpoint(self._chunk, *arguments, use_reentrant=False)
                          if torch.is_grad_enabled() else self._chunk(*arguments))
                chunks.append(result[0])
                masks.append(result[1])
            features[indices], available[indices] = torch.cat(chunks), torch.cat(masks)
        output = attention.out_proj(features)
        return ConeAttentionOutput(torch.where(available[:, None], output, 0.0))


def assert_no_view(model) -> None:
    # 检查具体已删除分支，不误杀正常的非空块Token表示。
    forbidden = ("sparse_mixture", "sparse_view", "full_sparse", "sparse_fusion")
    names = list(model.state_dict()) + [name for name, _ in model.named_modules()]
    if any(part in name.lower() for name in names for part in forbidden):
        raise ValueError("deleted view branch found in model")


def apply_component_core(model, spec):
    spec.validate()
    # 构造替代层不推进训练RNG，公共层保持相同初值。
    with torch.random.fork_rng(devices=[]):
        variant = spec.variant
        if variant in {"M1", "M1_C4"}:
            tokenizer = model.tokenizer
            tokenizer.__class__ = MeanTokenizer
            del tokenizer.pool_queries, tokenizer.key_projection
            del tokenizer.value_projection, tokenizer.query_fusion
            # 联合实验只叠加旧C4，不引入新的均值/统计融合方式。
            if variant == "M1_C4":
                if len(model.encoder.layers) != 3:
                    raise ValueError("M1_C4 requires exactly three encoder layers")
                variant = "C4"
        if variant == "M2":
            tokenizer = model.tokenizer
            dim, stats = tokenizer.event_embed_dim, tokenizer.statistic_dim
            tokenizer.token_projection[0] = SelectedLinear(
                tokenizer.token_projection[0], list(range(dim)) + list(range(dim + stats, 2 * dim + stats)))
        elif variant == "M3":
            model.encoder = FFNEncoder(model.encoder)
        elif variant == "M4":
            dim = model.token_dim
            model.decoder.classifier[0] = SelectedLinear(model.decoder.classifier[0], list(range(dim)))
            model.event_refinement_head.event_projection[0] = SelectedLinear(
                model.event_refinement_head.event_projection[0], list(range(dim)) + [dim + 6])
        elif variant == "M5":
            head = model.event_refinement_head
            head.scales = head.scales[-1:]
            head.cell_encoders = nn.ModuleList([head.cell_encoders[-1]])
            head.temporal_fusers = nn.ModuleList([head.temporal_fusers[-1]])
            context = head.context_dim
            columns = list(range(context)) + list(range(3 * context, 4 * context + 2))
            head.event_classifier[0] = SelectedLinear(head.event_classifier[0], columns).linear
        elif variant.startswith("C"):
            for layer in model.encoder.layers:
                original = layer.attention
                if variant == "C1":
                    layer.attention = GridLocalAttention(original, spec.local_neighbors)
                elif variant == "C2":
                    original.windows = matched_box_windows(original.windows)
                    original.fixed_radius = True
                elif variant == "C3":
                    original.windows = original.windows[1:2]
                    original.shape_embedding = nn.Parameter(original.shape_embedding[1:2].detach().clone())
                elif variant in {"C4", "C5"}:
                    layer.attention = SummaryAttention(original, past_only=variant == "C4", mean=variant == "C5")
    assert_no_view(model)
    model.ablation_spec = spec.to_dict()
    return model
