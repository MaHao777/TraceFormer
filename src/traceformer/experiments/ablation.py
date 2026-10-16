"""可追溯的结构消融，只在显式实验入口启用。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
from torch import nn

from traceformer.config import TraceFormerConfig
from traceformer.experiments.pooling import PoolingTokenizer
from traceformer.models.cone_attention import BidirectionalConeAttention, ConeAttentionOutput
from traceformer.models.cone_neighbor_builder import ConeGeometryContext
from traceformer.models.event_refinement import EventRefinementOutput
from traceformer.models.spatiotemporal_cone import TraceFormer


@dataclass(frozen=True)
class AblationSpec:
    backbone: str = "cone"
    sparse: bool = True
    refinement: str = "multiscale"
    version: int = 1
    pooling: str = "attention"

    def to_dict(self) -> dict:
        # 默认 attention 沿用旧契约，历史消融权重无需改写。
        result = asdict(self)
        if self.pooling == "attention":
            del result["pooling"]
        return result

    def validate(self) -> AblationSpec:
        if self.backbone not in {"cone", "window", "past", "no_position"}:
            raise ValueError("unknown ablation backbone")
        if self.refinement not in {"none", "multiscale", "same_scale"}:
            raise ValueError("unknown ablation refinement")
        if type(self.sparse) is not bool or self.version != 1:
            raise ValueError("unsupported ablation contract")
        if self.pooling not in {"attention", "mean", "statistics"}:
            raise ValueError("unknown token pooling")
        if self.pooling != "attention" and (self.backbone != "cone" or self.sparse or self.refinement != "none"):
            raise ValueError("pooling ablation requires cone backbone without Sparse/refinement")
        return self

    def validate_stage(self, stage: str) -> None:
        self.validate()
        valid = {
            "full": not self.sparse and self.refinement == "none",
            "sparse_mixture": self.sparse and self.refinement == "none",
            "refinement": self.refinement != "none",
        }
        if not valid.get(stage, False):
            raise ValueError("ablation modules do not match training stage")


class NoRefinement(nn.Module):
    def forward(self, *, base_event_logits: torch.Tensor, **_: Any) -> EventRefinementOutput:
        # 空 cell 使辅助损失为零；逐事件残差严格为零且无参数。
        return EventRefinementOutput(
            cell_logits=base_event_logits.new_empty(0),
            event_cell_indices=torch.empty(0, dtype=torch.long, device=base_event_logits.device),
            event_logits=torch.zeros_like(base_event_logits),
        )


class NoSparse(nn.Module):
    def forward(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("disabled Sparse module must not receive a sparse view")


class ZeroPositionBias(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.new_zeros(features.shape[:-1])


class PastOnlyAttention(BidirectionalConeAttention):
    def forward(
        self, tokens: torch.Tensor, token_coords: torch.Tensor, token_batch_indices: torch.Tensor,
        *, geometry: ConeGeometryContext | None = None,
    ) -> ConeAttentionOutput:
        if len(tokens) == 0:
            return ConeAttentionOutput(tokens)
        # 单方向消融沿用 Encoder 在本次前向准备的几何索引。
        neighborhoods = self.neighbor_builder(
            token_coords, token_batch_indices, geometry=geometry
        )
        past, _ = self._attend_direction(
            query=self._reshape_heads(self.query_projection(tokens)),
            key=self._reshape_heads(self.key_projection(tokens)),
            value=self._reshape_heads(self.value_projection(tokens)),
            token_coords=token_coords,
            neighbor_indices=neighborhoods.past_indices,
            direction_index=0,
        )
        # 单方向不需要门控或方向 dropout，不将其宣称为整网因果模型。
        return ConeAttentionOutput(self.output_projection(past.reshape(len(tokens), self.token_dim)))


def apply_ablation(model: TraceFormer, config: TraceFormerConfig, spec: AblationSpec) -> TraceFormer:
    spec.validate()
    model.ablation_spec = spec.to_dict()
    if spec.pooling != "attention":
        model.tokenizer = PoolingTokenizer.from_attention(model.tokenizer, spec.pooling)
    if not spec.sparse:
        model.sparse_mixture = NoSparse()
    if spec.refinement == "none":
        model.event_refinement_head = NoRefinement()
    if spec.refinement == "same_scale":
        coarse = config.frame_refinement.scales[-1]
        if len(config.frame_refinement.scales) != 3:
            raise ValueError("same-scale refinement requires three branches")
        # 实验契约记录尺度替换，正式配置仍要求严格递增的三个尺度。
        model.event_refinement_head.scales = (coarse,) * 3
    for layer in model.encoder.layers:
        attention = layer.attention
        if spec.backbone == "window":
            builder = attention.neighbor_builder
            # 按每层真实时间跨度匹配连续空间平均截面积，保留原 K 与排序。
            heads = []
            for head in builder.cone_heads:
                horizon = max(1, math.ceil(head.max_time_distance * builder.layer_time_scale))
                radius = math.sqrt(sum(
                    (head.base_radius + head.slope * d) ** 2 for d in range(1, horizon + 1)
                ) / horizon)
                heads.append(replace(head, base_radius=radius, slope=0.0))
            builder.cone_heads = tuple(heads)
        elif spec.backbone == "no_position":
            attention.relative_position = ZeroPositionBias()
        elif spec.backbone == "past":
            # 构造替代注意力时恢复 RNG，保证后续公共模块的初始化配对。
            with torch.random.fork_rng(devices=[]):
                past = PastOnlyAttention(
                    token_dim=attention.token_dim,
                    num_heads=attention.num_heads,
                    cone_heads=tuple(config.model.cone_heads),
                    max_neighbors_per_cone=config.model.max_neighbors_per_cone,
                    layer_time_scale=attention.neighbor_builder.layer_time_scale,
                    dropout=config.model.dropout,
                    direction_dropout=0.0,
                )
            past.load_state_dict(attention.state_dict(), strict=True)
            # 删除不参与计算的门控和未来嵌入，使参数量对应实际结构。
            del past.direction_gate
            past.direction_embeddings = nn.Parameter(past.direction_embeddings[:1].detach().clone())
            layer.attention = past
    return model
