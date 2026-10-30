"""可追溯的结构消融，只在显式实验入口启用。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from traceformer.config import TraceFormerConfig
from traceformer.experiments.attention import (
    PointwiseEncoder,
    SlidingLocalAttention,
    matched_box_windows,
)
from traceformer.experiments.pooling import PoolingTokenizer
from traceformer.models.event_refinement import EventRefinementOutput
from traceformer.models.spatiotemporal_cone import TraceFormer


@dataclass(frozen=True)
class LocalPoolingSpec:
    """六组实验只改变块内聚合和编码器交互，不开放隐式结构开关。"""

    variant: str = "S0"
    version: int = 4

    def validate(self) -> LocalPoolingSpec:
        if self.version != 4 or self.variant not in {"S0", "S1", "S2", "L0", "L1", "L2"}:
            raise ValueError("unknown local pooling experiment contract")
        return self

    def validate_stage(self, stage: str) -> None:
        self.validate()
        if stage != "full":
            raise ValueError("local pooling experiments require joint full training")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class StructureImprovementSpec:
    """两位分别启用A和B；共同配置作为替换前模板。"""

    variant: str = "00"
    version: int = 3

    def validate(self) -> StructureImprovementSpec:
        if self.version != 3 or self.variant not in {"00", "10", "01", "11"}:
            raise ValueError("unknown structure improvement contract")
        return self

    def validate_stage(self, stage: str) -> None:
        self.validate()
        if stage != "full":
            raise ValueError("structure improvements require joint full training")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ComponentCoreSpec:
    """新版仅描述当前单视图实验，不提供已删除分支的开关。"""

    variant: str = "Full"
    version: int = 2
    local_neighbors: int = 26

    def validate(self) -> ComponentCoreSpec:
        if self.version != 2 or self.variant not in {"Full", "M1_C4", *(f"M{i}" for i in range(1, 6)),
                                                     *(f"C{i}" for i in range(1, 6))}:
            raise ValueError("unknown component/core ablation")
        if type(self.local_neighbors) is not int or self.local_neighbors not in {6, 26}:
            raise ValueError("local_neighbors must be 6 or 26")
        if self.variant != "C1" and self.local_neighbors != 26:
            raise ValueError("local_neighbors only applies to C1")
        return self

    def validate_stage(self, stage: str) -> None:
        self.validate()
        if stage != "full":
            raise ValueError("component/core ablations require joint full training")

    def to_dict(self) -> dict:
        self.validate()
        result = {"version": 2, "variant": self.variant}
        if self.variant == "C1":
            result["local_neighbors"] = self.local_neighbors
        return result


def parse_ablation(
    value: dict,
) -> AblationSpec | ComponentCoreSpec | StructureImprovementSpec | LocalPoolingSpec:
    # 旧协议仍保持原含义；新实验契约拒绝任何额外的分支字段。
    if value.get("version") == 4:
        return LocalPoolingSpec(**value).validate()
    if value.get("version") == 3:
        return StructureImprovementSpec(**value).validate()
    return (ComponentCoreSpec(**value) if value.get("version") == 2 else AblationSpec(**value)).validate()


@dataclass(frozen=True)
class AblationSpec:
    backbone: str = "cone"
    sparse: bool = False
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
        if self.backbone not in {"cone", "box", "pointwise", "local", "single_box", "single_cone"}:
            raise ValueError("unknown ablation backbone")
        if self.refinement not in {"none", "multiscale", "same_scale"}:
            raise ValueError("unknown ablation refinement")
        if self.sparse is not False or self.version != 1:
            raise ValueError("unsupported ablation contract")
        if self.pooling not in {"attention", "mean", "statistics"}:
            raise ValueError("unknown token pooling")
        if self.backbone != "cone" and (self.sparse or self.refinement != "none"):
            raise ValueError("attention ablation requires backbone without Sparse/refinement")
        if self.pooling != "attention" and (self.backbone != "cone" or self.sparse or self.refinement != "none"):
            raise ValueError("pooling ablation requires cone backbone without Sparse/refinement")
        return self

    def validate_stage(self, stage: str) -> None:
        self.validate()
        valid = {
            "full": not self.sparse and self.refinement == "none",
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




def apply_ablation(
    model: TraceFormer, config: TraceFormerConfig,
    spec: AblationSpec | ComponentCoreSpec | StructureImprovementSpec | LocalPoolingSpec,
) -> TraceFormer:
    spec.validate()
    if isinstance(spec, LocalPoolingSpec):
        from traceformer.experiments.local_pooling import apply_local_pooling
        return apply_local_pooling(model, spec)
    if isinstance(spec, StructureImprovementSpec):
        from traceformer.experiments.structure import apply_structure_improvements
        return apply_structure_improvements(model, config, spec)
    if isinstance(spec, ComponentCoreSpec):
        from traceformer.experiments.component_core import apply_component_core
        return apply_component_core(model, spec)
    model.ablation_spec = spec.to_dict()
    if spec.backbone == "box":
        # 不重新构造注意力，确保 C1/C2 所有可学习参数具有相同初值。
        for layer in model.encoder.layers:
            layer.attention.windows = matched_box_windows(config.model.cone_windows)
            layer.attention.fixed_radius = True
    elif spec.backbone == "pointwise":
        model.encoder = PointwiseEncoder(model.encoder, config.model.dropout)
    elif spec.backbone in {"local", "single_box", "single_cone"}:
        for layer in model.encoder.layers:
            attention = layer.attention
            # 单尺度固定为中间锥，直接保留公共权重，不推进初始化随机数。
            attention.windows = (config.model.cone_windows[1],)
            attention.shape_embedding = nn.Parameter(attention.shape_embedding[1:2].detach().clone())
            attention.fixed_radius = spec.backbone != "single_cone"
            if spec.backbone == "local":
                layer.attention = SlidingLocalAttention(attention)
    if spec.pooling != "attention":
        model.tokenizer = PoolingTokenizer.from_attention(model.tokenizer, spec.pooling)
    if spec.refinement == "none":
        model.event_refinement_head = NoRefinement()
    if spec.refinement == "same_scale":
        coarse = config.frame_refinement.scales[-1]
        if len(config.frame_refinement.scales) != 3:
            raise ValueError("same-scale refinement requires three branches")
        # 实验契约记录尺度替换，正式配置仍要求严格递增的三个尺度。
        model.event_refinement_head.scales = (coarse,) * 3
    return model
