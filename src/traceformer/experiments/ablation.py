"""可追溯的结构消融，只在显式实验入口启用。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from traceformer.config import TraceFormerConfig
from traceformer.experiments.attention import PointwiseEncoder, matched_box_windows
from traceformer.experiments.pooling import PoolingTokenizer
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
        if self.backbone not in {"cone", "box", "pointwise"}:
            raise ValueError("unknown ablation backbone")
        if self.refinement not in {"none", "multiscale", "same_scale"}:
            raise ValueError("unknown ablation refinement")
        if type(self.sparse) is not bool or self.version != 1:
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


def apply_ablation(model: TraceFormer, config: TraceFormerConfig, spec: AblationSpec) -> TraceFormer:
    spec.validate()
    model.ablation_spec = spec.to_dict()
    if spec.backbone == "box":
        # 不重新构造注意力，确保 C1/C2 所有可学习参数具有相同初值。
        for layer in model.encoder.layers:
            layer.attention.windows = matched_box_windows(config.model.cone_windows)
            layer.attention.fixed_radius = True
    elif spec.backbone == "pointwise":
        model.encoder = PointwiseEncoder(model.encoder, config.model.dropout)
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
    return model
