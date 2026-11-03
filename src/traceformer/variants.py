"""S1 与八组论文消融使用同一构造和初始化路径。"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch
from torch import nn

from traceformer.config import ConeWindowConfig
from traceformer.models.context import FFNEncoder, MeanTokenizer, SummaryAttention
from traceformer.models.event_refinement import EventRefinementOutput
from traceformer.models.factory import build_base
from traceformer.models.past_local import PastLocalAttention, PastLocalEncoder
from traceformer.models.spatial_pooling import SpatialConvTokenizer

VARIANTS = ("S1", "S0", "A1", "A2", "P1", "G1", "G2", "F1", "T1")


def execution_contract():
    """分块影响训练执行顺序，恢复时必须保持一致。"""
    values = {}
    for name, default in (("TRACEFORMER_SPATIAL_TOKEN_CHUNK", "4096"),
                          ("TRACEFORMER_S1_DIRECT_QUERY_CHUNK", "256")):
        value = os.environ.get(name, default)
        if not value.isdecimal() or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
        values[name] = int(value)
    return values


@dataclass(frozen=True)
class PaperSpec:
    variant: str = "S1"
    version: int | None = None

    def __post_init__(self):
        if self.version is None:
            object.__setattr__(self, "version", 4 if self.variant in {"S1", "S0"} else 6)
        self.validate()

    def validate(self):
        expected = 4 if self.variant in {"S1", "S0"} else 6
        if self.variant not in VARIANTS or self.version != expected:
            raise ValueError("unsupported paper variant/checkpoint contract")
        return self

    def validate_stage(self, stage):
        self.validate()
        if stage != "full":
            raise ValueError("paper models require joint full training")

    def to_dict(self):
        return asdict(self.validate())


class NoRefinement(nn.Module):
    def forward(self, *, base_event_logits, **_):
        # A2 不含精修参数，辅助 cell 损失与逐事件残差均为零。
        return EventRefinementOutput(
            cell_logits=base_event_logits.new_empty(0),
            event_cell_indices=torch.empty(0, dtype=torch.long, device=base_event_logits.device),
            event_logits=torch.zeros_like(base_event_logits),
        )


def _base_variant(model, spatial):
    # 保留论文构造顺序和随机流，删除临时查询池化参数后仅保留真实模型。
    with torch.random.fork_rng(devices=[]):
        tokenizer = model.tokenizer
        tokenizer.__class__ = MeanTokenizer
        del tokenizer.pool_queries, tokenizer.key_projection
        del tokenizer.value_projection, tokenizer.query_fusion
        for layer in model.encoder.layers:
            layer.attention = SummaryAttention(layer.attention, past_only=True)
        if spatial:
            model.tokenizer = SpatialConvTokenizer.from_mean(model.tokenizer, 1)
    return model


def _matched_windows(windows):
    result = []
    for window in windows:
        volume = sum((2 * (window.spatial_radius * d // window.time_horizon) + 1) ** 2
                     for d in range(1, window.time_horizon + 1))
        radius = min(range(window.spatial_radius + 1), key=lambda r: abs(
            window.time_horizon * (2 * r + 1) ** 2 - volume))
        result.append(ConeWindowConfig(radius, window.time_horizon))
    return tuple(result)


def build_model(config, *, variant="S1", ablation=None):
    """默认构建论文 S1；checkpoint 使用原 schema-8 与消融契约。"""
    spec = ablation if ablation is not None else PaperSpec(variant)
    spec.validate()
    model = build_base(config)
    if spec.variant in {"S1", "S0"}:
        _base_variant(model, spec.variant == "S1")
    else:
        # 新增层不推进公共 RNG；P1、G2 保留对应 S1 初始系数。
        with torch.random.fork_rng(devices=[]):
            _base_variant(model, True)
            if spec.variant == "A1":
                model.encoder = FFNEncoder(model.encoder)
            elif spec.variant == "A2":
                model.event_refinement_head = NoRefinement()
            elif spec.variant == "P1":
                original = model.tokenizer.spatial_kernels[0]
                kernel = nn.Conv2d(original.in_channels, original.out_channels, 1,
                                   groups=original.groups, bias=False)
                with torch.no_grad():
                    kernel.weight.copy_(original.weight[:, :, 1:2, 1:2])
                model.tokenizer.spatial_kernels[0] = kernel
            else:
                for layer in model.encoder.layers:
                    attention = layer.attention
                    if spec.variant == "G1":
                        attention.windows = _matched_windows(attention.windows)
                        attention.fixed_radius = True
                    elif spec.variant == "G2":
                        attention.windows = attention.windows[1:2]
                        attention.shape_embedding = nn.Parameter(
                            attention.shape_embedding[1:2].detach().clone())
                    elif spec.variant == "F1":
                        dim = attention.pool_value.out_features
                        dropout = attention.center_attention.dropout
                        attention.mean = True
                        attention.projection = nn.Sequential(
                            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout),
                            nn.Linear(2 * dim, dim))
                        del attention.center_attention
                    elif spec.variant == "T1":
                        layer.attention = PastLocalAttention(
                            attention, windows=attention.windows,
                            chunk_size=os.environ.get("TRACEFORMER_S1_DIRECT_QUERY_CHUNK", "256"))
                if spec.variant == "T1":
                    model.encoder = PastLocalEncoder(model.encoder)
    model.ablation_spec = spec.to_dict()
    return model
