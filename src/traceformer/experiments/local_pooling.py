"""均值/空间卷积池化与过去锥局部注意力的六组配对构造。"""

from __future__ import annotations

import torch

from traceformer.experiments.ablation import ComponentCoreSpec
from traceformer.experiments.component_core import apply_component_core, assert_no_view
from traceformer.experiments.past_local import PastLocalAttention, PastLocalEncoder
from traceformer.experiments.spatial_pooling import SpatialConvTokenizer

VARIANTS = ("S0", "S1", "S2", "L0", "L1", "L2")


def apply_local_pooling(model, spec):
    spec.validate()
    # 统一从同一 M1+C4 初值开始，新增模块构造不推进公共训练 RNG。
    with torch.random.fork_rng(devices=[]):
        apply_component_core(model, ComponentCoreSpec("M1_C4"))
        kernels = int(spec.variant[1])
        if kernels:
            model.tokenizer = SpatialConvTokenizer.from_mean(model.tokenizer, kernels)
        if spec.variant.startswith("L"):
            for layer in model.encoder.layers:
                layer.attention = PastLocalAttention(layer.attention)
            model.encoder = PastLocalEncoder(model.encoder)
    assert_no_view(model)
    model.ablation_spec = spec.to_dict()
    return model
