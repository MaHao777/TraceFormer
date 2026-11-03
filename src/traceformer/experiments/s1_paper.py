"""以S1为共同底座的结构删除与创新点替代对照。"""

import hashlib
import os

import torch
from torch import nn

from traceformer.experiments.ablation import LocalPoolingSpec, NoRefinement, S1PaperSpec
from traceformer.experiments.attention import matched_box_windows
from traceformer.experiments.component_core import FFNEncoder, assert_no_view
from traceformer.experiments.local_pooling import apply_local_pooling
from traceformer.experiments.past_local import PastLocalAttention, PastLocalEncoder

VARIANTS = ("S1", "S0", "A1", "A2", "P1", "G1", "G2", "F1", "T1")
NEW_VARIANTS = VARIANTS[2:]
LABELS = {"S1": "完整S1", "S0": "均值聚合", "A1": "去编码器跨Token交互", "A2": "去神经精修Head",
          "P1": "子格1×1卷积与GELU", "G1": "固定窗口摘要", "G2": "中间单锥摘要",
          "F1": "均值摘要与近容量MLP", "T1": "同域直接Token注意力"}
STRUCTURE_ROWS = ("S1", "S0", "A1", "A2")
MECHANISM_ROWS = ("S1", "S0", "P1", "G1", "G2", "F1", "T1")


def specification(variant):
    if variant in {"S1", "S0"}:
        return LocalPoolingSpec(variant)
    return S1PaperSpec(variant).validate()


def apply_s1_paper(model, spec):
    spec.validate()
    # 所有替代先构造同一S1，新增层不推进公共训练随机流。
    with torch.random.fork_rng(devices=[]):
        apply_local_pooling(model, LocalPoolingSpec("S1"))
        if spec.variant == "A1":
            model.encoder = FFNEncoder(model.encoder)
        elif spec.variant == "A2":
            model.event_refinement_head = NoRefinement()
        elif spec.variant == "P1":
            original = model.tokenizer.spatial_kernels[0]
            kernel = nn.Conv2d(original.in_channels, original.out_channels, 1,
                               groups=original.groups, bias=False)
            # 共享3×3核的中心系数，只移除邻格连接，GELU及回读保持一致。
            with torch.no_grad():
                kernel.weight.copy_(original.weight[:, :, 1:2, 1:2])
            model.tokenizer.spatial_kernels[0] = kernel
        else:
            for layer in model.encoder.layers:
                attention = layer.attention
                if spec.variant == "G1":
                    attention.windows = matched_box_windows(attention.windows)
                    attention.fixed_radius = True
                elif spec.variant == "G2":
                    attention.windows = attention.windows[1:2]
                    attention.shape_embedding = nn.Parameter(attention.shape_embedding[1:2].detach().clone())
                elif spec.variant == "F1":
                    dim = attention.pool_value.out_features
                    dropout = attention.center_attention.dropout
                    attention.mean = True
                    attention.projection = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(),
                                                          nn.Dropout(dropout), nn.Linear(2*dim, dim))
                    del attention.center_attention
                elif spec.variant == "T1":
                    layer.attention = PastLocalAttention(attention, windows=attention.windows,
                        chunk_size=os.environ.get("TRACEFORMER_S1_DIRECT_QUERY_CHUNK", "256"))
            if spec.variant == "T1":
                model.encoder = PastLocalEncoder(model.encoder)
    assert_no_view(model)
    model.ablation_spec = spec.to_dict()
    return model


def describe_s1(model, variant):
    layers = model.encoder.layers
    attention = getattr(layers[0], "attention", None)
    kernels = getattr(model.tokenizer, "spatial_kernels", [])
    return {"variant": variant, "label": LABELS[variant], "ablation": model.ablation_spec,
            "parameters": sum(p.numel() for p in model.parameters()), "layers": len(layers),
            "pool_kernel_sizes": [list(k.kernel_size) for k in kernels], "statistics": 7, "queries": 0,
            "directions": [-1] if attention is not None else [], "metric": "chebyshev", "cross_token_interaction": attention is not None,
            "windows": [[w.spatial_radius, w.time_horizon] for w in attention.windows] if attention else [],
            "fixed_radius": getattr(attention, "fixed_radius", False),
            "direct_union_candidates": len(attention.offsets) if isinstance(attention, PastLocalAttention) else None,
            "refinement": type(model.event_refinement_head).__name__,
            "fusion": "none" if variant == "A1" else "mean_mlp" if variant == "F1"
                      else "direct_tokens" if variant == "T1" else "center_summary",
            "claim": "same support union, deduplicated; no neighbor cap" if variant == "T1" else LABELS[variant]}


def initialization_audit(config):
    """核对所有保留权重与RNG，显式核对降维核与中间尺度切片。"""
    from traceformer.training.engine import build_model

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.training.seed)
        baseline = build_model(config, ablation=LocalPoolingSpec("S1"))
        reference, rng = baseline.state_dict(), torch.get_rng_state()
        rows = []
        for variant in VARIANTS:
            torch.manual_seed(config.training.seed)
            model = build_model(config, ablation=specification(variant))
            if not torch.equal(rng, torch.get_rng_state()):
                raise ValueError(f"S1 common training RNG changed: {variant}")
            digest, shared = hashlib.sha256(), 0
            for name, value in model.state_dict().items():
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                source = name.replace(".attention.attention.", ".attention.center_attention.")
                if variant == "A1" and name.startswith("encoder.layers."):
                    source = source.replace(".norm.", ".feed_forward_norm.").replace(".ffn.", ".feed_forward.")
                if variant == "F1" and ".attention.projection." in name:
                    continue
                expected = reference.get(source)
                if variant == "P1" and name == "tokenizer.spatial_kernels.0.weight":
                    expected = expected[:, :, 1:2, 1:2]
                if variant == "G2" and name.endswith(".shape_embedding"):
                    expected = expected[1:2]
                if expected is None or not torch.equal(value, expected):
                    raise ValueError(f"S1 shared initialization differs: {variant}/{name}")
                shared += value.numel()
            rows.append({**describe_s1(model, variant), "shared_parameters_verified": shared,
                         "initial_state_sha256": digest.hexdigest()})
    return {"passed": True, "seed": config.training.seed, "common_parameters_bitwise_equal": True,
            "training_rng_equal": True, "variants": rows}
