"""均值/单核池化与小局部/三尺度中心欧氏过去锥的四组构造。"""

import hashlib

import torch

from traceformer.experiments.ablation import ComponentCoreSpec, EuclideanPoolingSpec
from traceformer.experiments.component_core import apply_component_core, assert_no_view
from traceformer.experiments.past_local import PastLocalAttention, PastLocalEncoder
from traceformer.experiments.spatial_pooling import SpatialConvTokenizer

VARIANTS = ("E0", "E1", "E2", "E3")
GEOMETRY = {"metric": "euclidean", "directions": [-1], "exclude_current_time": True,
            "boundary": "H^2*(dx^2+dy^2)<=R^2*d^2; 1<=d<=H",
            "units": "token grid", "local_windows": [[2, 2]],
            "summary_windows": [[8, 2], [4, 4], [2, 8]]}


def apply_euclidean_pooling(model, spec):
    spec.validate()
    # 公共参数沿用同一 M1+C4 初值；新增卷积不推进训练随机流。
    with torch.random.fork_rng(devices=[]):
        apply_component_core(model, ComponentCoreSpec("M1_C4"))
        if spec.variant in {"E1", "E3"}:
            model.tokenizer = SpatialConvTokenizer.from_mean(model.tokenizer, 1)
        for layer in model.encoder.layers:
            if spec.variant in {"E0", "E1"}:
                layer.attention = PastLocalAttention(layer.attention, metric="euclidean")
            else:
                layer.attention.metric = "euclidean"
        if spec.variant in {"E0", "E1"}:
            model.encoder = PastLocalEncoder(model.encoder)
    assert_no_view(model)
    model.ablation_spec = spec.to_dict()
    return model


def describe_euclidean(model):
    """记录替换后的真实结构，避免把配置模板中的 Query 数当作实际结构。"""
    return {"geometry": GEOMETRY, "pooling": "single_depthwise_3x3_gelu"
            if isinstance(model.tokenizer, SpatialConvTokenizer) else "mean_event_embeddings",
            "attention": "local" if isinstance(model.encoder, PastLocalEncoder) else "center_summary",
            "windows_per_layer": [[[w.spatial_radius, w.time_horizon] for w in layer.attention.windows]
                                  for layer in model.encoder.layers],
            "layers": len(model.encoder.layers), "queries": 0, "statistics": 7,
            "decoder": type(model.decoder).__name__,
            "refinement": type(model.event_refinement_head).__name__,
            "parameters": sum(p.numel() for p in model.parameters())}


def initialization_audit(config):
    """逐位核对公共权重、卷积配对和构造后的 RNG，并保存实际结构与初值哈希。"""
    from traceformer.training.engine import build_model

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.training.seed)
        reference_model = build_model(config, ablation=ComponentCoreSpec("M1_C4"))
        reference = reference_model.state_dict()
        expected_rng = torch.get_rng_state()
        rows, convolution = [], None
        for variant in VARIANTS:
            torch.manual_seed(config.training.seed)
            model = build_model(config, ablation=EuclideanPoolingSpec(variant))
            if not torch.equal(expected_rng, torch.get_rng_state()):
                raise ValueError("Euclidean model construction changed the common training RNG")
            digest = hashlib.sha256()
            for name, tensor in model.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                if name.startswith("tokenizer.spatial_kernels."):
                    if convolution is None:
                        convolution = tensor.clone()
                    elif not torch.equal(convolution, tensor):
                        raise ValueError("Euclidean paired convolution initialization differs")
                    continue
                source = name.replace(".attention.attention.", ".attention.center_attention.")
                if source not in reference or not torch.equal(tensor, reference[source]):
                    raise ValueError(f"Euclidean common initialization differs: {variant}/{name}")
            rows.append({"variant": variant, "ablation": model.ablation_spec,
                         "initial_state_sha256": digest.hexdigest(), **describe_euclidean(model)})
    return {"passed": True, "seed": config.training.seed, "common_parameters_bitwise_equal": True,
            "training_rng_equal": True, "convolution_pair_equal": True, "variants": rows}
