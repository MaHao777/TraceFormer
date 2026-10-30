"""首轮验证筛查后的摘要权重定位；仍为冻结推理干预，不是重训消融。"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from diagnose_original_neural_paths import run
from torch.nn import functional as F

from traceformer.models import prefix_cone

VARIANTS = (
    "full", "no_refinement",
    "uniform_summary_weights", "uniform_weights_0", "uniform_weights_1", "uniform_weights_2",
    "shuffled_summary_weights", "uniform_center_attention",
)


def uniform_attention(module, args, kwargs, result):
    """保留V和输出投影，仅将中心对可用摘要的注意力改为均匀分配。"""
    query, _, values = args
    if query.shape[1] != 1 or not module.batch_first or module.training:
        raise ValueError("requires the existing single-query evaluation attention")
    width = module.embed_dim
    projected = F.linear(values, module.in_proj_weight[2 * width:],
                         module.in_proj_bias[2 * width:])
    valid = ~kwargs["key_padding_mask"]
    mean = (projected * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True)
    return F.linear(mean[:, None], module.out_proj.weight, module.out_proj.bias), None


@contextmanager
def intervene(model, name, seed):
    """三个层使用原有权重；不创建新结构、不调整任何参数。"""
    with ExitStack() as stack:
        if name == "uniform_center_attention":
            for layer in model.encoder.layers:
                hook = layer.attention.center_attention.register_forward_hook(
                    uniform_attention, with_kwargs=True)
                stack.callback(hook.remove)
        if name.startswith("uniform_weights_") or name in {
                "uniform_summary_weights", "shuffled_summary_weights"}:
            original = prefix_cone.prefix_cone_pool
            generator = torch.Generator().manual_seed(seed)
            calls = 0

            def pool(values, weights, geometry, windows, *args, **kwargs):
                nonlocal calls
                index = calls % len(model.encoder.layers)
                calls += 1
                if name == "uniform_summary_weights" or name == f"uniform_weights_{index}":
                    weights = torch.ones_like(weights)
                elif name == "shuffled_summary_weights":
                    # 保持每个样本的权重分布，仅破坏权重与邻居特征的对应。
                    weights = weights.clone()
                    for sample in geometry:
                        indices = sample.indices
                        permutation = indices[torch.randperm(len(indices), generator=generator)]
                        weights[indices] = weights[permutation].clone()
                return original(values, weights, geometry, windows, *args, **kwargs)

            stack.enter_context(patch.object(prefix_cone, "prefix_cone_pool", pool))
        yield


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), max_seconds=600, variants=VARIANTS, intervention=intervene,
        purpose="post-screening validation follow-up: learned summary weighting versus center attention; "
                "NOT retrained ablation; variants chosen after the first validation screening")
