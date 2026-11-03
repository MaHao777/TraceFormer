"""实际占用邻居统计独立于网络计时，避免用理论体积替代数据中的支持域。"""

import torch

from traceformer.experiments.ablation import LocalPoolingSpec
from traceformer.experiments.attention import matched_box_windows
from traceformer.experiments.component_suite import read_json
from traceformer.experiments.past_local import PastLocalAttention
from traceformer.experiments.runtime import setup_device
from traceformer.models.prefix_cone import prefix_cone_pool, prepare_prefix_geometry
from traceformer.training.engine import _dataset, _file_sha256, build_model


def measure_support(config, bundle, smoke):
    runtime = setup_device(config)
    model = build_model(config, ablation=LocalPoolingSpec("S1")).to(runtime.device).eval()
    union = PastLocalAttention(model.encoder.layers[0].attention, windows=config.model.cone_windows,
                               chunk_size=256).to(runtime.device)
    windows = config.model.cone_windows
    box = matched_box_windows(windows)
    files = []
    with torch.inference_mode():
        for sequence in _dataset(config, "val"):
            tokenized = model.tokenizer(sequence.events.to(runtime.device))
            coords, batches = tokenized.token_coords, tokenized.token_batch_indices
            geometry = prepare_prefix_geometry(coords, batches)
            values = tokenized.tokens.new_zeros((len(coords), 1))
            weights = torch.ones_like(values)
            stats = {}
            for label, supports, fixed in (("cone", windows, False), ("box", box, True)):
                _, counts = prefix_cone_pool(values, weights, geometry, supports,
                                             fixed_radius=fixed, directions=(-1,))
                stats[label] = [{"radius": window.spatial_radius, "horizon": window.time_horizon,
                    "members": int(counts[:, i].sum()), "mean": float(counts[:, i].float().mean()),
                    "p95": float(torch.quantile(counts[:, i].float(), 0.95))}
                    for i, window in enumerate(supports)]
            counts = torch.cat([(union._neighbor_indices(sample) >= 0).sum(1) for sample in geometry])
            stats["union"] = {"members": int(counts.sum()), "mean": float(counts.float().mean()),
                              "p95": float(torch.quantile(counts.float(), 0.95))}
            files.append({"file": sequence.source_path.name, "source_sha256": _file_sha256(sequence.source_path),
                          "tokens": len(coords), **stats})
    identities = read_json(bundle / "dataset_manifest.json")["val"]
    expected = sorted(identities)[:2] if smoke else sorted(identities)
    if [row["file"] for row in files] != expected:
        raise ValueError("support audit lacks complete validation input coverage")
    return {"runtime": runtime.metadata, "smoke": smoke, "split": "val", "files": files,
            "direct_union_candidates": len(union.offsets), "timing_scope": "diagnostic only; excluded from benchmark"}
