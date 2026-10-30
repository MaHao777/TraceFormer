"""用真实检查点核验旧Torch与融合CPU执行的全量验证输出和掩码。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from traceformer.config import load_config
from traceformer.experiments.ablation import AblationSpec
from traceformer.training.engine import (
    _dataset,
    _file_sha256,
    _metrics_at_threshold,
    _validate_checkpoint_contract,
    _write_json,
    build_model,
)


def check(config_path, checkpoint_path, output):
    config = load_config(config_path)
    torch.set_num_threads(config.training.num_threads)
    saved = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    spec = AblationSpec(**saved["ablation"])
    if spec.backbone != "local":
        raise ValueError("requires an A0 checkpoint")
    _validate_checkpoint_contract(saved, config, ablation=spec)
    model = build_model(config, ablation=spec).eval()
    model.load_state_dict(saved["model_state"], strict=True)
    threshold = saved["decision_threshold"]
    comparisons, original_results, fused_results = {}, [], []
    with torch.inference_mode():
        for sequence in _dataset(config, "val"):
            values, seconds = {}, {}
            for backend in ("torch", "fused"):
                for layer in model.encoder.layers:
                    layer.attention.cpu_backend = backend
                started = time.perf_counter()
                values[backend] = model(sequence.events)
                seconds[backend] = time.perf_counter() - started
            for key, value in values["torch"].items():
                torch.testing.assert_close(value, values["fused"][key], atol=3e-5, rtol=3e-5, msg=key)
            original, fused = values["torch"]["event_logits"], values["fused"]["event_logits"]
            different = int(((original.sigmoid() >= threshold) != (fused.sigmoid() >= threshold)).sum())
            comparisons[sequence.source_path.stem] = {
                "events": sequence.length, "max_logit_absolute_error": float((original - fused).abs().max()),
                "different_mask_events": different, "seconds": seconds,
            }
            original_results.append((original, sequence))
            fused_results.append((fused, sequence))
            print(f"{sequence.source_path.stem}: mask differences={different}, "
                  f"torch={seconds['torch']:.3f}s fused={seconds['fused']:.3f}s", flush=True)
    original_metrics = _metrics_at_threshold(original_results, config, threshold)
    fused_metrics = _metrics_at_threshold(fused_results, config, threshold)
    _write_json(output, {
        "status": "passed", "checkpoint_sha256": _file_sha256(checkpoint_path),
        "sequences": len(comparisons), "threshold": threshold, "per_sequence": comparisons,
        "all_masks_equal": all(v["different_mask_events"] == 0 for v in comparisons.values()),
        "all_metrics_equal": original_metrics == fused_metrics,
        "torch_metrics": original_metrics, "fused_metrics": fused_metrics,
        "note": "float32 outputs checked within 3e-5; timings are diagnostics, not final benchmark",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    check(args.config, args.checkpoint, args.output)
