"""独立 CPU 复评、场景诊断及原始事件顺序预测缓存。"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import psutil
import torch

from traceformer.config import TraceFormerConfig
from traceformer.data import is_dense_balanced_event_stream
from traceformer.experiments.ablation import AblationSpec
from traceformer.experiments.attention import neighbor_count_summary
from traceformer.models.prefix_cone import (
    PrefixConeEncoder,
    prefix_cone_pool,
    prepare_prefix_geometry,
)
from traceformer.training.engine import (
    _atomic_torch_save,
    _dataset,
    _dataset_manifest,
    _manifest_identity_digest,
    _metrics_at_threshold,
    _validate_checkpoint_contract,
    _write_json,
    build_model,
)
from traceformer.training.runtime import prepare_device


@torch.no_grad()
def evaluate_ablation(
    config: TraceFormerConfig, checkpoint_path: Path, spec: AblationSpec, *, save_predictions: bool,
    neighbor_diagnostics: bool = False,
) -> dict:
    device = prepare_device("cpu", num_threads=config.training.num_threads).device
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    _validate_checkpoint_contract(checkpoint, config, ablation=spec)
    dataset = _dataset(config, "val")
    if _manifest_identity_digest(_dataset_manifest(dataset)) != checkpoint["validation_manifest_sha256"]:
        raise ValueError("independent evaluation dataset mismatch")
    model = build_model(config, ablation=spec).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    threshold = float(checkpoint["decision_threshold"])
    results, groups, details, elapsed = [], {"sparse_eligible": [], "ordinary": []}, {}, []
    neighbor_counts = []
    prediction_dir = checkpoint_path.parent / "predictions"
    if save_predictions:
        prediction_dir.mkdir(exist_ok=True)
    # 固定前向一次作为预热；计时包含子流模拟与 CPU 邻域查找，排除磁盘读取。
    warm = dataset[0]
    model(warm.events, torch.zeros(warm.length, dtype=torch.long))
    del warm
    for sequence in dataset:
        batches = torch.zeros(sequence.length, dtype=torch.long)
        started = time.perf_counter()
        output = model(sequence.events, batches)
        seconds = time.perf_counter() - started
        elapsed.append(seconds)
        logits = output["event_logits"].detach().clone()
        if logits.shape != (sequence.length,) or not torch.isfinite(logits).all():
            raise ValueError("invalid ablation event output")
        entry = (logits, sequence)
        results.append(entry)
        eligible = is_dense_balanced_event_stream(
            sequence.events, min_events=config.training.dense_sequence_event_threshold,
            max_polarity_fraction=config.training.dense_sequence_background_polarity_max,
        )
        group = "sparse_eligible" if eligible else "ordinary"
        groups[group].append(entry)
        current = _metrics_at_threshold([entry], config, threshold)
        base = _metrics_at_threshold([(output["base_event_logits"], sequence)], config, threshold)
        details[sequence.source_path.stem] = {
            "group": group, "events": sequence.length, "seconds": seconds,
            "metrics": current, "before_refinement_same_threshold": base,
            "refinement_count_delta": {
                key: current["event"][key] - base["event"][key] for key in ("tp", "fp", "fn")
            },
            "token_count": len(output["token_features"]),
        }
        if neighbor_diagnostics and isinstance(model.encoder, PrefixConeEncoder) and len(output["token_coords"]):
            # 几何统计放在计时区间外；三层使用相同区域，只需统计一次。
            attention = model.encoder.layers[0].attention
            geometry = prepare_prefix_geometry(output["token_coords"], output["token_batch_indices"])
            ones = logits.new_ones((len(output["token_coords"]), 1))
            _, counts = prefix_cone_pool(
                ones, ones, geometry, attention.windows, fixed_radius=attention.fixed_radius,
            )
            neighbor_counts.append(counts)
            details[sequence.source_path.stem]["neighbor_counts"] = neighbor_count_summary(
                counts, attention.windows,
            )
        if save_predictions:
            _atomic_torch_save({
                "source_path": str(sequence.source_path), "event_logits": logits,
                "base_event_logits": output["base_event_logits"].detach().clone(),
                "threshold": threshold,
            }, prediction_dir / f"{sequence.source_path.stem}.pt")
        print(f"  independent evaluation {sequence.source_path.name}: {seconds:.2f}s", flush=True)
        del output
    memory = psutil.Process().memory_info()
    report = {
        "metrics": _metrics_at_threshold(results, config, threshold),
        "fixed_threshold": _metrics_at_threshold(results, config, 0.5),
        "groups": {name: _metrics_at_threshold(items, config, threshold)
                   for name, items in groups.items() if items},
        "per_file": details,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mean_seconds_per_sequence": statistics.mean(elapsed),
        "process_peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
        "memory_note": "process peak RSS includes model, evaluation data and diagnostic caches; CPU has no CUDA VRAM",
        "ablation": checkpoint["ablation"],
        "neighbor_counts": neighbor_count_summary(
            torch.cat(neighbor_counts), model.encoder.layers[0].attention.windows,
        ) if neighbor_counts else [],
        "neighbor_count_note": "actual occupied tokens per region; geometric diagnostics excluded from forward timing",
    }
    _write_json(checkpoint_path.parent / "reproduced_metrics.json", report["metrics"])
    _write_json(checkpoint_path.parent / "ablation_report.json", report)
    return report
