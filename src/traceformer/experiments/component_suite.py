"""单视图消融的前置验收、资源门槛与窗口几何诊断。"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import psutil
import torch
from torch.nn import functional as F

from traceformer.config import load_config
from traceformer.experiments.attention import matched_box_windows
from traceformer.experiments.component_core import assert_no_view
from traceformer.models.prefix_cone import prepare_prefix_geometry
from traceformer.training.engine import (
    _dataset,
    _dataset_manifest,
    _dense_sequence_loss_weight,
    _event_refinement_loss,
    _file_sha256,
    _manifest_identity_digest,
    _serializable,
    _validate_checkpoint_contract,
    build_model,
)
from traceformer.training.losses import EventFocalDiceLoss
from traceformer.training.sampling import sample_event_indices

VARIANTS = ("Full", "M1", "M2", "M3", "M4", "M5", "C1", "C2", "C3", "C4", "C5")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_baseline(root: Path) -> dict | None:
    """等待完整流程；失败与结果不一致不能作为可复用的Full。"""
    status = read_json(root / "status.json")
    if status["status"] == "failed":
        raise RuntimeError("baseline failed; preserve its outputs and inspect its logs")
    if status["status"] != "complete":
        return None
    training = root / "training"
    history = read_json(training / "history.json")
    if [row["epoch"] for row in history] != list(range(1, 25)):
        raise ValueError("baseline requires 24 completed epochs")
    config = load_config(training / "config.yaml")
    settings = config.training
    if (settings.seed, settings.epochs, settings.device, settings.num_threads,
            settings.learning_rate, settings.weight_decay, settings.stage, settings.mixed_precision
        ) != (37, 24, "cpu", 8, 0.0007, 0.0002, "full", False):
        raise ValueError("baseline training protocol differs")
    if settings.max_train_files is not None or settings.max_val_files is not None:
        raise ValueError("baseline cannot use partial partitions")
    if read_json(training / "metrics.json") != read_json(training / "reproduced_metrics.json"):
        raise ValueError("baseline validation was not independently reproduced")
    if read_json(root / "paper_verification.json").get("unchanged") is not True:
        raise ValueError("baseline paper integrity audit failed")
    initial, independent = (read_json(root / "test" / name)
                            for name in ("report.json", "independent_report.json"))
    checkpoint = torch.load(training / "best.pt", map_location="cpu", weights_only=True)
    _validate_checkpoint_contract(checkpoint, config)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    assert_no_view(model)
    digest = _file_sha256(training / "best.pt")
    for name in ("competition", "checkpoint_sha256", "threshold", "per_sequence",
                 "validation_identity", "parameters", "sequences", "ablation"):
        if initial[name] != independent[name]:
            raise ValueError(f"baseline test rescore mismatch: {name}")
    if set(initial["files"]) != set(independent["files"]):
        raise ValueError("baseline test file sets differ")
    for name, record in initial["files"].items():
        for key in ("events", "source_sha256", "prediction_sha256"):
            if record[key] != independent["files"][name][key]:
                raise ValueError(f"baseline test file identity mismatch: {name}")
    if (initial["sequences"] != 24 or len(initial["files"]) != 24
            or initial["checkpoint_sha256"] != digest or initial["ablation"] is not None
            or initial["threshold"] != checkpoint["decision_threshold"]):
        raise ValueError("baseline test/checkpoint identity mismatch")
    if len(_dataset(config, "train")) != 99 or len(_dataset(config, "val")) != 24:
        raise ValueError("requires complete 99/24 development partitions")
    manifest = _dataset_manifest(_dataset(config, "train"), _dataset(config, "val"))
    if _manifest_identity_digest(manifest) != checkpoint["dataset_manifest_sha256"]:
        raise ValueError("baseline development data changed")
    return {"checkpoint": str((training / "best.pt").resolve()), "sha256": digest,
            "config": str((training / "config.yaml").resolve()),
            "config_sha256": _file_sha256(training / "config.yaml"),
            "dataset_manifest_sha256": _manifest_identity_digest(manifest),
            "parameters": sum(p.numel() for p in model.parameters()),
            "test_report": str((root / "test/independent_report.json").resolve()),
            "protocol": _serializable(asdict(settings))}


def production_loss(model, sequence, config, criterion, *, sampling_generator=None):
    """资源预检沿用真实训练采样、主损失、精修监督及序列加权。"""
    # 资源检查和正式训练一样，让索引及采样生成器跟随事件设备。
    batches = torch.zeros(sequence.length, dtype=torch.long, device=sequence.events.device)
    output = model(sequence.events, batches)
    indices = sample_event_indices(
        sequence.labels, logits=output["event_logits"],
        background_ratio=config.training.background_ratio,
        hard_negative_fraction=config.training.hard_negative_fraction,
        max_background_without_positive=config.training.max_background_without_positive,
        # 连续计时沿用正式训练的持久采样器；旧资源检查保留原默认行为。
        generator=(sampling_generator if sampling_generator is not None else
                   torch.Generator(device=sequence.events.device).manual_seed(config.training.seed)),
    )
    losses = criterion(
        output["event_logits"][indices], sequence.labels[indices], batches[indices],
        object_ids=sequence.object_ids[indices], timestamps_ms=sequence.timestamps_ms[indices],
        frame_ms=config.evaluation.frame_ms, full_logits=output["event_logits"],
        full_targets=sequence.labels, full_batch_indices=batches,
        full_timestamps_ms=sequence.timestamps_ms, full_coordinates=sequence.events[:, :2],
    )
    total = (losses["total"] + config.frame_refinement.loss_weight
             * _event_refinement_loss(output, sequence.labels, criterion, config))
    return total * _dense_sequence_loss_weight(sequence.events, sequence.labels, config)


def resource_gate(config, spec, filename: str, *, repeats=3) -> dict:
    torch.set_num_threads(config.training.num_threads)
    torch.manual_seed(config.training.seed)
    dataset = _dataset(config, "train")
    sequence = dataset[next(i for i, p in enumerate(dataset.files) if p.name == filename)]
    model = build_model(config, ablation=spec).train()
    assert_no_view(model)
    criterion = EventFocalDiceLoss(**asdict(config.loss))
    seconds = []
    for repeat in range(repeats + 1):
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss = production_loss(model, sequence, config, criterion)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite resource-gate loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError(f"missing/nonfinite gradient: {name}")
        if repeat:
            seconds.append(time.perf_counter() - started)
        del loss
    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", memory.rss)
    return {"variant": spec.to_dict(), "sequence": filename, "seconds": seconds,
            "median_seconds": statistics.median(seconds), "peak_rss_bytes": peak,
            "physical_ram_bytes": psutil.virtual_memory().total,
            "memory_passed": peak <= psutil.virtual_memory().total * 0.7,
            "parameters": sum(p.numel() for p in model.parameters()), "finite_gradients": True}


def choose_local_neighbors(full: list[dict], local: list[dict]) -> bool:
    """按预先冻结的资源规则选择，不读取任何精度指标。"""
    if {r["sequence"] for r in full} != {r["sequence"] for r in local}:
        raise ValueError("resource-gate sequence sets differ")
    reference = {row["sequence"]: row for row in full}
    return all(row["memory_passed"] and row["finite_gradients"] and
               row["median_seconds"] <= 2 * reference[row["sequence"]]["median_seconds"]
               for row in local)


def window_diagnostics(coords, batches, windows) -> list[dict]:
    """标量占用前缀表统计交并集，避免逐中心的N平方枚举。"""
    boxes = matched_box_windows(windows)
    totals = [dict(tokens=0, cone=0, box=0, intersection=0, union=0,
                   cone_empty=0, box_empty=0, jaccard_sum=0.0) for _ in range(2 * len(windows))]
    for sample in prepare_prefix_geometry(coords, batches):
        times, height, width = sample.shape
        grid = torch.zeros(times * height * width, dtype=torch.int64)
        grid[sample.flat_indices] = 1
        integral = F.pad(grid.reshape(times, height, width).cumsum(1).cumsum(2), (1, 0, 1, 0))
        for index, (cone, box) in enumerate(zip(windows, boxes, strict=True)):
            for side, direction in enumerate((-1, 1)):
                accumulator = totals[2 * index + side]
                for start in range(0, len(sample.coords), 1024):
                    xyz = sample.coords[start:start + 1024]
                    distance = torch.arange(1, cone.time_horizon + 1)
                    frame = xyz[:, 2:3] + direction * distance
                    valid = (frame >= 0) & (frame < times)
                    frame = frame.clamp(0, times - 1)
                    cone_radius = cone.spatial_radius * distance // cone.time_horizon

                    def count(radius):
                        x0, x1 = ((xyz[:, 0:1] - radius).clamp(0, width),
                                  (xyz[:, 0:1] + radius + 1).clamp(0, width))
                        y0, y1 = ((xyz[:, 1:2] - radius).clamp(0, height),
                                  (xyz[:, 1:2] + radius + 1).clamp(0, height))
                        value = (integral[frame, y1, x1] - integral[frame, y0, x1]
                                 - integral[frame, y1, x0] + integral[frame, y0, x0])
                        return (value * valid).sum(1)

                    cone_count, box_count = count(cone_radius), count(box.spatial_radius)
                    intersection = count(cone_radius.clamp_max(box.spatial_radius))
                    union = cone_count + box_count - intersection
                    accumulator["tokens"] += len(xyz)
                    for name, value in (("cone", cone_count), ("box", box_count),
                                        ("intersection", intersection), ("union", union)):
                        accumulator[name] += int(value.sum())
                    accumulator["cone_empty"] += int((cone_count == 0).sum())
                    accumulator["box_empty"] += int((box_count == 0).sum())
                    accumulator["jaccard_sum"] += float(torch.where(
                        union > 0, intersection.double() / union.clamp_min(1), 1.0).sum())
    return [{**row, "scale": index // 2, "direction": ("past", "future")[index % 2],
             "cone_mean": row["cone"] / max(1, row["tokens"]),
             "box_mean": row["box"] / max(1, row["tokens"]),
             "cone_empty_fraction": row["cone_empty"] / max(1, row["tokens"]),
             "box_empty_fraction": row["box_empty"] / max(1, row["tokens"]),
             "jaccard_mean": row["jaccard_sum"] / max(1, row["tokens"]),
             "jaccard_micro": row["intersection"] / row["union"] if row["union"] else 1.0}
            for index, row in enumerate(totals)]
