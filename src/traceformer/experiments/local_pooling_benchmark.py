"""六组实验的完整网络实测；缓存重评分时间不进入速度比较。"""

from __future__ import annotations

import copy
import gc
import math
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from traceformer.experiments.component_suite import production_loss
from traceformer.experiments.local_pooling import VARIANTS
from traceformer.training.engine import (
    _dataset,
    _file_sha256,
    _validate_checkpoint_contract,
    build_model,
)
from traceformer.training.losses import EventFocalDiceLoss
from traceformer.training.runtime import prepare_device


def numerical_check(model, device) -> dict:
    """同权重小样本覆盖完整前向、门控和新增卷积的梯度。"""
    reference = copy.deepcopy(model).cpu().eval()
    model.eval()
    tokenizer = model.tokenizer
    events = torch.tensor([
        [min(tokenizer.width - 1, (x + local_x) * tokenizer.spatial_block_size_x),
         min(tokenizer.height - 1, (y + local_y) * tokenizer.spatial_block_size_y),
         (t + offset) / tokenizer.num_time_blocks, polarity]
        for x in range(2) for y in range(2) for t in range(min(4, tokenizer.num_time_blocks))
        for local_x in (0.125, 0.375, 0.625, 0.875)
        for local_y in (0.125, 0.375, 0.625, 0.875)
        for offset, polarity in ((0.1, 0.0), (0.7, 1.0))
    ], dtype=torch.float32)
    actual = model(events.to(device))["event_logits"]
    expected = reference(events)["event_logits"]
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-4, rtol=3e-4)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (name, left), (_, right) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if left.grad is None or right.grad is None:
            raise ValueError(f"numerical check missing gradient: {name}")
        if not torch.isfinite(left.grad).all() or not torch.isfinite(right.grad).all():
            raise ValueError(f"numerical check nonfinite gradient: {name}")
        torch.testing.assert_close(left.grad.cpu(), right.grad, atol=5e-4, rtol=5e-3)
    difference = float((actual.cpu() - expected).abs().max().detach())
    model.zero_grad(set_to_none=True)
    return {"passed": True, "target_device": str(device),
            "real_cuda_checked": device.type == "cuda",
            "max_logit_absolute_difference": difference,
            "parameter_gradients_checked": True}


def _synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak(device) -> int:
    # reserved 包含缓存，防止只看 allocated 而低估独占运行需要的显存。
    return int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0


def _inference_report(model, dataset, device, smoke):
    # 提前载入事件，单独记录完整网络前向；不把磁盘和评分混入耗时。
    inputs = [(sequence.source_path, sequence.events.to(device)) for sequence in dataset]
    rows = []
    warmup, repeats = (0, 1) if smoke else (1, 3)
    _reset_peak(device)
    with torch.inference_mode():
        for path, events in inputs:
            batches = torch.zeros(len(events), device=device, dtype=torch.long)
            durations = []
            for repeat in range(warmup + repeats):
                _synchronize(device)
                started = time.perf_counter()
                output = model(events, batches)
                _synchronize(device)
                elapsed = time.perf_counter() - started
                if repeat >= warmup:
                    durations.append(elapsed)
                del output
            rows.append({"file": path.name, "events": len(events),
                         "source_sha256": _file_sha256(path), "warmup": warmup, "repeats": repeats,
                         "seconds": durations, "median_seconds": statistics.median(durations)})
    values = [value for row in rows for value in row["seconds"]]
    return {"sequences": rows, "mean_seconds": statistics.mean(values),
            "p95_seconds": float(np.quantile(values, 0.95)), "total_seconds": sum(values),
            "peak_memory_bytes": _peak(device),
            "scope": "full network forward; GPU-resident inputs; synchronized wall time; no IO/scoring"}


def run_prediction_benchmark(config, spec, *, checkpoint, test_root, smoke=False) -> dict:
    """只测最佳权重在测试集的预测时间，不执行训练步骤或预算判断。"""
    if config.training.mixed_precision:
        raise ValueError("prediction timing requires FP32")
    runtime = prepare_device(config.training.device, num_threads=config.training.num_threads)
    device = runtime.device
    if device.type != "cuda" and not smoke:
        raise ValueError("formal prediction timing requires a real CUDA GPU")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _validate_checkpoint_contract(saved, config, ablation=spec)
    model = build_model(config, ablation=spec).to(device)
    model.load_state_dict(saved["model_state"], strict=True)
    numerical = numerical_check(model, device)
    model.eval()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    test_config = replace(config, dataset=replace(config.dataset, root=Path(test_root)))
    dataset = _dataset(test_config, "test")
    if len(dataset) != (2 if smoke else 24):
        raise ValueError("prediction timing requires the complete test partition (2 files for smoke)")
    inference = _inference_report(model, dataset, device, smoke)
    return {"schema": 1, "variant": spec.to_dict(), "runtime": runtime.metadata,
            "device": device.type, "precision": "fp32", "smoke": bool(smoke),
            "checkpoint_sha256": _file_sha256(checkpoint), "split": "test", "timing_only": True,
            "budget_enforced": False, "parameters": sum(p.numel() for p in model.parameters()),
            "numerical_check": numerical, "inference": inference,
            "peak_memory_bytes": inference["peak_memory_bytes"],
            "total_memory_bytes": runtime.metadata.get("total_memory_bytes", 0)}


def run_benchmark(config, spec, *, checkpoint=None, smoke=False) -> dict:
    """原始事件已在GPU上；完整训练步骤包含AdamW，不计I/O和评分。"""
    if config.training.mixed_precision:
        raise ValueError("six-group benchmark requires FP32")
    runtime = prepare_device(config.training.device, num_threads=config.training.num_threads)
    device = runtime.device
    if device.type != "cuda" and not smoke:
        raise ValueError("formal resource gates require a real CUDA GPU")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    train, validation = _dataset(config, "train"), _dataset(config, "val")
    if len(validation) != (2 if smoke else 24):
        raise ValueError("benchmark requires complete validation partition (2 files for smoke)")
    if len(train) != (2 if smoke else 99):
        raise ValueError("benchmark requires complete training partition (2 files for smoke)")
    torch.manual_seed(config.training.seed)
    model = build_model(config, ablation=spec).to(device)
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _validate_checkpoint_contract(saved, config, ablation=spec)
        model.load_state_dict(saved["model_state"], strict=True)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    numerical = numerical_check(model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                                 weight_decay=config.training.weight_decay)
    criterion = EventFocalDiceLoss(**asdict(config.loss))
    sampling_generator = torch.Generator(device=device).manual_seed(config.training.seed)
    names = ([path.name for path in train.files[:2]] if smoke
             else ["train_094.npz", "train_096.npz"])
    training = []
    warmup, repeats = (0, 1) if smoke else (2, 5)
    model.train()
    for name in names:
        index = next((i for i, path in enumerate(train.files) if path.name == name), None)
        if index is None:
            raise ValueError(f"resource gate requires full training sequence {name}")
        sequence = train[index]
        sequence = replace(sequence, **{key: getattr(sequence, key).to(device)
                           for key in ("events", "labels", "object_ids", "timestamps_ms")})
        durations = []
        for repeat in range(warmup + repeats):
            if repeat == warmup:
                _reset_peak(device)
            _synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = production_loss(model, sequence, config, criterion,
                                   sampling_generator=sampling_generator)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite benchmark loss")
            loss.backward()
            if any(parameter.grad is None for parameter in model.parameters() if parameter.requires_grad):
                raise ValueError("missing benchmark parameter gradient")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite benchmark gradient")
            optimizer.step()
            _synchronize(device)
            elapsed = time.perf_counter() - started
            if repeat >= warmup:
                durations.append(elapsed)
        training.append({"file": name, "events": sequence.length,
                         "source_sha256": _file_sha256(sequence.source_path),
                         "warmup": warmup, "repeats": repeats, "seconds": durations,
                         "median_seconds": statistics.median(durations),
                         "peak_memory_bytes": _peak(device), "finite_gradients": True})
        del loss, sequence
    # 训练计时会更新临时权重；推理前恢复真实best或原始初值。
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.load_state_dict(initial, strict=True)
    model.eval()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    inference = _inference_report(model, validation, device, smoke)
    return {"schema": 1, "variant": spec.to_dict(), "runtime": runtime.metadata,
            "device": device.type, "precision": "fp32", "smoke": bool(smoke),
            "checkpoint_sha256": _file_sha256(checkpoint) if checkpoint is not None else None,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "numerical_check": numerical,
            "training": {"sequences": training, "optimizer": "AdamW", "includes_optimizer_step": True},
            "inference": inference,
            "peak_memory_bytes": max([inference["peak_memory_bytes"],
                                      *(row["peak_memory_bytes"] for row in training)]),
            "total_memory_bytes": runtime.metadata.get("total_memory_bytes", 0)}


def _positive(value, label) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid benchmark value: {label}")
    return float(value)


def assess_budget(reports: dict) -> dict:
    """六组齐全且同设备同输入，才按1.25倍与90%显存门槛验收。"""
    if set(reports) != set(VARIANTS):
        raise ValueError("resource reports must contain all six variants")
    baseline = reports["S0"]
    smoke = baseline["smoke"]
    if type(smoke) is not bool:
        raise ValueError("resource report smoke flag must be boolean")
    failures, rows = [], []
    expected_train = [row["file"] for row in baseline["training"]["sequences"]]
    expected_val = [row["file"] for row in baseline["inference"]["sequences"]]
    if not smoke and (expected_train != ["train_094.npz", "train_096.npz"]
                      or expected_val != [f"val_{index:03}.npz" for index in range(24)]):
        raise ValueError("formal resource report has incomplete benchmark inputs")
    reference_train = {row["file"]: row for row in baseline["training"]["sequences"]}
    reference_val = {row["file"]: row for row in baseline["inference"]["sequences"]}
    for name in VARIANTS:
        report = reports[name]
        if (type(report["smoke"]) is not bool or report["smoke"] != smoke
                or report["device"] != baseline["device"]
                or report["runtime"] != baseline["runtime"] or report["precision"] != "fp32"):
            raise ValueError("resource reports differ in device/runtime/precision/protocol")
        if not smoke and (report["device"] != "cuda"
                          or report["runtime"].get("cuda_kernel_verified") is not True):
            raise ValueError("formal resource gate requires verified CUDA")
        legacy = name == "S0" and report["variant"].get("version") == 2
        expected_contract = {"variant": "M1_C4" if legacy else name, "version": 2 if legacy else 4}
        if report["variant"] != expected_contract:
            raise ValueError("resource report variant identity differs")
        if (report["numerical_check"].get("passed") is not True
                or report["numerical_check"].get("parameter_gradients_checked") is not True):
            raise ValueError("resource report numerical check failed")
        if not smoke and report["numerical_check"].get("real_cuda_checked") is not True:
            raise ValueError("resource report has no real CUDA numerical check")
        if (report["training"].get("includes_optimizer_step") is not True
                or report["training"].get("optimizer") != "AdamW"):
            raise ValueError("training benchmark must include optimizer step")
        train_rows, val_rows = report["training"]["sequences"], report["inference"]["sequences"]
        if ([row["file"] for row in train_rows] != expected_train
                or [row["file"] for row in val_rows] != expected_val):
            raise ValueError("resource reports use different sequence sets")
        for part, reference, warm, repeats in ((train_rows, reference_train, 2, 5),
                                               (val_rows, reference_val, 1, 3)):
            for row in part:
                other = reference[row["file"]]
                digest = row["source_sha256"]
                if (type(row["events"]) is not int or row["events"] <= 0
                        or not isinstance(digest, str) or len(digest) != 64
                        or any(character not in "0123456789abcdef" for character in digest)):
                    raise ValueError("resource report has invalid source identity")
                if (row["source_sha256"], row["events"]) != (other["source_sha256"], other["events"]):
                    raise ValueError("resource reports use different source data")
                if not smoke and (row["warmup"] != warm or row["repeats"] != repeats):
                    raise ValueError("resource report has incomplete warmup/repeats")
                if len(row["seconds"]) != row["repeats"]:
                    raise ValueError("resource report timing sample count differs")
                if not row["seconds"]:
                    raise ValueError("resource report has no timing samples")
                for value in row["seconds"]:
                    _positive(value, "seconds")
                if not math.isclose(row["median_seconds"], statistics.median(row["seconds"]),
                                    rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError("resource report median differs from timing samples")
        if not all(row.get("finite_gradients") is True for row in train_rows):
            raise ValueError("resource report gradients failed")
        values = [value for row in val_rows for value in row["seconds"]]
        for field, actual in (("mean_seconds", statistics.mean(values)),
                              ("p95_seconds", float(np.quantile(values, 0.95)))):
            if not math.isclose(report["inference"][field], actual, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("resource report aggregate differs from timing samples")
        phase_peaks = [report["inference"]["peak_memory_bytes"],
                       *(row["peak_memory_bytes"] for row in train_rows)]
        if any(type(peak) is not int or peak < 0 or (report["device"] == "cuda" and peak == 0)
               for peak in phase_peaks):
            raise ValueError("invalid benchmark memory")
        if report["peak_memory_bytes"] != max(phase_peaks):
            raise ValueError("resource report peak differs from measured phases")
        if not smoke and (report["total_memory_bytes"] != baseline["total_memory_bytes"]
                          or report["total_memory_bytes"] != report["runtime"]["total_memory_bytes"]):
            raise ValueError("resource report device memory differs")
        train_ratios = {row["file"]: _positive(row["median_seconds"], "train median")
                        / _positive(reference_train[row["file"]]["median_seconds"], "S0 train median")
                        for row in train_rows}
        mean_ratio = (_positive(report["inference"]["mean_seconds"], "inference mean")
                      / _positive(baseline["inference"]["mean_seconds"], "S0 inference mean"))
        p95_ratio = (_positive(report["inference"]["p95_seconds"], "inference p95")
                     / _positive(baseline["inference"]["p95_seconds"], "S0 inference p95"))
        peak_ratio = (report["peak_memory_bytes"] / _positive(baseline["peak_memory_bytes"], "S0 memory")
                      if report["device"] == "cuda" else None)
        memory_fraction = (report["peak_memory_bytes"] / _positive(report["total_memory_bytes"], "GPU memory")
                           if report["device"] == "cuda" else None)
        checks = {"training_time": max(train_ratios.values()) <= 1.25,
                  "inference_mean": mean_ratio <= 1.25, "inference_p95": p95_ratio <= 1.25,
                  "memory_ratio": peak_ratio is None or peak_ratio <= 1.25,
                  "device_memory": memory_fraction is None or memory_fraction <= 0.9}
        # CPU微型闭环只验证接口，不用噪声计时冒充GPU预算验收。
        if not smoke:
            failures.extend(f"{name}:{key}" for key, passed in checks.items() if not passed)
        rows.append({"variant": name, "training_ratios": train_ratios,
                     "inference_mean_ratio": mean_ratio, "inference_p95_ratio": p95_ratio,
                     "memory_ratio": peak_ratio, "device_memory_fraction": memory_fraction,
                     "checks": checks})
    return {"passed": not failures, "validated_cuda_budget": not smoke and not failures,
            "smoke": smoke, "limit_ratio": 1.25, "max_device_memory_fraction": 0.9,
            "rows": rows, "failures": failures}
