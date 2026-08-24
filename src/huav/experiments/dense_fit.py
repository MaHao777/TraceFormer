"""Short, gated capacity tests for weak dense HUAV training sequences."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from torch.nn.utils import clip_grad_norm_

from huav.config import HUAVConfig, load_config
from huav.data import LoadedEventSequence, NPZEventDataset
from huav.training.engine import _metrics_at_threshold, build_model
from huav.training.losses import EventFocalDiceLoss
from huav.training.runtime import prepare_device
from huav.training.sampling import sample_event_indices

EXPERIMENT_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_files(value: str) -> tuple[str, ...]:
    files = tuple(item.strip() for item in value.split(",") if item.strip())
    if not files:
        raise ValueError("at least one training file is required")
    if len(set(files)) != len(files):
        raise ValueError("training files must be unique")
    if any(Path(name).name != name or not name.endswith(".npz") for name in files):
        raise ValueError("training files must be plain NPZ filenames")
    return files


def capacity_config(base: HUAVConfig, *, seed: int) -> HUAVConfig:
    """Remove regularization and make the classification loss see all events."""
    return replace(
        base,
        model=replace(base.model, dropout=0.0, direction_dropout=0.0),
        training=replace(
            base.training,
            device="cuda",
            learning_rate=7e-4,
            weight_decay=0.0,
            seed=seed,
            background_ratio=200,
            hard_negative_fraction=0.0,
            dense_sequence_weight=1.0,
            gradient_clip_norm=5.0,
            mixed_precision=True,
            mixed_precision_dtype="bfloat16",
            num_threads=8,
        ),
        threshold_augmentation=replace(
            base.threshold_augmentation,
            enabled=False,
        ),
        evaluation=replace(base.evaluation, optimize_threshold=False),
    ).validate()


def cosine_rate(step: int, maximum_steps: int, peak: float) -> float:
    if step <= 0 or maximum_steps <= 0 or step > maximum_steps:
        raise ValueError("step must be within the positive training range")
    progress = (step - 1) / max(1, maximum_steps - 1)
    multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(peak * multiplier)


def decision_from_evaluation(
    evaluation: dict[str, Any],
    *,
    min_event_iou: float,
    min_event_pd: float,
    min_object_pd: float,
    min_file_event_iou: float,
    min_file_object_pd: float,
) -> dict[str, Any]:
    aggregate = evaluation["aggregate"]
    per_file = evaluation["per_file"]
    event_iou = float(aggregate["event"]["iou"])
    event_pd = float(aggregate["event"]["pd"])
    object_pd = float(aggregate["ev_uav_compatible"]["pd"])
    worst_file_event_iou = min(
        float(metrics["event"]["iou"]) for metrics in per_file.values()
    )
    worst_file_object_pd = min(
        float(metrics["ev_uav_compatible"]["pd"])
        for metrics in per_file.values()
    )
    checks = {
        "event_iou": event_iou >= min_event_iou,
        "event_pd": event_pd >= min_event_pd,
        "object_pd": object_pd >= min_object_pd,
        "worst_file_event_iou": worst_file_event_iou >= min_file_event_iou,
        "worst_file_object_pd": worst_file_object_pd >= min_file_object_pd,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "event_iou": event_iou,
            "event_pd": event_pd,
            "object_pd": object_pd,
            "worst_file_event_iou": worst_file_event_iou,
            "worst_file_object_pd": worst_file_object_pd,
        },
        "required": {
            "event_iou": min_event_iou,
            "event_pd": min_event_pd,
            "object_pd": min_object_pd,
            "worst_file_event_iou": min_file_event_iou,
            "worst_file_object_pd": min_file_object_pd,
        },
    }


def _selected_dataset(
    config: HUAVConfig,
    data_root: Path,
    filenames: Iterable[str],
) -> NPZEventDataset:
    dataset = NPZEventDataset(
        data_root / "train",
        width=config.resolution.width,
        height=config.resolution.height,
        duration_ms=config.dataset.duration_ms,
    )
    requested = tuple(filenames)
    by_name = {path.name: path for path in dataset.files}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise FileNotFoundError(f"missing dense training file(s): {', '.join(missing)}")
    dataset.files = [by_name[name] for name in requested]
    return dataset


@torch.no_grad()
def _infer(
    model: torch.nn.Module,
    dataset: NPZEventDataset,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> list[tuple[torch.Tensor, LoadedEventSequence]]:
    model.eval()
    results: list[tuple[torch.Tensor, LoadedEventSequence]] = []
    for index in range(len(dataset)):
        sequence = dataset[index]
        events = sequence.events.to(device)
        batch_indices = torch.zeros(len(events), dtype=torch.long, device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            logits = model(events, batch_indices)["event_logits"]
        results.append((logits.detach().cpu(), sequence))
    return results


def _event_metrics_at_threshold(
    sequence_results: list[tuple[torch.Tensor, LoadedEventSequence]],
    threshold: float,
) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for logits, sequence in sequence_results:
        prediction = torch.sigmoid(logits) >= threshold
        target = sequence.labels > 0.5
        tp += int(torch.sum(prediction & target).item())
        fp += int(torch.sum(prediction & ~target).item())
        fn += int(torch.sum(~prediction & target).item())
        tn += int(torch.sum(~prediction & ~target).item())
    union = tp + fp + fn
    positives = tp + fn
    return {
        "iou": float(tp / union) if union else 0.0,
        "pd": float(tp / positives) if positives else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "event_count": tp + fp + fn + tn,
    }


def _best_event_threshold(
    sequence_results: list[tuple[torch.Tensor, LoadedEventSequence]],
    *,
    steps: int,
) -> tuple[float, dict[str, float | int]]:
    best_threshold = 0.5
    best_metrics: dict[str, float | int] | None = None
    for candidate in torch.linspace(0.01, 0.99, steps=steps).tolist():
        metrics = _event_metrics_at_threshold(sequence_results, float(candidate))
        quality = (
            float(metrics["iou"]),
            float(metrics["pd"]),
            -abs(float(candidate) - 0.5),
        )
        if best_metrics is None:
            best_threshold = float(candidate)
            best_metrics = metrics
            best_quality = quality
        elif quality > best_quality:
            best_threshold = float(candidate)
            best_metrics = metrics
            best_quality = quality
    if best_metrics is None:
        raise RuntimeError("event threshold search produced no result")
    return best_threshold, best_metrics


def _full_evaluation(
    model: torch.nn.Module,
    dataset: NPZEventDataset,
    config: HUAVConfig,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    results = _infer(model, dataset, device=device, amp_enabled=amp_enabled)
    threshold, _ = _best_event_threshold(results, steps=99)
    aggregate = _metrics_at_threshold(results, config, threshold)
    per_file = {
        sequence.source_path.name: _metrics_at_threshold(
            [(logits, sequence)], config, threshold
        )
        for logits, sequence in results
    }
    return {
        "threshold_selection": "maximum aggregate training event IoU",
        "threshold": threshold,
        "aggregate": aggregate,
        "per_file": per_file,
    }


def _reproducible_evaluation(
    model: torch.nn.Module,
    dataset: NPZEventDataset,
    config: HUAVConfig,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Use FP32 so CUDA reduction noise cannot move threshold-boundary events."""
    return _full_evaluation(
        model,
        dataset,
        config,
        device=device,
        amp_enabled=False,
    )


def _criterion(config: HUAVConfig) -> EventFocalDiceLoss:
    return EventFocalDiceLoss(
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma,
        lambda_dice=config.loss.lambda_dice,
        lambda_object_frame=config.loss.lambda_object_frame,
        object_frame_logit_margin=config.loss.object_frame_logit_margin,
        lambda_false_alarm=config.loss.lambda_false_alarm,
        false_alarm_logit_margin=config.loss.false_alarm_logit_margin,
        false_alarm_topk_per_frame=config.loss.false_alarm_topk_per_frame,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    files = parse_files(args.files)
    base = load_config(args.base_config)
    config = capacity_config(base, seed=args.seed)
    data_root = Path(args.data_root).resolve()
    dataset = _selected_dataset(config, data_root, files)
    runtime = prepare_device(args.device, num_threads=config.training.num_threads)
    device = runtime.device
    amp_enabled = bool(config.training.mixed_precision and device.type == "cuda")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    configuration = _jsonable(asdict(config))
    configuration["dataset"]["root"] = str(data_root)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(configuration, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(
        output_dir / "run_spec.json",
        {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "files": files,
            "maximum_steps": args.max_steps,
            "evaluation_interval": args.eval_interval,
            "seed": args.seed,
            "device": args.device,
            "runtime": runtime.metadata,
        },
    )

    model = build_model(config).to(device)
    criterion = _criterion(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_iou = -1.0
    best_step = 0
    success_streak = 0
    started = time.perf_counter()
    interval_losses: list[float] = []
    interval_gradients: list[float] = []
    completed_steps = 0

    for step in range(1, args.max_steps + 1):
        completed_steps = step
        model.train()
        sequence = dataset[(step - 1) % len(dataset)]
        events = sequence.events.to(device)
        labels = sequence.labels.to(device)
        object_ids = sequence.object_ids.to(device)
        timestamps_ms = sequence.timestamps_ms.to(device)
        batch_indices = torch.zeros(len(events), dtype=torch.long, device=device)
        learning_rate = cosine_rate(step, args.max_steps, config.training.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            output = model(events, batch_indices)
            query_indices = sample_event_indices(
                labels,
                logits=output["event_logits"],
                background_ratio=config.training.background_ratio,
                hard_negative_fraction=config.training.hard_negative_fraction,
            )
            if len(query_indices) != sequence.length:
                raise RuntimeError(
                    f"{sequence.source_path.name}: capacity loss did not cover all events"
                )
            losses = criterion(
                output["event_logits"][query_indices],
                labels[query_indices],
                batch_indices[query_indices],
                object_ids=object_ids[query_indices],
                timestamps_ms=timestamps_ms[query_indices],
                frame_ms=config.evaluation.frame_ms,
                full_logits=output["event_logits"],
                full_targets=labels,
                full_batch_indices=batch_indices,
                full_timestamps_ms=timestamps_ms,
                full_coordinates=events[:, :2],
            )
        if not torch.isfinite(losses["total"]):
            raise RuntimeError(f"non-finite loss at step {step}")
        losses["total"].backward()
        gradient_norm = clip_grad_norm_(
            model.parameters(), config.training.gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        interval_losses.append(float(losses["total"].detach().item()))
        interval_gradients.append(float(gradient_norm.detach().item()))

        should_evaluate = step % args.eval_interval == 0 or step == args.max_steps
        if not should_evaluate:
            continue
        results = _infer(model, dataset, device=device, amp_enabled=amp_enabled)
        threshold, event = _best_event_threshold(results, steps=41)
        improved = float(event["iou"]) > best_iou
        if improved:
            best_iou = float(event["iou"])
            best_step = step
            torch.save(
                {
                    "schema_version": EXPERIMENT_SCHEMA_VERSION,
                    "experiment": "dense_fit_capacity",
                    "model_state": model.state_dict(),
                    "step": step,
                    "files": files,
                    "threshold": threshold,
                    "event": event,
                },
                output_dir / "best_fit.pt",
            )
        gate_candidate = (
            float(event["iou"]) >= args.min_event_iou
            and float(event["pd"]) >= args.min_event_pd
        )
        decision: dict[str, Any] | None = None
        if gate_candidate:
            evaluation = {
                "threshold": threshold,
                "aggregate": _metrics_at_threshold(results, config, threshold),
                "per_file": {
                    sequence.source_path.name: _metrics_at_threshold(
                        [(logits, sequence)], config, threshold
                    )
                    for logits, sequence in results
                },
            }
            decision = decision_from_evaluation(
                evaluation,
                min_event_iou=args.min_event_iou,
                min_event_pd=args.min_event_pd,
                min_object_pd=args.min_object_pd,
                min_file_event_iou=args.min_file_event_iou,
                min_file_object_pd=args.min_file_object_pd,
            )
        success_streak = success_streak + 1 if decision and decision["passed"] else 0
        record = {
            "step": step,
            "file": sequence.source_path.name,
            "learning_rate": learning_rate,
            "mean_loss": sum(interval_losses) / len(interval_losses),
            "gradient_norm_mean": sum(interval_gradients) / len(interval_gradients),
            "gradient_norm_max": max(interval_gradients),
            "gradient_clipped_fraction": sum(
                value > config.training.gradient_clip_norm
                for value in interval_gradients
            )
            / len(interval_gradients),
            "threshold": threshold,
            "event": event,
            "best_event_iou": best_iou,
            "success_streak": success_streak,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        _write_json(output_dir / "history.json", history)
        print(
            f"DENSE_FIT step={step}/{args.max_steps} "
            f"iou={float(event['iou']):.6f} pd={float(event['pd']):.6f} "
            f"threshold={threshold:.3f} best={best_iou:.6f} "
            f"streak={success_streak}",
            flush=True,
        )
        interval_losses.clear()
        interval_gradients.clear()
        if success_streak >= 2:
            break

    checkpoint = torch.load(output_dir / "best_fit.pt", map_location=device, weights_only=True)
    best_model = build_model(config).to(device)
    best_model.load_state_dict(checkpoint["model_state"])
    evaluation = _reproducible_evaluation(
        best_model,
        dataset,
        config,
        device=device,
    )
    _write_json(output_dir / "best_evaluation.json", evaluation)
    independent_model = build_model(config).to(device)
    independent_model.load_state_dict(checkpoint["model_state"])
    independent_evaluation = _reproducible_evaluation(
        independent_model,
        dataset,
        config,
        device=device,
    )
    _write_json(output_dir / "independent_evaluation.json", independent_evaluation)
    if evaluation != independent_evaluation:
        raise RuntimeError("independent dense-fit evaluation did not reproduce")
    decision = decision_from_evaluation(
        evaluation,
        min_event_iou=args.min_event_iou,
        min_event_pd=args.min_event_pd,
        min_object_pd=args.min_object_pd,
        min_file_event_iou=args.min_file_event_iou,
        min_file_object_pd=args.min_file_object_pd,
    )
    decision.update(
        {
            "files": files,
            "completed_steps": completed_steps,
            "best_step": best_step,
            "best_event_iou_during_training": best_iou,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "independent_evaluation_exact": True,
        }
    )
    _write_json(output_dir / "decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--files", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--eval-interval", type=int, required=True)
    parser.add_argument("--min-event-iou", type=float, required=True)
    parser.add_argument("--min-event-pd", type=float, required=True)
    parser.add_argument("--min-object-pd", type=float, required=True)
    parser.add_argument("--min-file-event-iou", type=float, required=True)
    parser.add_argument("--min-file-object-pd", type=float, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_steps <= 0 or args.eval_interval <= 0:
        raise SystemExit("training and evaluation step counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
