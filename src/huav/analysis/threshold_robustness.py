"""Evaluate checkpoints on deterministic high-trigger-threshold event views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from huav.config import HUAVConfig, load_config
from huav.data import LoadedEventSequence, NPZEventDataset, simulate_high_threshold_events
from huav.training.engine import _validate_checkpoint_contract, build_model
from huav.training.metrics import (
    EventMetricAccumulator,
    EVUAVMetricAccumulator,
    competition_metrics,
)
from huav.training.runtime import prepare_device


def _new_accumulators(
    config: HUAVConfig, threshold: float
) -> tuple[EventMetricAccumulator, EVUAVMetricAccumulator]:
    return (
        EventMetricAccumulator(threshold=threshold),
        EVUAVMetricAccumulator(
            width=config.resolution.width,
            height=config.resolution.height,
            frame_ms=config.evaluation.frame_ms,
            object_correct_fraction=config.evaluation.object_correct_fraction,
            threshold=threshold,
        ),
    )


def _computed_metrics(
    accumulators: tuple[EventMetricAccumulator, EVUAVMetricAccumulator],
    config: HUAVConfig,
) -> dict[str, Any]:
    event = accumulators[0].compute()
    compatible = accumulators[1].compute()
    return {
        "competition": competition_metrics(
            event,
            compatible,
            width=config.resolution.width,
            height=config.resolution.height,
        ),
        "event": event,
        "ev_uav_compatible": compatible,
    }


def _update_accumulators(
    accumulators: tuple[EventMetricAccumulator, EVUAVMetricAccumulator],
    logits: torch.Tensor,
    sequence: LoadedEventSequence,
    indices: torch.Tensor,
) -> None:
    labels = sequence.labels[indices]
    accumulators[0].update(logits, labels)
    accumulators[1].update(
        logits,
        labels,
        sequence.object_ids[indices],
        sequence.timestamps_ms[indices],
        sequence.events[indices, :2],
    )


@torch.no_grad()
def evaluate_threshold_robustness(
    config: HUAVConfig,
    checkpoint_path: str | Path,
    *,
    threshold_ratios: tuple[float, ...] = (1.5, 2.0),
    device: str = "cpu",
    files: tuple[str, ...] | None = ("val_023.npz",),
    progress: bool = False,
) -> dict[str, Any]:
    """Compare retained-stream and fixed-original-universe stress metrics."""

    config.validate()
    ratios = tuple(float(value) for value in threshold_ratios)
    if not ratios or len(set(ratios)) != len(ratios) or any(value <= 1 for value in ratios):
        raise ValueError("threshold ratios must be unique and greater than 1")
    runtime = prepare_device(device, num_threads=config.training.num_threads)
    torch_device = runtime.device
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=True)
    _validate_checkpoint_contract(checkpoint, config)

    model = build_model(config).to(torch_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    decision_threshold = float(
        checkpoint.get("decision_threshold", config.inference.sigmoid_threshold)
    )
    dataset = NPZEventDataset(
        Path(config.dataset.root) / "val",
        width=config.resolution.width,
        height=config.resolution.height,
        duration_ms=config.dataset.duration_ms,
    )
    requested = set(files or ())
    selected_indices = [
        index
        for index, path in enumerate(dataset.files)
        if not requested or path.name in requested
    ]
    selected_names = {dataset.files[index].name for index in selected_indices}
    missing = requested - selected_names
    if missing:
        raise ValueError(f"validation files not found: {sorted(missing)}")
    if not selected_indices:
        raise ValueError("no validation files selected")

    results: dict[str, Any] = {}
    for ratio in ratios:
        retained_accumulators = _new_accumulators(config, decision_threshold)
        fixed_accumulators = _new_accumulators(config, decision_threshold)
        counts = {
            "source_events": 0,
            "retained_events": 0,
            "source_positive_events": 0,
            "retained_positive_events": 0,
        }
        sequence_rows: list[dict[str, Any]] = []
        for offset, dataset_index in enumerate(selected_indices):
            sequence = dataset[dataset_index]
            view = simulate_high_threshold_events(
                sequence.events,
                width=config.resolution.width,
                height=config.resolution.height,
                threshold_ratio=ratio,
            )
            sparse_events = view.events.to(torch_device)
            sparse_logits = model(sparse_events)["event_logits"].detach().cpu()
            indices = view.source_indices.cpu()
            _update_accumulators(
                retained_accumulators, sparse_logits, sequence, indices
            )
            fixed_logits = torch.full(
                (sequence.length,), -100.0, dtype=sparse_logits.dtype
            )
            fixed_logits[indices] = sparse_logits
            all_indices = torch.arange(sequence.length, dtype=torch.long)
            _update_accumulators(
                fixed_accumulators, fixed_logits, sequence, all_indices
            )
            source_positive = int((sequence.labels > 0.5).sum().item())
            retained_positive = int((sequence.labels[indices] > 0.5).sum().item())
            counts["source_events"] += sequence.length
            counts["retained_events"] += int(len(indices))
            counts["source_positive_events"] += source_positive
            counts["retained_positive_events"] += retained_positive
            sequence_rows.append(
                {
                    "file": sequence.source_path.name,
                    "source_events": sequence.length,
                    "retained_events": int(len(indices)),
                    "event_retention": float(len(indices) / sequence.length),
                    "source_positive_events": source_positive,
                    "retained_positive_events": retained_positive,
                    "positive_event_retention": (
                        float(retained_positive / source_positive)
                        if source_positive
                        else 0.0
                    ),
                }
            )
            if progress:
                print(
                    f"  ratio {ratio:g} {offset + 1:03d}/{len(selected_indices):03d} "
                    f"{sequence.source_path.name}: {len(indices):,}/{sequence.length:,}",
                    flush=True,
                )
        results[f"{ratio:g}"] = {
            "counts": counts,
            "event_retention": float(
                counts["retained_events"] / counts["source_events"]
            ),
            "positive_event_retention": float(
                counts["retained_positive_events"]
                / counts["source_positive_events"]
            ),
            "retained_stream": _computed_metrics(retained_accumulators, config),
            "fixed_original_universe": _computed_metrics(fixed_accumulators, config),
            "sequences": sequence_rows,
        }

    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "threshold": decision_threshold,
        "device": torch_device.type,
        "runtime": runtime.metadata,
        "validation_files": [dataset.files[index].name for index in selected_indices],
        "threshold_ratios": list(ratios),
        "results": results,
        "metric_note": (
            "retained_stream scores only emitted events; fixed_original_universe "
            "scatters missing events to a negative logit and retains the original "
            "event/object-frame denominator"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a HUAV checkpoint under higher trigger thresholds"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-ratios", type=float, nargs="+", default=(1.5, 2.0))
    parser.add_argument("--files", nargs="*", default=("val_023.npz",))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    report = evaluate_threshold_robustness(
        load_config(arguments.config),
        arguments.checkpoint,
        threshold_ratios=tuple(arguments.threshold_ratios),
        device=arguments.device,
        files=tuple(arguments.files) if arguments.files else None,
        progress=not arguments.quiet,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
