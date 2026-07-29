"""Gate adaptive-threshold training on HUAV's complete largest sequence."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from huav.config import load_config
from huav.data import NPZEventDataset
from huav.training.engine import (
    _amp_is_enabled,
    _autocast_dtype,
    _configure_training_stage,
    _criterion,
    _dataset_manifest,
    _dense_sequence_loss_weight,
    _initialize_detector_weights,
    _manifest_identity_digest,
    _threshold_loss_inputs,
    build_model,
)
from huav.training.runtime import prepare_device
from huav.training.sampling import sample_event_indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resource-check adaptive calibration with a complete HUAV "
            "sequence before smoke or full training"
        )
    )
    parser.add_argument(
        "--config",
        default="configs/huav_adaptive_hierarchical_autodl.yaml",
        help="adaptive calibrator-stage configuration",
    )
    parser.add_argument(
        "--initialize-detector",
        required=True,
        help="schema 4/5 non-adaptive detector checkpoint",
    )
    parser.add_argument(
        "--sequence",
        default="train_094.npz",
        help="NPZ filename under the configured train split",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def _gradient_summary(model: torch.nn.Module) -> dict[str, Any]:
    trainable_parameter_count = 0
    gradient_parameter_count = 0
    finite_gradients = True
    gradient_names: list[str] = []
    parameters_without_grad: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_parameter_count += parameter.numel()
        if parameter.grad is None:
            parameters_without_grad.append(name)
            continue
        gradient_parameter_count += parameter.numel()
        gradient_names.append(name)
        finite_gradients = finite_gradients and bool(
            torch.isfinite(parameter.grad).all()
        )
    finite_gradients = bool(
        finite_gradients
        and gradient_parameter_count > 0
    )
    return {
        "trainable_parameter_count": trainable_parameter_count,
        "gradient_parameter_count": gradient_parameter_count,
        "finite_gradients": finite_gradients,
        "gradient_names": gradient_names,
        "parameters_without_grad": parameters_without_grad,
    }


def _validate_gradient_gate(summary: dict[str, Any]) -> None:
    missing = summary["parameters_without_grad"]
    if missing:
        raise ValueError(
            f"calibrator parameters are missing gradients: {missing}"
        )
    if (
        summary["gradient_parameter_count"]
        != summary["trainable_parameter_count"]
    ):
        raise ValueError(
            "gradient parameter count does not match trainable parameter count"
        )
    if not summary["finite_gradients"]:
        raise ValueError("calibrator gradients contain non-finite values")
    invalid_names = [
        name
        for name in summary["gradient_names"]
        if not name.startswith("calibrator.")
    ]
    if invalid_names:
        raise ValueError(
            f"gradients must belong only to calibrator parameters: {invalid_names}"
        )


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _find_sequence(
    dataset: NPZEventDataset,
    sequence_name: str,
    parser: argparse.ArgumentParser,
) -> Any:
    matching = [
        index
        for index, path in enumerate(dataset.files)
        if path.name == sequence_name
    ]
    if len(matching) != 1:
        parser.error(
            f"expected exactly one {sequence_name} under {dataset.root}"
        )
    return dataset[matching[0]]


def run_check(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    if config.training.training_stage != "calibrator":
        raise ValueError(
            "largest-sequence initialization gate requires training_stage=calibrator"
        )
    if config.model.adaptive_threshold_mode == "none":
        raise ValueError(
            "largest-sequence initialization gate requires adaptive thresholding"
        )
    runtime = prepare_device(
        arguments.device,
        num_threads=config.training.num_threads,
    )
    device = runtime.device
    _seed_everything(config.training.seed, device)
    train_dataset = NPZEventDataset(
        Path(config.dataset.root) / "train",
        width=config.resolution.width,
        height=config.resolution.height,
        duration_ms=config.dataset.duration_ms,
    )
    validation_dataset = NPZEventDataset(
        Path(config.dataset.root) / "val",
        width=config.resolution.width,
        height=config.resolution.height,
        duration_ms=config.dataset.duration_ms,
    )
    sequence = _find_sequence(
        train_dataset,
        arguments.sequence,
        build_parser(),
    )
    validation_manifest_digest = _manifest_identity_digest(
        _dataset_manifest(validation_dataset)
    )
    model = build_model(config).to(device)
    initialization = _initialize_detector_weights(
        model,
        config,
        Path(arguments.initialize_detector),
        validation_manifest_digest,
        device=device,
    )
    _configure_training_stage(model, "calibrator")
    detector_frozen = all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("calibrator.")
    )
    calibrator_only_trainable = all(
        name.startswith("calibrator.")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not detector_frozen or not calibrator_only_trainable:
        raise RuntimeError(
            "calibrator resource gate must freeze every detector parameter"
        )

    criterion = _criterion(config)
    events = sequence.events.to(device)
    labels = sequence.labels.to(device)
    object_ids = sequence.object_ids.to(device)
    timestamps_ms = sequence.timestamps_ms.to(device)
    batch_indices = torch.zeros(
        len(events), dtype=torch.long, device=device
    )
    sampling_generator = torch.Generator(device=device.type).manual_seed(
        config.training.seed
    )
    amp_enabled = _amp_is_enabled(config.training.mixed_precision, device)
    amp_dtype = _autocast_dtype(config.training.mixed_precision_dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = perf_counter()

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        output = model(events, batch_indices)
        if output["event_logits"].numel() != sequence.length:
            raise RuntimeError(
                "model output must contain one logit per original event"
            )
        query_indices = sample_event_indices(
            labels,
            logits=output["event_logits"],
            background_ratio=config.training.background_ratio,
            hard_negative_fraction=config.training.hard_negative_fraction,
            generator=sampling_generator,
            max_background_without_positive=(
                config.training.max_background_without_positive
            ),
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
            **_threshold_loss_inputs(output),
        )
        sequence_weight = _dense_sequence_loss_weight(
            events,
            labels,
            config,
        )
        losses["total"] = losses["total"] * sequence_weight
    finite_loss = bool(torch.isfinite(losses["total"]))
    if not finite_loss:
        raise RuntimeError("largest-sequence total loss is not finite")
    losses["total"].backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = 0
        peak_reserved = 0
    elapsed_seconds = perf_counter() - started
    gradients = _gradient_summary(model)
    _validate_gradient_gate(gradients)
    if any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("calibrator.")
    ):
        raise RuntimeError("frozen detector received gradients")

    result = {
        "sequence": str(sequence.source_path.resolve()),
        "event_count": sequence.length,
        "all_event_logits_count": int(output["event_logits"].numel()),
        "token_count": int(output["token_features"].shape[0]),
        "group_count": int(output["threshold_group_coords"].shape[0]),
        "neighbor_count": int(output["threshold_neighbor_edges"].shape[1]),
        "loss_query_count": int(len(query_indices)),
        "sequence_loss_weight": sequence_weight,
        "loss": float(losses["total"].detach().float().item()),
        "losses": {
            name: float(value.detach().float().item())
            for name, value in losses.items()
        },
        "finite_loss": finite_loss,
        **gradients,
        "detector_frozen": detector_frozen,
        "calibrator_only_gradients": all(
            name.startswith("calibrator.")
            for name in gradients["gradient_names"]
        ),
        "max_memory_allocated_bytes": peak_allocated,
        "max_memory_reserved_bytes": peak_reserved,
        "elapsed_seconds": elapsed_seconds,
        "source_checkpoint": initialization["source_checkpoint"],
        "source_checkpoint_sha256": initialization[
            "source_checkpoint_sha256"
        ],
        "runtime": runtime.metadata,
    }
    if not math.isfinite(result["elapsed_seconds"]):
        raise RuntimeError("invalid runtime measurement")
    return result


def main() -> None:
    arguments = build_parser().parse_args()
    result = run_check(arguments)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
