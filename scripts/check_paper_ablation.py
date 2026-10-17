"""真实大文件消融资源预检；随机权重仅用于前向/反向检查。"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import psutil
import torch

from traceformer.config import load_config
from traceformer.data import NPZEventDataset
from traceformer.experiments.ablation import AblationSpec
from traceformer.training.engine import (
    _add_sparse_mixture_losses,
    _dense_sequence_loss_weight,
    _event_refinement_loss,
    _forward_with_sparse_mixture,
    _sparse_mixture_auxiliary_losses,
    _write_json,
    build_model,
)
from traceformer.training.losses import EventFocalDiceLoss
from traceformer.training.runtime import prepare_device
from traceformer.training.sampling import sample_event_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/traceformer_score_optimized.yaml")
    parser.add_argument("--backbone", choices=["cone"], default="cone")
    parser.add_argument("--stage", choices=["full", "sparse_mixture", "refinement"], default="full")
    parser.add_argument("--pooling", choices=["attention", "mean", "statistics"], default="attention")
    parser.add_argument("--sequence", default="train_096.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config = replace(config, training=replace(config.training, stage=args.stage))
    spec = AblationSpec(args.backbone, args.stage != "full",
                        "multiscale" if args.stage == "refinement" else "none", pooling=args.pooling)
    prepare_device("cpu", num_threads=config.training.num_threads)
    torch.manual_seed(config.training.seed)
    model = build_model(config, ablation=spec)
    model.configure_training_stage(args.stage)
    model.train()
    dataset = NPZEventDataset(Path(config.dataset.root) / "train",
                              width=config.resolution.width, height=config.resolution.height,
                              duration_ms=config.dataset.duration_ms)
    sequence = dataset[next(i for i, p in enumerate(dataset.files) if p.name == args.sequence)]
    print(f"preflight {args.backbone}/{args.stage} {args.sequence}: {sequence.length} events", flush=True)
    events, labels = sequence.events, sequence.labels
    batches = torch.zeros(sequence.length, dtype=torch.long)
    generator = torch.Generator().manual_seed(config.training.seed)
    criterion = EventFocalDiceLoss(**asdict(config.loss))
    started = time.perf_counter()
    output, view = _forward_with_sparse_mixture(model, sequence, events, batches, config, mixture_weight=1.0)
    queries = sample_event_indices(
        labels, logits=output["event_logits"], background_ratio=config.training.background_ratio,
        hard_negative_fraction=config.training.hard_negative_fraction, generator=generator,
        max_background_without_positive=config.training.max_background_without_positive,
    )
    losses = criterion(
        output["event_logits"][queries], labels[queries], batches[queries],
        object_ids=sequence.object_ids[queries], timestamps_ms=sequence.timestamps_ms[queries],
        frame_ms=config.evaluation.frame_ms, full_logits=output["event_logits"], full_targets=labels,
        full_batch_indices=batches, full_timestamps_ms=sequence.timestamps_ms, full_coordinates=events[:, :2],
    )
    refinement = _event_refinement_loss(output, labels, criterion, config)
    total = (losses["total"] + config.frame_refinement.loss_weight * refinement) * _dense_sequence_loss_weight(events, labels, config)
    auxiliary = _sparse_mixture_auxiliary_losses(
        output, labels, sequence.object_ids, sequence.timestamps_ms, events, batches, queries,
        criterion, config, sampling_generator=generator,
    )
    if view is not None:
        total = _add_sparse_mixture_losses(total, auxiliary, config)
    if not torch.isfinite(total):
        raise RuntimeError("nonfinite resource-check loss")
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("missing or nonfinite resource-check gradients")
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError("frozen parameter received gradient")
    if output["event_logits"].shape != (sequence.length,):
        raise RuntimeError("original event universe was changed")
    memory = psutil.Process().memory_info()
    report = {
        "status": "passed", "sequence": args.sequence, "stage": args.stage, "spec": spec.to_dict(),
        "events": sequence.length, "tokens": len(output["token_features"]),
        "loss": float(total.detach()), "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
