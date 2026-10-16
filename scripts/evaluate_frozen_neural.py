"""使用匹配 schema 的归档源码，严格复评发布主网络或参考网络。"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from evaluate_external_baseline import probability_metrics, select_threshold

from traceformer.config import load_config as scoring_config
from traceformer.postprocessing.contracts import EventCache
from traceformer.postprocessing.submission import write_prediction_npz
from traceformer.training.engine import _file_sha256, _write_json


@torch.no_grad()
def evaluate(args):
    sys.path.insert(0, str(args.source / "src"))
    legacy_config = importlib.import_module("huav.config")
    engine = importlib.import_module("huav.training.engine")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    original = json.loads(json.dumps(raw))
    # 仅删除旧解析器没有的训练控制字段；推理结构和门控配置必须完整匹配。
    omitted = {k: v for k, v in raw["training"].items()
               if k not in legacy_config.TrainingConfig.__dataclass_fields__}
    raw["training"] = {k: v for k, v in raw["training"].items() if k not in omitted}
    raw["dataset"]["root"] = str(args.dataset_root)
    raw["training"]["output_root"] = str(output)
    raw["training"]["num_threads"] = args.num_threads
    runtime_config = output / "config.yaml"
    runtime_config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = legacy_config.load_config(runtime_config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != engine.CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("legacy source/checkpoint schema differs")
    for field in ("model", "resolution", "sparse_mixture"):
        checkpoint_field = {"model": "model_config", "resolution": "resolution",
                            "sparse_mixture": "sparse_mixture_config"}[field]
        if field in original and checkpoint.get(checkpoint_field) != engine._serializable(asdict(getattr(config, field))):
            raise ValueError(f"checkpoint {field} configuration differs")
    # 测试推理仍核对训练时的验证身份，但不使用测试标签选择阈值。
    validation = engine._dataset(config, "val")
    identity = engine._manifest_identity_digest(engine._dataset_manifest(validation))
    if identity != checkpoint["validation_manifest_sha256"]:
        raise ValueError("checkpoint validation data identity differs")
    dataset = validation if args.split == "val" else engine._dataset(config, args.split)
    if len(validation) != 24 or len(dataset) != 24:
        raise ValueError("requires all 24 sequences in the selected partition")
    model = engine.build_model(config).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if args.audit_only:
        _write_json(output / "contract_audit.json", {"status": "contract_verified",
            "schema": checkpoint["schema_version"], "validation_identity": identity,
            "parameters": sum(p.numel() for p in model.parameters()),
            "omitted_training_fields": omitted, "checkpoint_sha256": _file_sha256(args.checkpoint)})
        print("Frozen checkpoint/config/data/source contract verified", flush=True)
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime unavailable")
    torch.set_num_threads(args.num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model.to(device)
    threshold = float(checkpoint["decision_threshold"])
    results, records = [], {}
    for sequence in dataset:
        events = sequence.events.to(device)
        batch = torch.zeros(len(events), dtype=torch.long, device=device)
        # 原始标签只留给评分器；推理序列显式清空标签与目标身份。
        inference_sequence = replace(sequence, labels=torch.zeros_like(sequence.labels),
                                     object_ids=torch.zeros_like(sequence.object_ids))
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        if checkpoint["schema_version"] == 5:
            prediction, sparse = engine._forward_with_sparse_mixture(
                model, inference_sequence, events, batch, config, mixture_weight=1.0)
        else:
            prediction, sparse = model(events, batch), None
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        probability = torch.sigmoid(prediction["event_logits"]).cpu()
        if probability.shape != (sequence.length,) or not torch.isfinite(probability).all():
            raise ValueError("invalid full-event output")
        name = sequence.source_path.name
        source_hash = _file_sha256(sequence.source_path)
        torch.save({"event_probabilities": probability, "source_sha256": source_hash},
                   output / f"{sequence.source_path.stem}.pt")
        write_prediction_npz(output / "predictions" / name, probability.numpy() >= threshold)
        if checkpoint["schema_version"] == 5:
            with np.load(sequence.source_path, allow_pickle=False) as data:
                cache = EventCache(name=name, locations=data["ev_loc"].astype(np.int64),
                    raw_features=data["evs_norm"][:, :4].astype(np.float32),
                    base_probability=probability.numpy(),
                    token_features=prediction["token_features"].cpu().numpy(),
                    event_token_indices=prediction["event_token_indices"].cpu().numpy(),
                    prediction=probability.numpy() >= threshold)
            (output / "cache").mkdir(exist_ok=True)
            cache.save(output / "cache" / name, include_labels=False)
        records[name] = {"events": sequence.length, "source_sha256": source_hash,
                         "sparse_view": sparse is not None, "forward_seconds": elapsed,
                         "prediction_sha256": _file_sha256(output / "predictions" / name)}
        results.append((probability, sequence))
        print(f"schema{checkpoint['schema_version']} {name}: {sequence.length} events", flush=True)
    score_config = scoring_config(args.scoring_config)
    report = {"status": "complete", "checkpoint_sha256": _file_sha256(args.checkpoint),
        "original_config_sha256": _file_sha256(args.config),
        "runtime_config_sha256": _file_sha256(runtime_config), "omitted_training_fields": omitted,
        "legacy_source": str(args.source), "source_engine_sha256": _file_sha256(Path(engine.__file__)),
        "schema": checkpoint["schema_version"], "parameters": sum(p.numel() for p in model.parameters()),
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                        "threads": args.num_threads, "precision": "FP32", "tf32": False},
        "split": args.split, "inference_annotations": "zeroed labels and object IDs",
        "test_threshold_selection": False,
        "validation_identity": identity,
        "adapter_sha256": _file_sha256(Path(__file__)),
        "release_threshold": probability_metrics(results, score_config, threshold),
        "files": records,
        "timing_note": "Single forward excluding preprocessing, not an end-to-end speed benchmark."}
    if args.split == "val":
        report["fixed_threshold"] = probability_metrics(results, score_config, 0.5)
        report["selected_threshold"] = select_threshold(results, score_config)
    _write_json(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "config", "checkpoint", "dataset-root", "scoring-config", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--num-threads", type=int, default=8)
    evaluate(parser.parse_args())
