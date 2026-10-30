"""直接评估当前神经网络，保存逐事件概率和掩码；测试集使用验证阈值。"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from traceformer.config import load_config
from traceformer.data import NPZEventDataset
from traceformer.experiments.ablation import parse_ablation
from traceformer.postprocessing.cache import infer_event_output
from traceformer.training.engine import (
    _dataset,
    _dataset_manifest,
    _file_sha256,
    _manifest_identity_digest,
    _metrics_at_threshold,
    _validate_checkpoint_contract,
    _write_json,
    build_model,
    load_inference_model,
)


def verify_raw_score(original: dict, independent: dict) -> None:
    """使用另一套原始事件计数实现交叉核对，不重新选择阈值。"""
    counts = original["counts"]
    expected = {
        "intersection": independent["tp"],
        "union": independent["tp"] + independent["fp"] + independent["fn"],
        "true_positive": independent["tp"],
        "gt_positive": independent["tp"] + independent["fn"],
        "detected_objects": independent["detected_objects"], "objects": independent["objects"],
        "false_alarms": independent["false_components"], "frames": independent["frames"],
    }
    if counts != expected or any(abs(original[k] - independent[k]) > 1e-12
                                 for k in ("score", "iou", "seg_acc", "pd", "fa")):
        raise ValueError("independent raw-event scoring differs")


def evaluate(args):
    config = load_config(args.config)
    torch.set_num_threads(args.num_threads or config.training.num_threads)
    if args.device == "cuda":
        # 独立预测固定FP32，避免主机默认TF32设置改变对照协议。
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    ablation_path = getattr(args, "ablation", None)
    if ablation_path is None:
        model, checkpoint = load_inference_model(
            config, args.checkpoint, device=torch.device(args.device)
        )
    else:
        # 实验入口要求显式契约；正式模型加载器仍拒绝所有消融权重。
        ablation = parse_ablation(json.loads(ablation_path.read_text(encoding="utf-8")))
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        _validate_checkpoint_contract(checkpoint, config, ablation=ablation)
        model = build_model(config, ablation=ablation).to(args.device).eval()
        model.load_state_dict(checkpoint["model_state"], strict=True)
    identity = _manifest_identity_digest(_dataset_manifest(_dataset(config, "val")))
    if identity != checkpoint["validation_manifest_sha256"]:
        raise ValueError("checkpoint validation data identity differs")
    threshold = float(checkpoint["decision_threshold"])
    dataset = NPZEventDataset(
        args.dataset_root / args.split,
        width=config.resolution.width,
        height=config.resolution.height,
        duration_ms=config.dataset.duration_ms,
        limit_files=getattr(args, "max_files", None),
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    probabilities = output / "probabilities"
    masks = output / "predictions"
    probabilities.mkdir(exist_ok=True)
    masks.mkdir(exist_ok=True)
    results, records = [], {}
    probability_scores, target_values = [], []
    if args.rescore_only:
        expected = {p.stem for p in dataset.files}
        if {p.stem for p in probabilities.glob("*.pt")} != expected:
            raise ValueError("saved prediction file set differs from dataset")
    with torch.inference_mode():
        for sequence in dataset:
            name = sequence.source_path.name
            path = probabilities / f"{sequence.source_path.stem}.pt"
            source_hash = _file_sha256(sequence.source_path)
            started = time.perf_counter()
            if args.rescore_only:
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if saved["source_sha256"] != source_hash:
                    raise ValueError("prediction source hash differs")
                probability = saved["event_probabilities"]
            else:
                # 标签与目标 ID 只送入评分器；网络推理显式使用清零的副本。
                inference_sequence = replace(
                    sequence,
                    labels=torch.zeros_like(sequence.labels),
                    object_ids=torch.zeros_like(sequence.object_ids),
                )
                prediction = infer_event_output(model, config, inference_sequence)
                probability = prediction["event_logits"].sigmoid().cpu()
                torch.save({"event_probabilities": probability, "source_sha256": source_hash}, path)
            if probability.shape != (sequence.length,) or not torch.isfinite(probability).all():
                raise ValueError("invalid aligned probabilities")
            if bool(((probability < 0) | (probability > 1)).any()):
                raise ValueError("probabilities must be in [0,1]")
            probability_scores.append(probability.numpy())
            target_values.append(sequence.labels.numpy())
            prediction = probability >= threshold
            mask_path = masks / name
            if args.rescore_only:
                with np.load(mask_path, allow_pickle=False) as saved_mask:
                    if not np.array_equal(saved_mask["prediction"], prediction.numpy()):
                        raise ValueError("saved mask differs from frozen-threshold probabilities")
            else:
                np.savez_compressed(mask_path, prediction=prediction.numpy())
            # 二值 logits 避免概率转回 logits 后的阈值舍入误差。
            results.append((torch.where(prediction, 20.0, -20.0), sequence))
            records[name] = {
                "events": sequence.length,
                "source_sha256": source_hash,
                "prediction_sha256": _file_sha256(mask_path),
                "seconds": time.perf_counter() - started,
            }
            print(f"{args.split} {name}: {sequence.length:,} events", flush=True)
    report = _metrics_at_threshold(results, config, 0.5)
    report["average_precision"] = float(average_precision_score(
        np.concatenate(target_values), np.concatenate(probability_scores)))
    report.update(
        threshold=threshold,
        split=args.split,
        sequences=len(dataset),
        postprocessing=False,
        test_threshold_selection=False,
        checkpoint_sha256=_file_sha256(args.checkpoint),
        config_sha256=_file_sha256(args.config),
        adapter_sha256=_file_sha256(Path(__file__)),
        ablation=checkpoint.get("ablation"),
        inference_annotations="zeroed labels and object IDs",
        architecture=config.model.architecture,
        parameters=sum(p.numel() for p in model.parameters()),
        files=records,
        validation_identity=identity,
    )
    report["per_sequence"] = {
        sequence.source_path.name: _metrics_at_threshold([(logits, sequence)], config, 0.5)[
            "competition"
        ]
        for logits, sequence in results
    }
    report.pop("validation_files")
    destination = output / ("independent_report.json" if args.rescore_only else "report.json")
    if args.rescore_only:
        before = json.loads((output / "report.json").read_text(encoding="utf-8"))
        if (
            before["competition"] != report["competition"]
            or before["checkpoint_sha256"] != report["checkpoint_sha256"]
            or before["threshold"] != report["threshold"]
            or before["ablation"] != report["ablation"]
            or before["split"] != report["split"]
        ):
            raise ValueError("independent scoring differs from original report")
    _write_json(destination, report)
    print(json.dumps(report["competition"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "dataset-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--rescore-only", action="store_true")
    parser.add_argument("--ablation", type=Path, help="显式消融契约 ablation.json")
    parser.add_argument("--max-files", type=int, help="仅用于smoke的文件数限制，正式评估不设置")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
