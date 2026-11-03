"""纯神经推理与独立原始事件评分，不依赖外部后处理或验证目录。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from traceformer.data import NPZEventDataset
from traceformer.scoring import BinaryCounts, score_binary
from traceformer.training.engine import (
    _dataset_manifest,
    _file_sha256,
    _manifest_identity_digest,
    _metrics_at_threshold,
    _validate_checkpoint_contract,
    _write_json,
)
from traceformer.training.runtime import prepare_device
from traceformer.variants import PaperSpec, build_model


def load_model(config, checkpoint_path, *, device="cpu"):
    config.validate()
    runtime = prepare_device(device, num_threads=config.training.num_threads)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(checkpoint_path, map_location=runtime.device, weights_only=True)
    spec = PaperSpec(**checkpoint["ablation"])
    _validate_checkpoint_contract(checkpoint, config, ablation=spec)
    threshold = checkpoint["decision_threshold"]
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("invalid checkpoint threshold")
    model = build_model(config, ablation=spec).to(runtime.device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint


def run_predictions(config, checkpoint_path, input_dir, output_dir, *, device="cpu",
                    evaluate=False, split=None, predictions_dir=None):
    """保存原序概率及掩码；评分只接收标注，网络始终只接收四列事件。"""
    model, checkpoint = load_model(config, checkpoint_path, device=device)
    threshold = float(checkpoint["decision_threshold"])
    dataset = NPZEventDataset(input_dir, width=config.resolution.width,
                              height=config.resolution.height,
                              duration_ms=config.dataset.duration_ms,
                              require_annotations=evaluate)
    if evaluate and split == "val":
        identity = _manifest_identity_digest(_dataset_manifest(dataset))
        if identity != checkpoint["validation_manifest_sha256"]:
            raise ValueError("checkpoint validation dataset identity differs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions = output_dir / "predictions"
    if predictions_dir is None:
        predictions.mkdir()
    else:
        predictions_dir = Path(predictions_dir)
        if {p.name for p in predictions_dir.glob("*.npz")} != {p.name for p in dataset.files}:
            raise ValueError("saved prediction file set differs")
    rows, scores, targets, per_file = [], [], [], {}
    independent = BinaryCounts()
    with torch.inference_mode():
        for sequence in dataset:
            source_sha = _file_sha256(sequence.source_path)
            if predictions_dir is None:
                events = sequence.events.to(device)
                logits = model(events, torch.zeros(len(events), dtype=torch.long, device=device))["event_logits"]
                probability = logits.sigmoid().cpu()
                mask = probability >= threshold
                np.savez_compressed(predictions / sequence.source_path.name,
                                    event_probabilities=probability.numpy(), prediction=mask.numpy(),
                                    source_sha256=np.array(source_sha), threshold=np.array(threshold))
            else:
                with np.load(predictions_dir / sequence.source_path.name, allow_pickle=False) as saved:
                    if str(saved["source_sha256"]) != source_sha or float(saved["threshold"]) != threshold:
                        raise ValueError("prediction source/threshold differs")
                    probability = torch.from_numpy(saved["event_probabilities"].copy())
                    mask = probability >= threshold
                    if not np.array_equal(mask.numpy(), saved["prediction"]):
                        raise ValueError("saved mask differs from frozen-threshold probability")
            if (probability.shape != (sequence.length,) or not torch.isfinite(probability).all()
                    or bool(((probability < 0) | (probability > 1)).any())):
                raise ValueError("invalid event-aligned probabilities")
            per_file[sequence.source_path.name] = {"events": sequence.length, "source_sha256": source_sha}
            if evaluate:
                # 用已冻结的二值工作点评分，避免概率反变换造成边界舍入。
                binary_logits = torch.where(mask, 20.0, -20.0)
                rows.append((binary_logits, sequence))
                scores.append(probability.numpy())
                targets.append(sequence.labels.numpy())
                locations = np.column_stack((sequence.events[:, :2].numpy(), sequence.timestamps_ms.numpy()))
                counts = score_binary(locations, sequence.labels.numpy(), sequence.object_ids.numpy(),
                                      mask.numpy(), frame_ms=config.evaluation.frame_ms,
                                      object_correct_fraction=config.evaluation.object_correct_fraction)
                independent += counts
                per_file[sequence.source_path.name]["competition"] = _metrics_at_threshold(
                    [(binary_logits, sequence)], config, 0.5)["competition"]
            print(f"{sequence.source_path.name}: {sequence.length:,} events", flush=True)
    report = {"threshold": threshold, "checkpoint_sha256": _file_sha256(Path(checkpoint_path)),
              "ablation": checkpoint["ablation"], "split": split,
              "parameters": sum(p.numel() for p in model.parameters()), "files": per_file,
              "postprocessing": False, "test_threshold_selection": False,
              "device": device, "torch_version": torch.__version__}
    if evaluate:
        metrics = _metrics_at_threshold(rows, config, 0.5)
        report.update({key: metrics[key] for key in ("competition", "event", "ev_uav_compatible")})
        report["average_precision"] = float(average_precision_score(np.concatenate(targets), np.concatenate(scores)))
        verified = independent.metrics(width=config.resolution.width, height=config.resolution.height)
        counts = metrics["competition"]["counts"]
        expected = {"intersection": verified["tp"], "union": verified["tp"] + verified["fp"] + verified["fn"],
                    "true_positive": verified["tp"], "gt_positive": verified["tp"] + verified["fn"],
                    "detected_objects": verified["detected_objects"], "objects": verified["objects"],
                    "false_alarms": verified["false_components"], "frames": verified["frames"]}
        if counts != expected or any(abs(metrics["competition"][k] - verified[k]) > 1e-12
                                     for k in ("score", "iou", "seg_acc", "pd", "fa")):
            raise ValueError("independent official counts differ")
        report["independent_counts_verified"] = True
    _write_json(output_dir / "metrics.json", report)
    return report
