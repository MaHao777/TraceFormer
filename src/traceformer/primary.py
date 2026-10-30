"""用户指定的冻结主模型：网络、两套事件树、主轨迹与直接分支。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from traceformer.config import load_config
from traceformer.data.dataset import LoadedEventSequence
from traceformer.gpu.runtime import validate_backends
from traceformer.postprocessing.cache import infer_event_cache
from traceformer.postprocessing.contracts import sha256_file
from traceformer.postprocessing.direct_verifier import event_branches
from traceformer.postprocessing.learning import positive_prediction_threads, predict_event_scores
from traceformer.postprocessing.release import event_validation_gate, load_event_release
from traceformer.postprocessing.submission import audit_submission, write_prediction_npz
from traceformer.training.engine import load_inference_model

DEFAULT_MAIN_MANIFEST = Path(__file__).resolve().parents[2] / "configs/main_model.json"


def load_primary_manifest(path=DEFAULT_MAIN_MANIFEST):
    """保留真实门槛结果；用户指定的冻结选择与自动验收分别记录。"""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"best.pt", "config.yaml", "event_release/release_report.json", "validation.json"}
    if (manifest["schema_version"] != 1 or not required.issubset(manifest["files"])
            or manifest["selection"]["method"] != "user_requested"):
        raise ValueError("invalid primary model manifest")
    root = (path.parent / manifest["artifact_root"]).resolve()
    for name, expected in manifest["files"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"primary model hash differs: {name}")
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    gate = event_validation_gate(validation["event_release"], validation["original"])
    if gate != manifest["selection"]["validation_gate"]:
        raise ValueError("primary selection validation gate differs")
    if (validation["protocol"]["event_release_report_sha256"]
            != sha256_file(root / "event_release/release_report.json")):
        raise ValueError("primary validation release differs")
    load_event_release(root / "event_release", checkpoint=root / "best.pt",
                       config=root / "config.yaml", load_models=False)
    return root, manifest


def _inference_input(path, config):
    # 只读取推理字段；真实测试文件不需要提供label/name。
    with np.load(path, allow_pickle=False) as saved:
        raw = saved["ev"]
        source = {"locations": saved["ev_loc"].astype(np.int64),
                  "raw_features": saved["evs_norm"][:, :4].astype(np.float32)}
    timestamps = raw["t"].astype(np.float64)
    if (not len(raw) or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) < 0) or timestamps[0] < 0
            or timestamps[-1] > config.dataset.duration_ms + 1e-6
            or np.any((raw["x"] < 0) | (raw["x"] >= config.resolution.width))
            or np.any((raw["y"] < 0) | (raw["y"] >= config.resolution.height))
            or not np.isin(raw["p"], (0, 1)).all()):
        raise ValueError(f"invalid inference events: {path.name}")
    events = torch.from_numpy(np.column_stack((
        raw["x"].astype(np.float32), raw["y"].astype(np.float32),
        (timestamps / config.dataset.duration_ms).astype(np.float32), raw["p"].astype(np.float32),
    )))
    sequence = LoadedEventSequence(events, torch.zeros(len(events)),
        torch.zeros(len(events), dtype=torch.long), torch.from_numpy(timestamps), path)
    return sequence, source


def predict_primary_model(output_dir=None, *, manifest_path=DEFAULT_MAIN_MANIFEST,
                          input_dir=None, split="test", device="cpu", prediction_threads=8,
                          num_threads=None, export_submission=True, progress=print):
    """使用同一冻结完整管线预测；评分和训练均不进入此入口。"""
    if split not in {"val", "test"}:
        raise ValueError("primary prediction split must be val or test")
    prediction_threads = positive_prediction_threads(prediction_threads)
    root, manifest = load_primary_manifest(manifest_path)
    config = load_config(root / "config.yaml")
    # 运行线程独立于冻结训练配置，避免旧配置的8线程覆盖当前14线程。
    num_threads = positive_prediction_threads(
        manifest["runtime"]["num_threads"] if num_threads is None else num_threads)
    torch.set_num_threads(num_threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(device == "cuda")
    validate_backends(prediction_backend=device, feature_backend=device)
    model, checkpoint = load_inference_model(config, root / "best.pt", device=torch.device(device))
    models, release = load_event_release(root / "event_release", checkpoint=root / "best.pt",
                                         config=root / "config.yaml", backend=device)
    if input_dir is None:
        input_dir = Path(manifest_path).resolve().parent / manifest["default_inputs"][split]
    input_dir = Path(input_dir).resolve()
    paths = sorted(input_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no inference NPZ files in {input_dir}")
    if output_dir is None:
        output_dir = root / "predictions" / f"{split}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    records = {}
    with torch.inference_mode():
        for path in paths:
            sequence, source = _inference_input(path, config)
            cache = infer_event_cache(model, config, sequence,
                                              checkpoint["decision_threshold"], source=source)
            scores = predict_event_scores(cache, models, num_threads=prediction_threads,
                                          feature_backend=device)
            _, _, tracked, direct = event_branches(cache, scores, release["event_contract"])
            prediction = tracked | direct
            destination = write_prediction_npz(output_dir / "predictions" / path.name, prediction)
            records[path.name] = {"events": len(prediction), "positive": int(prediction.sum()),
                                   "prediction_sha256": sha256_file(destination)}
            progress(f"{path.name}: {int(prediction.sum()):,} / {len(prediction):,} positive")
    report = {"model": "primary", "split": split, "output_dir": str(output_dir),
              "manifest_sha256": sha256_file(manifest_path), "selection": manifest["selection"],
              "validation_labels_accessed_during_inference": False, "device": device,
              "active_event_models": list(models), "prediction_threads": prediction_threads,
              "torch_threads": torch.get_num_threads(), "files": records}
    if export_submission:
        audit = audit_submission(output_dir / "predictions", input_dir, output_dir / "submission.zip",
            expected_files=len(paths), expected_rows=sum(row["events"] for row in records.values()))
        report["submission_zip"] = audit["zip"]
    (output_dir / "prediction_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
