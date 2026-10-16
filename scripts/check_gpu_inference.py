"""检查 GPU 邻居、特征、树分数及当前模型输出，不训练或搜索阈值。"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from traceformer.config import load_config
from traceformer.data.dataset import LoadedEventSequence, NPZEventDataset
from traceformer.gpu.runtime import require_cuda
from traceformer.models.cone_neighbor_builder import ConeNeighborBuilder
from traceformer.models.gpu_neighbors import configure_neighbor_backend
from traceformer.postprocessing.contracts import EventCache, sha256_file
from traceformer.postprocessing.features import build_event_features, make_context
from traceformer.postprocessing.gpu import CudaEventFeatures
from traceformer.postprocessing.learning import (
    load_event_models,
    positive_prediction_threads,
    predict_event_scores,
)
from traceformer.postprocessing.submission import evaluate_predictions, write_prediction_npz
from traceformer.training.engine import (
    _dataset_manifest,
    _manifest_identity_digest,
    forward_with_sparse_mixture,
    load_inference_model,
)

REPO = Path(__file__).resolve().parents[1]


def difference(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("comparison shape differs")
    # 相同 Inf 不能算作通过，NaN 也须在输出差异计算前拒绝。
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("comparison requires finite values")
    return {"elements": int(left.size), "different": int(np.count_nonzero(left != right)),
            "maximum_absolute_difference": float(np.max(np.abs(left.astype(np.float64) - right), initial=0))}


@torch.inference_mode()
def check(arguments, *, assets=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device, _ = require_cuda()
    torch.set_num_threads(14)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if assets is None:
        assets = load_assets(arguments, device)
    config, model, checkpoint, cpu_trees, gpu_trees = assets
    # 全集核验复用只读权重，但每条序列重新准备几何并先恢复 CPU 查询。
    configure_neighbor_backend(model, "cpu")
    if arguments.sequence:
        dataset = NPZEventDataset(arguments.sequence.parent, width=config.resolution.width,
            height=config.resolution.height, duration_ms=config.dataset.duration_ms)
        index = next(i for i, path in enumerate(dataset.files) if path.name == arguments.sequence.name)
        original = dataset[index]
        with np.load(arguments.sequence, allow_pickle=False) as saved:
            locations = saved["ev_loc"].astype(np.int64)
            raw = saved["evs_norm"][:, :4].astype(np.float32)
    else:
        rng = np.random.default_rng(37)
        locations = np.column_stack((rng.integers(0, 32, 512), rng.integers(0, 24, 512),
                                     rng.integers(0, min(500, int(config.dataset.duration_ms)), 512)))
        values = np.column_stack((locations[:, :2], locations[:, 2] / config.dataset.duration_ms,
                                  rng.choice([-1, 1], 512))).astype(np.float32)
        original = LoadedEventSequence(events=torch.from_numpy(values), labels=torch.zeros(512),
            object_ids=torch.zeros(512, dtype=torch.int64), timestamps_ms=torch.from_numpy(locations[:, 2].astype(np.float64)),
            source_path=Path("synthetic.npz"))
        raw = values
    # 来源文件的标签/身份不进入前向；参考计算也只使用相同的事件输入。
    sequence = LoadedEventSequence(events=original.events, labels=torch.zeros(original.length),
        object_ids=torch.zeros(original.length, dtype=torch.int64), timestamps_ms=original.timestamps_ms,
        source_path=original.source_path)
    events = sequence.events.to(device)
    batches = torch.zeros(len(events), dtype=torch.int64, device=device)
    tokenized = model.tokenizer(events, batches)
    report = {"status": "checking", "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "model_comparison": "same CUDA network; CPU neighbor queries versus CUDA neighbor queries",
        "checkpoint_sha256": sha256_file(arguments.checkpoint), "sequence": str(original.source_path),
        "source_sha256": sha256_file(original.source_path) if arguments.sequence else None,
        "events": original.length, "neighbors": {}, "model_outputs": {}, "tree_scores": {}}
    for index, layer in enumerate(model.encoder.layers):
        builder = layer.attention.neighbor_builder
        expected = builder(tokenized.token_coords, tokenized.token_batch_indices)
        other = ConeNeighborBuilder(cone_heads=builder.cone_heads,
            max_neighbors_per_cone=builder.max_neighbors_per_cone,
            layer_time_scale=builder.layer_time_scale, backend="cuda")
        actual = other(tokenized.token_coords, tokenized.token_batch_indices)
        report["neighbors"][str(index)] = {name: difference(getattr(expected, name).cpu().numpy(),
            getattr(actual, name).cpu().numpy()) for name in ("past_indices", "future_indices")}
    cpu_query, _ = forward_with_sparse_mixture(model, sequence, events, batches, config, compute_sparse_logits=False)
    configure_neighbor_backend(model, "cuda")
    gpu_query, _ = forward_with_sparse_mixture(model, sequence, events, batches, config, compute_sparse_logits=False)
    for name, expected in cpu_query.items():
        if isinstance(expected, torch.Tensor):
            report["model_outputs"][name] = difference(expected.cpu().numpy(), gpu_query[name].cpu().numpy())
    probability = torch.sigmoid(gpu_query["event_logits"]).cpu().numpy()
    cache = EventCache(name=original.source_path.name, locations=locations, raw_features=raw,
        base_probability=probability,
        token_features=gpu_query["token_features"].cpu().numpy().astype(np.float16).astype(np.float32),
        event_token_indices=gpu_query["event_token_indices"].cpu().numpy(),
        prediction=probability >= float(checkpoint["decision_threshold"]))
    names = tuple(arguments.model_names)
    expected = predict_event_scores(cache, cpu_trees, num_threads=arguments.prediction_threads)
    for backend in ("cpu", "cuda"):
        actual = predict_event_scores(cache, gpu_trees, feature_backend=backend,
                                      num_threads=arguments.prediction_threads)
        report["tree_scores"][backend] = {name: difference(expected[name], actual[name]) for name in names}
    gpu_features = CudaEventFeatures(cache)
    # 分块对照复用 CPU 上下文，避免对最大序列反复计算同一几何统计。
    cpu_context = make_context(cache.locations, cache.base_probability)
    feature_differences = 0
    feature_error = 0.0
    for start in range(0, len(locations), 200_000):
        stop = min(start + 200_000, len(locations))
        row = difference(build_event_features(cache, indices=np.arange(start, stop), context=cpu_context),
                         gpu_features.rows(start, stop).cpu().numpy())
        feature_differences += row["different"]
        feature_error = max(feature_error, row["maximum_absolute_difference"])
    report["features"] = {"different": feature_differences, "maximum_absolute_difference": feature_error}
    threshold = float(checkpoint["decision_threshold"])
    cpu_mask = torch.sigmoid(cpu_query["event_logits"]).cpu().numpy() >= threshold
    gpu_mask = probability >= threshold
    report["decision_mask"] = difference(cpu_mask, gpu_mask)
    if getattr(arguments, "prediction_dir", None):
        write_prediction_npz(arguments.prediction_dir / original.source_path.name, gpu_mask)
    rows = [*report["model_outputs"].values(), report["features"], report["decision_mask"]]
    rows.extend(row for layers in report["neighbors"].values() for row in layers.values())
    rows.extend(row for mode in report["tree_scores"].values() for row in mode.values())
    report["status"] = "verified" if all(row["different"] == 0 for row in rows) else "mismatch"
    report["real_cuda_verified"] = report["status"] == "verified"
    return report


def load_assets(arguments, device):
    """全集检查只加载一次权重，保留原加载契约及树模型检查。"""
    config = load_config(arguments.config)
    model, checkpoint = load_inference_model(config, arguments.checkpoint, device=device)
    names = tuple(arguments.model_names)
    cpu_trees = load_event_models(arguments.model_dir, names=names)
    gpu_trees = load_event_models(arguments.model_dir, names=names, backend="cuda")
    return config, model, checkpoint, cpu_trees, gpu_trees


def check_split(arguments):
    """验证真实 val/test；标签只在预测写出后由独立评分器读取。"""
    device, _ = require_cuda()
    assets = load_assets(arguments, device)
    reports, metrics = {}, {}
    splits = ("val", "test") if arguments.split == "all" else (arguments.split,)
    if "val" in splits:
        config, _, checkpoint, _, _ = assets
        validation = NPZEventDataset(arguments.dataset / "val", width=config.resolution.width,
            height=config.resolution.height, duration_ms=config.dataset.duration_ms)
        # 验证集必须与保存权重的训练记录相同，不用标签选择阈值。
        if _manifest_identity_digest(_dataset_manifest(validation)) != checkpoint["validation_manifest_sha256"]:
            raise ValueError("checkpoint validation dataset manifest differs")
    for split in splits:
        paths = sorted((arguments.dataset / split).glob("*.npz"))
        if not paths:
            raise ValueError(f"no sequences in {split}")
        for path in paths:
            current = copy.copy(arguments)
            current.sequence = path
            current.prediction_dir = arguments.output.parent / "current_predictions" / split
            row = check(current, assets=assets)
            reports[f"{split}/{path.name}"] = row
            destination = arguments.output.parent / "checks" / split / (path.stem + ".json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            print(f"{split}/{path.name}: {row['status']}", flush=True)
            if row["status"] != "verified":
                return {"status": "mismatch", "real_cuda_verified": False, "files": reports}
        metrics[split] = evaluate_predictions(current.prediction_dir, arguments.dataset / split)
    return {"status": "verified", "real_cuda_verified": True, "sequences": len(reports),
            "files": reports, "independent_metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true", help="在独立 CPU simulator 进程执行同一份内核")
    parser.add_argument("--config", type=Path, default=REPO / "configs/traceformer_score_optimized.yaml")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "outputs/cpu_refinement_20260830_154410/best.pt")
    parser.add_argument("--model-dir", type=Path, default=REPO / "outputs/paper_cuda_20260928/copyback_20260929/artifacts/event_release/events")
    parser.add_argument("--model-names", nargs="+", choices=("small", "large"), default=["large"])
    parser.add_argument("--prediction-threads", type=positive_prediction_threads, default=8)
    parser.add_argument("--sequence", type=Path)
    parser.add_argument("--split", choices=("val", "test", "all"), help="复用权重检查完整数据分区")
    parser.add_argument("--dataset", type=Path, default=REPO.parent / "EV-UAV-dataset")
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.split and (arguments.sequence or arguments.simulate):
        parser.error("--split cannot be combined with --sequence or --simulate")
    if arguments.simulate:
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/gpu_kernel_cases.py", "-q"],
            cwd=REPO, env=dict(os.environ, NUMBA_ENABLE_CUDASIM="1"), capture_output=True,
            text=True, encoding="utf-8", timeout=180)
        report = {"status": "simulator_verified" if result.returncode == 0 else "simulator_failed",
                  "real_cuda_verified": False, "output": result.stdout, "errors": result.stderr}
    else:
        try:
            report = check_split(arguments) if arguments.split else check(arguments)
        except (ImportError, RuntimeError, ValueError) as error:
            report = {"status": "unavailable_or_failed", "real_cuda_verified": False, "error": str(error)}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] in {"verified", "simulator_verified"} else 1)


if __name__ == "__main__":
    main()
