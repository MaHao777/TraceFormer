"""计时已发布完整系统，并逐位核对全部测试掩码。"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from traceformer.gpu.runtime import add_backend_arguments, validate_backends
from traceformer.postprocessing.contracts import EventCache, density_group, sha256_file
from traceformer.postprocessing.direct_verifier import event_branches
from traceformer.postprocessing.learning import (
    load_event_models,
    positive_prediction_threads,
    predict_event_scores,
    required_event_models,
)
from traceformer.postprocessing.submission import (
    evaluate_predictions,
    load_prediction,
    write_prediction_npz,
)

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "outputs/paper_cuda_20260928/copyback_20260929"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_detector(name, frozen, device):
    """加载主检测器归档源码并验证权重、配置与源码哈希。"""
    report = read(frozen / name / "report.json")
    schema = report["schema"]
    source = ARCHIVE / f"legacy_schema{schema}/src"
    # 旧包使用同一名称；切换只影响后续导入，不修改归档源码。
    for key in tuple(sys.modules):
        if key == "huav" or key.startswith("huav."):
            del sys.modules[key]
    sys.path.insert(0, str(source))
    try:
        config_module = importlib.import_module("huav.config")
        engine = importlib.import_module("huav.training.engine")
    finally:
        sys.path.remove(str(source))
    checkpoint_path = ARCHIVE / f"artifacts/{name}/best.pt"
    config_path = ARCHIVE / f"artifacts/{name}/config.yaml"
    for path, field in ((checkpoint_path, "checkpoint_sha256"),
                        (config_path, "original_config_sha256"),
                        (Path(engine.__file__), "source_engine_sha256")):
        if sha256_file(path).lower() != report[field].lower():
            raise ValueError(f"released {name} {field} differs")
    config = config_module.load_config(frozen / name / "config.yaml")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != engine.CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema differs")
    for field, stored in (("model", "model_config"), ("resolution", "resolution"),
                          ("sparse_mixture", "sparse_mixture_config")):
        if hasattr(config, field) and stored in checkpoint:
            if checkpoint[stored] != engine._serializable(asdict(getattr(config, field))):
                raise ValueError(f"checkpoint {name}/{field} differs")
    model = engine.build_model(config).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != report["parameters"]:
        raise ValueError("neural parameter count differs")
    return model, config, engine, report


@torch.no_grad()
def infer(
    source, raw_features, locations, path, main, boosters, release, *,
    prediction_threads=8,
    feature_backend="cpu",
):
    """计时从内存输入的归一化开始，到最终布尔掩码结束。"""
    model, config, engine, report = main
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    timestamp = source["t"].astype(np.float64)
    events = torch.from_numpy(np.column_stack((
        source["x"].astype(np.float32), source["y"].astype(np.float32),
        (timestamp / config.dataset.duration_ms).astype(np.float32),
        source["p"].astype(np.float32))))
    # 推理只使用 x/y/t/p；标签和目标身份始终为零。
    sequence_type = engine.NPZEventDataset.__getitem__.__globals__["LoadedEventSequence"]
    sequence = sequence_type(events=events, labels=torch.zeros(len(events)),
        object_ids=torch.zeros(len(events), dtype=torch.long),
        timestamps_ms=torch.from_numpy(timestamp), source_path=path)
    events = events.to(device)
    batch = torch.zeros(len(events), dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prepared = time.perf_counter()
    prediction, sparse = engine._forward_with_sparse_mixture(
        model, sequence, events, batch, config, mixture_weight=1.0)
    probability_tensor = torch.sigmoid(prediction["event_logits"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    neural_done = time.perf_counter()
    # GPU 网络与传回 CPU 的时间分开记录，后处理继续使用发布实现。
    probability = probability_tensor.cpu().numpy()
    token_features = prediction["token_features"].cpu().numpy()
    event_token_indices = prediction["event_token_indices"].cpu().numpy()
    copied = time.perf_counter()
    # 保留发布缓存的半精度量化，不改变后处理输入格式。
    cache = EventCache(name=path.name, locations=locations, raw_features=raw_features,
        base_probability=probability,
        token_features=token_features.astype(np.float16).astype(np.float32),
        event_token_indices=event_token_indices,
        prediction=probability >= report["release_threshold"]["threshold"])
    group, _ = density_group(locations)
    scores = predict_event_scores(cache, boosters, num_threads=prediction_threads,
                                   feature_backend=feature_backend)
    # 单独记录事件树及特征耗时，保留原后处理合计字段。
    scores_done = time.perf_counter()
    _, _, tracked, direct = event_branches(cache, scores, release["event_contract"])
    # 合并掩码也计入主轨迹阶段，不再加载第二套检测器。
    final = direct | tracked
    ended = time.perf_counter()
    timings = {"input_seconds": prepared-started, "main_seconds": neural_done-prepared,
        "device_to_host_seconds": copied-neural_done,
        "verification_tracking_seconds": ended-copied,
        "event_scores_seconds": scores_done-copied,
        "tracking_seconds": ended-scores_done, "total_seconds": ended-started}
    masks = {"complete": final, "neural": cache.prediction,
             "direct": direct, "event_track": final}
    return masks, group, sparse is not None, timings


def benchmark(args):
    prediction_backend = getattr(args, "prediction_backend", "cpu")
    feature_backend = getattr(args, "feature_backend", "cpu")
    validate_backends(prediction_backend=prediction_backend, feature_backend=feature_backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    # CUDA 原子累加会扰动半精度量化边界；使用可重复的归约实现。
    torch.use_deterministic_algorithms(device.type == "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    main = load_detector("main", args.frozen, device)

    release_path = ARCHIVE / "artifacts/event_release"
    release = read(release_path / "release_report.json")
    for name, expected in release["event_model_sha256"].items():
        if sha256_file(release_path / "events" / name).lower() != expected.lower():
            raise ValueError("released tree model differs")
    boosters = load_event_models(
        release_path / "events", names=required_event_models(release["event_contract"]),
        backend=prediction_backend,
    )
    tree_counts = {name: {"trees": model.num_trees(),
        "leaves": sum(tree["num_leaves"] for tree in model.dump_model()["tree_info"])}
        for name, model in boosters.items()}
    audit = read(args.frozen / "dataset_audit.json")["partitions"]["test"]["files"]
    paths = sorted((args.dataset / "test").glob("*.npz"))
    if len(paths) != 24 or {p.name for p in paths} != set(audit):
        raise ValueError("requires all 24 published test files")

    prediction_root = args.output.parent / (args.output.stem + "_predictions")

    def run(path, pass_index):
        if sha256_file(path).lower() != audit[path.name]["sha256"].lower():
            raise ValueError("raw test data changed")
        # 文件解压、输入复制、模型加载和结果核验均位于计时范围之外。
        with np.load(path, allow_pickle=False) as data:
            source = data["ev"]
            locations = data["ev_loc"].astype(np.int64)
            features = data["evs_norm"][:, :4].astype(np.float32)
        masks, group, sparse, timings = infer(
            source, features, locations, path, main, boosters, release,
            prediction_threads=args.prediction_threads,
            feature_backend=feature_backend)
        differences = {}
        for stage in ("complete", "neural", "direct", "event_track"):
            differences[stage] = int(np.count_nonzero(
                masks[stage] != load_prediction(args.frozen / "complete" / ("event_track" if stage == "complete" else stage) / path.name)))
            if differences[stage]:
                raise ValueError(f"frozen {stage} mask differs: {path.name}")
        # CUDA 浮点差异如实计数；重复运行必须得到相同的最终决策。
        if pass_index >= 0:
            for stage in ("complete", "neural", "direct", "event_track"):
                saved = prediction_root / stage / path.name
                if pass_index == 0:
                    write_prediction_npz(saved, masks[stage])
                elif not np.array_equal(masks[stage], load_prediction(saved)):
                    raise ValueError(f"repeat {pass_index} {stage} mask differs: {path.name}")
        return {"events": len(source), "density_group": group, "sparse_view": sparse,
                "source_sha256": audit[path.name]["sha256"], "timings": timings,
                "cpu_mask_differences": differences,
                "frozen_masks_exact": not any(differences.values())}

    # 固定最大序列预热一次，覆盖主网络、稀疏视图、树模型。
    warmup = max(paths, key=lambda p: audit[p.name]["events"])
    print(f"warmup {warmup.name}", flush=True)
    warmup_record = run(warmup, -1)
    print(f"warmup complete: {warmup_record['timings']['total_seconds']:.3f}s; "
          f"CPU differences={warmup_record['cpu_mask_differences']}", flush=True)
    records = {}
    for repeat in range(args.repeats):
        for path in paths:
            row = run(path, repeat)
            records.setdefault(path.name, []).append(row)
            print(f"pass {repeat+1} {path.name}: {row['timings']['total_seconds']:.3f}s "
                  f"CPU differences={row['cpu_mask_differences']}", flush=True)
    # 每个序列的重复平均和原始测量同时保存，不能只挑最快的一次。
    for name, rows in records.items():
        records[name] = {**rows[0], "timings": {key: float(np.mean([
            row["timings"][key] for row in rows])) for key in rows[0]["timings"]},
            "timed_passes": [row["timings"] for row in rows]}
    totals = np.array([row["timings"]["total_seconds"] for row in records.values()])
    result = {"status": "verified", "split": "test", "sequences": 24,
        "frozen_masks_exact": all(r["frozen_masks_exact"] for r in records.values()),
        "repeat_masks_exact": True, "timed_passes": args.repeats, "warmup_sequence": warmup.name,
        "environment": {"processor": args.processor, "platform": platform.platform(),
            "torch": torch.__version__, "device": str(device), "precision": "FP32",
            "cuda_build": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32": False,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory
                if device.type == "cuda" else None,
            "torch_threads": args.num_threads,
            "lightgbm_prediction_threads": args.prediction_threads if prediction_backend == "cpu" else None,
            "cuda_event_tree_block_threads": 128 if prediction_backend == "cuda" else None},
        "active_event_models": list(boosters),
        "prediction_backend": prediction_backend,
        "feature_backend": feature_backend,
        "scope": "In-memory input normalization through final mask; includes main full/sparse "
                 "forward, quantized feature construction, active tree ensembles, tracking. Excludes model loading, disk "
                 "I/O, scoring, and correctness checks. CUDA synchronized at timing boundaries; "
                 "device transfers included in total and recorded separately. All repeated passes "
                 "after one full-system warmup are averaged; repeat masks must match exactly.",
        "neural_parameters": {"main": main[3]["parameters"], "total": main[3]["parameters"]},
        "tree_models": tree_counts, "mean_ms_per_sequence": float(totals.mean()*1000),
        "median_ms_per_sequence": float(np.median(totals)*1000),
        "min_ms_per_sequence": float(totals.min()*1000), "max_ms_per_sequence": float(totals.max()*1000),
        "component_mean_ms": {key: float(np.mean([row["timings"][key] for row in records.values()])*1000)
                              for key in next(iter(records.values()))["timings"]},
        "files": records, "script_sha256": sha256_file(__file__),
        "independent_metrics": {stage: evaluate_predictions(prediction_root / stage,
                                 args.dataset / "test")
                                for stage in ("neural", "direct", "event_track", "complete")},
        "prediction_sha256": {stage: {p.name: sha256_file(p) for p in
            sorted((prediction_root / stage).glob("*.npz"))}
            for stage in ("neural", "direct", "event_track", "complete")},
        "checkpoint_sha256": {name: value[3]["checkpoint_sha256"]
                              for name, value in (("main", main),)},
        "tree_model_sha256": release["event_model_sha256"]}
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(f"verified all 24 sequences; mean {result['mean_ms_per_sequence']:.2f} ms/sequence; "
          f"CPU masks exact={result['frozen_masks_exact']}")


def build_parser():
    """解析入口参数，不执行计时或全集推理。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=14)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--prediction-threads", type=positive_prediction_threads, default=8)
    add_backend_arguments(parser, trees=True)
    return parser


if __name__ == "__main__":
    benchmark(build_parser().parse_args())
