"""测量当前固定网络、事件树和主轨迹；默认使用现有 GPU 加速链路。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import lightgbm
import numpy as np
import psutil
import torch

from traceformer.config import load_config
from traceformer.data.dataset import LoadedEventSequence
from traceformer.gpu.runtime import add_backend_arguments, validate_backends
from traceformer.postprocessing.cache import infer_event_cache, infer_event_output
from traceformer.postprocessing.contracts import sha256_file
from traceformer.postprocessing.direct_verifier import event_branches
from traceformer.postprocessing.learning import positive_prediction_threads, predict_event_scores
from traceformer.postprocessing.release import event_validation_gate, load_event_release
from traceformer.postprocessing.submission import load_prediction, write_prediction_npz
from traceformer.training.engine import load_inference_model


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "event-release-dir", "dataset-root", "output", "validation-report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--mode", choices=("network", "pipeline"), default="pipeline")
    parser.add_argument("--diagnostic", action="store_true", help="明确评测未通过本地门槛的候选，不改变正式验收状态")
    parser.add_argument("--reference-dir", type=Path, help="验证集本地最终掩码")
    parser.add_argument("--validation-evidence", type=Path, help="测试集计时前通过的云端验证报告")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--num-threads", type=int, default=14)
    parser.add_argument("--prediction-threads", type=positive_prediction_threads, default=8,
                        help="CPU LightGBM线程数；CUDA树不使用此参数")
    parser.add_argument("--repeats", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--profile", action="store_true", help="额外做一轮阶段同步诊断")
    parser.add_argument("--export-track2", action="store_true", help="用已保存测试掩码导出TXT和ZIP，不重复前向")
    add_backend_arguments(parser, trees=True)
    parser.set_defaults(prediction_backend="cuda", feature_backend="cuda")
    return parser


def run(args):
    if args.num_threads <= 0:
        raise ValueError("thread count must be positive")
    if args.export_track2 and (args.split != "test" or args.diagnostic):
        raise ValueError("Track 2 export requires a non-diagnostic test run")
    # 与现有 CUDA 验证器一致，在初始化 CUDA 前配置可重复的矩阵运算。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    validate_backends(prediction_backend=args.prediction_backend, feature_backend=args.feature_backend)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    config = load_config(args.config)
    model, checkpoint = load_inference_model(config, args.checkpoint, device=device)
    models, release = load_event_release(
        args.event_release_dir, checkpoint=args.checkpoint, config=args.config,
        backend=args.prediction_backend, load_models=args.mode == "pipeline",
    )
    identity = {"checkpoint_sha256": sha256_file(args.checkpoint),
                "release_report_sha256": sha256_file(args.event_release_dir / "release_report.json"),
                "prediction_backend": args.prediction_backend, "feature_backend": args.feature_backend,
                "mode": args.mode, "diagnostic": args.diagnostic,
                "torch_threads": args.num_threads, "prediction_threads": args.prediction_threads,
                "thread_environment": {name: os.environ.get(name) for name in (
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    local_gate = event_validation_gate(validation["event_release"], validation["original"])
    if (not local_gate["passed"] and not args.diagnostic
            or validation["protocol"]["event_release_report_sha256"] != identity["release_report_sha256"]):
        raise ValueError("local validation did not accept this release")
    if args.split == "test":
        if args.validation_evidence is None:
            raise ValueError("test timing requires --validation-evidence")
        evidence = json.loads(args.validation_evidence.read_text(encoding="utf-8"))
        if (evidence["split"] != "val"
                or (not evidence["reference_masks_exact"] and not args.diagnostic)
                or (args.diagnostic and len(evidence["reference_mask_differences"]) != 24)
                or not evidence["repeat_masks_exact"] or evidence["identity"] != identity
                or evidence["environment"]["device"] != str(device)):
            raise ValueError("cloud validation evidence did not verify this system")
    elif args.reference_dir is None:
        raise ValueError("validation timing requires --reference-dir")
    torch.set_num_threads(args.num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(device.type == "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_root = args.output.with_suffix("").with_name(args.output.stem + "_predictions")
    if args.output.exists() or prediction_root.exists():
        raise FileExistsError("benchmark output already exists")
    prediction_root.mkdir()
    paths = sorted((args.dataset_root / args.split).glob("*.npz"))
    if len(paths) != 24:
        raise ValueError("requires all 24 partition files")
    if args.split == "val" and {p.name for p in args.reference_dir.glob("*.npz")} != {p.name for p in paths}:
        raise ValueError("reference mask file set differs")
    kernel_report = None
    if device.type == "cuda" and args.split == "val":
        # 复用已有数值检查，验证当前网络、112维特征与本次发布的活动树。
        kernel_report = args.output.with_name(args.output.stem + "_kernels.json")
        command = [sys.executable, str(Path(__file__).with_name("check_gpu_inference.py")),
                   "--config", str(args.config), "--checkpoint", str(args.checkpoint),
                   "--model-dir", str(args.event_release_dir / "events"),
                   "--model-names", *models, "--dataset", str(args.dataset_root), "--split", "val",
                   "--output", str(kernel_report)]
        if models:
            with kernel_report.with_suffix(".log").open("w", encoding="utf-8") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            numerical = json.loads(kernel_report.read_text(encoding="utf-8"))
            if numerical["status"] != "verified" or numerical["sequences"] != 24:
                raise ValueError("CUDA numerical validation did not pass")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def infer(path, *, profile=False):
        # 磁盘解压与标注均不参与正式计时；仅保留输入的四个字段。
        with np.load(path, allow_pickle=False) as saved:
            raw = saved["ev"]
            source = {"locations": saved["ev_loc"].astype(np.int64),
                      "raw_features": saved["evs_norm"][:, :4].astype(np.float32)}
        forward_times = []
        hooks = []
        if profile:
            def before(_module, _args):
                sync()
                forward_times.append(time.perf_counter())

            def after(_module, _args, _output):
                sync()
                forward_times.append(time.perf_counter())

            hooks = [model.register_forward_pre_hook(before), model.register_forward_hook(after)]
        try:
            sync()
            started = time.perf_counter()
            timestamps = raw["t"].astype(np.float64)
            events = torch.from_numpy(np.column_stack((
                raw["x"].astype(np.float32), raw["y"].astype(np.float32),
                (timestamps / config.dataset.duration_ms).astype(np.float32), raw["p"].astype(np.float32),
            )))
            sequence = LoadedEventSequence(
                events, torch.zeros(len(events)), torch.zeros(len(events), dtype=torch.long),
                torch.from_numpy(timestamps), path,
            )
            prepared = time.perf_counter()
            if args.mode == "network":
                output = infer_event_output(model, config, sequence)
                final = (output["event_logits"].sigmoid() >= checkpoint["decision_threshold"]).cpu().numpy()
            else:
                cache = infer_event_cache(model, config, sequence, checkpoint["decision_threshold"], source=source)
            if profile:
                sync()
            cached = time.perf_counter()
            scores = predict_event_scores(cache, models, num_threads=args.prediction_threads,
                                          feature_backend=args.feature_backend) if args.mode == "pipeline" else None
            if profile:
                sync()
            scored = time.perf_counter()
            if args.mode == "pipeline":
                _, _, tracked, direct = event_branches(cache, scores, release["event_contract"])
                final = tracked | direct
            sync()
            ended = time.perf_counter()
            if args.mode == "network" and not torch.isfinite(output["event_logits"]).all().item():
                raise ValueError("network returned nonfinite logits")
            diagnostic = {}
            if profile:
                diagnostic = {"input_seconds": prepared - started,
                              "network_sparse_cache_seconds": cached - prepared,
                              "network_forward_seconds": sum(forward_times[i + 1] - forward_times[i]
                                                             for i in range(0, len(forward_times), 2)),
                              "features_trees_seconds": scored - cached,
                              "tracking_seconds": ended - scored,
                              "instrumented_total_seconds": ended - started}
            return final, len(events), False, ended - started, diagnostic
        finally:
            for hook in hooks:
                hook.remove()

    def count(path):
        with np.load(path, allow_pickle=False) as saved:
            return len(saved["ev_loc"])

    records = {p.name: {"source_sha256": sha256_file(p), "seconds": []} for p in paths}
    differences = {}
    warmup = max(paths, key=count)
    with torch.inference_mode():
        infer(warmup)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for repeat in range(args.repeats):
            for path in paths:
                mask, events, sparse, seconds, _ = infer(path)
                destination = prediction_root / path.name
                if repeat == 0:
                    write_prediction_npz(destination, mask)
                    if args.reference_dir is not None:
                        differences[path.name] = int(np.count_nonzero(mask != load_prediction(args.reference_dir / path.name)))
                elif not np.array_equal(mask, load_prediction(destination)):
                    raise ValueError(f"repeat mask differs: {path.name}")
                records[path.name].update(events=events, sparse_view=sparse)
                records[path.name]["seconds"].append(seconds)
                print(f"pass {repeat + 1} {path.name}: {seconds:.4f}s", flush=True)
        peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        if args.profile:
            for path in paths:
                mask, _, _, _, diagnostic = infer(path, profile=True)
                if not np.array_equal(mask, load_prediction(prediction_root / path.name)):
                    raise ValueError("profiling changed predictions")
                records[path.name]["profile"] = diagnostic
    means = np.array([np.mean(row["seconds"]) for row in records.values()])
    total_events = sum(row["events"] for row in records.values())
    result = {
        "identity": identity, "split": args.split, "repeat_masks_exact": True,
        "mode": args.mode, "diagnostic": args.diagnostic, "local_validation_gate": local_gate,
        "reference_masks_exact": bool(differences) and not any(differences.values()),
        "reference_mask_differences": differences, "warmup_sequence": warmup.name,
        "timed_passes": args.repeats, "mean_ms_per_sequence": float(means.mean() * 1000),
        "median_ms_per_sequence": float(np.median(means) * 1000),
        "p95_ms_per_sequence": float(np.percentile(means, 95) * 1000),
        "max_ms_per_sequence": float(means.max() * 1000),
        "events_per_second": float(total_events / means.sum()),
        "peak_allocated_vram_bytes": peak_vram, "active_event_models": list(models),
        "neural_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "tree_models": {name: {"trees": booster.num_trees(),
                               "leaves": sum(tree["num_leaves"] for tree in booster.dump_model()["tree_info"])}
                        for name, booster in models.items()},
        "environment": {"platform": platform.platform(), "processor": platform.processor(),
                        "torch": torch.__version__, "cuda_build": torch.version.cuda,
                        "lightgbm": lightgbm.__version__,
                        "device": str(device), "threads": args.num_threads, "precision": "FP32",
                        # 保存实际运行值，避免仅凭命令默认值判断线程口径。
                        "torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads(),
                        "cpu_tree_threads_requested": args.prediction_threads,
                        "thread_environment": identity["thread_environment"],
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "ram_bytes": psutil.virtual_memory().total, "tf32": False},
        "scope": "In-memory normalization through final mask, including sparse view and transfers; "
                 + ("pipeline also includes token quantization, features, active trees and CPU tracking; "
                    if args.mode == "pipeline" else "network mode excludes all postprocessing and token-cache construction; ")
                 + "excludes loading, disk IO, "
                 "scoring and checks. Formal total synchronizes only at start/end. Profile is separate "
                 "and instrumented; network_forward_seconds is nested within network_sparse_cache_seconds.",
        "files": records, "script_sha256": sha256_file(__file__),
    }
    if kernel_report is not None and kernel_report.is_file():
        result["gpu_kernel_report_sha256"] = sha256_file(kernel_report)
    if args.export_track2:
        from predict_postprocess_track2 import _write_track2_txt

        # 导出使用正式入口的九位小数格式，计时后的掩码直接复用。
        txt_dir = prediction_root.parent / (prediction_root.name + "_txt")
        txt_dir.mkdir()
        for path in paths:
            _write_track2_txt(path, txt_dir / f"{path.stem}.txt", load_prediction(prediction_root / path.name))
        zip_path = prediction_root.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(txt_dir.glob("*.txt")):
                archive.write(path, path.name)
        result["submission_zip_sha256"] = sha256_file(zip_path)
        result["submission_zip"] = str(zip_path.resolve())
        result["submission_txt"] = str(txt_dir.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.split == "val" and not result["reference_masks_exact"] and not args.diagnostic:
        raise ValueError("CUDA validation masks differ; inspect report before test inference")
    print(f"mean {result['mean_ms_per_sequence']:.3f} ms/sequence", flush=True)


if __name__ == "__main__":
    run(build_parser().parse_args())
