"""完成四组注意力的原始掩码复评、三轮计时和论文图表。"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import psutil
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluate_attention_test import evaluate as evaluate_test
from evaluate_attention_test import verify_raw_score

from traceformer.config import load_config
from traceformer.data import NPZEventDataset
from traceformer.experiments.ablation import AblationSpec
from traceformer.postprocessing.submission import evaluate_predictions
from traceformer.training.engine import (
    _dataset,
    _dataset_manifest,
    _file_sha256,
    _manifest_identity_digest,
    _metrics_at_threshold,
    _validate_checkpoint_contract,
    _write_json,
    build_model,
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_job(root, job):
    record = read(root / job["id"] / "done.json")
    config = load_config(job["config"])
    path = Path(record["checkpoint"])
    if (_file_sha256(path) != record["sha256"]
            or _file_sha256(Path(job["config"])) != job["config_sha256"]):
        raise ValueError("checkpoint or configuration changed")
    saved = torch.load(path, weights_only=True, map_location="cpu")
    spec = AblationSpec(**job["spec"])
    _validate_checkpoint_contract(saved, config, ablation=spec)
    model = build_model(config, ablation=spec).eval()
    model.load_state_dict(saved["model_state"], strict=True)
    return config, model, path


def benchmark(root, identifier):
    suite = read(root / "suite.json")
    job = next(j for j in suite["jobs"] if j["result_id"] == identifier)
    config, model, checkpoint = load_job(root, job)
    torch.set_num_threads(config.training.num_threads)
    dataset = NPZEventDataset(Path(suite["test_dataset_root"]) / "test",
                              width=config.resolution.width, height=config.resolution.height,
                              duration_ms=config.dataset.duration_ms)
    if len(dataset) != 24:
        raise ValueError("benchmark requires all 24 test files")
    rows = {}
    with torch.inference_mode():
        warm = dataset[0]
        model(warm.events)
        del warm
        for repeat in range(3):
            for sequence in dataset:
                # NPZ读取和输入解析在计时外，计时包含完整神经前向和几何构建。
                started = time.perf_counter()
                output = model(sequence.events)
                seconds = time.perf_counter() - started
                if not torch.isfinite(output["event_logits"]).all():
                    raise ValueError("nonfinite benchmark output")
                rows.setdefault(sequence.source_path.stem, []).append(seconds)
                del output
            print(f"benchmark {identifier}: pass {repeat + 1}/3", flush=True)
    means = [statistics.mean(v) for v in rows.values()]
    memory = psutil.Process().memory_info()
    (root / "benchmark").mkdir(exist_ok=True)
    _write_json(root / "benchmark" / f"{identifier}.json", {
        "variant": identifier, "checkpoint_sha256": _file_sha256(checkpoint),
        "device": "cpu", "precision": "float32", "threads": config.training.num_threads,
        "passes": 3, "sequences": len(dataset), "per_sequence_seconds": rows,
        "mean_seconds": statistics.mean(means), "median_seconds": statistics.median(means),
        "p95_seconds": float(np.quantile(means, 0.95)),
        "process_peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
        "memory_note": "fresh process peak RSS includes model, input loading and forward temporaries",
        "boundary": "tokenizer, geometry, attention, FFN, decoder; excludes disk, scoring and export",
    })


def validation_audit(root, suite, job):
    record = read(root / job["id"] / "done.json")
    attempt = Path(record["checkpoint"]).parent
    config = load_config(job["config"])
    torch.set_num_threads(config.training.num_threads)
    dataset = _dataset(config, "val")
    identity = _manifest_identity_digest(_dataset_manifest(_dataset(config, "train"), dataset))
    if identity != suite["dataset_manifest_sha256"]:
        raise ValueError("training/validation data changed before final audit")
    report = read(Path(record["report"]))
    results = []
    masks = root / "validation_masks" / job["result_id"]
    masks.mkdir(parents=True, exist_ok=True)
    for sequence in dataset:
        saved = torch.load(attempt / "predictions" / f"{sequence.source_path.stem}.pt",
                           weights_only=True, map_location="cpu")
        if (saved["threshold"] != report["metrics"]["threshold"]
                or saved["event_logits"].shape != (sequence.length,)
                or not torch.isfinite(saved["event_logits"]).all()):
            raise ValueError("invalid validation cache")
        mask = saved["event_logits"].sigmoid() >= saved["threshold"]
        np.savez_compressed(masks / sequence.source_path.name, prediction=mask.numpy())
        results.append((saved["event_logits"], sequence))
    raw = evaluate_predictions(masks, Path(config.dataset.root) / "val")
    verify_raw_score(report["metrics"]["competition"], raw["aggregate"])
    for name, result in raw["files"].items():
        verify_raw_score(report["per_file"][Path(name).stem]["metrics"]["competition"], result)
    curves = []
    (root / "curves").mkdir(exist_ok=True)
    # 曲线只在验证集计算，不依据测试曲线改变任何操作点。
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = _metrics_at_threshold(results, config, float(threshold))["competition"]
        curves.append({"threshold": float(threshold), **metrics})
    _write_json(root / "validation_masks" / f"{job['result_id']}_raw_report.json", raw)
    _write_json(root / "curves" / f"{job['result_id']}.json", curves)


def render(root, suite, test, benchmarks):
    target = root / "figures"
    target.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    descriptions = ["648 possible raw neighbors\nDirect token attention",
                    "Same fixed window (4,4)\nPast / future summaries",
                    "One cone (4,4)\nPast / future summaries",
                    "Cones (8,2), (4,4), (2,8)\nSix directional summaries"]
    for index, (ax, description) in enumerate(zip(axes, descriptions, strict=True)):
        ax.axis("off")
        ax.text(.5, .95, f"A{index}", ha="center", va="top", fontsize=15, weight="bold")
        ax.text(.5, .65, description, ha="center", va="center", fontsize=10,
                bbox={"boxstyle": "round,pad=.6", "facecolor": "#e7f0f7", "edgecolor": "#387ba8"})
        ax.annotate("", xy=(.5, .27), xytext=(.5, .46), arrowprops={"arrowstyle": "->"})
        ax.text(.5, .18, "Center query + residual", ha="center", fontsize=10)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(target / f"structure.{extension}", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for job in suite["jobs"]:
        variant = job["result_id"]
        curve = read(root / "curves" / f"{variant}.json")
        axes[0].plot([c["fa"] * 1e6 for c in curve], [c["pd"] * 100 for c in curve],
                     marker=".", label=variant)
        row = next(r for r in test["results"] if r["variant"] == variant)
        seconds = benchmarks[variant]["mean_seconds"]
        axes[1].scatter(seconds * 1000, row["iou"] * 100)
        axes[1].annotate(variant, (seconds * 1000, row["iou"] * 100), xytext=(4, 4),
                         textcoords="offset points")
    axes[0].set(xlabel="Validation Fa (1e-6)", ylabel="Validation Pd (%)")
    axes[0].legend()
    axes[1].set(xlabel="CPU time (ms / sequence)", ylabel="Test IoU (%)")
    for ax in axes:
        ax.grid(alpha=.25)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(target / f"tradeoff.{extension}", dpi=180)
    plt.close(figure)


def finish(root):
    suite = read(root / "suite.json")
    if (suite.get("protocol") != "center_attention_four_variants"
            or read(root / "status.json")["status"] != "complete"):
        raise ValueError("all four validation jobs must be complete")
    if suite.get("test_dataset_manifest_sha256"):
        identity = _manifest_identity_digest(_dataset_manifest(NPZEventDataset(
            Path(suite["test_dataset_root"]) / "test")))
        if identity != suite["test_dataset_manifest_sha256"]:
            raise ValueError("test content changed after protocol freeze")
    for job in suite["jobs"]:
        validation_audit(root, suite, job)
    test_root = root / "test"
    if not (test_root / "status.json").exists() or read(test_root / "status.json")["status"] != "complete":
        evaluate_test(SimpleNamespace(suite=root, dataset_root=Path(suite["test_dataset_root"]),
                                      output=test_root))
    benchmarks = {}
    for job in suite["jobs"]:
        variant = job["result_id"]
        path = root / "benchmark" / f"{variant}.json"
        if not path.exists():
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--suite", str(root),
                            "--benchmark-job", variant], check=True)
        benchmarks[variant] = read(path)
    test = read(test_root / "report.json")
    render(root, suite, test, benchmarks)
    lines = ["# 中心锚定多锥注意力：四组整体结构消融", "",
             "seed37，24轮，纯主干；所有测试阈值均由验证集选定。", "",
             "|组别|参数|验证Score|测试Score|IoU %|ACC %|Pd %|Fa (1e-6)|CPU ms|RSS MiB|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in test["results"]:
        variant = row["variant"]
        speed = benchmarks[variant]
        lines.append(f"|{variant}|{row['parameters']}|{row['validation_score']:.6f}|{row['score']:.6f}|"
                     f"{row['iou']*100:.3f}|{row['seg_acc']*100:.3f}|{row['pd']*100:.3f}|"
                     f"{row['fa']*1e6:.4f}|{speed['mean_seconds']*1000:.2f}|"
                     f"{speed['process_peak_rss_bytes']/2**20:.1f}|")
    lines.extend(["", "相邻比较和整体比较：", ""])
    for row in test["comparisons"]:
        lines.append(f"- {row['comparison']}：Score {row['score']:+.6f}，"
                     f"IoU {row['iou']*100:+.3f} pp，Pd {row['pd']*100:+.3f} pp，"
                     f"Fa {row['fa']*1e6:+.4f} × 1e-6。")
    lines.extend(["", "A0→A1衡量摘要式中心交互。四组匹配最大范围，不匹配参数/邻居数/FLOPs。",
                  "A2是预先指定的中间锥。单种子结果不支持稳定性或统计显著性结论。",
                  "RSS为独立进程峰值，含模型、输入加载及前向临时张量。"])
    (root / "final_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(root / "final_report.json", {"status": "complete", "test": test,
                                             "benchmarks": benchmarks, "validation_raw_audit": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--benchmark-job")
    args = parser.parse_args()
    if args.benchmark_job:
        benchmark(args.suite.resolve(), args.benchmark_job)
    else:
        finish(args.suite.resolve())
