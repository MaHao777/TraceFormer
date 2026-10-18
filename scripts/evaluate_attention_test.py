"""复用三组消融权重和验证阈值，在完整带标签测试集推理并独立计分。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from traceformer.data import NPZEventDataset
from traceformer.postprocessing.submission import evaluate_predictions
from traceformer.training.engine import _dataset_manifest, _file_sha256, _write_json


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def evaluate(args) -> None:
    suite_root = args.suite.resolve()
    suite = read(suite_root / "suite.json")
    if (suite.get("protocol") != "attention_three_variants"
            or read(suite_root / "status.json")["status"] != "complete"
            or len(suite["jobs"]) != 3
            or {job["result_id"] for job in suite["jobs"]} != {"C0", "C1", "C2"}):
        raise ValueError("requires the completed three-variant attention suite")
    source = Path(suite["source_snapshot"])
    for relative, digest in read(suite_root / "source_hashes.json").items():
        if _file_sha256(source / relative) != digest:
            raise ValueError(f"frozen training source changed: {relative}")
    dataset = NPZEventDataset(args.dataset_root / "test")
    if len(dataset) != 24:
        raise ValueError("requires all 24 labeled test sequences")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    adapters = root / "adapters"
    adapters.mkdir()
    for path in (Path(__file__), Path(__file__).with_name("evaluate_neural_checkpoint.py")):
        shutil.copy2(path, adapters / path.name)
    _write_json(root / "dataset_manifest.json", _dataset_manifest(dataset))
    _write_json(root / "source_hashes.json", read(suite_root / "source_hashes.json"))
    environment = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONIOENCODING="utf-8",
                       PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    rows = []
    for job in suite["jobs"]:
        variant = job["result_id"]
        done = read(suite_root / job["id"] / "done.json")
        checkpoint = Path(done["checkpoint"])
        if (_file_sha256(checkpoint) != done["sha256"]
                or _file_sha256(Path(job["config"])) != job["config_sha256"]):
            raise ValueError(f"trained checkpoint or config changed: {variant}")
        _write_json(root / "status.json", {"status": "running", "variant": variant})
        output = root / variant
        command = [
            sys.executable, "-u", str(adapters / "evaluate_neural_checkpoint.py"),
            "--config", job["config"], "--checkpoint", str(checkpoint),
            "--ablation", str(checkpoint.parent / "ablation.json"),
            "--dataset-root", str(args.dataset_root.resolve()), "--split", "test",
            "--device", "cpu", "--num-threads", "8", "--output", str(output),
        ]
        # 每组推理与概率复评分开进程，沿用训练时冻结的全部模型源码。
        with (root / "logs" / f"{variant}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
            subprocess.run([*command, "--rescore-only"], check=True, env=environment,
                           stdout=log, stderr=subprocess.STDOUT)
        report = read(output / "independent_report.json")
        if report["sequences"] != 24 or report["ablation"] != job["spec"]:
            raise ValueError("test partition or ablation contract differs")
        raw = evaluate_predictions(output / "predictions", args.dataset_root / "test")
        verify_raw_score(report["competition"], raw["aggregate"])
        for name, metrics in report["per_sequence"].items():
            verify_raw_score(metrics, raw["files"][name])
        _write_json(output / "raw_independent_report.json", raw)
        rows.append({
            "variant": variant, "seed": job["seed"], "backbone": job["spec"]["backbone"],
            "threshold": report["threshold"], "parameters": report["parameters"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "validation_score": read(checkpoint.parent / "metrics.json")["competition"]["score"],
            **{key: report["competition"][key] for key in ("score", "iou", "seg_acc", "pd", "fa")},
        })
        print(f"{variant} test Score={rows[-1]['score']:.9f}; both independent scorers agree", flush=True)
    by_id = {row["variant"]: row for row in rows}
    comparisons = [{"comparison": f"{a}-{b}", **{
        key: by_id[a][key] - by_id[b][key] for key in ("score", "iou", "seg_acc", "pd", "fa")
    }} for a, b in suite["comparisons"]]
    _write_json(root / "report.json", {
        "status": "complete", "suite": str(suite_root), "split": "test", "sequences": 24,
        "test_threshold_selection": False, "postprocessing": False,
        "inference_annotations": "zeroed labels and object IDs", "results": rows,
        "comparisons": comparisons,
        "adapter_sha256": {p.name: _file_sha256(p) for p in adapters.glob("*.py")},
    })
    lines = ["# 三组注意力消融：带标签测试集", "",
             "99训练/24验证选模；复用各组验证阈值，24测试文件，不进行测试调参或后处理。",
             "推理标签清零，另进程复评分数，并与原始事件的另一计数实现逐文件一致。", "",
             "|组别|验证阈值|验证Score|测试Score|IoU|SegAcc|Pd|Fa|",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"|{row['variant']}|{row['threshold']:.4f}|{row['validation_score']:.6f}|"
                     f"{row['score']:.6f}|{row['iou']:.6f}|{row['seg_acc']:.6f}|"
                     f"{row['pd']:.6f}|{row['fa']:.8g}|")
    lines.extend(["", "配对差值：", ""])
    for row in comparisons:
        lines.append(f"- {row['comparison']}：ΔScore={row['score']:+.6f}，"
                     f"ΔIoU={row['iou']:+.6f}，ΔPd={row['pd']:+.6f}，ΔFa={row['fa']:+.8g}")
    lines.extend(["", "seed37单次训练；不据此声称多种子稳定性或统计显著性。"])
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(root / "status.json", {"status": "complete", "completed_variants": 3})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("suite", "dataset-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    evaluate(parser.parse_args())
