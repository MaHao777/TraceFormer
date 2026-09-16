"""论文消融 DAG、配对配置及结果汇总。"""

from __future__ import annotations

import csv
import io
import json
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from traceformer.config import TraceFormerConfig
from traceformer.experiments.ablation import AblationSpec
from traceformer.training.engine import _atomic_write_text, _file_sha256, _serializable, _write_json


@dataclass(frozen=True)
class Job:
    name: str
    stage: str
    spec: AblationSpec
    parent: str | None
    result_id: str | None


def stage_jobs() -> list[Job]:
    # 公共上游只训练一次；机制消融各自重训主干与下游。
    jobs = [
        Job("B", "full", AblationSpec(sparse=False, refinement="none"), None, "A0"),
        Job("BS", "sparse_mixture", AblationSpec(refinement="none"), "B", "A1"),
        Job("BR", "refinement", AblationSpec(sparse=False), "B", "A2"),
        Job("BSR", "refinement", AblationSpec(), "BS", "A3"),
        Job("same_scale", "refinement", AblationSpec(refinement="same_scale"), "BS", "A7"),
    ]
    for backbone, result_id in (("window", "A4"), ("past", "A5"), ("no_position", "A6")):
        jobs.extend([
            Job(f"{backbone}_B", "full", AblationSpec(backbone, False, "none"), None, None),
            Job(f"{backbone}_BS", "sparse_mixture", AblationSpec(backbone, True, "none"),
                f"{backbone}_B", None),
            Job(f"{backbone}_BSR", "refinement", AblationSpec(backbone),
                f"{backbone}_BS", result_id),
        ])
    return jobs


def job_config(base: TraceFormerConfig, job: Job, seed: int, output: Path) -> TraceFormerConfig:
    job.spec.validate_stage(job.stage)
    return replace(base, training=replace(
        base.training, stage=job.stage, seed=seed, device="cpu", mixed_precision=False,
        epochs=18 if job.stage == "full" else 24, save_training_state_interval=10000,
        output_root=output,
    )).validate()


def prepare_suite(base: TraceFormerConfig, root: Path, seeds: list[int]) -> dict:
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be nonempty and unique")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    jobs = []
    for seed in seeds:
        for job in stage_jobs():
            identifier = f"seed{seed}/{job.name}"
            config = job_config(base, job, seed, root)
            config_path = root / "configs" / f"seed{seed}_{job.name}.yaml"
            config_path.parent.mkdir(exist_ok=True)
            _atomic_write_text(config_path, yaml.safe_dump(
                _serializable(asdict(config)), allow_unicode=True, sort_keys=False,
            ))
            jobs.append({
                "id": identifier, "seed": seed, "name": job.name, "stage": job.stage,
                "spec": asdict(job.spec), "result_id": job.result_id,
                "config": str(config_path.resolve()),
                "config_sha256": _file_sha256(config_path),
                "parent": f"seed{seed}/{job.parent}" if job.parent else None,
            })
    manifest = {
        "schema": 1, "seeds": seeds, "jobs": jobs, "device": "cpu", "precision": "float32",
        "evaluation": "99 train / 24 validation; validation selects checkpoints and thresholds",
        "root": str(root.resolve()), "source_snapshot": str((root / "source").resolve()),
    }
    _write_json(root / "suite.json", manifest)
    return manifest


def summarize(root: Path) -> None:
    """逐步生成完整 24 文件指标表，并保留配对差值和未完成状态。"""
    suite = json.loads((root / "suite.json").read_text(encoding="utf-8"))
    rows = []
    groups = []
    for job in suite["jobs"]:
        done_path = root / job["id"] / "done.json"
        if not job["result_id"] or not done_path.exists():
            continue
        done = json.loads(done_path.read_text(encoding="utf-8"))
        report = json.loads(Path(done["report"]).read_text(encoding="utf-8"))
        rows.append({
            "variant": job["result_id"], "seed": job["seed"],
            **{k: report["metrics"]["competition"][k]
               for k in ("score", "iou", "seg_acc", "pd", "fa")},
            "threshold": report["metrics"]["threshold"],
            "parameters": report["parameters"],
            "seconds_per_sequence": report["mean_seconds_per_sequence"],
            "process_peak_rss_bytes": report["process_peak_rss_bytes"],
            "score_threshold_05": report["fixed_threshold"]["competition"]["score"],
        })
        for group, result in report["groups"].items():
            groups.append({"variant": job["result_id"], "seed": job["seed"], "group": group,
                           "files": result["validation_files"], **{
                               k: result["competition"][k] for k in ("score", "iou", "pd", "fa")
                           }})
    for name, records in (("results.csv", rows), ("groups.csv", groups)):
        if records:
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(sorted(records, key=lambda row: (row["variant"], row["seed"])))
            _atomic_write_text(root / name, stream.getvalue())
    aggregate = {}
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            "completed_seeds": [row["seed"] for row in selected],
            **{metric: {"mean": statistics.mean(row[metric] for row in selected),
                        "std": statistics.stdev(row[metric] for row in selected)
                        if len(selected) > 1 else None}
               for metric in ("score", "iou", "seg_acc", "pd", "fa")},
        }
    paired = []
    comparisons = [("A1", "A0"), ("A2", "A0"), ("A3", "A1"), ("A3", "A2"),
                   ("A3", "A4"), ("A3", "A5"), ("A3", "A6"), ("A3", "A7")]
    for seed in suite["seeds"]:
        by_id = {row["variant"]: row for row in rows if row["seed"] == seed}
        for left, right in comparisons:
            if left in by_id and right in by_id:
                paired.append({"seed": seed, "comparison": f"{left}-{right}", **{
                    k: by_id[left][k] - by_id[right][k] for k in ("score", "iou", "pd", "fa")
                }})
        if all(key in by_id for key in ("A0", "A1", "A2", "A3")):
            paired.append({"seed": seed, "comparison": "A3-A2-A1+A0", **{
                k: by_id["A3"][k] - by_id["A2"][k] - by_id["A1"][k] + by_id["A0"][k]
                for k in ("score", "iou", "pd", "fa")
            }})
    _write_json(root / "aggregate.json", {"aggregate": aggregate, "paired": paired})
    lines = ["# TraceFormer CPU 核心消融", "", f"完成 {len(rows)}/{8 * len(suite['seeds'])} 份结果。",
             "验证集参与 checkpoint 与阈值选择，结果不代表独立测试集。", "",
             "|配置|种子数|Score|IoU|SegAcc|Pd|Fa|", "|---|---:|---:|---:|---:|---:|---:|"]
    for variant, result in aggregate.items():
        cells = []
        for metric in ("score", "iou", "seg_acc", "pd", "fa"):
            item = result[metric]
            deviation = f" ± {item['std']:.6g}" if item["std"] is not None else " (单种子)"
            cells.append(f"{item['mean']:.6g}{deviation}")
        lines.append(f"|{variant}|{len(result['completed_seeds'])}|" + "|".join(cells) + "|")
    _atomic_write_text(root / "results.md", "\n".join(lines) + "\n")
