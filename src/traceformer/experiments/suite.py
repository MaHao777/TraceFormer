"""论文消融 DAG、配对配置及结果汇总。"""

from __future__ import annotations

import csv
import io
import json
import shutil
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import yaml

from traceformer.config import ModelConfig, TraceFormerConfig, load_config
from traceformer.experiments.ablation import AblationSpec
from traceformer.training.engine import (
    _atomic_write_text,
    _dataset,
    _dataset_manifest,
    _file_sha256,
    _manifest_identity_digest,
    _serializable,
    _validate_checkpoint_contract,
    _write_json,
)


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
    return jobs


def job_config(
    base: TraceFormerConfig, job: Job, seed: int, output: Path, *,
    epochs: int | None = None, save_training_state_interval: int = 10000,
) -> TraceFormerConfig:
    job.spec.validate_stage(job.stage)
    return replace(base, training=replace(
        base.training, stage=job.stage, seed=seed, device="cpu", mixed_precision=False,
        epochs=epochs if epochs is not None else (18 if job.stage == "full" else 24),
        save_training_state_interval=save_training_state_interval,
        output_root=output,
    )).validate()


def simple_stage_jobs() -> list[Job]:
    # E1/E2 复用，队列只补训新三锥完整网络精修。
    return [
        Job("B", "full", AblationSpec(sparse=False, refinement="none"), None, "E1"),
        Job("BS", "sparse_mixture", AblationSpec(refinement="none"), "B", "E2"),
        Job("BSR", "refinement", AblationSpec(), "BS", "E3"),
    ]


def pooling_stage_jobs() -> list[Job]:
    # 先用新运行时复评历史 attention 基线，再从头训练两组编码替代方案。
    return [
        Job("attention_B", "full", AblationSpec(sparse=False, refinement="none"), None, "T2"),
        Job("statistics_B", "full", AblationSpec(sparse=False, refinement="none", pooling="statistics"), None, "T0"),
        Job("mean_B", "full", AblationSpec(sparse=False, refinement="none", pooling="mean"), None, "T1"),
    ]


def attention_stage_jobs() -> list[Job]:
    # 三组均从相同种子随机初始化，不复用含 Sparse/精修梯度的旧主干。
    return [
        Job(f"{backbone}_B", "full", AblationSpec(backbone, False, "none"), None, f"C{i}")
        for i, backbone in enumerate(("pointwise", "box", "cone"))
    ]


def prepare_attention_suite(base: TraceFormerConfig, root: Path, seeds: list[int]) -> dict:
    if base.model != ModelConfig():
        raise ValueError("attention suite requires the registered tokenizer and three-cone model")
    # 准备时锁定数据；每个训练进程开始前再次核对，防止三组使用不同文件。
    dataset_manifest = _dataset_manifest(_dataset(base, "train"), _dataset(base, "val"))
    manifest = prepare_suite(
        base, root, seeds, definitions=attention_stage_jobs(), epochs=24,
        save_training_state_interval=1,
    )
    manifest.update(
        protocol="attention_three_variants",
        dataset_manifest_sha256=_manifest_identity_digest(dataset_manifest),
        comparisons=[["C1", "C0"], ["C2", "C1"], ["C2", "C0"]],
        figure_variants=["C0", "C1", "C2"],
        variant_labels={"C0": "逐 Token MLP", "C1": "三尺度固定窗口注意力",
                        "C2": "三锥注意力"},
        attention_protocol={
            "epochs": 24, "initialization": "random, paired common weights and RNG",
            "cone_windows": [[8, 2], [4, 4], [2, 8]],
            "box_windows": [[6, 2], [3, 4], [1, 8]],
            "box_volume_relative_difference": [338 / 370 - 1, 196 / 164 - 1, 72 / 64 - 1],
            "parameters": {"C0": 164077, "C1": 163972, "C2": 163972},
            "pointwise_hidden_dim": 164,
            "limitations": "approximate volume match; validation selects checkpoint/threshold; single-seed evidence",
        },
    )
    _write_json(root / "dataset_manifest.json", dataset_manifest)
    _write_json(root / "suite.json", manifest)
    return manifest


def prepare_pooling_suite(source: Path, root: Path) -> dict:
    """锁定同协议 attention checkpoint；新队列必须重新复评才能完成基线。"""
    source = source.resolve()
    original = json.loads((source / "suite.json").read_text(encoding="utf-8"))
    job = next(j for j in original["jobs"] if j["id"] == "seed37/B")
    if _file_sha256(Path(job["config"])) != job["config_sha256"]:
        raise ValueError("pooling baseline config hash mismatch")
    base = load_config(job["config"])
    expected = job_config(base, pooling_stage_jobs()[0], 37, base.training.output_root)
    if base != expected or base.training.max_train_files is not None or base.training.max_val_files is not None:
        raise ValueError("pooling baseline training protocol mismatch")
    train, val = _dataset(base, "train"), _dataset(base, "val")
    if len(train) != 99 or len(val) != 24:
        raise ValueError("pooling ablation requires 99 train / 24 validation files")
    done = json.loads((source / "seed37/B/done.json").read_text(encoding="utf-8"))
    checkpoint = Path(done["checkpoint"])
    if _file_sha256(checkpoint) != done["sha256"]:
        raise ValueError("pooling baseline checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _validate_checkpoint_contract(saved, base, ablation=pooling_stage_jobs()[0].spec)
    if saved.get("training_seed") != 37 or saved.get("training_stage") != "full":
        raise ValueError("pooling baseline seed/stage mismatch")
    if saved["dataset_manifest_sha256"] != _manifest_identity_digest(_dataset_manifest(train, val)):
        raise ValueError("pooling baseline dataset changed")
    run = checkpoint.parent
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    report = json.loads(Path(done["report"]).read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if len(history) != 18 or report["metrics"] != metrics or metrics != json.loads(
        (run / "reproduced_metrics.json").read_text(encoding="utf-8")
    ) or metrics["validation_files"] != 24:
        raise ValueError("pooling baseline is not fully trained and independently verified")
    manifest = prepare_suite(base, root, [37], definitions=pooling_stage_jobs())
    manifest.update(
        protocol="pooling_three_variants", reused_from=str(source),
        comparisons=[["T1", "T0"], ["T2", "T1"], ["T2", "T0"]],
        figure_variants=["T0", "T1", "T2"],
        variant_labels={"T0": "七维统计 MLP 编码", "T1": "逐事件 MLP + 均值池化", "T2": "逐事件 MLP + 四查询 Attention pooling"},
        pooling_protocol={
            "shared": "7 block statistics, position, token mapping, cone backbone, decoder inputs, seed 37, 18 epochs",
            "statistics_encoder": "7 -> 64 -> 64, LayerNorm and GELU",
            "mean_encoder": "shared event MLP and value projection -> arithmetic mean -> 64 -> 192 -> 64 MLP",
            "capacity_note": "mean 152010 vs attention 151882 parameters; statistics 122762; no dummy parameters",
            "baseline_checkpoint": str(checkpoint), "baseline_sha256": done["sha256"],
        },
    )
    attempt = root / "seed37/attention_B/attempt_001"
    attempt.mkdir(parents=True)
    # 不复制 done 标记，新队列会先重载基线并完整复评。
    for name in ("best.pt", "metrics.json", "history.json", "config.yaml", "dataset_manifest.json", "ablation.json", "summary.json"):
        shutil.copy2(run / name, attempt / name)
    _write_json(root / "suite.json", manifest)
    return manifest


def prepare_suite(
    base: TraceFormerConfig, root: Path, seeds: list[int], *, definitions: list[Job] | None = None,
    epochs: int | None = None, save_training_state_interval: int = 10000,
) -> dict:
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be nonempty and unique")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    jobs = []
    for seed in seeds:
        for job in definitions if definitions is not None else stage_jobs():
            identifier = f"seed{seed}/{job.name}"
            config = job_config(base, job, seed, root, epochs=epochs,
                                save_training_state_interval=save_training_state_interval)
            config_path = root / "configs" / f"seed{seed}_{job.name}.yaml"
            config_path.parent.mkdir(exist_ok=True)
            _atomic_write_text(config_path, yaml.safe_dump(
                _serializable(asdict(config)), allow_unicode=True, sort_keys=False,
            ))
            jobs.append({
                "id": identifier, "seed": seed, "name": job.name, "stage": job.stage,
                "spec": job.spec.to_dict(), "result_id": job.result_id,
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


def prepare_simple_suite(source: Path, root: Path) -> dict:
    """核验完整训练和独立复评产物后，建立不影响原队列的四组实验。"""
    source = source.resolve()
    originals = json.loads((source / "suite.json").read_text(encoding="utf-8"))
    by_id = {job["id"]: job for job in originals["jobs"]}
    base = load_config(by_id["seed37/B"]["config"])
    live_manifest = _manifest_identity_digest(_dataset_manifest(_dataset(base, "train"), _dataset(base, "val")))
    reused = {}
    for definition in simple_stage_jobs()[:2]:
        identifier = f"seed37/{definition.name}"
        job = by_id[identifier]
        if _file_sha256(Path(job["config"])) != job["config_sha256"]:
            raise ValueError("reuse source configuration changed")
        done = json.loads((source / identifier / "done.json").read_text(encoding="utf-8"))
        checkpoint_path = Path(done["checkpoint"])
        if _file_sha256(checkpoint_path) != done["sha256"]:
            raise ValueError("reuse checkpoint hash mismatch")
        config = load_config(job["config"])
        expected = job_config(base, definition, 37, Path(config.training.output_root))
        if _serializable(asdict(config)) != _serializable(asdict(expected)):
            raise ValueError("reuse configuration differs from paired training protocol")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _validate_checkpoint_contract(checkpoint, config, ablation=definition.spec)
        if checkpoint.get("training_seed") != 37 or checkpoint.get("training_stage") != definition.stage:
            raise ValueError("reuse seed/stage mismatch")
        if checkpoint.get("dataset_manifest_sha256") != live_manifest:
            raise ValueError("reuse dataset changed")
        run = checkpoint_path.parent
        history = json.loads((run / "history.json").read_text(encoding="utf-8"))
        if len(history) != expected.training.epochs:
            raise ValueError("reuse training did not finish its full epoch budget")
        report = json.loads(Path(done["report"]).read_text(encoding="utf-8"))
        saved = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        reproduced = json.loads((run / "reproduced_metrics.json").read_text(encoding="utf-8"))
        if report["metrics"] != saved or reproduced != saved or saved["validation_files"] != 24:
            raise ValueError("reuse independent 24-file evaluation mismatch")
        if definition.name == "BS":
            if checkpoint["full_initialization"]["sha256"] != reused["seed37/B"]["sha256"]:
                raise ValueError("reuse Sparse checkpoint has a different base")
        reused[identifier] = {**done, "reused_from": str(source / identifier),
                              "report_sha256": _file_sha256(Path(done["report"]))}
    manifest = prepare_suite(base, root, [37], definitions=simple_stage_jobs())
    manifest.update(
        protocol="simple_three_variants", reused_from=str(source),
        comparisons=[["E2", "E1"], ["E3", "E2"]],
        figure_variants=["E1", "E2", "E3"],
        variant_labels={"E1": "时空锥主干 B", "E2": "B + Sparse", "E3": "B + Sparse + 精修"},
    )
    for identifier, done in reused.items():
        directory = root / identifier
        directory.mkdir(parents=True)
        _write_json(directory / "done.json", done)
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
    comparisons = suite.get("comparisons", [
        ("A1", "A0"), ("A2", "A0"), ("A3", "A1"), ("A3", "A2"),
        ("A3", "A7"),
    ])
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
    total_results = sum(job["result_id"] is not None for job in suite["jobs"])
    lines = ["# TraceFormer CPU 核心消融", "", f"完成 {len(rows)}/{total_results} 份结果。",
             "验证集参与 checkpoint 与阈值选择，结果不代表独立测试集。", "",
             "|配置|种子数|Score|IoU|SegAcc|Pd|Fa|", "|---|---:|---:|---:|---:|---:|---:|"]
    for variant, result in aggregate.items():
        cells = []
        for metric in ("score", "iou", "seg_acc", "pd", "fa"):
            item = result[metric]
            deviation = f" ± {item['std']:.6g}" if item["std"] is not None else " (单种子)"
            cells.append(f"{item['mean']:.6g}{deviation}")
        lines.append(f"|{variant}|{len(result['completed_seeds'])}|" + "|".join(cells) + "|")
    if suite.get("variant_labels"):
        lines.extend(["", "配置说明：", ""])
        lines.extend(f"- {key}：{value}" for key, value in suite["variant_labels"].items())
    if suite.get("protocol") in {"pooling_three_variants", "attention_three_variants"}:
        note = ("所有组关闭 Sparse 和精修；三组从头训练24轮，C0/C1/C2近似匹配参数量。"
                "固定窗口仅近似等体积，实际邻居数见各组 ablation_report.json。"
                if suite["protocol"] == "attention_three_variants" else
                "所有组关闭 Sparse 和精修；T1/T2 近似匹配参数量，T0 为较小统计基线。"
                "T2 复用完整训练后重新复评，T0/T1 从头训练。")
        lines.extend(["", note, "", "|配置|阈值|Score@0.5|参数量|秒/序列|峰值RSS MiB|",
                      "|---|---:|---:|---:|---:|---:|"])
        for row in sorted(rows, key=lambda r: r["variant"]):
            lines.append(f"|{row['variant']}|{row['threshold']:.4f}|{row['score_threshold_05']:.6f}|{row['parameters']}|{row['seconds_per_sequence']:.3f}|{row['process_peak_rss_bytes'] / 2**20:.1f}|")
        lines.extend(["", "同种子差值：", ""])
        for row in paired:
            lines.append(f"- {row['comparison']}：ΔScore={row['score']:+.6f}，ΔIoU={row['iou']:+.6f}，ΔPd={row['pd']:+.6f}，ΔFa={row['fa']:+.8g}")
    _atomic_write_text(root / "results.md", "\n".join(lines) + "\n")
