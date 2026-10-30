"""建立三种子核心机制队列，审计并重新复评 seed37 的完整归档。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
from run_paper_ablation import source_snapshot

from traceformer.config import load_config
from traceformer.experiments.ablation import AblationSpec
from traceformer.experiments.suite import Job, job_config, prepare_suite
from traceformer.training.engine import (
    _dataset,
    _dataset_manifest,
    _file_sha256,
    _manifest_identity_digest,
    _serializable,
    _validate_checkpoint_contract,
    _write_json,
)

REPO = Path(__file__).resolve().parents[3]
DEFINITIONS = [
    Job("cone", "full", AblationSpec(sparse=False, refinement="none"), None, "C"),
    Job("window", "full", AblationSpec("window", False, "none"), None, "W"),
    Job("mean", "full", AblationSpec(sparse=False, refinement="none", pooling="mean"), None, "M"),
]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(root: Path) -> dict:
    pooling = REPO / "outputs/paper_pooling_cpu_20260918_234003"
    simple = REPO / "outputs/paper_ablation_simple_cpu_20260917_231507"
    originals = [pooling / "seed37/attention_B", simple / "seed37/window_B",
                 pooling / "seed37/mean_B"]
    base = load_config(read(originals[0] / "done.json")["config"])
    datasets = [_dataset(base, split) for split in ("train", "val")]
    if list(map(len, datasets)) != [99, 24]:
        raise ValueError("formal protocol requires all 99 train and 24 validation files")
    live_digest = _manifest_identity_digest(_dataset_manifest(*datasets))
    # 复用必须来自当前相同模型/训练/数据实现，汇总和绘图脚本不参与模型定义。
    for relative, digest in read(pooling / "source_hashes.json").items():
        normalized = relative.replace("\\", "/")
        if normalized.startswith(("src/traceformer/models/", "src/traceformer/training/",
                                  "src/traceformer/data/")) or normalized in {
            "src/traceformer/config.py", "src/traceformer/experiments/ablation.py",
            "src/traceformer/experiments/pooling.py",
        }:
            if _file_sha256(REPO / relative) != digest:
                raise ValueError(f"archived training implementation differs: {relative}")
    audited = []
    for definition, original in zip(DEFINITIONS, originals, strict=True):
        done = read(original / "done.json")
        config = load_config(done["config"])
        expected = job_config(base, definition, 37, config.training.output_root)
        if _serializable(asdict(config)) != _serializable(asdict(expected)):
            raise ValueError(f"paired training configuration differs: {definition.name}")
        checkpoint = Path(done["checkpoint"])
        if _file_sha256(checkpoint) != done["sha256"]:
            raise ValueError("archived checkpoint hash changed")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _validate_checkpoint_contract(saved, config, ablation=definition.spec)
        if saved["training_seed"] != 37 or saved["dataset_manifest_sha256"] != live_digest:
            raise ValueError("archived training seed or dataset differs")
        run = checkpoint.parent
        metrics = read(run / "metrics.json")
        if len(read(run / "history.json")) != 18 or metrics["validation_files"] != 24:
            raise ValueError("archived training/evaluation is incomplete")
        if metrics != read(run / "reproduced_metrics.json") or metrics != read(
            Path(done["report"])
        )["metrics"]:
            raise ValueError("archived independent evaluation mismatch")
        audited.append({"variant": definition.result_id, "checkpoint": str(checkpoint),
                        "sha256": done["sha256"], "report_sha256": _file_sha256(Path(done["report"]))})
    manifest = prepare_suite(base, root, [37, 73, 101], definitions=DEFINITIONS)
    manifest.update(
        protocol="focused_mechanism_validation", comparisons=[["C", "W"], ["C", "M"]],
        variant_labels={"C": "Attention + query-centered cone", "W": "Attention + fixed window",
                        "M": "Capacity-matched mean + query-centered cone"},
        reuse_audit=audited, claim_gate={
            "all_three_paired_score_differences_positive": True,
            "minimum_mean_score_gain": 0.001,
            "minimum_mean_ordinary_group_score_gain": 0.0,
            "minimum_mean_val023_pd_gain": -0.01,
            "scope": "validation consistency across seeds, not independent-test significance",
        },
    )
    for definition, original in zip(DEFINITIONS, originals, strict=True):
        run = Path(read(original / "done.json")["checkpoint"]).parent
        attempt = root / f"seed37/{definition.name}/attempt_001"
        attempt.mkdir(parents=True)
        for name in ("best.pt", "metrics.json", "history.json", "config.yaml",
                     "dataset_manifest.json", "ablation.json", "summary.json"):
            shutil.copy2(run / name, attempt / name)
    # 不复制 done；队列首先独立重载和复评，再运行六个新训练。
    _write_json(root / "suite.json", manifest)
    source_snapshot(root)
    _write_json(root / "status.json", {"status": "prepared", "pending_reevaluations": 3,
                                       "pending_training_stages": 6, "updated_at": time.time()})
    print(root, flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    prepare(parser.parse_args().suite.resolve())
