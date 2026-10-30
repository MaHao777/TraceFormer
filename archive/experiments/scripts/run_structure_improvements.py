"""复用00，冻结并串行训练10/01/11，自动完成验证与测试预测。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import psutil
import torch
import yaml
from run_component_core import (
    evaluate_and_audit,
    execute,
    process_alive,
    verify_checks,
)
from run_paper_ablation import completed_job, queue_lock, source_snapshot

from traceformer.config import load_config
from traceformer.experiments.ablation import StructureImprovementSpec
from traceformer.experiments.component_suite import audit_baseline, read_json, resource_gate
from traceformer.experiments.structure import describe_structure
from traceformer.training.engine import (
    _atomic_write_text,
    _file_sha256,
    _serializable,
    _write_json,
    build_model,
)

REPO = Path(__file__).resolve().parents[3]
VARIANTS = ("00", "10", "01", "11")
SEQUENCES = ("train_094.npz", "train_096.npz")


def environment(source):
    return dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONUNBUFFERED="1",
                PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="8",
                MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1")


def prepare(root, baseline, full):
    if not (root / "source").exists():
        source_snapshot(root)
    source = root / "source"
    config = load_config(full["config"])
    # 同一00权重分别使用原始和当前源码，比较24文件逐事件输出。
    for label, runtime in (("original", baseline / "source"), ("current", source)):
        output = root / "equivalence" / label
        if not (output / "report.json").exists():
            command = [sys.executable, "-u", str(source / "scripts/check_component_full_equivalence.py"),
                       "--config", full["config"], "--checkpoint", full["checkpoint"],
                       "--output", str(output)]
            execute(root, "equivalence_" + label, command, environment(runtime))
    old_dir, new_dir = root / "equivalence/original", root / "equivalence/current"
    if len(list(old_dir.glob("*.pt"))) != 24 or len(list(new_dir.glob("*.pt"))) != 24:
        raise ValueError("00 equivalence requires all 24 files")
    for path in sorted(old_dir.glob("*.pt")):
        old, new = torch.load(path, weights_only=True), torch.load(new_dir / path.name, weights_only=True)
        if any(not torch.equal(old[key], new[key]) for key in old):
            raise ValueError(f"00 source equivalence failed: {path.name}")
    _write_json(root / "baseline_equivalence.json", {"exact": True, "files": 24, "baseline": full})
    expected = read_json(full["test_report"])["files"]
    candidates = [REPO.parent / "EV-UAV-dataset", Path(config.dataset.root)]
    matches = [p for p in candidates if {f.name for f in (p / "test").glob("*.npz")} == set(expected)
               and all(_file_sha256(p / "test" / name) == row["source_sha256"]
                       for name, row in expected.items())]
    if not matches:
        raise ValueError("test partition differs from 00")
    (root / "configs").mkdir(exist_ok=True)
    jobs = []
    for variant in VARIANTS[1:]:
        path = root / "configs" / f"{variant}.yaml"
        current = replace(config, training=replace(config.training, output_root=root,
                                                  save_training_state_interval=1))
        _atomic_write_text(path, yaml.safe_dump(_serializable(asdict(current)),
                                               allow_unicode=True, sort_keys=False))
        spec = StructureImprovementSpec(variant)
        torch.manual_seed(config.training.seed)
        model = build_model(config, ablation=spec)
        jobs.append({"id": f"seed37/{variant}", "name": variant, "stage": "full", "seed": 37,
                     "spec": spec.to_dict(), "result_id": variant, "parent": None,
                     "config": str(path), "config_sha256": _file_sha256(path),
                     "actual_structure": describe_structure(model)})
    suite = {"schema": 3, "protocol": "structure_improvements", "seeds": [37], "jobs": jobs,
             "source_snapshot": str(source), "baseline": full,
             "dataset_manifest_sha256": full["dataset_manifest_sha256"],
             "test_root": str(matches[0].resolve()), "baseline_retrained": False,
             "embedding_formula": "0.75*(mean(event_encoder(events))+Linear(statistics)) + 0.25*Query(events)",
             "limitations": "single seed; historical test inspected; B bundles direction/gating/depth; full network noncausal"}
    _write_json(root / "suite.json", suite)
    return suite


def validate_suite(root, suite, full):
    if suite["baseline"] != full:
        raise ValueError("00 identity changed")
    for name, digest in read_json(root / "source_hashes.json").items():
        if _file_sha256(root / "source" / name) != digest:
            raise ValueError(f"frozen source changed: {name}")
    for job in suite["jobs"]:
        if _file_sha256(Path(job["config"])) != job["config_sha256"]:
            raise ValueError("job configuration changed")


def gates(root, suite, env):
    records = []
    for job in suite["jobs"]:
        for sequence in SEQUENCES:
            path = root / "preflight" / f"{job['name']}_{Path(sequence).stem}.json"
            if not path.exists():
                execute(root, path.stem, [sys.executable, "-u",
                        str(root / "source/scripts/run_structure_improvements.py"),
                        "--root", str(root), "--gate", job["name"], "--config", job["config"],
                        "--sequence", sequence, "--output", str(path)], env, capture=path)
            record = read_json(path)
            if (record["variant"] != job["spec"] or record["sequence"] != sequence
                    or not record["finite_gradients"] or not record["memory_passed"]):
                raise ValueError(f"resource gate failed: {path}")
            records.append(record)
    _write_json(root / "resource_gate.json", {"passed": True, "records": records})


def summarize(root, suite):
    rows = [read_json(root / "results" / f"{v}.json") for v in VARIANTS
            if (root / "results" / f"{v}.json").exists()]
    lines = ["# A/B结构对照", "", "00复用；10仅A、01仅B、11同时A+B。均为神经网络，无额外后处理。", ""]
    for split in ("val", "test"):
        lines += [f"## {split}", "", "|组|参数|AP|IoU|SegAcc|Pd|Fa|Score|阈值|相对00|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        baseline = next((r[split]["competition"]["score"] for r in rows if r["variant"] == "00"), None)
        for row in rows:
            report, metrics = row[split], row[split]["competition"]
            delta = f"{metrics['score'] - baseline:+.6f}" if baseline is not None else "pending"
            lines.append(f"|{row['variant']}|{report['parameters']}|{report['average_precision']:.6f}|"
                         f"{metrics['iou']:.6f}|{metrics['seg_acc']:.6f}|{metrics['pd']:.6f}|"
                         f"{metrics['fa']:.8g}|{metrics['score']:.6f}|{report['threshold']:.6f}|{delta}|")
        lines.append("")
    complete = {r["variant"] for r in rows} == set(VARIANTS)
    _write_json(root / "results.json", {"rows": rows, "complete": complete, "baseline": suite["baseline"]})
    _atomic_write_text(root / "results.md", "\n".join(lines))


def finish(root, suite, variant, config_path, checkpoint, env, spec=None):
    config = load_config(config_path)
    reports = {}
    for split, data in (("val", Path(config.dataset.root)), ("test", Path(suite["test_root"]))):
        output = root / split / variant
        report = evaluate_and_audit(root, f"{split}_{variant}", root / "source", config_path,
                                    checkpoint, data, split, output, env, spec)
        if variant == "00":
            expected = (read_json(Path(checkpoint).parent / "metrics.json")["competition"]
                        if split == "val" else read_json(suite["baseline"]["test_report"])["competition"])
            if report["competition"] != expected:
                raise ValueError("00 reevaluation differs from existing result")
        reports[split] = {key: report[key] for key in (
            "competition", "average_precision", "threshold", "parameters", "checkpoint_sha256")}
        reports[split]["report"] = str(output / "independent_report.json")
        reports[split]["predictions"] = str(output / "predictions")
        reports[split]["probabilities"] = str(output / "probabilities")
    _write_json(root / "results" / f"{variant}.json", {"variant": variant, **reports})
    summarize(root, suite)


def run(root, baseline):
    acquired = False
    try:
        with queue_lock(root):
            acquired = True
            full = audit_baseline(baseline)
            if full is None:
                raise ValueError("00 is not complete")
            verify_checks(root, root / "source" if (root / "source").exists() else REPO)
            suite = (read_json(root / "suite.json") if (root / "suite.json").exists()
                     else prepare(root, baseline, full))
            validate_suite(root, suite, full)
            env = environment(root / "source")
            (root / "results").mkdir(exist_ok=True)
            if not (root / "resource_gate.json").exists():
                gates(root, suite, env)
            finish(root, suite, "00", full["config"], full["checkpoint"], env)
            for job in suite["jobs"]:
                if completed_job(root, job["id"]) is None:
                    execute(root, "train_" + job["name"], [sys.executable, "-u",
                            str(root / "source/scripts/run_paper_ablation.py"), "--suite", str(root),
                            "--worker", job["id"]], env)
                done = completed_job(root, job["id"])
                path = Path(done["checkpoint"])
                finish(root, suite, job["name"], job["config"], path, env, path.parent / "ablation.json")
            hashes = {p.relative_to(root).as_posix(): _file_sha256(p)
                      for directory in ("val", "test", "results", "seed37", "preflight")
                      for p in sorted((root / directory).rglob("*")) if p.is_file()}
            _write_json(root / "completion_audit.json", {"complete": True, "sha256": hashes,
                       "new_training_runs": 3, "baseline_retrained": False,
                       "source_sha256": read_json(root / "source_hashes.json")})
            _write_json(root / "status.json", {"status": "complete", "variants": list(VARIANTS),
                       "new_training_runs": 3, "updated_at": time.time()})
    except Exception:
        if acquired:
            _write_json(root / "status.json", {"status": "failed", "error": traceback.format_exc(),
                                               "updated_at": time.time()})
        raise


def active_processes(root):
    # 同时检查子孙训练/评估进程，协调器意外退出后不重复启动。
    found = []
    for process in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if (process.pid != os.getpid() and "python" in (process.info["name"] or "").lower()
                    and str(root) in command):
                found.append(process.pid)
        except psutil.Error:
            continue
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=REPO / "outputs/no_sparse_local_20261025")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gate", choices=VARIANTS[1:])
    parser.add_argument("--sequence", choices=SEQUENCES)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root, baseline = args.root.resolve(), args.baseline.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.gate:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = resource_gate(load_config(args.config), StructureImprovementSpec(args.gate),
                               args.sequence, repeats=1)
        _write_json(args.output, result)
        if not result["memory_passed"]:
            raise ValueError("resource gate exceeds RAM budget")
    elif args.run:
        run(root, baseline)
    else:
        dispatch = root / "dispatch"
        dispatch.mkdir(exist_ok=True)
        with queue_lock(dispatch):
            status = read_json(root / "status.json") if (root / "status.json").exists() else {}
            active = active_processes(root)
            if active or any(process_alive(status.get(k), root) for k in ("queue_pid", "worker_pid")):
                print(json.dumps({**status, "status": "already_running", "active_pids": active}))
                return
            if status.get("status") == "complete":
                print(json.dumps(status))
                return
            if status.get("status") == "failed":
                raise RuntimeError("prior phase failed; inspect status/logs before recovery with --run")
            verify_checks(root, root / "source" if (root / "source").exists() else REPO)
            with (root / "coordinator.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                           "--root", str(root), "--baseline", str(baseline), "--run"],
                           stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            _write_json(root / "status.json", {"status": "starting", "queue_pid": process.pid,
                                               "updated_at": time.time()})
            print(f"started coordinator PID={process.pid}")


if __name__ == "__main__":
    main()
