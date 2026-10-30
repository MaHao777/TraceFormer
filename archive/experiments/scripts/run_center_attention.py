"""四组结构实验：冻结源码、审计复用、真实资源门槛、训练和自动完整复评。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import psutil
import torch

from traceformer.config import load_config
from traceformer.data import NPZEventDataset
from traceformer.experiments.suite import summarize
from traceformer.training.engine import (
    _dataset_manifest,
    _file_sha256,
    _manifest_identity_digest,
    _write_json,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_logged(command, path, environment):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"running {' '.join(map(str, command))}; log={path}", flush=True)
    with path.open("a", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=True)


def reuse_a3(root, source_root, environment):
    """只复用同协议旧C2，且当前代码必须逐事件精确复现其缓存。"""
    suite = read(root / "suite.json")
    target = next(j for j in suite["jobs"] if j["result_id"] == "A3")
    if (root / target["id"] / "done.json").exists():
        return
    old_suite = read(source_root / "suite.json")
    old_job = next(j for j in old_suite["jobs"] if j["result_id"] == "C2")
    done = read(source_root / old_job["id"] / "done.json")
    original = Path(done["checkpoint"]).parent
    audit = {"source": str(source_root), "source_checkpoint_sha256": done["sha256"],
             "accepted": False}
    try:
        if (old_suite.get("protocol") != "attention_three_variants"
                or old_job["spec"] != target["spec"]
                or old_suite["dataset_manifest_sha256"] != suite["dataset_manifest_sha256"]
                or _file_sha256(Path(done["checkpoint"])) != done["sha256"]
                or _file_sha256(Path(old_job["config"])) != old_job["config_sha256"]):
            raise ValueError("old C2 identity, dataset or contract differs")
        left, right = asdict(load_config(old_job["config"])), asdict(load_config(target["config"]))
        for config in (left, right):
            config["training"].pop("output_root")
        if left != right or len(read(original / "history.json")) != 24:
            raise ValueError("old C2 training protocol differs")
        if read(original / "metrics.json") != read(original / "reproduced_metrics.json"):
            raise ValueError("old C2 was not independently reproduced")
        # 核验训练/数据/模型源码；仅记录已知的前缀求和推理优化差异。
        old_hashes = read(source_root / "source_hashes.json")
        new_hashes = read(root / "source_hashes.json")
        differences = []
        for relative, digest in old_hashes.items():
            normalized = relative.replace("\\", "/")
            if normalized.startswith(("src/traceformer/models/", "src/traceformer/training/",
                                      "src/traceformer/data/")) or normalized == "src/traceformer/config.py":
                if new_hashes.get(relative) != digest:
                    differences.append(normalized)
        audit["runtime_differences"] = differences
        if set(differences) - {"src/traceformer/models/prefix_cone.py"}:
            raise ValueError("unapproved model/training/data differences require a fresh A3")
    except ValueError as error:
        audit["reason"] = str(error)
        _write_json(root / "reuse_a3.json", audit)
        return
    attempt = root / target["id"] / "attempt_001"
    attempt.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "config.yaml", "metrics.json", "history.json", "summary.json",
                 "ablation.json", "dataset_manifest.json"):
        shutil.copy2(original / name, attempt / name)
    command = [sys.executable, "-u", str(Path(suite["source_snapshot"]) / "scripts/run_paper_ablation.py"),
               "--suite", str(root), "--worker", target["id"]]
    run_logged(command, root / "logs/reuse_A3.log", environment)
    matches = {}
    for path in sorted((original / "predictions").glob("*.pt")):
        old = torch.load(path, map_location="cpu", weights_only=True)
        new = torch.load(attempt / "predictions" / path.name, map_location="cpu", weights_only=True)
        matches[path.stem] = bool(torch.equal(old["event_logits"], new["event_logits"]))
    audit.update(per_sequence_logits_equal=matches, accepted=len(matches) == 24 and all(matches.values()))
    _write_json(root / "reuse_a3.json", audit)
    if not audit["accepted"]:
        # 将拒绝记录保存在本次实验目录内，随后从头训练A3。
        rejected = attempt.with_name("reuse_rejected")
        if not attempt.resolve().is_relative_to(root) or not rejected.resolve().is_relative_to(root):
            raise ValueError("reuse rejection paths must stay in the experiment directory")
        attempt.rename(rejected)
        (root / target["id"] / "done.json").unlink()
    summarize(root)


def execute(args):
    root = args.suite.resolve()
    repo = Path(__file__).resolve().parents[1]
    if (root / "execution.json").exists():
        active = read(root / "execution.json")
        pid = active.get("pid")
        if active.get("status") == "running" and pid and pid != os.getpid() and psutil.pid_exists(pid):
            if str(root) in psutil.Process(pid).cmdline():
                print(f"experiment already running under PID {pid}: {root}", flush=True)
                return
    if not root.exists():
        subprocess.run([sys.executable, "-u", str(repo / "scripts/run_paper_ablation.py"),
                        "--prepare-center-attention", "--config", str(args.config.resolve()),
                        "--suite", str(root), "--seeds", "37"], check=True)
    suite = read(root / "suite.json")
    if suite["protocol"] != "center_attention_four_variants" or suite["seeds"] != [37]:
        raise ValueError("requires the confirmed four-group seed37 protocol")
    test_root = args.dataset_root.resolve()
    if len(list((test_root / "test").glob("*.npz"))) != 24:
        raise ValueError("requires all 24 test sequences")
    if suite.get("test_dataset_root", str(test_root)) != str(test_root):
        raise ValueError("test source changed when resuming")
    test_manifest = _dataset_manifest(NPZEventDataset(test_root / "test"))
    test_digest = _manifest_identity_digest(test_manifest)
    if suite.get("test_dataset_manifest_sha256", test_digest) != test_digest:
        raise ValueError("test content changed when resuming")
    suite["test_dataset_root"] = str(test_root)
    suite["test_dataset_manifest_sha256"] = test_digest
    _write_json(root / "test_dataset_manifest.json", test_manifest)
    _write_json(root / "suite.json", suite)
    source = Path(suite["source_snapshot"])
    environment = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONIOENCODING="utf-8",
                       PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
                       NUMBA_NUM_THREADS="8", TRACEFORMER_LOCAL_CPU_BACKEND=suite.get(
                           "runtime_overrides", {}).get("local_cpu_backend", "torch"))
    def phase(name):
        _write_json(root / "execution.json", {"status": "running", "phase": name,
                                               "pid": os.getpid(), "updated_at": time.time()})
    phase("reuse_a3")
    if args.reuse_from:
        reuse_a3(root, args.reuse_from.resolve(), environment)
    phase("resource_preflight")
    for job in suite["jobs"]:
        for sequence in ("train_096", "train_094"):
            variant = job["result_id"]
            output = root / "preflight" / f"{variant}_{sequence}.json"
            if not output.exists():
                command = [sys.executable, "-u", str(source / "scripts/check_paper_ablation.py"),
                           "--config", job["config"], "--backbone", job["spec"]["backbone"],
                           "--sequence", f"{sequence}.npz", "--output", str(output)]
                run_logged(command, root / "logs" / f"preflight_{variant}_{sequence}.log", environment)
            report = read(output)
            if (report["status"] != "passed" or report["spec"] != job["spec"]
                    or report["config_sha256"] != job["config_sha256"]
                    or report["source_sha256"] != _file_sha256(
                        Path(load_config(job["config"]).dataset.root) / "train" / f"{sequence}.npz")):
                raise ValueError("resource preflight did not pass")
    phase("training_and_validation")
    run_logged([sys.executable, "-u", str(source / "scripts/run_paper_ablation.py"),
                "--suite", str(root)], root / "logs/queue.log", environment)
    phase("test_independent_scoring_and_benchmark")
    run_logged([sys.executable, "-u", str(source / "scripts/finish_center_attention.py"),
                "--suite", str(root)], root / "logs/finalization.log", environment)
    _write_json(root / "execution.json", {"status": "complete", "updated_at": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/traceformer_prefix_cone.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("../EV-UAV-dataset"))
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args()
    try:
        execute(args)
    except BaseException:
        if args.suite.exists():
            _write_json(args.suite / "execution.json", {"status": "failed", "traceback": traceback.format_exc(),
                                                       "updated_at": time.time()})
        raise
