"""CPU 论文消融队列：预检、隔离子进程、断点续训与自动汇总。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import psutil

from traceformer.config import load_config
from traceformer.experiments.ablation import AblationSpec
from traceformer.experiments.figures import render_figures, select_windows
from traceformer.experiments.report import evaluate_ablation
from traceformer.experiments.suite import (
    prepare_pooling_suite,
    prepare_simple_suite,
    prepare_suite,
    summarize,
)
from traceformer.training.engine import _file_sha256, _write_json, run_training


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def queue_lock(root: Path):
    # 操作系统文件锁随进程退出释放，避免重复启动两个 CPU 队列。
    with (root / "queue.lock").open("a+b") as stream:
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def completed_job(root: Path, identifier: str) -> dict | None:
    path = root / identifier / "done.json"
    if not path.exists():
        return None
    record = read_json(path)
    if _file_sha256(Path(record["checkpoint"])) != record["sha256"]:
        raise ValueError(f"completed checkpoint changed: {identifier}")
    if record.get("report_sha256") and _file_sha256(Path(record["report"])) != record["report_sha256"]:
        raise ValueError(f"reused report changed: {identifier}")
    return record


def worker(root: Path, identifier: str) -> None:
    suite = read_json(root / "suite.json")
    job = next(item for item in suite["jobs"] if item["id"] == identifier)
    if _file_sha256(Path(job["config"])) != job["config_sha256"]:
        raise ValueError("ablation stage configuration changed")
    config = load_config(job["config"])
    spec = AblationSpec(**job["spec"])
    folder = root / identifier
    folder.mkdir(parents=True, exist_ok=True)
    attempts = sorted(folder.glob("attempt_*"))
    prior = attempts[-1] if attempts else None
    if prior is not None and (prior / "summary.json").exists():
        # 训练已经结束而复评被打断时，直接重做只读复评。
        attempt = prior
    else:
        attempt = folder / f"attempt_{len(attempts) + 1:03d}"
        kwargs = {}
        if prior is not None and (prior / "training_state.pt").exists():
            kwargs["resume_state_path"] = prior / "training_state.pt"
        elif job["parent"]:
            parent = completed_job(root, job["parent"])
            if parent is None:
                raise RuntimeError("upstream job is not complete")
            key = "initial_refinement_checkpoint_path" if job["stage"] == "refinement" else "initial_full_checkpoint_path"
            kwargs[key] = parent["checkpoint"]
        run_training(config, output_dir=attempt, ablation=spec, progress=True,
                     config_source=job["config"], note=f"paper ablation {identifier}", **kwargs)
    checkpoint = attempt / "best.pt"
    # 与训练进程分离复评，释放训练分配器的内存，并测量独立推理峰值。
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--suite", str(root),
               "--evaluate", identifier, "--attempt", str(attempt)]
    subprocess.run(command, check=True)
    _write_json(folder / "done.json", {
        "checkpoint": str(checkpoint.resolve()), "sha256": _file_sha256(checkpoint),
        "report": str((attempt / "ablation_report.json").resolve()),
        "config": job["config"], "ablation": job["spec"], "completed_at": time.time(),
    })


def evaluate_worker(root: Path, identifier: str, attempt: Path) -> None:
    job = next(item for item in read_json(root / "suite.json")["jobs"] if item["id"] == identifier)
    report = evaluate_ablation(load_config(job["config"]), attempt / "best.pt",
                               AblationSpec(**job["spec"]), save_predictions=bool(job["result_id"]))
    if report["metrics"] != read_json(attempt / "metrics.json"):
        raise RuntimeError("independent FP32 metrics do not exactly reproduce saved best metrics")


def source_snapshot(root: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    target = root / "source"
    target.mkdir()
    for directory in ("src", "scripts", "configs"):
        shutil.copytree(repo / directory, target / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "AGENTS.md"):
        shutil.copy2(repo / name, target / name)
    hashes = {str(path.relative_to(target)): _file_sha256(path)
              for path in target.rglob("*") if path.is_file()}
    _write_json(root / "source_hashes.json", hashes)


def queue(root: Path) -> None:
    suite = read_json(root / "suite.json")
    # 父队列意外退出时，已有工作进程可能仍在训练；禁止重复接管。
    status_path = root / "status.json"
    if status_path.exists():
        previous = read_json(status_path)
        worker_pid = previous.get("worker_pid")
        if previous.get("status") == "running" and worker_pid and psutil.pid_exists(worker_pid):
            command = psutil.Process(worker_pid).cmdline()
            if str(root) in command and "--worker" in command:
                raise RuntimeError(f"existing worker is still running: PID {worker_pid}")
    source = Path(suite["source_snapshot"])
    for relative, digest in read_json(root / "source_hashes.json").items():
        if _file_sha256(source / relative) != digest:
            raise ValueError(f"experiment source changed: {relative}")
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    environment = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONUNBUFFERED="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    executable = source / "scripts" / "run_paper_ablation.py"
    started = time.time()
    for index, job in enumerate(suite["jobs"]):
        if _file_sha256(Path(job["config"])) != job["config_sha256"]:
            raise ValueError(f"stage configuration changed: {job['id']}")
        if completed_job(root, job["id"]) is not None:
            continue
        log = logs / f"{job['id'].replace('/', '_')}.log"
        command = [sys.executable, "-u", str(executable), "--suite", str(root), "--worker", job["id"]]
        with log.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=environment, cwd=source)
            _write_json(root / "status.json", {
                "status": "running", "job": job["id"], "stage_index": index + 1,
                "stages_total": len(suite["jobs"]), "queue_pid": os.getpid(), "worker_pid": process.pid,
                "log": str(log), "started_at": started, "updated_at": time.time(),
            })
            print(f"[{index+1}/{len(suite['jobs'])}] {job['id']} PID={process.pid} log={log}", flush=True)
            code = process.wait()
        if code:
            _write_json(root / "status.json", {
                "status": "failed", "job": job["id"], "returncode": code,
                "log": str(log), "updated_at": time.time(),
            })
            raise RuntimeError(f"ablation job failed: {job['id']}, exit={code}")
        summarize(root)
        if job["name"] == "no_position_BSR":
            render_figures(root, job["seed"])
    summarize(root)
    if suite.get("figure_variants"):
        for seed in suite["seeds"]:
            render_figures(root, seed)
    _write_json(root / "status.json", {"status": "complete", "completed_stages": len(suite["jobs"]),
                                       "updated_at": time.time()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepare-simple-from", type=Path)
    parser.add_argument("--prepare-pooling-from", type=Path)
    parser.add_argument("--config", default="configs/traceformer_score_optimized.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[37, 73, 101])
    parser.add_argument("--worker")
    parser.add_argument("--evaluate")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    root = args.suite.resolve()
    if args.prepare_pooling_from:
        # 仅允许此次已审计的聚合入口/实验契约改动，其余模型和训练实现必须相同。
        repo = Path(__file__).resolve().parents[1]
        allowed = {"src/traceformer/models/event_tokenizer.py", "src/traceformer/training/engine.py"}
        hashes = read_json(args.prepare_pooling_from / "source_hashes.json")
        for relative, digest in hashes.items():
            normalized = relative.replace("\\", "/")
            if normalized.startswith(("src/traceformer/models/", "src/traceformer/training/", "src/traceformer/data/")) or normalized == "src/traceformer/config.py":
                if normalized not in allowed and _file_sha256(repo / relative) != digest:
                    raise ValueError(f"pooling comparison runtime changed: {relative}")
        prepare_pooling_suite(args.prepare_pooling_from, root)
        shutil.copy2(args.prepare_pooling_from / "visualization_windows.json", root / "visualization_windows.json")
        source_snapshot(root)
        summarize(root)
        _write_json(root / "status.json", {"status": "prepared", "pending_training_stages": 2,
                                           "pending_baseline_reevaluation": True, "updated_at": time.time()})
        print(root, flush=True)
    elif args.prepare_simple_from:
        # 只允许复用相同的模型、训练与数据处理实现；汇总/队列代码可更新。
        repo = Path(__file__).resolve().parents[1]
        hashes = read_json(args.prepare_simple_from / "source_hashes.json")
        for relative, digest in hashes.items():
            normalized = relative.replace("\\", "/")
            if normalized.startswith(("src/traceformer/models/", "src/traceformer/training/", "src/traceformer/data/")) or normalized in {
                "src/traceformer/config.py", "src/traceformer/experiments/ablation.py",
            }:
                if _file_sha256(repo / relative) != digest:
                    raise ValueError(f"reuse runtime code changed: {relative}")
        prepare_simple_suite(args.prepare_simple_from, root)
        shutil.copy2(args.prepare_simple_from / "visualization_windows.json", root / "visualization_windows.json")
        source_snapshot(root)
        summarize(root)
        _write_json(root / "status.json", {"status": "prepared", "reused_results": 2, "pending_training_stages": 2,
                                           "updated_at": time.time()})
        print(root, flush=True)
    elif args.prepare:
        config = load_config(args.config)
        for split, expected in (("train", 99), ("val", 24)):
            files = list((Path(config.dataset.root) / split).glob("*.npz"))
            if len(files) != expected:
                raise ValueError(f"expected {expected} {split} files, found {len(files)}")
        if config.training.max_train_files is not None or config.training.max_val_files is not None:
            raise ValueError("formal ablation must use every file")
        prepare_suite(config, root, args.seeds)
        select_windows(config, root)
        source_snapshot(root)
        summarize(root)
        _write_json(root / "status.json", {"status": "prepared", "updated_at": time.time()})
        print(root, flush=True)
    elif args.worker:
        worker(root, args.worker)
    elif args.evaluate:
        if args.attempt is None:
            parser.error("--evaluate requires --attempt")
        evaluate_worker(root, args.evaluate, args.attempt)
    elif args.summarize:
        summarize(root)
    else:
        with queue_lock(root):
            queue(root)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
