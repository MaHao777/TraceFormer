"""保留原实验快照，以融合CPU执行恢复A0，并与先行A1/A2协调收尾。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil
import torch
from run_center_attention import read, run_logged
from run_paper_ablation import source_snapshot

from traceformer.experiments.figures import render_figures
from traceformer.experiments.suite import summarize
from traceformer.training.engine import _file_sha256, _write_json


def execute(root):
    acceleration = root / "acceleration"
    acceleration.mkdir(exist_ok=True)
    equivalence = read(root / "diagnostics/fused_A0_validation_equivalence.json")
    if not equivalence["all_masks_equal"] or not equivalence["all_metrics_equal"]:
        raise ValueError("A0 validation equivalence must pass before replacing its runtime")
    suite = read(root / "suite.json")
    status = read(root / "status.json")
    fast_track = read(root / "fast_track.json")
    if status.get("job") != "seed37/local_B":
        raise ValueError("requires the original A0 worker to be waiting")
    old_worker = psutil.Process(status["worker_pid"])
    if "seed37/local_B" not in old_worker.cmdline() or str(root) not in old_worker.cmdline():
        raise ValueError("unexpected original A0 process")
    # 停止旧进程前确认模型、优化器、采样及RNG状态都已保存。
    state = root / "seed37/local_B/attempt_001/training_state.pt"
    if not state.exists():
        raise ValueError("missing exact-state A0 checkpoint")
    resume_epoch = int(torch.load(state, weights_only=False, map_location="cpu")["epoch"])
    for name in ("suite.json", "source_hashes.json", "status.json", "execution.json"):
        shutil.copy2(root / name, acceleration / f"original_{name}")
    shutil.copy2(state, acceleration / f"a0_resume_epoch_{resume_epoch:03d}.pt")
    runtime = acceleration / "runtime_v2"
    runtime.mkdir()
    source_snapshot(runtime)
    original_hashes = read(acceleration / "original_source_hashes.json")
    new_hashes = read(runtime / "source_hashes.json")
    for relative, digest in original_hashes.items():
        normalized = relative.replace("\\", "/")
        if normalized.startswith(("src/traceformer/models/", "src/traceformer/training/",
                                  "src/traceformer/data/")) or normalized == "src/traceformer/config.py":
            if new_hashes.get(relative) != digest:
                raise ValueError(f"core model/training/data changed: {relative}")
    source = runtime / "source"
    suite.update(source_snapshot=str(source), runtime_overrides={"local_cpu_backend": "fused"},
                 runtime_revision={"original_source": suite["source_snapshot"], "reason": "CPU execution acceleration",
                                   "resume_epoch": resume_epoch, "data_epochs_losses_unchanged": True})
    # 旧快照不改写，新快照和原manifest均可独立审计。
    _write_json(root / "suite.json", suite)
    _write_json(root / "source_hashes.json", new_hashes)
    environment = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONUNBUFFERED="1",
                       PYTHONIOENCODING="utf-8", TRACEFORMER_LOCAL_CPU_BACKEND="fused",
                       OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
                       NUMBA_NUM_THREADS="8")
    record = {"status": "running", "phase": "accelerated_parallel_training", "pid": os.getpid(),
              "fast_track_pid": fast_track["pid"], "resume_checkpoint_sha256": _file_sha256(state)}
    # 先停只负责等待的旧协调器，避免旧队列把主动切换误报为训练失败。
    for pid in (read(acceleration / "original_execution.json")["pid"], status["queue_pid"], old_worker.pid):
        process = psutil.Process(pid)
        if str(root) not in process.cmdline():
            raise ValueError("refusing to stop a process from another experiment")
        process.terminate()
        process.wait(timeout=15)
    _write_json(root / "execution.json", record)
    log = root / "logs/accelerated_A0.log"
    command = [sys.executable, "-u", str(source / "scripts/run_paper_ablation.py"),
               "--suite", str(root), "--worker", "seed37/local_B"]
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        record["worker_pid"] = process.pid
        _write_json(root / "execution.json", record)
        _write_json(root / "status.json", {"status": "running", "job": "seed37/local_B",
                    "worker_pid": process.pid, "queue_pid": os.getpid(), "log": str(log),
                    "parallel_fast_track_pid": fast_track["pid"]})
        if process.wait():
            raise RuntimeError("accelerated A0 failed")
    while any(not (root / job["id"] / "done.json").exists() for job in suite["jobs"]):
        if read(root / "fast_track.json")["status"] == "complete":
            raise RuntimeError("fast track finished without all expected results")
        if not psutil.pid_exists(fast_track["pid"]):
            raise RuntimeError("fast track exited before its expected results completed")
        time.sleep(15)
    summarize(root)
    render_figures(root, 37)
    _write_json(root / "status.json", {"status": "complete", "completed_stages": 4})
    record["phase"] = "test_independent_scoring_and_benchmark"
    _write_json(root / "execution.json", record)
    run_logged([sys.executable, "-u", str(source / "scripts/finish_center_attention.py"),
                "--suite", str(root)], root / "logs/finalization.log", environment)
    _write_json(root / "execution.json", {"status": "complete", "updated_at": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args.suite.resolve())
    except BaseException:
        _write_json(args.suite / "acceleration_error.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise
