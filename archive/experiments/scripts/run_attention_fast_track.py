"""保留A0进程内状态，先完成同协议A1/A2，释放慢基线的排队阻塞。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from traceformer.training.engine import _file_sha256, _write_json


def run(root: Path):
    suite = json.loads((root / "suite.json").read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if status.get("job") != "seed37/local_B":
        raise ValueError("fast track requires A0 to be the current queued worker")
    worker = psutil.Process(status["worker_pid"])
    if "seed37/local_B" not in worker.cmdline() or str(root) not in worker.cmdline():
        raise ValueError("unexpected A0 process identity")
    source = Path(suite["source_snapshot"])
    environment = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONIOENCODING="utf-8",
                       PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                       OPENBLAS_NUM_THREADS="1")
    record = {"pid": os.getpid(), "a0_pid": worker.pid, "completed": [], "status": "running"}
    worker.suspend()
    try:
        for variant in ("A1", "A2"):
            job = next(j for j in suite["jobs"] if j["result_id"] == variant)
            if (root / job["id"] / "done.json").exists():
                record["completed"].append(variant)
                continue
            if _file_sha256(Path(job["config"])) != job["config_sha256"]:
                raise ValueError("fast-track training configuration changed")
            log = root / "logs" / f"{job['id'].replace('/', '_')}.log"
            command = [sys.executable, "-u", str(source / "scripts/run_paper_ablation.py"),
                       "--suite", str(root), "--worker", job["id"]]
            with log.open("a", encoding="utf-8") as stream:
                process = subprocess.Popen(command, env=environment, stdout=stream,
                                           stderr=subprocess.STDOUT)
                record.update(variant=variant, worker_pid=process.pid, log=str(log),
                              updated_at=time.time())
                _write_json(root / "fast_track.json", record)
                if process.wait():
                    raise RuntimeError(f"fast-track {variant} failed")
            record["completed"].append(variant)
        record.update(status="complete", updated_at=time.time())
    finally:
        # 即使快速分支失败也恢复A0；优化部署若已替换旧进程，则不再操作旧PID。
        if worker.is_running():
            worker.resume()
        _write_json(root / "fast_track.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    run(parser.parse_args().suite.resolve())
