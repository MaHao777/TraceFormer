"""串行执行同一 CUDA 主机的冻结模型比较；记录失败并保留所有现场。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from traceformer.training.engine import _file_sha256, _write_json


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    # 上传包内所有文件逐个验哈希，避免错用其他数据或权重。
    manifest = json.loads((root / "transfer_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        if _file_sha256(root / relative) != expected["sha256"]:
            raise ValueError(f"transfer hash mismatch: {relative}")
    environment = dict(os.environ, PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    python = sys.executable
    output = root / "results"
    output.mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    status = {"status": "running", "pid": os.getpid(), "started_at": time.time(), "stages": {}}
    _write_json(root / "status.json", status)
    commands = []
    for name, schema in (("TraceFormer-main", 5), ("TraceFormer-reference", 4)):
        artifact = "main" if schema == 5 else "reference"
        commands.append((name, [python, "scripts/evaluate_frozen_neural.py", "--source", f"legacy_schema{schema}",
            "--config", f"artifacts/{artifact}/config.yaml", "--checkpoint", f"artifacts/{artifact}/best.pt",
            "--dataset-root", str(root / "data/ev-uav"), "--scoring-config", "configs/comparison.yaml",
            "--output", str(output / name)]))
    commands.append(("TraceFormer-complete", [python, "scripts/evaluate_frozen_system.py",
        "--cache", str(output / "TraceFormer-main/cache"), "--reference", str(output / "TraceFormer-reference/predictions"),
        "--release", "artifacts/event_release", "--dataset-root", str(root / "data/ev-uav"),
        "--output", str(output / "TraceFormer-complete")]))
    for name in ("PACT", "EV-SpSegNet"):
        commands.append((name, [python, "scripts/evaluate_external_baseline.py", "--model", name,
            "--manifest", "upstream_manifest.json", "--config", "configs/comparison.yaml",
            "--output", str(output / name)]))
    for name, command in commands:
        status.update(current=name, updated_at=time.time())
        _write_json(root / "status.json", status)
        log = root / "logs" / f"{name}.log"
        if (output / name / "report.json").exists():
            report = json.loads((output / name / "report.json").read_text(encoding="utf-8"))
            if report["status"] == "complete" and (output / name / "independent_report.json").exists():
                status["stages"][name] = {"status": "verified_existing"}
                continue
        with log.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            status.update(worker_pid=process.pid)
            _write_json(root / "status.json", status)
            exit_code = process.wait()
            report_path = output / name / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
            result = {"exit_code": exit_code, "status": report.get("status", "failed"), "log": str(log)}
            if exit_code == 0 and report.get("status") == "complete":
                if name == "TraceFormer-complete":
                    # 二值结果重新读取并评分，不能只复用推理进程里的累加器。
                    rescoring = [python, "-m", "traceformer.cli.postprocess", "evaluate",
                        "--prediction-dir", str(output / name / "complete"), "--dataset-root", str(root / "data/ev-uav/val"),
                        "--output", str(output / name / "independent_report.json")]
                else:
                    rescoring = [python, "scripts/rescore_external_predictions.py", "--predictions", str(output / name),
                                 "--config", "configs/comparison.yaml"]
                result["independent_exit_code"] = subprocess.call(rescoring, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                if result["independent_exit_code"]:
                    result["status"] = "independent_scoring_failed"
                elif name == "TraceFormer-complete":
                    independent = json.loads((output / name / "independent_report.json").read_text(encoding="utf-8"))
                    if independent != report["stages"]["complete"]:
                        result["status"] = "independent_scoring_failed"
            status["stages"][name] = result
            _write_json(root / "status.json", status)
    status.update(status="finished" if all(s["status"] in {"complete", "verified_existing"} for s in status["stages"].values())
                  else "finished_with_pending_or_failures", updated_at=time.time(), worker_pid=None)
    _write_json(root / "status.json", status)


if __name__ == "__main__":
    main()
