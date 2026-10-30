"""有限批次结束后核验产物，修正独立计分入口路径并保留原始错误记录。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from traceformer.training.engine import _file_sha256, _write_json


def main():
    root = Path(__file__).resolve().parents[1]
    deadline = time.monotonic() + 5400
    while time.monotonic() < deadline:
        status = json.loads((root / "status.json").read_text(encoding="utf-8"))
        process = Path(f"/proc/{status['pid']}/cmdline")
        if status["status"] != "running" or not process.exists() or b"run_paper_cuda.py" not in process.read_bytes():
            break
        time.sleep(15)
    else:
        raise TimeoutError("remote batch did not finish within the finite audit window")
    manifest = json.loads((root / "transfer_manifest.json").read_text(encoding="utf-8"))
    for relative, record in manifest["files"].items():
        if _file_sha256(root / relative) != record["sha256"]:
            raise ValueError(f"final input/source hash differs: {relative}")
    verified = {}
    for name, state in status["stages"].items():
        folder = root / "results" / name
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        if report.get("status") != "complete":
            verified[name] = {"status": report.get("status", "failed"), "original_stage": state}
            continue
        if name == "TraceFormer-complete":
            # 只重做独立计分，不重新生成神经特征或预测掩码。
            subprocess.run([sys.executable, "-m", "traceformer.cli.postprocess", "evaluate",
                "--prediction-dir", str(folder / "complete"), "--dataset-root", str(root / "data/ev-uav/val"),
                "--output", str(folder / "independent_report.json")], cwd=root, check=True)
            independent = json.loads((folder / "independent_report.json").read_text(encoding="utf-8"))
            if independent != report["stages"]["complete"]:
                raise ValueError("complete-system independent scoring differs")
        else:
            independent = json.loads((folder / "independent_report.json").read_text(encoding="utf-8"))
            if independent.get("status") != "complete":
                raise ValueError(f"missing independent scoring for {name}")
        verified[name] = {"status": "verified", "report_sha256": _file_sha256(report_path),
                         "independent_report_sha256": _file_sha256(folder / "independent_report.json"),
                         "original_stage": state}
    _write_json(root / "postflight_audit.json", {"status": "audited", "stages": verified,
        "original_queue_status": status["status"], "source_and_input_hashes_verified": True,
        "complete_system_rescore_note": "Canonical NPZ folder is data/ev-uav/val; original queue/logs preserved."})
    print("CUDA batch postflight audit complete", flush=True)


if __name__ == "__main__":
    main()
