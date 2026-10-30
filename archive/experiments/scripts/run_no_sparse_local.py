"""串行训练无 Sparse 网络，独立复评并用验证阈值评估测试集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "source"
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1")
    status = {"pid": os.getpid(), "started_at": datetime.now(
        timezone(timedelta(hours=8))).isoformat(), "postprocessing": False,
        "paper_modified_by_this_run": False, "root": str(root)}

    def execute(phase: str, command: list[str]) -> None:
        # 每个阶段使用新进程，复评不复用训练进程中的模型实例。
        status.update(status="running", phase=phase, command=command)
        write_json(root / "status.json", status)
        with (root / f"{phase}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", *command], cwd=repository,
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        execute("training", ["-m", "traceformer.cli.train", "--config", str(root / "config.yaml"),
                             "--device", "cpu", "--output-dir", str(root / "training"),
                             "--note", "删除 Sparse 分支；其余网络从头联合训练；不修改论文"])
        training = root / "training"
        execute("validation", ["-m", "traceformer.cli.evaluate", "--config", str(training / "config.yaml"),
                               "--checkpoint", str(training / "best.pt"), "--device", "cpu",
                               "--output", str(training / "reproduced_metrics.json")])
        saved = json.loads((training / "metrics.json").read_text(encoding="utf-8"))
        reproduced = json.loads((training / "reproduced_metrics.json").read_text(encoding="utf-8"))
        if saved != reproduced:
            raise ValueError("saved validation metrics were not exactly reproduced")
        evaluator = source / "scripts/evaluate_neural_checkpoint.py"
        command = [str(evaluator), "--config", str(training / "config.yaml"), "--checkpoint",
                   str(training / "best.pt"), "--dataset-root", str(args.test_root.resolve()),
                   "--split", "test", "--output", str(root / "test"), "--device", "cpu",
                   "--num-threads", "8"]
        execute("test", command)
        execute("test_rescore", [*command, "--rescore-only"])
        report = json.loads((root / "test/report.json").read_text(encoding="utf-8"))
        baseline_path = root / "before/baseline_test_metrics.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["aggregate"]
        comparison = {"baseline": baseline, "without_sparse": report["competition"],
                      "delta": {name: report["competition"][name] - baseline[name]
                                for name in ("score", "iou", "seg_acc", "pd", "fa")},
                      "parameters": report["parameters"], "threshold": report["threshold"],
                      "protocol": "single seed; neural-only; thresholds selected on validation"}
        write_json(root / "test_comparison.json", comparison)
        paper = Path("D:/Paper/TraceFormer")
        expected = json.loads((root / "paper_before_sha256.json").read_text(encoding="utf-8"))
        changed = [name for name, digest in expected.items()
                   if not (paper / name).is_file()
                   or hashlib.sha256((paper / name).read_bytes()).hexdigest() != digest]
        write_json(root / "paper_verification.json", {"unchanged": not changed, "changed": changed})
        status.update(status="complete", phase="complete", comparison=comparison,
                      paper_unchanged=not changed)
        write_json(root / "status.json", status)
    except Exception:
        status.update(status="failed", error=traceback.format_exc())
        write_json(root / "status.json", status)
        raise


if __name__ == "__main__":
    main()
