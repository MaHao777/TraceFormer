"""按原生Windows方案顺序预测完整val/test，保留失败现场并打包回传。"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from event_io import THRESHOLD, audit_input, selected_files, sha256, write_json

BUNDLE = Path(__file__).resolve().parents[1]
MODELS = ("ev_spsegnet", "pact", "pointevent")


def inventory(data_root, expected_count):
    splits = {}
    for split in ("val", "test"):
        files = selected_files(data_root, split, None)
        if len(files) != expected_count:
            raise ValueError(f"{split}: expected {expected_count} files, found {len(files)}")
        records = []
        for path in files:
            record, raw, _ = audit_input(path)
            # Pd必须使用原始目标ID，预检额外核对它与网络输入中的ID相同。
            with np.load(path, allow_pickle=False) as data:
                ids = raw["name"] if raw.dtype.names else raw[:, 5]
                if not np.array_equal(ids, data["evs_norm"][:, 5]):
                    raise ValueError(f"Object ID order differs: {path}")
            records.append(record)
        splits[split] = records
    return dict(expected_files_per_split=expected_count, splits=splits)


def command(arguments, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    with log.open("w", encoding="utf-8") as stream:
        stream.write(subprocess.list2cmdline(arguments) + "\n")
        stream.flush()
        with subprocess.Popen(arguments, cwd=BUNDLE, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            return process.wait()


def unused_path(path):
    suffix = 1
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{suffix:03d}{path.suffix}")
        suffix += 1
    return candidate


def reuse_or_preserve(folder, records, split, checkpoint_hash):
    report_path = folder / "report.json"
    if not folder.exists():
        return False
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] == "complete":
            expected = {r["file"]: (r["sha256"], r["events"]) for r in records}
            actual = {r["file"]: (r["sha256"], r["events"]) for r in report["sequences"]}
            if (expected != actual or report["split"] != split
                    or report["threshold"] != THRESHOLD or report["backend"] != "torch-reference"
                    or report["checkpoint_sha256"] != checkpoint_hash):
                raise ValueError(f"Completed run has different identities/protocol: {folder}")
            return True
    # 失败任务移到当前split内部的新目录，保留现场后重跑该模型。
    root = folder.parent.resolve()
    folder = folder.resolve()
    folder.relative_to(root)
    backup_root = root / "_failed_attempts"
    backup_root.mkdir(exist_ok=True)
    attempt = 1
    while (backup_root / f"{folder.name}_{attempt:03d}").exists():
        attempt += 1
    destination = backup_root / f"{folder.name}_{attempt:03d}"
    destination.resolve().relative_to(root)
    folder.rename(destination)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=BUNDLE / "results_full_20261102")
    parser.add_argument("--expected-count", type=int, default=24)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    root, data_root = args.output_root.resolve(), args.data_root.resolve()
    snapshot = inventory(data_root, args.expected_count)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "dataset_inventory.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != snapshot:
        raise ValueError("Output folder belongs to different dataset files; use a new output root")
    write_json(path, snapshot)
    shutil.copyfile(BUNDLE / "qualitative_selection.json", root / "qualitative_selection.json")
    if args.audit_only:
        print("Audited full val/test inputs; no neural inference was run.")
        return
    logs = root / "logs"
    preflight = command([sys.executable, str(BUNDLE / "tools/check_environment.py"),
                         "--output", str(root / "environment_windows.json")],
                        unused_path(logs / "environment.log"))
    if preflight:
        raise RuntimeError("GPU environment check failed; inspect logs/environment.log")
    # 独立进程释放模型显存；不并行占用同一张4090。
    from predict import CHECKPOINTS

    status_path = root / "run_status.json"
    status = dict(status="running", jobs=[], threshold=THRESHOLD, backend="torch-reference",
                  execution_helpers_sha256={p.name: sha256(p) for p in (BUNDLE / "tools").glob("*.py")})
    write_json(status_path, status)
    for split in ("val", "test"):
        for model in MODELS:
            job = dict(split=split, model=model)
            try:
                reused = reuse_or_preserve(root / split / model, snapshot["splits"][split],
                                           split, sha256(BUNDLE / CHECKPOINTS[model]))
                if reused:
                    job["status"] = "reused_complete"
                    print(f"Reuse complete run: {split}/{model}", flush=True)
                else:
                    code = command([sys.executable, str(BUNDLE / "tools/predict.py"),
                                    "--model", model, "--backend", "torch-reference",
                                    "--data-root", str(data_root), "--split", split,
                                    "--output-root", str(root / split)],
                                   unused_path(logs / f"{split}_{model}.log"))
                    job.update(status="complete" if code == 0 else "failed", exit_code=code)
            except Exception as error:
                job.update(status="failed", error=str(error))
                print(f"Failed: {split}/{model}: {error}", flush=True)
            status["jobs"].append(job)
            write_json(status_path, status)
    status["status"] = "partial" if any(j["status"] == "failed" for j in status["jobs"]) else "complete"
    write_json(status_path, status)
    from analyze_results import collect_full

    archive = unused_path(root.parent / f"{root.name}_return.zip")
    summary = collect_full(root, archive)
    if status["status"] != "complete" or summary["status"] != "complete":
        raise SystemExit("Partial run returned; rerun the same command to retry failed model/split jobs.")


if __name__ == "__main__":
    main()
