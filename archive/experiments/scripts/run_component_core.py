"""等待无额外视图Full验收，再启动唯一的十组联合训练队列。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import psutil
import torch
import yaml
from evaluate_attention_test import verify_raw_score
from run_paper_ablation import completed_job, queue_lock, source_snapshot

from traceformer.config import load_config
from traceformer.experiments.ablation import ComponentCoreSpec, parse_ablation
from traceformer.experiments.component_core import assert_no_view
from traceformer.experiments.component_suite import (
    VARIANTS,
    audit_baseline,
    choose_local_neighbors,
    read_json,
    resource_gate,
    window_diagnostics,
)
from traceformer.postprocessing.submission import evaluate_predictions
from traceformer.training.engine import (
    _atomic_write_text,
    _dataset,
    _file_sha256,
    _serializable,
    _write_json,
    build_model,
)

REPO = Path(__file__).resolve().parents[3]
SEQUENCES = ("train_094.npz", "train_096.npz")


def source_identity(base=REPO):
    # 只纳入本实验会执行的源码，训练中的旧快照完全不改动。
    return {path.relative_to(base).as_posix(): _file_sha256(path)
            for directory in ("src", "scripts") for path in sorted((base / directory).rglob("*.py"))}


def verify_checks(root, source=REPO):
    record = read_json(root / "implementation_checks.json")
    if record.get("passed") is not True or record["source_sha256"] != source_identity(source):
        raise ValueError("implementation checks are missing or source changed; rerun verification")


def process_alive(pid, root):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        process = psutil.Process(pid)
        return process.is_running() and str(root) in " ".join(process.cmdline())
    except (psutil.Error, TypeError):
        return False


def active_processes(root):
    # 覆盖协调器退出但评估孙进程仍在运行的情况。
    result = []
    scripts = ("run_component_core.py", "run_paper_ablation.py", "evaluate_neural_checkpoint.py",
               "check_component_full_equivalence.py")
    for process in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if (process.pid != os.getpid() and "python" in (process.info["name"] or "").lower()
                    and str(root) in command
                    and any(name in command for name in scripts)):
                result.append(process.pid)
        except psutil.Error:
            continue
    return result


def execute(root, phase, command, environment, *, capture=None):
    """持久记录当前子进程；协调器失联时也不会重复启动它。"""
    (root / "logs").mkdir(exist_ok=True)
    log = root / "logs" / f"{phase}.log"
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=environment, cwd=REPO,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        _write_json(root / "status.json", {"status": "running", "phase": phase,
                    "queue_pid": os.getpid(), "worker_pid": process.pid,
                    "command": command, "log": str(log), "updated_at": time.time()})
        code = process.wait()
    if code:
        raise RuntimeError(f"{phase} failed, exit={code}; see {log}")
    if capture is not None and not capture.is_file():
        raise ValueError(f"missing phase result: {capture}")


def evaluate_command(source, config, checkpoint, dataset_root, split, output, spec=None):
    command = [sys.executable, "-u", str(source / "scripts/evaluate_neural_checkpoint.py"),
               "--config", str(config), "--checkpoint", str(checkpoint),
               "--dataset-root", str(dataset_root), "--split", split,
               "--device", "cpu", "--num-threads", "8", "--output", str(output)]
    if spec:
        command += ["--ablation", str(spec)]
    return command


def evaluate_and_audit(root, phase, source, config, checkpoint, data, split, output, env, spec=None):
    command = evaluate_command(source, config, checkpoint, data, split, output, spec)
    expected_spec = read_json(spec) if spec else None
    prior = read_json(output / "report.json") if (output / "report.json").exists() else None
    reusable = (prior is not None and prior["checkpoint_sha256"] == _file_sha256(Path(checkpoint))
                and prior["config_sha256"] == _file_sha256(Path(config))
                and prior["ablation"] == expected_spec and prior["split"] == split)
    if not reusable:
        execute(root, phase, command, env)
    execute(root, phase + "_rescore", [*command, "--rescore-only"], env)
    report = read_json(output / "independent_report.json")
    if report["sequences"] != 24:
        raise ValueError("evaluation requires all 24 files")
    raw = evaluate_predictions(output / "predictions", data / split)
    verify_raw_score(report["competition"], raw["aggregate"])
    for name, metrics in report["per_sequence"].items():
        verify_raw_score(metrics, raw["files"][name])
    _write_json(output / "raw_independent_report.json", raw)
    return report


def prepare(root, baseline, full):
    if not (root / "source").exists():
        source_snapshot(root)
    for name, digest in read_json(root / "source_hashes.json").items():
        if _file_sha256(root / "source" / name) != digest:
            raise ValueError(f"frozen preparation source changed: {name}")
    config = load_config(full["config"])
    data = Path(config.dataset.root)
    source = root / "source"
    env = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONUNBUFFERED="1",
               PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="1")
    # 两套源码读同一权重，逐事件比较完整验证集，不改写Full本身。
    old_source = baseline / "source"
    old_env = dict(env, PYTHONPATH=str(old_source / "src"))
    for label, current_source, current_env in (("original", old_source, old_env),
                                              ("current", source, env)):
        output = root / "equivalence" / label
        command = [sys.executable, "-u", str(source / "scripts/check_component_full_equivalence.py"),
                   "--config", full["config"], "--checkpoint", full["checkpoint"], "--output", str(output)]
        if label == "current":
            command.append("--adapted")
        execute(root, "equivalence_" + label, command, current_env, capture=output / "report.json")
    for path in sorted((root / "equivalence/original").glob("*.pt")):
        old = torch.load(path, weights_only=True)
        new = torch.load(root / "equivalence/current" / path.name, weights_only=True)
        if any(not torch.equal(old[key], new[key]) for key in old):
            raise ValueError(f"Full exact logit/mapping equivalence failed: {path.name}")
    _write_json(root / "full_equivalence.json", {"exact": True, "files": 24,
                                               "baseline": full})
    configs = root / "configs"
    configs.mkdir(exist_ok=True)
    jobs = []
    for variant in VARIANTS[1:]:
        path = configs / f"{variant}.yaml"
        current = replace(config, training=replace(config.training, output_root=root,
                                                  save_training_state_interval=1))
        _atomic_write_text(path, yaml.safe_dump(_serializable(asdict(current)),
                                               allow_unicode=True, sort_keys=False))
        jobs.append({"id": f"seed37/{variant}", "name": variant, "stage": "full", "seed": 37,
                     "spec": ComponentCoreSpec(variant).to_dict(), "result_id": variant,
                     "parent": None, "config": str(path.resolve()),
                     "config_sha256": _file_sha256(path)})
    suite = {"schema": 2, "protocol": "component_core_no_view", "seeds": [37], "jobs": jobs,
             "source_snapshot": str(source), "baseline": full,
             "dataset_manifest_sha256": full["dataset_manifest_sha256"],
             "test_root": str(data.parent.parent / "EV-UAV-dataset"),
             "limitations": "single seed; historical test set already inspected; no test tuning"}
    # 数据路径从前置测试的源目录定位，不默认引入另一份数据。
    test_roots = [REPO.parent / "EV-UAV-dataset", data]
    matches = [p for p in test_roots if len(list((p / "test").glob("*.npz"))) == 24]
    if not matches:
        raise ValueError("cannot locate the complete baseline test partition")
    expected = read_json(full["test_report"])["files"]
    matches = [p for p in matches if all(_file_sha256(p / "test" / name) == row["source_sha256"]
                                       for name, row in expected.items())]
    if not matches:
        raise ValueError("test data does not match baseline source hashes")
    suite["test_root"] = str(matches[0].resolve())
    _write_json(root / "suite.json", suite)
    _write_json(root / "baseline_audit.json", full)
    return suite, env


def gates(root, suite, env):
    """所有资源门槛在前置运行结束后串行执行。"""
    source = Path(suite["source_snapshot"])
    (root / "preflight").mkdir(exist_ok=True)
    config = suite["jobs"][0]["config"]

    def check(variant, neighbors=26, repeats=1):
        results = []
        for sequence in SEQUENCES:
            output = root / "preflight" / f"{variant}_{neighbors}_{Path(sequence).stem}.json"
            if not output.exists():
                command = [sys.executable, "-u", str(source / "scripts/run_component_core.py"),
                           "--root", str(root), "--gate", variant, "--config", str(config),
                           "--sequence", sequence, "--neighbors", str(neighbors),
                           "--repeats", str(repeats), "--output", str(output)]
                try:
                    execute(root, output.stem, command, env, capture=output)
                except RuntimeError:
                    if variant != "C1":
                        raise
                    log = (root / "logs" / f"{output.stem}.log").read_text(encoding="utf-8", errors="replace")
                    if not any(marker in log.lower() for marker in (
                            "out of memory", "cannot allocate", "can't allocate", "not enough memory",
                            "memoryerror", "defaultcpuallocator")):
                        raise
                    # 26邻接内存失败也按同一预定规则回退，日志完整保留。
                    _write_json(output, {"sequence": sequence, "memory_passed": False,
                                         "finite_gradients": False, "median_seconds": 0.0,
                                         "error": "preflight subprocess failed; inspect log"})
            result = read_json(output)
            if not result["memory_passed"] or not result["finite_gradients"]:
                if variant != "C1":
                    raise ValueError(f"resource gate failed: {variant} {sequence}")
            results.append(result)
        return results

    full = check("Full", repeats=3)
    local = check("C1", repeats=3)
    selected = 26
    if not choose_local_neighbors(full, local):
        selected = 6
        local = check("C1", neighbors=6, repeats=3)
        if not choose_local_neighbors(full, local):
            # 只停止该组，其余已获授权的结构仍继续执行。
            suite["skipped_variants"] = {"C1": "both 26 and 6 neighbors fail the frozen resource budget"}
    for job in suite["jobs"]:
        if job["name"] == "C1":
            job["spec"] = ComponentCoreSpec("C1", local_neighbors=selected).to_dict()
        else:
            check(job["name"])
    _write_json(root / "suite.json", suite)
    _write_json(root / "resource_gate.json", {"passed": not suite.get("skipped_variants"),
                "runnable_passed": True, "skipped_variants": suite.get("skipped_variants", {}),
                "neighbors": selected, "full": full, "local": local})


def summarize(root, suite):
    rows = [read_json(path) for path in (root / "results").glob("*.json")]
    by_id = {row["variant"]: row for row in rows}
    lines = ["# 无稀疏视图神经网络消融", "", "seed37；验证选模型与阈值；全量测试。", ""]
    columns = ("score", "iou", "seg_acc", "pd", "fa", "parameters", "seconds", "peak_rss_bytes")
    for label, variants in (("Component Ablation", VARIANTS[:6]),
                            ("Core Architecture Ablation", ("Full", *VARIANTS[6:]))):
        lines += [f"## {label}", "", "|模型|" + "|".join(columns) + "|ΔScore|",
                  "|---|" + "---|" * (len(columns) + 1)]
        for variant in variants:
            if variant not in by_id:
                if variant in suite.get("skipped_variants", {}):
                    lines.append("|" + variant + "|资源门槛失败|" + "—|" * len(columns))
                continue
            row = by_id[variant]
            delta = row["score"] - by_id["Full"]["score"]
            lines.append("|" + variant + "|" + "|".join(str(row[c]) for c in columns)
                         + f"|{delta:+.9f}|")
        lines.append("")
    lines += ["M3保留Head上下文；C4整网非因果；C3/M5同时改变参数量；单种子不作显著性结论。"]
    _atomic_write_text(root / "results.md", "\n".join(lines) + "\n")
    _write_json(root / "results.json", {"rows": rows, "baseline": suite["baseline"],
                                        "complete": set(by_id) == set(VARIANTS)})


def finish_variant(root, suite, variant, config, checkpoint, env, spec=None):
    """测试、复评分、独立原始掩码计数与性能测量均各自保存。"""
    source = Path(suite["source_snapshot"])
    output = root / "test" / variant
    report = evaluate_and_audit(root, "test_" + variant, source, config, checkpoint,
                               Path(suite["test_root"]), "test", output, env, spec)
    if variant == "Full" and report["competition"] != read_json(suite["baseline"]["test_report"])["competition"]:
        raise ValueError("reused Full test metrics differ from the baseline")
    benchmark = root / "benchmark" / f"{variant}.json"
    command = [sys.executable, "-u", str(source / "scripts/run_component_core.py"),
               "--root", str(root), "--benchmark", "--config", str(config),
               "--checkpoint", str(checkpoint), "--output", str(benchmark)]
    if spec:
        command += ["--spec", str(spec)]
    prior = read_json(benchmark) if benchmark.exists() else {}
    if prior.get("checkpoint_sha256") != _file_sha256(Path(checkpoint)):
        execute(root, "benchmark_" + variant, command, env, capture=benchmark)
    timing = read_json(benchmark)
    row = {"variant": variant, **report["competition"], "parameters": report["parameters"],
           "average_precision": report["average_precision"], "threshold": report["threshold"],
           "seconds": timing["mean_seconds"], "peak_rss_bytes": timing["peak_rss_bytes"],
           "checkpoint_sha256": report["checkpoint_sha256"], "report": str(output / "independent_report.json")}
    _write_json(root / "results" / f"{variant}.json", row)
    summarize(root, suite)


def run(root, baseline):
    acquired = False
    try:
        with queue_lock(root):
            acquired = True
            full = audit_baseline(baseline)
            if full is None:
                raise RuntimeError("baseline is not complete; no training may start")
            verify_checks(root, root / "source" if (root / "source").exists() else REPO)
            if not (root / "suite.json").exists():
                suite, env = prepare(root, baseline, full)
            else:
                suite = read_json(root / "suite.json")
                if suite["baseline"] != full:
                    raise ValueError("baseline identity changed")
                env = dict(os.environ, PYTHONPATH=str(root / "source/src"), PYTHONUNBUFFERED="1",
                           PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                           OPENBLAS_NUM_THREADS="1")
            for name, digest in read_json(root / "source_hashes.json").items():
                if _file_sha256(root / "source" / name) != digest:
                    raise ValueError(f"frozen source changed: {name}")
            if not (root / "resource_gate.json").exists():
                gates(root, suite, env)
            for directory in ("results", "benchmark"):
                (root / directory).mkdir(exist_ok=True)
            finish_variant(root, suite, "Full", full["config"], full["checkpoint"], env)
            for job in suite["jobs"]:
                if job["name"] in suite.get("skipped_variants", {}):
                    continue
                if _file_sha256(Path(job["config"])) != job["config_sha256"]:
                    raise ValueError("job configuration changed")
                if completed_job(root, job["id"]) is None:
                    execute(root, "train_" + job["name"], [sys.executable, "-u",
                            str(root / "source/scripts/run_paper_ablation.py"),
                            "--suite", str(root), "--worker", job["id"]], env)
                done = completed_job(root, job["id"])
                path = Path(done["checkpoint"])
                finish_variant(root, suite, job["name"], job["config"], path, env,
                               path.parent / "ablation.json")
            hashes = {p.relative_to(root).as_posix(): _file_sha256(p)
                      for directory in ("test", "results", "benchmark", "seed37")
                      for p in sorted((root / directory).rglob("*")) if p.is_file()}
            skipped = suite.get("skipped_variants", {})
            _write_json(root / "completion_audit.json", {"complete": not skipped, "sha256": hashes,
                       "variants": list(VARIANTS), "source_sha256": read_json(root / "source_hashes.json")})
            _write_json(root / "status.json", {"status": "needs_attention" if skipped else "complete",
                       "variants": 11 - len(skipped), "skipped_variants": skipped,
                       "new_training_runs": 10 - len(skipped), "updated_at": time.time()})
    except Exception:
        if acquired:
            _write_json(root / "status.json", {"status": "failed", "error": traceback.format_exc(),
                                               "updated_at": time.time()})
        raise


def benchmark(args):
    config = load_config(args.config)
    torch.set_num_threads(8)
    spec = parse_ablation(read_json(args.spec)) if args.spec else None
    model = build_model(config, ablation=spec).eval()
    saved = torch.load(args.checkpoint, weights_only=True)
    from traceformer.training.engine import _validate_checkpoint_contract
    _validate_checkpoint_contract(saved, config, ablation=spec)
    model.load_state_dict(saved["model_state"], strict=True)
    assert_no_view(model)
    elapsed, diagnostics = {}, {}
    with torch.inference_mode():
        for sequence in _dataset(config, "val"):
            model(sequence.events)
            seconds = []
            for _ in range(3):
                start = time.perf_counter()
                output = model(sequence.events)
                seconds.append(time.perf_counter() - start)
            elapsed[sequence.source_path.name] = seconds
            del output
    memory = psutil.Process().memory_info()
    if spec is not None and spec.variant == "C2":
        # 几何诊断不污染推理峰值；只重做Tokenizer来获取坐标。
        with torch.inference_mode():
            for sequence in _dataset(config, "val"):
                tokenized = model.tokenizer(sequence.events)
                diagnostics[sequence.source_path.name] = window_diagnostics(
                    tokenized.token_coords, tokenized.token_batch_indices, config.model.cone_windows)
    _write_json(args.output, {"mean_seconds": statistics.mean(statistics.median(v) for v in elapsed.values()),
                "checkpoint_sha256": _file_sha256(args.checkpoint),
                "per_file_seconds": elapsed, "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
                "memory_note": "fresh-process peak RSS including sequence input; diagnostics after timed forward",
                "window_diagnostics": diagnostics, "repeats": 3, "warmup_per_file": 1})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=REPO / "outputs/no_sparse_local_20261025")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gate", choices=VARIANTS)
    parser.add_argument("--neighbors", type=int, choices=(6, 26), default=26)
    parser.add_argument("--sequence", choices=SEQUENCES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    root, baseline = args.root.resolve(), args.baseline.resolve()
    if args.gate:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = resource_gate(load_config(args.config), ComponentCoreSpec(args.gate,
                              local_neighbors=args.neighbors), args.sequence, repeats=args.repeats)
        _write_json(args.output, result)
        return
    if args.benchmark:
        benchmark(args)
        return
    root.mkdir(parents=True, exist_ok=True)
    if args.run:
        run(root, baseline)
        return
    # 调度入口很快返回；真正队列在隐藏子进程中持有同一文件锁。
    dispatch = root / "dispatch"
    dispatch.mkdir(exist_ok=True)
    with queue_lock(dispatch):
        status = read_json(root / "status.json") if (root / "status.json").exists() else {}
        if status.get("status") in {"complete", "needs_attention"}:
            print(json.dumps(status, ensure_ascii=False))
            return
        active = active_processes(root)
        if active or any(process_alive(status.get(key), root) for key in ("queue_pid", "worker_pid")):
            print(json.dumps({**status, "status": "already_running", "active_pids": active}, ensure_ascii=False))
            return
        if status.get("status") == "failed":
            raise RuntimeError("prior phase failed; inspect status/logs before explicit recovery")
        if audit_baseline(baseline) is None:
            status = {"status": "waiting_for_baseline", "baseline": str(baseline),
                      "updated_at": time.time()}
            _write_json(root / "status.json", status)
            print(json.dumps(status, ensure_ascii=False))
            return
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
