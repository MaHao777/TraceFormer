"""实验共用执行功能：冻结身份、进程、锁、恢复与独立评估。"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
import traceback
import zipfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import psutil
import torch

from traceformer.config import load_config
from traceformer.experiments.ablation import parse_ablation
from traceformer.experiments.component_suite import read_json
from traceformer.experiments.evaluation import verify_raw_score
from traceformer.postprocessing.submission import evaluate_predictions
from traceformer.training.engine import (
    _file_sha256,
    _validate_checkpoint_contract,
    _write_json,
    run_training,
)
from traceformer.training.runtime import prepare_device


class Tee:
    """同一输出写入终端与实验日志。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def exclusive_run(root):
    """非阻塞系统文件锁随进程退出释放。"""
    with (root / "run.lock").open("a+b") as stream:
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


def setup_device(config):
    """保留精度与 CUDA 规则，不自动切换设备。"""
    runtime = prepare_device(config.training.device, num_threads=config.training.num_threads)
    if config.training.mixed_precision:
        if runtime.device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 was requested but this CUDA environment does not support BF16")
    if runtime.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return runtime


def freeze_execution_environment(config, root, label):
    """同一实验恢复时固定主机、GPU、驱动与运行时。"""
    runtime = setup_device(config)
    environment = {"runtime": runtime.metadata, "host": platform.node(), "python": sys.version,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if runtime.device.type == "cuda":
        properties = torch.cuda.get_device_properties(runtime.device)
        environment["device_uuid"] = str(getattr(properties, "uuid", "unavailable"))
        try:
            driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, check=True, timeout=15)
            environment["driver_versions"] = driver.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError) as error:
            environment["driver_query_error"] = type(error).__name__
    path = root / "execution_environment.json"
    if path.exists() and read_json(path) != environment:
        raise ValueError(f"{label} execution environment changed; preserve the original run")
    _write_json(path, environment)


def validate_benchmark_coverage(report, checkpoint, identities, inference_split="val"):
    """新科学实验要求真实数值/梯度检查和完整输入覆盖，不设置速度预算。"""
    smoke = report.get("smoke")
    if type(smoke) is not bool or report.get("precision") != "fp32":
        raise ValueError("benchmark precision/protocol differs")
    numerical = report.get("numerical_check", {})
    if not numerical.get("passed") or not numerical.get("parameter_gradients_checked"):
        raise ValueError("benchmark numerical/gradient check failed")
    if not smoke and (report.get("device") != "cuda" or not numerical.get("real_cuda_checked")):
        raise ValueError("formal preflight requires real CUDA numerical checks")
    parts = [("inference", inference_split)]
    if checkpoint is None:
        parts.append(("training", "train"))
    for part, split in parts:
        expected = sorted(identities[split])[:2] if smoke else (
            ["train_094.npz", "train_096.npz"] if split == "train" else sorted(identities[split]))
        rows = report.get(part, {}).get("sequences", [])
        if [row["file"] for row in rows] != expected:
            raise ValueError("benchmark has incomplete sequence coverage")
        if part == "training" and not all(row.get("finite_gradients") for row in rows):
            raise ValueError("full-sequence gradient check failed")


class ExperimentRuntime:
    """一份冻结包对应一个执行环境；具体阶段由 workflow 定义。"""

    experiment = ""

    def __init__(self, bundle, output=None):
        self.bundle = Path(bundle).resolve()
        self.output = Path(output).resolve() if output is not None else self.bundle / "output"

    def verify_package(self):
        manifest = read_json(self.bundle / "package_manifest.json")
        for name, digest in manifest["sha256"].items():
            path = self.bundle / name
            if not path.is_file() or _file_sha256(path) != digest:
                raise ValueError(f"package file changed or missing: {name}")
        expected = {name for name in manifest["sha256"] if name.endswith(".py")}
        actual = {p.relative_to(self.bundle).as_posix() for p in self.bundle.rglob("*.py")
                  if p.relative_to(self.bundle).parts[0] != "output"
                  and "__pycache__" not in p.parts}
        if actual != expected:
            raise ValueError("unexpected/missing Python source in upload directory")
        return _file_sha256(self.bundle / "package_manifest.json")

    def verify_data(self, data_root, test_root, *, need_test):
        identities = read_json(self.bundle / "dataset_manifest.json")
        checked = {}
        for split in ("train", "val", "test") if need_test else ("train", "val"):
            folder = (test_root if split == "test" else data_root) / split
            paths = {p.name: p for p in folder.glob("*.npz")}
            if set(paths) != set(identities[split]):
                raise ValueError(f"{split}: NPZ file set differs from packaged experiment data")
            for name, path in paths.items():
                if _file_sha256(path) != identities[split][name]["sha256"]:
                    raise ValueError(f"data identity differs: {split}/{name}")
            checked[split] = {"files": len(paths), "root": str(folder)}
        return checked

    def guard_active_workers(self, root):
        # 协调器退出后 worker 可能仍活着；核对完整命令，避免误认复用 PID。
        for status in root.rglob("status.json"):
            state = read_json(status)
            if state.get("status") != "running":
                continue
            pid, command = state.get("worker_pid"), state.get("command")
            if not isinstance(pid, int) or pid <= 0 or not isinstance(command, list):
                continue
            if str(self.bundle / "run.py") not in command and "traceformer.experiments.evaluation" not in command:
                continue
            try:
                process = psutil.Process(pid)
                created = state.get("worker_created_at")
                if created is not None and process.create_time() != created:
                    continue
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE and process.cmdline() == command:
                    raise RuntimeError(f"active worker PID {pid} still belongs to this experiment")
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied as error:
                raise RuntimeError(f"cannot verify recorded worker PID {pid}; preserve output") from error

    def execute(self, root, phase, command):
        print(f"phase: {phase}", flush=True)
        environment = dict(os.environ, PYTHONPATH=str(self.bundle / "src"), PYTHONUNBUFFERED="1",
                           PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        with subprocess.Popen(command, cwd=self.bundle, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as child:
            try:
                created = psutil.Process(child.pid).create_time()
            except psutil.NoSuchProcess:
                created = None
            _write_json(root / "status.json", {"status": "running", "phase": phase,
                        "coordinator_pid": os.getpid(), "worker_pid": child.pid,
                        "worker_created_at": created, "command": command})
            for line in child.stdout:
                print(line, end="", flush=True)
            code = child.wait()
        # 已退出的 PID 不再属于本实验，避免 Windows 复用 PID 后误阻止恢复。
        _write_json(root / "status.json", {"status": "phase_complete" if code == 0 else "phase_failed",
                    "phase": phase, "worker_pid": None, "exit_code": code, "command": command})
        if code:
            raise RuntimeError(f"{phase} failed (exit {code}); inspect workflow.log")

    def completed_checkpoint(self, folder):
        pointer = folder / "training.json"
        if not pointer.exists():
            raise ValueError(f"no completed training for {folder.name}; run --stage train first")
        done = read_json(pointer)
        checkpoint = Path(done["checkpoint"])
        if _file_sha256(checkpoint) != done["sha256"]:
            raise ValueError("completed training checkpoint changed")
        spec = parse_ablation(read_json(folder / "ablation.json"))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _validate_checkpoint_contract(payload, load_config(folder / "config.yaml"), ablation=spec)
        return checkpoint

    def training(self, folder, args):
        if (folder / "training.json").exists():
            return self.completed_checkpoint(folder)
        attempts = sorted((folder / "training").glob("attempt_*"))
        previous = attempts[-1] if attempts else None
        if previous is not None and (previous / "summary.json").exists():
            attempt = previous
        else:
            attempt = folder / "training" / f"attempt_{len(attempts) + 1:03d}"
            command = [sys.executable, "-u", str(self.bundle / "run.py"), "--worker", "train",
                       "--root", str(folder), "--config", str(folder / "config.yaml"),
                       "--attempt", str(attempt)]
            if previous is not None and (previous / "training_state.pt").exists():
                command += ["--resume-state", str(previous / "training_state.pt")]
            self.execute(folder, f"training_{folder.name}", command)
        checkpoint = attempt / "best.pt"
        _write_json(folder / "training.json", {"checkpoint": str(checkpoint),
                    "sha256": _file_sha256(checkpoint), "config": str(attempt / "config.yaml"),
                    "reused": False, "canonical_variant": folder.name,
                    "ablation": read_json(folder / "ablation.json")})
        return self.completed_checkpoint(folder)

    def predict_variant(self, folder, config, test_root, checkpoint, args, *, splits=("val", "test")):
        # 概率重评分与原始掩码计数保留两套独立实现，均使用冻结验证阈值。
        for split in splits:
            data = Path(config.dataset.root) if split == "val" else test_root
            output = folder / split
            command = [sys.executable, "-u", "-m", "traceformer.experiments.evaluation",
                       "--config", str(folder / "config.yaml"), "--checkpoint", str(checkpoint),
                       "--dataset-root", str(data), "--split", split, "--output", str(output),
                       "--device", config.training.device, "--num-threads", str(config.training.num_threads),
                       "--ablation", str(folder / "ablation.json")]
            if args.smoke:
                command += ["--max-files", "2"]
            if not (output / "report.json").exists():
                self.execute(folder, f"predict_{split}", command)
            self.execute(folder, f"rescore_{split}", [*command, "--rescore-only"])
            report = read_json(output / "independent_report.json")
            if report["checkpoint_sha256"] != _file_sha256(checkpoint):
                raise ValueError("prediction checkpoint identity differs")
            if report["sequences"] != (2 if args.smoke else 24) or report["test_threshold_selection"]:
                raise ValueError("evaluation partition count or frozen threshold protocol differs")
            raw = evaluate_predictions(output / "predictions", data / split, names=report["files"])
            verify_raw_score(report["competition"], raw["aggregate"])
            for name, metric in report["per_sequence"].items():
                verify_raw_score(metric, raw["files"][name])
            _write_json(output / "raw_independent_report.json", raw)
            if split == "val" and report["competition"] != read_json(checkpoint.parent / "metrics.json")["competition"]:
                raise ValueError("independent validation does not reproduce saved best counts")

    def archive_results(self, root):
        archive = self.output / ("smoke_results.zip" if root.name == "smoke" else "results.zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.name != "run.lock":
                    zipped.write(path, path.relative_to(root.parent).as_posix())
            for path in sorted((self.bundle / "baseline").glob("*")):
                if path.is_file():
                    zipped.write(path, "baseline/" + path.name)
        # 回传前验证ZIP条目CRC，并提供整个归档的SHA256供本地核验。
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("result archive CRC verification failed")
        checksum = archive.with_suffix(".zip.sha256")
        checksum.write_text(f"{_file_sha256(archive)}  {archive.name}\n", encoding="utf-8")
        print(f"complete; copy back {archive} and {checksum.name}", flush=True)

    def worker(self, args):
        if args.worker != "train":
            raise ValueError(f"unknown worker: {args.worker}")
        config = load_config(args.config)
        setup_device(config)
        spec = parse_ablation(read_json(args.root / "ablation.json"))
        run_training(config, output_dir=args.attempt, progress=True, ablation=spec,
                     config_source=args.config, resume_state_path=args.resume_state,
                     note=f"{self.experiment} {args.root.name}")

    def run(self, args):
        if args.worker:
            self.worker(args)
            return
        root = self.output / ("smoke" if args.smoke else "full")
        root.mkdir(parents=True, exist_ok=True)
        with exclusive_run(root), (root / "workflow.log").open("a", encoding="utf-8") as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                self.guard_active_workers(root)
                try:
                    previous = read_json(root / "status.json") if (root / "status.json").exists() else {}
                    if previous.get("status") == "failed" and not args.resume:
                        raise ValueError("previous phase failed; inspect workflow.log then rerun with --resume")
                    digest = self.verify_package()
                    config, test_root = self.configure(args, root, digest)
                    runtime = setup_device(config)
                    _write_json(root / "environment.json", {**runtime.metadata, "python": sys.version,
                        "platform": platform.platform(),
                        "precision": "bf16" if config.training.mixed_precision else "fp32",
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ("torch", "numpy", "scipy", "PyYAML", "scikit-learn", "matplotlib", "psutil")}})
                    self.stages(root, config, test_root, args)
                except Exception:
                    _write_json(root / "status.json", {"status": "failed", "error": traceback.format_exc()})
                    raise
