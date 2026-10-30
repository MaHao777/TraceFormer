"""实验统一入口：查看、打包、运行、续跑与历史复现。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = {
    "mean_past_gated": ("M1+C4", "均值池化与三层门控过去锥"),
    "local_pooling_six": ("S0/S1/S2/L0/L1/L2", "池化与过去锥局部注意力六组对照"),
}
REPO = Path(__file__).resolve().parents[3]


def run_arguments(parser, *, workers=False):
    parser.add_argument("--stage", choices=("all", "preflight", "train", "predict"), default="all")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    if workers:
        parser.add_argument("--output", type=Path)
        parser.add_argument("--worker", choices=("gate", "train", "benchmark", "prediction_benchmark"),
                            help=argparse.SUPPRESS)
        for name in ("root", "config", "attempt", "resume-state", "checkpoint", "report"):
            parser.add_argument("--" + name, type=Path, help=argparse.SUPPRESS)


def bundled_main(bundle, argv=None):
    """上传包和本地冻结源码都从这里选择 workflow。"""
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "package_manifest.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=f"冻结实验：{manifest['experiment']}")
    run_arguments(parser, workers=True)
    args = parser.parse_args(argv)
    if manifest["experiment"] == "mean_past_gated":
        from traceformer.experiments.workflows.mean_past import MeanPastWorkflow
        workflow = MeanPastWorkflow(bundle, args.output)
    elif manifest["experiment"] == "local_pooling_six":
        if args.precision not in (None, "fp32"):
            parser.error("local_pooling_six requires CUDA FP32")
        from traceformer.experiments.workflows.local_pooling import LocalPoolingWorkflow
        workflow = LocalPoolingWorkflow(bundle, args.output)
    else:
        parser.error("unknown maintained experiment")
    workflow.run(args)


def packaging_arguments(parser):
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="默认使用无 Sparse 完整训练配置")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--without-baseline", action="store_true")
    parser.add_argument("--references", type=Path, help="可选历史参考指标 JSON")


def build_from_arguments(args, destination):
    from traceformer.experiments.packaging import build_package
    baseline = None
    if args.baseline_root is not None and args.without_baseline:
        raise ValueError("--baseline-root and --without-baseline cannot be combined")
    if args.experiment == "local_pooling_six" and not args.without_baseline:
        baseline = args.baseline_root or REPO / "remote_result/26-10-28"
    elif args.baseline_root is not None:
        raise ValueError("only local_pooling_six accepts --baseline-root")
    references = json.loads(args.references.read_text(encoding="utf-8")) if args.references else None
    return build_package(args.experiment, destination,
                         args.config or REPO / "outputs/no_sparse_local_20261025/training/config.yaml",
                         args.data_root or REPO / "data/ev-uav",
                         args.test_root or REPO.parent / "EV-UAV-dataset",
                         baseline_root=baseline, references=references)


def run_repository(args):
    """先冻结独立源码；后续 worker 始终导入该快照。"""
    output = args.output.resolve()
    bundle = output / "source"
    if not bundle.exists():
        if output.exists():
            raise FileExistsError("output exists without this experiment's frozen source; use a new --output")
        if args.resume:
            raise ValueError("cannot resume: frozen source is missing")
        build_from_arguments(args, bundle)
        # 首次运行传入打包时采用的数据目录；子进程 cwd 已切到输出目录。
        args.data_root = args.data_root or REPO / "data/ev-uav"
        args.test_root = args.test_root or REPO.parent / "EV-UAV-dataset"
    else:
        manifest = json.loads((bundle / "package_manifest.json").read_text(encoding="utf-8"))
        if manifest["experiment"] != args.experiment:
            raise ValueError("output belongs to a different experiment")
        # 当前工作副本也要与快照一致，避免误以为恢复运行了新实现。
        from traceformer.training.engine import _file_sha256
        current = {p.relative_to(REPO).as_posix(): _file_sha256(p)
                   for p in (REPO / "src/traceformer").rglob("*.py") if "__pycache__" not in p.parts}
        frozen = {k: v for k, v in manifest["sha256"].items() if k.startswith("src/") and k.endswith(".py")}
        if current != frozen:
            raise ValueError("repository source changed; resume with the original frozen source/run.py")
        if args.baseline_root is not None or args.without_baseline or args.references is not None:
            raise ValueError("baseline/references are fixed at creation; omit packaging options on resume")
        if args.config is not None and _file_sha256(args.config) != manifest["sha256"]["config.yaml"]:
            # 打包配置会规范化路径，恢复配置由 experiment.json 严格核验。
            original = json.loads((bundle / "creation.json").read_text(encoding="utf-8"))
            if _file_sha256(args.config) != original["config_source_sha256"]:
                raise ValueError("configuration source changed since the run was frozen")
    command = [sys.executable, "-u", str(bundle / "run.py"), "--output", str(output), "--stage", args.stage]
    for name in ("data_root", "test_root", "device", "precision"):
        value = getattr(args, name)
        if value is not None:
            if isinstance(value, Path):
                value = value.resolve()
            command += ["--" + name.replace("_", "-"), str(value)]
    for flag in ("smoke", "resume"):
        if getattr(args, flag):
            command.append("--" + flag)
    result = subprocess.run(command, cwd=output, env=dict(os.environ, PYTHONPATH=""))
    return result.returncode


def status(output, smoke=False):
    """状态只读取产物，不导入训练或评分依赖。"""
    root = output.resolve()
    if not (root / "status.json").exists():
        root = root / ("smoke" if smoke else "full")
    path = root / "status.json"
    if not path.exists():
        raise FileNotFoundError(f"no experiment status: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    result["output"] = str(root)
    result["logs"] = str(root / "workflow.log")
    result["training"] = {p.parent.relative_to(root).as_posix(): json.loads(p.read_text(encoding="utf-8"))
                          for p in root.rglob("training.json")}
    result["results"] = str(root / "results.md") if (root / "results.md").exists() else None
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="traceformer-experiment", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="查看维护中与归档实验")
    bundle = commands.add_parser("bundle", help="生成独立上传包")
    packaging_arguments(bundle)
    bundle.add_argument("--data-root", type=Path)
    bundle.add_argument("--test-root", type=Path)
    run = commands.add_parser("run", help="冻结源码并执行实验")
    packaging_arguments(run)
    run_arguments(run)
    state = commands.add_parser("status", help="查看运行状态和产物位置")
    state.add_argument("--output", required=True, type=Path)
    state.add_argument("--smoke", action="store_true")
    archived = commands.add_parser("archive-run", help="隔离运行匹配源码的历史实验")
    archived.add_argument("archive_id")
    archived.add_argument("--source-root", type=Path)
    archived.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            from traceformer.experiments.archive import archive_inventory
            print(json.dumps({"maintained": {k: {"variants": v[0], "description": v[1]}
                                              for k, v in EXPERIMENTS.items()},
                              "archived": archive_inventory(REPO)}, ensure_ascii=False, indent=2))
        elif args.command == "bundle":
            print(build_from_arguments(args, args.output))
        elif args.command == "run":
            code = run_repository(args)
            if code:
                raise SystemExit(code)
        elif args.command == "status":
            status(args.output, args.smoke)
        else:
            from traceformer.experiments.archive import run_archive
            raise SystemExit(run_archive(REPO, args.archive_id, args.source_root, args.arguments))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
