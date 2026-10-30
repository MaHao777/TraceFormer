"""六组池化与过去锥局部注意力；耗时和显存仅记录。"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from traceformer.config import load_config
from traceformer.experiments.ablation import ComponentCoreSpec, LocalPoolingSpec, parse_ablation
from traceformer.experiments.component_suite import read_json
from traceformer.experiments.local_pooling_benchmark import run_benchmark, run_prediction_benchmark
from traceformer.experiments.runtime import ExperimentRuntime, setup_device
from traceformer.training.engine import _atomic_write_text, _file_sha256, _serializable, _write_json

VARIANTS = ("S0", "S1", "S2", "L0", "L1", "L2")
CHUNKS = {"TRACEFORMER_LOCAL_QUERY_CHUNK": "8192", "TRACEFORMER_SPATIAL_TOKEN_CHUNK": "4096"}


class LocalPoolingWorkflow(ExperimentRuntime):
    """六组统一训练后评分，保留 S0 原字节和原契约。"""

    experiment = "local_pooling_six"

    def configure(self, args, root, package_digest):
        prior = read_json(root / "experiment.json") if (root / "experiment.json").exists() else None
        saved = prior["config"] if prior else None
        data_root = args.data_root or (Path(saved["dataset"]["root"]) if saved else None)
        if data_root is None:
            raise ValueError("first run requires --data-root containing train/ and val/")
        data_root = data_root.resolve()
        test_root = (args.test_root or (Path(prior["test_root"]) if prior else data_root)).resolve()
        device = args.device or (saved["training"]["device"] if saved else "cuda")
        if device != "cuda" and not args.smoke:
            raise ValueError("formal six-group experiments require CUDA FP32; CPU is smoke-only")
        base = load_config(self.bundle / "config.yaml")
        config = replace(base, dataset=replace(base.dataset, root=data_root), training=replace(
            base.training, device=device, output_root=root, mixed_precision=False,
            epochs=1 if args.smoke else 24, seed=37, num_threads=1 if args.smoke else 8,
            max_train_files=2 if args.smoke else None, max_val_files=2 if args.smoke else None,
            save_training_state_interval=1,
        )).validate()
        if not args.smoke:
            m = config.model
            actual = (m.spatial_block_size_x, m.spatial_block_size_y, m.num_time_blocks,
                      m.token_dim, m.event_embed_dim, m.num_heads, m.num_layers)
            if actual != (8, 8, 64, 64, 64, 4, 3):
                raise ValueError("formal experiment requires 8x8/64 time blocks/64 dimensions/4 heads/3 layers")
            if ((config.resolution.width, config.resolution.height, config.dataset.duration_ms)
                    != (346, 260, 8000)
                    or [(window.spatial_radius, window.time_horizon) for window in m.cone_windows]
                    != [(8, 2), (4, 4), (2, 8)]):
                raise ValueError("formal experiment resolution/duration/summary cone geometry differs")
        # 恢复沿用原分块，新的默认值不改变已有训练的随机数调用顺序。
        saved_chunks = prior.get("execution_chunks", {}) if prior else {}
        chunks = {name: os.environ.get(name, saved_chunks.get(name, value)) for name, value in CHUNKS.items()}
        for name, value in chunks.items():
            if not value.isdecimal() or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
            os.environ[name] = value
        contract = {"variants": list(VARIANTS), "config": _serializable(asdict(config)),
                    "test_root": str(test_root), "precision": "fp32", "smoke": args.smoke,
                    "execution_chunks": chunks, "package_sha256": package_digest,
                    "resource_policy": "timing_only"}
        comparable = ({**prior, "test_root": contract["test_root"]}
                      if prior and not any((root / name / "test/report.json").exists() for name in VARIANTS)
                      else prior)
        if prior is not None and comparable != contract:
            raise ValueError("experiment settings/source changed; use original settings and preserve this output")
        checked = self.verify_data(data_root, test_root, need_test=args.stage in {"all", "predict"})
        _write_json(root / "experiment.json", contract)
        _write_json(root / "data_verification.json", checked)
        _write_json(root / "dataset_identity.json", read_json(self.bundle / "dataset_manifest.json"))
        _write_json(root / "package_manifest.json", read_json(self.bundle / "package_manifest.json"))
        _atomic_write_text(root / "config.yaml", yaml.safe_dump(contract["config"], allow_unicode=True,
                                                              sort_keys=False))
        for name in VARIANTS:
            folder = root / name
            folder.mkdir(exist_ok=True)
            spec = self.specification(name, args.smoke)
            _write_json(folder / "ablation.json", spec.to_dict())
            cfg = replace(config, training=replace(config.training, output_root=folder))
            _atomic_write_text(folder / "config.yaml", yaml.safe_dump(
                _serializable(asdict(cfg)), allow_unicode=True, sort_keys=False))
        return config, test_root

    def specification(self, variant, smoke=False):
        # S0直接读取原权重与原契约，不伪造或重写历史检查点。
        if variant == "S0" and not smoke and (self.bundle / "baseline/best.pt").exists():
            return ComponentCoreSpec("M1_C4")
        return LocalPoolingSpec(variant=variant)

    def benchmark_worker(self, args):
        config = load_config(args.config)
        spec = parse_ablation(read_json(args.root / "ablation.json"))
        # 基准模块统一执行数值对照与完整训练/推理计时，不复制其实现。
        if args.worker == "prediction_benchmark":
            report = run_prediction_benchmark(config, spec, checkpoint=args.checkpoint,
                                              test_root=args.test_root, smoke=args.smoke)
        else:
            report = run_benchmark(config, spec, checkpoint=args.checkpoint, smoke=args.smoke)
        _write_json(args.report, report)

    def validate_benchmark_identity(self, report, spec, checkpoint, runtime, identities, *, inference_split="val"):
        """缓存计时仍需对应当前GPU、权重、结构和全部原始输入。"""
        expected_hash = _file_sha256(checkpoint) if checkpoint is not None else None
        if report.get("checkpoint_sha256") != expected_hash or report.get("variant") != spec.to_dict():
            raise ValueError("cached benchmark checkpoint/variant identity differs")
        if report.get("runtime") != runtime:
            raise ValueError("cached benchmark runtime differs from the current device")
        for part, split in (("training", "train"), ("inference", inference_split)):
            for row in report.get(part, {}).get("sequences", []):
                source = identities[split].get(row["file"])
                if source is None or row.get("source_sha256") != source["sha256"]:
                    raise ValueError("cached benchmark source data identity differs")

    def benchmark_suite(self, root, args, stage, *, test_root=None):
        report_root = root / "benchmark" / stage
        report_root.mkdir(parents=True, exist_ok=True)
        reports = {}
        runtime = setup_device(load_config(root / "S0/config.yaml")).metadata
        identities = read_json(self.bundle / "dataset_manifest.json")
        for name in VARIANTS:
            folder = root / name
            destination = report_root / f"{name}.json"
            checkpoint = None
            if stage == "trained":
                checkpoint = self.completed_checkpoint(folder)
            elif name == "S0" and not args.smoke and (self.bundle / "baseline/best.pt").exists():
                checkpoint = self.bundle / "baseline/best.pt"
            # 旧版 trained 报告包含训练步骤与验证计时，保留原文件并另测测试预测。
            if destination.exists() and stage == "trained" and not read_json(destination).get("timing_only"):
                destination = report_root / f"{name}_prediction.json"
            if not destination.exists():
                worker = "prediction_benchmark" if stage == "trained" else "benchmark"
                command = [sys.executable, "-u", str(self.bundle / "run.py"), "--worker", worker,
                           "--root", str(folder), "--config", str(folder / "config.yaml"),
                           "--report", str(destination)]
                if stage == "trained":
                    command += ["--test-root", str(test_root)]
                if checkpoint is not None:
                    command += ["--checkpoint", str(checkpoint)]
                if args.smoke:
                    command.append("--smoke")
                self.execute(root, f"{stage}_benchmark_{name}", command)
            reports[name] = read_json(destination)
            self.validate_benchmark_identity(reports[name], self.specification(name, args.smoke),
                                        checkpoint, runtime, identities,
                                        inference_split="test" if stage == "trained" else "val")
        # 耗时和显存只记录，不再按相对比例或百分位中断实验。
        _write_json(root / f"{stage}_timing.json", {"budget_enforced": False, "reports": reports})
        return reports

    def baseline(self, root, config, test_root, args):
        folder = root / "S0"
        pointer = folder / "training.json"
        if pointer.exists():
            checkpoint = self.completed_checkpoint(folder)
            if not read_json(pointer).get("reused", False):
                return checkpoint
        if args.smoke or not (self.bundle / "baseline/best.pt").exists():
            return self.training(folder, args)
        identity = read_json(self.bundle / "baseline/identity.json")
        checkpoint = self.bundle / "baseline/best.pt"
        if _file_sha256(checkpoint) != identity["checkpoint_sha256"]:
            raise ValueError("imported baseline checkpoint identity differs")
        _write_json(pointer, {"checkpoint": str(checkpoint), "sha256": identity["checkpoint_sha256"],
                             "reused": True, "canonical_variant": "S0",
                             "ablation": ComponentCoreSpec("M1_C4").to_dict()})
        # 先独立复评全部验证文件；复现失败不得开始任何新组训练。
        try:
            self.predict_variant(folder, config, test_root, checkpoint, args, splits=("val",))
        except Exception:
            pointer.unlink(missing_ok=True)
            raise
        _write_json(root / "baseline_audit.json", {"passed": True, "reused": True,
                    "checkpoint_sha256": identity["checkpoint_sha256"], "validation_reproduced": True,
                    "source_identity": identity})
        return checkpoint

    def finish(self, root, reports, args):
        rows = []
        for name in VARIANTS:
            splits = {split: read_json(root / name / split / "independent_report.json")
                      for split in ("val", "test")}
            rows.append({"variant": name, **splits, "benchmark": reports[name],
                         "training": read_json(root / name / "training.json")})
        comparisons = {}
        by_name = {row["variant"]: row for row in rows}
        for split in ("val", "test"):
            score = {name: row[split]["competition"]["score"] for name, row in by_name.items()}
            comparisons[split] = {f"{a}-{b}": score[a] - score[b]
                                  for a, b in (("L0", "S0"), ("S1", "S0"), ("L1", "L0"),
                                               ("S2", "S1"), ("L2", "L1"))}
            comparisons[split]["single_kernel_interaction"] = (score["L1"] - score["L0"]) - (score["S1"] - score["S0"])
        _write_json(root / "results.json", {"complete": True, "diagnostic_only": args.smoke,
                    "rows": rows, "comparisons": comparisons,
                    "experiment": read_json(root / "experiment.json"),
                    "limitations": "single seed; encoder geometry and interaction change together; convolution includes nonlinearity"})
        lines = ["# 均值/空间卷积池化 × 过去锥局部注意力", "",
                 "耗时和显存仅记录，不设置预算门槛；CPU smoke仅检查接口。", ""]
        for split in ("val", "test"):
            lines += [f"## {split}", "", "|模型|参数|Score|IoU|SegAcc|Pd|Fa|AP|阈值|ΔScore/S0|",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for row in rows:
                report = row[split]
                m = report["competition"]
                delta = m["score"] - by_name["S0"][split]["competition"]["score"]
                lines.append(f"|{row['variant']}|{report['parameters']}|{m['score']:.9f}|{m['iou']:.6f}|"
                             f"{m['seg_acc']:.6f}|{m['pd']:.6f}|{m['fa']:.8g}|"
                             f"{report['average_precision']:.6f}|{report['threshold']:.6f}|{delta:+.9f}|")
            lines.append("")
        lines += ["## 测试集完整网络预测时间", "",
                  "|模型|前向平均毫秒|前向P95毫秒|峰值MiB|前向相对S0|",
                  "|---|---:|---:|---:|---:|"]
        baseline_mean = reports["S0"]["inference"]["mean_seconds"]
        for name in VARIANTS:
            bench = reports[name]
            inference = bench["inference"]
            lines.append(f"|{name}|{1000 * inference['mean_seconds']:.3f}|"
                         f"{1000 * inference['p95_seconds']:.3f}|{bench['peak_memory_bytes'] / 2**20:.1f}|"
                         f"{inference['mean_seconds'] / baseline_mean:.4f}|")
        lines += ["", "逐文件预测耗时见benchmark/trained及trained_timing.json；无时间预算门槛。",
                  "计时不使用读取缓存概率后的重评分耗时；测试固定验证阈值。"]
        _atomic_write_text(root / "results.md", "\n".join(lines) + "\n")
        hashes = {path.relative_to(root).as_posix(): _file_sha256(path) for path in sorted(root.rglob("*"))
                  if path.is_file() and path.name not in {"completion_audit.json", "workflow.log", "status.json", "run.lock"}}
        _write_json(root / "completion_audit.json", {"complete": True, "sha256": hashes,
                    "variants": list(VARIANTS), "partitions": {s: rows[0][s]["sequences"] for s in ("val", "test")},
                    "budget_enforced": False, "timing_split": "test", "diagnostic_only": args.smoke})
        _write_json(root / "status.json", {"status": "complete", "variants": list(VARIANTS),
                                           "diagnostic_only": args.smoke})
        self.archive_results(root)

    def worker(self, args):
        if args.worker in {"benchmark", "prediction_benchmark"}:
            self.benchmark_worker(args)
        else:
            super().worker(args)

    def stages(self, root, config, test_root, args):
        if args.stage == "preflight":
            # 显式诊断才运行耗时预检，不以相对预算中止训练。
            self.benchmark_suite(root, args, "preflight")
            _write_json(root / "status.json", {"status": "preflight_complete", "diagnostic_only": args.smoke})
            return
        if args.stage in {"all", "train"}:
            self.baseline(root, config, test_root, args)
            for name in VARIANTS[1:]:
                self.training(root / name, args)
        for name in VARIANTS:
            self.completed_checkpoint(root / name)
        if args.stage == "predict":
            self.baseline(root, config, test_root, args)
        if args.stage in {"all", "predict"}:
            for name in VARIANTS:
                self.predict_variant(root / name, config, test_root,
                                     self.completed_checkpoint(root / name), args)
            reports = self.benchmark_suite(root, args, "trained", test_root=test_root)
            self.finish(root, reports, args)
        else:
            _write_json(root / "status.json", {"status": "training_complete", "diagnostic_only": args.smoke})
