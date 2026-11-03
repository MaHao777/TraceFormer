"""七种S1论文筛选消融；复用公共训练、计时、复评和恢复运行层。"""

import shutil
from pathlib import Path

from traceformer.experiments.ablation import S1PaperSpec, parse_ablation
from traceformer.experiments.component_suite import read_json
from traceformer.experiments.runtime import (
    freeze_execution_environment,
    validate_benchmark_coverage,
)
from traceformer.experiments.s1_paper import (
    LABELS,
    MECHANISM_ROWS,
    STRUCTURE_ROWS,
    VARIANTS,
    initialization_audit,
    specification,
)
from traceformer.experiments.workflows.local_pooling import CHUNKS, LocalPoolingWorkflow
from traceformer.training.engine import _file_sha256, _write_json


class S1PaperWorkflow(LocalPoolingWorkflow):
    experiment = "s1_paper_ablation"
    variants = VARIANTS
    comparison_pairs = tuple((name, "S1") for name in VARIANTS[1:])
    interaction_pairs = None
    title = "S1论文模型：结构消融与创新点消融（seed37筛选）"
    limitations = "single seed screening; no VTR; matched union comparison does not isolate coordinate anchoring"
    execution_chunks = {**CHUNKS, "TRACEFORMER_S1_DIRECT_QUERY_CHUNK": "256"}

    def scientific_contract(self):
        return {"profile": "screen", "ablation_version": 6, "seed": 37,
                "mandatory_preflight": True, "structure_rows": list(STRUCTURE_ROWS),
                "mechanism_rows": list(MECHANISM_ROWS), "direct_support": "three-cone union, deduplicated, no cap"}

    def specification(self, variant, smoke=False):
        source = self.bundle / "baseline" / variant / "identity.json"
        if not smoke and variant in {"S1", "S0"} and source.is_file():
            return parse_ablation(read_json(source)["ablation"])
        return specification(variant)

    def configure(self, args, root, package_digest):
        config, test_root = super().configure(args, root, package_digest)
        S1PaperSpec().validate_stage(config.training.stage)
        audit = initialization_audit(config)
        if read_json(self.bundle / "initialization_audit.json") != audit:
            raise ValueError("S1 initialization differs from the frozen package")
        path = root / "initialization_audit.json"
        if path.exists() and read_json(path) != audit:
            raise ValueError("S1 paired initialization changed")
        _write_json(path, audit)
        for row in audit["variants"]:
            _write_json(root / row["variant"] / "structure.json",
                        {**row, "checkpoint_contract": self.specification(row["variant"], args.smoke).to_dict()})
        freeze_execution_environment(config, root, "S1 paper")
        return config, test_root

    def baseline(self, root, config, test_root, args):
        # 两份历史基线均独立复现验证计数后，才允许开始七种新结构训练。
        audits = {}
        for variant in ("S1", "S0"):
            folder = root / variant
            source = self.bundle / "baseline" / variant
            if args.smoke or not source.exists():
                self.training(folder, args)
                continue
            identity = read_json(source / "identity.json")
            imported = folder / "training/imported"
            imported.mkdir(parents=True, exist_ok=True)
            for name, expected in identity["source_files"].items():
                if _file_sha256(source / name) != expected:
                    raise ValueError("S1 reference source identity differs")
                target = imported / name
                if target.exists() and _file_sha256(target) != expected:
                    raise ValueError("S1 imported baseline changed")
                if not target.exists():
                    shutil.copy2(source / name, target)
            pointer = folder / "training.json"
            _write_json(pointer, {"checkpoint": str(imported / "best.pt"),
                "sha256": identity["checkpoint_sha256"], "config": str(imported / "config.yaml"),
                "reused": True, "canonical_variant": variant, "ablation": identity["ablation"]})
            try:
                checkpoint = self.completed_checkpoint(folder)
                self.predict_variant(folder, config, test_root, checkpoint, args, splits=("val",))
            except Exception:
                pointer.unlink(missing_ok=True)
                raise
            audits[variant] = {"checkpoint_sha256": identity["checkpoint_sha256"],
                               "validation_reproduced": True, "original_bytes_preserved": True}
        _write_json(root / "baseline_audit.json", {"passed": True, "references": audits})
        return self.completed_checkpoint(root / "S1")

    def validate_benchmark_identity(self, report, spec, checkpoint, runtime, identities, *, inference_split="val"):
        super().validate_benchmark_identity(report, spec, checkpoint, runtime, identities,
                                            inference_split=inference_split)
        validate_benchmark_coverage(report, checkpoint, identities, inference_split)

    def stages(self, root, config, test_root, args):
        support = root / "support_geometry.json"
        if not support.exists():
            import sys
            command = [sys.executable, "-u", str(self.bundle / "run.py"), "--worker", "support",
                       "--root", str(root), "--config", str(root / "S1/config.yaml"), "--report", str(support)]
            if args.smoke:
                command.append("--smoke")
            self.execute(root, "support_geometry", command)
        recorded = read_json(support)
        identities = read_json(self.bundle / "dataset_manifest.json")["val"]
        expected = sorted(identities)[:2] if args.smoke else sorted(identities)
        if (recorded["smoke"] != args.smoke or recorded["direct_union_candidates"] != 552
                or [row["file"] for row in recorded["files"]] != expected
                or any(row["source_sha256"] != identities[row["file"]]["sha256"] for row in recorded["files"])):
            raise ValueError("cached support geometry identity differs")
        if args.stage in {"all", "train"}:
            self.benchmark_suite(root, args, "preflight")
        super().stages(root, config, test_root, args)

    def worker(self, args):
        if args.worker == "support":
            from traceformer.config import load_config
            from traceformer.experiments.s1_support import measure_support
            _write_json(args.report, measure_support(load_config(args.config), self.bundle, args.smoke))
        else:
            super().worker(args)

    def extra_results(self, root, rows, comparisons):
        by_name = {row["variant"]: row for row in rows}
        tables = {"structure": list(STRUCTURE_ROWS), "mechanism": list(MECHANISM_ROWS)}
        lines = ["", "以下均为seed37筛选；所有差值相对S1，不作多种子显著性解释。"]
        for key, title in (("structure", "模型结构消融"), ("mechanism", "主要创新点消融")):
            lines += ["", f"## {title}", "", "|组|定义|参数|验证Score|测试Score|测试Δ/S1|", "|---|---|---:|---:|---:|---:|"]
            for name in tables[key]:
                row = by_name[name]
                delta = row["test"]["competition"]["score"] - by_name["S1"]["test"]["competition"]["score"]
                lines.append(f"|{name}|{LABELS[name]}|{row['test']['parameters']}|"
                    f"{row['val']['competition']['score']:.9f}|{row['test']['competition']['score']:.9f}|{delta:+.9f}|")
        lines += ["", "## 训练记录", "", "|组|训练来源|完成轮数|训练与逐轮验证分钟|", "|---|---|---:|---:|"]
        for row in rows:
            history = read_json(Path(row["training"]["checkpoint"]).parent / "history.json")
            seconds = sum(epoch["optimization"]["elapsed_seconds"] for epoch in history)
            row["training"]["completed_epochs"] = len(history)
            row["training"]["epoch_seconds_including_validation"] = seconds
            lines.append(f"|{row['variant']}|{'历史原字节复用' if row['training']['reused'] else '从头训练'}|{len(history)}|{seconds/60:.2f}|")
        lines += ["", "固定窗口近似体积匹配，不是严格等邻居数；G2同时减少尺度与槽位。",
                  "T1匹配三锥并集，不单独证明中心坐标锚定；参数与计算开销均如实记录。",
                  "A1的精修Head仍有上下文；仅过去编码器不意味着整网严格因果。",
                  "P1与S1共享卷积中心初值；F1的64→128→64 MLP接近原MHA容量。"]
        return {"ablation_tables": tables, "profile": "screen", "seed": 37,
                "initialization_audit": read_json(root / "initialization_audit.json"),
                "support_geometry": read_json(root / "support_geometry.json"),
                "baseline_audit": read_json(root / "baseline_audit.json")}, lines
