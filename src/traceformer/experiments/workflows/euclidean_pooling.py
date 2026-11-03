"""四组欧氏锥的科学配置；执行、恢复、计时及独立评估共用现有运行层。"""

from pathlib import Path

from traceformer.experiments.ablation import EuclideanPoolingSpec
from traceformer.experiments.component_suite import read_json
from traceformer.experiments.euclidean_pooling import GEOMETRY, VARIANTS, initialization_audit
from traceformer.experiments.runtime import (
    freeze_execution_environment,
    validate_benchmark_coverage,
)
from traceformer.experiments.workflows.local_pooling import LocalPoolingWorkflow
from traceformer.training.engine import _write_json


class EuclideanPoolingWorkflow(LocalPoolingWorkflow):
    experiment = "euclidean_pooling_four"
    variants = VARIANTS
    comparison_pairs = (("E1", "E0"), ("E3", "E2"), ("E2", "E0"), ("E3", "E1"), ("E3", "E0"))
    interaction_pairs = (("E3", "E2"), ("E1", "E0"))
    title = "均值/单核池化 × 局部/中心锚定欧氏过去锥"

    def scientific_contract(self):
        return {"geometry": GEOMETRY, "ablation_version": 5, "baseline_reused": False,
                "mandatory_preflight": True,
                "actual_windows": {v: GEOMETRY["local_windows" if v in {"E0", "E1"} else "summary_windows"]
                                   for v in VARIANTS}}

    def specification(self, variant, smoke=False):
        return EuclideanPoolingSpec(variant)

    def configure(self, args, root, package_digest):
        if (self.bundle / "baseline").exists():
            raise ValueError("Euclidean experiments must not import historical trained baselines")
        config, test_root = super().configure(args, root, package_digest)
        EuclideanPoolingSpec().validate_stage(config.training.stage)
        audit = initialization_audit(config)
        if read_json(self.bundle / "initialization_audit.json") != audit:
            raise ValueError("Euclidean initialization differs from the frozen package")
        path = root / "initialization_audit.json"
        if path.exists() and read_json(path) != audit:
            raise ValueError("Euclidean paired initialization changed")
        _write_json(path, audit)
        for row in audit["variants"]:
            _write_json(root / row["variant"] / "structure.json", row)
        freeze_execution_environment(config, root, "Euclidean")
        return config, test_root

    def baseline(self, root, config, test_root, args):
        return self.training(root / "E0", args)

    def validate_benchmark_identity(self, report, spec, checkpoint, runtime, identities, *, inference_split="val"):
        super().validate_benchmark_identity(report, spec, checkpoint, runtime, identities,
                                            inference_split=inference_split)
        validate_benchmark_coverage(report, checkpoint, identities, inference_split)

    def stages(self, root, config, test_root, args):
        # 显式 train 也执行首次预检，避免绕过新几何的真实 CUDA 检查。
        if args.stage in {"all", "train"}:
            self.benchmark_suite(root, args, "preflight")
        super().stages(root, config, test_root, args)

    def extra_results(self, root, rows, comparisons):
        lines = ["", "## 配对 Score 差值", "", "|比较|验证|测试|", "|---|---:|---:|"]
        for name in comparisons["val"]:
            lines.append(f"|{name}|{comparisons['val'][name]:+.9f}|{comparisons['test'][name]:+.9f}|")
        lines += ["", "## 训练耗时", "", "|模型|完成轮数|训练与逐轮验证秒数|", "|---|---:|---:|"]
        diagnostics = {}
        for row in rows:
            history = read_json(Path(row["training"]["checkpoint"]).parent / "history.json")
            elapsed = sum(epoch["optimization"]["elapsed_seconds"] for epoch in history)
            row["training"]["epoch_seconds_including_validation"] = elapsed
            row["training"]["completed_epochs"] = len(history)
            lines.append(f"|{row['variant']}|{len(history)}|{elapsed:.3f}|")
            diagnostics[row["variant"]] = {
                split: {name: row[split]["per_sequence"][name] for name in names
                        if name in row[split]["per_sequence"]}
                for split, names in (("val", ("val_023.npz",)),
                                     ("test", ("test_022.npz", "test_023.npz")))}
        lines += ["", "## 困难序列诊断", "", "|模型|序列|Score|Pd|Fa|", "|---|---|---:|---:|---:|"]
        for variant, splits in diagnostics.items():
            for sequences in splits.values():
                for name, metric in sequences.items():
                    lines.append(f"|{variant}|{name}|{metric['score']:.9f}|{metric['pd']:.6f}|{metric['fa']:.8g}|")
        lines += ["", "单种子观察；两类注意力同时改变支持域和交互方式，单核处理包含GELU。",
                  "编码器只读过去；精修Head与全序列统计仍保留，整网不宣称严格因果。",
                  "困难序列仅用于诊断，排序使用全量全局计数。"]
        return {"geometry": GEOMETRY, "initialization_audit": read_json(root / "initialization_audit.json"),
                "difficult_sequences": diagnostics}, lines
