"""核验S1/S0的原始权重和训练协议；不从云端绝对路径读取文件。"""

from pathlib import Path

import torch

from traceformer.experiments.ablation import ComponentCoreSpec, LocalPoolingSpec, parse_ablation
from traceformer.experiments.component_suite import read_json
from traceformer.experiments.packaging import validate_baseline


def collect_references(root, config, identities):
    if root is None:
        return {}
    root = Path(root).resolve()
    result = {}
    for variant in ("S1", "S0"):
        folder = root / variant
        pointer = folder / "training.json"
        if not pointer.is_file():
            raise ValueError(f"S1 reference training pointer missing: {variant}")
        runs = sorted((folder / "training").glob("attempt_*/best.pt"))
        # 旧S0回传位于full的同级baseline，不追随已失效的云端绝对路径。
        source = folder if runs else root.parent / "baseline"
        checkpoint_path = runs[-1] if runs else source / "best.pt"
        if not checkpoint_path.is_file():
            raise ValueError(f"S1 reference checkpoint missing: {variant}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        allowed = [LocalPoolingSpec(variant).to_dict()]
        if variant == "S0":
            allowed.append(ComponentCoreSpec("M1_C4").to_dict())
        if checkpoint.get("ablation") not in allowed:
            raise ValueError(f"S1 reference variant contract differs: {variant}")
        validated = validate_baseline(source, config, identities,
            spec=parse_ablation(checkpoint["ablation"]), identity_file=root / "dataset_identity.json",
            pointer_file=pointer, canonical_variant=variant)
        _, run, identity = validated
        history, metrics = read_json(run / "history.json"), read_json(run / "metrics.json")
        selected = checkpoint["epoch"]
        if (history[selected - 1]["validation"]["competition"] != metrics["competition"]
                or max(item["validation"]["competition"]["score"] for item in history)
                != metrics["competition"]["score"]):
            raise ValueError(f"S1 reference saved best does not match training history: {variant}")
        for split in ("val", "test"):
            report = read_json(folder / split / "independent_report.json")
            if (report["checkpoint_sha256"] != identity["checkpoint_sha256"]
                    or report["threshold"] != metrics["threshold"]
                    or set(report["files"]) != set(identities[split])
                    or any(item["source_sha256"] != identities[split][name]["sha256"]
                           for name, item in report["files"].items())):
                raise ValueError(f"S1 reference independent report identity differs: {variant}/{split}")
        identity["original_validation"] = metrics["competition"]
        result[variant] = validated
    return result
