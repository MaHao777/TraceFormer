"""冻结权重与阈值，仅在验证集检查事件记忆对应关系的敏感性。"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
from finish_neural_event_memory import (
    checked_hash,
    entry_key,
    load_models,
    read,
    report_path,
    validate_lock,
    write,
)
from run_neural_event_memory import DEFAULT_ROOT, source_model
from sklearn.metrics import average_precision_score

from traceformer.experiments.event_memory import (
    mean_memory_within_tokens,
    shuffle_memory_within_tokens,
)
from traceformer.postprocessing.contracts import sha256_file
from traceformer.postprocessing.metrics import BinaryCounts, score_binary


@torch.no_grad()
def run(root, variant="C"):
    # 普通参数构造与训练/缓存保持一致，避免inference参数切换CPU数值路径。
    lock, lock_hash, selections = validate_lock(root)
    reference = read(report_path(root, "val"))
    if reference.get("status") != "complete" or reference.get("lock_sha256") != lock_hash:
        raise ValueError("alignment diagnostics require the completed locked validation report")
    output = root / "diagnostics" / ("memory_alignment.json" if variant == "C"
                                     else "gated_memory_alignment.json")
    if output.exists():
        raise FileExistsError(output)
    source, _, _ = source_model()
    entries = [entry for entry in lock["entries"] if entry["variant"] == variant]
    models, _ = load_models(source, root, dict(lock, entries=entries), selections)
    controls = ("aligned", "token_mean", "within_token_shuffle", "zero")
    groups = {f"{entry_key(entry)}/{control}": {"phase": entry["phase"], "seed": entry["seed"],
              "control": control, "threshold": entry["threshold"], "files": {}}
              for entry in entries for control in controls}
    logits_all = {key: [] for key in groups}
    totals = {key: BinaryCounts() for key in groups}
    labels_all = []
    manifest = read(root / "cache/val/manifest.json")
    for name, identity in manifest["files"].items():
        path = root / "cache/val" / f"{name}.pt"
        checked_hash(path, identity["cache_sha256"])
        cached = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        # 仅白名单神经输入进入前向，标注只交给计分函数。
        item = {key: cached[key] for key in ("tokens", "penultimate", "coords", "indices", "locals", "memory")}
        labels = cached["labels"].numpy().astype(bool)
        labels_all.append(labels)
        locations = np.column_stack((cached["xy"].numpy(), cached["timestamps_ms"].numpy())).astype(np.int64)
        memories = {"aligned": item["memory"],
                    "token_mean": mean_memory_within_tokens(item["memory"], item["indices"]),
                    "within_token_shuffle": shuffle_memory_within_tokens(
                        item["memory"], item["indices"], generator=torch.Generator().manual_seed(
                            20261024 + int(name.split("_")[-1]))),
                    "zero": torch.zeros_like(item["memory"])}
        for key, model in models.items():
            for control, memory in memories.items():
                group_key = f"{key}/{control}"
                logits = model(dict(item, memory=memory)).cpu()
                prediction = logits.sigmoid().numpy() >= groups[group_key]["threshold"]
                counts = score_binary(locations, labels, cached["object_ids"].numpy(), prediction)
                totals[group_key] += counts
                groups[group_key]["files"][name] = counts.metrics()
                logits_all[group_key].append(logits.numpy())
        print(f"memory alignment validation {name}", flush=True)
    labels = np.concatenate(labels_all)
    for key, group in groups.items():
        group["aggregate"] = totals[key].metrics()
        group["foreground_ap"] = float(average_precision_score(labels, np.concatenate(logits_all[key])))
        if group["control"] == "aligned":
            original = reference["groups"][key.split("/")[0]]
            if (group["aggregate"] != original["aggregate"]
                    or not np.isclose(group["foreground_ap"], original["foreground_ap"], rtol=0, atol=1e-12)):
                raise ValueError("unaltered alignment diagnostics do not reproduce raw validation")
    write(output, {"status": "complete", "lock_sha256": lock_hash, "variant": variant,
                   "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                   "post_test_validation_followup": variant == "G",
                   "validation_report_sha256": sha256_file(report_path(root, "val")),
                   "script_sha256": sha256_file(Path(__file__)), "split": "val", "groups": groups,
                   "scope": f"frozen trained {variant} weights and validation-selected thresholds; "
                            "no retraining or threshold tuning; interventions may cause input "
                            "distribution shift; sensitivity evidence, not trained ablation or "
                            "a separate generalization experiment",
                   "shuffle": "joint 12-channel rows, only within each token; deterministic "
                              "seed 20261024 + sequence index, shared across head seeds"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--variant", choices=("C", "G"), default="C")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    run(args.root.resolve(), args.variant)


if __name__ == "__main__":
    main()
