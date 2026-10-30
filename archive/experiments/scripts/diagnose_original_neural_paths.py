"""冻结原始主模型，筛查已有神经信息路径；不训练、不访问测试集。"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from traceformer.config import load_config
from traceformer.models import prefix_cone
from traceformer.models.cone_transformer import ConeAttentionOutput
from traceformer.postprocessing.contracts import sha256_file
from traceformer.postprocessing.metrics import BinaryCounts, score_binary
from traceformer.training.engine import (
    _dataset,
    _dataset_manifest,
    _manifest_identity_digest,
    load_inference_model,
)

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "outputs/main_model"
VARIANTS = (
    "full", "no_refinement",
    "no_messages", "no_message_0", "no_message_1", "no_message_2",
    "scale_0", "scale_1", "scale_2", "past_only", "future_only",
    "uniform_summary_weights", "shuffled_summaries", "token_mean_event_locals",
)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@contextmanager
def intervene(model, name, seed):
    """仅改变前向张量；退出时恢复全部 hook 与函数，不改网络参数。"""
    with ExitStack() as stack:
        if name.startswith("no_message"):
            indices = range(len(model.encoder.layers)) if name == "no_messages" else [int(name[-1])]
            for index in indices:
                hook = model.encoder.layers[index].attention.register_forward_hook(
                    lambda module, args, result: ConeAttentionOutput(torch.zeros_like(result.features)))
                stack.callback(hook.remove)
        if name == "token_mean_event_locals":
            def mean_locals(module, args, result):
                # Token保持原值，仅删除同一Token内事件局部特征的差别。
                inverse = result.event_token_indices
                local = result.event_local_features
                total = local.new_zeros((len(result.tokens), local.shape[1]))
                total.index_add_(0, inverse, local)
                counts = torch.bincount(inverse, minlength=len(total))
                return replace(result, event_local_features=(total / counts[:, None])[inverse])

            hook = model.tokenizer.register_forward_hook(mean_locals)
            stack.callback(hook.remove)
        if (name.startswith("scale_") or name in {
                "past_only", "future_only", "uniform_summary_weights", "shuffled_summaries"}):
            original = prefix_cone.prefix_cone_pool
            generator = torch.Generator().manual_seed(seed)

            def pool(values, weights, geometry, windows, *args, **kwargs):
                if name == "uniform_summary_weights":
                    weights = torch.ones_like(weights)
                summaries, counts = original(values, weights, geometry, windows, *args, **kwargs)
                if name == "shuffled_summaries":
                    # 一个样本内联合置换六摘要与成员数，保持槽间关系和边缘分布。
                    summaries, counts = summaries.clone(), counts.clone()
                    for sample in geometry:
                        indices = sample.indices
                        permutation = indices[torch.randperm(len(indices), generator=generator)]
                        summaries[indices] = summaries[permutation].clone()
                        counts[indices] = counts[permutation].clone()
                else:
                    keep = torch.ones(counts.shape[1], dtype=torch.bool, device=counts.device)
                    if name.startswith("scale_"):
                        keep[:] = False
                        index = int(name[-1])
                        keep[2 * index:2 * index + 2] = True
                    elif name == "past_only":
                        keep[1::2] = False
                    elif name == "future_only":
                        keep[::2] = False
                    counts = counts.masked_fill(~keep[None], 0)
                return summaries, counts

            stack.enter_context(patch.object(prefix_cone, "prefix_cone_pool", pool))
        yield


@torch.inference_mode()
def predict(model, sequence, config, variant, *, intervention=intervene):
    # 前向仅读取x/y/t/p；真实标签和目标身份由调用方隔离在评分路径。
    with intervention(model, variant, 20261024 + int(sequence.source_path.stem.split("_")[-1])):
        return model(sequence.events)


def run(output, *, threads=8, max_seconds=900, variants=VARIANTS, intervention=intervene,
        purpose="original-neural-path screening; NOT retrained ablation"):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(SOURCE / "config.yaml")
    torch.set_num_threads(threads)
    # 必须在正常上下文构造参数，避免inference tensor改变CPU数值路径。
    model, checkpoint = load_inference_model(config, SOURCE / "best.pt", device=torch.device("cpu"))
    dataset = _dataset(config, "val")
    manifest = _dataset_manifest(dataset)
    if _manifest_identity_digest(manifest) != checkpoint["validation_manifest_sha256"]:
        raise ValueError("original checkpoint validation identity differs")
    if checkpoint["ablation"] is not None or checkpoint["training_stage"] != "full":
        raise ValueError("requires the original jointly trained production neural checkpoint")
    original_state = {key: value.clone() for key, value in model.state_dict().items()}
    protocol = {"purpose": purpose,
                "checkpoint_sha256": sha256_file(SOURCE / "best.pt"),
                "config_sha256": sha256_file(SOURCE / "config.yaml"),
                "script_sha256": sha256_file(__file__), "variants": variants,
                "threshold": checkpoint["decision_threshold"], "split": "val", "files": len(dataset),
                "dtype": "float32", "threads": threads, "max_seconds": max_seconds,
                "test_accessed": False, "training": False, "threshold_selection": False,
                "forward_annotations_cleared": True, "shuffle_seed": 20261024,
                "parameters": {name: sum(p.numel() for p in child.parameters())
                               for name, child in model.named_children()},
                "limitations": ["interventions may shift the learned feature distribution",
                                "fixed-threshold score confounds calibration and discrimination",
                                "single trained seed; full validation set; no causal novelty claim"]}
    write(output / "protocol.json", protocol)
    write(output / "dataset_manifest.json", manifest)
    (output / "logits").mkdir()
    probabilities = {name: [] for name in variants}
    labels, files, totals, groups = [], {}, {}, {}
    inference_seconds = dict.fromkeys(variants, 0.0)
    for name in variants:
        totals[name] = BinaryCounts()
        groups[name] = {"ordinary": BinaryCounts(), "dense": BinaryCounts()}
    threshold = checkpoint["decision_threshold"]
    for index in range(len(dataset)):
        sequence = dataset[index]
        blind = replace(sequence, labels=torch.zeros_like(sequence.labels),
                        object_ids=torch.zeros_like(sequence.object_ids))
        with torch.inference_mode():
            began = time.perf_counter()
            original = model(blind.events)
            inference_seconds["full"] += time.perf_counter() - began
        if index == 0:
            first = predict(model, blind, config, "full", intervention=intervention)
            if not torch.equal(first["event_logits"], original["event_logits"]):
                raise ValueError("prepared sparse path does not exactly match production forward")
            for name in variants[2:]:
                altered = predict(model, blind, config, name, intervention=intervention)["event_logits"]
                if altered.shape != original["event_logits"].shape or not torch.isfinite(altered).all():
                    raise ValueError(f"invalid intervention output: {name}")
            restored = predict(model, blind, config, "full", intervention=intervention)
            if not torch.equal(restored["event_logits"], original["event_logits"]):
                raise ValueError("temporary intervention was not exactly restored")
            write(output / "preflight.json", {"source": sequence.source_path.name,
                  "production_forward_bit_exact": True, "restoration_bit_exact": True,
                  "all_interventions_finite": True, "seconds": time.perf_counter() - started})
            print("preflight passed; production and restored logits bit exact", flush=True)
        values = {"full": original["event_logits"],
                  "no_refinement": original["base_event_logits"]}
        for name in variants[2:]:
            began = time.perf_counter()
            values[name] = predict(model, blind, config, name, intervention=intervention)["event_logits"]
            inference_seconds[name] += time.perf_counter() - began
        raw_logits = {name: value.detach().cpu().numpy() for name, value in values.items()}
        np.savez_compressed(output / "logits" / sequence.source_path.name, **raw_logits)
        with np.load(sequence.source_path, allow_pickle=False) as archive:
            raw = archive["ev"]
        locations = np.column_stack((raw["x"], raw["y"], raw["t"])).astype(np.int64)
        target = sequence.labels.numpy().astype(bool)
        identities = sequence.object_ids.numpy()
        labels.append(target)
        current = {"events": len(target), "variants": {}}
        for name in variants:
            probability = torch.sigmoid(torch.from_numpy(raw_logits[name])).numpy()
            probabilities[name].append(probability)
            count = score_binary(locations, target, identities, probability >= threshold,
                                 frame_ms=config.evaluation.frame_ms,
                                 object_correct_fraction=config.evaluation.object_correct_fraction)
            totals[name] += count
            groups[name]["ordinary"] += count
            current["variants"][name] = count.metrics()
        files[sequence.source_path.name] = current
        elapsed = time.perf_counter() - started
        write(output / "status.json", {"status": "running", "files_done": index + 1,
                                        "files_total": len(dataset), "seconds": elapsed})
        print(f"{index + 1}/{len(dataset)} {sequence.source_path.name} "
              f"elapsed={elapsed:.1f}s", flush=True)
        if elapsed > max_seconds and index + 1 < len(dataset):
            raise TimeoutError("screening budget exhausted; completed sequence outputs are preserved")
    target = np.concatenate(labels)
    results = {}
    for name in variants:
        results[name] = {**totals[name].metrics(),
                         "ap": float(average_precision_score(target, np.concatenate(probabilities[name]))),
                         "groups": {key: value.metrics() for key, value in groups[name].items()},
                         "inference_seconds": inference_seconds[name]}
    reference = json.loads((SOURCE / "validation.json").read_text(encoding="utf-8"))["objectness"]["aggregate"]
    for name in asdict(BinaryCounts()):
        if results["full"][name] != reference[name]:
            raise ValueError(f"original validation count failed reproduction: {name}")
    unchanged = all(torch.equal(value, original_state[key]) for key, value in model.state_dict().items())
    if not unchanged:
        raise ValueError("diagnostic unexpectedly mutated original parameters")
    report = {"status": "complete", "protocol": protocol, "results": results, "files": files,
              "original_global_counts_reproduced": True, "weights_unchanged": unchanged,
              "elapsed_seconds": time.perf_counter() - started}
    write(output / "report.json", report)
    write(output / "status.json", {"status": "complete", "seconds": report["elapsed_seconds"]})
    for name, value in results.items():
        print(f"{name:28} Score={value['score']:.6f} AP={value['ap']:.6f} IoU={value['iou']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=900)
    args = parser.parse_args()
    run(args.output.resolve(), threads=args.threads, max_seconds=args.max_seconds)
