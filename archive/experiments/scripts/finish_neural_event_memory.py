"""在评估锁下，从原始事件验证纯神经模型并另进程独立复评。"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from run_neural_event_memory import DEFAULT_ROOT, SOURCE, Readout, source_dir, source_model
from sklearn.metrics import average_precision_score

from traceformer.data import NPZEventDataset
from traceformer.experiments.event_memory import (
    SamePixelEventMemory,
    shuffle_memory_within_tokens,
)
from traceformer.models.prefix_cone import prepare_prefix_geometry
from traceformer.postprocessing.contracts import sha256_file
from traceformer.postprocessing.metrics import BinaryCounts, score_binary
from traceformer.postprocessing.submission import evaluate_predictions, write_prediction_npz

INPUT_FIELDS = ("tokens", "penultimate", "coords", "indices", "locals", "memory")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def checked_hash(path, expected):
    actual = sha256_file(path)
    if actual.lower() != str(expected).lower():
        raise ValueError(f"identity mismatch: {path}")
    return actual


def rooted(root, relative):
    """锁内文件限定在实验目录，避免误用其他实验的同名结果。"""
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("locked artifact paths must be relative to the experiment root")
    return path


def entry_key(entry):
    return f"{entry['phase']}_seed{entry['seed']}_{entry['variant']}"


def check_metrics(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("metric fields differ")
    for name, value in actual.items():
        reference = expected[name]
        if isinstance(reference, int):
            equal = value == reference
        else:
            equal = math.isclose(float(value), float(reference), rel_tol=0, abs_tol=1e-12)
        if not equal:
            raise ValueError(f"metric mismatch for {name}: {value} != {reference}")


def validate_lock(root):
    """先验证整个矩阵，再读取任何验证或测试输入。"""
    path = root / "evaluation_lock.json"
    if not path.is_file():
        raise FileNotFoundError("evaluation_lock.json is required before evaluation")
    lock = read(path)
    if lock.get("schema_version") != 1 or not lock.get("entries"):
        raise ValueError("evaluation lock requires schema_version=1 and nonempty entries")
    checked_hash(SOURCE / "best.pt", lock["source_checkpoint_sha256"])
    checked_hash(SOURCE / "config.yaml", lock["source_config_sha256"])
    protocols, selections, keys = {}, {}, set()
    for entry in lock["entries"]:
        phase, variant, seed = entry["phase"], entry["variant"], entry["seed"]
        if phase not in {"frozen", "adaptive"} or variant not in {"B", "C", "U", "G", "S"}:
            raise ValueError("unsupported locked neural model")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("locked seed must be a nonnegative integer")
        threshold = float(entry["threshold"])
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("locked threshold must be finite and strictly between zero and one")
        key = entry_key(entry)
        if key in keys:
            raise ValueError("duplicate model in evaluation lock")
        keys.add(key)
        if phase not in protocols:
            protocol_path = root / phase / "protocol.json"
            checked_hash(protocol_path, lock["protocol_sha256"][phase])
            protocols[phase] = read(protocol_path)
        protocol = protocols[phase]
        if (protocol["source_checkpoint_sha256"].lower()
                != lock["source_checkpoint_sha256"].lower()):
            raise ValueError("phase protocol uses another pretrained source")
        for split, expected_count in (("val", 24), ("train", 99)):
            manifest_path = root / "cache" / split / "manifest.json"
            checked_hash(manifest_path, protocol[f"{split}_cache_manifest_sha256"])
            manifest = read(manifest_path)
            if (manifest.get("status") != "complete"
                    or len(manifest.get("files", {})) != expected_count
                    or manifest["checkpoint_sha256"].lower()
                    != lock["source_checkpoint_sha256"].lower()
                    or manifest["config_sha256"].lower() != lock["source_config_sha256"].lower()
                    or manifest.get("annotations_used_in_features") is not False
                    or manifest.get("dtype") != "float32; raw timestamps float64"):
                raise ValueError("locked training cache does not match the pure FP32 source")
        checkpoint_path = rooted(root, entry["checkpoint"])
        checked_hash(checkpoint_path, entry["sha256"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (checkpoint["variant"] != variant or checkpoint["seed"] != seed
                or checkpoint["adaptive"] != (phase == "adaptive")
                or checkpoint["protocol"] != protocol):
            raise ValueError("locked checkpoint metadata differs from its phase protocol")
        result_path = rooted(root, entry["validation_result"])
        checked_hash(result_path, entry["validation_result_sha256"])
        selected = read(result_path)[variant]
        if (selected["checkpoint_sha256"].lower() != entry["sha256"].lower()
                or float(selected["threshold"]) != threshold
                or selected["epoch"] != checkpoint["epoch"]
                or selected["validation_ap"] != checkpoint["validation_ap"]):
            raise ValueError("locked checkpoint or threshold differs from validation selection")
        selections[key] = selected
    return lock, sha256_file(path), selections


@torch.inference_mode()
def raw_inputs(source, sequence, *, need_memory):
    """标注与身份清零后，仅从原始坐标、时间、极性构造神经输入。"""
    blind = replace(sequence, labels=torch.zeros_like(sequence.labels),
                    object_ids=torch.zeros_like(sequence.object_ids))
    if bool(blind.labels.any()) or bool(blind.object_ids.any()):
        raise ValueError("annotation clearing failed")
    tokenized = source.tokenizer(blind.events)
    geometry = prepare_prefix_geometry(tokenized.token_coords, tokenized.token_batch_indices)
    encoded = tokenized.tokens
    for layer in source.encoder.layers[:-1]:
        encoded = layer(encoded, tokenized.token_coords, tokenized.token_batch_indices,
                        geometry=geometry)
    penultimate = encoded
    encoded = source.encoder.layers[-1](encoded, tokenized.token_coords,
                                       tokenized.token_batch_indices, geometry=geometry)
    item = {"tokens": source.encoder.final_norm(encoded), "penultimate": penultimate,
            "coords": tokenized.token_coords, "indices": tokenized.event_token_indices,
            "locals": tokenized.event_local_features}
    if need_memory:
        features = SamePixelEventMemory()(blind.events[:, :2].numpy(),
                                         blind.timestamps_ms.numpy(), blind.events[:, 3].numpy())
        item["memory"] = torch.from_numpy(features)
    return item


def with_shuffled_memory(item, name):
    output = dict(item)
    seed = 20261024 + int(name.split("_")[-1])
    output["shuffled_memory"] = shuffle_memory_within_tokens(
        item["memory"], item["indices"], generator=torch.Generator().manual_seed(seed))
    return output


def cached_inputs(root, name, source_hash, manifest, *, need_memory):
    identity = manifest["files"][name]
    if identity["source_sha256"].lower() != source_hash.lower():
        raise ValueError("validation raw source differs from the training cache source")
    path = root / "cache/val" / f"{name}.pt"
    checked_hash(path, identity["cache_sha256"])
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    # 仅取前向特征白名单；旧输出、标签、对象身份不进入缓存适配器。
    return {key: saved[key] for key in INPUT_FIELDS if key != "memory" or need_memory}


def report_path(root, split):
    return root / "evaluation" / split / "report.json"


def load_models(source, root, lock, selections):
    trunk = sum(p.numel() for p in source.tokenizer.parameters()) + sum(
        p.numel() for p in source.encoder.parameters())
    if trunk != 157187:
        raise ValueError("unexpected shared backbone parameter count")
    models, inventory = {}, {}
    for entry in lock["entries"]:
        key = entry_key(entry)
        model = Readout(entry["variant"], source.encoder, adaptive=entry["phase"] == "adaptive")
        checkpoint = torch.load(rooted(root, entry["checkpoint"]),
                                map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        head = sum(p.numel() for p in model.decoder.parameters())
        trainable = sum(p.numel() for p in model.parameters())
        if (trainable != selections[key]["readout_trainable_parameters"]
                or trunk + head != selections[key]["neural_parameters"]):
            raise ValueError("saved validation parameter count differs from actual neural graph")
        # 适应末层替换原末层；完整网络只统计一份共享主干。
        inventory[key] = {"shared_trunk_parameters": trunk, "head_parameters": head,
                          "total_neural_parameters": trunk + head,
                          "optimized_parameters": trainable,
                          "encoder_scope": "last block and final norm adapted" if model.tail
                          is not None else "frozen pretrained trunk"}
        models[key] = model
    return models, inventory


def check_validation_gate(root, lock_hash):
    path = report_path(root, "val")
    independent_path = path.with_name("independent_rescore.json")
    if not path.is_file() or not independent_path.is_file():
        raise ValueError("test requires completed raw/cached validation and independent rescore")
    report, independent = read(path), read(independent_path)
    if (report.get("status") != "complete" or report.get("lock_sha256") != lock_hash
            or not report.get("raw_cached_equivalence")
            or independent.get("status") != "verified"
            or independent.get("lock_sha256") != lock_hash
            or independent.get("evaluation_report_sha256") != sha256_file(path)):
        raise ValueError("validation acceptance does not belong to the current complete lock")


@torch.no_grad()
def evaluate(root, split, *, atol):
    # 模型参数按训练时的普通Tensor构造；仅实际特征提取使用inference_mode。
    lock, lock_hash, selections = validate_lock(root)
    if split == "test":
        check_validation_gate(root, lock_hash)
    destination = report_path(root, split)
    if destination.exists():
        raise FileExistsError(f"evaluation already exists; preserve it: {destination}")
    source, config, _ = source_model()
    models, inventory = load_models(source, root, lock, selections)
    dataset = NPZEventDataset(source_dir(config, split))
    if len(dataset) != 24:
        raise ValueError("evaluation requires every sequence in the 24-file partition")
    manifest = read(root / "cache/val/manifest.json") if split == "val" else None
    need_memory = any(entry["variant"] != "B" for entry in lock["entries"])
    need_shuffle = any(entry["variant"] == "S" for entry in lock["entries"])
    started, shared_seconds = time.perf_counter(), 0.0
    labels_all, sources = [], {}
    aggregates = {key: BinaryCounts() for key in models}
    groups = {entry_key(entry): {"phase": entry["phase"], "seed": entry["seed"],
              "variant": entry["variant"], "threshold": entry["threshold"],
              "checkpoint_sha256": entry["sha256"], "files": {},
              "parameters": inventory[entry_key(entry)]} for entry in lock["entries"]}
    all_logits = {key: [] for key in models}
    for sequence in dataset:
        name = sequence.source_path.stem
        source_hash = sha256_file(sequence.source_path)
        sources[sequence.source_path.name] = source_hash
        encode_started = time.perf_counter()
        item = raw_inputs(source, sequence, need_memory=need_memory)
        if need_shuffle:
            item = with_shuffled_memory(item, name)
        shared_seconds += time.perf_counter() - encode_started
        cache = None
        if split == "val":
            cache = cached_inputs(root, name, source_hash, manifest, need_memory=need_memory)
            if need_shuffle:
                cache = with_shuffled_memory(cache, name)
        labels = sequence.labels.numpy().astype(bool)
        labels_all.append(labels)
        locations = np.column_stack((sequence.events[:, :2].numpy(),
                                     sequence.timestamps_ms.numpy())).astype(np.int64)
        for key, model in models.items():
            head_started = time.perf_counter()
            logits = model(item).detach().cpu()
            forward_seconds = time.perf_counter() - head_started
            if logits.shape != sequence.labels.shape or not bool(torch.isfinite(logits).all()):
                raise ValueError("neural logits are nonfinite or do not preserve original event rows")
            threshold = float(groups[key]["threshold"])
            prediction = logits.sigmoid().numpy() >= threshold
            agreement = None
            if cache is not None:
                cached_logits = model(cache).detach().cpu()
                exact = torch.equal(logits, cached_logits)
                maximum = float((logits - cached_logits).abs().max())
                mask_exact = np.array_equal(prediction, cached_logits.sigmoid().numpy() >= threshold)
                if not torch.allclose(logits, cached_logits, rtol=0, atol=atol) or not mask_exact:
                    raise ValueError(f"raw/cache disagreement for {key}/{name}: {maximum}")
                agreement = {"bit_exact": exact, "maximum_absolute_logit_error": maximum,
                             "mask_exact": mask_exact, "absolute_tolerance": atol}
            group_dir = destination.parent / key
            logit_path = group_dir / "logits" / f"{name}.npz"
            logit_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(logit_path, logits=logits.numpy(), source_sha256=source_hash,
                                checkpoint_sha256=groups[key]["checkpoint_sha256"])
            mask_path = write_prediction_npz(group_dir / "masks" / f"{name}.npz", prediction)
            counts = score_binary(locations, labels, sequence.object_ids.numpy(), prediction)
            aggregates[key] += counts
            groups[key]["files"][sequence.source_path.name] = {
                "metrics": counts.metrics(), "source_sha256": source_hash,
                "logits_sha256": sha256_file(logit_path), "mask_sha256": sha256_file(mask_path),
                "raw_cached_agreement": agreement, "head_or_tail_seconds": forward_seconds}
            all_logits[key].append(logits.numpy())
        print(f"raw neural {split} {name}: {len(models)} locked models", flush=True)
    labels_array = np.concatenate(labels_all)
    for key, group in groups.items():
        group["aggregate"] = aggregates[key].metrics()
        group["foreground_ap"] = float(average_precision_score(labels_array,
                                                               np.concatenate(all_logits[key])))
        if split == "val":
            selected = selections[key]
            check_metrics(group["aggregate"], selected["aggregate"])
            expected_files = {Path(name).stem: row for name, row in selected["files"].items()}
            if set(expected_files) != {Path(name).stem for name in group["files"]}:
                raise ValueError("validation selection does not cover the full raw partition")
            for name, row in group["files"].items():
                check_metrics(row["metrics"], expected_files[Path(name).stem])
            if not math.isclose(group["foreground_ap"], selected["validation_ap"],
                                rel_tol=0, abs_tol=1e-8):
                raise ValueError("raw validation AP differs from checkpoint selection")
    if sha256_file(root / "evaluation_lock.json") != lock_hash:
        raise ValueError("evaluation lock changed during inference")
    report = {"status": "complete", "split": split, "lock_sha256": lock_hash,
              "source_checkpoint_sha256": lock["source_checkpoint_sha256"],
              "source_config_sha256": lock["source_config_sha256"],
              "evaluation_script_sha256": sha256_file(Path(__file__)), "inference_pid": os.getpid(),
              "annotations_cleared_in_forward": True, "teacher_outputs_used_as_inputs": False,
              "postprocessing": False, "new_test_threshold_selection": False,
              "scope": "head seeds share one pretrained trunk; adaptive replaces its last block; "
                       "previously inspected benchmark, exploratory generalization limits",
              "raw_cached_equivalence": split == "val", "sources": sources,
              "shared_encoder_and_memory_seconds": shared_seconds,
              "timing_scope": "shared feature extraction once per sequence and per-entry heads/tails; "
                              "not independently measured single-model latency",
              "elapsed_seconds": time.perf_counter() - started, "groups": groups}
    write(destination, report)
    print(json.dumps({"status": "complete", "split": split, "report": str(destination)}), flush=True)
    return report


def rescore(root, split):
    lock, lock_hash, _ = validate_lock(root)
    path = report_path(root, split)
    report = read(path)
    if (report.get("status") != "complete" or report["lock_sha256"] != lock_hash
            or report["inference_pid"] == os.getpid()):
        raise ValueError("independent rescore requires a complete locked report from another process")
    if set(report["groups"]) != {entry_key(entry) for entry in lock["entries"]}:
        raise ValueError("saved evaluation matrix differs from the lock")
    source, config, _ = source_model()
    del source
    dataset_root = source_dir(config, split)
    actual_sources = {p.name: sha256_file(p) for p in sorted(dataset_root.glob("*.npz"))}
    if actual_sources != report["sources"] or len(actual_sources) != 24:
        raise ValueError("raw partition changed after inference")
    started, verified = time.perf_counter(), {}
    for entry in lock["entries"]:
        key = entry_key(entry)
        group = report["groups"][key]
        directory = path.parent / key
        for name, row in group["files"].items():
            checked_hash(directory / "masks" / name, row["mask_sha256"])
            logit_path = directory / "logits" / name
            checked_hash(logit_path, row["logits_sha256"])
            with np.load(logit_path, allow_pickle=False) as saved:
                if (str(saved["source_sha256"].item()) != actual_sources[name]
                        or str(saved["checkpoint_sha256"].item()) != group["checkpoint_sha256"]):
                    raise ValueError("saved logits refer to another source or checkpoint")
                prediction = torch.from_numpy(saved["logits"]).sigmoid().numpy() >= group["threshold"]
            with np.load(directory / "masks" / name, allow_pickle=False) as saved:
                if not np.array_equal(prediction, saved["prediction"].astype(bool)):
                    raise ValueError("saved mask differs from its frozen-threshold neural logits")
        independent = evaluate_predictions(directory / "masks", dataset_root)
        check_metrics(independent["aggregate"], group["aggregate"])
        for name, metrics in independent["files"].items():
            check_metrics(metrics, group["files"][name]["metrics"])
        verified[key] = independent
    result = {"status": "verified", "split": split, "lock_sha256": lock_hash,
              "evaluation_report_sha256": sha256_file(path), "independent_pid": os.getpid(),
              "inference_pid": report["inference_pid"], "groups": verified,
              "elapsed_seconds": time.perf_counter() - started}
    write(path.with_name("independent_rescore.json"), result)
    print(json.dumps({"status": "verified", "split": split, "models": len(verified)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    if args.threads < 1 or not math.isfinite(args.atol) or not 0 <= args.atol <= 1e-6:
        parser.error("threads must be positive and atol must be between zero and 1e-6")
    torch.set_num_threads(args.threads)
    root = args.root.resolve()
    if args.rescore:
        rescore(root, args.split)
    else:
        evaluate(root, args.split, atol=args.atol)


if __name__ == "__main__":
    main()
