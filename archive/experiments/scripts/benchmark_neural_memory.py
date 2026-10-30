"""对锁定神经模型逐个计时，覆盖原始事件归一化到二值掩码。"""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import psutil
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
from run_neural_event_memory import DEFAULT_ROOT, source_dir, source_model

from traceformer.experiments.event_memory import SamePixelEventMemory
from traceformer.models.prefix_cone import prepare_prefix_geometry
from traceformer.postprocessing.contracts import sha256_file


@torch.inference_mode()
def raw_forward(source, readout, raw, threshold, duration_ms):
    """计时内部重新归一化和分块；适应模型直接替换最后一层。"""
    xy, timestamps, polarity = raw
    events = torch.from_numpy(np.column_stack((xy, timestamps / duration_ms, polarity))
                              .astype(np.float32))
    tokenized = source.tokenizer(events)
    geometry = prepare_prefix_geometry(tokenized.token_coords, tokenized.token_batch_indices)
    encoded = tokenized.tokens
    layers = source.encoder.layers if readout.tail is None else source.encoder.layers[:-1]
    for layer in layers:
        encoded = layer(encoded, tokenized.token_coords, tokenized.token_batch_indices,
                        geometry=geometry)
    item = {"coords": tokenized.token_coords, "indices": tokenized.event_token_indices,
            "locals": tokenized.event_local_features}
    if readout.tail is None:
        item["tokens"] = source.encoder.final_norm(encoded)
    else:
        item["penultimate"] = encoded
    if readout.variant != "B":
        item["memory"] = torch.from_numpy(SamePixelEventMemory()(xy, timestamps, polarity))
    return readout(item).sigmoid().numpy() >= threshold


def run(args):
    root = args.root.resolve()
    lock, lock_hash, selected = validate_lock(root)
    report = read(report_path(root, args.split))
    independent = read(report_path(root, args.split).with_name("independent_rescore.json"))
    if (report.get("status") != "complete" or report.get("lock_sha256") != lock_hash
            or independent.get("status") != "verified"
            or independent.get("evaluation_report_sha256") != sha256_file(report_path(root, args.split))):
        raise ValueError("runtime requires independently verified inference under the same lock")
    entries = {entry_key(entry): entry for entry in lock["entries"]}
    if not set(args.models) <= entries.keys():
        raise ValueError("every timed model must belong to the evaluation lock")
    destination = (args.output.resolve() if args.output is not None else
                   root / "evaluation" / args.split / "standalone_runtime.json")
    if not destination.is_relative_to(root):
        raise ValueError("runtime evidence must remain inside the experiment root")
    if destination.exists():
        raise FileExistsError(destination)
    source, config, _ = source_model()
    subset = dict(lock, entries=[entries[key] for key in args.models])
    models, inventory = load_models(source, root, subset, selected)
    raw_inputs = {}
    for path in sorted(source_dir(config, args.split).glob("*.npz")):
        checked_hash(path, report["sources"][path.name])
        with np.load(path, allow_pickle=False) as saved:
            raw = saved["ev"]
            # 从原文件仅取无标注的四个输入字段；解压和磁盘读取不计时。
            raw_inputs[path.name] = (np.column_stack((raw["x"], raw["y"])),
                                     raw["t"].astype(np.float64), raw["p"].copy())
    if set(raw_inputs) != set(report["sources"]) or len(raw_inputs) != 24:
        raise ValueError("standalone runtime must cover every partition sequence")
    largest = max(raw_inputs, key=lambda name: len(raw_inputs[name][1]))
    torch.set_num_threads(args.threads)
    process, groups = psutil.Process(), {}
    for key, model in models.items():
        threshold = entries[key]["threshold"]
        # 每个独立图先在最大序列热身；包含记忆提取的Numba首次编译。
        raw_forward(source, model, raw_inputs[largest], threshold, config.dataset.duration_ms)
        samples = {name: [] for name in raw_inputs}
        for repeat in range(args.repeats):
            for name, raw in raw_inputs.items():
                started = time.perf_counter()
                prediction = raw_forward(source, model, raw, threshold, config.dataset.duration_ms)
                elapsed = time.perf_counter() - started
                # 所有计时轮都校验最终掩码；检查与磁盘读取置于计时区间之外。
                mask_path = report_path(root, args.split).parent / key / "masks" / name
                checked_hash(mask_path, report["groups"][key]["files"][name]["mask_sha256"])
                with np.load(mask_path, allow_pickle=False) as saved:
                    if not np.array_equal(prediction, saved["prediction"].astype(bool)):
                        raise ValueError(f"standalone mask differs: {key}/{name}")
                samples[name].append(elapsed)
            print(f"standalone runtime {key} pass {repeat + 1}/{args.repeats}", flush=True)
        means = np.asarray([np.mean(values) for values in samples.values()])
        groups[key] = {"parameters": inventory[key], "seconds_by_sequence_and_pass": samples,
                       "mean_ms": float(means.mean() * 1000),
                       "median_ms": float(np.median(means) * 1000),
                       "p95_ms": float(np.quantile(means, .95) * 1000),
                       "maximum_ms": float(means.max() * 1000),
                       "partition_seconds_per_pass": float(means.sum()), "masks_exact": True}
    if sha256_file(root / "evaluation_lock.json") != lock_hash:
        raise ValueError("lock changed during runtime measurement")
    write(destination, {"status": "verified", "split": args.split, "lock_sha256": lock_hash,
                        "evaluation_report_sha256": sha256_file(report_path(root, args.split)),
                        "script_sha256": sha256_file(Path(__file__)),
                        "pid": os.getpid(), "device": "cpu", "processor": platform.processor(),
                        "torch_version": torch.__version__, "threads": args.threads,
                        "physical_memory_bytes": psutil.virtual_memory().total,
                        "process_peak_working_set_bytes": process.memory_info().peak_wset,
                        "repeats": args.repeats, "warmup_sequence": largest,
                        "scope": "each complete network separately; in-memory raw x/y/t/p through "
                                 "normalization, tokenizer, encoder, event memory if applicable, "
                                 "readout and mask; offline bidirectional; disk/scoring/export excluded",
                        "groups": groups})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, help="单独保存补充结构计时，不覆盖既有证据")
    parser.add_argument("--models", nargs="+", default=["frozen_seed37_B", "frozen_seed37_C",
                                                        "adaptive_seed37_B", "adaptive_seed37_C"])
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1 or len(set(args.models)) != len(args.models):
        parser.error("positive threads/repeats and distinct models are required")
    run(args)


if __name__ == "__main__":
    main()
