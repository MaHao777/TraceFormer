"""诊断三锥池化的模块、算子与分块开销；仅在当前进程临时替换分块。"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from traceformer.config import load_config
from traceformer.data.dataset import NPZEventDataset
from traceformer.models import prefix_cone
from traceformer.postprocessing.cache import infer_event_output
from traceformer.postprocessing.contracts import sha256_file
from traceformer.postprocessing.submission import load_prediction
from traceformer.training.engine import load_inference_model


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class Probe:
    """模块同步计时有嵌套；算子追踪模式只添加范围标记。"""

    def __init__(self, synchronized=True):
        self.synchronized = synchronized
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.geometry = []

    def wrap(self, name, function):
        def measured(*args, **kwargs):
            if self.synchronized:
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.profiler.record_function(name):
                result = function(*args, **kwargs)
            if self.synchronized:
                torch.cuda.synchronize()
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1
            if name == "encoder_geometry":
                self.geometry.extend({"tokens": len(sample.indices), "grid": sample.shape} for sample in result)
            return result
        return measured

    def install(self, stack, model):
        for name in ("tokenizer", "encoder", "decoder", "sparse_mixture", "event_refinement_head"):
            module = getattr(model, name)
            stack.enter_context(patch.object(module, "forward", self.wrap(name, module.forward)))
        for index, block in enumerate(model.encoder.layers):
            for name, module in ((f"layer_{index}_attention", block.attention),
                                 ("center_attention", block.attention.center_attention),
                                 ("feed_forward", block.feed_forward)):
                stack.enter_context(patch.object(module, "forward", self.wrap(name, module.forward)))
        stack.enter_context(patch.object(prefix_cone, "prepare_prefix_geometry",
                            self.wrap("encoder_geometry", prefix_cone.prepare_prefix_geometry)))
        stack.enter_context(patch.object(prefix_cone, "prefix_cone_pool",
                            self.wrap("prefix_pool", prefix_cone.prefix_cone_pool)))
        stack.enter_context(patch.object(prefix_cone._ConeSums, "forward", staticmethod(
                            self.wrap("prefix_sums", prefix_cone._ConeSums.forward))))


def arrays(output):
    return {name: value.detach().cpu().numpy() for name, value in output.items() if isinstance(value, torch.Tensor)}


def compare(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("network tensor set differs")
    result = {}
    for name, left in actual.items():
        right = expected[name]
        if left.shape != right.shape:
            raise ValueError(f"tensor shape differs: {name}")
        if not np.isfinite(left).all():
            raise ValueError(f"nonfinite tensor: {name}")
        count = int(np.count_nonzero(left != right))
        if count:
            result[name] = {"different": count, "max_abs": float(np.max(np.abs(left.astype(float) - right)))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, nargs="+", default=[1024, 4096, 16384])
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(14)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_config(args.run_dir / "artifacts/config.yaml")
    model, checkpoint = load_inference_model(config, args.run_dir / "artifacts/best.pt", device=torch.device("cuda"))
    dataset = NPZEventDataset(args.dataset / "test", width=config.resolution.width,
                              height=config.resolution.height, duration_ms=config.dataset.duration_ms)
    sequences = [dataset[index] for index in range(len(dataset))]
    if len(sequences) != 24 or args.chunks[0] != 1024 or any(chunk <= 0 for chunk in args.chunks):
        raise ValueError("requires 24 test files and baseline chunk 1024")
    maximum = max(sequences, key=lambda sequence: sequence.length)
    reference = args.run_dir / "results/test_network_predictions"
    threshold = checkpoint["decision_threshold"]
    baseline = {}

    def infer(sequence):
        output = infer_event_output(model, config, sequence)
        mask = (output["event_logits"].sigmoid() >= threshold).cpu().numpy()
        if not np.array_equal(mask, load_prediction(reference / sequence.source_path.name)):
            raise ValueError(f"fixed prediction differs: {sequence.source_path.name}")
        return output

    with torch.inference_mode():
        infer(maximum)
        components = {}
        for sequence in sequences:
            probe = Probe()
            with ExitStack() as stack:
                probe.install(stack, model)
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = infer(sequence)
                torch.cuda.synchronize()
                total = time.perf_counter() - start
            baseline[sequence.source_path.name] = arrays(output)
            del output
            components[sequence.source_path.name] = {"seconds": dict(probe.seconds), "calls": dict(probe.calls),
                "geometry": probe.geometry, "instrumented_seconds": total, "events": sequence.length}
            print("components", sequence.source_path.name, total, flush=True)
        keys = set().union(*(row["seconds"] for row in components.values()))
        save(args.output / "components.json", {"files": components,
             "mean_ms": {key: float(np.mean([row["seconds"].get(key, 0) for row in components.values()]) * 1000) for key in sorted(keys)},
             "scope": "Synchronized nested module times; do not sum parent and child phases."})

        # 逐个改变分块大小，不修改模型数学定义或仓库源码；三轮交错顺序减少运行顺序偏差。
        original_pool = prefix_cone.prefix_cone_pool
        reports = {chunk: {"files": {s.source_path.name: {"seconds": [], "tensor_differences": {}} for s in sequences},
                          "peak_allocated_bytes": 0} for chunk in args.chunks}

        def use_chunk(chunk):
            def pooled(values, weights, geometry, windows, chunk_size=1024, **kwargs):
                return original_pool(values, weights, geometry, windows, chunk, **kwargs)
            return patch.object(prefix_cone, "prefix_cone_pool", pooled)

        for chunk in args.chunks:
            with use_chunk(chunk):
                infer(maximum)
        for repeat in range(3):
            order = args.chunks if repeat == 0 else list(reversed(args.chunks)) if repeat == 1 else args.chunks[1:] + args.chunks[:1]
            for chunk in order:
                report = reports[chunk]
                with use_chunk(chunk):
                    torch.cuda.reset_peak_memory_stats()
                    for sequence in sequences:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        output = infer_event_output(model, config, sequence)
                        mask = (output["event_logits"].sigmoid() >= threshold).cpu().numpy()
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        row = report["files"][sequence.source_path.name]
                        row["seconds"].append(elapsed)
                        if not np.array_equal(mask, load_prediction(reference / sequence.source_path.name)):
                            raise ValueError("chunk experiment changed final mask")
                        differences = compare(arrays(output), baseline[sequence.source_path.name])
                        row["tensor_differences"].update(differences)
                        del output
                    report["peak_allocated_bytes"] = max(report["peak_allocated_bytes"], torch.cuda.max_memory_allocated())
                report["mean_ms"] = float(np.mean([np.mean(row["seconds"]) for row in report["files"].values()]) * 1000)
                save(args.output / "chunks.json", reports)
                print("chunk", chunk, "pass", repeat + 1, report["mean_ms"], flush=True)

        # 算子跟踪只选普通文件与最大文件；与主计时独立，不用于正式速度比较。
        for sequence in (sequences[0], maximum):
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            probe = Probe(synchronized=False)
            with ExitStack() as stack:
                probe.install(stack, model)
                with torch.profiler.profile(activities=activities) as profile:
                    output = infer(sequence)
                    torch.cuda.synchronize()
            difference = compare(arrays(output), baseline[sequence.source_path.name])
            rows = [{"name": row.key, "calls": row.count, "self_cpu_us": row.self_cpu_time_total,
                     "cpu_us": row.cpu_time_total, "self_device_us": row.self_device_time_total,
                     "device_us": row.device_time_total} for row in profile.key_averages()]
            device_events = [event for event in profile.events() if str(event.device_type).endswith("CUDA")]
            save(args.output / f"operators_{sequence.source_path.stem}.json", {
                "tensor_differences": difference, "operators": rows,
                "cuda_launch_calls": sum(row["calls"] for row in rows if row["name"] == "cudaLaunchKernel"),
                # CUDA事件含嵌套范围及内存操作，不能冒充互不重叠的内核时间总和。
                "cuda_device_events_including_ranges": len(device_events), "events": sequence.length})
            print("operators", sequence.source_path.name, len(device_events), flush=True)

    save(args.output / "provenance.json", {"script_sha256": sha256_file(__file__),
         "checkpoint_sha256": sha256_file(args.run_dir / "artifacts/best.pt"),
         "source_sha256": sha256_file(prefix_cone.__file__), "torch_threads": torch.get_num_threads(),
         "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
         "gpu": torch.cuda.get_device_name(), "all_24_masks_exact": True,
         "all_chunk_tensors_exact": all(not row["tensor_differences"] for r in reports.values() for row in r["files"].values()),
         "neural_parameters": sum(p.numel() for p in model.parameters()),
         "expected_rectangle_queries": {name: sum(math.ceil(g["tokens"] / 1024) * 6 * len(model.encoder.layers) for g in row["geometry"]) for name, row in components.items()}})


if __name__ == "__main__":
    main()
