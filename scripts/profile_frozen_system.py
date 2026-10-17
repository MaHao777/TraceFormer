"""细分冻结系统的同步耗时；速度比较仍使用不插桩的 benchmark。"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import ExitStack
from unittest.mock import patch

import benchmark_frozen_system as benchmark
import numpy as np
import torch

from traceformer.gpu.runtime import validate_backends
from traceformer.postprocessing import learning
from traceformer.postprocessing.gpu import CudaEventFeatures
from traceformer.postprocessing.learning import load_event_models, required_event_models
from traceformer.postprocessing.submission import load_prediction


class Timers:
    """同步后记录墙钟时间；父阶段包含子阶段，不重复相加。"""

    def __init__(self, device):
        self.device = device
        self.values = defaultdict(float)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def wrap(self, name, function):
        def measured(*args, **kwargs):
            self.synchronize()
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.synchronize()
                self.values[name] += time.perf_counter() - started
        return measured


def profile(args):
    validate_backends(prediction_backend=args.prediction_backend, feature_backend=args.feature_backend)
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(args.device == "cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    main = benchmark.load_detector("main", args.frozen, device)
    reference = benchmark.load_detector("reference", args.frozen, device)
    reuse = benchmark.can_reuse_reference_logits(main, reference)
    release_path = benchmark.ARCHIVE / "artifacts/event_release"
    release = benchmark.read(release_path / "release_report.json")
    boosters = load_event_models(release_path / "events", names=required_event_models(release["event_contract"]),
                                 backend=args.prediction_backend)
    timers = Timers(device)
    paths = sorted((args.dataset / "test").glob("*.npz"))
    audit = benchmark.read(args.frozen / "dataset_audit.json")["partitions"]["test"]["files"]
    if len(paths) != 24 or {p.name for p in paths} != set(audit):
        raise ValueError("requires all 24 frozen test sequences")
    records = {}
    with ExitStack() as stack:
        classes = {type(layer.attention.neighbor_builder)
                   for bundle in (main, reference) for layer in bundle[0].encoder.layers}
        for cls in classes:
            stack.enter_context(patch.object(cls, "__call__", timers.wrap("neighbors", cls.__call__)))
        for name, bundle in (("main", main), ("reference", reference)):
            for field in ("tokenizer", "encoder", "decoder", "sparse_mixture"):
                module = getattr(bundle[0], field, None)
                if module is not None:
                    stack.enter_context(patch.object(module, "forward", timers.wrap(f"{name}_{field}", module.forward)))

        for owner, field, label in ((learning, "make_context", "feature_context"),
                                   (learning, "build_event_features", "feature_rows"),
                                   (CudaEventFeatures, "__init__", "feature_context"),
                                   (CudaEventFeatures, "rows", "feature_rows")):
            stack.enter_context(patch.object(owner, field, timers.wrap(label, getattr(owner, field))))
        for model in boosters.values():
            stack.enter_context(patch.object(model, "predict", timers.wrap("tree_prediction", model.predict)))

        def run(path):
            if benchmark.sha256_file(path).lower() != audit[path.name]["sha256"].lower():
                raise ValueError("raw test data differs")
            with np.load(path, allow_pickle=False) as data:
                source = data["ev"]
                raw = data["evs_norm"][:, :4].astype(np.float32)
                locations = data["ev_loc"].astype(np.int64)
            timers.values.clear()
            masks, _, _, timings = benchmark.infer(source, raw, locations, path, main, reference,
                boosters, release, prediction_threads=args.prediction_threads,
                reuse_reference_logits=reuse, feature_backend=args.feature_backend)
            for stage in ("complete", "neural", "direct", "event_track"):
                if not np.array_equal(masks[stage], load_prediction(args.frozen / "complete" / stage / path.name)):
                    raise ValueError(f"frozen {stage} mask differs: {path.name}")
            return {"components_seconds": dict(timers.values), "timings": timings}

        run(max(paths, key=lambda p: audit[p.name]["events"]))
        for path in paths:
            records[path.name] = run(path)
            print(f"profile {path.name}: {records[path.name]['timings']['total_seconds']:.3f}s", flush=True)
    keys = set().union(*(row["components_seconds"] for row in records.values()))
    report = {"status": "verified", "sequences": len(records),
        "scope": "Synchronized instrumented wall times; nested phases overlap. Use the separate uninstrumented benchmark for speed comparisons.",
         "prediction_backend": args.prediction_backend,
        "feature_backend": args.feature_backend,
        "component_mean_ms": {key: float(np.mean([row["components_seconds"].get(key, 0) for row in records.values()])*1000) for key in sorted(keys)},
        "files": records, "script_sha256": benchmark.sha256_file(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    profile(benchmark.build_parser().parse_args())
