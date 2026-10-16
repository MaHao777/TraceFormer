"""从同主机新生成的神经特征重放冻结发布系统，并独立计分。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from traceformer.postprocessing.contracts import EventCache, density_group
from traceformer.postprocessing.direct_verifier import event_branches
from traceformer.postprocessing.learning import (
    load_event_models,
    positive_prediction_threads,
    predict_event_scores,
    required_event_models,
)
from traceformer.postprocessing.submission import (
    evaluate_predictions,
    load_prediction,
    write_prediction_npz,
)
from traceformer.postprocessing.tracking import rescue_reference_tracks
from traceformer.training.engine import _file_sha256, _write_json


def evaluate(args):
    release = json.loads((args.release / "release_report.json").read_text(encoding="utf-8"))
    for name, expected in release["event_model_sha256"].items():
        if _file_sha256(args.release / "events" / name).lower() != expected.lower():
            raise ValueError("frozen event model hash differs")
    models = load_event_models(
        args.release / "events", names=required_event_models(release["event_contract"])
    )
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(args.cache.glob("*.npz"))
    source = args.dataset_root / args.split
    names = {p.name for p in paths}
    if names != {p.name for p in source.glob("*.npz")} or len(paths) != 24:
        raise ValueError("requires exactly the 24 selected partition files")
    if names != {p.name for p in args.reference.glob("*.npz")}:
        raise ValueError("reference predictions differ from selected partition")
    records = {}
    for path in paths:
        cache = EventCache.load(path, allow_labels=False)
        with np.load(source / path.name, allow_pickle=False) as data:
            if not np.array_equal(cache.locations, data["ev_loc"].astype(np.int64)):
                raise ValueError("cache event order differs from raw data")
        scores = predict_event_scores(cache, models, num_threads=args.prediction_threads)
        _, _, tracked, direct = event_branches(cache, scores, release["event_contract"])
        group, _ = density_group(cache.locations)
        reference = load_prediction(args.reference / path.name)
        final, rescue = rescue_reference_tracks(cache.locations, direct | tracked, reference, density_group=group)
        for stage, mask in {"neural": cache.prediction, "direct": direct,
                            "event_track": direct | tracked, "complete": final}.items():
            write_prediction_npz(args.output / stage / path.name, mask)
        records[path.name] = {"rescue": rescue, "density_group": group,
            "source_sha256": _file_sha256(source / path.name),
            "cache_sha256": _file_sha256(path), "reference_sha256": _file_sha256(args.reference / path.name),
            "final_sha256": _file_sha256(args.output / "complete" / path.name)}
        print(f"complete system {path.name}", flush=True)
    report = {"status": "complete", "labels_accessed_during_inference": False,
        "prediction_threads": args.prediction_threads, "active_event_models": list(models),
        "split": args.split, "test_threshold_selection": False,
        "adapter_sha256": _file_sha256(Path(__file__)),
        "release_report_sha256": _file_sha256(args.release / "release_report.json"), "files": records,
        "stages": {stage: evaluate_predictions(args.output / stage, source)
                   for stage in ("neural", "direct", "event_track", "complete")}}
    _write_json(args.output / "report.json", report)


def build_parser():
    """共用正整数校验，便于检查入口默认值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "reference", "release", "dataset-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--prediction-threads", type=positive_prediction_threads, default=8)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
