"""本地重训事件后处理；只读取 train/val，不接触测试集。"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import lightgbm
import numpy as np
import psutil
import torch

from traceformer.config import load_config
from traceformer.data import NPZEventDataset
from traceformer.postprocessing.cache import _source_arrays, cache_split, infer_event_cache
from traceformer.postprocessing.contracts import PostprocessProtocol, sha256_file
from traceformer.postprocessing.learning import predict_event_scores, train_event_models
from traceformer.postprocessing.pipeline import train_pipeline
from traceformer.training.engine import load_inference_model


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args):
    config = load_config(args.config)
    if config.training.num_threads != 8:
        raise ValueError("local protocol requires saved config with 8 CPU threads")
    torch.set_num_threads(8)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = {"status": "preflight", "test_inference": False}

    def phase(value):
        status.update(status=value, elapsed_seconds=time.perf_counter() - started)
        write_json(output / "status.json", status)
        print(value, flush=True)

    try:
        shutil.copy2(args.config, output / "config.yaml")
        shutil.copy2(args.checkpoint, output / "best.pt")
        train = NPZEventDataset(Path(config.dataset.root) / "train")
        validation = NPZEventDataset(Path(config.dataset.root) / "val")
        if (len(train), len(validation)) != (99, 24):
            raise ValueError("requires 99 training and 24 validation sequences")
        # 先核对原网络训练的数据身份，避免旧权重与被替换的数据混用。
        original = json.loads((args.checkpoint.parent / "dataset_manifest.json").read_text(encoding="utf-8"))
        rows = original["files"]
        actual = {f"{split}/{path.name}" for split, data in (("train", train), ("val", validation))
                  for path in data.files}
        if actual != {f"{row['split']}/{Path(row['path']).name}" for row in rows}:
            raise ValueError("checkpoint dataset file set differs")
        for row in rows:
            path = Path(config.dataset.root) / row["split"] / Path(row["path"]).name
            if sha256_file(path).lower() != row["sha256"].lower():
                raise ValueError(f"checkpoint data identity differs: {path.name}")
        write_json(output / "data_identity_audit.json", {"passed": True, "files": len(rows),
                   "checkpoint_dataset_manifest_sha256": sha256_file(args.checkpoint.parent / "dataset_manifest.json")})
        repo = Path(__file__).resolve().parents[1]
        sources = sorted((repo / "src").rglob("*.py")) + sorted((repo / "scripts").glob("*.py"))
        sources += [repo / "pyproject.toml"]
        write_json(output / "source_hashes.json", {str(p.relative_to(repo)): sha256_file(p) for p in sources})
        with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sources:
                archive.write(path, str(path.relative_to(repo)))
        write_json(output / "environment.json", {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "lightgbm": lightgbm.__version__, "numpy": np.__version__, "threads": 8,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "config_sha256": sha256_file(args.config),
        })
        # 仅读取 NPY 头选择最大训练序列，不解压或读取测试分区。
        def event_count(path):
            with zipfile.ZipFile(path) as archive, archive.open("ev.npy") as stream:
                version = np.lib.format.read_magic(stream)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                shape, _, _ = readers[version](stream)
                return shape[0]

        largest = max(range(len(train)), key=lambda index: event_count(train.files[index]))
        model, checkpoint = load_inference_model(config, args.checkpoint, device=torch.device("cpu"))
        sequence = train[largest]
        smoke = output / "preflight"
        smoke.mkdir()
        with torch.inference_mode():
            cache = infer_event_cache(
                model, config, sequence, checkpoint["decision_threshold"],
                source=_source_arrays(sequence.source_path, include_labels=True),
            )
        cache.save(smoke / cache.name, include_labels=True)
        models, _ = train_event_models(smoke, [cache.name], smoke / "models",
                                      replace(PostprocessProtocol(), small_rounds=2, large_rounds=2))
        scores = predict_event_scores(cache, models, num_threads=8)
        if not all(np.isfinite(value).all() for value in scores.values()):
            raise ValueError("preflight tree scores are nonfinite")
        write_json(smoke / "report.json", {
            "passed": True, "sequence": cache.name, "events": len(cache.prediction),
            "process_peak_rss_bytes": psutil.Process().memory_info().peak_wset,
            "available_memory_bytes": psutil.virtual_memory().available,
        })
        del model, cache, scores, models, sequence
        import gc

        gc.collect()
        for split in ("train", "val"):
            phase(f"cache_{split}")
            cache_split(config, args.checkpoint, output, split=split, device="cpu", progress=True)
        phase("training")
        report = train_pipeline(output / "train", output / "training", PostprocessProtocol(),
                                event_only=True, progress=lambda message: print(message, flush=True))
        if report["status"] != "event_released":
            phase(report["status"])
            return
        phase("validation")
        command = [sys.executable, str(repo / "scripts/evaluate_postprocess_stages.py"),
                   "--event-only", "--cache-dir", str(output / "val"),
                   "--training-dir", str(output / "training"),
                   "--event-release-dir", str(output / "training/event_release"),
                   "--dataset-root", str(Path(config.dataset.root) / "val"),
                   "--output-dir", str(output / "validation")]
        subprocess.run(command, check=True)
        subprocess.run(command + ["--rescore-only"], check=True)
        result = json.loads((output / "validation/independent_validation.json").read_text(encoding="utf-8"))
        baseline = json.loads((args.checkpoint.parent / "metrics.json").read_text(encoding="utf-8"))
        if result["objectness"]["aggregate"]["score"] != baseline["competition"]["score"]:
            raise ValueError("neural cache metrics do not reproduce checkpoint baseline")
        phase("validation_accepted" if result["validation_gate"]["passed"] else "rejected_validation_gate")
    except Exception as error:
        status["error"] = repr(error)
        phase("failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
