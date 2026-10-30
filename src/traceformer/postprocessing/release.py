"""当前事件模型发布来源与盲推缓存的统一检查。"""

import json
from pathlib import Path

from .contracts import (
    CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    CacheManifest,
    sha256_file,
)
from .learning import load_event_models, required_event_models


def validate_cache_manifest(directory, *, split):
    """逐个检查缓存哈希，拒绝未完成、混入或已更换的缓存。"""
    directory = Path(directory)
    manifest = CacheManifest.read(directory / "manifest.json")
    if (manifest.split != split or manifest.labels_included != (split == "train")
            or manifest.cache_schema_version != CACHE_SCHEMA_VERSION
            or manifest.feature_schema_version != FEATURE_SCHEMA_VERSION):
        raise ValueError("cache manifest split or feature schema differs")
    names = sorted(path.name for path in directory.glob("*.npz"))
    if names != sorted(row["name"] for row in manifest.files):
        raise ValueError("cache manifest file set differs")
    for row in manifest.files:
        if sha256_file(directory / row["name"]) != row["cache_sha256"]:
            raise ValueError(f"cache hash differs: {row['name']}")
    return manifest


def load_event_release(directory, *, cache_dir=None, checkpoint=None, config=None, backend="cpu", load_models=True):
    """按发布目录相对路径加载，来源核验不依赖原主机的绝对路径。"""
    directory = Path(directory)
    report = json.loads((directory / "release_report.json").read_text(encoding="utf-8"))
    provenance = report["provenance"]
    manifest_path = directory / "training_manifest.json"
    if sha256_file(manifest_path) != provenance["training_manifest_sha256"]:
        raise ValueError("release training manifest hash differs")
    training_manifest = CacheManifest.read(manifest_path)
    if training_manifest.split != "train" or not training_manifest.labels_included:
        raise ValueError("release training manifest split differs")
    for key in ("checkpoint_sha256", "config_sha256", "cache_schema_version", "feature_schema_version"):
        if getattr(training_manifest, key) != provenance[key]:
            raise ValueError(f"release training manifest {key} differs")
    snapshot = directory / "development_contract.json"
    if sha256_file(snapshot) != report["source_training_report_sha256"]:
        raise ValueError("release development contract hash differs")
    development = json.loads(snapshot.read_text(encoding="utf-8"))
    training_names = sorted(row["name"] for row in training_manifest.files)
    if (len(training_names) != report["training_files"]
            or training_names != sorted(report["model"]["training_sequences"])
            or training_names != sorted(development["training_names"] + development["holdout_names"])):
        raise ValueError("release training sequence set differs")
    if not development["event_stage"]["gate"]["passed"]:
        raise ValueError("event development gate did not pass")
    contract = {key: development["event_stage"][key] for key in report["event_contract"]}
    if contract != report["event_contract"]:
        raise ValueError("release event contract differs")
    if (provenance["cache_schema_version"] != CACHE_SCHEMA_VERSION
            or provenance["feature_schema_version"] != FEATURE_SCHEMA_VERSION
            or provenance["token_feature_dtype"] != "float16"):
        raise ValueError("release feature schema differs")
    if cache_dir is not None:
        manifest = validate_cache_manifest(cache_dir, split="val")
        for key in ("checkpoint_sha256", "config_sha256"):
            if getattr(manifest, key) != provenance[key]:
                raise ValueError(f"release cache {key} differs")
    for path, key in ((checkpoint, "checkpoint_sha256"), (config, "config_sha256")):
        if path is not None and sha256_file(path) != provenance[key]:
            raise ValueError(f"release {key} differs")
    for name, expected in report["event_model_sha256"].items():
        if sha256_file(directory / "events" / name) != expected:
            raise ValueError(f"release event model hash differs: {name}")
    models = load_event_models(
        directory / "events", names=required_event_models(report["event_contract"]), backend=backend,
    ) if load_models else {}
    return models, report


def event_validation_gate(candidate, baseline):
    """沿用事件发布门槛；困难文件只用于诊断和验收。"""
    result = {
        "score_gain": candidate["aggregate"]["score"] - baseline["aggregate"]["score"],
        "val_023_pd_gain": candidate["files"]["val_023.npz"]["pd"]
        - baseline["files"]["val_023.npz"]["pd"],
        "fa": candidate["aggregate"]["fa"],
    }
    result["passed"] = bool(
        result["score_gain"] > 0 and result["val_023_pd_gain"] >= 0
        and result["fa"] <= 6e-6
    )
    return result
