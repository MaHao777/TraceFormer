"""从当前HUAV检查点生成逐事件后处理缓存。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from huav.config import HUAVConfig
from huav.data.dataset import NPZEventDataset
from huav.training.engine import forward_with_sparse_mixture, load_inference_model

from .contracts import (
    CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    CacheManifest,
    EventCache,
    sha256_file,
)


def _source_arrays(path: Path, *, include_labels: bool) -> dict[str, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as saved:
        locations = saved["ev_loc"].astype(np.int64, copy=True)
        raw = saved["evs_norm"][:, :4].astype(np.float32, copy=True)
        labels = saved["evs_norm"][:, 4].astype(bool, copy=True) if include_labels else None
        target_ids = saved["evs_norm"][:, 5].astype(np.int64, copy=True) if include_labels else None
    return {
        "locations": locations,
        "raw_features": raw,
        "labels": labels,
        "target_ids": target_ids,
    }


def cache_split(
    config: HUAVConfig,
    checkpoint_path: str | Path,
    output_root: str | Path,
    *,
    split: str,
    device: str = "cpu",
    progress: bool = False,
) -> CacheManifest:
    """生成训练或盲推缓存；验证缓存从源头排除标签。"""

    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    torch_device = torch.device(device)
    checkpoint_path = Path(checkpoint_path).resolve()
    model, checkpoint = load_inference_model(config, checkpoint_path, device=torch_device)
    threshold = float(checkpoint.get("decision_threshold", config.inference.sigmoid_threshold))
    dataset = NPZEventDataset(Path(config.dataset.root) / split)
    output_dir = Path(output_root) / split
    output_dir.mkdir(parents=True, exist_ok=False)
    include_labels = split == "train"
    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sequence = dataset[index]
            events = sequence.events.to(torch_device)
            batch_indices = torch.zeros(len(events), dtype=torch.long, device=torch_device)
            output, sparse_view = forward_with_sparse_mixture(
                model,
                sequence,
                events,
                batch_indices,
                config,
                mixture_weight=1.0,
            )
            source = _source_arrays(sequence.source_path, include_labels=include_labels)
            probability = torch.sigmoid(output["event_logits"]).cpu().numpy()
            cache = EventCache(
                name=sequence.source_path.name,
                locations=source["locations"],
                raw_features=source["raw_features"],
                base_probability=probability,
                token_features=output["token_features"].detach().cpu().numpy(),
                event_token_indices=(output["event_token_indices"].detach().cpu().numpy()),
                prediction=probability >= threshold,
                labels=source["labels"],
                target_ids=source["target_ids"],
            )
            cache_path = output_dir / sequence.source_path.name
            cache.save(cache_path, include_labels=include_labels)
            records.append(
                {
                    "name": sequence.source_path.name,
                    "events": sequence.length,
                    "tokens": int(len(cache.token_features)),
                    "sparse_view": sparse_view is not None,
                    "source_sha256": sha256_file(sequence.source_path),
                    "cache_sha256": sha256_file(cache_path),
                }
            )
            if progress:
                print(
                    f"  {index + 1:03d}/{len(dataset):03d} "
                    f"{sequence.source_path.name}: {sequence.length:,} events",
                    flush=True,
                )
    manifest = CacheManifest(
        split=split,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        checkpoint_sha256=sha256_file(checkpoint_path),
        config_sha256=sha256_file(Path(output_root) / "config.yaml")
        if (Path(output_root) / "config.yaml").is_file()
        else "",
        files=tuple(records),
        labels_included=include_labels,
    )
    manifest.write(output_dir / "manifest.json")
    (output_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "path": str(checkpoint_path),
                "sha256": manifest.checkpoint_sha256,
                "decision_threshold": threshold,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
