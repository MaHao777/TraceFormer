"""原第2层三种消息路径的读出恢复：缓存表示，重训原解码器与精修头。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import psutil
import torch
from archive_paths import source_file
from diagnose_original_neural_paths import intervene as message_intervention
from diagnose_summary_weighting import intervene as weight_intervention
from run_vtr_structure_ablation import PreparedScoring
from sklearn.metrics import average_precision_score
from torch import nn

from traceformer.config import load_config
from traceformer.experiments.cached_loss import CachedSequenceLoss
from traceformer.postprocessing.contracts import sha256_file
from traceformer.postprocessing.metrics import BinaryCounts, score_binary
from traceformer.training.engine import (
    _atomic_torch_save,
    _dataset,
    _dataset_manifest,
    _dense_sequence_loss_weight,
    _manifest_identity_digest,
    _write_json,
    build_model,
    load_inference_model,
)
from traceformer.training.losses import frame_cell_objectness_loss
from traceformer.training.sampling import sample_event_indices

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "outputs/main_model"
VARIANTS = ("learned", "uniform_layer2", "no_message_layer2")
FEATURES = ("indices", "locals", "xy", "normalized_time", "batches")
SOURCE_FILES = (
    "scripts/run_original_readout_recovery.py", "scripts/diagnose_original_neural_paths.py",
    "scripts/diagnose_summary_weighting.py", "scripts/run_vtr_structure_ablation.py",
    "src/traceformer/config.py", "src/traceformer/models/event_tokenizer.py",
    "src/traceformer/models/prefix_cone.py", "src/traceformer/models/cone_transformer.py",
    "src/traceformer/models/spatiotemporal_cone.py", "src/traceformer/models/event_decoder.py",
    "src/traceformer/models/event_refinement.py",
    "src/traceformer/experiments/cached_loss.py", "src/traceformer/training/engine.py",
    "src/traceformer/training/losses.py", "src/traceformer/training/sampling.py",
    "src/traceformer/postprocessing/metrics.py",
)


def read(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def intervention(model, variant):
    if variant == "uniform_layer2":
        return weight_intervention(model, "uniform_weights_1", 37)
    if variant == "no_message_layer2":
        return message_intervention(model, "no_message_1", 37)
    if variant != "learned":
        raise ValueError("unknown original neural path")
    return nullcontext()


class OriginalReadout(nn.Module):
    """只组合原有两个模块；参数、层次与生产读出完全相同。"""

    def __init__(self, model):
        super().__init__()
        self.decoder = copy.deepcopy(model.decoder)
        self.refinement = copy.deepcopy(model.event_refinement_head)

    def forward(self, common, tokens):
        if set(common) != set(FEATURES):
            raise ValueError("readout accepts neural features only")
        base = self.decoder(tokens, common["indices"], common["locals"])
        residual = self.refinement(token_features=tokens, event_token_indices=common["indices"],
            event_local_features=common["locals"], event_xy=common["xy"],
            normalized_time=common["normalized_time"], event_batch_indices=common["batches"],
            base_event_logits=base)
        return base + residual.event_logits, residual


def source_model():
    config = load_config(SOURCE / "config.yaml")
    expected = read(REPO / "configs/main_model.json")["files"]
    for name in ("best.pt", "config.yaml"):
        if sha256_file(SOURCE / name) != expected[name]:
            raise ValueError(f"original primary neural artifact changed: {name}")
    model, checkpoint = load_inference_model(config, SOURCE / "best.pt", device=torch.device("cpu"))
    if checkpoint["ablation"] is not None or checkpoint["training_stage"] != "full":
        raise ValueError("requires the original jointly trained neural network")
    return model, config, checkpoint


def extract(model, sequence, config):
    # 标注清零后才调用原网络；真实标注在返回特征之后单独保存。
    blind = replace(sequence, labels=torch.zeros_like(sequence.labels),
                    object_ids=torch.zeros_like(sequence.object_ids))
    captured, tokens, references = [], {}, {}
    handle = model.tokenizer.register_forward_hook(lambda module, args, result: captured.append(result))
    try:
        common = None
        for variant in VARIANTS:
            captured.clear()
            with torch.inference_mode(), intervention(model, variant):
                output = model(blind.events)
            item = captured[0]
            current = dict(zip(FEATURES, (item.event_token_indices, item.event_local_features,
                item.event_xy, item.normalized_time, item.event_batch_indices), strict=True))
            if common is None:
                # 在正常上下文clone，得到可供后续反向保存的普通Tensor。
                common = {name: value.clone() for name, value in current.items()}
            elif not all(torch.equal(common[name], value) for name, value in current.items()):
                raise ValueError("intervention unexpectedly changed event geometry or local features")
            tokens[variant] = output["token_features"].clone()
            references[variant] = output["event_logits"].clone()
    finally:
        handle.remove()
    return {"features": common, "tokens": tokens, "reference_logits": references}


def prepare(root):
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=False)
    model, config, checkpoint = source_model()
    original_head = OriginalReadout(model).eval()
    datasets = {split: _dataset(config, split) for split in ("train", "val")}
    if tuple(map(len, datasets.values())) != (99, 24):
        raise ValueError("requires all 99/24 original files")
    identities = _dataset_manifest(*datasets.values())
    if _manifest_identity_digest(identities) != checkpoint["dataset_manifest_sha256"]:
        raise ValueError("source training/validation data identity changed")
    provenance = {"checkpoint_sha256": sha256_file(SOURCE / "best.pt"),
                  "config_sha256": sha256_file(SOURCE / "config.yaml"),
                  "sources": {name: sha256_file(source_file(name)) for name in SOURCE_FILES}}
    _write_json(root / "provenance.json", provenance)
    _write_json(root / "dataset_manifest.json", identities)
    for name in SOURCE_FILES:
        destination = root / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file(name), destination)
    cache_manifest = {"status": "running", "files": {}, "provenance": provenance,
                      "annotations_in_features": False, "variants": VARIANTS}
    for split, dataset in datasets.items():
        directory = root / "cache" / split
        directory.mkdir(parents=True)
        for index, sequence in enumerate(dataset):
            data = extract(model, sequence, config)
            if index == 0 or sequence.source_path.stem == "train_096":
                with torch.inference_mode():
                    for variant in VARIANTS:
                        actual, _ = original_head(data["features"], data["tokens"][variant])
                        if not torch.equal(actual, data["reference_logits"][variant]):
                            raise ValueError("original readout on cache differs from raw neural forward")
            data["supervision"] = {"labels": sequence.labels, "object_ids": sequence.object_ids,
                "timestamps_ms": sequence.timestamps_ms,
                "weight": _dense_sequence_loss_weight(sequence.events, sequence.labels, config)}
            path = directory / f"{sequence.source_path.stem}.pt"
            _atomic_torch_save(data, path)
            cache_manifest["files"][f"{split}/{path.stem}"] = {
                "source_sha256": sha256_file(sequence.source_path), "cache_sha256": sha256_file(path),
                "events": sequence.length, "tokens": len(data["tokens"]["learned"])}
            cache_manifest["seconds"] = time.perf_counter() - started
            _write_json(root / "cache_manifest.json", cache_manifest)
            print(f"cache {split} {index + 1}/{len(dataset)} {path.stem} "
                  f"{cache_manifest['seconds']:.1f}s", flush=True)
    cache_manifest.update(status="complete", seconds=time.perf_counter() - started)
    _write_json(root / "cache_manifest.json", cache_manifest)


def load_data(root, config, *, splits=("train", "val")):
    manifest = read(root / "cache_manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("incomplete original neural cache")
    expected = manifest["provenance"]
    for name, digest in expected["sources"].items():
        if sha256_file(REPO / name) != digest:
            raise ValueError(f"executed source changed: {name}")
    if sha256_file(SOURCE / "best.pt") != expected["checkpoint_sha256"]:
        raise ValueError("pretrained neural weights changed")
    result = {split: [] for split in splits}
    for key, identity in manifest["files"].items():
        split, name = key.split("/")
        if split not in result:
            continue
        path = root / "cache" / f"{key}.pt"
        if sha256_file(path) != identity["cache_sha256"]:
            raise ValueError("cached neural representation changed")
        item = torch.load(path, weights_only=True, mmap=True)
        if any(value.is_inference() for value in item["features"].values()):
            raise ValueError("training cache must contain normal tensors")
        item["name"] = name
        target = item["supervision"]
        if split == "train":
            item["criterion"] = CachedSequenceLoss(target["labels"], target["object_ids"],
                target["timestamps_ms"], item["features"]["xy"], config)
        else:
            locations = np.column_stack((item["features"]["xy"].numpy(),
                                         target["timestamps_ms"].numpy())).astype(np.int64)
            item["scorer"] = PreparedScoring(locations, target["labels"].numpy().astype(bool),
                                            target["object_ids"].numpy())
        result[split].append(item)
    return result


def fresh_heads(config, seed=37):
    torch.manual_seed(seed)
    template = OriginalReadout(build_model(config))
    heads = {name: copy.deepcopy(template) for name in VARIANTS}
    if len({state_digest(head) for head in heads.values()}) != 1:
        raise ValueError("paired readouts have different initial weights")
    if any(sum(p.numel() for p in head.parameters()) != 14963 for head in heads.values()):
        raise ValueError("not the original neural readout")
    return heads


def training_step(heads, item, config, generator, dropout_seed, optimizers=None):
    outputs = {}
    for name, head in heads.items():
        head.train()
        head.zero_grad(set_to_none=True)
        # 三组相同结构和几何；显式恢复随机种子以配对全部Dropout。
        torch.manual_seed(dropout_seed)
        outputs[name] = head(item["features"], item["tokens"][name])
    common = torch.stack([out[0].detach() for out in outputs.values()]).mean(0)
    target = item["supervision"]
    query = sample_event_indices(target["labels"], logits=common,
        background_ratio=config.training.background_ratio,
        hard_negative_fraction=config.training.hard_negative_fraction, generator=generator,
        max_background_without_positive=config.training.max_background_without_positive)
    losses = {}
    for name, (logits, residual) in outputs.items():
        terms = item["criterion"](logits, query)
        auxiliary = frame_cell_objectness_loss(residual.cell_logits, residual.event_cell_indices,
            target["labels"], alpha=config.loss.focal_alpha, gamma=config.loss.focal_gamma)
        loss = (terms["total"] + config.frame_refinement.loss_weight * auxiliary) * target["weight"]
        if not torch.isfinite(loss):
            raise ValueError("nonfinite recovery loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(heads[name].parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError("nonfinite recovery gradients")
        if optimizers is not None:
            optimizers[name].step()
        losses[name] = float(loss.detach())
    return losses, query


@torch.no_grad()
def validation(heads, data):
    predictions = {name: {} for name in VARIANTS}
    for name, head in heads.items():
        head.eval()
        for item in data:
            logits, _ = head(item["features"], item["tokens"][name])
            predictions[name][item["name"]] = logits
    labels = np.concatenate([item["supervision"]["labels"].numpy() for item in data])
    aps = {name: float(average_precision_score(labels,
        np.concatenate([values[item["name"]].sigmoid().numpy() for item in data])))
        for name, values in predictions.items()}
    return predictions, aps


def train(root, *, budget_seconds=1200):
    started = time.perf_counter()
    _, config, _ = source_model()
    data = load_data(root, config)
    preparation = read(root / "cache_manifest.json")["seconds"]
    # 只按大小和实测成本锁定共同预算，尚未读取任何恢复后的验证分数。
    ordered = sorted(data["train"], key=lambda item: len(item["supervision"]["labels"]))
    probe = fresh_heads(config)
    profiles = []
    for item in (ordered[0], ordered[len(ordered) // 2], ordered[-1]):
        began = time.perf_counter()
        training_step(probe, item, config, torch.Generator().manual_seed(37), 37)
        profiles.append({"file": item["name"], "events": len(item["supervision"]["labels"]),
                         "seconds": time.perf_counter() - began})
    del probe
    coefficient = max(row["seconds"] / max(row["events"], 1) for row in profiles)
    # 小文件固定开销会令线性上界很保守；以中/大文件上界加小文件开销估计。
    slope = max(row["seconds"] / row["events"] for row in profiles[1:])
    per_epoch = sum(max(profiles[0]["seconds"], slope * len(item["supervision"]["labels"]))
                    for item in data["train"]) * 1.15 + 15
    available = budget_seconds - preparation - (time.perf_counter() - started) - 180
    epochs = min(6, math.floor(available / per_epoch))
    if epochs < 2:
        _write_json(root / "budget_gate.json", {"status": "insufficient", "profiles": profiles,
            "estimated_epoch_seconds": per_epoch, "available_seconds": available})
        raise RuntimeError("common two-epoch recovery does not fit the requested compute budget")
    heads = fresh_heads(config)
    protocol = {"scope": "conditional original readout recovery, not end-to-end/from-scratch ablation",
        "variants": VARIANTS, "seed": 37, "epochs": epochs, "train_files": 99, "val_files": 24,
        "readout_parameters": 14963, "frozen_representation": "original primary neural network",
        "initial_state_sha256": state_digest(heads["learned"]), "test_accessed": False,
        "learning_rate": .001, "weight_decay": .0002, "dropout_pairing": "same seed per sequence and step",
        "sampling": "all positives; 12:1 negatives; half hard from symmetric mean of three heads",
        "loss": "original four main terms plus cell auxiliary; original dense sequence weight; no Sparse auxiliary",
        "checkpoint_selection": "maximum validation global probability AP; earliest exact tie",
        "threshold_selection": "validation only; original 19 coarse plus 9 refinement grid",
        "compute_budget_seconds": budget_seconds, "preparation_seconds": preparation,
        "profiles": profiles, "estimated_epoch_seconds": per_epoch,
        "largest_observed_seconds_per_event": coefficient, "budget_locked_before_validation": True,
        "limitations": "single seed; original frozen trunk learned with all original paths and validation selection"}
    _write_json(root / "protocol.json", protocol)
    (root / "heads").mkdir()
    _atomic_torch_save(heads["learned"].state_dict(), root / "initial_readout.pt")
    print(f"LOCKED {epochs} common epochs; estimate {per_epoch:.1f}s/epoch; "
          f"preparation={preparation:.1f}s", flush=True)
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0002)
                  for name, head in heads.items()}
    best = dict.fromkeys(VARIANTS, -1.)
    history = []
    order_generator = torch.Generator().manual_seed(37)
    sampling_generator = torch.Generator().manual_seed(137)
    for epoch in range(1, epochs + 1):
        began = time.perf_counter()
        lr = .001 * (.1 + .9 * .5 * (1 + math.cos(math.pi * (epoch - 1) / max(1, epochs - 1))))
        for optimizer in optimizers.values():
            optimizer.param_groups[0]["lr"] = lr
        losses, query_digest = {name: [] for name in VARIANTS}, hashlib.sha256()
        order = torch.randperm(len(data["train"]), generator=order_generator).tolist()
        for step, index in enumerate(order):
            item = data["train"][index]
            values, query = training_step(heads, item, config, sampling_generator,
                                          37000000 + epoch * 1000 + step, optimizers)
            query_digest.update(item["name"].encode())
            query_digest.update(query.numpy().tobytes())
            for name, value in values.items():
                losses[name].append(value)
            if (step + 1) % 25 == 0:
                print(f"epoch {epoch}/{epochs} step {step + 1}/99 {time.perf_counter() - began:.1f}s", flush=True)
        train_seconds = time.perf_counter() - began
        _, aps = validation(heads, data["val"])
        for name in VARIANTS:
            if aps[name] > best[name]:
                best[name] = aps[name]
                _atomic_torch_save({"model_state": heads[name].state_dict(), "variant": name,
                    "epoch": epoch, "ap": aps[name], "protocol": protocol}, root / "heads" / f"{name}.pt")
        history.append({"epoch": epoch, "learning_rate": lr,
            "train_loss": {name: float(np.mean(values)) for name, values in losses.items()},
            "validation_ap": aps, "shared_query_sha256": query_digest.hexdigest(),
            "train_seconds": train_seconds, "epoch_seconds": time.perf_counter() - began})
        _write_json(root / "history.json", history)
        _atomic_torch_save({"epoch": epoch, "heads": {name: head.state_dict() for name, head in heads.items()},
            "optimizers": {name: value.state_dict() for name, value in optimizers.items()},
            "order_generator": order_generator.get_state(), "sampling_generator": sampling_generator.get_state(),
            "protocol": protocol, "history": history}, root / "training_state.pt")
        _write_json(root / "status.json", {"status": "training", "epoch": epoch, "epochs": epochs,
            "seconds_with_preparation": preparation + time.perf_counter() - started})
        print(f"epoch {epoch}/{epochs} AP={aps} total={history[-1]['epoch_seconds']:.1f}s", flush=True)
    memory = psutil.Process().memory_info()
    _write_json(root / "training_complete.json", {"status": "complete", "best_ap": best,
        "epochs": epochs, "optimizer_steps_per_arm": epochs * 99,
        "seconds": time.perf_counter() - started, "preparation_seconds": preparation,
        "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss)})


def score(logits, data, threshold):
    files, total = {}, BinaryCounts()
    for item in data:
        counts = item["scorer"].score(logits[item["name"]].sigmoid().numpy() >= threshold)
        total += counts
        files[item["name"]] = counts.metrics()
    return {"threshold": threshold, "aggregate": total.metrics(), "files": files}


def evaluate(root):
    started = time.perf_counter()
    source, config, checkpoint = source_model()
    data = load_data(root, config, splits=("val",))["val"]
    heads = fresh_heads(config)
    selections = {}
    for name, head in heads.items():
        saved = torch.load(root / "heads" / f"{name}.pt", weights_only=True)
        head.load_state_dict(saved["model_state"], strict=True)
        selections[name] = {"epoch": saved["epoch"], "ap": saved["ap"],
                            "checkpoint_sha256": sha256_file(root / "heads" / f"{name}.pt")}
    predictions, aps = validation(heads, data)
    results = {}
    directory = root / "validation_logits"
    directory.mkdir()
    for item in data:
        np.savez_compressed(directory / f"{item['name']}.npz",
                            **{name: predictions[name][item["name"]].numpy() for name in VARIANTS})
    for name in VARIANTS:
        if aps[name] != selections[name]["ap"]:
            raise ValueError("selected head does not reproduce its recorded validation AP")
        coarse = [score(predictions[name], data, float(t)) for t in torch.linspace(.05, .95, 19)]
        def key(row):
            return row["aggregate"]["score"], -abs(row["threshold"] - .5)
        best = max(coarse, key=key)
        fine = [score(predictions[name], data, float(t)) for t in torch.linspace(
            max(.01, best["threshold"] - .05), min(.99, best["threshold"] + .05), 9)]
        results[name] = {**selections[name], "selected": max(coarse + fine, key=key),
            "original_threshold": score(predictions[name], data, checkpoint["decision_threshold"]),
            "threshold_search": [{"threshold": row["threshold"], "score": row["aggregate"]["score"]}
                                 for row in coarse + fine]}
        print(f"selected {name} AP={aps[name]:.6f} Score={results[name]['selected']['aggregate']['score']:.6f}", flush=True)
    _write_json(root / "validation_report.json", {"status": "selected", "results": results})
    # 将选中原读出装回真实生产网络，用原始NPZ前向和另一计数实现复核。
    verified = {}
    for name, head in heads.items():
        source.decoder = copy.deepcopy(head.decoder).eval()
        source.event_refinement_head = copy.deepcopy(head.refinement).eval()
        total, files, max_error = BinaryCounts(), {}, 0.
        for sequence in _dataset(config, "val"):
            blind = replace(sequence, labels=torch.zeros_like(sequence.labels),
                            object_ids=torch.zeros_like(sequence.object_ids))
            with torch.inference_mode(), intervention(source, name):
                actual = source(blind.events)
            logits = actual["event_logits"]
            cached = predictions[name][sequence.source_path.stem]
            error = float((logits - cached).abs().max())
            max_error = max(max_error, error)
            if not torch.equal(logits, cached):
                raise ValueError(f"raw/cached recovered neural logits differ: {name} {sequence.source_path.name} {error}")
            locations = np.column_stack((sequence.events[:, :2].numpy(),
                                         sequence.timestamps_ms.numpy())).astype(np.int64)
            counts = score_binary(locations, sequence.labels.numpy().astype(bool),
                sequence.object_ids.numpy(), logits.sigmoid().numpy() >= results[name]["selected"]["threshold"])
            expected = results[name]["selected"]["files"][sequence.source_path.stem]
            if any(asdict(counts)[key] != expected[key] for key in asdict(counts)):
                raise ValueError("independent raw-file counts differ")
            files[sequence.source_path.name] = counts.metrics()
            total += counts
        verified[name] = {"max_logit_error": max_error, "bit_exact": True,
                          "aggregate": total.metrics(), "files": files}
        print(f"raw original graph verified {name}: 24/24 bit exact and independent counts exact", flush=True)
    _write_json(root / "independent_verification.json", verified)
    _write_json(root / "evaluation_complete.json", {"status": "complete", "seconds": time.perf_counter() - started,
        "test_accessed": False, "raw_models_verified": 3, "raw_sequences_per_model": 24})
    _write_json(root / "status.json", {"status": "recovery_complete_not_goal_complete"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate", "all"), default="all")
    parser.add_argument("--budget-seconds", type=int, default=1200)
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = args.root.resolve()
    if args.stage in {"prepare", "all"}:
        prepare(root)
    if args.stage in {"train", "all"}:
        train(root, budget_seconds=args.budget_seconds)
    if args.stage in {"evaluate", "all"}:
        evaluate(root)
