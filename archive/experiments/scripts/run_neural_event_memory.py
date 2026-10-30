"""纯神经事件记忆实验：共享FP32编码缓存、配对读出训练和末层适应。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil
import torch
from archive_paths import source_file
from run_vtr_structure_ablation import PreparedScoring
from sklearn.metrics import average_precision_score
from torch import nn

from traceformer.config import load_config
from traceformer.data import NPZEventDataset
from traceformer.experiments.ablation import AblationSpec
from traceformer.experiments.cached_loss import CachedSequenceLoss
from traceformer.experiments.event_memory import (
    MemoryConcatDecoder,
    MemoryDecoder,
    SamePixelEventMemory,
    shuffle_memory_within_tokens,
)
from traceformer.models.event_decoder import EventDecoder
from traceformer.models.prefix_cone import prepare_prefix_geometry
from traceformer.postprocessing.contracts import sha256_file
from traceformer.postprocessing.metrics import BinaryCounts
from traceformer.training.engine import (
    _atomic_torch_save,
    _dense_sequence_loss_weight,
    _validate_checkpoint_contract,
    build_model,
)
from traceformer.training.sampling import sample_event_indices

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "outputs/cone_attention_cpu_20261017_235917/seed37/cone_B/attempt_001"
DEFAULT_ROOT = REPO / "outputs/neural_event_memory_20261024"
VARIANTS = ("B", "C", "U", "G")
COUNT_KEYS = tuple(asdict(BinaryCounts()))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def source_model():
    config = load_config(SOURCE / "config.yaml")
    checkpoint = torch.load(SOURCE / "best.pt", map_location="cpu", weights_only=True)
    spec = AblationSpec(**checkpoint["ablation"])
    if spec.sparse or spec.refinement != "none":
        raise ValueError("the shared encoder must be backbone-only")
    _validate_checkpoint_contract(checkpoint, config, ablation=spec)
    model = build_model(config, ablation=spec).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, config, checkpoint


def source_dir(config, split):
    return (Path(config.dataset.root) / split if split != "test"
            else REPO.parent / "EV-UAV-dataset/test")


@torch.inference_mode()
def encode_sequence(model, sequence, memory):
    """只把原始事件送入网络；标注仅附加给训练和独立计分。"""
    event = sequence.events
    tokenized = model.tokenizer(event)
    geometry = prepare_prefix_geometry(tokenized.token_coords, tokenized.token_batch_indices)
    encoded = tokenized.tokens
    for layer in model.encoder.layers[:-1]:
        encoded = layer(encoded, tokenized.token_coords, tokenized.token_batch_indices, geometry=geometry)
    penultimate = encoded.clone()
    encoded = model.encoder.layers[-1](encoded, tokenized.token_coords,
                                       tokenized.token_batch_indices, geometry=geometry)
    tokens = model.encoder.final_norm(encoded)
    reference = model.decoder(tokens, tokenized.event_token_indices, tokenized.event_local_features)
    features = memory(event[:, :2].numpy(), sequence.timestamps_ms.numpy(), event[:, 3].numpy())
    return {
        "tokens": tokens.clone(), "penultimate": penultimate,
        "coords": tokenized.token_coords.clone(), "indices": tokenized.event_token_indices.clone(),
        "locals": tokenized.event_local_features.clone(), "memory": torch.from_numpy(features),
        "labels": sequence.labels.clone(), "object_ids": sequence.object_ids.clone(),
        "timestamps_ms": sequence.timestamps_ms.clone(), "xy": event[:, :2].clone(),
        "reference_logits": reference.clone(),
    }


def prepare(root, split):
    started = time.perf_counter()
    model, config, checkpoint = source_model()
    directory = root / "cache" / split
    directory.mkdir(parents=True, exist_ok=True)
    dataset = NPZEventDataset(source_dir(config, split))
    expected_count = 99 if split == "train" else 24
    if len(dataset) != expected_count:
        raise ValueError("incomplete source partition")
    original = read(SOURCE / "dataset_manifest.json")
    expected = {Path(row["path"]).name: row["sha256"] for row in original["files"]
                if row["split"] == split}
    identity_path = directory / "manifest.json"
    prior = read(identity_path) if identity_path.exists() else {}
    previous = prior.get("files", {})
    record = {"checkpoint_sha256": sha256_file(SOURCE / "best.pt"),
              "config_sha256": sha256_file(SOURCE / "config.yaml"), "split": split,
              "dtype": "float32; raw timestamps float64", "files": {},
              "source_script_sha256": sha256_file(Path(__file__)),
              "memory_protocol": "same/opposite polarity x strict past/future; tau=1,10,50ms",
              "annotations_used_in_features": False,
              "feature_function_sha256": hashlib.sha256(inspect.getsource(encode_sequence).encode()).hexdigest(),
              "feature_sources": {name: sha256_file(source_file(name)) for name in (
                  "src/traceformer/experiments/event_memory.py",
                  "src/traceformer/models/event_tokenizer.py",
                  "src/traceformer/models/prefix_cone.py",
                  "src/traceformer/models/cone_transformer.py")}}
    for key in ("checkpoint_sha256", "config_sha256", "memory_protocol", "feature_function_sha256", "feature_sources"):
        if prior and prior.get(key) != record[key]:
            raise ValueError(f"cached feature definition changed: {key}; use a new cache root")
    memory = SamePixelEventMemory()
    for index, sequence in enumerate(dataset):
        name = sequence.source_path.stem
        path = directory / f"{name}.pt"
        identity = sha256_file(sequence.source_path)
        if split != "test" and identity.lower() != expected[sequence.source_path.name].lower():
            raise ValueError("training/validation data differs from pretrained encoder")
        if name in previous and path.exists():
            item = previous[name]
            if item["source_sha256"] != identity or item["cache_sha256"] != sha256_file(path):
                raise ValueError("existing cache identity changed")
        else:
            data = encode_sequence(model, sequence, memory)
            # 原网络完整前向另验首文件，避免缓存遗漏真实神经分支。
            if index == 0:
                with torch.inference_mode():
                    direct = model(sequence.events)["event_logits"]
                if not torch.equal(data["reference_logits"], direct):
                    raise ValueError("cached encoder path differs from full original neural model")
            data["weight"] = _dense_sequence_loss_weight(sequence.events, sequence.labels, config)
            _atomic_torch_save(data, path)
            item = {"source_sha256": identity, "cache_sha256": sha256_file(path),
                    "events": sequence.length, "tokens": len(data["tokens"])}
        record["files"][name] = item
        record["elapsed_seconds"] = time.perf_counter() - started
        write(identity_path, record)
        print(f"cache {split} {index + 1}/{len(dataset)} {name} {record['elapsed_seconds']:.1f}s", flush=True)
    record["status"] = "complete"
    record["threshold_reference"] = float(checkpoint["decision_threshold"])
    write(identity_path, record)


def load_partition(root, split, config, *, loss):
    manifest = read(root / "cache" / split / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("cache extraction is incomplete")
    if (manifest["checkpoint_sha256"] != sha256_file(SOURCE / "best.pt")
            or manifest["config_sha256"] != sha256_file(SOURCE / "config.yaml")):
        raise ValueError("cached encoder identity differs from the training encoder")
    for name, expected in manifest["feature_sources"].items():
        if sha256_file(source_file(name)) != expected:
            raise ValueError(f"cached neural feature source changed: {name}")
    output = []
    for name, identity in manifest["files"].items():
        path = root / "cache" / split / f"{name}.pt"
        if sha256_file(path) != identity["cache_sha256"]:
            raise ValueError("cache hash changed")
        item = torch.load(path, weights_only=True, mmap=True)
        item["name"] = name
        item["locations"] = np.column_stack((item["xy"].numpy(), item["timestamps_ms"].numpy())).astype(np.int64)
        item["scorer"] = PreparedScoring(item["locations"], item["labels"].numpy().astype(bool),
                                          item["object_ids"].numpy())
        if loss:
            item["criterion"] = CachedSequenceLoss(item["labels"], item["object_ids"],
                                                    item["timestamps_ms"], item["xy"], config)
        output.append(item)
    return output


class Readout(nn.Module):
    def __init__(self, variant, encoder, *, adaptive=False):
        super().__init__()
        self.variant = variant
        self.decoder = (EventDecoder(token_dim=64, dropout=.1) if variant == "B" else
                        MemoryConcatDecoder() if variant == "C" else
                        MemoryDecoder(fusion="additive" if variant == "U" else "gated"))
        self.tail = copy.deepcopy(encoder.layers[-1]) if adaptive else None
        self.final_norm = copy.deepcopy(encoder.final_norm) if adaptive else None

    def forward(self, item):
        if self.tail is None:
            tokens = item["tokens"]
        else:
            batches = torch.zeros(len(item["coords"]), dtype=torch.long)
            tokens = self.final_norm(self.tail(item["penultimate"], item["coords"], batches))
        if self.variant == "B":
            return self.decoder(tokens, item["indices"], item["locals"])
        memory = item["memory"]
        if self.variant == "S":
            # 固定逐文件映射；只破坏记忆与事件的对应，不改变随机训练预算。
            memory = item["shuffled_memory"]
        return self.decoder(tokens, item["indices"], item["locals"], memory)


@torch.no_grad()
def predict(models, data):
    result = {name: {} for name in models}
    for model in models.values():
        model.eval()
    for item in data:
        for name, model in models.items():
            result[name][item["name"]] = model(item).cpu()
    return result


def aggregate(files):
    counts = BinaryCounts(**{key: sum(row[key] for row in files.values()) for key in COUNT_KEYS})
    return counts.metrics()


def score_logits(logits, data, threshold):
    files = {item["name"]: item["scorer"].score(
        (logits[item["name"]].sigmoid().numpy() >= threshold)).metrics() for item in data}
    return {"threshold": threshold, "aggregate": aggregate(files), "files": files}


def select_threshold(logits, data):
    # 与原协议同样的19点粗搜和9点细搜；只接收验证分区。
    if any(not item["name"].startswith("val_") for item in data):
        raise ValueError("decision threshold selection is validation-only")
    rows = [score_logits(logits, data, float(t)) for t in torch.linspace(.05, .95, 19)]
    best = max(rows, key=lambda row: (row["aggregate"]["score"], -abs(row["threshold"] - .5)))
    lower, upper = max(.01, best["threshold"] - .05), min(.99, best["threshold"] + .05)
    rows += [score_logits(logits, data, float(t)) for t in torch.linspace(lower, upper, 9)]
    return max(rows, key=lambda row: (row["aggregate"]["score"], -abs(row["threshold"] - .5)))


def ap_scores(logits, data):
    labels = np.concatenate([item["labels"].numpy() for item in data])
    return {name: float(average_precision_score(labels, np.concatenate(
        [entries[item["name"]].numpy() for item in data]))) for name, entries in logits.items()}


def train(root, seeds, variants, epochs, *, adaptive=False, warm_start=None):
    source, config, source_checkpoint = source_model()
    data = load_partition(root, "train", config, loss=True)
    validation = load_partition(root, "val", config, loss=False)
    if "S" in variants:
        for item in (*data, *validation):
            seed = 20261024 + int(item["name"].split("_")[-1])
            item["shuffled_memory"] = shuffle_memory_within_tokens(
                item["memory"], item["indices"], generator=torch.Generator().manual_seed(seed))
    phase = "adaptive" if adaptive else "frozen"
    protocol_path = root / phase / "protocol.json"
    protocol = {"phase": phase, "planned_seeds": [37, 73, 101], "variants": variants, "epochs": epochs,
                "seed37_encoder_pretraining": "shared 24-epoch backbone-only A3, fixed tokenizer and first two blocks",
                "encoder_updates": "last encoder block and final LayerNorm" if adaptive else "none",
                "selection": "highest complete-validation foreground AP; threshold by validation Score after checkpoint selection",
                "sampling": "all positives, 12:1 negative cap, half common hard negatives from mean detached logits of all paired arms",
                "learning_rate": .0003 if adaptive else .001, "weight_decay": .0002,
                "loss": "unchanged focal + .75 Dice + .25 object-frame + .05 full-event false-alarm; dense weight retained",
                "new_test_data_used": False, "postprocessing": False,
                "source_checkpoint_sha256": sha256_file(SOURCE / "best.pt"),
                "train_cache_manifest_sha256": sha256_file(root / "cache/train/manifest.json"),
                "val_cache_manifest_sha256": sha256_file(root / "cache/val/manifest.json"),
                "script_sha256": sha256_file(Path(__file__)),
                "neural_source_sha256": {name: sha256_file(REPO / name) for name in (
                    "src/traceformer/experiments/event_memory.py",
                    "src/traceformer/experiments/cached_loss.py",
                    "src/traceformer/models/event_decoder.py",
                    "src/traceformer/models/prefix_cone.py")}}
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError("training protocol changed; choose a new phase directory")
    write(protocol_path, protocol)
    for seed in seeds:
        folder = root / phase / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "results.json").exists():
            continue
        models, optimizers, rng, best, history = {}, {}, {}, {}, []
        initial = {"source_checkpoint_sha256": protocol["source_checkpoint_sha256"],
                   "warm_start": {}, "decoder_initial_sha256": {}}
        for variant in variants:
            torch.manual_seed(seed)
            models[variant] = Readout(variant, source.encoder, adaptive=adaptive)
            if warm_start:
                parent = Path(warm_start) / f"seed{seed}" / f"{variant}_best.pt"
                start = torch.load(parent, weights_only=True)
                initial["warm_start"][variant] = {"path": str(parent.resolve()), "sha256": sha256_file(parent)}
                decoder_state = {key.removeprefix("decoder."): value for key, value in start["model_state"].items()
                                 if key.startswith("decoder.")}
                models[variant].decoder.load_state_dict(decoder_state, strict=True)
            rng[variant] = torch.Generator().manual_seed(seed + 1000).get_state()
            optimizers[variant] = torch.optim.AdamW(models[variant].parameters(),
                                                    lr=protocol["learning_rate"], weight_decay=.0002)
            best[variant] = -1.0
            digest = hashlib.sha256()
            for name, value in models[variant].decoder.state_dict().items():
                digest.update(name.encode())
                digest.update(value.contiguous().numpy().tobytes())
            initial["decoder_initial_sha256"][variant] = digest.hexdigest()
        initial_path = folder / "initialization.json"
        if initial_path.exists() and read(initial_path) != initial:
            raise ValueError("paired initial weights or warm-start checkpoints changed")
        write(initial_path, initial)
        order_rng = torch.Generator().manual_seed(seed + 2000)
        sample_rng = torch.Generator().manual_seed(seed + 3000)
        state_path = folder / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, weights_only=True)
            if state["protocol"] != protocol:
                raise ValueError("resume protocol mismatch")
            history, best, rng = state["history"], state["best"], state["rng"]
            for name in variants:
                models[name].load_state_dict(state["models"][name])
                optimizers[name].load_state_dict(state["optimizers"][name])
            order_rng.set_state(state["order_rng"])
            sample_rng.set_state(state["sample_rng"])
        for epoch in range(len(history), epochs):
            started = time.perf_counter()
            lr = protocol["learning_rate"] * (.1 + .9 * .5 * (1 + math.cos(math.pi * epoch / max(epochs - 1, 1))))
            for name, model in models.items():
                model.train()
                for group in optimizers[name].param_groups:
                    group["lr"] = lr
            totals = {name: 0.0 for name in variants}
            order = torch.randperm(len(data), generator=order_rng).tolist()
            for step, index in enumerate(order):
                item = data[index]
                outputs = {}
                for name, model in models.items():
                    torch.set_rng_state(rng[name])
                    optimizers[name].zero_grad(set_to_none=True)
                    outputs[name] = model(item)
                    rng[name] = torch.get_rng_state()
                reference = torch.stack([out.detach() for out in outputs.values()]).mean(0)
                selected = sample_event_indices(item["labels"], logits=reference, background_ratio=12,
                                                 hard_negative_fraction=.5, generator=sample_rng)
                for name, out in outputs.items():
                    losses = item["criterion"](out, selected)
                    total = losses["total"] * item["weight"]
                    if not torch.isfinite(total):
                        raise RuntimeError("nonfinite neural training loss")
                    total.backward()
                    norm = nn.utils.clip_grad_norm_(models[name].parameters(), 1.)
                    if not torch.isfinite(norm):
                        raise RuntimeError("nonfinite neural gradient")
                    optimizers[name].step()
                    totals[name] += float(total.detach())
                if (step + 1) % 20 == 0:
                    write(root / "status.json", {"phase": phase, "seed": seed, "epoch": epoch + 1,
                          "step": step + 1, "steps": len(data), "pid": os.getpid(),
                          "elapsed_epoch_seconds": time.perf_counter() - started})
                    print(f"{phase} seed{seed} epoch{epoch+1}/{epochs} {step+1}/99 {totals}", flush=True)
                del outputs, reference, losses, total
            val_logits = predict(models, validation)
            aps = ap_scores(val_logits, validation)
            record = {"epoch": epoch + 1, "learning_rate": lr, "validation_ap": aps,
                      "train_loss": {name: value / len(data) for name, value in totals.items()},
                      "seconds": time.perf_counter() - started}
            history.append(record)
            for name, value in aps.items():
                if value > best[name]:
                    best[name] = value
                    _atomic_torch_save({"model_state": models[name].state_dict(), "epoch": epoch + 1,
                                        "validation_ap": value, "variant": name, "seed": seed,
                                        "adaptive": adaptive, "protocol": protocol}, folder / f"{name}_best.pt")
            _atomic_torch_save({"protocol": protocol, "history": history, "best": best, "rng": rng,
                               "models": {k: v.state_dict() for k, v in models.items()},
                               "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
                               "order_rng": order_rng.get_state(), "sample_rng": sample_rng.get_state()}, state_path)
            write(folder / "history.json", history)
            print(json.dumps({"phase": phase, "seed": seed, **record}), flush=True)
        results = {}
        for name, model in models.items():
            checkpoint = torch.load(folder / f"{name}_best.pt", weights_only=True)
            model.load_state_dict(checkpoint["model_state"])
            logits = predict({name: model}, validation)[name]
            result = select_threshold(logits, validation)
            result.update(validation_ap=checkpoint["validation_ap"], epoch=checkpoint["epoch"],
                          readout_trainable_parameters=sum(p.numel() for p in model.parameters()),
                          neural_parameters=157187 + sum(p.numel() for p in model.decoder.parameters()),
                          checkpoint_sha256=sha256_file(folder / f"{name}_best.pt"))
            results[name] = result
        # 已训练原A3读出是额外实际门槛，不能只胜过尚未恢复的随机重训B。
        results["original_A3"] = score_logits({item["name"]: item["reference_logits"]
                                               for item in validation}, validation,
                                              float(source_checkpoint["decision_threshold"]))
        write(folder / "results.json", results)
        write(root / "status.json", {"phase": phase, "seed": seed, "status": "seed_complete", "pid": os.getpid()})
        print(json.dumps({"seed": seed, "phase": phase,
                          "validation_scores": {k: v["aggregate"]["score"] for k, v in results.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prepare", choices=("train", "val", "test"))
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[37])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=(*VARIANTS, "S"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--warm-start", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(args.root, args.prepare)
    elif args.train:
        train(args.root, args.seeds, args.variants, args.epochs,
              adaptive=args.adaptive, warm_start=args.warm_start)
    else:
        parser.error("select --prepare or --train")
    print(json.dumps({"pid": os.getpid(), "rss_bytes": psutil.Process().memory_info().rss}), flush=True)


if __name__ == "__main__":
    main()
