"""保留官方网络与原生输入，按原始事件顺序使用统一评分器验证。"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from traceformer.config import load_config
from traceformer.training.engine import _dataset, _file_sha256, _metrics_at_threshold, _write_json


def audit_data(dataset) -> dict:
    records = {}
    for sequence in dataset:
        with np.load(sequence.source_path, allow_pickle=False) as bundle:
            events, native, locations = bundle["ev"], bundle["evs_norm"], bundle["ev_loc"]
            if native.shape != (len(events), 6) or locations.shape != (len(events), 3):
                raise ValueError("native event count/shape differs from raw data")
            expected = np.column_stack((events["x"], events["y"], events["t"].astype(np.int64)))
            if not np.array_equal(locations, expected):
                raise ValueError("native x/y/millisecond coordinates differ from raw event order")
            if not np.array_equal(native[:, 4] > 0, events["label"] > 0) or not np.array_equal(
                native[:, 5], events["name"]
            ):
                raise ValueError("native annotations differ from raw data")
            if not np.isfinite(native[:, :4]).all():
                raise ValueError("nonfinite upstream features")
            records[sequence.source_path.stem] = {
                "events": len(events), "sha256": _file_sha256(sequence.source_path),
            }
    if len(records) != 24:
        raise ValueError("external comparison requires all 24 validation files")
    return records


def probability_metrics(results, config, threshold):
    # 用明确的二值掩码送入同一个评分器，避免 probability→logit 往返误差。
    masks = [(torch.where(p >= threshold, 20.0, -20.0), sequence) for p, sequence in results]
    metrics = _metrics_at_threshold(masks, config, 0.5)
    metrics["threshold"] = float(threshold)
    return metrics


def select_threshold(results, config):
    # 完全匹配现有 19 点粗搜索、9 点局部细搜索及近 0.5 的平分规则。
    steps = config.evaluation.threshold_steps
    coarse = torch.unique(torch.cat((torch.linspace(0.05, 0.95, steps), torch.tensor([0.5]))))
    evaluated = {}

    def consider(candidate):
        key = float(candidate)
        if key not in evaluated:
            evaluated[key] = probability_metrics(results, config, key)

    def best():
        return max(evaluated.values(), key=lambda m: (
            m["competition"]["score"], -abs(m["threshold"] - 0.5)
        ))

    for threshold in coarse.tolist():
        consider(threshold)
    center, step = best()["threshold"], 0.9 / max(1, steps - 1)
    for threshold in torch.linspace(max(0.01, center - step), min(0.99, center + step), 9).tolist():
        consider(threshold)
    return best()


@torch.no_grad()
def evaluate(arguments):
    config = load_config(arguments.config)
    dataset = _dataset(config, "val")
    audited = audit_data(dataset)
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))[arguments.model]
    report = {"model": arguments.model, "upstream": manifest, "dataset": audited,
              "environment": {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
              "checkpoint_training_membership": "not independently documented",
              "score_area": [346, 260], "threshold_protocol": "same validation coarse/refined search",
              "evaluation_config_sha256": _file_sha256(arguments.config),
              "adapter_sha256": _file_sha256(Path(__file__)),
              "timing_comparability": "same-host TraceFormer GPU reevaluation still required"}
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    pending = []
    if arguments.audit_only:
        pending.append("audit-only requested")
    if not torch.cuda.is_available():
        pending.append("CUDA runtime unavailable")
    if manifest.get("checkpoint_status") != "downloaded":
        pending.append("official checkpoint unavailable")
    if pending:
        report.update(status="data_audited_inference_pending",
                      pending_reasons=pending)
        _write_json(output / "report.json", report)
        print(report["status"], flush=True)
        return
    source = Path(manifest["source"])
    report["upstream_config_sha256"] = _file_sha256(source / "configs/evisseg_evuav.yaml")
    report["environment"].update(cuda=torch.version.cuda, device=torch.cuda.get_device_name())
    checkpoint = Path(manifest["checkpoint"])
    if _file_sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("official checkpoint hash differs from downloaded artifact")
    sys.path.insert(0, str(source))
    # 上游配置模块在 import 时解析 argv；仅提供它自己的官方 YAML。
    sys.argv = [str(source / "test.py"), "--config", str(source / "configs/evisseg_evuav.yaml")]
    upstream_config = importlib.import_module("configs.configs").cfg
    upstream_config.root = str(config.dataset.root)
    upstream_config.model_path = str(checkpoint)
    upstream_dataset = importlib.import_module("dataset.ev_uav").EvUAV(upstream_config, mode="val")
    upstream_dataset.file_list = [path.name for path in dataset.files]
    module, constructor = (("model.PACT", "PACT") if arguments.model == "PACT"
                           else ("model.evspsegnet", "evspsegnet"))
    model = getattr(importlib.import_module(module), constructor)(upstream_config).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results, elapsed = [], {}
    for index, sequence in enumerate(dataset):
        native = upstream_dataset[index]
        batch = upstream_dataset.custom_collate([native])
        if not torch.equal(batch["locs"][:, 1:], torch.as_tensor(native["ev_loc"]).long()):
            raise ValueError("upstream collate changed event ordering")
        torch.cuda.synchronize()
        started = time.perf_counter()
        predictions, voxel = model(batch["voxel_ev"])[:2]
        torch.cuda.synchronize()
        elapsed[sequence.source_path.stem] = time.perf_counter() - started
        if not torch.equal(voxel.indices, batch["voxel_ev"].indices):
            raise ValueError("decoder voxel order differs; explicit coordinate remapping is required")
        probability = predictions[batch["p2v_map"].long().cuda()].reshape(-1).cpu()
        if probability.shape != (sequence.length,) or not torch.isfinite(probability).all():
            raise ValueError("invalid full-event output")
        if bool(((probability < 0) | (probability > 1)).any()):
            raise ValueError("upstream output is not a probability")
        torch.save({"event_probabilities": probability, "source_sha256": audited[
            sequence.source_path.stem]["sha256"]}, output / f"{sequence.source_path.stem}.pt")
        results.append((probability, sequence))
        print(f"{arguments.model} {sequence.source_path.name}: {len(probability)} events", flush=True)
    report.update(status="complete", fixed_threshold=probability_metrics(results, config, 0.5),
                  selected_threshold=select_threshold(results, config),
                  parameters=sum(p.numel() for p in model.parameters()),
                  neural_forward_seconds=elapsed,
                  timing_note="Single forward per sequence; excludes preprocessing and is not an end-to-end speed benchmark.")
    _write_json(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["PACT", "EV-SpSegNet"], required=True)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/baselines/upstream_manifest.json"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    evaluate(parser.parse_args())
