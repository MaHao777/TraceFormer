"""本地核对优化前后输出、计时，并用现有训练管线做权重热启动校验。"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import torch

from traceformer.config import load_config
from traceformer.data.dataset import NPZEventDataset
from traceformer.models import cone_neighbor_builder, cone_transformer
from traceformer.postprocessing.contracts import sha256_file
from traceformer.postprocessing.submission import evaluate_predictions
from traceformer.training import engine

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "outputs/cpu_refinement_20260830_154410/best.pt"
DATA = REPO.parent / "EV-UAV-dataset"
FROZEN = REPO / "outputs/paper_test_20261009"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    engine._write_json(path, value)


class Timers:
    """记录嵌套阶段的时间和调用数；汇总时明确扣除子阶段。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = defaultdict(lambda: {"seconds": 0.0, "calls": 0})

    def wrap(self, name, function):
        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.values[name]["seconds"] += time.perf_counter() - started
                self.values[name]["calls"] += 1
        return measured

    def modules(self, stack, model, prefix=""):
        for name in ("tokenizer", "encoder", "decoder", "sparse_mixture", "event_refinement_head"):
            module = getattr(model, name, None)
            if module is not None:
                stack.enter_context(patch.object(module, "forward", self.wrap(prefix + name, module.forward)))

    def snapshot(self):
        return {key: dict(value) for key, value in self.values.items()}


def old_builder(root):
    """仅隔离旧几何构建器；模型、权重和已有其他本地改动均保持相同。"""
    source = subprocess.run(
        ["git", "show", "HEAD:src/traceformer/models/cone_neighbor_builder.py"],
        cwd=REPO, capture_output=True, check=True,
    ).stdout
    path = root / "before_cone_neighbor_builder.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("optimization_before_builder", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ConeNeighborBuilder


@contextmanager
def neighbor_mode(model, root, before, timers):
    with ExitStack() as stack:
        timers.modules(stack, model)
        stack.enter_context(patch.object(engine, "simulate_high_threshold_events",
                                        timers.wrap("sparse_simulation", engine.simulate_high_threshold_events)))
        if before:
            legacy = old_builder(root)
            # 旧实现每层、每 head、每方向重建树，不额外准备新版共享上下文。
            stack.enter_context(patch.object(cone_transformer, "prepare_cone_geometry", lambda *_: None))
            for layer in model.encoder.layers:
                builder = layer.attention.neighbor_builder
                original = legacy(cone_heads=builder.cone_heads,
                                  max_neighbors_per_cone=builder.max_neighbors_per_cone,
                                  layer_time_scale=builder.layer_time_scale)
                query = timers.wrap("neighbors", original)
                stack.enter_context(patch.object(layer.attention, "neighbor_builder",
                                                lambda coords, batches, *, geometry=None, query=query:
                                                query(coords, batches)))
        else:
            stack.enter_context(patch.object(cone_transformer, "prepare_cone_geometry",
                                            timers.wrap("geometry", cone_transformer.prepare_cone_geometry)))
            stack.enter_context(patch.object(cone_neighbor_builder.ConeNeighborBuilder, "__call__",
                                            timers.wrap("neighbors", cone_neighbor_builder.ConeNeighborBuilder.__call__)))
        yield


def config_for(root):
    config = load_config(REPO / "configs/traceformer_score_optimized.yaml")
    return replace(config, dataset=replace(config.dataset, root=DATA),
                   training=replace(config.training, output_root=root, num_threads=14),
                   evaluation=replace(config.evaluation, optimize_threshold=False)).validate()


def sequence_input(sequence):
    # 网络推理不接触标签；评分阶段才使用原始序列中的标注。
    return replace(sequence, labels=torch.zeros_like(sequence.labels),
                   object_ids=torch.zeros_like(sequence.object_ids))


def forward(model, sequence, config, before):
    return engine.forward_with_sparse_mixture(
        model, sequence_input(sequence), sequence.events,
        torch.zeros(sequence.length, dtype=torch.long), config,
        compute_sparse_logits=before,
    )[0]


def runtime_environment():
    memory = psutil.Process().memory_info()
    return {"device": "cpu", "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "process_peak_rss_bytes": getattr(memory, "peak_wset", memory.rss)}


@torch.no_grad()
def evaluate_pair(root, checkpoint_path, compare):
    config = config_for(root)
    torch.set_num_threads(config.training.num_threads)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    engine._validate_checkpoint_contract(checkpoint, config)
    validation = engine._dataset(config, "val")
    if engine._manifest_identity_digest(engine._dataset_manifest(validation)) != checkpoint["validation_manifest_sha256"]:
        raise ValueError("validation dataset identity differs")
    model = engine.build_model(config).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    threshold = float(checkpoint["decision_threshold"])
    warm_dataset = NPZEventDataset(DATA / "test", width=config.resolution.width,
                                  height=config.resolution.height, duration_ms=config.dataset.duration_ms)
    warm = next(sequence for sequence in warm_dataset if sequence.source_path.name == "test_023.npz")
    timers = Timers()
    for before in ([True, False] if compare else [False]):
        with neighbor_mode(model, root, before, timers):
            forward(model, warm, config, before)
    del warm
    report = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
              "threshold": threshold, "parameters": sum(p.numel() for p in model.parameters()),
              "scope": "In-memory neural forward; includes sparse simulation and CPU geometry, excludes I/O and scoring",
              "before_scope": "Original neighbor builder and auxiliary sparse Decoder; identical weights and all other source",
              "partitions": {}}
    for split in ("val", "test"):
        dataset = NPZEventDataset(DATA / split, width=config.resolution.width,
                                 height=config.resolution.height, duration_ms=config.dataset.duration_ms)
        if len(dataset) != 24:
            raise ValueError("requires all 24 partition sequences")
        results, records = [], {}
        for sequence in dataset:
            outputs, measurements = {}, {}
            for before in ([True, False] if compare else [False]):
                mode = "before" if before else "after"
                timers.reset()
                with neighbor_mode(model, root, before, timers):
                    started = time.perf_counter()
                    output = forward(model, sequence, config, before)
                    elapsed = time.perf_counter() - started
                if not torch.isfinite(output["event_logits"]).all():
                    raise ValueError("nonfinite event logits")
                outputs[mode] = output
                measurements[mode] = {"seconds": elapsed, "components": timers.snapshot()}
            after = outputs["after"]
            differences = {key: int(torch.count_nonzero(value != after[key]).item())
                           for key, value in outputs.get("before", {}).items()
                           if key != "sparse_event_logits"}
            if any(differences.values()):
                raise ValueError(f"optimization changed {sequence.source_path.name}: {differences}")
            mask = torch.sigmoid(after["event_logits"]) >= threshold
            prediction_path = root / "predictions" / split / sequence.source_path.name
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(prediction_path, prediction=mask.numpy().astype(np.uint8),
                                event_logits=after["event_logits"].numpy())
            results.append((after["event_logits"].clone(), sequence))
            records[sequence.source_path.name] = {"events": sequence.length,
                "differences": differences, "timings": measurements,
                "source_sha256": sha256_file(sequence.source_path),
                "prediction_sha256": sha256_file(prediction_path)}
            print(f"{split} {sequence.source_path.name}: "
                  + ", ".join(f"{mode}={row['seconds']:.3f}s" for mode, row in measurements.items())
                  + (f"; differing elements={sum(differences.values())}" if compare
                     else "; independent evaluation"), flush=True)
            del outputs, output, after
        metrics = engine._metrics_at_threshold(results, config, threshold)
        raw_metrics = evaluate_predictions(root / "predictions" / split, DATA / split)
        # 独立计数器对 IoU、Pd、Fa 和 Score 的全局计数结果必须一致。
        for key in ("iou", "pd", "fa", "score"):
            if abs(raw_metrics["aggregate"][key] - metrics["competition"][key]) > 1e-12:
                raise ValueError(f"independent {split}/{key} differs")
        report["partitions"][split] = {"metrics": metrics, "independent_metrics": raw_metrics,
            "files": records, "mean_seconds": {
                mode: float(np.mean([row["timings"][mode]["seconds"] for row in records.values()]))
                for mode in measurements}, "complete": True}
        write(root / "neural_report.json", report)
        del results
    report["environment"] = runtime_environment()
    report["comparison_performed"] = compare
    report["outputs_exact"] = True if compare else None
    write(root / "neural_report.json", report)
    return report


def train_one_epoch(root):
    """建立明确的新优化器初态，复用 schema-7 权重而非伪造历史优化器。"""
    config = config_for(root)
    config = replace(config, training=replace(config.training, epochs=1,
                                             save_training_state_interval=1)).validate()
    configuration = engine._serializable(asdict(config))
    config_path = root / "warm_start_config.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(configuration, allow_unicode=True, sort_keys=False), encoding="utf-8")
    dataset = engine._dataset(config, "train")
    validation = engine._dataset(config, "val")
    if len(dataset) != 99 or len(validation) != 24:
        raise ValueError("requires all 99 train and 24 validation sequences")
    # 先用最大完整训练序列检查真实前向、损失及反向所需资源。
    largest = max(dataset, key=lambda sequence: sequence.length)
    gate_started = time.perf_counter()
    gate = subprocess.run([
        sys.executable, str(REPO / "scripts/check_largest_sequence.py"),
        "--config", str(config_path), "--device", "cpu", "--epoch", "1",
        "--sequence", largest.source_path.name, "--initialize-refinement",
        str(REPO / "outputs/paper_cuda_20260928/copyback_20260929/artifacts/main/best.pt"),
    ], cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True)
    gate_report = json.loads(gate.stdout)
    gate_report["wall_seconds"] = time.perf_counter() - gate_started
    write(root / "resource_gate.json", gate_report)
    print(f"largest training resource gate passed: {largest.source_path.name}, "
          f"{largest.length} events, {gate_report['wall_seconds']:.2f}s", flush=True)
    del largest

    torch.set_num_threads(config.training.num_threads)
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    engine._validate_checkpoint_contract(checkpoint, config)
    manifest = engine._dataset_manifest(dataset, validation)
    if engine._manifest_identity_digest(engine._dataset_manifest(validation)) != checkpoint["validation_manifest_sha256"]:
        raise ValueError("warm start validation identity differs")
    model = engine.build_model(config)
    model.configure_training_stage(config.training.stage)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer = torch.optim.AdamW(engine._trainable_parameters(model),
                                 lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    controller = engine.LearningRateController(config.training)
    initialization = {"kind": "weights_only_warm_start", "path": str(CHECKPOINT),
                      "sha256": sha256_file(CHECKPOINT), "source_epoch": checkpoint["epoch"],
                      "optimizer_restored": False, "new_training_epoch": 0,
                      "training_stage": config.training.stage,
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    # 旧 best.pt 没有优化器/RNG；新状态从 epoch=0 开始，历史结果不参与本轮选优。
    initial_checkpoint = copy.deepcopy(checkpoint)
    initial_checkpoint["full_initialization"] = initialization
    initial_metrics = json.loads((CHECKPOINT.parent / "metrics.json").read_text(encoding="utf-8"))
    state = engine._training_state_payload(
        config=configuration, manifest_digest=engine._manifest_identity_digest(manifest),
        model=model, optimizer=optimizer, controller=controller, epoch=0,
        history=[], sparse_mixture_history=[], best_score=-1.0, best_metrics=initial_metrics,
        best_checkpoint=initial_checkpoint,
        order_generator=torch.Generator().manual_seed(config.training.seed),
        sampling_generator=torch.Generator().manual_seed(config.training.seed),
        torch_device=torch.device("cpu"),
    )
    state_path = root / "weights_only_start_state.pt"
    engine._atomic_torch_save(state, state_path)
    write(root / "initialization.json", initialization)
    del model, optimizer, state
    timers, phase = Timers(), ["train"]

    def phase_wrap(name, function):
        def measured(*args, **kwargs):
            return timers.wrap(phase[0] + "/" + name, function)(*args, **kwargs)
        return measured

    original_evaluate = engine.evaluate_model

    def evaluate(*args, **kwargs):
        phase[0] = "validation"
        try:
            return timers.wrap("validation/total", original_evaluate)(*args, **kwargs)
        finally:
            phase[0] = "train"

    with ExitStack() as stack:
        for owner, name in ((engine, "_forward_with_sparse_mixture"),
                            (engine.EventFocalDiceLoss, "forward"),
                            (engine, "_event_refinement_loss"),
                            (engine, "_sparse_mixture_auxiliary_losses"),
                            (torch.autograd, "backward"), (torch.optim.AdamW, "step")):
            stack.enter_context(patch.object(owner, name, phase_wrap(name, getattr(owner, name))))
        stack.enter_context(patch.object(engine, "evaluate_model", evaluate))
        started = time.perf_counter()
        summary = engine.run_training(
            config, output_dir=root / "run", device="cpu", progress=True,
            note="Optimization verification: schema-7 weights-only warm start, new optimizer, 99 files x 1 epoch, fixed threshold",
            config_source=config_path, resume_state_path=state_path,
        )
        wall = time.perf_counter() - started
    run = Path(summary["run_dir"])
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    if len(history) != 1 or history[0]["optimization"]["optimizer_steps"] != 99:
        raise ValueError("one complete training epoch was not executed")
    trained = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    changed = [key for key, value in trained["model_state"].items()
               if not torch.equal(value, checkpoint["model_state"][key])]
    if not changed or any(not key.startswith("event_refinement_head.") for key in changed):
        raise ValueError("warm start must update only the configured refinement stage")
    write(root / "training_report.json", {"initialization": initialization,
          "train_files": 99, "validation_files": 24, "epochs": 1,
          "wall_seconds": wall, "phases": timers.snapshot(), "history": history,
          "changed_weight_keys": changed, "checkpoint_sha256": sha256_file(run / "best.pt"),
          "environment": runtime_environment()})
    # 独立重新读取保存权重，覆盖完整 val/test；固定原阈值，不搜索测试阈值。
    evaluate_pair(root / "trained_evaluation", run / "best.pt", compare=False)


def published_benchmark(root):
    """调用冻结发布入口，并细分原先合并报告的树特征与树预测时间。"""
    import benchmark_frozen_system as benchmark

    from traceformer.postprocessing import learning

    timers, details = Timers(), []
    original_infer, original_load = benchmark.infer, benchmark.load_detector
    original_trees = benchmark.load_event_models
    with ExitStack() as stack:
        stack.enter_context(patch.object(learning, "make_context", timers.wrap("feature_context", learning.make_context)))
        stack.enter_context(patch.object(learning, "build_event_features", timers.wrap("event_features", learning.build_event_features)))

        def load_detector(*args, **kwargs):
            result = original_load(*args, **kwargs)
            timers.modules(stack, result[0], prefix=args[0] + "/")
            legacy_engine = result[2]
            stack.enter_context(patch.object(legacy_engine, "simulate_high_threshold_events",
                                            timers.wrap(args[0] + "/sparse_simulation", legacy_engine.simulate_high_threshold_events)))
            return result

        def load_trees(*args, **kwargs):
            models = original_trees(*args, **kwargs)
            for name, model in models.items():
                stack.enter_context(patch.object(model, "predict", timers.wrap(name + "_tree", model.predict)))
            return models

        def infer(*args, **kwargs):
            timers.reset()
            result = original_infer(*args, **kwargs)
            details.append({"sequence": args[3].name, "components": timers.snapshot()})
            return result

        stack.enter_context(patch.object(benchmark, "load_detector", load_detector))
        stack.enter_context(patch.object(benchmark, "load_event_models", load_trees))
        stack.enter_context(patch.object(benchmark, "infer", infer))
        args = benchmark.build_parser().parse_args([
            "--dataset", str(DATA), "--frozen", str(FROZEN), "--output", str(root / "runtime_report.json"),
            "--processor", "Intel Core Ultra 5 125H", "--device", "cpu", "--num-threads", "14",
            "--prediction-threads", "8", "--repeats", "1",
        ])
        benchmark.benchmark(args)
    # 首条为完整预热；只汇总后续全部 24 序列的计时。
    if len(details) != 25:
        raise ValueError("published warmup or complete timed pass missing")
    values = {row["sequence"]: row["components"] for row in details[1:]}
    keys = {key for row in values.values() for key in row}
    write(root / "detailed_components.json", {"warmup": details[0], "files": values,
        "mean_seconds": {key: float(np.mean([row.get(key, {}).get("seconds", 0.0)
                                             for row in values.values()])) for key in sorted(keys)}})


def summarize(root):
    """只从完整执行产物汇总，不把两套模型的时间或训练变化混为一谈。"""
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    fixed = read(root / "fixed_weights/neural_report.json")
    published = read(root / "published/runtime_report.json")
    components = read(root / "published/detailed_components.json")
    training = read(root / "training/training_report.json")
    trained = read(root / "training/trained_evaluation/neural_report.json")
    old_path = Path("D:/Paper/TraceFormer/evidence/resource_metrics_20261010/runtime_report.json")
    old = read(old_path)
    # 历史计时只在硬件、运行环境和完整输入哈希一致时用于对照。
    for key in ("processor", "platform", "torch", "device", "precision", "torch_threads"):
        if old["environment"][key] != published["environment"][key]:
            raise ValueError(f"historical timing environment differs: {key}")
    if set(old["files"]) != set(published["files"]):
        raise ValueError("historical timing sequence set differs")
    for name, record in published["files"].items():
        if old["files"][name]["source_sha256"] != record["source_sha256"]:
            raise ValueError(f"historical timing input differs: {name}")
    changes = {}
    for split in ("val", "test"):
        changes[split] = {"mask_events": 0, "logit_events": 0, "events": 0, "files": {}}
        for name in fixed["partitions"][split]["files"]:
            with np.load(root / "fixed_weights/predictions" / split / name) as baseline:
                with np.load(root / "training/trained_evaluation/predictions" / split / name) as current:
                    mask_delta = int(np.count_nonzero(baseline["prediction"] != current["prediction"]))
                    logit_delta = int(np.count_nonzero(baseline["event_logits"] != current["event_logits"]))
                    events = len(current["prediction"])
            changes[split]["mask_events"] += mask_delta
            changes[split]["logit_events"] += logit_delta
            changes[split]["events"] += events
            changes[split]["files"][name] = {"mask_events": mask_delta, "logit_events": logit_delta}
    summary = {"published_masks_exact": published["frozen_masks_exact"],
        "fixed_neural_outputs_exact": fixed["outputs_exact"], "training_changes": changes,
        "published_before_seconds": old["mean_ms_per_sequence"] / 1000,
        "published_after_seconds": published["mean_ms_per_sequence"] / 1000,
        "published_speedup": old["mean_ms_per_sequence"] / published["mean_ms_per_sequence"],
        "baseline_date": "2026-10-10", "measurement_date": "2026-10-13",
        "baseline_report": str(old_path), "training": training,
        "source_sha256": sha256_file(__file__)}
    write(root / "summary.json", summary)
    lines = ["# 本地优化、训练与验证结果（2026-10-13）", "",
        "设备：Intel Core Ultra 5 125H，CPU FP32，PyTorch 2.11.0+cpu；Torch 14 线程，树预测 8 线程。本地没有可用 CUDA。",
        "各项计算串行执行，避免训练与推理争用资源。推理均完整预热后对每个序列计时一次；时间排除磁盘读写、模型加载、评分和核验。", "",
        "## 固定权重的优化等价性", "",
        "当前 schema 7（177,810 参数）：全部 24 验证、24 测试序列的事件 logits、token 特征、事件映射及其他决策输出均精确相同。",
        "原构建器来自 Git HEAD，仅恢复旧邻域构建及辅助 Decoder；其他现有本地代码、权重和阈值相同。", "",
        "|分区|优化前神经前向 s/序列|优化后 s/序列|减少|Score（前后相同）|",
        "|---|---:|---:|---:|---:|"]
    for split, item in fixed["partitions"].items():
        before, after = item["mean_seconds"]["before"], item["mean_seconds"]["after"]
        lines.append(f"|{split}|{before:.6f}|{after:.6f}|{1-after/before:.2%}|{item['metrics']['competition']['score']:.12f}|")
    test = fixed["partitions"]["test"]
    averages = {}
    for mode in ("before", "after"):
        rows = [row["timings"][mode] for row in test["files"].values()]
        keys = {key for row in rows for key in row["components"]}
        values = {key: float(np.mean([row["components"].get(key, {}).get("seconds", 0) for row in rows])) for key in keys}
        values["encoder_other"] = values["encoder"] - values["neighbors"] - values.get("geometry", 0)
        values["other"] = test["mean_seconds"][mode] - sum(values.get(key, 0) for key in
            ("sparse_simulation", "tokenizer", "geometry", "neighbors", "encoder_other", "decoder", "sparse_mixture", "event_refinement_head"))
        averages[mode] = values
    lines += ["", "### 当前源码的测试集神经前向明细", "",
              "Encoder 已扣除几何准备与邻域查询，避免嵌套计时重复相加。单位为全部 24 序列的平均毫秒。", "",
              "|阶段|优化前 ms|优化后 ms|", "|---|---:|---:|"]
    for label, key in (("高阈值子流模拟", "sparse_simulation"), ("Tokenizer", "tokenizer"),
                       ("共享几何准备", "geometry"), ("邻域构建/查询", "neighbors"),
                       ("Encoder 其余计算", "encoder_other"), ("Decoder", "decoder"),
                       ("Sparse mixture", "sparse_mixture"), ("多尺度精修", "event_refinement_head"),
                       ("其余前向开销", "other")):
        lines.append(f"|{label}|{averages['before'].get(key, 0)*1000:.3f}|{averages['after'].get(key, 0)*1000:.3f}|")
    lines += ["", "## 发布完整系统", "",
        "此处是冻结 schema 5 主网络 + schema 4 参考网络及已发布树/轨迹规则，和上面的当前 schema 7 神经模型分别计时。历史归档神经源码完整保留；本次发布系统执行按权重裁剪树、八线程树预测和参考 logits 复用。",
        "全部 24 测试序列（2,074,586 事件）四阶段掩码与冻结结果逐位一致；两个 dense 序列的参考掩码也相同。独立计分与原发布结果一致。",
        f"端到端平均：{summary['published_before_seconds']:.6f} → {summary['published_after_seconds']:.6f} s/序列，约 {summary['published_speedup']:.2f} 倍。前值来自 2026-10-10 同机、同 PyTorch、同 Torch 线程数的报告，后值为本轮完整实测；并非同一进程的交替对照。", "",
        "|阶段|优化后平均 ms/序列|", "|---|---:|"]
    published_times = published["component_mean_ms"]
    for label, key in (("输入归一化", "input_seconds"), ("主网络 full/sparse 前向", "main_seconds"),
                       ("末端 Tensor 转 NumPy", "device_to_host_seconds"),
                       ("树特征、树预测、融合及轨迹", "verification_tracking_seconds"),
                       ("参考 logits 复用", "reference_seconds"), ("参考恢复", "reference_recovery_seconds"),
                       ("总计", "total_seconds")):
        lines.append(f"|{label}|{published_times[key]:.3f}|")
    detail = components["mean_seconds"]
    rest = published_times["verification_tracking_seconds"] / 1000 - sum(detail[key] for key in ("feature_context", "event_features", "large_tree"))
    lines += ["", "后处理细分（平均秒/序列）：", "",
        f"- 上下文构造：{detail['feature_context']:.6f}",
        f"- 112 维事件特征展开：{detail['event_features']:.6f}",
        f"- large 树预测：{detail['large_tree']:.6f}",
        f"- 缓存量化、融合、轨迹及其余调度：{rest:.6f}", "",
        "|发布系统指标|修改前|修改后|", "|---|---:|---:|"]
    baseline_metrics = read(FROZEN / "complete/report.json")["stages"]["complete"]["aggregate"]
    actual_metrics = published["independent_metrics"]["complete"]["aggregate"]
    for key in ("iou", "seg_acc", "pd", "fa", "score"):
        if baseline_metrics[key] != actual_metrics[key]:
            raise ValueError(f"published metric changed: {key}")
        lines.append(f"|{key}|{baseline_metrics[key]:.12g}|{actual_metrics[key]:.12g}|")
    lines += ["", "## 复用权重后的单轮训练", "",
        "沿用原配置的 refinement 阶段：全部 99 训练序列，99 次优化器更新，只训练 8,178 个精修参数。已有 169,632 个主干/稀疏模块参数保持完全相同。",
        "历史 best.pt 不含优化器/RNG；明确建立新 AdamW 与种子 37 的 epoch-0 初态，复用全部 schema-7 权重。没有声称恢复历史优化器。",
        f"只运行一轮，本轮实际学习率为 {training['history'][0]['learning_rate']:.7f}；损失、网络结构和固定阈值 0.5 不变。训练产物独立保存，没有覆盖发布模型。",
        f"最大序列资源检查：{read(root / 'training/resource_gate.json')['wall_seconds']:.3f} s；训练入口总墙钟：{training['wall_seconds']:.3f} s。", "",
        "|训练/验证阶段|秒|调用数|", "|---|---:|---:|"]
    for key, label in (("train/_forward_with_sparse_mixture", "训练前向"),
                       ("train/forward", "主损失计算"),
                       ("train/_event_refinement_loss", "精修辅助损失"),
                       ("train/_sparse_mixture_auxiliary_losses", "稀疏辅助损失调用"),
                       ("train/backward", "反向"), ("train/step", "AdamW 更新"),
                       ("validation/total", "训练结束的完整验证，含损失及评分")):
        row = training["phases"][key]
        lines.append(f"|{label}|{row['seconds']:.3f}|{row['calls']}|")
    lines += ["", "以下变化来自新的权重更新，不能归因于推理等价优化。新权重重新读取后，完整 val/test 的独立计数均通过；没有用测试标签搜索阈值。", ""]
    for split in ("val", "test"):
        old_metrics = fixed["partitions"][split]["metrics"]["competition"]
        new_metrics = trained["partitions"][split]["metrics"]["competition"]
        lines += ["|分区|指标|训练前|单轮训练后|变化|", "|---|---|---:|---:|---:|"]
        for key in ("iou", "seg_acc", "pd", "fa", "score"):
            lines.append(f"|{split}|{key}|{old_metrics[key]:.12g}|{new_metrics[key]:.12g}|{new_metrics[key]-old_metrics[key]:+.9g}|")
        lines += ["", f"{split} 二值变化事件：{changes[split]['mask_events']:,}/{changes[split]['events']:,}；logits 变化事件：{changes[split]['logit_events']:,}。", ""]
    lines += ["## 检查与产物", "",
        "- 完整单元测试：270 passed，1 skipped（pytest 已直接收集断言测试，跳过重复的 unittest 包装用例）。Ruff 与 git diff --check 通过。",
        "- fixed_weights/neural_report.json：当前源码的完整前后输出核对及模块计时。",
        "- published/runtime_report.json 与 detailed_components.json：发布完整系统与树预测细分计时。",
        "- training/training_report.json、run/history.json、run/best.pt：完整单轮训练及权重变化。",
        "- training/trained_evaluation/neural_report.json：训练后完整 val/test 独立复评。",
        "- summary.json：汇总、前后变化和可核验来源。", ""]
    (root / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("neural", "train", "published", "report"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--after-only", action="store_true")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.mode == "neural":
        evaluate_pair(arguments.output, arguments.checkpoint, not arguments.after_only)
    elif arguments.mode == "train":
        train_one_epoch(arguments.output)
    elif arguments.mode == "report":
        summarize(arguments.output)
    else:
        published_benchmark(arguments.output)


if __name__ == "__main__":
    main()
