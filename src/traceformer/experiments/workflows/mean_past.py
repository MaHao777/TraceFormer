"""M1+C4 均值过去锥：预检、训练、独立复评与汇总。"""

from __future__ import annotations

import copy
import gc
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import psutil
import torch
import yaml

from traceformer.config import load_config
from traceformer.experiments.ablation import ComponentCoreSpec
from traceformer.experiments.component_suite import production_loss, read_json
from traceformer.experiments.runtime import ExperimentRuntime, setup_device
from traceformer.training.engine import (
    _atomic_write_text,
    _dataset,
    _file_sha256,
    _serializable,
    _write_json,
    build_model,
)
from traceformer.training.losses import EventFocalDiceLoss

SPEC = ComponentCoreSpec("M1_C4")


class MeanPastWorkflow(ExperimentRuntime):
    """单组流程保留原预检与精度选项。"""

    experiment = "mean_past_gated"

    def configure(self, args, root, package_digest):
        prior = read_json(root / "experiment.json") if (root / "experiment.json").exists() else None
        saved = prior["config"] if prior else None
        data_root = (args.data_root or (Path(saved["dataset"]["root"]) if saved else None))
        if data_root is None:
            raise ValueError("first run requires --data-root containing train/ and val/")
        data_root = data_root.resolve()
        test_root = (args.test_root or (Path(prior["test_root"]) if prior else data_root)).resolve()
        precision = args.precision or (prior["precision"] if prior else "fp32")
        device = args.device or (saved["training"]["device"] if saved else "cuda")
        base = load_config(self.bundle / "config.yaml")
        config = replace(base, dataset=replace(base.dataset, root=data_root), training=replace(
            base.training, device=device, output_root=root, mixed_precision=precision == "bf16",
            mixed_precision_dtype="bfloat16", epochs=1 if args.smoke else 24,
            max_train_files=2 if args.smoke else None, max_val_files=2 if args.smoke else None,
            save_training_state_interval=1,
        )).validate()
        contract = {"ablation": SPEC.to_dict(), "config": _serializable(asdict(config)),
                    "test_root": str(test_root), "precision": precision,
                    "smoke": args.smoke, "package_sha256": package_digest}
        # 仅训练时可稍后提供测试根目录，但测试文件身份仍必须与打包清单相同。
        comparable = ({**prior, "test_root": contract["test_root"]}
                      if prior and not (root / "test/report.json").exists() else prior)
        if prior is not None and comparable != contract:
            raise ValueError("experiment settings/source changed; preserve existing output and use the original settings")
        checked = self.verify_data(data_root, test_root, need_test=args.stage in {"all", "predict"})
        _write_json(root / "experiment.json", contract)
        _write_json(root / "data_verification.json", checked)
        _write_json(root / "package_manifest.json", read_json(self.bundle / "package_manifest.json"))
        _write_json(root / "dataset_identity.json", read_json(self.bundle / "dataset_manifest.json"))
        _write_json(root / "ablation.json", SPEC.to_dict())
        _atomic_write_text(root / "config.yaml", yaml.safe_dump(contract["config"], allow_unicode=True, sort_keys=False))
        return config, test_root

    def structure(self, model):
        return {"pooling": "mean(event_encoder(events))", "queries": 0,
                "statistics": "seven statistics concatenated independently",
                "layers": len(model.encoder.layers), "directions": [-1], "pool_weight_gate": True,
                "parameters": sum(p.numel() for p in model.parameters()),
                "windows": [[[w.spatial_radius, w.time_horizon] for w in layer.attention.windows]
                            for layer in model.encoder.layers]}

    def gate_worker(self, args):
        config = load_config(args.config)
        runtime = setup_device(config)
        device = runtime.device
        torch.manual_seed(config.training.seed)
        model = build_model(config, ablation=SPEC).to(device).train()
        numerical = self.numerical_check(model, device)
        criterion = EventFocalDiceLoss(**asdict(config.loss))
        dataset = _dataset(config, "train")
        # smoke检查首条数据；正式运行检查两条最大序列的完整前向与反向。
        names = [dataset.files[0].name] if args.smoke else ["train_094.npz", "train_096.npz"]
        rows = []
        for name in names:
            sequence = dataset[next(i for i, p in enumerate(dataset.files) if p.name == name)]
            sequence = replace(sequence, **{key: getattr(sequence, key).to(device)
                               for key in ("events", "labels", "object_ids", "timestamps_ms")})
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=config.training.mixed_precision):
                loss = production_loss(model, sequence, config, criterion)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite gate loss")
            loss.backward()
            for key, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError(f"missing/nonfinite gradient: {key}")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_reserved(device)
                budget = torch.cuda.get_device_properties(device).total_memory * 0.9
            else:
                memory = psutil.Process().memory_info()
                peak = getattr(memory, "peak_wset", memory.rss)
                budget = psutil.virtual_memory().total * 0.7
            if peak > budget:
                raise ValueError("resource gate exceeds device memory budget")
            rows.append({"file": name, "events": sequence.length, "loss": float(loss.detach()),
                         "seconds": time.perf_counter() - started, "peak_memory_bytes": peak,
                         "memory_budget_bytes": int(budget), "finite_gradients": True})
            print(f"resource gate passed: {name}", flush=True)
        _write_json(args.root / "preflight.json", {"passed": True, "runtime": runtime.metadata,
                    "structure": self.structure(model), "numerical_check": numerical,
                    "sequences": rows, "smoke": args.smoke})

    def numerical_check(self, model, device):
        """用同一参数比较CPU与目标设备的小样本输出和梯度。"""
        reference = copy.deepcopy(model).cpu().eval()
        model.eval()
        events = torch.tensor([[8.0 * x + 0.2, 8.0 * y + 0.4, (t + offset) / 64, polarity]
                               for x in range(2) for y in range(2) for t in range(4)
                               for offset, polarity in ((0.1, 0.0), (0.7, 1.0))])
        # 使用包内配置的分辨率；正式实验与本地微型自检均生成合法坐标。
        events[:, 0].clamp_(max=model.tokenizer.width - 1)
        events[:, 1].clamp_(max=model.tokenizer.height - 1)
        actual = model(events.to(device))["event_logits"]
        expected = reference(events)["event_logits"]
        torch.testing.assert_close(actual.cpu(), expected, atol=3e-4, rtol=3e-4)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for (name, left), (_, right) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
            if left.grad is None or right.grad is None:
                raise ValueError(f"numerical check missing gradient: {name}")
            torch.testing.assert_close(left.grad.cpu(), right.grad, atol=5e-4, rtol=5e-3)
        difference = float((actual.cpu() - expected).abs().max().detach())
        model.zero_grad(set_to_none=True)
        model.train()
        return {"passed": True, "target_device": str(device), "real_cuda_checked": device.type == "cuda",
                "max_logit_absolute_difference": difference, "parameter_gradients_checked": True}

    def finish(self, root):
        reports = {s: read_json(root / s / "independent_report.json") for s in ("val", "test")}
        references = read_json(self.bundle / "references.json")
        _write_json(root / "results.json", {"model": "M1_C4", "remote": reports,
                    "local_references": references, "experiment": read_json(root / "experiment.json"),
                    "note": "local references used CPU; remote CUDA/precision may change numerical trajectories"})
        lines = ["# M1纯均值 + 三层门控过去锥", "", "|来源/模型|划分|Score|IoU|SegAcc|Pd|Fa|AP|",
                 "|---|---|---:|---:|---:|---:|---:|---:|"]
        for split, report in reports.items():
            for label, metrics, ap in [("remote/M1_C4", report["competition"], report["average_precision"]),
                                      *[("local/" + name, row[split]["competition"], row[split]["average_precision"])
                                        for name, row in references.items()]]:
                ap_text = f"{ap:.6f}" if ap is not None else "—"
                lines.append(f"|{label}|{split}|{metrics['score']:.6f}|{metrics['iou']:.6f}|"
                             f"{metrics['seg_acc']:.6f}|{metrics['pd']:.6f}|{metrics['fa']:.8g}|{ap_text}|")
        lines += ["", "本地参考来自CPU；单种子、跨设备比较不等于逐位复现。测试固定验证阈值。"]
        _atomic_write_text(root / "results.md", "\n".join(lines) + "\n")
        hashes = {p.relative_to(root).as_posix(): _file_sha256(p) for p in root.rglob("*")
                  if p.is_file() and p.name not in {"completion_audit.json", "workflow.log", "status.json", "run.lock"}}
        _write_json(root / "completion_audit.json", {"complete": True, "sha256": hashes,
                    "partitions": {s: r["sequences"] for s, r in reports.items()}, "model": SPEC.to_dict()})
        _write_json(root / "status.json", {"status": "complete", "model": "M1_C4"})
        self.archive_results(root)

    def worker(self, args):
        if args.worker == "gate":
            self.gate_worker(args)
        else:
            super().worker(args)

    def stages(self, root, config, test_root, args):
        # 预检只在尚未完成时执行；训练完成后只补预测和汇总。
        if args.stage in {"all", "preflight", "train"} and not (root / "preflight.json").exists():
            command = [sys.executable, "-u", str(self.bundle / "run.py"), "--worker", "gate",
                       "--root", str(root), "--config", str(root / "config.yaml")]
            if args.smoke:
                command.append("--smoke")
            self.execute(root, "preflight", command)
        if args.stage == "preflight":
            _write_json(root / "status.json", {"status": "preflight_complete"})
            return
        if args.stage in {"all", "train"}:
            checkpoint = self.training(root, args)
            gc.collect()
        else:
            checkpoint = self.completed_checkpoint(root)
        if args.stage in {"all", "predict"}:
            self.predict_variant(root, config, test_root, checkpoint, args)
            self.finish(root)
        else:
            _write_json(root / "status.json", {"status": "training_complete", "checkpoint": str(checkpoint)})
