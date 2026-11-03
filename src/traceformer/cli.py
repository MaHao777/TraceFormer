"""预处理、训练、评估和推理的轻量命令入口。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from traceformer.config import load_config
from traceformer.data import NPZEventDataset
from traceformer.inference import run_predictions
from traceformer.training.engine import _dataset_manifest, _run_training_impl, _write_json
from traceformer.variants import VARIANTS, PaperSpec


def _parser(action):
    parser = argparse.ArgumentParser(prog=f"traceformer-{action}")
    parser.add_argument("--config", type=Path, default=Path("configs/s1.yaml"))
    parser.add_argument("--data-root", type=Path, help="directory containing train/val/test")
    parser.add_argument("--num-threads", type=int)
    if action != "preprocess":
        parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser


def _config(args):
    config = load_config(args.config)
    if args.data_root is not None:
        config = replace(config, dataset=replace(config.dataset, root=args.data_root.resolve()))
    changes = {}
    for name in ("device", "num_threads", "epochs", "max_train_files", "max_val_files"):
        if getattr(args, name, None) is not None:
            changes[name] = getattr(args, name)
    config = replace(config, training=replace(config.training, **changes)).validate()
    if config.training.stage != "full" or config.training.mixed_precision:
        raise ValueError("paper protocol requires full joint training and FP32 (AMP disabled)")
    return config


def _execute(parser, operation):
    args = parser.parse_args()
    try:
        report = operation(args, _config(args))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def preprocess():
    parser = _parser("preprocess")
    parser.add_argument("--split", choices=("train", "val", "test"), nargs="+", default=["train", "val"])
    parser.add_argument("--unlabeled", action="store_true", help="validate x/y/t/p only")
    parser.add_argument("--output", type=Path, default=Path("outputs/preprocess_manifest.json"))

    def operation(args, config):
        datasets, statistics = [], {}
        for split in args.split:
            dataset = NPZEventDataset(Path(config.dataset.root) / split,
                                      width=config.resolution.width, height=config.resolution.height,
                                      duration_ms=config.dataset.duration_ms,
                                      require_annotations=not args.unlabeled)
            # 与训练共享读取及在线预处理，不创建会改变事件顺序的中间数据集。
            statistics[split] = {"files": len(dataset), "events": sum(item.length for item in dataset)}
            datasets.append(dataset)
        report = {"duration_ms": config.dataset.duration_ms, "statistics": statistics,
                  "annotations_required": not args.unlabeled, "dataset": _dataset_manifest(*datasets)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.output, report)
        return {"manifest": str(args.output.resolve()), "statistics": statistics}

    _execute(parser, operation)


def train():
    parser = _parser("train")
    parser.add_argument("--variant", choices=VARIANTS, default="S1")
    parser.add_argument("--ablations-config", type=Path, default=Path("configs/ablations.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-files", type=int)
    parser.add_argument("--max-val-files", type=int)

    def operation(args, config):
        suite = yaml.safe_load(args.ablations_config.read_text(encoding="utf-8"))
        spec = PaperSpec(**suite["variants"][args.variant]["contract"])
        if spec.variant != args.variant:
            raise ValueError("variant configuration does not match requested variant")
        if config.training.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        args.output_dir.mkdir(parents=True, exist_ok=False)
        # 与论文预检一致：完整查询支持域，仅执行分块以控制内存。
        return _run_training_impl(config, output_dir=args.output_dir,
                                  device=config.training.device, progress=True,
                                  resume_state_path=args.resume_state, ablation=spec)

    _execute(parser, operation)


def evaluate():
    parser = _parser("evaluate")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/s1_best.pt"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, help="independently rescore saved NPZ predictions")

    def operation(args, config):
        return run_predictions(config, args.checkpoint, Path(config.dataset.root) / args.split,
                               args.output_dir, device=config.training.device, evaluate=True,
                               split=args.split, predictions_dir=args.predictions_dir)

    _execute(parser, operation)


def predict():
    parser = _parser("predict")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/s1_best.pt"))
    parser.add_argument("--input-dir", type=Path, required=True, help="NPZ directory; labels are optional")
    parser.add_argument("--output-dir", type=Path, required=True)

    def operation(args, config):
        return run_predictions(config, args.checkpoint, args.input_dir,
                               args.output_dir, device=config.training.device)

    _execute(parser, operation)
