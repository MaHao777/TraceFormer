"""当前TraceFormer模型后处理级联命令行。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from traceformer.config import load_config
from traceformer.postprocessing.cache import cache_split
from traceformer.postprocessing.contracts import PostprocessProtocol
from traceformer.postprocessing.direct_verifier import train_direct_verifier
from traceformer.postprocessing.pipeline import (
    predict_pipeline,
    release_event_models,
    train_pipeline,
)
from traceformer.postprocessing.submission import audit_submission, evaluate_predictions


def _protocol(arguments: argparse.Namespace) -> PostprocessProtocol:
    return PostprocessProtocol(
        seed=37,
        holdout_files=arguments.holdout_files,
        folds=5,
        small_rounds=arguments.small_rounds,
        large_rounds=arguments.large_rounds,
        blend_step=0.05,
    )


def _add_training_controls(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--holdout-files", type=int, default=20)
    parser.add_argument("--small-rounds", type=int, default=997)
    parser.add_argument("--large-rounds", type=int, default=1998)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traceformer-postprocess",
        description="缓存、训练、盲推、复评并审计当前TraceFormer模型的后处理级联",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    cache = commands.add_parser("cache", help="生成当前模型逐事件特征缓存")
    cache.add_argument("--config", required=True)
    cache.add_argument("--checkpoint", required=True)
    cache.add_argument("--output-dir", required=True)
    cache.add_argument("--split", choices=("train", "val"), required=True)
    cache.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    cache.add_argument("--quiet", action="store_true")

    train = commands.add_parser("train", help="按79/20协议训练并执行分阶段门槛")
    train.add_argument("--cache-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--control-dir")
    _add_training_controls(train)

    release_event = commands.add_parser(
        "release-event",
        help="冻结开发参数并用全部训练缓存发布事件模型",
    )
    release_event.add_argument("--cache-dir", required=True)
    release_event.add_argument("--training-report", required=True)
    release_event.add_argument("--output-dir", required=True)

    direct_verifier = commands.add_parser(
        "train-direct-verifier",
        help="训练只审查direct-only组件的时序证据门",
    )
    direct_verifier.add_argument("--cache-dir", required=True)
    direct_verifier.add_argument("--training-report", required=True)
    direct_verifier.add_argument("--output-dir", required=True)

    predict = commands.add_parser("predict", help="使用冻结发布级联进行盲推")
    predict.add_argument("--cache-dir", required=True)
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--output-dir", required=True)
    predict.add_argument("--quiet", action="store_true")

    evaluate = commands.add_parser("evaluate", help="从原始NPZ独立复评二值结果")
    evaluate.add_argument("--prediction-dir", required=True)
    evaluate.add_argument("--dataset-root", required=True)
    evaluate.add_argument("--output", required=True)

    audit = commands.add_parser("audit", help="写出TXT、生成平铺ZIP并逐行审计")
    audit.add_argument("--prediction-dir", required=True)
    audit.add_argument("--dataset-root", required=True)
    audit.add_argument("--zip", required=True)
    audit.add_argument("--expected-files", type=int, default=24)
    audit.add_argument("--expected-rows", type=int, default=1_424_330)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        if arguments.command == "cache":
            output_root = Path(arguments.output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            copied_config = output_root / "config.yaml"
            source_config = Path(arguments.config).resolve()
            if copied_config.exists():
                if copied_config.read_bytes() != source_config.read_bytes():
                    raise ValueError("output cache already contains a different config.yaml")
            else:
                shutil.copy2(source_config, copied_config)
            result = cache_split(
                load_config(source_config),
                arguments.checkpoint,
                output_root,
                split=arguments.split,
                device=arguments.device,
                progress=not arguments.quiet,
            )
            payload = {"split": result.split, "files": len(result.files)}
        elif arguments.command == "train":
            payload = train_pipeline(
                arguments.cache_dir,
                arguments.output_dir,
                _protocol(arguments),
                control_dir=arguments.control_dir,
                progress=lambda message: print(message, flush=True),
            )
        elif arguments.command == "release-event":
            payload = release_event_models(
                arguments.cache_dir,
                arguments.training_report,
                arguments.output_dir,
                progress=lambda message: print(message, flush=True),
            )
        elif arguments.command == "train-direct-verifier":
            payload = train_direct_verifier(
                arguments.cache_dir,
                arguments.training_report,
                arguments.output_dir,
                progress=lambda message: print(message, flush=True),
            )
        elif arguments.command == "predict":
            sink = (lambda _message: None) if arguments.quiet else print
            payload = predict_pipeline(
                arguments.cache_dir,
                arguments.model_dir,
                arguments.output_dir,
                progress=sink,
            )
        elif arguments.command == "evaluate":
            payload = evaluate_predictions(arguments.prediction_dir, arguments.dataset_root)
            output_path = Path(arguments.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            payload = audit_submission(
                arguments.prediction_dir,
                arguments.dataset_root,
                arguments.zip,
                expected_files=arguments.expected_files,
                expected_rows=arguments.expected_rows,
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
