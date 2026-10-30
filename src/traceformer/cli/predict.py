"""Command-line entry point for generating TraceFormer competition submission files."""

from __future__ import annotations

import argparse
from pathlib import Path

from traceformer.artifacts import (
    rebuild_output_index,
    record_submission,
    resolve_submission_output,
)
from traceformer.config import load_config
from traceformer.postprocessing.learning import positive_prediction_threads
from traceformer.primary import DEFAULT_MAIN_MANIFEST, predict_primary_model
from traceformer.training.engine import generate_submission_package


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="traceformer-predict",
        description="默认使用冻结主模型及完整后处理；显式checkpoint可预测单网络",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to the fixed schema-7 YAML configuration",
    )
    parser.add_argument(
        "--checkpoint",
        help="显式指定单网络best.pt；省略时使用主模型与完整后处理",
    )
    parser.add_argument(
        "--output-dir",
        help="主模型默认按时间新建输出目录；单网络默认CHECKPOINT_DIR/submission",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="hide per-file progress messages",
    )
    parser.add_argument("--main-manifest", type=Path, default=DEFAULT_MAIN_MANIFEST,
                        help="冻结主模型清单，含权重、后处理及用户选择记录")
    parser.add_argument("--input-dir", type=Path, help="主模型输入NPZ目录")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--predictions-only", action="store_true", help="只保存NPZ掩码，不导出TXT/ZIP")
    parser.add_argument("--prediction-threads", type=positive_prediction_threads, default=8)
    parser.add_argument("--num-threads", type=positive_prediction_threads,
                        help="主模型Torch运行线程，默认采用清单中的14线程")
    arguments = parser.parse_args()
    if arguments.checkpoint is None:
        if arguments.config is not None:
            parser.error("--checkpoint is required with --config for single-network prediction")
        try:
            report = predict_primary_model(arguments.output_dir, manifest_path=arguments.main_manifest,
                input_dir=arguments.input_dir, split=arguments.split, device=arguments.device,
                prediction_threads=arguments.prediction_threads,
                num_threads=arguments.num_threads,
                export_submission=not arguments.predictions_only,
                progress=(lambda _: None) if arguments.quiet else print)
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(f"primary model predictions: {report['output_dir']}")
        if "submission_zip" in report:
            print(f"submission ZIP: {report['submission_zip']}")
        return
    if arguments.input_dir is not None or arguments.predictions_only:
        parser.error("--input-dir and --predictions-only apply to the complete primary model")
    checkpoint_path = Path(arguments.checkpoint)
    output_dir = resolve_submission_output(checkpoint_path, arguments.output_dir)
    try:
        zip_path = generate_submission_package(
            load_config(arguments.config or "configs/traceformer_score_optimized.yaml"),
            checkpoint_path,
            output_dir,
            device=arguments.device,
            progress=not arguments.quiet,
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    try:
        record_submission(checkpoint_path.parent, zip_path)
        rebuild_output_index(checkpoint_path.parent.parent)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(
            f"submission generation succeeded but artifact update failed: {error}"
        )
    print(f"submission ZIP: {zip_path}")
    print(f"     TXT files: {zip_path.parent}")


if __name__ == "__main__":
    main()
