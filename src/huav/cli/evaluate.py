"""Command-line checkpoint evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huav.artifacts import (
    rebuild_output_index,
    record_evaluation,
    resolve_evaluation_output,
)
from huav.config import load_config
from huav.training.engine import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="huav-evaluate", description="Evaluate a HUAV checkpoint"
    )
    parser.add_argument(
        "--config",
        default="configs/huav_base.yaml",
        help="path to the YAML configuration",
    )
    parser.add_argument("--checkpoint", required=True, help="path to best.pt")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu", help="evaluation device"
    )
    parser.add_argument("--output", help="optional metrics JSON path")
    parser.add_argument(
        "--quiet", action="store_true", help="hide per-file progress messages"
    )
    arguments = parser.parse_args()
    try:
        config = load_config(arguments.config)
        metrics = evaluate_checkpoint(
            config,
            arguments.checkpoint,
            device=arguments.device,
            progress=not arguments.quiet,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    try:
        checkpoint_path = Path(arguments.checkpoint)
        output_path = resolve_evaluation_output(checkpoint_path, arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        record_evaluation(checkpoint_path.parent, output_path)
        rebuild_output_index(checkpoint_path.parent.parent)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(f"evaluation succeeded but artifact update failed: {error}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
