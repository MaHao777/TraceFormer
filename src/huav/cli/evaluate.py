"""Command-line checkpoint evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser.add_argument(
        "--reoptimize-threshold",
        action="store_true",
        help="ignore the stored threshold and run configured threshold search",
    )
    arguments = parser.parse_args()
    try:
        config = load_config(arguments.config)
        metrics = evaluate_checkpoint(
            config,
            arguments.checkpoint,
            device=arguments.device,
            progress=not arguments.quiet,
            reoptimize_threshold=arguments.reoptimize_threshold,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    if arguments.output:
        Path(arguments.output).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
