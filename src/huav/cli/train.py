"""Command-line entry point for dual-view HUAV training."""

from __future__ import annotations

import argparse

from huav.config import load_config
from huav.training.engine import run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="huav-train", description="Train and validate the HUAV dual-view event model"
    )
    parser.add_argument(
        "--config",
        default="configs/huav_base.yaml",
        help="path to the YAML configuration",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), help="override the configured device"
    )
    parser.add_argument(
        "--output-dir",
        help="exact run directory; defaults to outputs/cpu_<timestamp>",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="hide per-file progress messages"
    )
    arguments = parser.parse_args()
    try:
        config = load_config(arguments.config)
        summary = run_training(
            config,
            output_dir=arguments.output_dir,
            device=arguments.device,
            progress=not arguments.quiet,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"run artifacts: {summary['run_dir']}")


if __name__ == "__main__":
    main()
