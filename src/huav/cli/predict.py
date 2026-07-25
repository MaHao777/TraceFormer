"""Command-line entry point for generating HUAV competition submission files."""

from __future__ import annotations

import argparse

from huav.config import load_config
from huav.training.engine import generate_submission_package


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="huav-predict",
        description="Generate competition submission TXT files and ZIP from a trained checkpoint",
    )
    parser.add_argument(
        "--config",
        default="configs/huav_base.yaml",
        help="path to the YAML configuration (default: configs/huav_base.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="path to the best.pt checkpoint file",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/submission",
        help="output directory for TXT files and ZIP (default: outputs/submission)",
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
    arguments = parser.parse_args()
    try:
        zip_path = generate_submission_package(
            load_config(arguments.config),
            arguments.checkpoint,
            arguments.output_dir,
            device=arguments.device,
            progress=not arguments.quiet,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"submission ZIP: {zip_path}")
    print(f"     TXT files: {zip_path.parent}")


if __name__ == "__main__":
    main()
