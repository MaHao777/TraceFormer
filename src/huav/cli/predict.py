"""Command-line entry point for generating HUAV competition submission files."""

from __future__ import annotations

import argparse
from pathlib import Path

from huav.artifacts import (
    rebuild_output_index,
    record_submission,
    resolve_submission_output,
)
from huav.config import load_config
from huav.training.engine import generate_submission_package


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="huav-predict",
        description="Generate competition submission TXT files and ZIP from a trained checkpoint",
    )
    parser.add_argument(
        "--config",
        default="configs/huav_score_optimized.yaml",
        help="path to the fixed schema-7 YAML configuration",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="path to the best.pt checkpoint file",
    )
    parser.add_argument(
        "--output-dir",
        help="output directory for TXT files and ZIP; defaults to CHECKPOINT_DIR/submission",
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
    checkpoint_path = Path(arguments.checkpoint)
    output_dir = resolve_submission_output(checkpoint_path, arguments.output_dir)
    try:
        zip_path = generate_submission_package(
            load_config(arguments.config),
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
