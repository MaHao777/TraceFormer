"""Command-line entry point for spatiotemporal-cone HUAV training."""

from __future__ import annotations

import argparse

from huav.config import load_config
from huav.training.engine import resume_training, run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="huav-train",
        description="Train and validate the HUAV spatiotemporal-cone event model",
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
        help="exact run directory; defaults to outputs/<device>_<timestamp>",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="hide per-file progress messages"
    )
    parser.add_argument(
        "--note",
        help="short description of the main change in this experiment",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="BEST_PT",
        help="resume training from a best.pt checkpoint",
    )
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=None,
        metavar="N",
        help="number of extra epochs when resuming (config.epochs is ignored)",
    )
    arguments = parser.parse_args()
    try:
        config = load_config(arguments.config)
        if arguments.resume:
            additional_epochs = (
                arguments.additional_epochs
                if arguments.additional_epochs is not None
                else config.training.epochs
            )
            summary = resume_training(
                config,
                arguments.resume,
                additional_epochs=additional_epochs,
                output_dir=arguments.output_dir,
                device=arguments.device,
                progress=not arguments.quiet,
                note=arguments.note,
                config_source=arguments.config,
            )
        else:
            summary = run_training(
                config,
                output_dir=arguments.output_dir,
                device=arguments.device,
                progress=not arguments.quiet,
                note=arguments.note,
                config_source=arguments.config,
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"run artifacts: {summary['run_dir']}")


if __name__ == "__main__":
    main()
