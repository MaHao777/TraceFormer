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
        help="exact run directory; defaults to outputs/cpu_<timestamp>",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="hide per-file progress messages"
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="BEST_PT",
        help="resume training from a best.pt checkpoint",
    )
    checkpoint_group.add_argument(
        "--initialize-detector",
        type=str,
        default=None,
        metavar="BEST_PT",
        help="initialize detector weights without resuming training",
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
        if arguments.additional_epochs is not None and not arguments.resume:
            parser.error("--additional-epochs is only valid with --resume")
        if (
            arguments.additional_epochs is not None
            and arguments.additional_epochs <= 0
        ):
            parser.error("--additional-epochs must be positive")
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
            )
        else:
            summary = run_training(
                config,
                output_dir=arguments.output_dir,
                device=arguments.device,
                progress=not arguments.quiet,
                initialize_detector=arguments.initialize_detector,
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"run artifacts: {summary['run_dir']}")


if __name__ == "__main__":
    main()
