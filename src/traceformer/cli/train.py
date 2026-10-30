"""Command-line entry point for spatiotemporal-cone TraceFormer training."""

from __future__ import annotations

import argparse

from traceformer.config import load_config
from traceformer.training.engine import run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="traceformer-train",
        description="Train and validate the TraceFormer spatiotemporal-cone event model",
    )
    parser.add_argument(
        "--config",
        default="configs/traceformer_score_optimized.yaml",
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
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume-state",
        type=str,
        default=None,
        metavar="TRAINING_STATE_PT",
        help="branch from an exact optimizer/RNG training state",
    )
    resume_group.add_argument(
        "--initialize-refinement",
        type=str,
        default=None,
        metavar="BEST_PT",
        help="initialize matching three-cone base weights and train only event refinement",
    )
    arguments = parser.parse_args()
    try:
        config = load_config(arguments.config)
        summary = run_training(
            config,
            output_dir=arguments.output_dir,
            device=arguments.device,
            progress=not arguments.quiet,
            note=arguments.note,
            config_source=arguments.config,
            resume_state_path=arguments.resume_state,
            initial_refinement_checkpoint_path=arguments.initialize_refinement,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"run artifacts: {summary['run_dir']}")


if __name__ == "__main__":
    main()
