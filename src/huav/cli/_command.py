"""Shared CLI behavior for intentionally unavailable Phase 1 operations."""

from __future__ import annotations

import argparse

from huav.config import load_config


def run(command: str) -> None:
    parser = argparse.ArgumentParser(prog=f"huav-{command}", description=f"HUAV {command} (Phase 1 scaffold)")
    parser.add_argument("--config", default="configs/huav_base.yaml", help="path to a validated YAML configuration")
    arguments = parser.parse_args()
    try:
        load_config(arguments.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    raise SystemExit(f"{command}: Phase 1 scaffold only; operation not implemented.")
