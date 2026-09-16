"""Command-line entry point for rendering TraceFormer training curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from traceformer.visualization import render_training_curves


def _read_history(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("history.json must contain a JSON array")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError("every history entry must be a JSON object")
    return value


def _visualize_run(run_dir: Path) -> Path:
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        raise OSError(f"{history_path} does not exist")
    history = _read_history(history_path)
    return render_training_curves(
        history,
        run_dir / "training_curves.png",
        run_name=run_dir.name,
    )


def _visualize_all(output_root: Path) -> tuple[int, int, int]:
    if not output_root.is_dir():
        raise OSError(f"{output_root} is not a directory")
    generated = 0
    skipped = 0
    failed = 0
    for history_path in sorted(output_root.rglob("history.json")):
        try:
            history = _read_history(history_path)
            if not history:
                skipped += 1
                continue
            render_training_curves(
                history,
                history_path.parent / "training_curves.png",
                run_name=history_path.parent.name,
            )
            generated += 1
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failed += 1
            print(f"failed: {history_path}: {error}", file=sys.stderr)
    return generated, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="traceformer-visualize",
        description="Render TraceFormer training history as a PNG dashboard",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=Path, help="one run directory to render")
    mode.add_argument("--all", action="store_true", help="render every history.json")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="root scanned by --all (default: outputs)",
    )
    arguments = parser.parse_args()

    if arguments.run_dir is not None:
        try:
            output = _visualize_run(arguments.run_dir)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        print(f"training curves: {output.resolve()}")
        return

    try:
        generated, skipped, failed = _visualize_all(arguments.output_root)
    except OSError as error:
        parser.error(str(error))
    print(f"generated={generated} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
