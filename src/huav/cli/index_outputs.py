"""Build a chronological Markdown index for HUAV output artifacts."""

from __future__ import annotations

import argparse

from huav.artifacts import rebuild_output_index


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="huav-index-outputs",
        description="Rebuild the HUAV training output index",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="output root to scan (default: outputs)",
    )
    arguments = parser.parse_args()
    try:
        index_path = rebuild_output_index(arguments.output_root)
    except OSError as error:
        parser.error(str(error))
    print(f"output index: {index_path.resolve()}")


if __name__ == "__main__":
    main()
