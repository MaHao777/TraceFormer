"""Materialize a joint-training config without relative-path drift."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from huav.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a validated adaptive joint template while setting an "
            "absolute dataset root and experiment seed"
        )
    )
    parser.add_argument("--template", required=True, help="joint YAML template")
    parser.add_argument("--output", required=True, help="generated YAML path")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="HUAV dataset root containing train/ and val/",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing generated config",
    )
    return parser


def generate_joint_config(
    template: str | Path,
    output: str | Path,
    *,
    dataset_root: str | Path,
    seed: int,
    force: bool = False,
) -> Path:
    template_path = Path(template).resolve()
    output_path = Path(output).resolve()
    if output_path == template_path:
        raise ValueError("template and output must be different paths")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"output already exists (use --force to replace): {output_path}"
        )
    resolved_dataset_root = Path(dataset_root).resolve()
    template_config = load_config(template_path)
    resolved_output_root = Path(
        template_config.training.output_root
    ).resolve()
    if template_config.training.training_stage != "joint":
        raise ValueError("template must use training_stage=joint")
    if template_config.model.adaptive_threshold_mode == "none":
        raise ValueError("template must enable adaptive thresholding")

    with template_path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("template must be a YAML mapping")
    dataset = raw.get("dataset")
    training = raw.get("training")
    if not isinstance(dataset, dict) or not isinstance(training, dict):
        raise ValueError("template must contain dataset and training mappings")
    dataset["root"] = str(resolved_dataset_root)
    training["output_root"] = str(resolved_output_root)
    training["seed"] = seed

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(
                yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
            )
            stream.flush()
            os.fsync(stream.fileno())
        generated = load_config(temporary_path)
        if Path(generated.dataset.root) != resolved_dataset_root:
            raise RuntimeError(
                "generated dataset root did not remain absolute"
            )
        if Path(generated.training.output_root) != resolved_output_root:
            raise RuntimeError(
                "generated output root did not remain absolute"
            )
        if generated.training.seed != seed:
            raise RuntimeError(
                "generated seed does not match the requested seed"
            )
        if generated.training.training_stage != "joint":
            raise RuntimeError("generated config is not a joint config")
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def main() -> None:
    arguments = build_parser().parse_args()
    try:
        output = generate_joint_config(
            arguments.template,
            arguments.output,
            dataset_root=arguments.dataset_root,
            seed=arguments.seed,
            force=arguments.force,
        )
    except (OSError, RuntimeError, ValueError) as error:
        build_parser().error(str(error))
    print(output)


if __name__ == "__main__":
    main()
