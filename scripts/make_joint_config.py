"""Materialize a joint-training config without relative-path drift."""

from __future__ import annotations

import argparse
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
    return parser


def generate_joint_config(
    template: str | Path,
    output: str | Path,
    *,
    dataset_root: str | Path,
    seed: int,
) -> Path:
    template_path = Path(template).resolve()
    output_path = Path(output).resolve()
    resolved_dataset_root = Path(dataset_root).resolve()
    template_config = load_config(template_path)
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
    training["seed"] = seed

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    generated = load_config(output_path)
    if Path(generated.dataset.root) != resolved_dataset_root:
        raise RuntimeError("generated dataset root did not remain absolute")
    if generated.training.seed != seed:
        raise RuntimeError("generated seed does not match the requested seed")
    return output_path


def main() -> None:
    arguments = build_parser().parse_args()
    try:
        output = generate_joint_config(
            arguments.template,
            arguments.output,
            dataset_root=arguments.dataset_root,
            seed=arguments.seed,
        )
    except (OSError, RuntimeError, ValueError) as error:
        build_parser().error(str(error))
    print(output)


if __name__ == "__main__":
    main()
