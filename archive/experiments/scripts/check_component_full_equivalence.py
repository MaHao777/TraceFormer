"""在指定PYTHONPATH下记录完整验证集logits，独立比较两套源码。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from traceformer.config import load_config
from traceformer.training.engine import (
    _dataset,
    _file_sha256,
    _validate_checkpoint_contract,
    _write_json,
    build_model,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapted", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(config.training.num_threads)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _validate_checkpoint_contract(saved, config)
    spec = None
    if args.adapted:
        # 旧快照无需认识新契约；仅新源码执行Full实验适配路径。
        from traceformer.experiments.ablation import ComponentCoreSpec
        spec = ComponentCoreSpec("Full")
    model = build_model(config, ablation=spec).eval()
    model.load_state_dict(saved["model_state"], strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    with torch.inference_mode():
        for sequence in _dataset(config, "val"):
            output = model(sequence.events)
            path = args.output / f"{sequence.source_path.stem}.pt"
            torch.save({name: output[name] for name in
                        ("event_logits", "event_token_indices", "token_coords")}, path)
            records[path.name] = {"sha256": _file_sha256(path),
                                  "source_sha256": _file_sha256(sequence.source_path)}
    if len(records) != 24:
        raise ValueError("Full equivalence requires all 24 validation sequences")
    _write_json(args.output / "report.json", {"files": records,
                "checkpoint_sha256": _file_sha256(args.checkpoint), "adapted": args.adapted})


if __name__ == "__main__":
    main()
