#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ARMS="${ARMS:-a b c}"

cd "$PROJECT_ROOT"
mkdir -p outputs smoke_outputs logs

audit_submission() {
  local run_dir="$1"
  RUN_DIR="$run_dir" python - <<'PY'
from __future__ import annotations

import json
import os
import zipfile
from itertools import zip_longest
from pathlib import Path

import numpy as np

run_dir = Path(os.environ["RUN_DIR"])
validation_dir = Path("data/ev-uav/val")
submission_dir = run_dir / "submission"
source_files = sorted(validation_dir.glob("*.npz"))
txt_files = sorted(submission_dir.glob("*.txt"))
if len(source_files) != 24 or len(txt_files) != len(source_files):
    raise RuntimeError(
        f"expected 24 validation NPZ/TXT files, got {len(source_files)}/{len(txt_files)}"
    )

total_rows = 0
positive_rows = 0
for source_path in source_files:
    txt_path = submission_dir / f"{source_path.stem}.txt"
    with np.load(source_path, allow_pickle=False) as archive:
        events = archive["ev"]
    with txt_path.open("r", encoding="ascii") as stream:
        for line_number, (line, event) in enumerate(
            zip_longest(stream, events), start=1
        ):
            if line is None or event is None:
                raise RuntimeError(f"{txt_path.name}: row count mismatch")
            fields = line.split()
            if len(fields) != 5:
                raise RuntimeError(f"{txt_path.name}:{line_number}: expected five columns")
            x, y, timestamp, polarity, label = fields
            if (
                int(x) != int(event["x"])
                or int(y) != int(event["y"])
                or float(timestamp) != float(event["t"])
                or int(polarity) != int(event["p"])
            ):
                raise RuntimeError(
                    f"{txt_path.name}:{line_number}: source x y t p mismatch"
                )
            if label not in {"0", "1"}:
                raise RuntimeError(f"{txt_path.name}:{line_number}: non-binary label")
            total_rows += 1
            positive_rows += int(label)

zip_path = submission_dir / "submission.zip"
with zipfile.ZipFile(zip_path) as archive:
    members = archive.namelist()
expected_members = sorted(path.name for path in txt_files)
if sorted(members) != expected_members or any(
    "/" in member or not member.endswith(".txt") for member in members
):
    raise RuntimeError("submission ZIP must contain only root-level TXT files")

report = {
    "txt_files": len(txt_files),
    "total_rows": total_rows,
    "positive_rows": positive_rows,
    "source_order_and_values_match": True,
    "binary_labels_only": True,
    "zip_root_txt_only": True,
}
(run_dir / "submission_audit.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=False))
PY
}

run_resource_gate() {
  local arm="$1"
  local config="configs/huav_embedding_${arm}_autodl.yaml"
  local report="smoke_outputs/embedding_${arm}_train094_resource.json"
  python scripts/check_largest_sequence.py \
    --config "$config" \
    --sequence train_094.npz \
    --epoch 7 \
    --device cuda | tee "$report"
}

run_arm() {
  local arm="$1"
  local full_config="configs/huav_embedding_${arm}_autodl.yaml"
  local smoke_config="configs/huav_embedding_${arm}_autodl_smoke.yaml"
  local stamp
  local smoke_dir
  local run_dir

  stamp="$(date +%Y%m%d_%H%M%S)"
  smoke_dir="smoke_outputs/cuda_${stamp}_embedding_${arm}"
  python -m huav.cli.train \
    --config "$smoke_config" \
    --device cuda \
    --output-dir "$smoke_dir" \
    --note "embedding_${arm}_smoke"
  python -m huav.cli.evaluate \
    --config "$smoke_dir/config.yaml" \
    --checkpoint "$smoke_dir/best.pt" \
    --device cuda \
    --quiet
  cmp "$smoke_dir/metrics.json" "$smoke_dir/reproduced_metrics.json"

  stamp="$(date +%Y%m%d_%H%M%S)"
  run_dir="outputs/cuda_${stamp}_embedding_${arm}"
  python -m huav.cli.train \
    --config "$full_config" \
    --device cuda \
    --output-dir "$run_dir" \
    --note "embedding_${arm}_24epoch_99train_24val"
  python -m huav.cli.evaluate \
    --config "$run_dir/config.yaml" \
    --checkpoint "$run_dir/best.pt" \
    --device cuda \
    --quiet
  cmp "$run_dir/metrics.json" "$run_dir/reproduced_metrics.json"
  python -m huav.cli.predict \
    --config "$run_dir/config.yaml" \
    --checkpoint "$run_dir/best.pt" \
    --device cuda \
    --quiet
  audit_submission "$run_dir"
  printf '%s\n' "$run_dir" > "outputs/embedding_${arm}_latest_run.txt"
}

python -m huav.cli.check_gpu --device cuda --num-threads 8
python -m pytest -q
python -m ruff check src tests
git diff --check

for arm in b c; do
  run_resource_gate "$arm"
done

for arm in $ARMS; do
  run_arm "$arm"
done

python -m pytest -q
python -m ruff check src tests
git diff --check
