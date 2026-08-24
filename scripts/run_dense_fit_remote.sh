#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BASE_CONFIG="${BASE_CONFIG:-$PROJECT_ROOT/outputs/lr_strategy_cuda_20260822_190936/runs/baseline_legacy_seed73/config.yaml}"
STAMP="$(date +%Y%m%d_%H%M%S)"
PIPELINE_DIR="$PROJECT_ROOT/smoke_outputs/cuda_${STAMP}"

cd "$PROJECT_ROOT"
mkdir -p outputs smoke_outputs logs "$PIPELINE_DIR"

if [[ ! -f "$BASE_CONFIG" ]]; then
  printf 'missing saved six-layer config: %s\n' "$BASE_CONFIG" >&2
  exit 2
fi

python -m huav.cli.check_gpu --device cuda --num-threads 8

python -m huav.experiments.dense_fit \
  --base-config "$BASE_CONFIG" \
  --data-root data/ev-uav \
  --files train_090.npz \
  --output-dir "$PIPELINE_DIR/stage1_train090" \
  --device cuda \
  --seed 73 \
  --max-steps 400 \
  --eval-interval 20 \
  --min-event-iou 0.90 \
  --min-event-pd 0.95 \
  --min-object-pd 0.98 \
  --min-file-event-iou 0.90 \
  --min-file-object-pd 0.98

STAGE1_DIR="$PIPELINE_DIR/stage1_train090"
if ! python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1], encoding="utf-8"))["passed"] else 1)' "$STAGE1_DIR/decision.json"; then
  PIPELINE_DIR="$PIPELINE_DIR" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PIPELINE_DIR"])
stage1 = json.loads((root / "stage1_train090" / "decision.json").read_text(encoding="utf-8"))
(root / "final_decision.json").write_text(
    json.dumps(
        {"passed": False, "stopped_after_stage": 1, "stage1": stage1},
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY
  printf 'stage 1 failed; stage 2 was not started\n'
  printf '%s\n' "$PIPELINE_DIR" > smoke_outputs/dense_fit_latest_run.txt
  exit 0
fi

python -m huav.experiments.dense_fit \
  --base-config "$BASE_CONFIG" \
  --data-root data/ev-uav \
  --files train_088.npz,train_089.npz,train_090.npz,train_091.npz \
  --output-dir "$PIPELINE_DIR/stage2_train088_091" \
  --device cuda \
  --seed 73 \
  --max-steps 600 \
  --eval-interval 40 \
  --min-event-iou 0.80 \
  --min-event-pd 0.85 \
  --min-object-pd 0.90 \
  --min-file-event-iou 0.60 \
  --min-file-object-pd 0.75

PIPELINE_DIR="$PIPELINE_DIR" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PIPELINE_DIR"])
stage1 = json.loads((root / "stage1_train090" / "decision.json").read_text(encoding="utf-8"))
stage2 = json.loads((root / "stage2_train088_091" / "decision.json").read_text(encoding="utf-8"))
(root / "final_decision.json").write_text(
    json.dumps(
        {
            "passed": bool(stage1["passed"] and stage2["passed"]),
            "stopped_after_stage": 2,
            "stage1": stage1,
            "stage2": stage2,
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY

printf '%s\n' "$PIPELINE_DIR" > smoke_outputs/dense_fit_latest_run.txt
printf 'dense-fit pipeline directory: %s\n' "$PIPELINE_DIR"

