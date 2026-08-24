#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BASE_CONFIG="${1:-configs/huav_score_optimized_autodl.yaml}"
STAMP="${EXPERIMENT_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ROOT="$PROJECT_ROOT/outputs/lr_strategy_cuda_${STAMP}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"
mkdir -p "$EXPERIMENT_ROOT"
printf '%s\n' "$$" > "$EXPERIMENT_ROOT/launcher.pid"
printf '%s\n' "$PROJECT_ROOT" > "$EXPERIMENT_ROOT/project_root.txt"

"$PYTHON_BIN" -m pip install -e ".[dev]"
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" -m ruff check src tests
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git diff --check
else
  printf 'git metadata unavailable; source diff was checked before upload\n'
fi
"$PYTHON_BIN" -m huav.cli.check_gpu --device cuda --num-threads 8
"$PYTHON_BIN" scripts/check_largest_sequence.py \
  --config "$BASE_CONFIG" \
  --sequence train_094.npz \
  --epoch 6 \
  --device cuda

"$PYTHON_BIN" -m huav.experiments.lr_strategy \
  --base-config "$BASE_CONFIG" \
  --experiment-root "$EXPERIMENT_ROOT"

printf '%s\n' "$EXPERIMENT_ROOT" > "$PROJECT_ROOT/outputs/lr_strategy_latest_run.txt"
printf 'experiment root: %s\n' "$EXPERIMENT_ROOT"
