#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

case "$MODE" in
  smoke)
    CONFIG="configs/huav_score_optimized_autodl_smoke.yaml"
    RUN_PREFIX="score_optimized_autodl_smoke"
    RUN_ROOT="$PROJECT_ROOT/smoke_outputs"
    ;;
  full)
    CONFIG="configs/huav_score_optimized_autodl.yaml"
    RUN_PREFIX="score_optimized_autodl_full"
    RUN_ROOT="$PROJECT_ROOT/outputs"
    ;;
  *)
    echo "usage: bash scripts/autodl_train.sh [smoke|full]" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
python -m huav.cli.check_gpu --device cuda --num-threads 8

STAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="$RUN_ROOT/${RUN_PREFIX}_${STAMP}"
LOG_PATH="$RUN_ROOT/${RUN_PREFIX}_${STAMP}.log"
mkdir -p "$RUN_ROOT"

python -m huav.cli.train \
  --config "$CONFIG" \
  --device cuda \
  --output-dir "$RUN_DIR" \
  2>&1 | tee "$LOG_PATH"

printf 'run directory: %s\n' "$RUN_DIR"
printf 'console log: %s\n' "$LOG_PATH"
