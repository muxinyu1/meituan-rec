#!/usr/bin/env bash
# One-click helper to regenerate transformed train/val splits.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

TRAIN_INPUT=${TRAIN_INPUT:-"$PROJECT_ROOT/data/recsys_task_data/train_merged_imputed-20221014.csv"}
TRAIN_TRANSFORMED=${TRAIN_TRANSFORMED:-"$PROJECT_ROOT/data/recsys_task_data/train_merged_transformed-20221014.csv"}
TRAIN_OUTPUT=${TRAIN_OUTPUT:-"$PROJECT_ROOT/data/recsys_task_data/train_merged_transformed_train-20221014.csv"}
VAL_OUTPUT=${VAL_OUTPUT:-"$PROJECT_ROOT/data/recsys_task_data/train_merged_transformed_val-20221014.csv"}

run_step() {
  local step="$1"
  shift
  echo "[+] $step"
  "$@"
}

run_step "Transforming train dataset" \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/transform_features.py" \
  --input "$TRAIN_INPUT" \
  --output "$TRAIN_TRANSFORMED"

run_step "Splitting into train/val" \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/split_train_val.py" \
  --input "$TRAIN_TRANSFORMED" \
  --train-output "$TRAIN_OUTPUT" \
  --val-output "$VAL_OUTPUT"

cat <<EOF
Done.
Train split: $TRAIN_OUTPUT
Val split  : $VAL_OUTPUT
(Override defaults via TRAIN_INPUT, TRAIN_TRANSFORMED, TRAIN_OUTPUT, VAL_OUTPUT, PYTHON_BIN)
EOF
