#!/usr/bin/env bash
#
# AIgnition 3.0 — single entry point for the automated testing pipeline.
#
#   ./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>
#
# Defaults let it run locally with no arguments. At test time the harness calls:
#   ./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
#
# It (a) generates features from whatever is in DATA_DIR, then (b) loads the
# committed pickled model and writes probabilistic predictions to OUTPUT_PATH.
# Fully offline, deterministic (seeded), no interactive input, no network calls.
set -euo pipefail

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

# Internal, relative intermediate artifact (not a scored output).
FEATURES_PATH="${AIGNITION_FEATURES:-./features.pkl}"

# Reproducibility: stable hash seed for any dict/set ordering.
export PYTHONHASHSEED=0

# Resolve a Python interpreter (prefer python3, fall back to python).
if command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "ERROR: no python interpreter found on PATH" >&2
  exit 1
fi

# Run from the repo root regardless of where run.sh is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "[run.sh] Python: $($PY --version 2>&1)"
echo "[run.sh] 1/2 Generating features from $DATA_DIR ..."
"$PY" src/generate_features.py --data-dir "$DATA_DIR" --out "$FEATURES_PATH"

echo "[run.sh] 2/2 Loading model $MODEL_PATH and predicting ..."
"$PY" src/predict.py --features "$FEATURES_PATH" --model "$MODEL_PATH" --output "$OUTPUT_PATH"

echo "[run.sh] Done. Predictions written to $OUTPUT_PATH"
