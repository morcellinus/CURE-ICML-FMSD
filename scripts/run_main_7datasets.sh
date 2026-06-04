#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export OPENBLAS_NUM_THREADS="4"
export NUMEXPR_NUM_THREADS="4"
DATA_ROOT="$ROOT/data/uspds"
OUT_ROOT="$ROOT/results"
TAU_JSON="$ROOT/config/expected_main_taus.json"
DEVICE="${DEVICE:-cuda}"
TABICL_MODEL_PATH="${TABICL_MODEL_PATH:-auto}"
mkdir -p "$OUT_ROOT"

declare -A MAXLEN=(
  [noaa]=18159
  [meter]=22950
  [rialto]=82250
  [posture_no8]=163477
  [nomao]=34465
  [poker]=829201
  [agr_a]=30000
)

for ds in noaa meter rialto posture_no8 nomao poker agr_a; do
  TAU_VALUE="$(python - <<PY
import json
with open(${TAU_JSON@Q}, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(data[${ds@Q}])
PY
)"
  python "$ROOT/scripts/run_tabicl_cure.py" \
    --dataset "$ds" \
    --tau "$TAU_VALUE" \
    --data-root "$DATA_ROOT" \
    --output-root "$OUT_ROOT" \
    --max-stream-length "${MAXLEN[$ds]}" \
    --device "$DEVICE" \
    --tabicl-model-path "$TABICL_MODEL_PATH" \
    --threads 4
 done
