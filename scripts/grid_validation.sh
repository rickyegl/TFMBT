#!/usr/bin/env bash
# TimesFM short/medium-context sweep on the validation window (run locally).
# Long contexts (4096-16000) run on GitHub Actions: see ci/timesfm_plan.json.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
run() {
  local name="timesfm_$2_L$4$([ "${5:-}" = --no-flip ] && echo _noflip)"
  [ -f "results/preds_${name}_validation.npz" ] && return
  $PY scripts/run_timesfm.py --split validation "$@" 2>&1 | grep -v -i warn | tail -1
}
run --mode price --context 128
run --mode price --context 512
run --mode price5m --context 256
run --mode price5m --context 512
run --mode returns --context 512
run --mode price --context 1024
run --mode price5m --context 1024
run --mode price --context 2048 --no-flip
run --mode price --context 2048
run --mode returns --context 2048
run --mode price5m --context 2048
