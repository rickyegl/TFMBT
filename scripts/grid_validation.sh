#!/usr/bin/env bash
# TimesFM configuration sweep on the validation window (cheapest first).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
run() { $PY scripts/run_timesfm.py --split validation "$@" 2>&1 | grep -v -i warn | tail -1; }
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
run --mode price --context 4096
run --mode returns --context 4096
run --mode price --context 8192
