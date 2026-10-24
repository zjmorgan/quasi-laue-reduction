#!/bin/bash
# Run from anywhere; QLR_WORK (default <repo>/work) holds inputs and outputs, PYTHON the interpreter.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${QLR_WORK:-$REPO/work}"
PY="${PYTHON:-python}"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1
cd "$REPO"
PYTHONPATH="$REPO/src:$REPO/benchmarks" $PY benchmarks/bench_optimizer.py --out optimizer_equal > "$WORK/bench_optimizer_equal.log" 2>&1
