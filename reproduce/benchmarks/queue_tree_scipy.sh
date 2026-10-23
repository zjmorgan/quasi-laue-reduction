#!/bin/bash
# Run from anywhere; QLR_WORK (default <repo>/work) holds inputs and outputs, PYTHON the interpreter.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${QLR_WORK:-$REPO/work}"
PY="${PYTHON:-python}"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1
cd "$REPO"
PYTHONPATH="$REPO/src:$REPO/benchmarks" $PY benchmarks/bench_search_tree_scipy.py \
  --peaks "$WORK/peaks_1816_detcal.npz" --cell 61.5 61.5 95.9 90 90 120 --band 2 10 > "$WORK/bench_tree_scipy.log" 2>&1
