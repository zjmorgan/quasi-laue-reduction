#!/bin/bash
# Run from anywhere; QLR_WORK (default <repo>/work) holds inputs and outputs, PYTHON the interpreter.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${QLR_WORK:-$REPO/work}"
PY="${PYTHON:-python}"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1
cd "$WORK/null"
$PY "$HERE/null_mc.py" 100000 > null_mc.log 2>&1
$PY "$HERE/null_search.py" > null_search.log 2>&1
$PY "$HERE/null_accept.py" null_accept.jsonl > null_accept.log 2>&1
