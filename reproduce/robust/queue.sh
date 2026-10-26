#!/bin/bash
# Run from anywhere; QLR_WORK (default <repo>/work) holds inputs and outputs, PYTHON the interpreter.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${QLR_WORK:-$REPO/work}"
PY="${PYTHON:-python}"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1
cd "$WORK/robust"
$PY "$HERE/mesolite_peaks.py" > mesolite_peaks.log 2>&1
$PY "$HERE/seeds.py" > seeds.log 2>&1
$PY "$HERE/spurious_mandi.py" > spurious_mandi.log 2>&1
