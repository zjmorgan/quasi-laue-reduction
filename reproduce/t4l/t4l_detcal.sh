#!/bin/bash
# Run from anywhere; QLR_WORK (default <repo>/work) holds inputs and outputs, PYTHON the interpreter.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${QLR_WORK:-$REPO/work}"
PY="${PYTHON:-python}"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1
cd "$REPO"
F=/HFIR/CG4D/IPTS-37331/shared/autoreduce/CG4D_1816.lite.nxs.h5
for G in default none; do
  ARG=""; [ "$G" = default ] && ARG="--detcal default"
  echo "=== 1816 detcal=$G $(date)"
  $PY -m quasi_laue_reduction.cli --lite $F --cell 61.5 61.5 95.9 90 90 120 --centering P --band 2.0 10.0 \
      --seed 1 --restarts 6 $ARG 2>&1 | grep -vE "Notice|Information|^\[|Warning|fork|AddPeak"
done
echo T4L_DONE
