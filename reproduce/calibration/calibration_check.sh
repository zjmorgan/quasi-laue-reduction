#!/bin/bash
# Run from anywhere; QLR_WORK (default <repo>/work) holds inputs and outputs, PYTHON the interpreter.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${QLR_WORK:-$REPO/work}"
PY="${PYTHON:-python}"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1
cd "$REPO"
DETCAL=src/quasi_laue_reduction/calibration/IMAGINE_garnet_IPTS-37331_2026-10-01_lite.DetCal
echo "check_detcal start $(date)"
$PY scripts/check_detcal.py --work $WORK/garnet --cell 11.9386 11.9386 11.9386 90 90 90 --centering I \
    --runs 2022 2034 2046 2058 2070 2082 2094 2106 2118 2130 2142 2154 --detcal $DETCAL --pdf-run 2022 > $WORK/check_detcal.log 2>&1
echo "validation start $(date)"
$PY scripts/validate_calibration.py --name mesolite --cell 18.40 56.65 6.54 90 90 90 --centering F --tols 1.0 0.5 \
    --ipts 37331 --runs 1132 1200 2 --runs 1286 1372 2 --work $WORK/mesolite --detcal $DETCAL > $WORK/validate_mesolite.log 2>&1
echo "QUEUE_DONE $(date)"
