"""
Validate a geometry calibration on an independent rotation series.

The series is analysed twice, with nominal geometry and with the DetCal
applied (geometry fixed); only the goniometer, per-run angles and the cell
length ratios b/a, c/a are fitted. Held-out indexing and the refined ratios
are compared.

Example (mesolite)::

    python scripts/validate_calibration.py --name mesolite \\
        --cell 18.40 56.65 6.54 90 90 90 --centering F --tols 1.0 0.5 \\
        --ipts 37331 --runs 1132 1200 2 --runs 1286 1372 2 --work work/mesolite \\
        --detcal src/quasi_laue_reduction/calibration/IMAGINE_garnet_IPTS-37331_2026-10-01_lite.DetCal
"""

import argparse
import json
import os

import numpy as np

from quasi_laue_reduction.calibrate import (
    VALIDATION_STAGES,
    SeriesCalibration,
    extract_series,
    load_series,
    nominal_positions,
)
from quasi_laue_reduction.peaks import interpolate_positions


def detcal_positions(work, detcal, shape):
    from mantid.simpleapi import DeleteWorkspace, mtd

    from quasi_laue_reduction.detcal import apply_detcal
    from quasi_laue_reduction.loading import _counts_workspace
    from quasi_laue_reduction.workflow import QuasiLaue

    xml = open(os.path.join(work, "lite_corrected.xml")).read()
    n = int(np.prod(shape[:3]))
    _counts_workspace("_val", np.zeros(n), idf_xml=xml, instrument="IMAGINE")
    apply_detcal("_val", detcal, shape[1:3])
    pos = QuasiLaue("_val", pixel_shape=shape[1:3]).positions
    for name in list(mtd.getObjectNames()):
        DeleteWorkspace(name)
    return pos


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--name", required=True)
    p.add_argument("--cell", nargs=6, type=float, required=True)
    p.add_argument("--centering", default="P")
    p.add_argument("--ipts", type=int, required=True)
    p.add_argument("--runs", nargs=3, type=int, action="append", metavar=("FIRST", "LAST", "STEP"), required=True)
    p.add_argument("--band", nargs=2, type=float, default=(2.0, 10.0))
    p.add_argument("--tols", nargs=2, type=float, default=(0.5, 0.35))
    p.add_argument("--work", required=True)
    p.add_argument("--detcal", required=True)
    p.add_argument("--n-proc", type=int, default=16)
    args = p.parse_args()

    runs_list = [r for a, b, s in args.runs for r in range(a, b + 1, s)]
    lite = [f"/HFIR/CG4D/IPTS-{args.ipts}/shared/autoreduce/CG4D_{r}.lite.nxs.h5" for r in runs_list]
    extract_series(lite, args.work)

    nominal = nominal_positions(lite[0], args.work)
    geometries = {"nominal": nominal, "detcal": detcal_positions(args.work, args.detcal, nominal.shape)}

    results = {}
    for name, pos in geometries.items():
        runs = load_series(args.work)
        for r in runs:
            coords = np.load(os.path.join(args.work, f"peaks_{r['run']}.npz"))["coords"]
            r["xyz"] = interpolate_positions(pos, coords)
        cal = SeriesCalibration(runs, pos, args.cell, args.centering, band=args.band, tols=args.tols, n_proc=args.n_proc)
        cal.orientations(cache=os.path.join(args.work, f"U_runs_{name}.npy"))
        report = cal.fit(stages=VALIDATION_STAGES)
        results[name] = report
        print(name, json.dumps(report[-1]), flush=True)

    json.dump(results, open(os.path.join(args.work, "validation.json"), "w"), indent=1)
    for name, report in results.items():
        r = report[-1]
        print(f"{name:8s} held-out indexed {r['test']['indexed']}/{r['test']['of']} ({100 * r['test']['fraction']:.0f}%) "
              f"median {r['test']['median_deg']:.3f} deg; b/a {r.get('b_over_a')}, c/a {r.get('c_over_a')}")


if __name__ == "__main__":
    main()
