"""
Calibrate the IMAGINE array geometry from a reference-crystal rotation series.

Example (garnet, IPTS-37331 even runs 2022-2156)::

    python scripts/calibrate_geometry.py --name garnet \\
        --cell 11.9386 11.9386 11.9386 90 90 90 --centering I \\
        --ipts 37331 --runs 2022 2156 2 --work work/garnet

Peaks are extracted into --work (resumable), then the goniometer and the
array geometry are fitted stage by stage; the DetCal of the stage with the
best held-out indexing and a JSON report are written to --work.
"""

import argparse
import json
import os

from quasi_laue_reduction.calibrate import SeriesCalibration, extract_series, load_series, nominal_positions


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--name", required=True)
    p.add_argument("--cell", nargs=6, type=float, required=True)
    p.add_argument("--centering", default="P")
    p.add_argument("--ipts", type=int, required=True)
    p.add_argument("--runs", nargs=3, type=int, metavar=("FIRST", "LAST", "STEP"), required=True)
    p.add_argument("--band", nargs=2, type=float, default=(2.0, 10.0))
    p.add_argument("--tols", nargs=2, type=float, default=(0.5, 0.35), help="search and final tolerance (deg)")
    p.add_argument("--work", required=True)
    p.add_argument("--n-proc", type=int, default=16)
    p.add_argument("--extract-only", action="store_true")
    args = p.parse_args()

    first, last, step = args.runs
    lite = [f"/HFIR/CG4D/IPTS-{args.ipts}/shared/autoreduce/CG4D_{r}.lite.nxs.h5" for r in range(first, last + 1, step)]
    os.makedirs(args.work, exist_ok=True)

    extract_series(lite, args.work)
    if args.extract_only:
        return

    runs = load_series(args.work)
    positions = nominal_positions(lite[0], args.work)
    cal = SeriesCalibration(runs, positions, args.cell, args.centering, band=args.band, tols=args.tols, n_proc=args.n_proc)
    cal.orientations(cache=os.path.join(args.work, "U_runs.npy"))
    report = cal.fit(out_dir=args.work)

    xml = open(os.path.join(args.work, "lite_corrected.xml")).read()
    best = cal.write_detcal(os.path.join(args.work, f"IMAGINE_{args.name}"), xml, comment=f"{args.name} calibration,")
    json.dump({"report": report, "selected": best["stage"], "args": vars(args)}, open(os.path.join(args.work, "calibration.json"), "w"), indent=1)
    print("selected", best["stage"], flush=True)


if __name__ == "__main__":
    main()
