"""
Standard workflow on extracted runs with nominal geometry vs a DetCal.

Uses the images saved by scripts/calibrate_geometry.py (no re-reading of the
raw data). Every Mantid workspace is deleted after each run.

Example::

    python scripts/check_detcal.py --work work/garnet --cell 11.9386 11.9386 11.9386 90 90 90 \\
        --centering I --runs 2022 2034 2046 --detcal src/quasi_laue_reduction/calibration/IMAGINE_garnet_IPTS-37331_2026-10-01_lite.DetCal
"""

import argparse
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--work", required=True)
    p.add_argument("--cell", nargs=6, type=float, required=True)
    p.add_argument("--centering", default="P")
    p.add_argument("--runs", nargs="+", type=int, required=True)
    p.add_argument("--detcal", required=True)
    p.add_argument("--band", nargs=2, type=float, default=(2.0, 10.0))
    p.add_argument("--n-proc", type=int, default=16)
    p.add_argument("--pdf-run", type=int, default=None, help="write a peak PDF for this run with the DetCal")
    args = p.parse_args()

    from mantid.simpleapi import DeleteWorkspace, mtd

    from quasi_laue_reduction.detcal import apply_detcal
    from quasi_laue_reduction.loading import _counts_workspace
    from quasi_laue_reduction.workflow import QuasiLaue

    xml = open(os.path.join(args.work, "lite_corrected.xml")).read()
    out_fn = os.path.join(args.work, "check_detcal.jsonl")

    for run in args.runs:
        images = np.load(os.path.join(args.work, f"images_{run}.npz"))["images"].astype(float)
        axis0 = float(np.load(os.path.join(args.work, f"peaks_{run}.npz"))["axis0"])
        rec = {"run": run, "axis0": axis0}

        for geom in ("nominal", "detcal"):
            ws = "check"
            _counts_workspace(ws, images.ravel(), idf_xml=xml, instrument="IMAGINE")
            if geom == "detcal":
                apply_detcal(ws, args.detcal, images.shape[1:])
            ql = QuasiLaue(ws, pixel_shape=images.shape[1:])
            ql.find_peaks(method="local", n_sigma=8)
            kf = ql.kf_ki_dir.copy()
            ql.find_UB(*args.cell, args.band, centering=args.centering, n_proc=args.n_proc, seed=1)

            stats = {"peaks": len(kf)}
            for tol in (0.2, 0.35, 0.5):
                s = ql.opt.index_significance(kf, args.band, angle_tol=np.deg2rad(tol))
                stats[f"idx_{tol}"] = s["indexed"]
                if tol == 0.35:
                    stats["log10p"] = round(s["log10_p_value"], 1)
            ql.opt.index(kf, args.band, angle_tol=np.deg2rad(1.0))
            e = np.rad2deg(ql.opt.last_angle_error)
            stats["median_resid_within_1deg"] = round(float(np.median(e[e <= 1.0])), 3)
            rec[geom] = stats

            if geom == "detcal" and run == args.pdf_run:
                pred = ql.predicted_pixels(args.band, d_min=1.5)
                ql.plot_peaks_pdf(os.path.join(args.work, f"CG4D_{run}_peaks_detcal.pdf"), predicted=pred)

            for name in list(mtd.getObjectNames()):
                DeleteWorkspace(name)

        print(json.dumps(rec), flush=True)
        with open(out_fn, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
