"""
Command line: find peaks, determine UB and optionally integrate.

Example::

    quasi-laue --lite /HFIR/CG4D/IPTS-37331/shared/autoreduce/CG4D_1438.lite.nxs.h5 \\
        --cell 18.40 56.65 6.54 90 90 90 --centering F --band 2.8 4.6 \\
        --lattice Orthorhombic --refine --save-peaks CG4D_1438.integrate
"""

import argparse
import json
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--lite", help="autoreduced lite event NeXus file")
    src.add_argument("--tiff", help="CG4D image-plate TIFF")

    p.add_argument("--cell", nargs=6, type=float, required=True, metavar=("A", "B", "C", "ALPHA", "BETA", "GAMMA"))
    p.add_argument("--centering", default="P", help="conventional-cell centering (P, A, B, C, I, F, R)")
    p.add_argument("--band", nargs=2, type=float, required=True, metavar=("LMIN", "LMAX"), help="wavelength band in angstroms")
    p.add_argument("--lattice", default=None, help="lattice system for --refine, e.g. Hexagonal")
    p.add_argument("--refine", action="store_true", help="refine lattice constants after indexing")
    p.add_argument("--peak-method", default="local", choices=["local", "matched", "global"])
    p.add_argument("--n-sigma", type=float, default=8.0, help="local peak threshold")
    p.add_argument("--z-min", type=float, default=5.0, help="matched-filter peak threshold")
    p.add_argument("--n-proc", type=int, default=-1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--popsize", type=int, default=300)
    p.add_argument("--maxiter", type=int, default=1000)
    p.add_argument("--angle-tol", type=float, default=0.35, help="indexing tolerance in degrees")
    p.add_argument("--integrate", action="store_true", help="predict and integrate peaks")
    p.add_argument("--d-min", type=float, default=2.0, help="prediction resolution limit")
    p.add_argument("--save-peaks", default=None, help="write peaks with SaveIsawPeaks")
    p.add_argument("--save-ub", default=None, help="write UB with SaveIsawUB")
    p.add_argument("--detcal", default=None, help="DetCal calibration ('default' for the packaged one)")
    p.add_argument("--restarts", type=int, default=3, help="maximum orientation-search restarts")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from mantid.simpleapi import SaveIsawPeaks, SaveIsawUB

    from .loading import load_cg4d_image, load_lite
    from .workflow import QuasiLaue

    timing = {}

    t = time.perf_counter()
    detcal = args.detcal
    if detcal == "default":
        from .loading import calibration_file

        detcal = calibration_file()
    info = load_lite(args.lite, detcal=detcal) if args.lite else load_cg4d_image(args.tiff)
    timing["load"] = time.perf_counter() - t

    ql = QuasiLaue("data", pixel_shape=info["pixel_shape"])

    t = time.perf_counter()
    kwargs = {"local": {"n_sigma": args.n_sigma}, "matched": {"z_min": args.z_min}}.get(args.peak_method, {})
    ql.find_peaks(method=args.peak_method, **kwargs)
    timing["find_peaks"] = time.perf_counter() - t

    t = time.perf_counter()
    num = ql.find_UB(
        *args.cell,
        args.band,
        centering=args.centering,
        n_proc=args.n_proc,
        seed=args.seed,
        angle_tol_deg=args.angle_tol,
        popsize=args.popsize,
        maxiter=args.maxiter,
        n_restarts=args.restarts,
    )
    timing["find_UB"] = time.perf_counter() - t

    if args.refine:
        if args.lattice is None:
            raise SystemExit("--refine needs --lattice")
        t = time.perf_counter()
        ql.refine_UB(args.lattice, args.band)
        timing["refine"] = time.perf_counter() - t

    if args.save_ub:
        SaveIsawUB(InputWorkspace=ql.peaks_ws, Filename=args.save_ub)

    if args.integrate:
        t = time.perf_counter()
        ql.predict_peaks(args.band, d_min=args.d_min)
        ql.integrate_peaks()
        timing["integrate"] = time.perf_counter() - t

    if args.save_peaks:
        SaveIsawPeaks(InputWorkspace=ql.peaks_ws, Filename=args.save_peaks)

    print(
        json.dumps(
            {
                "indexed": int(num),
                "peaks": int(len(ql.found_coords)),
                "significance": ql.significance,
                "detcal": detcal,
                "cell": list(ql.opt.get_lattice_constants()),
                "timing_s": timing,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
