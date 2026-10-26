"""Rotation series of T4 lysozyme on IMAGINE-X (paper Table 3).

Run 1816 is indexed by the global search (seed 1, three restarts) and refined. Each later
run is predicted from the refined orientation of the previous run rotated by 10 degrees
about the vertical axis, scored, refined locally (orientation and c/a, hexagonal) and
scored again. For the first step both rotation senses are tried; the better one is kept
and the other is reported. Every run starts from the same cell (61.1, 61.1, 96.84);
informative peaks (chance probability <= 0.7 at 0.35 deg) are selected with that cell,
and scores count informative peaks within 0.35 deg.

This follows the script that produced the published table (recovered from the session
in which it was run). Peaks of each run are cached in $QLR_WORK/series/peaks_<run>.npz,
so the lite files ($QLR_CG4D_AUTOREDUCE, default /HFIR/CG4D/IPTS-37331/shared/autoreduce)
are read once.

Usage: python rotation_series.py
"""
import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB
from quasi_laue_reduction.peaks import find_peaks_matched, interpolate_positions, scattering_directions

AUTOREDUCE = _os.environ.get("QLR_CG4D_AUTOREDUCE", "/HFIR/CG4D/IPTS-37331/shared/autoreduce")
CELL = (61.1, 61.1, 96.84, 90, 90, 120)
BAND = (2.0, 10.0)
TOL = np.deg2rad(0.35)
STEP = 10.0
RUNS = [1816, 1818, 1820, 1822, 1824, 1826]
CACHE = _os.path.join(WORK_DIR, "series")


def peaks(run):
    """Matched-filter peaks (z >= 5) of one run with the garnet DetCal: kf_hat - ki_hat and heights."""
    f = _os.path.join(CACHE, f"peaks_{run}.npz")
    if _os.path.exists(f):
        d = np.load(f)
        return d["kf"], d["heights"]
    ref = _os.path.join(WORK_DIR, "figure", "t4l_1816.npz")
    if run == 1816 and _os.path.exists(ref):
        d = np.load(ref)
        images, P = d["images"].astype(float), d["positions"]
    else:
        from quasi_laue_reduction.loading import calibration_file, load_lite
        from quasi_laue_reduction.workflow import QuasiLaue

        info = load_lite(_os.path.join(AUTOREDUCE, f"CG4D_{run}.lite.nxs.h5"), ws="data", detcal=calibration_file())
        ql = QuasiLaue("data", pixel_shape=info["pixel_shape"])
        images, P = ql.extract_images().astype(float), ql.positions
    c, h, _ = find_peaks_matched(images, z_min=5)
    kf = scattering_directions(interpolate_positions(P, c))
    _os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(f, kf=kf, heights=h, coords=c)
    return kf, h


def score(o, kf, inf):
    s = o.index_significance(kf[inf], BAND, angle_tol=TOL)
    return {"indexed": int(s["indexed"]), "chance": round(float(s["expected_chance"]), 1),
            "log10p": round(float(s["log10_p_value"]), 1)}


def refine(o, kf, inf, tols):
    for tol in tols:
        o._score_angle_tol = np.deg2rad(tol)
        o.refine(kf[inf], BAND, "Hexagonal", error=0.05)
    return o


def evaluate(kf, U, do_refine=True):
    """Score at U with the fixed starting cell, then (optionally) refine locally and score again."""
    o = CalculateUB(*CELL)
    o.x = o.orientation_parameters_from_U(U)
    o.prepare_indexer(kf, BAND, ambiguity_power=0, force=True, lookup=False)
    inf = (-np.expm1(-o._candidate_counts * TOL**2 / 4)) <= 0.7
    predicted = score(o, kf, inf)
    if not do_refine:
        return predicted, None, None, inf
    refine(o, kf, inf, (1.0, 0.5, 0.35, 0.35, 0.2, 0.2))
    return predicted, score(o, kf, inf), o, inf


def angle_deg(U1, U2):
    return float(np.degrees(Rotation.from_matrix(U1 @ U2.T).magnitude()))


def ry(deg):
    return Rotation.from_euler("y", deg, degrees=True).as_matrix()


def main():
    kfs = {r: peaks(r) for r in RUNS}

    # run 1816: global search, then refinement (as for the other T4 lysozyme results)
    kf, h = kfs[1816]
    o = CalculateUB(*CELL)
    o.find_orientation(kf, BAND, h, n_proc=16, n_restarts=3, seed=1)
    inf = o.informative
    refine(o, kf, inf, (0.5, 0.35, 0.35, 0.2, 0.2))
    U = {1816: o.orientation_U(*o.x)}
    s, _, _, inf = evaluate(kf, U[1816], do_refine=False)
    print(json.dumps({"run": 1816, "source": "global search", "informative": int(inf.sum()), "score": s}), flush=True)

    sense = None
    for prev, run in zip(RUNS[:-1], RUNS[1:]):
        kf, _ = kfs[run]
        if sense is None:
            trials = {sg: evaluate(kf, ry(sg * STEP) @ U[prev], do_refine=False)[0] for sg in (1, -1)}
            sense = min(trials, key=lambda sg: trials[sg]["log10p"])
            print(json.dumps({"sense": sense, "opposite_sense_1818": trials[-sense]}), flush=True)
        Up = ry(sense * STEP) @ U[prev]
        predicted, refined, o, inf = evaluate(kf, Up)
        U[run] = o.orientation_U(*o.x)
        cell = o.get_lattice_constants()
        print(json.dumps({"run": run, "informative": int(inf.sum()), "predicted": predicted, "refined": refined,
                          "correction_deg": round(angle_deg(U[run], Up), 2),
                          "c_over_a": round(cell[2] / cell[0], 4)}), flush=True)

    total = Rotation.from_matrix(U[RUNS[-1]] @ U[RUNS[0]].T).as_rotvec()
    tilt = np.degrees(np.arccos(abs(total[1]) / np.linalg.norm(total)))
    print(json.dumps({"total_rotation_deg": round(float(np.degrees(np.linalg.norm(total))), 2),
                      "axis_from_vertical_deg": round(float(tilt), 2),
                      "expected_deg": STEP * (len(RUNS) - 1)}), flush=True)


if __name__ == "__main__":
    main()
