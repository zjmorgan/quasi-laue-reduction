"""Run 1820 of the T4 lysozyme rotation series indexed on its own by the global search.

Five seeds with the default search (three restarts, weight cap 2); each solution is
refined (orientation and c/a) and compared, up to the 622 symmetry, with the orientation
of run 1820 reached by the rotation series (rotation_series.py), which is rebuilt here.

Usage: python index_1820.py
"""
import os as _os
import json
import sys
import time

import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import rotation_series as rs
from quasi_laue_reduction.optimize import CalculateUB


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


SYM = [Rz(np.pi / 3 * k) @ m for k in range(6) for m in (np.eye(3), np.diag([1.0, -1.0, -1.0]))]


def series_orientation(kfs):
    o = CalculateUB(*rs.CELL)
    kf, h = kfs[1816]
    o.find_orientation(kf, rs.BAND, h, n_proc=16, n_restarts=3, seed=1)
    rs.refine(o, kf, o.informative, (0.5, 0.35, 0.35, 0.2, 0.2))
    U = o.orientation_U(*o.x)
    for run in (1818, 1820):
        _, _, oc, _ = rs.evaluate(kfs[run][0], rs.ry(-rs.STEP) @ U)
        U = oc.orientation_U(*oc.x)
    return U


def main():
    kfs = {r: rs.peaks(r) for r in (1816, 1818, 1820)}
    U_series = series_orientation(kfs)
    kf, h = kfs[1820]
    for seed in (1, 2, 3, 4, 5):
        t = time.perf_counter()
        o = CalculateUB(*rs.CELL)
        o.find_orientation(kf, rs.BAND, h, n_proc=16, seed=seed)
        accepted = bool(o.significance["accepted"])
        rs.refine(o, kf, o.informative, (0.5, 0.35, 0.35, 0.2, 0.2))
        U = o.orientation_U(*o.x)
        print(json.dumps({"seed": seed, "restart_scores": [round(r["log10_p_value"], 1) for r in o.restarts],
                          "accepted": accepted, "refined": rs.score(o, kf, o.informative),
                          "deg_from_series": round(min(rs.angle_deg(U @ S, U_series) for S in SYM), 2),
                          "time_s": round(time.perf_counter() - t)}), flush=True)


if __name__ == "__main__":
    main()
