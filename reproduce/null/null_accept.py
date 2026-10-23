"""Null searches with the production acceptance procedure: up to three restarts,
stopping at log10 p < -5, each restart scored by the more significant of its scores
at the final (0.35 deg) and search (1 deg) tolerances. Same scrambles and seeds as
work/null/null_search.py."""
import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, json, time
import numpy as np
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB, primitive_cell

W = (WORK_DIR + "/")
m = np.load(W + "figure/t4l_1816_matched.npz")
p = np.load(W + "robust/mesolite_peaks.npz")
cases = [("T4 lysozyme", m["kf"], m["heights"], (61.1, 61.1, 96.84, 90, 90, 120), "P", (2.0, 10.0), 20),
         ("mesolite", p["kf"], p["heights"], (18.39, 56.55, 6.54, 90, 90, 90), "F", (2.0, 4.5), 10)]
out = open(sys.argv[1], "a")
for name, kf, h, cell, cent, band, n in cases:
    for k in range(n):
        rng = np.random.default_rng(1000 + k)
        phi = rng.uniform(0, 2 * np.pi, len(kf))
        lab = kf + np.array([0, 0, 1.0])
        c, s = np.cos(phi), np.sin(phi)
        kf_rot = np.column_stack([c * lab[:, 0] - s * lab[:, 1], s * lab[:, 0] + c * lab[:, 1], lab[:, 2]])
        q = kf_rot - np.array([0, 0, 1.0])
        o = CalculateUB(*primitive_cell(*cell, cent))
        t = time.perf_counter()
        o.find_orientation(q, band, heights=h, n_proc=16, seed=k + 1)  # production defaults
        restarts = [round(r["log10_p_value"], 2) for r in o.restarts]
        rec = {"case": name, "scramble": k, "seconds": round(time.perf_counter() - t, 1),
               "restart_scores": restarts, "best": min(restarts), "accepted": min(restarts) < -5,
               "log10p_0.35_final": round(o.significance["log10_p_value"], 2)}
        out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
print("DONE", flush=True)
