"""Null searches: the full orientation search on data with no correct orientation.
Each observed peak direction is rotated about the incident beam by an independent random
angle, which keeps every peak's scattering angle (and therefore its candidate shell) but
destroys the lattice geometry. The significance the search reaches is the empirical
false-alarm level of the search itself."""
import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, json, time
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB, primitive_cell

W = (WORK_DIR + "/")
m = np.load(W + "figure/t4l_1816_matched.npz")
p = np.load(W + "robust/mesolite_peaks.npz")
cases = [("T4 lysozyme", m["kf"], m["heights"], (61.1, 61.1, 96.84, 90, 90, 120), "P", (2.0, 10.0), 20),
         ("mesolite", p["kf"], p["heights"], (18.39, 56.55, 6.54, 90, 90, 90), "F", (2.0, 4.5), 10)]
out = open("null_search.jsonl", "a")
for name, kf, h, cell, cent, band, n in cases:
    for k in range(n):
        rng = np.random.default_rng(1000 + k)
        phi = rng.uniform(0, 2 * np.pi, len(kf))
        lab = kf + np.array([0, 0, 1.0])            # q = kf - ki  ->  kf (beam along +z)
        c, s = np.cos(phi), np.sin(phi)
        kf_rot = np.column_stack([c * lab[:, 0] - s * lab[:, 1], s * lab[:, 0] + c * lab[:, 1], lab[:, 2]])
        q = kf_rot - np.array([0, 0, 1.0])
        o = CalculateUB(*primitive_cell(*cell, cent))
        t = time.perf_counter()
        o.find_orientation(q, band, heights=h, n_proc=16, n_restarts=1, accept_log10_p=None, seed=k + 1)
        sig = {tol: o.index_significance(q[o.informative], band, angle_tol=np.deg2rad(tol)) for tol in (0.2, 0.35)}
        rec = {"case": name, "scramble": k, "seconds": round(time.perf_counter() - t, 1), "n_search": int(len(o.informative)),
               **{f"log10p_{t}": round(v["log10_p_value"], 2) for t, v in sig.items()},
               **{f"indexed_{t}": v["indexed"] for t, v in sig.items()},
               **{f"chance_{t}": round(v["expected_chance"], 1) for t, v in sig.items()}}
        out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
print("DONE", flush=True)
