"""Blind indexing of MANDI garnet settings with added spurious peaks (directions from other settings)."""
import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, json, time
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB, primitive_cell, to_conventional

d = np.load("../mandi/mandi_garnet.npz")
cell, UBref = tuple(d["cell"]), d["UB"]
BAND, CENT = (1.0, 3.0), "I"
Uref = UBref @ np.linalg.inv(CalculateUB(*cell).reciprocal_lattice_B())
ops = list(Rotation.create_group("O").as_matrix())
runs = np.unique(d["run"])[::5][:20]
rng = np.random.default_rng(0)
out = open("spurious_mandi.jsonl", "a")
for frac in (0.1, 0.25, 0.5, 1.0):
    for run in runs:
        m = d["run"] == run
        R = d["R"][m][0]
        true_lab = d["kf"][m]
        pool = np.flatnonzero(d["run"] != run)
        n_sp = int(round(frac * m.sum()))
        sp_lab = d["kf"][rng.choice(pool, n_sp, replace=False)]
        lab = np.vstack([true_lab, sp_lab])
        kf = (lab - [0, 0, 1.0]) @ R
        is_true = np.r_[np.ones(m.sum(), bool), np.zeros(n_sp, bool)]
        heights = np.r_[np.maximum(d["intens"][m], 1), rng.choice(np.maximum(d["intens"][m], 1), n_sp)]
        t = time.perf_counter()
        prim = CalculateUB(*primitive_cell(*cell, CENT))
        prim.find_orientation(kf, BAND, heights=heights, n_proc=16, n_restarts=3, seed=1)
        conv, _ = to_conventional(prim, CENT, cell)
        dt = time.perf_counter() - t
        _, num, hkl, _ = conv.index(kf, BAND, angle_tol=np.deg2rad(0.35))
        ok = np.any(hkl != 0, axis=1)
        U = conv.orientation_U(*conv.x)
        mis = min(Rotation.from_matrix(U @ g @ Uref.T).magnitude() for g in ops)
        rec = {"frac": frac, "run": int(run), "n_true": int(m.sum()), "n_spurious": n_sp, "time_s": round(dt, 1),
               "true_indexed": float(ok[is_true].mean()), "spurious_indexed": float(ok[~is_true].mean()) if n_sp else None,
               "log10p": round(prim.significance["log10_p_value"], 1), "chance": round(prim.significance["expected_chance"], 1),
               "misorientation_deg": round(float(np.rad2deg(mis)), 3), "restarts": len(prim.restarts)}
        out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
