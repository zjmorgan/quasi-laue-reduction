"""Reliability: independent seeds for T4 lysozyme (IMAGINE-X, matched peaks) and mesolite (IMAGINE image plate)."""
import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, json, time, os
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB, primitive_cell, to_conventional

out = open("seeds.jsonl", "a")

def run_seeds(name, kf, heights, cell, band, cent, group, Uref=None, seeds=range(1, 11)):
    ops = list(Rotation.create_group(group).as_matrix())
    B = CalculateUB(*cell).reciprocal_lattice_B()
    for seed in seeds:
        t = time.perf_counter()
        prim = CalculateUB(*primitive_cell(*cell, cent))
        prim.find_orientation(kf, band, heights=heights, n_proc=16, n_restarts=3, seed=seed)
        conv, _ = to_conventional(prim, cent, cell)
        dt = time.perf_counter() - t
        U = conv.orientation_U(*conv.x)
        if Uref is None:
            Uref = U
        mis = min(Rotation.from_matrix(U @ g @ Uref.T).magnitude() for g in ops)
        sig = prim.significance
        rec = {"dataset": name, "seed": seed, "time_s": round(dt, 1), "restarts": len(prim.restarts),
               "log10p": round(sig["log10_p_value"], 1), "indexed": sig["indexed"], "n_search": sig["n_informative"],
               "misorientation_deg": round(float(np.rad2deg(mis)), 3)}
        out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)

# T4 lysozyme: matched-filter peaks, refined room-temperature cell; reference = refined solution
m = np.load("../figure/t4l_1816_matched.npz")
cellT = (61.1, 61.1, 96.84, 90, 90, 120)
UrefT = m["UB"] @ np.linalg.inv(CalculateUB(*cellT).reciprocal_lattice_B())
run_seeds("T4 lysozyme", m["kf"], m["heights"], cellT, (2.0, 10.0), "P", "D6", UrefT)

# mesolite: peaks from the image plate (saved by mesolite_peaks.py)
p = np.load("mesolite_peaks.npz")
run_seeds("mesolite", p["kf"], p["heights"], (18.39, 56.55, 6.54, 90, 90, 90), (2.0, 4.5), "F", "D2")
print("DONE", flush=True)
