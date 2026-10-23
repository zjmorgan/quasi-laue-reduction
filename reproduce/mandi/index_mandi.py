"""Blind quasi-Laue indexing of each MANDI garnet run; compare with the time-of-flight UB."""
import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, json, time
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB, primitive_cell, to_conventional

d = np.load("mandi_garnet.npz")
cell, UBref = tuple(d["cell"]), d["UB"]
BAND, CENT = (1.0, 2.0), "I"
B = CalculateUB(*cell).reciprocal_lattice_B()
Uref = UBref @ np.linalg.inv(B)
# proper rotations of the cubic lattice (24)
ops = [m for m in Rotation.create_group("O").as_matrix()]
out = open("index_mandi.jsonl", "a")
done = set()
try:
    done = {json.loads(l)["run"] for l in open("index_mandi.jsonl")}
except Exception:
    pass
for run in np.unique(d["run"]):
    if int(run) in done:
        continue
    m = d["run"] == run
    R = d["R"][m][0]
    dlab = d["kf"][m] - np.array([0, 0, 1.0])
    kf = dlab @ R                       # sample frame, R^T d
    h = d["intens"][m]
    t = time.perf_counter()
    prim = CalculateUB(*primitive_cell(*cell, CENT))
    prim.find_orientation(kf, BAND, heights=np.maximum(h, 1), n_proc=16, n_restarts=3, seed=1)
    conv, _ = to_conventional(prim, CENT, cell)
    dt = time.perf_counter() - t
    _, num, hkl, lam = conv.index(kf, BAND, angle_tol=np.deg2rad(0.35))
    U = conv.orientation_U(*conv.x)
    mis = min(Rotation.from_matrix(U @ g @ Uref.T).magnitude() for g in ops)
    # hkl agreement with the time-of-flight indexing (stored hkl are Friedel mates)
    href = -d["hkl"][m]
    G = U.T @ Uref                        # maps reference hkl frame to ours (a symmetry op if mis ~ 0)
    match = np.mean(np.all(np.rint(href @ np.rint(G).T) == hkl, axis=1)) if mis < np.deg2rad(2) else 0.0
    ang = np.rad2deg(conv.last_angle_error)
    ok = np.any(hkl != 0, axis=1)
    lam_err = np.median(np.abs(lam[ok] - d["lam"][m][ok]) / d["lam"][m][ok]) if mis < np.deg2rad(2) and ok.any() else np.nan
    rec = {"run": int(run), "n_peaks": int(m.sum()), "time_s": round(dt, 2), "indexed": int(num),
           "log10p": round(prim.significance["log10_p_value"], 1), "chance": round(prim.significance["expected_chance"], 1),
           "misorientation_deg": round(float(np.rad2deg(mis)), 3), "hkl_match": round(float(match), 3),
           "median_residual_deg": round(float(np.median(ang[ok])), 3) if ok.any() else None,
           "median_rel_lambda_err": None if np.isnan(lam_err) else round(float(lam_err), 4)}
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(json.dumps(rec), flush=True)
print("DONE", flush=True)
