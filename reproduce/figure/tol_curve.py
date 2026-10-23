import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, numpy as np
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB
d = np.load("t4l_1816.npz"); kf = np.load("../peaks_1816_detcal.npz")["kf_ki_dir"]
a, b, c, al, be, ga = d["cell"]
o = CalculateUB(a, b, c, al, be, ga)
B = o.reciprocal_lattice_B(); U = d["UB"] @ np.linalg.inv(B)
o.x = o.orientation_parameters_from_U(U)
tols = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.6, 0.8, 1.0])
rows = [o.index_significance(kf, (2.0, 10.0), angle_tol=np.deg2rad(t)) for t in tols]
obs = np.array([r["indexed"] for r in rows]); lam = np.array([r["expected_chance"] for r in rows])
lp = np.array([r["log10_p_value"] for r in rows])
for t, n, l, p in zip(tols, obs, lam, lp): print(t, n, round(l, 1), round(p, 1))
np.savez("tol_curve.npz", tols=tols, obs=obs, chance=lam, log10p=lp, n=len(kf))
