import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, numpy as np
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB, centering_filter
d = np.load("mandi_garnet.npz"); cell = tuple(d["cell"])
o = CalculateUB(*cell); B = o.reciprocal_lattice_B(); U = d["UB"] @ np.linalg.inv(B)
o.x = o.orientation_parameters_from_U(U); o._hkl_filter = centering_filter("I")
same = par = other = unidx = 0
for run in np.unique(d["run"]):
    m = d["run"] == run
    kf = (d["kf"][m] - [0, 0, 1.0]) @ d["R"][m][0]
    href0 = -np.rint(d["hkl"][m]).astype(int)
    Ur = o._wahba_refine_U(B, kf, href0)
    o.x = o.orientation_parameters_from_U(Ur)
    o.prepare_indexer(kf, (1.0, 2.0), force=True)
    _, _, hkl, lam = o.index(kf, (1.0, 2.0), angle_tol=np.deg2rad(0.35))
    href = -np.rint(d["hkl"][m]).astype(int)
    for a, b in zip(hkl, href):
        if np.all(a == b): same += 1
        elif np.any(a) and np.linalg.norm(np.cross(a, b)) == 0: par += 1
        elif not np.any(a): unidx += 1
        else: other += 1
print("per-run U from TOF hkl: same", same, "harmonic (parallel)", par, "unindexed", unidx, "different", other)
