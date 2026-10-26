import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, numpy as np
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB
d = np.load("t4l_1816.npz"); kf = np.load("../peaks_1816_detcal.npz")["kf_ki_dir"]
idx, h, coords = d["indexed"], d["heights"], d["coords"]
s2 = np.linalg.norm(kf, axis=1); tt = np.rad2deg(2 * np.arcsin(s2 / 2))
print("height median: indexed %.0f unindexed %.0f" % (np.median(h[idx]), np.median(h[~idx])))
print("height rank of unindexed (percentile among all):", np.round(np.median([np.mean(h < x) for x in h[~idx]]), 2))
print("2theta sorted unindexed:", np.round(np.sort(tt[~idx]), 1))
# fraction indexed by 2theta bin
for lo, hi in ((0, 25), (25, 35), (35, 45), (45, 60), (60, 90), (90, 180)):
    m = (tt >= lo) & (tt < hi)
    if m.any(): print(f"2theta {lo:3d}-{hi:3d}: n={m.sum():3d} indexed {idx[m].mean():.2f}")
# second crystal? DE on the unindexed subset only
a, b, c, al, be, ga = d["cell"]
o = CalculateUB(a, b, c, al, be, ga)
o.find_orientation(kf[~idx], (2.0, 10.0), h[~idx], n_proc=16, n_restarts=3, seed=3)
for t in (0.2, 0.35):
    s = o.index_significance(kf[~idx], (2.0, 10.0), angle_tol=np.deg2rad(t))
    print("second-lattice search on unindexed: tol", t, s)
U2 = o.orientation_U(*o.x); U1 = d["UB"] @ np.linalg.inv(o.reciprocal_lattice_B())
from scipy.spatial.transform import Rotation
print("misorientation to main (deg, no symmetry):", np.round(Rotation.from_matrix(U2 @ U1.T).magnitude() * 180 / np.pi, 2))
