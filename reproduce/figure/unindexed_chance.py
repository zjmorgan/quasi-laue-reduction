import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import numpy as np
from scipy.spatial.transform import Rotation
d = np.load("t4l_1816.npz"); kf = np.load("../peaks_1816_detcal.npz")["kf_ki_dir"]
UB, idx = d["UB"], d["indexed"]
u = kf / np.linalg.norm(kf, axis=1, keepdims=True); s2 = np.linalg.norm(kf, axis=1)
r = np.arange(-60, 61); H0 = np.array(np.meshgrid(r, r, np.arange(-90, 91), indexing="ij")).reshape(3, -1).T
def frac(UBm, sel, lo, hi, tol=0.35):
    q = H0 @ UBm.T; qn = np.linalg.norm(q, axis=1); k = (qn > 0) & (qn < 1 / 0.9)
    qu = q[k] / qn[k, None]; qn = qn[k]; hit = 0
    for i in np.flatnonzero(sel):
        l = s2[i] / qn; m = (l >= lo) & (l <= hi)
        if m.any() and np.rad2deg(np.arccos(np.clip((qu[m] @ u[i]).max(), -1, 1))) < tol: hit += 1
    return hit / sel.sum()
rng = np.random.default_rng(1); U = UB @ np.linalg.inv(np.linalg.qr(UB)[1])  # not used
B = np.linalg.qr(UB)[1]
B = np.linalg.cholesky(UB.T @ UB).T   # upper-triangular with same metric
rand = [Rotation.random(random_state=rng).as_matrix() @ B for _ in range(10)]
for lo, hi in ((1.0, 2.0), (2.0, 10.0)):
    print(f"lambda {lo}-{hi}: unindexed hit, solution {frac(UB, ~idx, lo, hi):.2f} | random:",
          np.round([frac(R, ~idx, lo, hi) for R in rand], 2))
