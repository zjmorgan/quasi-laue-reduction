"""Add per-peak angular error (deg, nearest in-band candidate) to t4l_1816_matched.npz."""
import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, numpy as np
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB
m = dict(np.load("t4l_1816_matched.npz"))
o = CalculateUB(*m["cell"]); o.x = o.orientation_parameters_from_U(m["UB"] @ np.linalg.inv(o.reciprocal_lattice_B()))
o.index(m["kf"], (2.0, 10.0), angle_tol=np.deg2rad(float(m["tol95"])))
err = np.rad2deg(o.last_angle_error)
m["angle_error"] = err
np.savez_compressed("t4l_1816_matched.npz", **m)
print("angle error quantiles (deg):", np.round(np.percentile(err, [10, 25, 50, 75, 90, 99]), 3), "no candidates:", int(np.sum(err > 90)))
