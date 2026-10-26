import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, numpy as np
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB
d = np.load("t4l_1816.npz"); kf = np.load("../peaks_1816_detcal.npz")["kf_ki_dir"]
o = CalculateUB(*d["cell"]); o.x = o.orientation_parameters_from_U(d["UB"] @ np.linalg.inv(o.reciprocal_lattice_B()))
for t in np.arange(0.60, 0.76, 0.01):
    s = o.index_significance(kf, (2.0, 10.0), angle_tol=np.deg2rad(t))
    print(round(t, 2), s["indexed"], round(s["indexed"] / len(kf), 3), round(s["expected_chance"], 1), round(s["log10_p_value"], 1))
    if s["indexed"] >= 0.95 * len(kf): break
