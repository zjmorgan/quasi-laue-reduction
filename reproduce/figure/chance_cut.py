import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, numpy as np, scipy.stats
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB
from quasi_laue_reduction.peaks import find_peaks_matched, find_peaks_local, interpolate_positions, scattering_directions
d = np.load("t4l_1816.npz"); images, P = d["images"], d["positions"]
o = CalculateUB(*d["cell"]); o.x = o.orientation_parameters_from_U(d["UB"] @ np.linalg.inv(o.reciprocal_lattice_B()))
for name, finder in (("matched5", lambda: find_peaks_matched(images, z_min=5)), ("local8", lambda: find_peaks_local(images, n_sigma=8))):
    c, h, _ = finder()
    kf = scattering_directions(interpolate_positions(P, c))
    T = np.deg2rad(0.35)
    _, _, hkl, _ = o.index(kf, (2.0, 10.0), angle_tol=T)
    ok = np.any(np.asarray(hkl) != 0, axis=1)
    p = -np.expm1(-o._n_dirs * T**2 / 4); pc = -np.expm1(-o._candidate_counts * T**2 / 4)
    print(name, len(kf), "p quantiles", np.round(np.percentile(p, [10, 25, 50, 75, 90]), 2))
    for cut in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01):
        for lab, pp in (("n_dirs", p), ("counts", pc)):
            m = pp <= cut
            lam = p[m].sum(); n = ok[m].sum()
            print(f"  cut {cut:4.2f} [{lab}] n={m.sum():4d} idx {n:4d} chance {lam:6.1f} log10p {scipy.stats.poisson.logsf(n - 1, lam) / np.log(10):6.1f}")
