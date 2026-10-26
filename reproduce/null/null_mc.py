"""Monte Carlo null distribution of the indexed count for uniformly random orientations,
compared with the analytical Poisson model of Section 3.6."""
import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, json, time
import numpy as np
import scipy.stats
from scipy.spatial.transform import Rotation
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB, primitive_cell

W = (WORK_DIR + "/")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100000

def datasets():
    m = np.load(W + "figure/t4l_1816_matched.npz")
    yield "T4 lysozyme (informative)", m["kf"][m["informative"]], (61.1, 61.1, 96.84, 90, 90, 120), "P", (2.0, 10.0)
    p = np.load(W + "robust/mesolite_peaks.npz")
    yield "mesolite", p["kf"], (18.39, 56.55, 6.54, 90, 90, 90), "F", (2.0, 4.5)
    g = np.load(W + "mandi/mandi_garnet.npz")
    for run in np.unique(g["run"])[[0, 40, 80]]:
        s = g["run"] == run
        yield f"garnet MANDI {run}", (g["kf"][s] - [0, 0, 1.0]) @ g["R"][s][0], tuple(g["cell"]), "I", (1.0, 3.0)

out = {}
rng = np.random.default_rng(0)
for name, kf, cell, cent, band in datasets():
    o = CalculateUB(*primitive_cell(*cell, cent))
    o.prepare_indexer(kf, band, ambiguity_power=0.0, force=True)
    Us = Rotation.random(N, random_state=rng).as_matrix()
    for tol_deg in (0.2, 0.35):
        tol = np.deg2rad(tol_deg)
        t = time.perf_counter()
        counts = np.array([o._index_from_U(U, kf, band, angle_tol=tol, score_only=True)[1] for U in Us])
        p = -np.expm1(-o._n_dirs * tol**2 / 4)
        lam, var_pb = p.sum(), (p * (1 - p)).sum()
        ks = np.arange(counts.max() + 6)
        emp_tail = np.array([(counts >= k).mean() for k in ks])
        poi_tail = scipy.stats.poisson.sf(ks - 1, lam)
        rec = {"n_peaks": len(kf), "tol_deg": tol_deg, "n_orient": N, "seconds": round(time.perf_counter() - t, 1),
               "lambda": round(lam, 2), "emp_mean": round(counts.mean(), 2),
               "var_poisson": round(lam, 2), "var_poisson_binomial": round(var_pb, 2), "emp_var": round(counts.var(), 2),
               "emp_max": int(counts.max()), "emp_p999": float(np.percentile(counts, 99.9))}
        # tail comparison at the empirical 99th and 99.9th percentiles
        for q in (99, 99.9):
            k = int(np.ceil(np.percentile(counts, q)))
            rec[f"k_q{q}"] = k
            rec[f"tail_emp_q{q}"] = float((counts >= k).mean())
            rec[f"tail_poisson_q{q}"] = float(scipy.stats.poisson.sf(k - 1, lam))
        print(name, json.dumps(rec), flush=True)
        out[f"{name}|{tol_deg}"] = dict(counts=counts, lam=lam, p=p)
np.savez_compressed("null_mc.npz", **{k: v["counts"] for k, v in out.items()},
                    **{k + "|lam": np.array(v["lam"]) for k, v in out.items()},
                    **{k + "|p": v["p"] for k, v in out.items()})
print("DONE", flush=True)
