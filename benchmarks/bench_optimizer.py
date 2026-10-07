"""
Differential evolution against particle swarm optimization for the global
orientation search. Everything except the optimizer is identical: the
informative-peak selection, coarse subset, objective, k-d tree, persistent
worker pool and the reassignment/Wahba refinement of find_orientation. PSO
replaces scipy.optimize.differential_evolution with the same signature and
evaluation budget (population popsize * 3, maxiter generations). One search
per run (no restarts); success means the refined orientation is within 1 deg
of the truth (up to lattice symmetry) and log10 p < -5.
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import scipy.optimize

from common import save
from quasi_laue_reduction.optimize import CalculateUB
from quasi_laue_reduction.simulate import misorientation_deg, rotational_symmetry_ops, simulate_laue

CELL = (61.1, 61.1, 96.84, 90.0, 90.0, 120.0)
BAND = (2.0, 10.0)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DE = scipy.optimize.differential_evolution
COUNT = {}


def de(*args, **kwargs):
    sol = _DE(*args, **kwargs)
    COUNT["nfev"] = int(sol.nfev)
    return sol


def make_pso(w, c1, c2):
    def pso(func, bounds, maxiter=1000, popsize=15, rng=None, workers=1, x0=None, **_):
        r = np.random.default_rng(rng)
        n, d = popsize * len(bounds), len(bounds)
        evaluate = (lambda X: np.asarray(workers(func, X), float)) if callable(workers) else (lambda X: np.array([func(x) for x in X]))
        X = r.random((n, d))
        if x0 is not None:
            X[0] = x0
        V = (r.random((n, d)) - 0.5) * 0.2
        F = evaluate(X)
        P, PF = X.copy(), F.copy()
        g = int(np.argmin(PF))
        nfev = n
        for _ in range(maxiter - 1):
            r1, r2 = r.random((n, d)), r.random((n, d))
            V = w * V + c1 * r1 * (P - X) + c2 * r2 * (P[g] - X)
            V = np.clip(V, -0.5, 0.5)
            X = X + V
            X[:, 1] %= 1.0                      # azimuth of the axis is periodic
            for k in (0, 2):                    # reflect at the other bounds
                lo, hi = X[:, k] < 0, X[:, k] > 1
                X[lo, k], X[hi, k] = -X[lo, k], 2 - X[hi, k]
                V[lo | hi, k] *= -1
            X = np.clip(X, 0, 1)
            F = evaluate(X)
            nfev += n
            better = F < PF
            P[better], PF[better] = X[better], F[better]
            g = int(np.argmin(PF))
        COUNT["nfev"] = nfev
        return scipy.optimize.OptimizeResult(x=P[g].copy(), fun=float(PF[g]), nfev=nfev, success=True)
    return pso


def problems(n_synth, rng):
    ops = rotational_symmetry_ops(CELL)
    ref = CalculateUB(*CELL)
    for k in range(n_synth):
        x = rng.random(3)
        kf, h, _ = simulate_laue(CELL, x, BAND, d_min=2.0, n_peaks=300, noise_deg=0.15, rng=int(rng.integers(1 << 30)))
        yield f"synthetic{k}", kf, h, ref.orientation_U(*x), ops
    d = np.load(os.path.join(ROOT, "work/figure/t4l_1816.npz"))
    p = np.load(os.path.join(ROOT, "work/peaks_1816_detcal.npz"))
    yield "run1816", p["kf_ki_dir"], p["heights"], d["UB"] @ np.linalg.inv(ref.reciprocal_lattice_B()), ops


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("--n-synth", type=int, default=6)
    a.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a.add_argument("--budgets", nargs="+", default=["60x100", "60x300", "300x300"])
    a.add_argument("--n-proc", type=int, default=16)
    args = a.parse_args()

    methods = {"DE": de}
    for w, c in itertools.product((0.5, 0.72), (1.0, 1.49)):
        methods[f"PSO w={w} c={c}"] = make_pso(w, c, c)

    for name, kf, h, U_true, ops in problems(args.n_synth, np.random.default_rng(7)):
        for budget in args.budgets:
            pop, it = (int(v) for v in budget.split("x"))
            for method, fn in methods.items():
                for seed in args.seeds:
                    scipy.optimize.differential_evolution = fn
                    opt = CalculateUB(*CELL)
                    t = time.perf_counter()
                    opt.find_orientation(kf, BAND, heights=h, n_proc=args.n_proc, seed=seed, n_restarts=1,
                                         accept_log10_p=None, popsize=pop, maxiter=it)
                    dt = time.perf_counter() - t
                    mis = misorientation_deg(U_true, opt.orientation_U(*opt.x), ops)
                    lp = opt.significance["log10_p_value"]
                    rec = {"problem": name, "method": method, "popsize": pop, "maxiter": it, "seed": seed,
                           "nfev": COUNT.get("nfev"), "seconds": round(dt, 2), "misorientation_deg": round(float(mis), 3),
                           "log10p": round(lp, 2), "success": bool(mis < 1.0 and lp < -5)}
                    save("optimizer", rec)
                    print(json.dumps(rec), flush=True)
    scipy.optimize.differential_evolution = _DE
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
