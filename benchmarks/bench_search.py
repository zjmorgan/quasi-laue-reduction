"""
Wall time of the differential-evolution stage, legacy (per-peak scan, SciPy
``workers=N``) vs current (KD-tree, persistent pool), for several worker
counts, with the same seed, weights and coarse subset.
"""

import time

import numpy as np
import scipy.optimize

import legacy_optimize_refine_ub as legacy
from common import arguments, load_peaks, save
from quasi_laue_reduction.optimize import CalculateUB


def extra(p):
    p.add_argument("--n-proc", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--variants", nargs="+", default=["current", "legacy"])
    p.add_argument("--popsize", type=int, default=60)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)


def main():
    args = arguments(__doc__, extra)
    kf, h, _ = load_peaks(args)
    band, cell = tuple(args.band), tuple(args.cell)

    w = CalculateUB.make_peak_weights(h)
    sub = CalculateUB(*cell).select_coarse_subset(kf, band, w)
    kf, w = kf[sub], w[sub]

    de = scipy.optimize.differential_evolution

    def seeded(*a, **k):
        k.setdefault("rng", args.seed)
        return de(*a, **k)

    legacy.scipy.optimize.differential_evolution = seeded

    for variant in args.variants:
        for n_proc in args.n_proc:
            if variant == "legacy":
                opt = legacy.CalculateUB(*cell)
                kw = {}
            else:
                opt = CalculateUB(*cell)
                kw = {"seed": args.seed}

            t = time.perf_counter()
            opt.minimize(
                kf, band, n_proc=n_proc, peak_weights=w, n_reassign=0,
                popsize=args.popsize, maxiter=args.maxiter, **kw,
            )
            elapsed = time.perf_counter() - t

            _, num, _, _ = opt._index_from_U(opt.orientation_U(*opt.x), kf, band)

            save("search", {
                "variant": variant, "n_proc": n_proc, "popsize": args.popsize,
                "maxiter": args.maxiter, "seed": args.seed, "seconds": elapsed,
                "indexed": int(num), "n_peaks": len(kf), "x": np.asarray(opt.x).tolist(),
            })


if __name__ == "__main__":
    main()
