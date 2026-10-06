"""
k-d tree objective distributed with SciPy's built-in pool (workers=N), for
comparison with the persistent pool: same peaks, seed, subset and DE
settings as bench_search.py. The optimizer is pickled with its k-d tree, so
each task re-sends the prepared data but does not rebuild it.
"""

import time

import numpy as np

import quasi_laue_reduction.optimize as qopt
from common import arguments, load_peaks, save
from quasi_laue_reduction.optimize import CalculateUB


class TreeWithState(CalculateUB):
    def __getstate__(self):
        return self.__dict__.copy()


class SciPyWorkers(int):
    """Stands in for ChunkedPoolMap: SciPy then uses its own multiprocessing pool."""

    def __new__(cls, opt, n_proc, start_method=None):
        return super().__new__(cls, n_proc)

    def close(self):
        pass


def extra(p):
    p.add_argument("--n-proc", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
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

    qopt.ChunkedPoolMap = SciPyWorkers

    for n_proc in args.n_proc:
        opt = TreeWithState(*cell)
        t = time.perf_counter()
        opt.minimize(kf, band, n_proc=n_proc, peak_weights=w, n_reassign=0,
                     popsize=args.popsize, maxiter=args.maxiter, seed=args.seed)
        elapsed = time.perf_counter() - t
        _, num, _, _ = opt._index_from_U(opt.orientation_U(*opt.x), kf, band)
        save("search", {
            "variant": "tree_scipy", "n_proc": n_proc, "popsize": args.popsize,
            "maxiter": args.maxiter, "seed": args.seed, "seconds": elapsed,
            "indexed": int(num), "n_peaks": len(kf), "x": np.asarray(opt.x).tolist(),
        })
        print(n_proc, round(elapsed, 1), num, flush=True)


if __name__ == "__main__":
    main()
