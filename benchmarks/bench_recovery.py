"""
Search reliability: fraction of seeds that recover the true orientation for
several DE sizes, plus the random-orientation null distribution of the
indexed count.
"""

import time

import numpy as np

from common import arguments, load_peaks, save
from quasi_laue_reduction.optimize import CalculateUB
from quasi_laue_reduction.simulate import misorientation_deg, rotational_symmetry_ops


def extra(p):
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--sizes", nargs="+", default=["60x300", "300x1000"], help="popsize x maxiter")
    p.add_argument("--n-proc", type=int, default=-1)


def main():
    args = arguments(__doc__, extra)
    kf, h, x_true = load_peaks(args)
    band, cell = tuple(args.band), tuple(args.cell)

    null = CalculateUB(*cell)
    null.prepare_indexer(kf, band, force=True)
    counts = [null._index_from_U(null.orientation_U(*x), kf, band)[1] for x in np.random.default_rng(0).random((2000, 3))]
    save("recovery_null", {"cell": cell, "band": band, "n_peaks": len(kf),
                           "p50": float(np.median(counts)), "p999": float(np.percentile(counts, 99.9)), "max": int(max(counts))})

    ops = rotational_symmetry_ops(cell)

    for size in args.sizes:
        pop, it = (int(v) for v in size.split("x"))
        for seed in args.seeds:
            opt = CalculateUB(*cell)
            t = time.perf_counter()
            _, _, _, num = opt.find_orientation(kf, band, heights=h, n_proc=args.n_proc, seed=seed, popsize=pop, maxiter=it)
            rec = {"popsize": pop, "maxiter": it, "seed": seed, "indexed": int(num), "n_peaks": len(kf),
                   "seconds": time.perf_counter() - t}
            if x_true is not None:
                rec["misorientation_deg"] = misorientation_deg(opt.orientation_U(*x_true), opt.orientation_U(*opt.x), ops)
            save("recovery", rec)


if __name__ == "__main__":
    main()
