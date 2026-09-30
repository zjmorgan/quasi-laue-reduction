"""
Time one objective evaluation: legacy per-peak shell scan vs vectorised scan
vs stacked KD-tree, on the same peaks, weights and random orientations.
"""

import time

import numpy as np

import legacy_optimize_refine_ub as legacy
from common import arguments, load_peaks, save
from quasi_laue_reduction import optimize
from quasi_laue_reduction.optimize import CalculateUB


def per_eval_ms(opt, kf, band, xs):
    t = time.perf_counter()
    for x in xs:
        opt._index_from_U(opt.orientation_U(*x), kf, band, score_only=True)
    return (time.perf_counter() - t) / len(xs) * 1e3


def main():
    args = arguments(__doc__, lambda p: p.add_argument("--n-coarse", type=int, default=80))
    kf, h, _ = load_peaks(args)
    band = tuple(args.band)
    cell = tuple(args.cell)

    w = CalculateUB.make_peak_weights(h)
    sub = CalculateUB(*cell).select_coarse_subset(kf, band, w, n_coarse=args.n_coarse)
    kf, w = kf[sub], w[sub]

    xs = np.random.default_rng(0).random((300, 3))
    rec = {"n_peaks": len(kf), "cell": cell, "band": band}

    old = legacy.CalculateUB(*cell)
    old.prepare_indexer(kf, band, peak_weights=w, force=True)
    rec["candidates_per_peak"] = float(np.mean(old._candidate_counts))
    rec["legacy_ms"] = per_eval_ms(old, kf, band, xs)

    for name, limit in (("scan_ms", 0), ("tree_ms", optimize.MAX_TREE_CANDIDATES)):
        optimize.MAX_TREE_CANDIDATES = limit
        new = CalculateUB(*cell)
        t = time.perf_counter()
        new.prepare_indexer(kf, band, peak_weights=w, force=True)
        rec[name.replace("_ms", "_prepare_s")] = time.perf_counter() - t
        rec[name] = per_eval_ms(new, kf, band, xs)

    # same objective values
    new = CalculateUB(*cell)
    new.prepare_indexer(kf, band, peak_weights=w, force=True)
    diffs = []
    for x in xs[:50]:
        U = new.orientation_U(*x)
        a = old._index_from_U(U, kf, band, score_only=True)
        b = new._index_from_U(U, kf, band, score_only=True)
        assert a[1] == b[1]
        diffs.append(abs(a[0] - b[0]) / max(1.0, abs(a[0])))
    rec["max_rel_cost_diff"] = float(max(diffs))
    rec["speedup_tree_vs_legacy"] = rec["legacy_ms"] / rec["tree_ms"]

    save("indexer", rec)


if __name__ == "__main__":
    main()
