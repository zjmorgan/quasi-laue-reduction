"""
Objective comparison: recovery rate of the true orientation over seeds for
the threshold score (at the final tolerance and with a wider search
tolerance) and the significance score, on synthetic data with realistic
angular noise (default: T4 lysozyme, full 2-10 A band, 0.2 degree noise).
"""

import time

import numpy as np

from common import T4L, X_TRUE, save
from quasi_laue_reduction.optimize import CalculateUB
from quasi_laue_reduction.simulate import misorientation_deg, rotational_symmetry_ops, simulate_laue

import argparse

CONFIGS = {
    "threshold": dict(score="threshold"),
    "threshold_search1deg": dict(score="threshold", search_tol_deg=1.0),
    "significance": dict(score="significance"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", nargs=6, type=float, default=T4L)
    p.add_argument("--band", nargs=2, type=float, default=(2.0, 10.0))
    p.add_argument("--noise", type=float, default=0.2)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--n-proc", type=int, default=16)
    args = p.parse_args()

    cell, band = tuple(args.cell), tuple(args.band)
    kf, h, _ = simulate_laue(cell, X_TRUE, band, d_min=2.5, n_peaks=99, noise_deg=args.noise, rng=11)
    ops = rotational_symmetry_ops(cell)
    U_true = CalculateUB(*cell).orientation_U(*X_TRUE)

    for name in args.configs:
        for seed in args.seeds:
            opt = CalculateUB(*cell)
            t = time.perf_counter()
            _, _, _, num = opt.find_orientation(kf, band, heights=h, n_proc=args.n_proc, seed=seed, **CONFIGS[name])
            mis = misorientation_deg(U_true, opt.orientation_U(*opt.x), ops)
            sig = opt.index_significance(kf, band, angle_tol=np.deg2rad(0.35))
            save("cost", {"config": name, "band": band, "noise_deg": args.noise, "seed": seed, "indexed": int(num),
                          "misorientation_deg": mis, "recovered": bool(mis < 0.5), "seconds": time.perf_counter() - t,
                          "log10_p_value": sig["log10_p_value"], "expected_chance": sig["expected_chance"]})


if __name__ == "__main__":
    main()
