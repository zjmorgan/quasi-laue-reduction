"""Shared inputs for the benchmarks."""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from quasi_laue_reduction.simulate import simulate_laue  # noqa: E402

T4L = (61.5, 61.5, 95.9, 90.0, 90.0, 120.0)
WL = (2.8, 4.6)
X_TRUE = (0.41, 0.77, 0.23)

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def arguments(description, extra=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--peaks", help="npz with kf_ki_dir and heights (default: synthetic T4 lysozyme)")
    p.add_argument("--cell", nargs=6, type=float, default=T4L)
    p.add_argument("--band", nargs=2, type=float, default=WL)
    if extra:
        extra(p)
    return p.parse_args()


def load_peaks(args, n_peaks=99, seed=11):
    """
    Peaks from ``--peaks`` or a synthetic pattern; returns kf, heights, x_true.
    """
    if args.peaks:
        d = np.load(args.peaks)
        return d["kf_ki_dir"], d["heights"], d["x_true"] if "x_true" in d else None
    kf, h, _ = simulate_laue(tuple(args.cell), X_TRUE, tuple(args.band), d_min=2.5, n_peaks=n_peaks, rng=seed)
    return kf, h, np.array(X_TRUE)


def save(name, record):
    os.makedirs(RESULTS, exist_ok=True)
    record["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(RESULTS, name + ".jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record))
