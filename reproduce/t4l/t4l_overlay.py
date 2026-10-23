"""1816 + garnet DetCal: sparse low-order overlay and an independent faint-spot test."""

import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import json
import os

import numpy as np
from mantid.simpleapi import SetUB, mtd
from scipy.spatial.transform import Rotation

from quasi_laue_reduction.loading import calibration_file, load_lite
from quasi_laue_reduction.optimize import CalculateUB
from quasi_laue_reduction.peaks import find_peaks_local
from quasi_laue_reduction.workflow import QuasiLaue

CELL = (61.5, 61.5, 95.9, 90.0, 90.0, 120.0)
BAND = (2.0, 10.0)
OUT = (WORK_DIR + "/")

info = load_lite("/HFIR/CG4D/IPTS-37331/shared/autoreduce/CG4D_1816.lite.nxs.h5", ws="data", detcal=calibration_file())
ql = QuasiLaue("data", pixel_shape=info["pixel_shape"])
images = ql.extract_images()
if not os.path.exists(OUT + "images_1816.npz"):
    np.savez_compressed(OUT + "images_1816.npz", images=images.astype(np.float32))

ql.find_peaks(method="local", n_sigma=8)
strong = ql.found_coords.copy()
ql.find_UB(*CELL, BAND, centering="P", n_proc=16, seed=1, n_restarts=6)
UB_sol = np.array(mtd[ql.peaks_ws].sample().getOrientedLattice().getUB())

# spots at a lower threshold; "faint" = not within 2 px of a strong peak
spots4, _, _ = find_peaks_local(images, n_sigma=4)


def near(a, b, r):
    """For each row of a (bank, i, j): is any row of b in the same bank within r pixels?"""
    out = np.zeros(len(a), dtype=bool)
    for bank in np.unique(a[:, 0]):
        ma, mb = a[:, 0] == bank, b[:, 0] == bank
        if not mb.any():
            continue
        d = np.hypot(a[ma, 1][:, None] - b[mb, 1][None], a[ma, 2][:, None] - b[mb, 2][None])
        out[np.flatnonzero(ma)] = d.min(axis=1) <= r
    return out


faint = spots4[~near(spots4, strong, 2.0)]
inner = lambda c: c[(c[:, 1] >= 10) & (c[:, 1] <= 117) & (c[:, 2] >= 10) & (c[:, 2] <= 117)]

B = CalculateUB(*CELL).reciprocal_lattice_B()
rng = np.random.default_rng(0)
UBs = {"solution": UB_sol}
for k in range(20):
    UBs[f"random{k}"] = Rotation.random(random_state=rng).as_matrix() @ B

res = {}
for name, UB in UBs.items():
    SetUB(Workspace=ql.peaks_ws, UB=UB.ravel().tolist())
    low = inner(ql.predicted_pixels(BAND, centering="P", d_min=10.0))
    dense = ql.predicted_pixels(BAND, centering="P", d_min=2.5)
    res[name] = {
        "low_order_predicted": int(len(low)),
        "low_order_on_a_spot": float(np.mean(near(low, spots4, 1.5))) if len(low) else np.nan,
        "faint_spots_on_prediction": float(np.mean(near(faint, dense, 1.0))),
        "dense_predicted": int(len(dense)),
    }

sol = res.pop("solution")
rand = {k: [v[k] for v in res.values()] for k in sol}
summary = {
    "strong_peaks": int(len(strong)),
    "faint_spots": int(len(faint)),
    "solution": sol,
    "random_mean": {k: float(np.nanmean(v)) for k, v in rand.items()},
    "random_max": {k: float(np.nanmax(v)) for k, v in rand.items()},
}
print(json.dumps(summary, indent=1), flush=True)
json.dump(summary, open(OUT + "t4l_overlay.json", "w"), indent=1)

# sparse overlay PDF for the solution
SetUB(Workspace=ql.peaks_ws, UB=UB_sol.ravel().tolist())
pred = ql.predicted_pixels(BAND, centering="P", d_min=10.0)
ql.plot_peaks_pdf(OUT + "CG4D_1816_peaks_detcal_dmin10.pdf", predicted=pred,
                  title="CG4D_1816 T4L + garnet DetCal, predictions d>10 A")
print("PDF written", flush=True)
