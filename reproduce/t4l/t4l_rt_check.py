"""1816 + garnet DetCal at a room-temperature T4L cell: refine c/a, overlay tests, PDFs."""

import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import json

import numpy as np
from mantid.simpleapi import SetUB, mtd
from scipy.spatial.transform import Rotation

from quasi_laue_reduction.detcal import apply_detcal
from quasi_laue_reduction.loading import _counts_workspace, calibration_file
from quasi_laue_reduction.optimize import CalculateUB
from quasi_laue_reduction.peaks import find_peaks_local
from quasi_laue_reduction.workflow import QuasiLaue

W = (WORK_DIR + "/")
BAND = (2.0, 10.0)

# ---- refine c/a from the 61.1/97.0 solution (a fixed)
d = np.load(W + "peaks_1816_detcal.npz")
kf, h = d["kf_ki_dir"], d["heights"]
o = CalculateUB(61.1, 61.1, 97.0, 90, 90, 120)
o.x = o.orientation_parameters_from_U(np.load(W + "U_1816_61.1_97.0.npy"))
for tol in (0.5, 0.35, 0.35, 0.2, 0.2):
    o._score_angle_tol = np.deg2rad(tol)
    _, _, _, sig = o.refine(kf, BAND, "Hexagonal", error=0.05)
stats = {t: o.index_significance(kf, BAND, angle_tol=np.deg2rad(t)) for t in (0.2, 0.35, 0.5)}
cell = o.get_lattice_constants()
print(json.dumps({"refined_c_over_a": round(o.c / o.a, 4), "c_at_a_61.1": round(o.c, 2),
                  **{f"at_{t}": f"{s['indexed']}/{len(kf)} vs {s['expected_chance']:.0f} log10p {s['log10_p_value']:.1f}" for t, s in stats.items()}}), flush=True)
UB = o.orientation_U(*o.x) @ o.reciprocal_lattice_B()

# ---- workspace from saved images with the DetCal
images = np.load(W + "images_1816.npz")["images"].astype(float)
xml = open(W + "garnet/lite_corrected.xml").read()
_counts_workspace("data", images.ravel(), idf_xml=xml, instrument="IMAGINE")
apply_detcal("data", calibration_file(), images.shape[1:])
ql = QuasiLaue("data", pixel_shape=images.shape[1:])
ql.find_peaks(method="local", n_sigma=8)
strong = ql.found_coords.copy()
ql.centering = "P"
spots4, _, _ = find_peaks_local(images, n_sigma=4)


def near(a, b, r):
    out = np.zeros(len(a), dtype=bool)
    for bank in np.unique(a[:, 0]):
        ma, mb = a[:, 0] == bank, b[:, 0] == bank
        if mb.any():
            dd = np.hypot(a[ma, 1][:, None] - b[mb, 1][None], a[ma, 2][:, None] - b[mb, 2][None])
            out[np.flatnonzero(ma)] = dd.min(axis=1) <= r
    return out


faint = spots4[~near(spots4, strong, 2.0)]
inner = lambda c: c[(c[:, 1] >= 10) & (c[:, 1] <= 117) & (c[:, 2] >= 10) & (c[:, 2] <= 117)]

B = CalculateUB(*cell).reciprocal_lattice_B()
rng = np.random.default_rng(0)
UBs = {"solution": UB, **{f"random{k}": Rotation.random(random_state=rng).as_matrix() @ B for k in range(20)}}
# the peaks workspace needs the refined lattice for prediction
from mantid.simpleapi import SetUB as _S
res = {}
for name, M in UBs.items():
    SetUB(Workspace=ql.peaks_ws, UB=M.ravel().tolist())
    low = inner(ql.predicted_pixels(BAND, centering="P", d_min=10.0))
    dense = ql.predicted_pixels(BAND, centering="P", d_min=2.5)
    res[name] = {"low_order_predicted": int(len(low)),
                 "low_order_on_a_spot": float(np.mean(near(low, spots4, 1.5))) if len(low) else np.nan,
                 "faint_spots_on_prediction": float(np.mean(near(faint, dense, 1.0)))}
sol = res.pop("solution")
rand = {k: [v[k] for v in res.values()] for k in sol}
print(json.dumps({"faint_spots": int(len(faint)), "solution": sol,
                  "random_mean": {k: round(float(np.nanmean(v)), 3) for k, v in rand.items()},
                  "random_max": {k: round(float(np.nanmax(v)), 3) for k, v in rand.items()}}, indent=1), flush=True)

# ---- PDFs with the refined room-temperature solution
SetUB(Workspace=ql.peaks_ws, UB=UB.ravel().tolist())
for dmin in (3.0, 10.0):
    pred = ql.predicted_pixels(BAND, centering="P", d_min=dmin)
    ql.plot_peaks_pdf(W + f"CG4D_1816_peaks_RTcell_dmin{dmin:g}.pdf", predicted=pred,
                      title=f"CG4D_1816 T4L RT cell c/a {o.c / o.a:.3f} + DetCal, pred d>{dmin:g}")
print("PDFs written", flush=True)
