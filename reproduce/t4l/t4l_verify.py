"""1816 with the garnet DetCal: seed reproducibility and a PDF with predictions."""

import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import json

import numpy as np
from mantid.simpleapi import DeleteWorkspace, mtd

from quasi_laue_reduction.loading import calibration_file, load_lite
from quasi_laue_reduction.optimize import CalculateUB
from quasi_laue_reduction.simulate import misorientation_deg, rotational_symmetry_ops
from quasi_laue_reduction.workflow import QuasiLaue

CELL = (61.5, 61.5, 95.9, 90.0, 90.0, 120.0)
BAND = (2.0, 10.0)
OUT = (WORK_DIR + "/")

info = load_lite("/HFIR/CG4D/IPTS-37331/shared/autoreduce/CG4D_1816.lite.nxs.h5", ws="data", detcal=calibration_file())
ql = QuasiLaue("data", pixel_shape=info["pixel_shape"])
ql.find_peaks(method="local", n_sigma=8)
kf, h = ql.kf_ki_dir.copy(), ql.heights.copy()
np.savez(OUT + "peaks_1816_detcal.npz", kf_ki_dir=kf, heights=h, coords=ql.coords)

ops = rotational_symmetry_ops(CELL)
sols = []
for seed in (1, 11, 21, 31):
    o = CalculateUB(*CELL)
    _, _, _, num = o.find_orientation(kf, BAND, heights=h, n_proc=16, seed=seed, n_restarts=6)
    sols.append({"seed": seed, "indexed": num, "log10p": o.significance["log10_p_value"],
                 "restarts": len(o.restarts), "U": o.orientation_U(*o.x)})
    print(json.dumps({k: v for k, v in sols[-1].items() if k != "U"}), flush=True)

mis = [[round(misorientation_deg(a["U"], b["U"], ops), 2) for b in sols] for a in sols]
print("misorientation matrix (deg):", json.dumps(mis), flush=True)
np.save(OUT + "U_1816_detcal_seeds.npy", np.array([s["U"] for s in sols]))

# PDF for the seed-1 solution with predicted reflections
ql.find_UB(*CELL, BAND, centering="P", n_proc=16, seed=1, n_restarts=6)
for d_min in (3.0,):
    pred = ql.predicted_pixels(BAND, d_min=d_min)
    ql.plot_peaks_pdf(OUT + f"CG4D_1816_peaks_detcal_dmin{d_min:g}.pdf", predicted=pred,
                      title=f"CG4D_1816 T4L with garnet DetCal, log10p {ql.significance['log10_p_value']:.1f}, pred d>{d_min:g}")
print("PDF written", flush=True)
for name in list(mtd.getObjectNames()):
    DeleteWorkspace(name)
