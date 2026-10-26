"""Prepare data for the IMAGINE-X T4L figure: refined RT UB, indexed peaks, pixel angles, predictions."""

import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import numpy as np
from mantid.simpleapi import PredictPeaks, SetUB, mtd

from quasi_laue_reduction.detcal import apply_detcal
from quasi_laue_reduction.loading import _counts_workspace, calibration_file
from quasi_laue_reduction.optimize import CalculateUB
from quasi_laue_reduction.workflow import QuasiLaue

W = (WORK_DIR + "/")
BAND = (2.0, 10.0)

d = np.load(W + "peaks_1816_detcal.npz")
kf, h, coords = d["kf_ki_dir"], d["heights"], d["coords"]
o = CalculateUB(61.1, 61.1, 97.0, 90, 90, 120)
o.x = o.orientation_parameters_from_U(np.load(W + "U_1816_61.1_97.0.npy"))
for tol in (0.5, 0.35, 0.35, 0.2, 0.2):
    o._score_angle_tol = np.deg2rad(tol)
    o.refine(kf, BAND, "Hexagonal", error=0.05)
_, num, hkl, lam = o.index(kf, BAND, angle_tol=np.deg2rad(0.35))
UB = o.orientation_U(*o.x) @ o.reciprocal_lattice_B()
print("cell", np.round(o.get_lattice_constants(), 2), "indexed", num, len(kf))

images = np.load(W + "images_1816.npz")["images"].astype(float)
xml = open(W + "garnet/lite_corrected.xml").read()
_counts_workspace("data", images.ravel(), idf_xml=xml, instrument="IMAGINE")
apply_detcal("data", calibration_file(), images.shape[1:])
ql = QuasiLaue("data", pixel_shape=images.shape[1:])
ql.find_peaks(method="local", n_sigma=8)
SetUB(Workspace=ql.peaks_ws, UB=UB.ravel().tolist())

pred = {}
for dmin in (3.0, 10.0):
    PredictPeaks(InputWorkspace=ql.peaks_ws, OutputWorkspace="pred", MinDSpacing=dmin, MaxDSpacing="inf",
                 WavelengthMin=BAND[0], WavelengthMax=BAND[1], ReflectionCondition="Primitive")
    pk = [p for p in mtd["pred"] if p.getDetectorID() >= 0]
    pix = np.column_stack(ql.pixels_of_detectors([p.getDetectorID() for p in pk]))
    pred[dmin] = dict(pix=pix, hkl=np.array([p.getHKL() for p in pk]),
                      lam=np.array([p.getWavelength() for p in pk]))

np.savez_compressed(W + "figure/t4l_1816.npz", images=images, positions=ql.positions, UB=UB,
                    cell=o.get_lattice_constants(), coords=coords, heights=h, hkl=hkl, lam=lam,
                    indexed=np.any(hkl != 0, axis=1),
                    pred3_pix=pred[3.0]["pix"], pred3_hkl=pred[3.0]["hkl"], pred3_lam=pred[3.0]["lam"],
                    pred10_pix=pred[10.0]["pix"], pred10_hkl=pred[10.0]["hkl"], pred10_lam=pred[10.0]["lam"])
print("saved", len(pred[3.0]["pix"]), len(pred[10.0]["pix"]))
