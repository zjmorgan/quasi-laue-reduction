"""MANDI garnet (runs 11736-11828): per-run peak directions + reference time-of-flight UB."""
import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import numpy as np
from mantid.simpleapi import Load
ws = Load("/SNS/MANDI/shared/calibration/2025A_blue_garnet/calibration_0.nxs", OutputWorkspace="pk")
ol = ws.sample().getOrientedLattice()
UB = np.array(ol.getUB())
rows = []
for p in ws:
    tt, az = p.getScattering(), p.getAzimuthal()
    kf = np.array([np.sin(tt) * np.cos(az), np.sin(tt) * np.sin(az), np.cos(tt)])
    rows.append((p.getRunNumber(), *kf, *p.getHKL(), p.getWavelength(), p.getIntensity(), *np.array(p.getGoniometerMatrix()).ravel()))
a = np.array(rows)
np.savez("mandi_garnet.npz", run=a[:, 0].astype(int), kf=a[:, 1:4], hkl=a[:, 4:7], lam=a[:, 7], intens=a[:, 8],
         R=a[:, 9:18].reshape(-1, 3, 3), UB=UB, cell=[ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()])
# sanity: does d = kf - ki equal lambda * R UB hkl ?
d = a[:, 1:4] - np.array([0, 0, 1.0])
pred = a[:, 7, None] * np.einsum("nij,jk,nk->ni", a[:, 9:18].reshape(-1, 3, 3), UB, a[:, 4:7])
err = np.rad2deg(np.arccos(np.clip(np.sum(d * pred, 1) / np.linalg.norm(d, axis=1) / np.linalg.norm(pred, axis=1), -1, 1)))
print("peaks", len(a), "check angle (deg) median", np.median(err).round(3), "max", err.max().round(3), "|d|/|pred| median", np.median(np.linalg.norm(d, axis=1) / np.linalg.norm(pred, axis=1)).round(4))
