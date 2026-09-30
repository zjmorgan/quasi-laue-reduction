"""
Mesolite (Fdd2), IMAGINE geometry on CG4D (IPTS-37331, run 1438).

A small F-centred cell: used as a geometry check, since a consistent
orientation is found but residuals are 0.5-3 degrees (see README).
"""

from quasi_laue_reduction.loading import load_lite
from quasi_laue_reduction.workflow import QuasiLaue

IPTS = 37331
run = 1438

filename = "/HFIR/CG4D/IPTS-{}/shared/autoreduce/CG4D_{}.lite.nxs.h5".format(IPTS, run)

### sample info ###
a, b, c = 18.40, 56.65, 6.54
alpha, beta, gamma = 90, 90, 90
centering = "F"
cell = "Orthorhombic"
### ----------- ###

wavelength_band = [2.8, 4.6]

info = load_lite(filename, ws="data")

ql = QuasiLaue("data", pixel_shape=info["pixel_shape"])
ql.find_peaks(method="local", n_sigma=8)
ql.find_UB(a, b, c, alpha, beta, gamma, wavelength_band, centering=centering, n_proc=-1, seed=1, angle_tol_deg=1.0)
ql.predict_peaks(wavelength_band, d_min=1.5)
ql.plot_detector()
