"""
T4 lysozyme, IMAGINE geometry on CG4D (IPTS-37331, run 1816).

Status: does not index yet. On synthetic data with this cell the search
recovers the orientation reliably, but on this run no orientation separates
from random; see README ("Test cases").
"""

from quasi_laue_reduction.loading import load_lite
from quasi_laue_reduction.workflow import QuasiLaue

IPTS = 37331
run = 1816

filename = "/HFIR/CG4D/IPTS-{}/shared/autoreduce/CG4D_{}.lite.nxs.h5".format(IPTS, run)

### sample info ###
# 5VNQ (80 K): a = b = 61.5, c = 95.9; room temperature ~ 61.2 / 96.8
a, b, c = 61.1, 61.1, 97.0
alpha, beta, gamma = 90, 90, 120
centering = "P"
cell = "Hexagonal"
### ----------- ###

wavelength_band = [2.8, 4.6]

info = load_lite(filename, ws="data")

ql = QuasiLaue("data", pixel_shape=info["pixel_shape"])
ql.find_peaks(method="local", n_sigma=8)
ql.find_UB(a, b, c, alpha, beta, gamma, wavelength_band, centering=centering, n_proc=-1, seed=1)
ql.predict_peaks(wavelength_band, d_min=2.5)
ql.plot_detector()
