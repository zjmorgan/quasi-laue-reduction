"""
T4 lysozyme on IMAGINE-X (IPTS-37331, run 1816), full 2-10 A band.

The settings of the paper: garnet-calibrated geometry, matched-filter peaks,
room-temperature cell. Indexing a protein cell at this band needs the
calibration.
"""

from quasi_laue_reduction.loading import calibration_file, load_lite
from quasi_laue_reduction.workflow import QuasiLaue

IPTS = 37331
run = 1816

filename = "/HFIR/CG4D/IPTS-{}/shared/autoreduce/CG4D_{}.lite.nxs.h5".format(IPTS, run)

### sample info ###
# room-temperature cell refined from this run (5VNQ at 80 K: c/a = 1.559)
a, b, c = 61.1, 61.1, 96.84
alpha, beta, gamma = 90, 90, 120
centering = "P"
cell = "Hexagonal"
### ----------- ###

wavelength_band = [2.0, 10.0]

info = load_lite(filename, ws="data", detcal=calibration_file())

ql = QuasiLaue("data", pixel_shape=info["pixel_shape"])
ql.find_peaks(method="matched", z_min=5)
ql.find_UB(a, b, c, alpha, beta, gamma, wavelength_band, centering=centering, n_proc=-1, seed=1)
ql.predict_peaks(wavelength_band, d_min=2.5)
ql.plot_detector()
