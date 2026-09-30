"""
Mesolite on the CG4D image plate: TIFF -> peaks -> UB -> refine -> integrate.
"""

from quasi_laue_reduction.loading import load_cg4d_image
from quasi_laue_reduction.workflow import QuasiLaue

filename = "/HFIR/CG4D/shared/images/ndip_data_test/meso_may/meso_2_15min_2-0_4-5_078.tif"

### sample info ###
a, b, c = 18.40, 56.67, 6.54
alpha, beta, gamma = 90, 90, 90
centering = "F"
cell = "Orthorhombic"
### ----------- ###

wavelength_band = [2.0, 4.5]

info = load_cg4d_image(filename, ws="data")

ql = QuasiLaue("data", pixel_shape=info["pixel_shape"])
ql.find_peaks(method="global", max_peaks=200, min_pix=3, perc=99.9)
ql.find_UB(a, b, c, alpha, beta, gamma, wavelength_band, centering=centering, n_proc=-1, seed=1)
ql.refine_UB(cell, wavelength_band)
ql.predict_peaks(wavelength_band, d_min=1.0)
ql.integrate_peaks(roi_pixels=10, plot=True)
