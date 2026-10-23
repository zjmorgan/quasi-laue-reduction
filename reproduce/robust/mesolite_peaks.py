import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, numpy as np
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.loading import load_cg4d_image
from quasi_laue_reduction.workflow import QuasiLaue
info = load_cg4d_image("/HFIR/CG4D/shared/images/ndip_data_test/meso_may/meso_2_15min_2-0_4-5_078.tif", ws="data")
ql = QuasiLaue("data", pixel_shape=info["pixel_shape"])
ql.find_peaks(method="local", n_sigma=8)
np.savez("mesolite_peaks.npz", kf=ql.kf_ki_dir, heights=ql.heights, coords=ql.coords)
print("mesolite peaks", len(ql.heights))
