"""Mesolite on the IMAGINE image plate (meso_2_15min_2-0_4-5_078.tif): blind indexing with timing."""
import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, json, time
import numpy as np
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.loading import load_cg4d_image
from quasi_laue_reduction.workflow import QuasiLaue

F = "/HFIR/CG4D/shared/images/ndip_data_test/meso_may/meso_2_15min_2-0_4-5_078.tif"
CELL, BAND = (18.39, 56.55, 6.54, 90, 90, 90), (2.0, 4.5)
info = load_cg4d_image(F, ws="data")
ql = QuasiLaue("data", pixel_shape=info["pixel_shape"])
images = ql.extract_images()
np.savez_compressed("images_078.npz", images=images.astype(np.float32))
for method, kw in (("local", {"n_sigma": 8}), ("matched", {"z_min": 5})):
    t0 = time.perf_counter()
    ql.find_peaks(method=method, **kw)
    tf = time.perf_counter() - t0
    t0 = time.perf_counter()
    num = ql.find_UB(*CELL, BAND, centering="F", n_proc=16, seed=1, n_restarts=3)
    ts = time.perf_counter() - t0
    sig = ql.significance
    o = ql.opt
    ang = np.rad2deg(o.last_angle_error)
    print(json.dumps({"finder": method, "peaks": int(len(ql.found_coords)), "search_peaks": sig["n_informative"],
                      "indexed_all": int(num), "search_indexed": sig["indexed"], "chance": round(sig["expected_chance"], 1),
                      "log10p": round(sig["log10_p_value"], 1), "median_residual_deg": round(float(np.median(ang[ang < 0.35 * 1.0001 / 1])), 3) if (ang < 0.35).any() else None,
                      "time_find_s": round(tf, 1), "time_search_s": round(ts, 1),
                      "UB": np.array(o.orientation_U(*o.x) @ o.reciprocal_lattice_B()).round(6).tolist()}), flush=True)
print("DONE", flush=True)
