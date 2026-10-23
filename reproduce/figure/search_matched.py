"""1816 from scratch (RT cell, DetCal geometry) with different peak sets."""
import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, json, time, numpy as np
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB
from quasi_laue_reduction.peaks import find_peaks_local, find_peaks_matched, interpolate_positions, scattering_directions

d = np.load("t4l_1816.npz"); images, P = d["images"], d["positions"]
ref_kf = np.load("../peaks_1816_detcal.npz")["kf_ki_dir"]
CELL = (61.1, 61.1, 96.84, 90, 90, 120); BAND = (2.0, 10.0)

def kf_of(coords):
    return scattering_directions(interpolate_positions(P, coords))

c8, h8, _ = find_peaks_local(images, n_sigma=8)
print("check: recomputed local-8 kf matches saved:", np.allclose(kf_of(c8), ref_kf, atol=1e-6), flush=True)
cm, hm, _ = find_peaks_matched(images, z_min=5)
xyz = interpolate_positions(P, cm); g = np.rad2deg(np.arctan2(xyz[:, 0], xyz[:, 2]))
sets = {"local8": (kf_of(c8), h8), "matched5_all": (kf_of(cm), hm),
        "matched5_absgamma<60": (kf_of(cm)[np.abs(g) < 60], hm[np.abs(g) < 60])}
for name, (kf, h) in sets.items():
    for seed in (1, 2):
        t = time.perf_counter()
        o = CalculateUB(*CELL)
        o.find_orientation(kf, BAND, h, n_proc=16, n_restarts=3, seed=seed)
        out = {"set": name, "seed": seed, "n_peaks": len(kf), "time_s": round(time.perf_counter() - t)}
        for tol in (0.2, 0.35):
            s = o.index_significance(kf, BAND, angle_tol=np.deg2rad(tol))
            out[f"own_{tol}"] = f"{s['indexed']} vs {s['expected_chance']:.0f} (log10p {s['log10_p_value']:.1f})"
        s = o.index_significance(ref_kf, BAND, angle_tol=np.deg2rad(0.2))
        out["ref282_0.2"] = f"{s['indexed']} vs {s['expected_chance']:.0f} (log10p {s['log10_p_value']:.1f})"
        print(json.dumps(out), flush=True)
print("DONE", flush=True)
