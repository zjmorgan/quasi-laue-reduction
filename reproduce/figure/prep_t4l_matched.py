import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import sys, json, numpy as np, scipy.stats
sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.optimize import CalculateUB
from quasi_laue_reduction.peaks import find_peaks_matched, interpolate_positions, scattering_directions
d = np.load("t4l_1816.npz"); images, P = d["images"], d["positions"]; BAND = (2.0, 10.0)

# garnet: fraction of peaks a 0.7 chance cut keeps (run 2022, same instrument geometry)
g = np.load("../garnet/images_2022.npz")["images"].astype(float)
cg, _, _ = find_peaks_matched(g, z_min=5)
og = CalculateUB(11.9386, 11.9386, 11.9386, 90, 90, 90)
og.prepare_indexer(scattering_directions(interpolate_positions(P, cg)), BAND, ambiguity_power=0, force=True, lookup=False)
pg = -np.expm1(-og._candidate_counts * np.deg2rad(0.35) ** 2 / 4)
print("garnet 2022 (primitive-cell counts, conservative): peaks", len(cg), "kept by 0.7 cut", int((pg <= 0.7).sum()), flush=True)

c, h, _ = find_peaks_matched(images, z_min=5)
kf = scattering_directions(interpolate_positions(P, c))
o = CalculateUB(*d["cell"]); o.x = o.orientation_parameters_from_U(d["UB"] @ np.linalg.inv(o.reciprocal_lattice_B()))
o.prepare_indexer(kf, BAND, ambiguity_power=0, force=True, lookup=False)
inf = (-np.expm1(-o._candidate_counts * np.deg2rad(0.35) ** 2 / 4)) <= 0.7
for tol in (0.5, 0.35, 0.35, 0.2, 0.2):
    o._score_angle_tol = np.deg2rad(tol); o.refine(kf[inf], BAND, "Hexagonal", error=0.05)
cell = o.get_lattice_constants(); UB = o.orientation_U(*o.x) @ o.reciprocal_lattice_B()
tols = np.round(np.arange(0.05, 1.001, 0.05), 2)
rows = [o.index_significance(kf[inf], BAND, angle_tol=np.deg2rad(t)) for t in tols]
obs = np.array([r["indexed"] for r in rows]); ch = np.array([r["expected_chance"] for r in rows])
t95 = None
for t in np.arange(0.3, 1.0, 0.01):
    if o.index_significance(kf[inf], BAND, angle_tol=np.deg2rad(t))["indexed"] >= 0.95 * inf.sum(): t95 = round(t, 2); break
_, _, hkl, lam = o.index(kf, BAND, angle_tol=np.deg2rad(t95))
print(json.dumps({"c_over_a": round(cell[2] / cell[0], 4), "n_peaks": len(kf), "n_search": int(inf.sum()), "tol95": t95,
                  "indexed_all_at_tol95": int(np.any(np.asarray(hkl) != 0, axis=1).sum()),
                  **{f"search_{t}": f"{o_} vs {c_:.0f} log10p {r['log10_p_value']:.1f}" for t, o_, c_, r in zip(tols, obs, ch, rows) if t in (0.2, 0.35, 0.5)}}), flush=True)
np.savez_compressed("t4l_1816_matched.npz", coords=c, heights=h, kf=kf, informative=inf, UB=UB, cell=cell,
                    hkl=np.asarray(hkl), lam=lam, tol95=t95, tols=tols, obs=obs, chance=ch)
