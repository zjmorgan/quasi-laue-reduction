import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import json
import numpy as np
from quasi_laue_reduction.optimize import CalculateUB
d = np.load(WORK_DIR + "/peaks_1816_detcal.npz"); kf, h = d["kf_ki_dir"], d["heights"]
band = (2.0, 10.0)
U80 = np.load(WORK_DIR + "/U_1816_detcal_seeds.npy")[0]
from quasi_laue_reduction.simulate import misorientation_deg, rotational_symmetry_ops
for a, c in ((61.3, 96.4), (61.2, 96.8), (61.1, 97.0)):
    cell = (a, a, c, 90.0, 90.0, 120.0)
    o = CalculateUB(*cell)
    _, _, _, num = o.find_orientation(kf, band, heights=h, n_proc=16, seed=1, n_restarts=6)
    s = {t: o.index_significance(kf, band, angle_tol=np.deg2rad(t)) for t in (0.2, 0.35, 0.5)}
    mis = misorientation_deg(U80, o.orientation_U(*o.x), rotational_symmetry_ops(cell))
    print(json.dumps({"a": a, "c": c, "c_over_a": round(c / a, 4), "restarts": len(o.restarts),
                      "misorientation_to_80K_solution_deg": round(mis, 2),
                      **{f"at_{t}": f"{s[t]['indexed']}/{len(kf)} vs {s[t]['expected_chance']:.0f} log10p {s[t]['log10_p_value']:.1f}" for t in s}}), flush=True)
    np.save(WORK_DIR + f"/U_1816_{a}_{c}.npy", o.orientation_U(*o.x))
print("CELLS_DONE")
