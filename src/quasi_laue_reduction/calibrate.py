"""
Detector-geometry calibration from a goniometer rotation series of a
reference crystal (e.g. garnet).

Observable: for every indexed peak, the direction of kf_hat - ki_hat must
match the direction of U_k B hkl. Directions are wavelength- and
scale-free, so the cell only enters through its shape.

Model
-----
Crystal:  U_k = Rot(axis, theta_k + dtheta_k) U0, with theta_k the logged
          goniometer angle (axis0) of run k; axis, U0 and the per-run
          offsets dtheta_k are fitted.
Detector: the two 40-camera arrays (banks 11-58 and 61-108) are moved as
          rigid bodies, the sample stays at the origin:
            yaw        rotation of array 2 about the vertical axis through
                       the sample, relative to array 1 (a common yaw equals
                       a goniometer offset and is not measurable)
            vertical   vertical shift of each array
            horizontal (x, z) translation of each array
          and optionally in-plane shifts of well-covered panels, constrained
          to zero mean per array.

Stages add these terms one at a time; each is judged by the fraction of
peaks of held-out runs (every other run) indexed within the final
tolerance after refitting only their angle offsets.
"""

import glob
import json
import os
import re
import time

import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation

from .detcal import panel_frames, write_detcal
from .loading import correct_grouped_panel_origin, read_lite
from .optimize import CalculateUB, centering_filter, primitive_cell, to_conventional
from .peaks import find_peaks_local, interpolate_positions
from .simulate import rotational_symmetry_ops

Z = np.array([0.0, 0.0, 1.0])
Y = np.array([0.0, 1.0, 0.0])
ARRAY = (np.arange(80) >= 40).astype(int)  # banks 11-58 / 61-108

STAGES = [
    ("A goniometer", {}),
    ("B +relative array yaw", {"yaw": True}),
    ("C +array vertical shifts", {"yaw": True, "vertical": True}),
    ("D +array horizontal translations", {"yaw": True, "vertical": True, "horizontal": True}),
    ("E +panel in-plane shifts", {"yaw": True, "vertical": True, "horizontal": True, "inplane": True}),
]

FLAGS = ("yaw", "vertical", "horizontal", "inplane", "ratios")

# Fixed-geometry validation: goniometer only, then + cell length ratios.
VALIDATION_STAGES = [("A goniometer", {}), ("R +cell ratios b/a, c/a", {"ratios": True})]


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------


def nominal_positions(lite_file, out_dir):
    """
    Pixel positions (80, 128, 128, 3) of the centre-corrected lite IDF.

    Built once with Mantid; the workspace is deleted afterwards.
    """
    fn = os.path.join(out_dir, "positions.npz")
    if os.path.exists(fn):
        return np.load(fn)["positions"]

    from mantid.simpleapi import DeleteWorkspace, mtd

    from .loading import _counts_workspace
    from .workflow import QuasiLaue

    info = read_lite(lite_file)
    xml, _ = correct_grouped_panel_origin(info["idf"], info["group"])
    _counts_workspace("_calib_geom", info["counts"], idf_xml=xml, instrument=info["instrument"])
    positions = QuasiLaue("_calib_geom", pixel_shape=info["pixel_shape"]).positions

    for ws in ("_calib_geom", "_calib_geom_detectors"):
        if mtd.doesExist(ws):
            DeleteWorkspace(ws)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(fn, positions=positions)
    with open(os.path.join(out_dir, "lite_corrected.xml"), "w") as f:
        f.write(xml)

    return positions


def extract_series(lite_files, out_dir, n_sigma=8.0):
    """
    Peaks (and images) for each run of a rotation series.

    Skips runs already extracted, so an interrupted extraction resumes.
    The goniometer angle is read from the run title (``axis0 = ...``).
    """
    os.makedirs(out_dir, exist_ok=True)
    positions = nominal_positions(lite_files[0], out_dir)

    for lite in lite_files:
        run = int(re.search(r"_(\d+)\.lite", lite).group(1))
        out = os.path.join(out_dir, f"peaks_{run}.npz")
        if os.path.exists(out):
            continue
        t = time.perf_counter()
        info = read_lite(lite)
        images = info["counts"].reshape(-1, *positions.shape[1:3])
        coords, heights, snr = find_peaks_local(images, n_sigma=n_sigma)
        m = re.search(r"axis0\s*=\s*([-\d.]+)", info["title"])
        np.savez(
            out,
            coords=coords,
            heights=heights,
            snr=snr,
            xyz=interpolate_positions(positions, coords),
            axis0=float(m.group(1)) if m else np.nan,
            title=info["title"],
        )
        np.savez_compressed(os.path.join(out_dir, f"images_{run}.npz"), images=images.astype(np.float32))
        print(f"{run} {info['title']!r} peaks={len(coords)} {time.perf_counter() - t:.0f}s", flush=True)


def load_series(out_dir):
    runs = []
    for fn in sorted(glob.glob(os.path.join(out_dir, "peaks_*.npz"))):
        d = np.load(fn)
        runs.append(
            {
                "run": int(re.search(r"peaks_(\d+)", fn).group(1)),
                "axis0": float(d["axis0"]),
                "xyz": d["xyz"],
                "bank": d["coords"][:, 0].astype(int),
                "heights": d["heights"],
            }
        )
    return runs


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------


def kf_ki(xyz):
    kf = xyz / np.linalg.norm(xyz, axis=1)[:, None]
    return kf - Z


def rot_about(axis, deg):
    return Rotation.from_rotvec(np.deg2rad(deg) * np.asarray(axis)).as_matrix()


class SeriesCalibration:
    """
    Joint calibration of goniometer and detector arrays from a series.

    Parameters
    ----------
    runs : list of dict
        From :func:`load_series`.
    positions : ndarray
        Nominal pixel positions (80, nx, ny, 3).
    cell : tuple
        Conventional cell (only its shape matters).
    centering : str
        Lattice centering.
    band : tuple
        Wavelength band.
    tols : tuple
        (search, final) assignment tolerances in degrees.
    n_proc : int
        Workers for the fresh orientation searches.
    """

    def __init__(self, runs, positions, cell, centering, band=(2.0, 10.0), tols=(0.5, 0.35), n_proc=16, min_panel_peaks=20):
        self.runs = runs
        self.positions = positions
        self.cell = tuple(cell)
        self.centering = centering
        self.band = tuple(band)
        self.tols = tuple(tols)
        self.n_proc = n_proc

        self.B = CalculateUB(*self.cell).reciprocal_lattice_B()
        self.ops = rotational_symmetry_ops(self.cell)

        frames = panel_frames(positions)
        self.panel_base, self.panel_up = frames["base"], frames["up"]

        counts = np.bincount(np.concatenate([np.asarray(r["bank"], dtype=int) for r in runs]), minlength=len(positions))
        self.fine_banks = np.flatnonzero(counts >= min_panel_peaks)

        n = len(runs)
        self.train = [k for k in range(n) if k % 2 == 0]
        self.test = [k for k in range(n) if k % 2 == 1]

    # ------------------------------------------------------------- orientation

    def _opt(self, U=None, cell=None):
        o = CalculateUB(*(self.cell if cell is None else cell))
        o._hkl_filter = centering_filter(self.centering)
        if U is not None:
            o.x = o.orientation_parameters_from_U(U)
        return o

    def _polish(self, U, r):
        o = self._opt(U)
        kf = kf_ki(r["xyz"])
        w = CalculateUB.make_peak_weights(r["heights"])
        for t in (1.0,) + self.tols[1:]:
            o.reassign_refine(kf, self.band, w, 0.5, np.deg2rad(t), n_cycles=3)
        return o.orientation_U(*o.x), o.index_significance(kf, self.band, angle_tol=np.deg2rad(self.tols[-1]))

    def _fresh(self, r, seed=1):
        p = CalculateUB(*primitive_cell(*self.cell, self.centering))
        p.find_orientation(kf_ki(r["xyz"]), self.band, heights=r["heights"], n_proc=self.n_proc, seed=seed)
        c, _ = to_conventional(p, self.centering, self.cell)
        return self._polish(c.orientation_U(*c.x), r)

    def orientations(self, cache=None):
        """
        Orientation of every run: fresh search on the first two runs,
        goniometer axis from them, then prediction + reassignment (fresh
        search as a fallback), refining the axis as larger rotations
        become available.
        """
        runs = self.runs
        r0, r1 = runs[0], runs[1]

        if cache and os.path.exists(cache) and len(np.load(cache)) == len(runs):
            for r, U in zip(runs, np.load(cache)):
                r["U"] = U
            ref = min(runs[1:], key=lambda r: abs(abs(r["axis0"] - r0["axis0"]) - 90))
            rv = Rotation.from_matrix(ref["U"] @ r0["U"].T).as_rotvec()
            self.axis = rv / np.linalg.norm(rv) * np.sign(ref["axis0"] - r0["axis0"])
            return

        U0, _ = self._fresh(r0)
        U1, _ = self._fresh(r1)
        dw = r1["axis0"] - r0["axis0"]
        R = min(
            ((U1 @ M) @ U0.T for M in self.ops),
            key=lambda R: abs(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())) - abs(dw)),
        )
        rv = Rotation.from_matrix(R).as_rotvec()
        axis = rv / np.linalg.norm(rv) * np.sign(dw)

        for r in runs:
            U_pred = rot_about(axis, r["axis0"] - r0["axis0"]) @ U0
            U, sig = self._polish(U_pred, r)
            r["fresh"] = False
            if sig["log10_p_value"] > -5:
                U, sig = self._fresh(r)
                U = min((U @ M for M in self.ops), key=lambda V: np.linalg.norm(V - U_pred))
                r["fresh"] = True
            r["U"], r["sig"] = U, sig
            dwk = r["axis0"] - r0["axis0"]
            if 10 < abs(dwk) < 170 and sig["log10_p_value"] < -5:
                rv = Rotation.from_matrix(U @ U0.T).as_rotvec()
                if np.linalg.norm(rv) > 0 and np.dot(rv, axis) * np.sign(dwk) > 0:
                    axis = rv / np.linalg.norm(rv) * np.sign(dwk)
            print(
                f"run {r['run']} axis0 {r['axis0']:g}: indexed {sig['indexed']}/{len(r['xyz'])}, "
                f"log10 p {sig['log10_p_value']:.1f}" + (" (fresh search)" if r["fresh"] else ""),
                flush=True,
            )

        self.axis = axis
        if cache:
            np.save(cache, np.array([r["U"] for r in runs]))

    # ------------------------------------------------------------- parameters

    def layout(self, cfg):
        n0 = 6 + len(self.runs)
        sizes = {"yaw": 1, "vertical": 2, "horizontal": 4, "inplane": 2 * len(self.fine_banks), "ratios": 2}
        out, i = {}, n0
        for name in FLAGS:
            if cfg.get(name):
                out[name] = slice(i, i + sizes[name])
                i += sizes[name]
        return out, i

    def unpack(self, p, cfg):
        lay, _ = self.layout(cfg)
        nf = len(self.fine_banks)

        def get(name, shape):
            return p[lay[name]].reshape(shape) if name in lay else np.zeros(shape)

        return {
            "axis": Rotation.from_rotvec(p[0:3]).apply(self.axis),
            "U0": Rotation.from_rotvec(p[3:6]).as_matrix() @ self.runs[0]["U"],
            "dw": p[6 : 6 + len(self.runs)],
            "yaw": get("yaw", (1,))[0],
            "vy": get("vertical", (2,)),
            "hxz": get("horizontal", (2, 2)),
            "pin": get("inplane", (nf, 2)),
            "cell": self._cell_from(get("ratios", (2,))),
        }

    def _cell_from(self, dratio):
        """Cell with b/a and c/a changed by ``dratio`` (a fixed)."""
        a, b, c, al, be, ga = self.cell
        return (a, a * (b / a + dratio[0]), a * (c / a + dratio[1]), al, be, ga)

    def extend(self, p, cfg_old, cfg_new):
        """Parameter vector for cfg_new carrying over values from cfg_old."""
        lay_old, _ = self.layout(cfg_old)
        lay_new, n = self.layout(cfg_new)
        q = np.zeros(n)
        q[: 6 + len(self.runs)] = p[: 6 + len(self.runs)]
        for name, sl in lay_new.items():
            if name in lay_old:
                q[sl] = p[lay_old[name]]
        return q

    # ------------------------------------------------------------- model

    def corrected_xyz(self, xyz, bank, q, cfg):
        a = ARRAY[bank]
        if cfg.get("yaw"):
            yaw = np.where(a == 1, q["yaw"], 0.0)
            xyz = np.einsum("nij,nj->ni", Rotation.from_rotvec(yaw[:, None] * Y).as_matrix(), xyz)
        if cfg.get("vertical"):
            xyz = xyz + q["vy"][a][:, None] * Y
        if cfg.get("horizontal"):
            xyz = xyz + np.column_stack([q["hxz"][a, 0], np.zeros(len(a)), q["hxz"][a, 1]])
        if cfg.get("inplane") and len(self.fine_banks):
            k = np.clip(np.searchsorted(self.fine_banks, bank), 0, len(self.fine_banks) - 1)
            fine = (self.fine_banks[k] == bank)[:, None]
            shift = q["pin"][k, 0][:, None] * self.panel_base[bank] + q["pin"][k, 1][:, None] * self.panel_up[bank]
            xyz = xyz + np.where(fine, shift, 0.0)
        return xyz

    def model_U(self, q, k):
        return rot_about(q["axis"], self.runs[k]["axis0"] - self.runs[0]["axis0"] + q["dw"][k]) @ q["U0"]

    def assign(self, k, q, cfg, tol):
        r = self.runs[k]
        o = self._opt(self.model_U(q, k), q["cell"])
        kf = kf_ki(self.corrected_xyz(r["xyz"], r["bank"], q, cfg))
        _, _, hkl, _ = o.index(kf, self.band, angle_tol=np.deg2rad(tol))
        return hkl

    def stack(self, sel, hkls):
        rows = {"xyz": [], "bank": [], "hkl": [], "axis0": [], "k": []}
        for k in sel:
            r, h = self.runs[k], hkls[k]
            m = np.any(h != 0, axis=1)
            rows["xyz"].append(r["xyz"][m])
            rows["bank"].append(r["bank"][m])
            rows["hkl"].append(h[m])
            rows["axis0"].append(np.full(m.sum(), r["axis0"]))
            rows["k"].append(np.full(m.sum(), k))
        return {key: np.concatenate(v) for key, v in rows.items()}

    def residuals(self, p, data, cfg, priors=True):
        q = self.unpack(p, cfg)
        d = kf_ki(self.corrected_xyz(data["xyz"], data["bank"], q, cfg))
        d /= np.linalg.norm(d, axis=1)[:, None]
        ang = data["axis0"] - self.runs[0]["axis0"] + q["dw"][data["k"]]
        Rk = Rotation.from_rotvec(np.deg2rad(ang)[:, None] * q["axis"]).as_matrix()
        B = CalculateUB(*q["cell"]).reciprocal_lattice_B() if cfg.get("ratios") else self.B
        g = np.einsum("nij,jk,nk->ni", Rk, q["U0"] @ B, data["hkl"])
        g /= np.linalg.norm(g, axis=1)[:, None]
        res = [(d - g).ravel()]
        if priors:
            obs = np.deg2rad(0.1)
            res.append(q["dw"] / 1.0 * obs)
            if cfg.get("yaw"):
                res.append(np.atleast_1d(q["yaw"]) / np.deg2rad(2.0) * obs)
            if cfg.get("vertical"):
                res.append(q["vy"] / 0.01 * obs)
            if cfg.get("horizontal"):
                res.append(q["hxz"].ravel() / 0.01 * obs)
            if cfg.get("inplane") and len(self.fine_banks):
                res.append(q["pin"].ravel() / 0.003 * obs)
                arr = ARRAY[self.fine_banks]
                for a in (0, 1):
                    if np.any(arr == a):
                        res.append(q["pin"][arr == a].mean(axis=0) / 0.0001 * obs)
        return np.concatenate(res)

    def evaluate(self, p, cfg, sel, refit_angles):
        """Indexed count and residuals for runs `sel` at the final tolerance."""
        p = p.copy()

        def assign_all(tol):
            q = self.unpack(p, cfg)
            return {k: self.assign(k, q, cfg, tol) for k in sel}

        if refit_angles:
            for tol in self.tols:
                hk = assign_all(tol)
                for k in sel:
                    dk = self.stack([k], hk)
                    if len(dk["hkl"]) < 3:
                        continue

                    def f(x, k=k, dk=dk):
                        pp = p.copy()
                        pp[6 + k] = x[0]
                        return self.residuals(pp, dk, cfg, priors=False)

                    p[6 + k] = scipy.optimize.least_squares(
                        f, [p[6 + k]], loss="soft_l1", f_scale=np.deg2rad(0.2)
                    ).x[0]

        data = self.stack(sel, assign_all(self.tols[1]))
        rr = self.residuals(p, data, cfg, priors=False).reshape(-1, 3)
        ang = np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(rr, axis=1) / 2, 0, 1)))
        total = sum(len(self.runs[k]["xyz"]) for k in sel)
        return {
            "indexed": int(len(ang)),
            "of": int(total),
            "fraction": float(len(ang) / total),
            "median_deg": float(np.median(ang)) if len(ang) else np.nan,
        }

    def fit(self, stages=STAGES, n_iter=3, out_dir=None):
        """
        Fit stage by stage; returns a report list and the parameter vectors.
        """
        cfg = {name: False for name in FLAGS}
        p = np.zeros(6 + len(self.runs))
        report, params = [], {}

        for stage, flags in stages:
            new = {name: False for name in FLAGS} | flags
            p = self.extend(p, cfg, new)
            cfg = new
            for it in range(n_iter):
                q = self.unpack(p, cfg)
                tol = self.tols[0] if it == 0 else self.tols[1]
                hkls = {k: self.assign(k, q, cfg, tol) for k in self.train}
                data = self.stack(self.train, hkls)
                sol = scipy.optimize.least_squares(
                    self.residuals, p, args=(data, cfg), loss="soft_l1", f_scale=np.deg2rad(0.2), x_scale="jac"
                )
                p = sol.x

            rec = {
                "stage": stage,
                "train": self.evaluate(p, cfg, self.train, refit_angles=False),
                "test": self.evaluate(p, cfg, self.test, refit_angles=True),
                "goniometer_axis": self.unpack(p, cfg)["axis"].round(5).tolist(),
                "dtheta_rms_deg": float(np.sqrt(np.mean(self.unpack(p, cfg)["dw"] ** 2))),
            }
            rec.update(self._uncertainties(sol, p, cfg))
            report.append(rec)
            params[stage] = (p.copy(), dict(cfg))
            print(json.dumps(rec), flush=True)
            if out_dir:
                np.save(os.path.join(out_dir, f"params_{stage[0]}.npy"), p)

        self.report, self.params = report, params
        return report

    def _uncertainties(self, sol, p, cfg):
        J = sol.jac
        chi2 = 2 * sol.cost / max(1, J.shape[0] - J.shape[1])
        cov = np.linalg.pinv(J.T @ J, rcond=1e-8, hermitian=True) * chi2
        sd = np.sqrt(np.maximum(np.diag(cov), 0))
        lay, _ = self.layout(cfg)
        out = {}

        def val(name, scale):
            return [f"{v * scale:.4f} +- {e * scale:.4f}" for v, e in zip(p[lay[name]], sd[lay[name]])]

        if cfg.get("yaw"):
            out["relative_yaw_deg"] = val("yaw", np.degrees(1.0))
        if cfg.get("vertical"):
            out["array_vertical_mm"] = val("vertical", 1e3)
        if cfg.get("horizontal"):
            out["array_horizontal_xz_mm"] = val("horizontal", 1e3)
        if cfg.get("inplane"):
            out["panel_inplane_mm_rms"] = float(np.sqrt(np.mean(p[lay["inplane"]] ** 2)) * 1e3)
        if cfg.get("ratios"):
            a, b, c = self.cell[:3]
            sl = lay["ratios"]
            out["b_over_a"] = f"{b / a + p[sl][0]:.5f} +- {sd[sl][0]:.5f}"
            out["c_over_a"] = f"{c / a + p[sl][1]:.5f} +- {sd[sl][1]:.5f}"
        return out

    # ------------------------------------------------------------- output

    def best_stage(self):
        return max(self.report, key=lambda r: r["test"]["fraction"])

    def calibrated_positions(self, stage):
        p, cfg = self.params[stage]
        q = self.unpack(p, cfg)
        n_banks, nx, ny, _ = self.positions.shape
        flat = self.positions.reshape(-1, 3)
        bank = np.repeat(np.arange(n_banks), nx * ny)
        return self.corrected_xyz(flat, bank, q, cfg).reshape(self.positions.shape)

    def write_detcal(self, prefix, idf_xml, stage=None, comment=""):
        rec = self.best_stage() if stage is None else next(r for r in self.report if r["stage"] == stage)
        pos = self.calibrated_positions(rec["stage"])
        banks = [int(b[4:]) for b in re.findall(r'<component type="(bank\d+)">', idf_xml)]
        note = (
            f"{comment} stage {rec['stage']}, held-out indexed {rec['test']['indexed']}/{rec['test']['of']} "
            f"at {self.tols[1]} deg"
        ).strip()
        write_detcal(prefix + "_lite.DetCal", pos, banks, comment=note)
        write_detcal(prefix + "_full.DetCal", pos, banks, pixels=(4 * pos.shape[1], 4 * pos.shape[2]), comment=note)
        return rec
