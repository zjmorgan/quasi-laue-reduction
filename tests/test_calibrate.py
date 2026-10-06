"""
Geometry calibration on a synthetic two-array detector with a known
misplacement (no Mantid needed).
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quasi_laue_reduction.calibrate import STAGES, SeriesCalibration, rot_about
from quasi_laue_reduction.optimize import CalculateUB, centering_filter

CELL = (11.9386, 11.9386, 11.9386, 90.0, 90.0, 90.0)
BAND = (2.0, 10.0)
R = 0.35
HALF = 0.06
T_DEG = [-167.5, -147.9, -128.3, -108.6, -89.0, 12.5, 32.1, 51.7, 71.4, 91.0]
Y_ROWS = np.arange(-3.5, 4.5) * 0.122

TRUE = {"yaw": np.deg2rad(-1.0), "vy": np.array([0.006, 0.005]), "hxz": np.array([[-0.0015, -0.0038], [0.0040, 0.0022]])}


def nominal_frames():
    c, b, u = [], [], []
    for t in np.deg2rad(T_DEG):
        for y in Y_ROWS:
            c.append([R * np.sin(t), y, R * np.cos(t)])
            b.append([np.cos(t), 0.0, -np.sin(t)])
            u.append([0.0, 1.0, 0.0])
    return np.array(c), np.array(b), np.array(u)


def positions_grid(c, b, u, n=8):
    s = np.linspace(-HALF, HALF, n)
    return c[:, None, None] + s[None, :, None, None] * b[:, None, None] + s[None, None, :, None] * u[:, None, None]


def true_frames(c, b, u):
    arr = (np.arange(80) >= 40).astype(int)
    rot = [np.eye(3), Rotation.from_rotvec(TRUE["yaw"] * np.array([0, 1, 0])).as_matrix()]
    ct = np.array([rot[a] @ ci for a, ci in zip(arr, c)])
    ct += np.column_stack([TRUE["hxz"][arr, 0], TRUE["vy"][arr], TRUE["hxz"][arr, 1]])
    bt = np.array([rot[a] @ bi for a, bi in zip(arr, b)])
    ut = np.array([rot[a] @ ui for a, ui in zip(arr, u)])
    return ct, bt, ut


def simulate_series(n_runs=13, rng=0):
    rng = np.random.default_rng(rng)
    c, b, u = nominal_frames()
    ct, bt, ut = true_frames(c, b, u)
    nt = np.cross(bt, ut)

    o = CalculateUB(*CELL)
    B = o.reciprocal_lattice_B()
    H, _, _ = o._build_hkl_table(B, q_max=1 / 0.9, hkl_filter=centering_filter("I"))
    U0 = Rotation.random(random_state=1).as_matrix()
    axis = np.array([0.0, -1.0, 0.0])

    runs = []
    for k, theta in enumerate(np.linspace(0, 180, n_runs)):
        U = rot_about(axis, theta) @ U0
        G = H @ (U @ B).T
        lam = -2 * G[:, 2] / np.sum(G**2, axis=1)
        ok = (lam >= BAND[0]) & (lam <= BAND[1])
        kf = np.array([0, 0, 1.0]) + lam[ok, None] * G[ok]
        kf /= np.linalg.norm(kf, axis=1)[:, None]

        xyz, bank = [], []
        for d in kf:
            den = nt @ d  # outward normals: a hit has den > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = np.where(den > 1e-9, np.sum(ct * nt, axis=1) / den, np.inf)
                p = np.where(np.isfinite(tt)[:, None], tt[:, None] * d, 0.0)
            s = np.sum((p - ct) * bt, axis=1)
            v = np.sum((p - ct) * ut, axis=1)
            inside = np.isfinite(tt) & (np.abs(s) < HALF) & (np.abs(v) < HALF)
            if not inside.any():
                continue
            j = int(np.flatnonzero(inside)[np.argmin(tt[inside])])
            sj, vj = s[j] + 2e-4 * rng.normal(), v[j] + 2e-4 * rng.normal()
            xyz.append(c[j] + sj * b[j] + vj * u[j])  # where the analysis thinks it is
            bank.append(j)
        xyz, bank = np.array(xyz).reshape(-1, 3), np.array(bank, dtype=int)
        pick = rng.choice(len(xyz), size=min(45, len(xyz)), replace=False)
        runs.append({"run": k, "axis0": float(theta), "xyz": xyz[pick], "bank": bank[pick],
                     "heights": np.ones(len(pick)), "U": U})
    return runs, positions_grid(c, b, u), axis


def test_series_calibration_recovers_array_misplacement():
    runs, positions, axis = simulate_series()
    cal = SeriesCalibration(runs, positions, CELL, "I", band=BAND, tols=(0.5, 0.35), min_panel_peaks=20)
    cal.axis = axis  # orientations given; the bootstrap is exercised on real data
    stages = [s for s in STAGES if s[0][0] in "ABCD"]
    report = cal.fit(stages=stages)

    D = next(r for r in report if r["stage"].startswith("D"))
    p, cfg = cal.params[D["stage"]]
    q = cal.unpack(p, cfg)

    assert np.degrees(q["yaw"]) == pytest.approx(np.degrees(TRUE["yaw"]), abs=0.05)
    assert np.allclose(q["vy"], TRUE["vy"], atol=0.0002)
    assert np.allclose(q["hxz"], TRUE["hxz"], atol=0.0003)
    assert D["test"]["fraction"] > 0.9
    assert report[0]["test"]["fraction"] < D["test"]["fraction"]
