import re

import h5py
import pytest
import numpy as np

from quasi_laue_reduction.loading import pixel_shape_from_idf, read_lite
from quasi_laue_reduction.peaks import (
    find_peaks_global,
    find_peaks_local,
    find_peaks_matched,
    interpolate_positions,
    scattering_directions,
)


def spot_images(rng, n_banks=3, shape=(128, 128), spots=6):
    """Poisson background with Gaussian spots; per-bank backgrounds differ."""
    ii, jj = np.mgrid[0 : shape[0], 0 : shape[1]]
    images, truth = [], []
    for b in range(n_banks):
        bg = 50.0 * (b + 1) ** 2
        lam = np.full(shape, bg)
        for _ in range(spots):
            ci, cj = rng.uniform(15, shape[0] - 15), rng.uniform(15, shape[1] - 15)
            lam += 40 * np.sqrt(bg) * np.exp(-0.5 * ((ii - ci) ** 2 + (jj - cj) ** 2) / 1.2**2)
            truth.append((b, ci, cj))
        images.append(rng.poisson(lam).astype(float))
    return np.array(images), np.array(truth)


def match(found, truth, tol):
    d = []
    for b, i, j in truth:
        m = found[:, 0] == b
        d.append(np.min(np.hypot(found[m, 1] - i, found[m, 2] - j)) if m.any() else np.inf)
    return np.array(d) < tol, np.array(d)


def test_local_finder_finds_spots_on_every_bank():
    images, truth = spot_images(np.random.default_rng(0))
    coords, heights, snr = find_peaks_local(images, n_sigma=8)
    found, dist = match(coords, truth, 1.0)
    assert found.all()
    assert np.median(dist) < 0.3  # sub-pixel centroids
    assert len(coords) <= len(truth) + 2


def test_global_finder_misses_low_background_banks():
    images, truth = spot_images(np.random.default_rng(0))
    coords, _ = find_peaks_global(images, perc=99.7)
    found, _ = match(coords, truth, 1.5)
    # one threshold for all banks: the quiet first bank is under-detected
    assert found[truth[:, 0] == 2].all()
    assert not found[truth[:, 0] == 0].all()


def faint_spots_with_stripe(rng, shape=(128, 128), spots=12, bg=50.0):
    """Faint, spread spots (peak ~4 sigma per pixel, ~11 sigma summed) and a bright stripe."""
    ii, jj = np.mgrid[0 : shape[0], 0 : shape[1]]
    lam = np.full(shape, bg)
    truth = []
    grid = [(ci, cj) for ci in (20, 40, 60) for cj in (20, 45, 70, 95)][:spots]
    for ci, cj in grid:
        ci, cj = ci + rng.uniform(-2, 2), cj + rng.uniform(-2, 2)
        lam += 4 * np.sqrt(bg) * np.exp(-0.5 * ((ii - ci) ** 2 + (jj - cj) ** 2) / 1.2**2)
        truth.append((0, ci, cj))
    lam[95:97, :] += 3 * np.sqrt(bg)  # line artefact along j
    return rng.poisson(lam).astype(float)[None], np.array(truth)


def test_matched_finder_recovers_faint_spots_and_ignores_stripes():
    images, truth = faint_spots_with_stripe(np.random.default_rng(1))

    local, _, _ = find_peaks_local(images, n_sigma=8)
    matched, _, z = find_peaks_matched(images, z_min=5)

    found_local, _ = match(local, truth, 1.5)
    found, dist = match(matched, truth, 1.5)

    assert found_local.mean() < 0.5
    assert found.mean() > 0.9
    assert np.median(dist[found]) < 0.5
    # no detections on the stripe once it is removed
    assert not np.any(np.abs(matched[:, 1] - 95.5) < 3)

    unstriped, _, _ = find_peaks_matched(images, z_min=5, destripe_lines=False)
    assert np.sum(np.abs(unstriped[:, 1] - 95.5) < 3) > 0


def test_interpolate_positions_is_bilinear():
    pos = np.zeros((1, 4, 4, 3))
    pos[0, :, :, 0] = np.arange(4)[:, None]
    pos[0, :, :, 1] = np.arange(4)[None, :]
    xyz = interpolate_positions(pos, [[0, 1.25, 2.5]])
    assert np.allclose(xyz, [[1.25, 2.5, 0.0]])


def test_scattering_directions_norm_is_two_sin_theta():
    tt = np.deg2rad([20, 60, 120])
    xyz = np.column_stack([np.sin(tt), np.zeros(3), np.cos(tt)]) * 0.35
    d = scattering_directions(xyz)
    assert np.allclose(np.linalg.norm(d, axis=1), 2 * np.sin(tt / 2))


IDF = '<type name="panel" is="rectangular_detector" xpixels="4" ypixels="2"/>'


def test_read_lite(tmp_path):
    fn = tmp_path / "TEST_1.lite.nxs.h5"
    counts = [[1.0, 2.0], [], [3.0], [0.5, 0.5, 1.0], [], [], [], [2.0]]
    with h5py.File(fn, "w") as f:
        e = f.create_group("mantid_workspace_1")
        ev = e.create_group("event_workspace")
        ev["indices"] = np.cumsum([0] + [len(c) for c in counts])
        ev["weight"] = np.array([w for c in counts for w in c], dtype=np.float32)
        e["instrument/instrument_xml/data"] = np.array([IDF.encode()])
        e["instrument/name"] = np.array([b"IMAGINE"])
        e["logs/run_number/value"] = np.array([b"1"])
        e["title"] = np.array([b"test"])
        e["logs/goniometer/rotation_matrix"] = np.eye(3).ravel()

    info = read_lite(str(fn))

    assert np.allclose(info["counts"], [3, 0, 3, 2, 0, 0, 0, 2])
    assert info["pixel_shape"] == (4, 2)
    assert info["instrument"] == "IMAGINE"
    assert info["run_number"] == "1"
    assert np.allclose(info["goniometer"], np.eye(3))


def test_pixel_shape_from_idf():
    assert pixel_shape_from_idf(IDF) == (4, 2)


def test_grouped_panel_origin_correction():
    from quasi_laue_reduction.loading import correct_grouped_panel_origin, grouping_size

    raw_step = 0.00023809375
    lite = (
        f'<type name="panel" is="rectangular_detector" type="pixel"\n'
        f'      xpixels="128" xstart="{-256 * raw_step!r}" xstep="{4 * raw_step!r}"\n'
        f'      ypixels="128" ystart="{-256 * raw_step!r}" ystep="{4 * raw_step!r}" >'
    )
    fixed, changed = correct_grouped_panel_origin(lite, 4)
    assert changed

    start = float(re.search(r'xstart="([-\d.eE+]+)"', fixed).group(1))
    # centre of the first 4 raw pixels
    assert start == pytest.approx(-256 * raw_step + 1.5 * raw_step)

    # already-correct panels are left alone
    again, changed = correct_grouped_panel_origin(fixed, 4)
    assert not changed and again == fixed

    assert grouping_size("0+1+2+3+512+513+514+515+1024+1025+1026+1027+1536+1537+1538+1539,4+5") == 4
    assert grouping_size("0,1,2") == 1


def test_masked_border_gives_no_corner_peaks():
    """Zeroed (masked) edges must not create peaks at the panel corners."""
    rng = np.random.default_rng(3)
    images, truth = spot_images(rng, n_banks=2)
    images[:, :8, :] = 0
    images[:, -8:, :] = 0
    images[:, :, :8] = 0
    images[:, :, -8:] = 0
    coords, _, _ = find_peaks_local(images, n_sigma=8)
    near_corner = (np.minimum(coords[:, 1], 127 - coords[:, 1]) < 14) & (np.minimum(coords[:, 2], 127 - coords[:, 2]) < 14)
    assert not near_corner.any()


def test_packaged_calibration_parses():
    import json
    import os

    from quasi_laue_reduction.detcal import read_detcal
    from quasi_laue_reduction.loading import calibration_file

    for res, n in (("lite", 128), ("full", 512)):
        fn = calibration_file(resolution=res)
        l1, panels = read_detcal(fn)
        assert len(panels) == 80
        assert all(p["nrows"] == n and p["ncols"] == n for p in panels.values())
        for p in panels.values():
            assert np.isclose(np.linalg.norm(p["base"]), 1, atol=1e-4)
            assert abs(np.dot(p["base"], p["up"])) < 1e-4
            assert 0.2 < np.linalg.norm(p["centre"]) < 0.8
    prov = json.load(open(calibration_file().replace("_lite.DetCal", ".json")))
    assert prov["validation"]["selected"].startswith("D")
    assert os.path.basename(calibration_file()) in prov["files"].values()
