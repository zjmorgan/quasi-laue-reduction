import h5py
import numpy as np

from quasi_laue_reduction.loading import pixel_shape_from_idf, read_lite
from quasi_laue_reduction.peaks import (
    find_peaks_global,
    find_peaks_local,
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
