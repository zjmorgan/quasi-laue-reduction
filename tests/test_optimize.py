import numpy as np
import pytest

from quasi_laue_reduction import optimize
from quasi_laue_reduction.optimize import (
    CalculateUB,
    centering_filter,
    primitive_cell,
    to_conventional,
)
from quasi_laue_reduction.simulate import (
    misorientation_deg,
    rotational_symmetry_ops,
    simulate_laue,
)

T4L = (61.5, 61.5, 95.9, 90.0, 90.0, 120.0)
WL = (2.8, 4.6)


def reference_index(opt, U, angle_tol):
    """Per-peak brute-force scan over each q-shell (the original indexer)."""
    n = len(opt._dnorm)
    w = opt._score_weights
    cap, bonus = opt._score_outlier_cap, opt._score_index_bonus
    cost, num = 0.0, 0
    hkl = np.zeros((n, 3), dtype=int)
    for i, (lo, hi) in enumerate(opt._candidate_ranges):
        if hi <= lo:
            cost += w[i] * cap**2
            continue
        cos = np.clip((opt._g[lo:hi] @ U.T) @ opt._dhat[i] / opt._q[lo:hi], -1, 1)
        j = int(np.argmax(cos))
        ang = float(np.arccos(cos[j]))
        cost += w[i] * min(ang / angle_tol, cap) ** 2
        if ang <= angle_tol:
            num += 1
            cost -= bonus * w[i]
            hkl[i] = opt._hkl[lo + j]
    return cost, num, hkl


def random_directions(n, rng):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v * rng.uniform(0.3, 1.9, n)[:, None]


@pytest.mark.parametrize("use_tree", [True, False])
def test_indexer_matches_bruteforce(monkeypatch, use_tree):
    if not use_tree:
        monkeypatch.setattr(optimize, "MAX_TREE_CANDIDATES", 0)

    rng = np.random.default_rng(1)
    kf = random_directions(40, rng)
    w = rng.uniform(0.5, 2, 40)

    opt = CalculateUB(*T4L)
    opt.prepare_indexer(kf, WL, peak_weights=w, force=True)

    assert (opt._tree is not None) == use_tree

    tol = 0.02
    for x in rng.random((20, 3)):
        U = opt.orientation_U(*x)
        cost, num, hkl, _ = opt._index_from_U(U, kf, WL, angle_tol=tol)
        ref_cost, ref_num, ref_hkl = reference_index(opt, U, tol)

        assert num == ref_num
        assert cost == pytest.approx(ref_cost, rel=1e-8, abs=1e-8)

        # only harmonic ties may differ, and then ours is the lowest order
        for i in np.flatnonzero(np.any(hkl != ref_hkl, axis=1)):
            assert np.allclose(np.cross(hkl[i], ref_hkl[i]), 0)
            assert np.linalg.norm(hkl[i]) <= np.linalg.norm(ref_hkl[i])


def test_orientation_roundtrip():
    opt = CalculateUB(5, 6, 7, 90, 90, 90)
    for x in np.random.default_rng(0).random((50, 3)):
        U = opt.orientation_U(*x)
        U2 = opt.orientation_U(*opt.orientation_parameters_from_U(U))
        assert np.allclose(U, U2, atol=1e-6)


def test_lookup_rebuilds_after_counts_only_prepare():
    rng = np.random.default_rng(2)
    opt = CalculateUB(*T4L)
    a, b = random_directions(20, rng), random_directions(30, rng)
    opt.prepare_indexer(a, WL, force=True)
    opt.prepare_indexer(b, WL, force=True, lookup=False)
    opt.x = rng.random(3)
    U = opt.orientation_U(*opt.x)
    _, num, hkl, _ = opt._index_from_U(U, b, WL, angle_tol=0.05)
    assert hkl.shape == (30, 3)
    assert num == reference_index(opt, U, 0.05)[1]


def test_table_extends_for_wider_peak_set():
    rng = np.random.default_rng(3)
    opt = CalculateUB(20, 22, 25, 90, 90, 90)
    small = random_directions(10, rng) * 0.3
    full = np.vstack([small, random_directions(10, rng)])
    opt.prepare_indexer(small, WL, force=True)
    q_max_small = opt._q.max()
    opt.prepare_indexer(full, WL)
    assert opt._q.max() > q_max_small
    fresh = CalculateUB(20, 22, 25, 90, 90, 90)
    fresh.prepare_indexer(full, WL, force=True)
    assert np.array_equal(opt._candidate_counts, fresh._candidate_counts)


def recover(cell, x_true, n_peaks=40, seed=0, **de):
    kf, h, _ = simulate_laue(cell, x_true, WL, d_min=1.5, n_peaks=n_peaks, rng=seed)
    opt = CalculateUB(*cell)
    _, _, _, num = opt.find_orientation(kf, WL, heights=h, n_proc=1, seed=seed, **de)
    ops = rotational_symmetry_ops(cell)
    mis = misorientation_deg(opt.orientation_U(*x_true), opt.orientation_U(*opt.x), ops)
    return num, len(kf), mis


def test_find_orientation_small_cell():
    num, n, mis = recover((8.0, 11.0, 14.0, 90, 90, 90), [0.3, 0.6, 0.4], popsize=30, maxiter=150)
    assert num >= 0.9 * n
    assert mis < 0.1


def test_centred_cell_through_primitive_search():
    conv = (14.0, 18.0, 22.0, 90.0, 90.0, 90.0)
    prim_cell = primitive_cell(*conv, "F")
    x_true = [0.7, 0.2, 0.55]

    # fundamental reflections of the primitive lattice = allowed F reflections
    kf, h, _ = simulate_laue(prim_cell, x_true, WL, d_min=1.5, n_peaks=40, rng=1)

    prim = CalculateUB(*prim_cell)
    # default (full-size) search: the small pop=60 x 300 search fails here
    prim.find_orientation(kf, WL, heights=h, n_proc=-1, seed=1)

    conv_opt, M = to_conventional(prim, "F", conv)
    _, num, hkl, _ = conv_opt.index(kf, WL)

    assert num >= 0.9 * len(kf)
    ok = np.any(hkl != 0, axis=1)
    assert np.all(centering_filter("F")(hkl[ok]))

    # true conventional orientation from the true primitive UB
    true_p = CalculateUB(*prim_cell)
    UB_c = true_p.orientation_U(*x_true) @ true_p.reciprocal_lattice_B() @ np.linalg.inv(M)
    u, _, vt = np.linalg.svd(UB_c @ np.linalg.inv(CalculateUB(*conv).reciprocal_lattice_B()))

    ops = rotational_symmetry_ops(conv)
    assert misorientation_deg(u @ vt, conv_opt.orientation_U(*conv_opt.x), ops) < 0.1


@pytest.mark.parametrize("start_method", ["fork", "forkserver"])
def test_parallel_matches_serial(start_method):
    cell = (8.0, 11.0, 14.0, 90, 90, 90)
    kf, h, _ = simulate_laue(cell, [0.3, 0.6, 0.4], WL, d_min=1.5, n_peaks=40, rng=0)
    xs = []
    for n_proc in (1, 3):
        opt = CalculateUB(*cell)
        opt.minimize(kf, WL, n_proc=n_proc, seed=5, popsize=10, maxiter=20, n_reassign=0, start_method=start_method)
        xs.append(opt.de_result.x)
    assert np.allclose(xs[0], xs[1])


def test_centering_matrices_are_right_handed():
    for key, P in optimize.CENTERING_MATRICES.items():
        assert np.linalg.det(P) > 0, key


def test_primitive_cell_volume():
    conv = (18.40, 56.65, 6.54, 90, 90, 90)
    for c, factor in [("P", 1), ("A", 2), ("B", 2), ("C", 2), ("I", 2), ("F", 4)]:
        vc = np.sqrt(np.linalg.det(CalculateUB(*conv).metric_G_tensor()))
        vp = np.sqrt(np.linalg.det(CalculateUB(*primitive_cell(*conv, c)).metric_G_tensor()))
        assert vc / vp == pytest.approx(factor)


def test_refine_recovers_cell():
    cell = (8.0, 11.0, 14.0, 90, 90, 90)
    x_true = [0.3, 0.6, 0.4]
    kf, _, _ = simulate_laue(cell, x_true, WL, d_min=1.5, n_peaks=60, noise_deg=0.02, rng=4)
    opt = CalculateUB(8.1, 10.9, 14.1, 90, 90, 90)
    opt.x = np.array(x_true)
    opt._score_angle_tol = np.deg2rad(1.0)
    opt.refine(kf, WL, "Orthorhombic")
    assert opt.a == pytest.approx(8.0, rel=2e-3)
    assert opt.b == pytest.approx(11.0, rel=2e-3)
    assert opt.c == pytest.approx(14.0, rel=2e-3)


@pytest.mark.slow
def test_find_orientation_protein_cell():
    """Full-size search on the T4 lysozyme cell (about a minute on 16 cores)."""
    kf, h, _ = simulate_laue(T4L, [0.41, 0.77, 0.23], WL, d_min=2.5, n_peaks=99, rng=11)
    opt = CalculateUB(*T4L)
    _, _, _, num = opt.find_orientation(kf, WL, heights=h, n_proc=-1, seed=1)
    ops = rotational_symmetry_ops(T4L)
    assert num >= 90
    assert misorientation_deg(opt.orientation_U(0.41, 0.77, 0.23), opt.orientation_U(*opt.x), ops) < 0.1
