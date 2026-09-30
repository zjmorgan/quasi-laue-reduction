"""
Synthetic quasi-Laue peak lists for a known orientation.

Used for tests and benchmarks: the recovered orientation can be compared
with the truth, and the search can be exercised without facility data.
"""

import numpy as np
import scipy.spatial

from .optimize import CalculateUB


def simulate_laue(
    cell,
    x,
    wavelength,
    d_min=2.5,
    n_peaks=99,
    detector_directions=None,
    coverage_deg=0.6,
    two_theta_range=(10.0, 80.0),
    noise_deg=0.08,
    intensity_power=4.0,
    rng=None,
):
    """
    Simulate observed peak directions for orientation parameters ``x``.

    Only fundamental reflections (gcd(h, k, l) = 1) are generated, since
    harmonics land on the same pixel. The beam travels along +z.

    Parameters
    ----------
    cell : tuple
        a, b, c, alpha, beta, gamma (angles in degrees).
    x : array_like
        Orientation parameters for :meth:`CalculateUB.orientation_U`.
    wavelength : tuple
        Wavelength band as (lambda_min, lambda_max).
    d_min : float, optional
        Resolution limit in angstroms.
    n_peaks : int, optional
        Number of peaks to keep.
    detector_directions : ndarray, optional
        Unit kf vectors of detector pixels with shape (M, 3). A reflection
        is observed if a pixel lies within ``coverage_deg``. If omitted,
        every direction within ``two_theta_range`` is observed.
    coverage_deg : float, optional
        Pixel matching tolerance for ``detector_directions``.
    two_theta_range : tuple, optional
        Scattering-angle window (degrees) when no pixels are given.
    noise_deg : float, optional
        Gaussian angular noise added to each kf direction.
    intensity_power : float, optional
        Peaks are drawn with probability proportional to d**power, mimicking
        the stronger low-resolution reflections that a peak finder keeps.
    rng : numpy.random.Generator or int, optional
        Random source.

    Returns
    -------
    kf_ki_dir : ndarray
        Simulated kf_hat - ki_hat with shape (n, 3).
    heights : ndarray
        Pseudo peak heights with shape (n,).
    hkl : ndarray
        True HKLs with shape (n, 3).
    """
    rng = np.random.default_rng(rng)

    opt = CalculateUB(*cell)
    B = opt.reciprocal_lattice_B()
    U = opt.orientation_U(*x)

    H, _, _ = opt._build_hkl_table(B, q_max=1.0 / d_min)
    H = H[np.gcd.reduce(np.abs(H), axis=1) == 1]

    G = H @ (U @ B).T
    G2 = np.sum(G**2, axis=1)

    # Laue condition with ki = z / lambda: |ki + G| = |ki|
    lam = -2.0 * G[:, 2] / G2

    ok = (lam >= wavelength[0]) & (lam <= wavelength[1])
    G, lam, H = G[ok], lam[ok], H[ok]

    kf = np.array([0.0, 0.0, 1.0]) + lam[:, None] * G

    if detector_directions is None:
        tt = np.degrees(np.arccos(np.clip(kf[:, 2], -1.0, 1.0)))
        seen = (tt >= two_theta_range[0]) & (tt <= two_theta_range[1])
    else:
        tree = scipy.spatial.cKDTree(detector_directions)
        dist, _ = tree.query(kf)
        seen = dist < np.deg2rad(coverage_deg)

    G, H, kf = G[seen], H[seen], kf[seen]

    if len(kf) == 0:
        raise ValueError("No reflections fall on the detector.")

    d = 1.0 / np.linalg.norm(G, axis=1)

    p = d**intensity_power
    p /= p.sum()

    pick = rng.choice(len(kf), size=min(n_peaks, len(kf)), replace=False, p=p)

    kf = kf[pick]

    noise = np.deg2rad(noise_deg) * rng.normal(size=kf.shape)
    noise -= np.sum(noise * kf, axis=1)[:, None] * kf

    kf = kf + noise
    kf /= np.linalg.norm(kf, axis=1)[:, None]

    heights = 1000.0 * (d[pick] / d[pick].min()) ** 2 * rng.uniform(0.7, 1.3, len(pick))

    return kf - np.array([0.0, 0.0, 1.0]), heights, H[pick]


def rotational_symmetry_ops(cell, tol=1e-6):
    """
    Proper rotations of the lattice metric as Cartesian matrices B R B^-1.

    These are the operations that map the lattice onto itself, so two
    orientations U1 and U2 are equivalent if U2 = U1 M for one of them.
    """
    import itertools

    opt = CalculateUB(*cell)
    B = opt.reciprocal_lattice_B()
    Gs = B.T @ B

    ops = []

    for m in itertools.product([-1, 0, 1], repeat=9):
        R = np.array(m).reshape(3, 3)
        if round(np.linalg.det(R)) == 1 and np.allclose(R.T @ Gs @ R, Gs, atol=tol * np.abs(Gs).max()):
            ops.append(B @ R @ np.linalg.inv(B))

    return ops


def misorientation_deg(U1, U2, ops):
    """
    Smallest rotation angle between U1 and U2 over lattice symmetry ops.
    """
    return min(
        np.degrees(np.arccos(np.clip((np.trace(U1.T @ U2 @ M.T) - 1.0) / 2.0, -1.0, 1.0)))
        for M in ops
    )
