# Baseline for benchmarks: the orientation optimizer as it stood before the
# KD-tree indexer and persistent worker pool (copied unchanged from
# /HFIR/CG4D/shared/instrument/mantid/optimize_refine_ub.py, 2026-07-05).
# Not used by the package.
import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

import scipy.linalg
import scipy.optimize
import scipy.spatial
import scipy.interpolate


class CalculateUB:
    """
    Optimizer of crystal orientation from Laue peaks and known lattice
    parameters.

    The main global search uses wavelength-limited HKL shells. For each peak,
    possible HKLs are precomputed from

        |B hkl| in [|d_i| / lambda_max, |d_i| / lambda_min]

    where d_i = kf_hat - ki_hat and |d_i| = 2 sin(theta_i).

    Attributes
    ----------
    a : float
        Lattice constant a in angstroms.
    b : float
        Lattice constant b in angstroms.
    c : float
        Lattice constant c in angstroms.
    alpha : float
        Lattice angle alpha in degrees.
    beta : float
        Lattice angle beta in degrees.
    gamma : float
        Lattice angle gamma in degrees.
    """

    def __init__(self, a, b, c, alpha, beta, gamma):
        """
        Initialize UB calculator.

        Parameters
        ----------
        a : float
            Lattice constant a in angstroms.
        b : float
            Lattice constant b in angstroms.
        c : float
            Lattice constant c in angstroms.
        alpha : float
            Lattice angle alpha in degrees.
        beta : float
            Lattice angle beta in degrees.
        gamma : float
            Lattice angle gamma in degrees.
        """
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        t = np.linspace(0.0, np.pi, 4096)
        cdf = (t - np.sin(t)) / np.pi

        self._angle = scipy.interpolate.interp1d(
            cdf,
            t,
            kind="linear",
            bounds_error=False,
            fill_value=(0.0, np.pi),
        )

        self.x = None

        self._hkl = None
        self._g = None
        self._q = None
        self._candidate_ranges = None
        self._candidate_counts = None
        self._cache_B = None
        self._cache_wavelength = None

        self._score_angle_tol = np.deg2rad(0.35)
        self._score_index_bonus = 4.0
        self._score_outlier_cap = 3.0
        self._score_weights = None

        self.last_cost = None
        self.last_num_indexed = None
        self.last_angle_error = None

    # -------------------------------------------------------------------------
    # Lattice geometry
    # -------------------------------------------------------------------------

    def metric_G_tensor(self):
        """
        Calculate the direct-space metric tensor G.

        Returns
        -------
        G : ndarray
            Direct-space metric tensor with shape (3, 3).
        """
        alpha = np.deg2rad(self.alpha)
        beta = np.deg2rad(self.beta)
        gamma = np.deg2rad(self.gamma)

        g11 = self.a**2
        g22 = self.b**2
        g33 = self.c**2
        g12 = self.a * self.b * np.cos(gamma)
        g13 = self.a * self.c * np.cos(beta)
        g23 = self.b * self.c * np.cos(alpha)

        return np.array(
            [
                [g11, g12, g13],
                [g12, g22, g23],
                [g13, g23, g33],
            ]
        )

    def metric_G_star_tensor(self):
        """
        Calculate the reciprocal-space metric tensor G*.

        Returns
        -------
        Gstar : ndarray
            Reciprocal-space metric tensor with shape (3, 3).
        """
        return np.linalg.inv(self.metric_G_tensor())

    def reciprocal_lattice_B(self):
        """
        Calculate the reciprocal lattice B matrix.

        Returns
        -------
        B : ndarray
            Reciprocal lattice matrix with shape (3, 3).
        """
        Gstar = self.metric_G_star_tensor()

        return scipy.linalg.cholesky(Gstar, lower=False)

    def UB_matrix(self, U, B):
        """
        Calculate the UB matrix.

        Parameters
        ----------
        U : ndarray
            Orientation matrix with shape (3, 3).
        B : ndarray
            Reciprocal lattice matrix with shape (3, 3).

        Returns
        -------
        UB : ndarray
            Oriented reciprocal lattice matrix with shape (3, 3).
        """
        return U @ B

    def cartesian_matrix_metric_tensor(self, a, b, c, alpha, beta, gamma):
        """
        Calculate B and G* for lattice constants.

        Parameters
        ----------
        a : float
            Lattice constant a in angstroms.
        b : float
            Lattice constant b in angstroms.
        c : float
            Lattice constant c in angstroms.
        alpha : float
            Lattice angle alpha in radians.
        beta : float
            Lattice angle beta in radians.
        gamma : float
            Lattice angle gamma in radians.

        Returns
        -------
        B : ndarray
            Reciprocal lattice matrix with shape (3, 3).
        Gstar : ndarray
            Reciprocal metric tensor with shape (3, 3).
        """
        G = np.array(
            [
                [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
                [b * a * np.cos(gamma), b**2, b * c * np.cos(alpha)],
                [c * a * np.cos(beta), c * b * np.cos(alpha), c**2],
            ]
        )

        Gstar = np.linalg.inv(G)
        B = scipy.linalg.cholesky(Gstar, lower=False)

        return B, Gstar

    # -------------------------------------------------------------------------
    # Orientation parameterization
    # -------------------------------------------------------------------------

    def orientation_U(self, u0, u1, u2):
        """
        Calculate orientation matrix U from three parameters in [0, 1].

        Parameters
        ----------
        u0 : float
            Axis parameter.
        u1 : float
            Axis azimuth parameter.
        u2 : float
            Rotation-angle parameter.

        Returns
        -------
        U : ndarray
            Orientation matrix with shape (3, 3).
        """
        u0 = float(np.clip(u0, 0.0, 1.0))
        u1 = float(np.mod(u1, 1.0))
        u2 = float(np.clip(u2, 0.0, 1.0))

        theta = np.arccos(1.0 - 2.0 * u0)
        phi = 2.0 * np.pi * u1

        w = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )

        omega = float(self._angle(u2))

        return scipy.spatial.transform.Rotation.from_rotvec(
            omega * w
        ).as_matrix()

    def orientation_parameters_from_U(self, U):
        """
        Convert orientation matrix U to the internal three-parameter form.

        Parameters
        ----------
        U : ndarray
            Orientation matrix with shape (3, 3).

        Returns
        -------
        x : ndarray
            Orientation parameters with shape (3,).
        """
        rot = scipy.spatial.transform.Rotation.from_matrix(U)
        rotvec = rot.as_rotvec()

        omega = np.linalg.norm(rotvec)

        if omega < 1e-14:
            return np.array([0.0, 0.0, 0.0])

        w = rotvec / omega

        u0 = 0.5 * (1.0 - w[2])
        u1 = np.arctan2(w[1], w[0]) / (2.0 * np.pi)
        u1 = np.mod(u1, 1.0)

        u2 = (omega - np.sin(omega)) / np.pi
        u2 = np.clip(u2, 0.0, 1.0)

        return np.array([u0, u1, u2])

    # -------------------------------------------------------------------------
    # HKL candidate table
    # -------------------------------------------------------------------------

    def _build_hkl_table(
        self,
        B,
        q_max,
        q_min=0.0,
        hkl_filter=None,
        max_grid_points=20_000_000,
    ):
        """
        Build a global HKL table up to a maximum reciprocal length.

        Parameters
        ----------
        B : ndarray
            Reciprocal lattice matrix with shape (3, 3).
        q_max : float
            Maximum allowed reciprocal length.
        q_min : float, optional
            Minimum allowed reciprocal length.
        hkl_filter : callable, optional
            Function accepting H with shape (N, 3) and returning a Boolean
            mask. This can be used for centering or systematic absences.
        max_grid_points : int, optional
            Maximum number of raw integer grid points before ellipsoid
            filtering.

        Returns
        -------
        H : ndarray
            Integer HKL table with shape (N, 3).
        G : ndarray
            Cartesian reciprocal vectors B hkl with shape (N, 3).
        q : ndarray
            Reciprocal lengths with shape (N,).
        """
        A = B.T @ B
        Ainv = np.linalg.inv(A)

        bounds = np.ceil(q_max * np.sqrt(np.diag(Ainv))).astype(int) + 1
        bounds = np.maximum(bounds, 1)

        n_grid = np.prod(2 * bounds + 1)

        if n_grid > max_grid_points:
            raise ValueError(
                "HKL search grid is too large. "
                f"Grid points: {n_grid}. Bounds: {bounds}. "
                "Reduce wavelength range, use a larger d_min, or increase "
                "max_grid_points if this is expected."
            )

        hs = np.arange(-bounds[0], bounds[0] + 1)
        ks = np.arange(-bounds[1], bounds[1] + 1)
        ls = np.arange(-bounds[2], bounds[2] + 1)

        H = np.array(np.meshgrid(hs, ks, ls, indexing="ij"))
        H = H.reshape(3, -1).T

        H = H[np.any(H != 0, axis=1)]

        if hkl_filter is not None:
            keep = np.asarray(hkl_filter(H), dtype=bool)
            H = H[keep]

        G = H @ B.T
        q = np.linalg.norm(G, axis=1)

        keep = (q >= q_min) & (q <= q_max)
        H = H[keep]
        G = G[keep]
        q = q[keep]

        order = np.argsort(q)

        return H[order], G[order], q[order]

    def _build_candidate_ranges(self, kf_ki_dir, wavelength, q):
        """
        Build peak-specific HKL q-shell ranges.

        Parameters
        ----------
        kf_ki_dir : ndarray
            Observed kf_hat - ki_hat vectors with shape (N, 3).
        wavelength : tuple
            Wavelength band as (lambda_min, lambda_max).
        q : ndarray
            Sorted reciprocal lengths from the HKL table.

        Returns
        -------
        ranges : list
            List of (lo, hi) index ranges into the sorted HKL table.
        counts : ndarray
            Candidate count for each peak.
        """
        wl_min, wl_max = wavelength

        dnorm = np.linalg.norm(kf_ki_dir, axis=1)

        qmins = dnorm / wl_max
        qmaxs = dnorm / wl_min

        ranges = []
        counts = np.zeros(len(kf_ki_dir), dtype=int)

        for i, (qmin, qmax) in enumerate(zip(qmins, qmaxs)):
            lo = np.searchsorted(q, qmin, side="left")
            hi = np.searchsorted(q, qmax, side="right")

            ranges.append((lo, hi))
            counts[i] = hi - lo

        return ranges, counts

    def prepare_indexer(
        self,
        kf_ki_dir,
        wavelength,
        hkl_filter=None,
        q_margin=1e-12,
        peak_weights=None,
        ambiguity_power=0.5,
        max_grid_points=20_000_000,
        force=False,
    ):
        """
        Precompute global HKL table and per-peak q-shell candidates.

        Parameters
        ----------
        kf_ki_dir : ndarray
            Observed kf_hat - ki_hat vectors with shape (N, 3).
        wavelength : tuple
            Wavelength band as (lambda_min, lambda_max).
        hkl_filter : callable, optional
            Optional systematic-absence or centering filter.
        q_margin : float, optional
            Extra reciprocal-length margin added to global q bounds.
        peak_weights : ndarray, optional
            Optional weights, for example intensity-derived weights.
        ambiguity_power : float, optional
            Penalizes peaks with many candidate HKLs in the DE score. Use
            0 for no ambiguity weighting, 0.5 for sqrt(N) weighting, and 1
            for full 1/N weighting.
        max_grid_points : int, optional
            Maximum raw integer grid size.
        force : bool, optional
            If True, force rebuild of the global HKL table.

        Returns
        -------
        None
            Updates cached HKL tables and candidate ranges.
        """
        kf_ki_dir = np.asarray(kf_ki_dir, dtype=float)
        wavelength = tuple(float(x) for x in wavelength)

        wl_min, wl_max = wavelength

        if wl_min <= 0 or wl_max <= wl_min:
            raise ValueError("wavelength must satisfy 0 < min < max.")

        B = self.reciprocal_lattice_B()

        dnorm = np.linalg.norm(kf_ki_dir, axis=1)

        if np.any(dnorm <= 0):
            raise ValueError("All kf_ki_dir vectors must have nonzero norm.")

        q_min = max(0.0, dnorm.min() / wl_max - q_margin)
        q_max = dnorm.max() / wl_min + q_margin

        needs_table = (
            force
            or self._hkl is None
            or self._g is None
            or self._q is None
            or self._cache_B is None
            or not np.allclose(B, self._cache_B)
            or self._cache_wavelength != wavelength
        )

        if needs_table:
            H, G, q = self._build_hkl_table(
                B,
                q_max=q_max,
                q_min=q_min,
                hkl_filter=hkl_filter,
                max_grid_points=max_grid_points,
            )

            self._hkl = H
            self._g = G
            self._q = q
            self._cache_B = B.copy()
            self._cache_wavelength = wavelength

        ranges, counts = self._build_candidate_ranges(
            kf_ki_dir,
            wavelength,
            self._q,
        )

        self._candidate_ranges = ranges
        self._candidate_counts = counts

        if peak_weights is None:
            weights = np.ones(len(kf_ki_dir), dtype=float)
        else:
            weights = np.asarray(peak_weights, dtype=float)
            if weights.shape != (len(kf_ki_dir),):
                raise ValueError("peak_weights must have shape (N_peaks,).")
            weights = np.maximum(weights, 0.0)

        if ambiguity_power != 0:
            denom = np.maximum(counts, 1).astype(float) ** ambiguity_power
            weights = weights / denom

        if np.max(weights) > 0:
            weights = weights / np.mean(weights[weights > 0])

        self._score_weights = weights

    # -------------------------------------------------------------------------
    # Indexing and scoring
    # -------------------------------------------------------------------------

    def _index_from_U(
        self,
        U,
        kf_ki_dir,
        wavelength,
        angle_tol=None,
        index_bonus=None,
        outlier_cap=None,
        score_only=False,
    ):
        """
        Index peaks for a proposed U using precomputed HKL q-shell candidates.

        Parameters
        ----------
        U : ndarray
            Orientation matrix with shape (3, 3).
        kf_ki_dir : ndarray
            Observed kf_hat - ki_hat vectors with shape (N, 3).
        wavelength : tuple
            Wavelength band as (lambda_min, lambda_max).
        angle_tol : float, optional
            Angular indexing tolerance in radians.
        index_bonus : float, optional
            Bonus subtracted from the objective for each indexed peak.
        outlier_cap : float, optional
            Maximum normalized angular residual.
        score_only : bool, optional
            If True, do not allocate output HKL and wavelength arrays.

        Returns
        -------
        cost : float
            Robust indexing objective.
        num : int
            Number of indexed peaks.
        hkl : ndarray
            Indexed HKLs with shape (N, 3). Unindexed peaks are [0, 0, 0].
        lamda : ndarray
            Indexed wavelengths with shape (N,). Unindexed peaks are inf.
        """
        if self._hkl is None or self._candidate_ranges is None:
            self.prepare_indexer(kf_ki_dir, wavelength)

        if angle_tol is None:
            angle_tol = self._score_angle_tol

        if index_bonus is None:
            index_bonus = self._score_index_bonus

        if outlier_cap is None:
            outlier_cap = self._score_outlier_cap

        kf_ki_dir = np.asarray(kf_ki_dir, dtype=float)

        dnorm = np.linalg.norm(kf_ki_dir, axis=1)
        dhat = kf_ki_dir / dnorm[:, None]

        n_peaks = len(kf_ki_dir)

        if score_only:
            hkl_out = None
            lam_out = None
        else:
            hkl_out = np.zeros((n_peaks, 3), dtype=int)
            lam_out = np.full(n_peaks, np.inf)

        angle_error = np.full(n_peaks, np.pi)
        weights = self._score_weights

        if weights is None:
            weights = np.ones(n_peaks, dtype=float)

        cost = 0.0
        num = 0
        indexed_weight = 0.0

        for i, (lo, hi) in enumerate(self._candidate_ranges):
            if hi <= lo:
                z = outlier_cap
                cost += weights[i] * z**2
                continue

            G0 = self._g[lo:hi]
            q0 = self._q[lo:hi]

            G_lab = G0 @ U.T

            cosang = (G_lab @ dhat[i]) / q0
            cosang = np.clip(cosang, -1.0, 1.0)

            j = int(np.argmax(cosang))
            best_cos = cosang[j]
            best_angle = float(np.arccos(best_cos))

            angle_error[i] = best_angle

            z = min(best_angle / angle_tol, outlier_cap)
            cost += weights[i] * z**2

            if best_angle <= angle_tol:
                num += 1
                indexed_weight += weights[i]

                if not score_only:
                    idx = lo + j
                    hkl_out[i] = self._hkl[idx]
                    lam_out[i] = dnorm[i] / self._q[idx]

        cost -= index_bonus * indexed_weight

        self.last_cost = cost
        self.last_num_indexed = num
        self.last_angle_error = angle_error

        if score_only:
            return cost, num, None, None

        return cost, num, hkl_out, lam_out

    def indexer(
        self,
        UB,
        kf_ki_dir,
        wavelength,
        tol=0.1,
        alpha=None,
    ):
        """
        Compatibility wrapper around the new candidate-HKL indexer.

        Parameters
        ----------
        UB : ndarray
            UB matrix with shape (3, 3).
        kf_ki_dir : ndarray
            Observed kf_hat - ki_hat vectors with shape (N, 3).
        wavelength : tuple
            Wavelength band as (lambda_min, lambda_max).
        tol : float, optional
            Angular tolerance in radians. This replaces the old fractional
            HKL tolerance.
        alpha : float, optional
            Unused compatibility argument.

        Returns
        -------
        cost : float
            Robust indexing objective.
        num : int
            Number of indexed peaks.
        hkl : ndarray
            Indexed HKLs with shape (N, 3).
        lamda : ndarray
            Indexed wavelengths with shape (N,).
        """
        B = self.reciprocal_lattice_B()
        U = UB @ np.linalg.inv(B)

        self.prepare_indexer(kf_ki_dir, wavelength)

        return self._index_from_U(
            U,
            kf_ki_dir,
            wavelength,
            angle_tol=tol,
            score_only=False,
        )

    def cost(self, param):
        """
        Cost function for a proposed orientation.

        Parameters
        ----------
        param : ndarray
            Orientation parameters with shape (3,).

        Returns
        -------
        cost : float
            Robust indexing objective.
        """
        U = self.orientation_U(*param)

        cost, _, _, _ = self._index_from_U(
            U,
            self.kf_ki_dir,
            self.wavelength,
            score_only=True,
        )

        return float(cost)

    def objective(self, x):
        """
        Objective function for differential evolution.

        Parameters
        ----------
        x : ndarray
            Orientation parameters with shape (3,).

        Returns
        -------
        cost : float
            Robust indexing objective.
        """
        return self.cost(x)

    # -------------------------------------------------------------------------
    # Global search and orientation refinement
    # -------------------------------------------------------------------------

    def _wahba_refine_U(self, B, kf_ki_dir, hkl, peak_weights=None):
        """
        Refine U from assigned HKLs using Wahba/Kabsch alignment.

        Parameters
        ----------
        B : ndarray
            Reciprocal lattice matrix with shape (3, 3).
        kf_ki_dir : ndarray
            Observed kf_hat - ki_hat vectors with shape (N, 3).
        hkl : ndarray
            Assigned HKLs with shape (N, 3).
        peak_weights : ndarray, optional
            Optional peak weights with shape (N,).

        Returns
        -------
        U : ndarray or None
            Refined orientation matrix, or None if refinement is not possible.
        """
        hkl = np.asarray(hkl)
        kf_ki_dir = np.asarray(kf_ki_dir, dtype=float)

        mask = np.any(hkl != 0, axis=1)

        if np.sum(mask) < 3:
            return None

        G = hkl[mask] @ B.T
        q = np.linalg.norm(G, axis=1)

        good = q > 0

        G = G[good]
        q = q[good]

        d = kf_ki_dir[mask][good]
        dnorm = np.linalg.norm(d, axis=1)

        good = dnorm > 0

        G = G[good]
        q = q[good]
        d = d[good]
        dnorm = dnorm[good]

        if len(G) < 3:
            return None

        ghat = G / q[:, None]
        dhat = d / dnorm[:, None]

        if peak_weights is None:
            weights = None
        else:
            weights = np.asarray(peak_weights, dtype=float)[mask][good]
            weights = np.maximum(weights, 0.0)

            if np.sum(weights) <= 0:
                weights = None

        try:
            rot, _ = scipy.spatial.transform.Rotation.align_vectors(
                dhat,
                ghat,
                weights=weights,
            )
        except Exception:
            return None

        return rot.as_matrix()

    def minimize(
        self,
        kf_ki_dir,
        wavelength,
        n_proc=-1,
        peak_weights=None,
        hkl_filter=None,
        angle_tol=np.deg2rad(0.35),
        index_bonus=4.0,
        outlier_cap=3.0,
        ambiguity_power=0.5,
        maxiter=300,
        popsize=60,
        mutation=(0.5, 1.2),
        recombination=0.5,
        strategy="rand1bin",
        polish=False,
        n_reassign=3,
        max_grid_points=20_000_000,
    ):
        """
        Search for the UB matrix.

        Parameters
        ----------
        kf_ki_dir : ndarray
            Observed kf_hat - ki_hat vectors with shape (N, 3).
        wavelength : tuple
            Wavelength band as (lambda_min, lambda_max).
        n_proc : int, optional
            Number of workers passed to differential_evolution.
        peak_weights : ndarray, optional
            Optional weights with shape (N,).
        hkl_filter : callable, optional
            Optional function for systematic absences or centering.
        angle_tol : float, optional
            Angular indexing tolerance in radians.
        index_bonus : float, optional
            Bonus subtracted from objective for each indexed peak.
        outlier_cap : float, optional
            Maximum normalized angular residual.
        ambiguity_power : float, optional
            Down-weights peaks with many candidate HKLs.
        maxiter : int, optional
            Maximum DE iterations.
        popsize : int, optional
            DE population size multiplier.
        mutation : tuple, optional
            DE mutation range.
        recombination : float, optional
            DE recombination value.
        strategy : str, optional
            DE strategy.
        polish : bool, optional
            Whether to use scipy's DE polishing.
        n_reassign : int, optional
            Number of assignment/Wahba refinement cycles after DE.
        max_grid_points : int, optional
            Maximum raw HKL grid size.

        Returns
        -------
        UB : ndarray
            Refined UB matrix with shape (3, 3).
        hkl : ndarray
            Indexed HKLs with shape (N, 3).
        lamda : ndarray
            Indexed wavelengths with shape (N,).
        """
        self.kf_ki_dir = np.asarray(kf_ki_dir, dtype=float)
        self.wavelength = tuple(float(x) for x in wavelength)

        self._score_angle_tol = angle_tol
        self._score_index_bonus = index_bonus
        self._score_outlier_cap = outlier_cap

        self.prepare_indexer(
            self.kf_ki_dir,
            self.wavelength,
            hkl_filter=hkl_filter,
            peak_weights=peak_weights,
            ambiguity_power=ambiguity_power,
            max_grid_points=max_grid_points,
            force=True,
        )

        kwargs = dict(
            func=self.objective,
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            init="sobol",
            updating="deferred",
            strategy=strategy,
            polish=polish,
            workers=n_proc,
        )

        if self.x is not None:
            kwargs["x0"] = np.asarray(self.x, dtype=float)

        sol = scipy.optimize.differential_evolution(**kwargs)

        self.x = np.asarray(sol.x, dtype=float)

        B = self.reciprocal_lattice_B()

        for _ in range(n_reassign):
            _, num, hkl, _ = self.index(
                self.kf_ki_dir,
                self.wavelength,
                angle_tol=angle_tol,
            )

            if num < 3:
                break

            U_new = self._wahba_refine_U(
                B,
                self.kf_ki_dir,
                hkl,
                peak_weights=peak_weights,
            )

            if U_new is None:
                break

            self.x = self.orientation_parameters_from_U(U_new)

        cost, num, hkl, lamda = self.index(
            self.kf_ki_dir,
            self.wavelength,
            angle_tol=angle_tol,
        )

        U = self.orientation_U(*self.x)
        UB = self.UB_matrix(U, B)

        self.last_cost = cost
        self.last_num_indexed = num

        return UB, hkl, lamda

    def index(self, kf_ki_dir, wavelength, angle_tol=None):
        """
        Index peaks using the current orientation parameters.

        Parameters
        ----------
        kf_ki_dir : ndarray
            Observed kf_hat - ki_hat vectors with shape (N, 3).
        wavelength : tuple
            Wavelength band as (lambda_min, lambda_max).
        angle_tol : float, optional
            Angular indexing tolerance in radians.

        Returns
        -------
        cost : float
            Robust indexing objective.
        num : int
            Number of indexed peaks.
        hkl : ndarray
            Indexed HKLs with shape (N, 3).
        lamda : ndarray
            Indexed wavelengths with shape (N,).
        """
        if self.x is None:
            raise RuntimeError("No orientation is set. Run minimize() first.")

        self.kf_ki_dir = np.asarray(kf_ki_dir, dtype=float)
        self.wavelength = tuple(float(x) for x in wavelength)

        self.prepare_indexer(self.kf_ki_dir, self.wavelength)

        U = self.orientation_U(*self.x)

        return self._index_from_U(
            U,
            self.kf_ki_dir,
            self.wavelength,
            angle_tol=angle_tol,
            score_only=False,
        )

    # -------------------------------------------------------------------------
    # Cell constraints for refinement
    # -------------------------------------------------------------------------

    def cubic(self, x):
        a, *params = x

        return (a, a, a, 90.0, 90.0, 90.0, *params)

    def rhombohedral(self, x):
        a, alpha, *params = x

        return (a, a, a, alpha, alpha, alpha, *params)

    def tetragonal(self, x):
        a, c, *params = x

        return (a, a, c, 90.0, 90.0, 90.0, *params)

    def hexagonal(self, x):
        a, c, *params = x

        return (a, a, c, 90.0, 90.0, 120.0, *params)

    def orthorhombic(self, x):
        a, b, c, *params = x

        return (a, b, c, 90.0, 90.0, 90.0, *params)

    def monoclinic(self, x):
        a, b, c, beta, *params = x

        return (a, b, c, 90.0, beta, 90.0, *params)

    def triclinic(self, x):
        a, b, c, alpha, beta, gamma, *params = x

        return (a, b, c, alpha, beta, gamma, *params)

    def _cell_function(self, cell):
        """
        Return the constraint function and starting cell parameters.

        Parameters
        ----------
        cell : str
            Cell constraint name.

        Returns
        -------
        fun : callable
            Constraint function.
        x0 : tuple
            Starting reduced cell parameters.
        """
        a, b, c, alpha, beta, gamma = self.get_lattice_constants()

        fun_dict = {
            "Cubic": self.cubic,
            "Rhombohedral": self.rhombohedral,
            "Tetragonal": self.tetragonal,
            "Hexagonal": self.hexagonal,
            "Orthorhombic": self.orthorhombic,
            "Monoclinic": self.monoclinic,
            "Triclinic": self.triclinic,
        }

        x0_dict = {
            "Cubic": (a,),
            "Rhombohedral": (a, alpha),
            "Tetragonal": (a, c),
            "Hexagonal": (a, c),
            "Orthorhombic": (a, b, c),
            "Monoclinic": (a, b, c, beta),
            "Triclinic": (a, b, c, alpha, beta, gamma),
        }

        if cell not in fun_dict:
            raise ValueError(f"Unknown cell constraint: {cell}")

        return fun_dict[cell], x0_dict[cell]

    def _expand_cell_uncertainty(self, cell, sig):
        """
        Expand reduced-cell uncertainties to six lattice constants.

        Parameters
        ----------
        cell : str
            Cell constraint name.
        sig : ndarray
            Reduced-parameter uncertainties.

        Returns
        -------
        uncertainties : tuple
            Uncertainties for a, b, c, alpha, beta, gamma.
        """
        if cell == "Cubic":
            return (sig[0], sig[0], sig[0], 0.0, 0.0, 0.0)

        if cell == "Rhombohedral":
            return (sig[0], sig[0], sig[0], sig[1], sig[1], sig[1])

        if cell == "Tetragonal":
            return (sig[0], sig[0], sig[1], 0.0, 0.0, 0.0)

        if cell == "Hexagonal":
            return (sig[0], sig[0], sig[1], 0.0, 0.0, 0.0)

        if cell == "Orthorhombic":
            return (sig[0], sig[1], sig[2], 0.0, 0.0, 0.0)

        if cell == "Monoclinic":
            return (sig[0], sig[1], sig[2], 0.0, sig[3], 0.0)

        return tuple(sig[:6])

    # -------------------------------------------------------------------------
    # Lattice and UB refinement
    # -------------------------------------------------------------------------

    def residual(self, x, kf_ki_dir, hkl, wavelength, fun):
        """
        Residual for least-squares cell and orientation refinement.

        Parameters
        ----------
        x : ndarray
            Reduced cell parameters followed by orientation parameters.
        kf_ki_dir : ndarray
            Observed kf_hat - ki_hat vectors with shape (N, 3).
        hkl : ndarray
            Fixed assigned HKLs with shape (N, 3).
        wavelength : tuple
            Wavelength band as (lambda_min, lambda_max).
        fun : callable
            Cell constraint function.

        Returns
        -------
        residual : ndarray
            Flattened vector residuals.
        """
        a, b, c, alpha, beta, gamma, *u = fun(x)

        constants = a, b, c, *np.deg2rad([alpha, beta, gamma])
        B, Gstar = self.cartesian_matrix_metric_tensor(*constants)
        U = self.orientation_U(*u)
        UB = U @ B

        q = np.sqrt(np.einsum("ij,lj,li->l", Gstar, hkl, hkl))

        dnorm = np.linalg.norm(kf_ki_dir, axis=1)
        lamda = dnorm / q
        lamda = np.clip(lamda, *wavelength)

        pred = lamda[:, None] * (hkl @ UB.T)
        vec = pred - kf_ki_dir

        return vec.ravel()

    def refine(self, kf_ki_dir, wavelength, cell="Triclinic", error=0.15):
        """
        Refine orientation and lattice constants after initial indexing.

        Parameters
        ----------
        kf_ki_dir : ndarray
            Observed kf_hat - ki_hat vectors with shape (N, 3).
        wavelength : tuple
            Wavelength band as (lambda_min, lambda_max).
        cell : str, optional
            Cell constraint name.
        error : float, optional
            Fractional bound around initial lattice constants.

        Returns
        -------
        UB : ndarray
            Refined UB matrix with shape (3, 3).
        hkl : ndarray
            Re-indexed HKLs with shape (N, 3).
        lamda : ndarray
            Re-indexed wavelengths with shape (N,).
        uncertainties : tuple
            Uncertainties for a, b, c, alpha, beta, gamma.
        """
        if self.x is None:
            raise RuntimeError("No orientation is set. Run minimize() first.")

        self.cell = cell

        fun, x0_cell = self._cell_function(cell)

        _, _, hkl, _ = self.index(kf_ki_dir, wavelength)
        mask = np.any(hkl != 0, axis=1)

        if np.sum(mask) < len(x0_cell) + 3:
            raise RuntimeError("Not enough indexed peaks for refinement.")

        x0 = tuple(x0_cell) + tuple(self.x)

        x_min = [(1.0 - error) * v for v in x0_cell] + [0.0, 0.0, 0.0]
        x_max = [(1.0 + error) * v for v in x0_cell] + [1.0, 1.0, 1.0]

        bounds = (np.array(x_min), np.array(x_max))

        args = (
            np.asarray(kf_ki_dir, dtype=float)[mask],
            hkl[mask],
            tuple(float(x) for x in wavelength),
            fun,
        )

        sol = scipy.optimize.least_squares(
            self.residual,
            x0=np.asarray(x0, dtype=float),
            args=args,
            bounds=bounds,
        )

        a, b, c, alpha, beta, gamma, *u = fun(sol.x)

        self.set_lattice_constants(a, b, c, alpha, beta, gamma)
        self.x = np.asarray(u, dtype=float)

        J = sol.jac
        JTJ = J.T @ J

        dof = max(1, sol.fun.size - sol.x.size)
        chi2dof = np.sum(sol.fun**2) / dof

        cov = np.linalg.pinv(JTJ) * chi2dof
        sig = np.sqrt(np.maximum(np.diag(cov), 0.0))

        uncertainties = self._expand_cell_uncertainty(cell, sig)

        self.prepare_indexer(
            kf_ki_dir,
            wavelength,
            force=True,
        )

        _, _, hkl, lamda = self.index(kf_ki_dir, wavelength)

        B = self.reciprocal_lattice_B()
        U = self.orientation_U(*self.x)
        UB = U @ B

        return UB, hkl, lamda, uncertainties

    # -------------------------------------------------------------------------
    # Getters and setters
    # -------------------------------------------------------------------------

    def softplus(self, z):
        """
        Numerically stable softplus.

        Parameters
        ----------
        z : ndarray
            Input values.

        Returns
        -------
        y : ndarray
            softplus(z).
        """
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

    def get_lattice_constants(self):
        """
        Return lattice constants.

        Returns
        -------
        constants : tuple
            a, b, c, alpha, beta, gamma.
        """
        return self.a, self.b, self.c, self.alpha, self.beta, self.gamma

    def set_lattice_constants(self, a, b, c, alpha, beta, gamma):
        """
        Set lattice constants.

        Parameters
        ----------
        a : float
            Lattice constant a in angstroms.
        b : float
            Lattice constant b in angstroms.
        c : float
            Lattice constant c in angstroms.
        alpha : float
            Lattice angle alpha in degrees.
        beta : float
            Lattice angle beta in degrees.
        gamma : float
            Lattice angle gamma in degrees.

        Returns
        -------
        None
            Updates the instance.
        """
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_orientation_parameters(self):
        """
        Return orientation parameters.

        Returns
        -------
        x : ndarray
            Orientation parameters with shape (3,).
        """
        return self.x

    def set_orientation_parameters(self, x):
        """
        Set orientation parameters.

        Parameters
        ----------
        x : ndarray
            Orientation parameters with shape (3,).

        Returns
        -------
        None
            Updates the instance.
        """
        self.x = np.asarray(x, dtype=float)