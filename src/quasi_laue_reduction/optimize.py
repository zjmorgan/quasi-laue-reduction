"""
Laue orientation search and lattice refinement from peak directions.

For a quasi-Laue measurement the wavelength of each reflection is unknown,
so a peak only fixes the direction of its scattering vector,

    d_i = kf_hat - ki_hat,    |d_i| = 2 sin(theta_i),

and the reflection lies somewhere on the segment d_i / lambda with lambda
in the incident band. For a trial orientation U the candidate reflections
of peak i are the HKLs whose reciprocal length falls in the shell

    |B hkl| in [|d_i| / lambda_max, |d_i| / lambda_min],

and the best candidate is the one whose direction U B hkl is closest to
d_i. The orientation is found by a global differential-evolution search
over a robust, capped angular objective, followed by Wahba (Kabsch)
reassignment cycles and an optional constrained least-squares refinement
of the lattice constants.
"""

import multiprocessing
import os

import numpy as np
import scipy.linalg
import scipy.interpolate
import scipy.optimize
import scipy.spatial

# Nearest-neighbour lookup of the best candidate per peak is only worth
# building a KD-tree for when the candidate count is moderate; beyond this
# the one-off indexing calls use the vectorised scan instead.
MAX_TREE_CANDIDATES = 5_000_000

# Separation along the fourth (peak label) axis of the stacked KD-tree. Any
# two unit vectors are at most 2 apart, so a gap > 2 keeps every query
# inside its own peak's shell.
_PEAK_AXIS_GAP = 4.0

# Worker state for the persistent differential-evolution pool.
_WORKER_OPT = None


def _init_worker(opt):
    global _WORKER_OPT
    _WORKER_OPT = opt
    os.environ["OMP_NUM_THREADS"] = "1"


def _evaluate_chunk(xs):
    return [_WORKER_OPT.objective(x) for x in xs]


class ChunkedPoolMap:
    """
    Map-like callable for ``differential_evolution(workers=...)``.

    SciPy's integer ``workers`` option pickles the objective (a bound method,
    and so the whole optimizer including its HKL table) with every task. Here
    the optimizer is handed to each worker once when the pool starts, and
    each generation sends only the trial vectors, split into one contiguous
    chunk per worker.

    Parameters
    ----------
    opt : CalculateUB
        Prepared optimizer whose ``objective`` is evaluated by the workers.
    n_proc : int
        Number of worker processes. ``-1`` uses all CPUs.
    start_method : str, optional
        Multiprocessing start method. ``"fork"`` (default where available)
        shares the prepared optimizer without pickling and starts fastest;
        ``"forkserver"`` or ``"spawn"`` avoid forking a multi-threaded
        process (e.g. inside Mantid Workbench) at the cost of pickling the
        optimizer once per worker, which then rebuilds its lookup.
    """

    def __init__(self, opt, n_proc, start_method=None):
        self.n_proc = multiprocessing.cpu_count() if n_proc in (-1, None) else n_proc
        if start_method is None:
            start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else None
        ctx = multiprocessing.get_context(start_method)
        self.pool = ctx.Pool(self.n_proc, initializer=_init_worker, initargs=(opt,))

    def __call__(self, func, iterable):
        xs = np.asarray(list(iterable))
        chunks = [c for c in np.array_split(xs, self.n_proc) if len(c)]
        out = self.pool.map(_evaluate_chunk, chunks)
        return [v for chunk in out for v in chunk]

    def close(self):
        self.pool.close()
        self.pool.join()


class CalculateUB:
    """
    Optimizer of crystal orientation from Laue peaks and known lattice
    parameters.

    Parameters
    ----------
    a, b, c : float
        Lattice constants in angstroms.
    alpha, beta, gamma : float
        Lattice angles in degrees.
    """

    def __init__(self, a, b, c, alpha, beta, gamma):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Rotation angle omega with density (1 - cos omega) / pi, so that
        # uniform (u0, u1, u2) in [0, 1]^3 samples SO(3) uniformly.
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
        self._q_bounds = (np.inf, -np.inf)
        self._hkl_filter = None

        self._tree = None
        self._tree_rows = None
        self._tree_key = None
        self._lookup_valid = False

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
        Direct-space metric tensor G with shape (3, 3).
        """
        alpha, beta, gamma = np.deg2rad([self.alpha, self.beta, self.gamma])

        return np.array(
            [
                [self.a**2, self.a * self.b * np.cos(gamma), self.a * self.c * np.cos(beta)],
                [self.a * self.b * np.cos(gamma), self.b**2, self.b * self.c * np.cos(alpha)],
                [self.a * self.c * np.cos(beta), self.b * self.c * np.cos(alpha), self.c**2],
            ]
        )

    def metric_G_star_tensor(self):
        """
        Reciprocal-space metric tensor G* with shape (3, 3).
        """
        return np.linalg.inv(self.metric_G_tensor())

    def reciprocal_lattice_B(self):
        """
        Reciprocal lattice B matrix (upper triangular, Busing-Levy/Mantid
        convention without 2 pi) with shape (3, 3).
        """
        return scipy.linalg.cholesky(self.metric_G_star_tensor(), lower=False)

    def UB_matrix(self, U, B):
        return U @ B

    def cartesian_matrix_metric_tensor(self, a, b, c, alpha, beta, gamma):
        """
        B and G* for lattice constants with angles in radians.
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
        Orientation matrix U from three parameters in [0, 1].

        ``u0`` and ``u1`` set the rotation axis uniformly on the sphere and
        ``u2`` the rotation angle, so a uniform draw is a uniform rotation.
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

        return scipy.spatial.transform.Rotation.from_rotvec(omega * w).as_matrix()

    def orientation_parameters_from_U(self, U):
        """
        Inverse of :meth:`orientation_U`.
        """
        rotvec = scipy.spatial.transform.Rotation.from_matrix(U).as_rotvec()

        omega = np.linalg.norm(rotvec)

        if omega < 1e-14:
            return np.array([0.0, 0.0, 0.0])

        w = rotvec / omega

        u0 = 0.5 * (1.0 - w[2])
        u1 = np.mod(np.arctan2(w[1], w[0]) / (2.0 * np.pi), 1.0)
        u2 = np.clip((omega - np.sin(omega)) / np.pi, 0.0, 1.0)

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
        Global HKL table sorted by reciprocal length.

        Parameters
        ----------
        B : ndarray
            Reciprocal lattice matrix with shape (3, 3).
        q_max, q_min : float
            Reciprocal-length limits.
        hkl_filter : callable, optional
            Function of H with shape (N, 3) returning a Boolean mask, for
            centering or systematic absences.
        max_grid_points : int, optional
            Maximum raw integer grid size before filtering.

        Returns
        -------
        H : ndarray
            Integer HKLs with shape (N, 3).
        G : ndarray
            Cartesian vectors B hkl with shape (N, 3).
        q : ndarray
            Reciprocal lengths with shape (N,), ascending.
        """
        Ainv = np.linalg.inv(B.T @ B)

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

        H = np.array(
            np.meshgrid(
                *[np.arange(-n, n + 1) for n in bounds],
                indexing="ij",
            )
        )
        H = H.reshape(3, -1).T
        H = H[np.any(H != 0, axis=1)]

        if hkl_filter is not None:
            H = H[np.asarray(hkl_filter(H), dtype=bool)]

        G = H @ B.T
        q = np.linalg.norm(G, axis=1)

        keep = (q >= q_min) & (q <= q_max)
        H, G, q = H[keep], G[keep], q[keep]

        order = np.argsort(q, kind="stable")

        return H[order], G[order], q[order]

    def _build_candidate_ranges(self, kf_ki_dir, wavelength, q):
        """
        Per-peak (lo, hi) index ranges of the q-shell into the sorted table.
        """
        wl_min, wl_max = wavelength

        dnorm = np.linalg.norm(kf_ki_dir, axis=1)

        lo = np.searchsorted(q, dnorm / wl_max, side="left")
        hi = np.searchsorted(q, dnorm / wl_min, side="right")

        return list(zip(lo.tolist(), hi.tolist())), hi - lo

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
        lookup=True,
    ):
        """
        Precompute the HKL table, per-peak q-shell candidates, score weights
        and the nearest-direction lookup.

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
            Down-weights peaks with many candidate HKLs: weight is divided
            by N_candidates**ambiguity_power (0 disables).
        max_grid_points : int, optional
            Maximum raw integer grid size.
        force : bool, optional
            Force a rebuild of the global HKL table.
        lookup : bool, optional
            Build the nearest-direction lookup (not needed when only the
            candidate counts are used).
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

        if hkl_filter is not None:
            self._hkl_filter = hkl_filter

        needs_table = (
            force
            or self._hkl is None
            or self._cache_B is None
            or not np.allclose(B, self._cache_B)
            or self._cache_wavelength != wavelength
            or q_min < self._q_bounds[0]
            or q_max > self._q_bounds[1]
        )

        if needs_table:
            self._hkl, self._g, self._q = self._build_hkl_table(
                B,
                q_max=q_max,
                q_min=q_min,
                hkl_filter=self._hkl_filter,
                max_grid_points=max_grid_points,
            )
            self._q_bounds = (q_min, q_max)
            self._cache_B = B.copy()
            self._cache_wavelength = wavelength
            self._tree_key = None

        ranges, counts = self._build_candidate_ranges(kf_ki_dir, wavelength, self._q)

        self._candidate_ranges = ranges
        self._candidate_counts = counts
        self._dnorm = dnorm
        self._dhat = kf_ki_dir / dnorm[:, None]
        self._nonempty = counts > 0

        if peak_weights is None:
            weights = np.ones(len(kf_ki_dir), dtype=float)
        else:
            weights = np.asarray(peak_weights, dtype=float)
            if weights.shape != (len(kf_ki_dir),):
                raise ValueError("peak_weights must have shape (N_peaks,).")
            weights = np.maximum(weights, 0.0)

        if ambiguity_power != 0:
            weights = weights / np.maximum(counts, 1).astype(float) ** ambiguity_power

        if np.max(weights) > 0:
            weights = weights / np.mean(weights[weights > 0])

        self._score_weights = weights

        self._prepared_kf = kf_ki_dir
        self._lookup_valid = False

        if lookup:
            self._prepare_lookup(kf_ki_dir)

    def _prepare_lookup(self, kf_ki_dir):
        """
        Build the stacked KD-tree of candidate directions, or the flattened
        segments for the vectorised scan when there are too many candidates.

        Each shell is reduced to one representative per reciprocal-lattice
        row: harmonics n*hkl are collinear, so only the lowest order (the
        first in the q-sorted table) can be the unique best match. Candidate
        directions of peak i are lifted to (ghat, i * gap), so the nearest
        neighbour of (U^T d_i, i * gap) is peak i's best in-shell candidate;
        chord length is monotone in angle, so this is the same argmax as a
        brute-force scan.
        """
        key = (id(self._hkl), kf_ki_dir.shape, hash(kf_ki_dir.tobytes()))

        if self._tree_key == key:
            self._lookup_valid = True
            return

        n_cand = int(np.sum(self._candidate_counts))

        H = self._hkl
        prim = H // np.gcd.reduce(np.abs(H), axis=1)[:, None]
        m = int(np.abs(prim).max()) + 1 if len(prim) else 1
        row_key = ((prim[:, 0] + m) * (2 * m + 1) + (prim[:, 1] + m)) * (2 * m + 1) + (prim[:, 2] + m)

        ghat = self._g / self._q[:, None]

        rows, labels = [], []

        for i, (lo, hi) in enumerate(self._candidate_ranges):
            if hi <= lo:
                continue
            _, first = np.unique(row_key[lo:hi], return_index=True)
            idx = lo + np.sort(first)
            rows.append(idx)
            labels.append(np.full(len(idx), i))

        if rows:
            rows = np.concatenate(rows)
            labels = np.concatenate(labels)
        else:
            rows = np.zeros(0, dtype=int)
            labels = np.zeros(0, dtype=int)

        if 0 < len(rows) <= MAX_TREE_CANDIDATES:
            pts = np.column_stack([ghat[rows], labels * _PEAK_AXIS_GAP])
            self._tree = scipy.spatial.cKDTree(pts, balanced_tree=False, compact_nodes=False)
        else:
            self._tree = None

        # per-peak shells for the scan path used when there is no tree
        self._seg_ghat = []
        if self._tree is None and len(rows):
            starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
            for lo, hi in zip(starts, np.r_[starts[1:], len(rows)]):
                self._seg_ghat.append((lo, np.ascontiguousarray(ghat[rows[lo:hi]])))

        self._tree_rows = rows
        self._n_candidates = n_cand
        self._tree_key = key
        self._lookup_valid = True

    # -------------------------------------------------------------------------
    # Indexing and scoring
    # -------------------------------------------------------------------------

    def _best_candidates(self, U):
        """
        Best candidate row and its angle for every peak with candidates.

        Returns
        -------
        angle : ndarray
            Best angular error (radians) for nonempty peaks, in peak order.
        row : ndarray
            Index into the HKL table of the best candidate.
        """
        ne = self._nonempty

        if not np.any(ne):
            return np.zeros(0), np.zeros(0, dtype=int)

        r = self._dhat[ne] @ U  # rows are U^T d_hat

        if self._tree is not None:
            lab = np.flatnonzero(ne) * _PEAK_AXIS_GAP
            dist, j = self._tree.query(np.column_stack([r, lab]))
            angle = 2.0 * np.arcsin(np.minimum(dist / 2.0, 1.0))
            return angle, self._tree_rows[j]

        best = np.empty(len(r))
        j = np.empty(len(r), dtype=int)

        for n, (lo, g) in enumerate(self._seg_ghat):
            cos = g @ r[n]
            k = int(np.argmax(cos))
            best[n] = cos[k]
            j[n] = lo + k

        return np.arccos(np.clip(best, -1.0, 1.0)), self._tree_rows[j]

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
        Index peaks for a proposed U using the precomputed candidates.

        The objective is sum_i w_i min(angle_i / tol, cap)^2 minus
        ``index_bonus`` times the weight of peaks within ``tol``; peaks with
        no candidates contribute w_i cap^2.

        Returns
        -------
        cost : float
            Robust indexing objective.
        num : int
            Number of indexed peaks.
        hkl : ndarray or None
            Indexed HKLs with shape (N, 3); unindexed peaks are [0, 0, 0].
        lamda : ndarray or None
            Indexed wavelengths with shape (N,); unindexed peaks are inf.
        """
        if self._hkl is None or self._candidate_ranges is None:
            self.prepare_indexer(kf_ki_dir, wavelength)

        if not self._lookup_valid:
            self._prepare_lookup(self._prepared_kf)

        angle_tol = self._score_angle_tol if angle_tol is None else angle_tol
        index_bonus = self._score_index_bonus if index_bonus is None else index_bonus
        outlier_cap = self._score_outlier_cap if outlier_cap is None else outlier_cap

        n_peaks = len(self._dnorm)
        weights = self._score_weights

        if weights is None:
            weights = np.ones(n_peaks, dtype=float)

        ne = self._nonempty
        best_angle, rows = self._best_candidates(U)

        angle_error = np.full(n_peaks, np.pi)
        angle_error[ne] = best_angle

        z = np.full(n_peaks, outlier_cap)
        z[ne] = np.minimum(best_angle / angle_tol, outlier_cap)

        ok = np.zeros(n_peaks, dtype=bool)
        ok[ne] = best_angle <= angle_tol

        cost = float(np.sum(weights * z**2) - index_bonus * np.sum(weights[ok]))
        num = int(np.sum(ok))

        self.last_cost = cost
        self.last_num_indexed = num
        self.last_angle_error = angle_error

        if score_only:
            return cost, num, None, None

        table_row = np.zeros(n_peaks, dtype=int)
        table_row[ne] = rows

        hkl_out = np.zeros((n_peaks, 3), dtype=int)
        lam_out = np.full(n_peaks, np.inf)

        hkl_out[ok] = self._hkl[table_row[ok]]
        lam_out[ok] = self._dnorm[ok] / self._q[table_row[ok]]

        return cost, num, hkl_out, lam_out

    def indexer(self, UB, kf_ki_dir, wavelength, tol=0.1):
        """
        Index peaks for a UB matrix; ``tol`` is an angle in radians.
        """
        U = UB @ np.linalg.inv(self.reciprocal_lattice_B())

        self.prepare_indexer(kf_ki_dir, wavelength)

        return self._index_from_U(U, kf_ki_dir, wavelength, angle_tol=tol)

    def cost(self, param):
        """
        Indexing objective for orientation parameters ``param``.
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
        return self.cost(x)

    # -------------------------------------------------------------------------
    # Global search and orientation refinement
    # -------------------------------------------------------------------------

    def _wahba_refine_U(self, B, kf_ki_dir, hkl, peak_weights=None):
        """
        Refine U from assigned HKLs by weighted Wahba/Kabsch alignment.

        Returns None if fewer than three peaks are assigned.
        """
        hkl = np.asarray(hkl)
        kf_ki_dir = np.asarray(kf_ki_dir, dtype=float)

        mask = np.any(hkl != 0, axis=1)

        if np.sum(mask) < 3:
            return None

        G = hkl[mask] @ B.T
        d = kf_ki_dir[mask]

        ghat = G / np.linalg.norm(G, axis=1)[:, None]
        dhat = d / np.linalg.norm(d, axis=1)[:, None]

        weights = None

        if peak_weights is not None:
            weights = np.maximum(np.asarray(peak_weights, dtype=float)[mask], 0.0)
            if np.sum(weights) <= 0:
                weights = None

        try:
            rot, _ = scipy.spatial.transform.Rotation.align_vectors(dhat, ghat, weights=weights)
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
        maxiter=1000,
        popsize=300,
        mutation=(0.5, 1.2),
        recombination=0.5,
        strategy="rand1bin",
        polish=False,
        n_reassign=3,
        max_grid_points=20_000_000,
        seed=None,
        start_method=None,
    ):
        """
        Global orientation search by parallel differential evolution.

        Parameters
        ----------
        kf_ki_dir : ndarray
            Observed kf_hat - ki_hat vectors with shape (N, 3).
        wavelength : tuple
            Wavelength band as (lambda_min, lambda_max).
        n_proc : int, optional
            Worker processes for the population evaluation (``-1`` all
            CPUs, ``1`` serial).
        peak_weights : ndarray, optional
            Optional weights with shape (N,).
        hkl_filter : callable, optional
            Optional function for systematic absences or centering.
        angle_tol : float, optional
            Angular indexing tolerance in radians.
        index_bonus, outlier_cap, ambiguity_power : float, optional
            Objective shape, see :meth:`_index_from_U` and
            :meth:`prepare_indexer`.
        maxiter, popsize, mutation, recombination, strategy, polish
            Passed to ``scipy.optimize.differential_evolution``. The default
            population (300 x 3) and generation count are sized for large
            protein cells, where the basin of the true orientation is only
            ~ angle_tol wide; smaller searches find it unreliably.
        n_reassign : int, optional
            Assignment/Wahba refinement cycles after DE.
        max_grid_points : int, optional
            Maximum raw HKL grid size.
        seed : int, optional
            Seed for the DE random number generator.
        start_method : str, optional
            Worker start method, see :class:`ChunkedPoolMap`.

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
            rng=seed,
        )

        if self.x is not None:
            kwargs["x0"] = np.asarray(self.x, dtype=float)

        pool = ChunkedPoolMap(self, n_proc, start_method) if n_proc != 1 else None

        try:
            sol = scipy.optimize.differential_evolution(**kwargs, workers=pool or 1)
        finally:
            if pool is not None:
                pool.close()

        self.de_result = sol
        self.x = np.asarray(sol.x, dtype=float)

        B = self.reciprocal_lattice_B()

        for _ in range(n_reassign):
            _, num, hkl, _ = self._index_from_U(
                self.orientation_U(*self.x),
                self.kf_ki_dir,
                self.wavelength,
                angle_tol=angle_tol,
            )

            if num < 3:
                break

            U_new = self._wahba_refine_U(B, self.kf_ki_dir, hkl, peak_weights=peak_weights)

            if U_new is None:
                break

            self.x = self.orientation_parameters_from_U(U_new)

        cost, num, hkl, lamda = self._index_from_U(
            self.orientation_U(*self.x),
            self.kf_ki_dir,
            self.wavelength,
            angle_tol=angle_tol,
        )

        UB = self.UB_matrix(self.orientation_U(*self.x), B)

        return UB, hkl, lamda

    def index(self, kf_ki_dir, wavelength, angle_tol=None, peak_weights=None, ambiguity_power=0.5):
        """
        Index peaks with the current orientation parameters.

        Returns
        -------
        cost, num, hkl, lamda
            See :meth:`_index_from_U`.
        """
        if self.x is None:
            raise RuntimeError("No orientation is set. Run minimize() first.")

        self.kf_ki_dir = np.asarray(kf_ki_dir, dtype=float)
        self.wavelength = tuple(float(x) for x in wavelength)

        self.prepare_indexer(
            self.kf_ki_dir,
            self.wavelength,
            peak_weights=peak_weights,
            ambiguity_power=ambiguity_power,
        )

        return self._index_from_U(
            self.orientation_U(*self.x),
            self.kf_ki_dir,
            self.wavelength,
            angle_tol=angle_tol,
        )

    # -------------------------------------------------------------------------
    # Full weighted protocol
    # -------------------------------------------------------------------------

    @staticmethod
    def make_peak_weights(heights, power=0.5, clip_percentiles=(10, 95), floor=1e-12):
        """
        Robust, mean-normalized optimizer weights from peak heights.

        Heights are clipped to the given percentiles of the positive values,
        scaled by their median and raised to ``power`` (0.5 is sqrt-height).
        """
        heights = np.asarray(heights, dtype=float)
        heights = np.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)
        heights = np.maximum(heights, 0.0)

        positive = heights[heights > 0]

        if positive.size == 0:
            return np.ones_like(heights)

        lo, hi = np.nanpercentile(positive, clip_percentiles)
        clipped = np.clip(heights, lo, hi)

        scale = max(np.nanmedian(clipped[clipped > 0]), floor)

        weights = np.maximum((clipped / scale) ** power, floor)

        return weights / np.mean(weights)

    def select_coarse_subset(
        self,
        kf_ki_dir,
        wavelength,
        weights,
        n_coarse=80,
        ambiguity_power=0.5,
        min_angle_deg=1.0,
    ):
        """
        Choose the peaks for the global search.

        Peaks are ranked by weight / N_candidates**ambiguity_power and taken
        greedily, skipping any within ``min_angle_deg`` of one already
        selected; the list is topped up by rank if that leaves too few.
        """
        self.prepare_indexer(
            kf_ki_dir,
            wavelength,
            peak_weights=weights,
            ambiguity_power=0.0,
            force=True,
            lookup=False,
        )

        counts = np.maximum(self._candidate_counts, 1)
        score = weights / counts**ambiguity_power

        order = np.argsort(score)[::-1]

        dirs = kf_ki_dir / np.linalg.norm(kf_ki_dir, axis=1)[:, None]
        cos_min_angle = np.cos(np.deg2rad(min_angle_deg))

        n_target = min(n_coarse, len(kf_ki_dir))
        selected = []

        for ind in order:
            if len(selected) >= n_target:
                break
            if self._candidate_counts[ind] <= 0:
                continue
            if not selected or np.max(dirs[selected] @ dirs[ind]) < cos_min_angle:
                selected.append(ind)

        chosen = set(selected)

        for ind in order:
            if len(selected) >= n_target:
                break
            if ind not in chosen:
                selected.append(ind)
                chosen.add(ind)

        return np.array(selected, dtype=int)

    def reassign_refine(
        self,
        kf_ki_dir,
        wavelength,
        weights,
        ambiguity_power,
        angle_tol,
        n_cycles=3,
    ):
        """
        Alternate HKL assignment and weighted Wahba refinement of U.
        """
        B = self.reciprocal_lattice_B()

        for _ in range(n_cycles):
            _, num, hkl, _ = self.index(
                kf_ki_dir,
                wavelength,
                angle_tol=angle_tol,
                peak_weights=weights,
                ambiguity_power=ambiguity_power,
            )

            if num < 3:
                break

            U_new = self._wahba_refine_U(B, kf_ki_dir, hkl, peak_weights=weights)

            if U_new is None:
                break

            self.x = self.orientation_parameters_from_U(U_new)

        return self.index(
            kf_ki_dir,
            wavelength,
            angle_tol=angle_tol,
            peak_weights=weights,
            ambiguity_power=ambiguity_power,
        )

    def find_orientation(
        self,
        kf_ki_dir,
        wavelength,
        heights=None,
        n_proc=-1,
        n_coarse=80,
        weight_power=0.5,
        ambiguity_power=0.5,
        angle_tol_deg=0.35,
        n_subset_reassign=2,
        n_full_reassign=3,
        seed=None,
        **de_kwargs,
    ):
        """
        Weighted two-stage orientation search.

        1. Global DE search on a coarse subset of strong, unambiguous and
           angularly distinct peaks.
        2. Weighted reassignment/Wahba cycles on the subset, then on all
           peaks.

        Parameters
        ----------
        kf_ki_dir : ndarray
            Observed kf_hat - ki_hat vectors with shape (N, 3).
        wavelength : tuple
            Wavelength band as (lambda_min, lambda_max).
        heights : ndarray, optional
            Peak heights for weighting; uniform if omitted.
        n_proc : int, optional
            Worker processes for DE.
        n_coarse : int, optional
            Size of the coarse subset.
        weight_power, ambiguity_power : float, optional
            Height compression and candidate-count penalty.
        angle_tol_deg : float, optional
            Angular indexing tolerance in degrees.
        n_subset_reassign, n_full_reassign : int, optional
            Reassignment cycles on the subset and on all peaks.
        seed : int, optional
            DE seed.
        **de_kwargs
            Passed to :meth:`minimize` (e.g. ``popsize``, ``maxiter``).

        Returns
        -------
        UB : ndarray
            UB matrix with shape (3, 3).
        hkl : ndarray
            Indexed HKLs with shape (N, 3).
        lamda : ndarray
            Indexed wavelengths with shape (N,).
        num : int
            Number of indexed peaks.
        """
        kf_ki_dir = np.asarray(kf_ki_dir, dtype=float)

        if heights is None:
            weights = np.ones(len(kf_ki_dir))
        else:
            weights = self.make_peak_weights(heights, power=weight_power)

        angle_tol = np.deg2rad(angle_tol_deg)

        subset = self.select_coarse_subset(
            kf_ki_dir,
            wavelength,
            weights,
            n_coarse=n_coarse,
            ambiguity_power=ambiguity_power,
        )

        self.minimize(
            kf_ki_dir[subset],
            wavelength,
            n_proc=n_proc,
            peak_weights=weights[subset],
            angle_tol=angle_tol,
            ambiguity_power=ambiguity_power,
            n_reassign=0,
            seed=seed,
            **de_kwargs,
        )

        self.reassign_refine(
            kf_ki_dir[subset],
            wavelength,
            weights[subset],
            ambiguity_power,
            angle_tol,
            n_cycles=n_subset_reassign,
        )

        _, num, hkl, lamda = self.reassign_refine(
            kf_ki_dir,
            wavelength,
            weights,
            ambiguity_power,
            angle_tol,
            n_cycles=n_full_reassign,
        )

        self.coarse_subset = subset
        self.peak_weights = weights

        UB = self.UB_matrix(self.orientation_U(*self.x), self.reciprocal_lattice_B())

        return UB, hkl, lamda, num

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
        Constraint function and starting reduced parameters for ``cell``.
        """
        a, b, c, alpha, beta, gamma = self.get_lattice_constants()

        fun_dict = {
            "Cubic": (self.cubic, (a,)),
            "Rhombohedral": (self.rhombohedral, (a, alpha)),
            "Tetragonal": (self.tetragonal, (a, c)),
            "Hexagonal": (self.hexagonal, (a, c)),
            "Orthorhombic": (self.orthorhombic, (a, b, c)),
            "Monoclinic": (self.monoclinic, (a, b, c, beta)),
            "Triclinic": (self.triclinic, (a, b, c, alpha, beta, gamma)),
        }

        if cell not in fun_dict:
            raise ValueError(f"Unknown cell constraint: {cell}")

        return fun_dict[cell]

    def _expand_cell_uncertainty(self, cell, sig):
        """
        Expand reduced-cell uncertainties to (a, b, c, alpha, beta, gamma).
        """
        expand = {
            "Cubic": lambda s: (s[0], s[0], s[0], 0.0, 0.0, 0.0),
            "Rhombohedral": lambda s: (s[0], s[0], s[0], s[1], s[1], s[1]),
            "Tetragonal": lambda s: (s[0], s[0], s[1], 0.0, 0.0, 0.0),
            "Hexagonal": lambda s: (s[0], s[0], s[1], 0.0, 0.0, 0.0),
            "Orthorhombic": lambda s: (s[0], s[1], s[2], 0.0, 0.0, 0.0),
            "Monoclinic": lambda s: (s[0], s[1], s[2], 0.0, s[3], 0.0),
        }

        return expand.get(cell, lambda s: tuple(s[:6]))(sig)

    # -------------------------------------------------------------------------
    # Lattice and UB refinement
    # -------------------------------------------------------------------------

    def residual(self, x, kf_ki_dir, hkl, wavelength, fun):
        """
        Vector residual lambda_i UB hkl_i - d_i for fixed assignments, with
        lambda_i = |d_i| / |B hkl_i| clipped to the band.
        """
        a, b, c, alpha, beta, gamma, *u = fun(x)

        constants = a, b, c, *np.deg2rad([alpha, beta, gamma])
        B, Gstar = self.cartesian_matrix_metric_tensor(*constants)
        UB = self.orientation_U(*u) @ B

        q = np.sqrt(np.einsum("ij,lj,li->l", Gstar, hkl, hkl))

        lamda = np.clip(np.linalg.norm(kf_ki_dir, axis=1) / q, *wavelength)

        return (lamda[:, None] * (hkl @ UB.T) - kf_ki_dir).ravel()

    def refine(self, kf_ki_dir, wavelength, cell="Triclinic", error=0.15):
        """
        Refine orientation and constrained lattice constants.

        Parameters
        ----------
        kf_ki_dir : ndarray
            Observed kf_hat - ki_hat vectors with shape (N, 3).
        wavelength : tuple
            Wavelength band as (lambda_min, lambda_max).
        cell : str, optional
            Lattice system constraint.
        error : float, optional
            Fractional bound around the starting lattice constants.

        Returns
        -------
        UB : ndarray
            Refined UB matrix.
        hkl : ndarray
            Re-indexed HKLs.
        lamda : ndarray
            Re-indexed wavelengths.
        uncertainties : tuple
            Uncertainties of a, b, c, alpha, beta, gamma.
        """
        if self.x is None:
            raise RuntimeError("No orientation is set. Run minimize() first.")

        self.cell = cell

        fun, x0_cell = self._cell_function(cell)

        kf_ki_dir = np.asarray(kf_ki_dir, dtype=float)
        wavelength = tuple(float(x) for x in wavelength)

        _, _, hkl, _ = self.index(kf_ki_dir, wavelength)
        mask = np.any(hkl != 0, axis=1)

        if np.sum(mask) < len(x0_cell) + 3:
            raise RuntimeError("Not enough indexed peaks for refinement.")

        x0 = np.array(tuple(x0_cell) + tuple(self.x), dtype=float)

        x_min = [(1.0 - error) * v for v in x0_cell] + [0.0, 0.0, 0.0]
        x_max = [(1.0 + error) * v for v in x0_cell] + [1.0, 1.0, 1.0]

        sol = scipy.optimize.least_squares(
            self.residual,
            x0=x0,
            args=(kf_ki_dir[mask], hkl[mask], wavelength, fun),
            bounds=(np.array(x_min), np.array(x_max)),
        )

        a, b, c, alpha, beta, gamma, *u = fun(sol.x)

        self.set_lattice_constants(a, b, c, alpha, beta, gamma)
        self.x = np.asarray(u, dtype=float)

        J = sol.jac
        dof = max(1, sol.fun.size - sol.x.size)
        chi2dof = np.sum(sol.fun**2) / dof

        cov = np.linalg.pinv(J.T @ J) * chi2dof
        sig = np.sqrt(np.maximum(np.diag(cov), 0.0))

        uncertainties = self._expand_cell_uncertainty(cell, sig)

        self.prepare_indexer(kf_ki_dir, wavelength, force=True)

        _, _, hkl, lamda = self.index(kf_ki_dir, wavelength)

        UB = self.orientation_U(*self.x) @ self.reciprocal_lattice_B()

        return UB, hkl, lamda, uncertainties

    # -------------------------------------------------------------------------
    # Getters and setters
    # -------------------------------------------------------------------------

    def get_lattice_constants(self):
        return self.a, self.b, self.c, self.alpha, self.beta, self.gamma

    def set_lattice_constants(self, a, b, c, alpha, beta, gamma):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_orientation_parameters(self):
        return self.x

    def set_orientation_parameters(self, x):
        self.x = np.asarray(x, dtype=float)

    def __getstate__(self):
        # The KD-tree is large and cheap to rebuild; workers started by
        # pickling (forkserver/spawn) rebuild it on first use.
        state = self.__dict__.copy()
        state["_tree"] = None
        state["_tree_key"] = None
        state["_lookup_valid"] = False
        return state


def primitive_cell(a, b, c, alpha, beta, gamma, centering):
    """
    Primitive cell of a centred conventional cell.

    Parameters
    ----------
    a, b, c, alpha, beta, gamma : float
        Conventional lattice constants (angles in degrees).
    centering : str
        Key of :data:`CENTERING_MATRICES`.

    Returns
    -------
    constants : tuple
        Primitive a, b, c, alpha, beta, gamma.
    """
    G = CalculateUB(a, b, c, alpha, beta, gamma).metric_G_tensor()
    P = CENTERING_MATRICES[centering]
    Gp = P.T @ G @ P
    L = np.sqrt(np.diag(Gp))

    def angle(i, j):
        return np.degrees(np.arccos(Gp[i, j] / (L[i] * L[j])))

    return (*L, angle(1, 2), angle(0, 2), angle(0, 1))


def conventional_hkl_transform(centering):
    """
    Matrix taking primitive-cell HKLs to conventional-cell HKLs.
    """
    return np.linalg.inv(CENTERING_MATRICES[centering]).T


# Columns are the primitive basis vectors in conventional coordinates; all
# are right-handed (positive determinant) so orientations stay proper
# rotations.
CENTERING_MATRICES = {
    "P": np.eye(3),
    "A": np.array([[2, 0, 0], [0, 1, -1], [0, 1, 1]]) / 2,
    "B": np.array([[1, 0, -1], [0, 2, 0], [1, 0, 1]]) / 2,
    "C": np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]]) / 2,
    "I": np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) / 2,
    "F": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2,
    "R": np.array([[2, -1, -1], [1, 1, -2], [1, 1, 1]]) / 3,  # obverse, hexagonal axes
}

# Mantid ReflectionCondition names for PredictPeaks.
REFLECTION_CONDITIONS = {
    "P": "Primitive",
    "A": "A-face centred",
    "B": "B-face centred",
    "C": "C-face centred",
    "I": "Body centred",
    "F": "All-face centred",
    "R": "Rhombohedrally centred, obverse",
    "R(obv)": "Rhombohedrally centred, obverse",
    "R(rev)": "Rhombohedrally centred, reverse",
}


def centering_filter(centering):
    """
    Allowed-reflection mask function for a conventional-cell centering.

    ``R`` means hexagonal axes, obverse setting; give rhombohedral-axis
    cells as ``P``.

    Returns
    -------
    fun : callable or None
        Function of H with shape (N, 3) returning a Boolean mask, or None
        for a primitive cell.
    """
    conditions = {
        "P": None,
        "A": lambda h, k, l: (k + l) % 2 == 0,
        "B": lambda h, k, l: (h + l) % 2 == 0,
        "C": lambda h, k, l: (h + k) % 2 == 0,
        "I": lambda h, k, l: (h + k + l) % 2 == 0,
        "F": lambda h, k, l: ((h + k) % 2 == 0) & ((k + l) % 2 == 0),
        "R": lambda h, k, l: (-h + k + l) % 3 == 0,
        "R(obv)": lambda h, k, l: (-h + k + l) % 3 == 0,
        "R(rev)": lambda h, k, l: (h - k + l) % 3 == 0,
    }

    cond = conditions[centering]

    if cond is None:
        return None

    return lambda H: cond(H[:, 0], H[:, 1], H[:, 2])


def to_conventional(opt, centering, conventional_cell):
    """
    Conventional-cell optimizer equivalent to a primitive-cell solution.

    Parameters
    ----------
    opt : CalculateUB
        Optimizer with the primitive cell and a solved orientation.
    centering : str
        Key of :data:`CENTERING_MATRICES`.
    conventional_cell : tuple
        Conventional a, b, c, alpha, beta, gamma.

    Returns
    -------
    conv : CalculateUB
        Optimizer with the conventional cell, the same crystal
        orientation and the centering reflection filter.
    M : ndarray
        HKL transform, primitive -> conventional.
    """
    M = conventional_hkl_transform(centering)

    UB_p = opt.orientation_U(*opt.x) @ opt.reciprocal_lattice_B()

    # UB_p h_p = UB_c h_c with h_c = M h_p
    UB_c = UB_p @ np.linalg.inv(M)

    conv = CalculateUB(*conventional_cell)
    U = UB_c @ np.linalg.inv(conv.reciprocal_lattice_B())

    # remove the residual from rounding in the conventional constants
    u, _, vt = np.linalg.svd(U)
    conv.x = conv.orientation_parameters_from_U(u @ vt)
    conv._hkl_filter = centering_filter(centering)

    for attr in ("_score_angle_tol", "_score_index_bonus", "_score_outlier_cap"):
        setattr(conv, attr, getattr(opt, attr))

    return conv, M
