"""
Two-dimensional Gaussian peak integration on detector images and a
k-nearest-neighbour kernel for smoothing peak shapes across a panel.
"""

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.optimize
from scipy.spatial import cKDTree


class ParallelProcessor:
    """
    Apply a function to the items of a dict, optionally in worker processes.
    """

    def __init__(self, n_proc=4):
        self.n_proc = n_proc

        if n_proc != 1:
            try:
                from mantid import config

                config["MultiThreaded.MaxCores"] = "1"
            except ImportError:
                pass

            for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
                os.environ[var] = "1"
            os.environ["TBB_THREAD_ENABLED"] = "0"

    def process_dict(self, data, func):
        self.function = func

        if self.n_proc == 1:
            return {k: func((k, v))[1] for k, v in data.items()}

        results = {}

        with ProcessPoolExecutor(max_workers=None if self.n_proc == -1 else self.n_proc) as executor:
            futures = [executor.submit(self.safe_function_wrapper, kv) for kv in data.items()]
            for future in as_completed(futures):
                try:
                    key, value = future.result()
                    results[key] = value
                except Exception as e:
                    print("Exception in pool: {}".format(e))
                    traceback.print_exc()

        return results

    def safe_function_wrapper(self, *args, **kwargs):
        try:
            return self.function(*args, **kwargs)
        except Exception as e:
            print("Exception in worker function: {}".format(e))
            traceback.print_exc()
            raise


class IntegratePeaks:
    """
    Fit a rotated 2D Gaussian plus constant background around each peak.

    Parameters
    ----------
    counts : ndarray
        Panel image with shape (nx, ny).
    x, y : array_like
        Predicted peak pixel positions.
    """

    def __init__(self, counts, x, y):
        self.counts = counts
        self.x = x
        self.y = y

    def fit(self, roi_pixels=50, sigma_1=None, sigma_2=None, theta=None, n_proc=1):
        """
        Fit every peak in a square region of interest.

        Shape parameters, if given, constrain the fit to within +-20% (and
        +-30 degrees) of the supplied values.

        Returns
        -------
        results : dict
            key -> (I, sigma_I, mu_1, mu_2, sigma_1, sigma_2, theta).
        """
        im = self.counts

        X, Y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]), indexing="ij")

        data = {}

        for i, (x_val, y_val) in enumerate(zip(self.x, self.y)):
            x_min = int(max(x_val - roi_pixels, 0))
            x_max = int(min(x_val + roi_pixels + 1, im.shape[0]))

            y_min = int(max(y_val - roi_pixels, 0))
            y_max = int(min(y_val + roi_pixels + 1, im.shape[1]))

            x = X[x_min:x_max, y_min:y_max].copy()
            y = Y[x_min:x_max, y_min:y_max].copy()
            z = im[x_min:x_max, y_min:y_max].copy()

            j, k = np.unravel_index(np.argwhere(z.ravel() > np.percentile(z, 95)), z.shape)

            if j.size == 0:  # flat (e.g. masked) region: start at the ROI centre
                j, k = np.array([x_val - x_min]), np.array([y_val - y_min])

            x0 = np.array(
                [
                    z.max(),
                    0.25 * (z.min() + z.max()),
                    np.mean(j) + x_min,
                    np.mean(k) + y_min,
                    roi_pixels / 6 if sigma_1 is None else sigma_1[i],
                    roi_pixels / 6 if sigma_2 is None else sigma_2[i],
                    0 if theta is None else theta[i],
                ]
            )

            xmin = np.array(
                [
                    z.min(),
                    z.min(),
                    x_min,
                    y_min,
                    1 if sigma_1 is None else 0.8 * sigma_1[i],
                    1 if sigma_2 is None else 0.8 * sigma_2[i],
                    -np.pi if theta is None else theta[i] - np.pi / 6,
                ]
            )

            xmax = np.array(
                [
                    2 * z.max(),
                    z.max(),
                    x_max,
                    y_max,
                    roi_pixels / 3 if sigma_1 is None else 1.2 * sigma_1[i],
                    roi_pixels / 3 if sigma_2 is None else 1.2 * sigma_2[i],
                    np.pi if theta is None else theta[i] + np.pi / 6,
                ]
            )

            data[i] = (x_val, y_val, x0, xmin, xmax, x, y, z)

        self.roi_pixels = roi_pixels

        return ParallelProcessor(n_proc).process_dict(data, self._fit)

    def _peak(self, x, y, A, B, mu_x, mu_y, sigma_1, sigma_2, theta):
        a = np.cos(theta) ** 2 / sigma_1**2 + np.sin(theta) ** 2 / sigma_2**2
        b = np.sin(theta) ** 2 / sigma_1**2 + np.cos(theta) ** 2 / sigma_2**2
        c = (1 / sigma_1**2 - 1 / sigma_2**2) * np.sin(2 * theta)

        dx = x - mu_x
        dy = y - mu_y

        return A * np.exp(-0.5 * (a * dx**2 + b * dy**2 + c * dx * dy)) + B

    def _intensity(self, A, B, sigma1, sigma2, cov_matrix):
        """
        Integrated Gaussian 2 pi A sigma_1 sigma_2 and its propagated error.

        The constant background B is modelled separately and does not
        enter the peak integral.
        """
        I = 2 * np.pi * A * sigma1 * sigma2

        dI = np.array(
            [
                2 * np.pi * sigma1 * sigma2,
                0.0,
                2 * np.pi * A * sigma2,
                2 * np.pi * A * sigma1,
            ]
        )

        return I, np.sqrt(dI @ cov_matrix @ dI)

    def _residual(self, params, x, y, z, x_val, y_val, lamda=0.01):
        A, B, mu_x, mu_y, *_ = params
        penalty = [lamda * (mu_x - x_val), lamda * (mu_y - y_val)]
        return (self._peak(x, y, *params) - z).ravel().tolist() + penalty

    def _fit(self, key_value):
        key, value = key_value

        x_val, y_val, x0, xmin, xmax, x, y, z = value

        I, sig = 0.0, 0.0
        mu_1, mu_2 = x_val, y_val
        sigma_1, sigma_2, theta = x0[4:]

        if np.all(x0 > xmin) and np.all(x0 < xmax):
            sol = scipy.optimize.least_squares(
                self._residual,
                x0=x0,
                bounds=np.array([xmin, xmax]),
                args=(x, y, z, x_val, y_val),
                loss="linear",
            )

            inv_cov = sol.jac.T @ sol.jac

            A, B, mu_1, mu_2, sigma_1, sigma_2, theta = sol.x

            if np.linalg.det(inv_cov) > 0:
                inds = [0, 1, 4, 5]
                cov = np.linalg.inv(inv_cov)[inds][:, inds]
                I, sig = self._intensity(A, B, sigma_1, sigma_2, cov)

        return key, (I, sig, mu_1, mu_2, sigma_1, sigma_2, theta)


class WeightedKernel:
    """
    Adaptive-bandwidth Gaussian kernel regression over k nearest neighbours.

    Parameters
    ----------
    k : int, optional
        Neighbours per query.
    lengthscale : tuple, optional
        (lx, ly) for anisotropic distances; isotropic if None.
    eps : float, optional
        Numerical floor.
    """

    def __init__(self, k=40, lengthscale=None, eps=1e-12):
        self.k = k
        self.lengthscale = lengthscale
        self.eps = eps

    def fit(self, x, y, values, weights=None):
        """
        values : (N,) or (N, D) samples; weights : (N,) confidences.
        """
        XY = np.column_stack([x, y])
        self.XY_raw = XY
        if self.lengthscale is not None:
            XY = XY / np.array(self.lengthscale, float)
        self.XY = XY
        self.tree = cKDTree(XY)
        self.values = np.asarray(values, float)
        self.weights = np.ones(XY.shape[0]) if weights is None else np.asarray(weights, float)
        return self

    def _local_bandwidth(self, dists):
        # median of nonzero neighbour distances; floor avoids h = 0 for
        # duplicate points
        nz = dists[dists > 0]
        return (np.median(nz) if nz.size else np.max(dists)) + 1e-9

    def predict(self, x, y):
        """
        Kernel-weighted mean at query points; returns shape (D, M).
        """
        XYq = np.column_stack([x, y])
        if self.lengthscale is not None:
            XYq = XYq / np.array(self.lengthscale, float)

        dists, idxs = self.tree.query(XYq, k=min(self.k, len(self.XY)))
        if dists.ndim == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        V = self.values[idxs]
        W = self.weights[idxs]

        hs = np.apply_along_axis(self._local_bandwidth, 1, dists)

        alpha = W * np.exp(-0.5 * (dists / (hs[:, None] + self.eps)) ** 2)

        den = np.sum(alpha, axis=1, keepdims=True) + self.eps
        while alpha.ndim < V.ndim:
            alpha = alpha[..., None]
            den = den[..., None]

        return np.sum(alpha * V, axis=1).T / den.squeeze()
