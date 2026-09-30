import os
import traceback

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial import cKDTree

from mantid import config

class ParallelProcessor:
    def __init__(self, n_proc=4):
        self.n_proc = n_proc

        config["MultiThreaded.MaxCores"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["TBB_THREAD_ENABLED"] = "0"

    def process_dict(self, data, func):
        self.function = func
        if self.n_proc > 1 or self.n_proc == -1:
            with ProcessPoolExecutor(max_workers=self.n_proc) as executor:
                futures = [
                    executor.submit(self.safe_function_wrapper, kv)
                    for kv in data.items()
                ]
                results = {}
                for future in as_completed(futures):
                    try:
                        key, value = future.result()
                        results[key] = value
                    except Exception as e:
                        print("Exception in pool: {}".format(e))
                        traceback.print_exc()
        else:
            results = {k: func((k, v))[1] for k, v in data.items()}
        return results

    def safe_function_wrapper(self, *args, **kwargs):
        try:
            return self.function(*args, **kwargs)
        except Exception as e:
            print("Exception in worker function: {}".format(e))
            traceback.print_exc()
            raise


class WeightedKernel:
    def __init__(self, k=40, lengthscale=None, eps=1e-12):
        """
        k: number of nearest neighbors per query
        lengthscale: None (isotropic) or (lx, ly) for anisotropic distance
        """
        self.k = k
        self.lengthscale = lengthscale
        self.eps = eps

    def fit(self, x, y, values, weights=None):
        """
        values: (N,) or (N,D) array of sample values
        weights: (N,) confidence weights (default 1)
        """
        XY = np.column_stack([x, y])
        self.XY_raw = XY
        if self.lengthscale is not None:
            lx, ly = self.lengthscale
            XY = XY / np.array([lx, ly], float)
        self.XY = XY
        self.tree = cKDTree(XY)
        self.values = np.asarray(values, float)
        if weights is None:
            weights = np.ones(XY.shape[0], float)
        self.weights = np.asarray(weights, float)
        return self

    def _local_bandwidth(self, dists):
        # Smooth, query-specific bandwidth = median of nonzero dists to neighbors
        # add small floor to avoid h=0 when duplicates exist
        nz = dists[dists > 0]
        return (np.median(nz) if nz.size else np.max(dists)) + 1e-9

    def predict(self, x, y):
        """
        XYq: (M,2) query points
        return_std: if True, returns (mean, std) using local weighted variance
        """
        XYq = np.column_stack([x, y])
        if self.lengthscale is not None:
            lx, ly = self.lengthscale
            XYq_scaled = XYq / np.array([lx, ly], float)
        else:
            XYq_scaled = XYq

        # k-NN search
        dists, idxs = self.tree.query(XYq_scaled, k=min(self.k, len(self.XY)))
        # Ensure 2D shapes even when k==1
        if dists.ndim == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        # Values/weights for neighbors
        V = self.values[idxs]           # (M,k,...) supports vector-valued
        W = self.weights[idxs]          # (M,k)

        # Per-query bandwidth (adaptive)
        hs = np.apply_along_axis(self._local_bandwidth, 1, dists)  # (M,)

        # Gaussian kernel
        K = np.exp(-0.5 * (dists / (hs[:, None] + self.eps))**2)    # (M,k)
        alpha = W * K                                               # (M,k)

        # Handle exact hits: if any dist==0, snap to weighted average of exact duplicates
        # exact = (dists <= 1e-15)
        # if exact.any():
            # For rows with exact hits, zero out non-exact neighbors
            # mask = (~exact)
            # alpha = np.where(mask, 0.0, W * 1.0)

        # Normalize and compute mean
        den = np.sum(alpha, axis=1, keepdims=True) + self.eps
        # Broadcast alpha to V's trailing dims
        while alpha.ndim < V.ndim:
            alpha = alpha[..., None]
            den = den[..., None]
        mean = np.sum(alpha * V, axis=1).T / den.squeeze()
        return mean
