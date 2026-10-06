"""
Mantid workflow: peak finding, UB determination, refinement, prediction
and integration on a one-bin detector workspace.
"""

import numpy as np

from mantid.kernel import config
from mantid.simpleapi import (
    AddPeak,
    CloneWorkspace,
    CopySample,
    CreatePeaksWorkspace,
    DeleteTableRows,
    DeleteWorkspace,
    PredictPeaks,
    PreprocessDetectorsToMD,
    SetUB,
    mtd,
)

from .integrate import IntegratePeaks, WeightedKernel
from .optimize import (
    REFLECTION_CONDITIONS,
    CalculateUB,
    primitive_cell,
    to_conventional,
)
from .peaks import (
    find_peaks_global,
    find_peaks_local,
    find_peaks_matched,
    interpolate_positions,
    scattering_directions,
)

# Pixels per panel by instrument; IMAGINE is resolved from the spectrum count.
PIXEL_SHAPES = {
    "CG4D": (5000, 1800),
    "MANDI": (256, 256),
    ("IMAGINE", 80 * 128 * 128): (128, 128),
    ("IMAGINE", 80 * 512 * 512): (512, 512),
}


class QuasiLaue:
    """
    Quasi-Laue reduction of a one-bin detector workspace.

    Parameters
    ----------
    ws : str
        Workspace name, e.g. from :func:`loading.load_lite`.
    pixel_shape : tuple, optional
        Pixels per panel (nx, ny); inferred from the instrument if omitted.
    """

    def __init__(self, ws, pixel_shape=None):
        # Peak vectors are kf - ki; Mantid's default (Inelastic) is ki - kf,
        # which would make every HKL the Friedel mate of ours.
        config["Q.convention"] = "Crystallography"

        self.ws = ws
        self.peaks_ws = ws + "_peaks"

        workspace = mtd[ws]

        info = workspace.componentInfo()
        self.instrument = info.name(info.root())

        n_spec = workspace.getNumberHistograms()

        if pixel_shape is None:
            pixel_shape = PIXEL_SHAPES.get(self.instrument, PIXEL_SHAPES.get((self.instrument, n_spec)))

        if pixel_shape is None:
            raise ValueError(f"Unknown pixel layout for {self.instrument}; pass pixel_shape.")

        self.pixel_shape = tuple(pixel_shape)

        det = ws + "_detectors"

        if not mtd.doesExist(det):
            PreprocessDetectorsToMD(InputWorkspace=ws, OutputWorkspace=det)

        table = mtd[det]

        L2 = np.array(table.column("L2"))
        tt = np.array(table.column("TwoTheta"))
        az = np.array(table.column("Azimuthal"))

        xyz = L2[:, None] * np.column_stack([np.sin(tt) * np.cos(az), np.sin(tt) * np.sin(az), np.cos(tt)])

        self.detector_ids = np.array(table.column("DetectorID")).reshape(-1, *self.pixel_shape)
        self.positions = xyz.reshape(*self.detector_ids.shape, 3)

        flat = self.detector_ids.ravel()
        self._id_order = np.argsort(flat)
        self._sorted_ids = flat[self._id_order]

        self.R = workspace.run().getGoniometer().getR()

        self.opt = None

    # -------------------------------------------------------------------------
    # Data access
    # -------------------------------------------------------------------------

    def extract_images(self):
        return mtd[self.ws].extractY().reshape(-1, *self.pixel_shape)

    def pixels_of_detectors(self, det_ids):
        """
        (bank, i, j) of detector IDs.
        """
        pos = np.searchsorted(self._sorted_ids, np.asarray(det_ids))
        return np.unravel_index(self._id_order[pos], self.detector_ids.shape)

    def peak_indices(self):
        """
        (bank, i, j) of the peaks in the peaks workspace.
        """
        ids = [peak.getDetectorID() for peak in mtd[self.peaks_ws]]
        return self.pixels_of_detectors(ids)

    # -------------------------------------------------------------------------
    # Peak finding
    # -------------------------------------------------------------------------

    def find_peaks(self, method="local", **kwargs):
        """
        Find peaks and their sample-frame scattering directions.

        Parameters
        ----------
        method : str, optional
            ``"local"`` (per-bank background-normalised threshold with
            sub-pixel centroids), ``"matched"`` (Gaussian matched filter
            with destriping, more sensitive to faint high-angle spots) or
            ``"global"`` (single absolute threshold, the original finder).
        **kwargs
            Passed to :func:`peaks.find_peaks_local`,
            :func:`peaks.find_peaks_matched` or
            :func:`peaks.find_peaks_global`.
        """
        images = self.extract_images()

        if method == "local":
            coords, heights, _ = find_peaks_local(images, **kwargs)
        elif method == "matched":
            coords, heights, _ = find_peaks_matched(images, **kwargs)
        elif method == "global":
            coords, heights = find_peaks_global(images, **kwargs)
        else:
            raise ValueError(f"Unknown peak finding method: {method}")

        xyz = interpolate_positions(self.positions, coords)

        # lab -> sample frame (Mantid convention Q_sample = R^T Q_lab)
        self.kf_ki_dir = scattering_directions(xyz) @ self.R
        self.heights = heights
        self.coords = coords
        self.found_coords = coords.copy()
        self.indexed_mask = None

        if mtd.doesExist(self.peaks_ws):
            DeleteWorkspace(self.peaks_ws)

        CreatePeaksWorkspace(
            InstrumentWorkspace=self.ws,
            NumberOfPeaks=0,
            OutputType="Peak",
            OutputWorkspace=self.peaks_ws,
        )

        pix = np.rint(coords).astype(int)
        pix[:, 1] = np.clip(pix[:, 1], 0, self.pixel_shape[0] - 1)
        pix[:, 2] = np.clip(pix[:, 2], 0, self.pixel_shape[1] - 1)

        for (bank, i, j), height in zip(pix, heights):
            AddPeak(
                PeaksWorkspace=self.peaks_ws,
                RunWorkspace=self.ws,
                TOF=1,
                DetectorID=int(self.detector_ids[bank, i, j]),
                Height=float(height),
                BinCount=0,
            )

        print("Found {} peaks".format(len(heights)))

    # -------------------------------------------------------------------------
    # UB
    # -------------------------------------------------------------------------

    def _apply_indexing(self, UB, hkl, lamda):
        """
        Set UB, wavelengths and HKLs on the peaks and drop unindexed peaks.
        """
        SetUB(Workspace=self.peaks_ws, UB=UB.ravel().tolist())
        SetUB(Workspace=self.ws, UB=UB.ravel().tolist())

        peaks = mtd[self.peaks_ws]

        for row, (lam, h) in enumerate(zip(lamda, hkl)):
            peak = peaks.getPeak(row)
            if np.isfinite(lam):
                peak.setWavelength(float(lam))
            peak.setHKL(*[float(v) for v in h])

        drop = np.flatnonzero(~np.any(np.asarray(hkl) != 0, axis=1))

        if len(drop):
            DeleteTableRows(TableWorkspace=self.peaks_ws, Rows=drop.tolist())

        keep = np.any(np.asarray(hkl) != 0, axis=1)

        self.kf_ki_dir = self.kf_ki_dir[keep]
        self.heights = self.heights[keep]
        self.coords = self.coords[keep]

    def find_UB(
        self,
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        wavelength,
        centering="P",
        n_proc=-1,
        seed=None,
        **kwargs,
    ):
        """
        Determine the orientation and index the peaks.

        The global search runs in the primitive cell of the (conventional)
        lattice given; the solution is then expressed in the conventional
        setting, with HKLs obeying the centering condition.

        Parameters
        ----------
        a, b, c, alpha, beta, gamma : float
            Conventional lattice constants (degrees).
        wavelength : tuple
            Wavelength band (lambda_min, lambda_max).
        centering : str, optional
            Lattice centering of the conventional cell.
        n_proc : int, optional
            Worker processes for differential evolution.
        seed : int, optional
            DE seed.
        **kwargs
            Passed to :meth:`CalculateUB.find_orientation`.
        """
        conventional = (a, b, c, alpha, beta, gamma)

        prim = CalculateUB(*primitive_cell(*conventional, centering))

        _, _, _, num = prim.find_orientation(
            self.kf_ki_dir,
            wavelength,
            heights=self.heights,
            n_proc=n_proc,
            seed=seed,
            **kwargs,
        )

        opt, _ = to_conventional(prim, centering, conventional)

        _, num, hkl, lamda = opt.index(self.kf_ki_dir, wavelength)

        sig = prim.significance
        self.significance = sig
        print(
            "Indexed {} of {} peaks; search peaks {} of {} indexed (random orientations: {:.1f} expected; "
            "log10 p = {:.1f}; {} restart(s))".format(
                num,
                len(self.kf_ki_dir),
                sig["indexed"],
                sig["n_informative"],
                sig["expected_chance"],
                sig["log10_p_value"],
                len(prim.restarts),
            )
        )

        UB = opt.orientation_U(*opt.x) @ opt.reciprocal_lattice_B()

        if len(self.found_coords) == len(hkl):
            self.indexed_mask = np.any(np.asarray(hkl) != 0, axis=1)

        self._apply_indexing(UB, hkl, lamda)

        self.opt = opt
        self.primitive_opt = prim
        self.centering = centering

        return num

    def refine_UB(self, cell, wavelength):
        """
        Constrained refinement of lattice constants and orientation.

        Parameters
        ----------
        cell : str
            Lattice system, e.g. ``"Hexagonal"``.
        wavelength : tuple
            Wavelength band.

        Returns
        -------
        uncertainties : tuple
            Uncertainties of a, b, c, alpha, beta, gamma.
        """
        UB, hkl, lamda, uncertainties = self.opt.refine(self.kf_ki_dir, wavelength, cell)

        self._apply_indexing(UB, hkl, lamda)

        mtd[self.peaks_ws].sample().getOrientedLattice().setError(*uncertainties)

        return uncertainties

    def predict_peaks(self, wavelength, centering=None, d_min=4.0):
        """
        Replace the peaks with all predicted reflections in the band.
        """
        centering = self.centering if centering is None else centering

        PredictPeaks(
            InputWorkspace=self.peaks_ws,
            OutputWorkspace=self.peaks_ws,
            MinDSpacing=d_min,
            MaxDSpacing="inf",
            WavelengthMin=wavelength[0],
            WavelengthMax=wavelength[1],
            ReflectionCondition=REFLECTION_CONDITIONS[centering],
        )

    def predicted_pixels(self, wavelength, centering=None, d_min=2.0):
        """
        (bank, i, j) of reflections predicted by the current UB, without
        replacing the peaks workspace.
        """
        centering = self.centering if centering is None else centering

        PredictPeaks(
            InputWorkspace=self.peaks_ws,
            OutputWorkspace=self.ws + "_predicted",
            MinDSpacing=d_min,
            MaxDSpacing="inf",
            WavelengthMin=wavelength[0],
            WavelengthMax=wavelength[1],
            ReflectionCondition=REFLECTION_CONDITIONS[centering],
        )

        ids = [p.getDetectorID() for p in mtd[self.ws + "_predicted"]]
        ids = [i for i in ids if i >= 0]

        return np.column_stack(self.pixels_of_detectors(ids)) if ids else np.zeros((0, 3))

    def plot_peaks_pdf(self, filename, predicted=None, n_sigma=8.0, title=None):
        """
        One PDF page per panel with the found peaks (green if indexed, red
        otherwise), optional predicted reflections and the significance map.

        Parameters
        ----------
        filename : str
            Output PDF.
        predicted : ndarray, optional
            (bank, i, j), e.g. from :meth:`predicted_pixels`.
        n_sigma : float, optional
            Threshold contour on the significance map.
        title : str, optional
            Page title prefix; defaults to the workspace name.

        Returns
        -------
        n_pages : int
            Number of pages written.
        """
        from .plots import plot_peak_panels

        return plot_peak_panels(
            self.extract_images(),
            self.found_coords,
            filename,
            indexed=self.indexed_mask,
            predicted=predicted,
            n_sigma=n_sigma,
            title=self.ws if title is None else title,
        )

    # -------------------------------------------------------------------------
    # Integration
    # -------------------------------------------------------------------------

    def integrate_peaks(self, roi_pixels=10, strong_isigi=5.0, n_proc=1, plot=False):
        """
        Integrate the (predicted) peaks by 2D Gaussian fitting per panel.

        A first pass fits free Gaussians in a ``roi_pixels`` box; the shapes
        of strong peaks (I/sigma > ``strong_isigi``) are then smoothed across
        the panel with :class:`WeightedKernel`, and a second pass in a box
        twice as large fits every peak with its shape constrained to the
        smoothed one.

        Returns
        -------
        figures : list
            Matplotlib figures per panel if ``plot`` is True.
        """
        images = self.extract_images()

        rows, hkls, lamdas, det_ids = [], [], [], []

        for row, peak in enumerate(mtd[self.peaks_ws]):
            det_id = peak.getDetectorID()
            if det_id >= 0:
                hkls.append(list(peak.getHKL()))
                lamdas.append(peak.getWavelength())
                det_ids.append(det_id)
            else:
                rows.append(row)

        if rows:
            DeleteTableRows(TableWorkspace=self.peaks_ws, Rows=rows)

        tmp = self.ws + "_tmp"

        CloneWorkspace(InputWorkspace=self.peaks_ws, OutputWorkspace=tmp)

        CreatePeaksWorkspace(
            InstrumentWorkspace=self.ws,
            NumberOfPeaks=0,
            OutputType="Peak",
            OutputWorkspace=self.peaks_ws,
        )

        CopySample(
            InputWorkspace=tmp,
            OutputWorkspace=self.peaks_ws,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

        DeleteWorkspace(tmp)

        bank, pi, pj = self.pixels_of_detectors(det_ids)

        hkls = np.array(hkls)
        lamdas = np.array(lamdas)

        figures = []

        for b in np.unique(bank):
            mask = bank == b
            x, y = pi[mask], pj[mask]
            image = images[b]

            intgr = IntegratePeaks(image, x, y)

            first = intgr.fit(roi_pixels=roi_pixels, n_proc=n_proc)

            shapes, weights = [], []

            for key in range(len(x)):
                I, sig, _, _, s1, s2, th = first[key]
                sx2 = s1**2 * np.cos(th) ** 2 + s2**2 * np.sin(th) ** 2
                sy2 = s1**2 * np.sin(th) ** 2 + s2**2 * np.cos(th) ** 2
                sxy = (s1**2 - s2**2) * np.sin(th) * np.cos(th)
                shapes.append((np.sqrt(sx2), np.sqrt(sy2), sxy))
                weights.append(I / sig if sig > 0 else 0.0)

            shapes = np.array(shapes)
            weights = np.array(weights)
            strong = weights > strong_isigi

            if strong.sum() >= 1:
                wk = WeightedKernel(k=5).fit(x[strong], y[strong], shapes[strong], weights[strong])
                sx, sy, sxy = wk.predict(x, y)
            else:
                sx, sy, sxy = shapes.T

            A = np.zeros((len(x), 2, 2))
            A[:, 0, 0] = sx**2
            A[:, 1, 1] = sy**2
            A[:, 0, 1] = A[:, 1, 0] = sxy

            vals, vecs = np.linalg.eigh(A)
            vals = np.clip(vals, 1e-12, None)
            Ap = vecs @ (vals[..., None] * np.swapaxes(vecs, -1, -2))

            sx = np.sqrt(Ap[:, 0, 0])
            sy = np.sqrt(Ap[:, 1, 1])
            sxy = Ap[:, 0, 1]

            theta = (0.5 * np.arctan2(2 * sxy, sx**2 - sy**2) + np.pi / 2) % np.pi - np.pi / 2

            half = np.hypot(0.5 * (sx**2 - sy**2), sxy)
            sigma_1 = np.sqrt(0.5 * (sx**2 + sy**2) + half)
            sigma_2 = np.sqrt(np.maximum(0.5 * (sx**2 + sy**2) - half, 1e-12))

            second = intgr.fit(2 * roi_pixels, sigma_1, sigma_2, theta, n_proc=n_proc)

            for key in range(len(x)):
                I, sig, mu_1, mu_2, *_ = second[key]

                row = int(np.clip(round(mu_1), 0, image.shape[0] - 1))
                col = int(np.clip(round(mu_2), 0, image.shape[1] - 1))

                AddPeak(
                    PeaksWorkspace=self.peaks_ws,
                    RunWorkspace=self.ws,
                    TOF=1,
                    DetectorID=int(self.detector_ids[b, row, col]),
                    Height=0,
                    BinCount=0,
                )

                peak = mtd[self.peaks_ws].getPeak(mtd[self.peaks_ws].getNumberPeaks() - 1)

                peak.setWavelength(float(lamdas[mask][key]))
                peak.setHKL(*[float(v) for v in hkls[mask][key]])
                peak.setIntensity(float(I))
                peak.setSigmaIntensity(float(sig))

            if plot:
                figures.append(self._plot_integration(image, x, y, second, 2 * roi_pixels, b))

        return figures

    def _plot_integration(self, image, x, y, results, roi_pixels, bank):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Rectangle

        fig, ax = plt.subplots(1, 1)
        ax.imshow(image.T, cmap="binary", origin="lower", vmin=0, vmax=np.percentile(image, 99))
        ax.minorticks_on()
        ax.set_title(f"bank {bank}")

        for key, (I, sig, mu_1, mu_2, s1, s2, th) in results.items():
            ax.add_patch(
                Ellipse(
                    xy=(mu_1, mu_2),
                    width=6 * s1,
                    height=6 * s2,
                    angle=np.rad2deg(th),
                    edgecolor="r",
                    facecolor="none",
                    zorder=100,
                )
            )
            ax.add_patch(
                Rectangle(
                    xy=(x[key] - roi_pixels, y[key] - roi_pixels),
                    width=2 * roi_pixels,
                    height=2 * roi_pixels,
                    edgecolor="w",
                    facecolor="none",
                )
            )

        return fig

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_detector(self, ax=None):
        """
        Detector counts in (gamma, nu) with the peaks overlaid.
        """
        import matplotlib.pyplot as plt

        data = mtd[self.ws].extractY().ravel()
        xyz = self.positions.reshape(-1, 3)
        kf = xyz / np.linalg.norm(xyz, axis=1)[:, None]

        nu = np.rad2deg(np.arcsin(kf[:, 1]))
        gamma = np.rad2deg(np.arctan2(kf[:, 0], kf[:, 2]))

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 6), layout="constrained")

        ax.scatter(gamma, nu, c=data, vmin=0, vmax=np.percentile(data, 99.9), cmap="binary", s=1, rasterized=True)

        peaks = mtd[self.peaks_ws]
        tt = np.array([p.getScattering() for p in peaks])
        az = np.array([p.getAzimuthal() for p in peaks])

        pk = np.column_stack([np.sin(tt) * np.cos(az), np.sin(tt) * np.sin(az), np.cos(tt)])

        ax.scatter(
            np.rad2deg(np.arctan2(pk[:, 0], pk[:, 2])),
            np.rad2deg(np.arcsin(pk[:, 1])),
            color="r",
            marker="o",
            facecolor="none",
            linewidths=0.2,
        )

        ax.set_aspect(1)
        ax.minorticks_on()
        ax.set_xlabel(r"$\gamma$ [deg.]")
        ax.set_ylabel(r"$\nu$ [deg.]")

        return ax
