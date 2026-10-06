"""
Peak finding on detector images and conversion to scattering directions.

Images are stacked per bank with shape (n_banks, nx, ny), in the same
order as the workspace spectra, so the flat index of a pixel is its
spectrum index.
"""

import warnings

import numpy as np
import scipy.ndimage
import skimage.feature


def find_peaks_global(images, max_peaks=50, min_pix=10, perc=99.9):
    """
    Local maxima above one absolute threshold shared by all banks.

    The threshold is the 95th percentile, over banks, of each bank's
    ``perc`` percentile. This is the original finder; it misses peaks on
    banks whose background differs strongly from the others.

    Parameters
    ----------
    images : ndarray
        Detector counts with shape (n_banks, nx, ny).
    max_peaks : int, optional
        Maximum peaks per bank.
    min_pix : int, optional
        Minimum separation and border exclusion in pixels.
    perc : float, optional
        Per-bank percentile used to set the threshold.

    Returns
    -------
    coords : ndarray
        (bank, i, j) of each peak with shape (N, 3), integer pixels.
    heights : ndarray
        Pixel counts at each peak with shape (N,).
    """
    vals = [np.nanpercentile(image, perc) for image in images]
    threshold = np.nanpercentile(vals, 95)

    coords = []

    for bank, image in enumerate(images):
        for i, j in skimage.feature.peak_local_max(
            image,
            num_peaks=max_peaks,
            min_distance=min_pix,
            threshold_abs=threshold,
            exclude_border=min_pix,
        ):
            coords.append((bank, i, j))

    coords = np.array(coords, dtype=int).reshape(-1, 3)
    heights = images[coords[:, 0], coords[:, 1], coords[:, 2]].astype(float)

    return coords, heights


def background_residual(image, background_size=15, border=8, mask=None):
    """
    Background-subtracted image and its robust noise level.

    Masked pixels (by default those with exactly zero counts, as written for
    masked detector edges) are filled with their nearest unmasked value
    before the median filter, so the background next to a masked region is
    not dragged towards zero; their residual is set to zero.

    Returns
    -------
    residual : ndarray
        Image minus median-filter background.
    sigma : float
        Robust (MAD) noise of the residual over unmasked pixels inside the
        border.
    mask : ndarray
        Boolean mask of masked pixels.
    """
    image = np.asarray(image, dtype=float)

    if mask is None:
        mask = image == 0

    filled = image
    if np.any(mask) and not np.all(mask):
        _, ind = scipy.ndimage.distance_transform_edt(mask, return_indices=True)
        filled = image[tuple(ind)]

    residual = image - scipy.ndimage.median_filter(filled, size=background_size)
    residual[mask] = 0.0

    inner = residual[border:-border, border:-border][~mask[border:-border, border:-border]]
    if inner.size == 0:
        return residual, np.inf, mask

    sigma = 1.4826 * np.median(np.abs(inner - np.median(inner))) + 1e-9

    return residual, sigma, mask


def find_peaks_local(
    images,
    n_sigma=8.0,
    border=8,
    min_pix=4,
    box=3,
    background_size=15,
    max_peaks=200,
):
    """
    Peaks above a per-bank, background-normalised threshold, with centroids.

    Each bank's background is a median filter; the noise level is the
    robust (MAD) sigma of the background-subtracted image. Candidates are
    local maxima of the 3x3-smoothed residual; a peak is kept if its
    residual exceeds ``n_sigma`` sigma, and its position is the
    residual-weighted centroid in a (2 box + 1)^2 window.

    Parameters
    ----------
    images : ndarray
        Detector counts with shape (n_banks, nx, ny).
    n_sigma : float, optional
        Detection threshold in units of the robust noise level.
    border : int, optional
        Pixels excluded at each panel edge.
    min_pix : int, optional
        Minimum separation of peaks in pixels.
    box : int, optional
        Half-width of the centroid window.
    background_size : int, optional
        Median-filter size for the background.
    max_peaks : int, optional
        Maximum peaks per bank.

    Returns
    -------
    coords : ndarray
        (bank, i, j) centroids with shape (N, 3); i, j are fractional.
    heights : ndarray
        Background-subtracted counts in the centroid window, shape (N,).
    snr : ndarray
        Peak residual over robust sigma, shape (N,).
    """
    coords, heights, snr = [], [], []

    for bank, image in enumerate(images):
        image = np.asarray(image, dtype=float)

        if not np.any(image > 0):
            continue

        residual, sigma, mask = background_residual(image, background_size, border)

        smooth = scipy.ndimage.uniform_filter(residual, 3)

        # no peaks within `box` pixels of masked pixels (centroid window)
        near_mask = scipy.ndimage.binary_dilation(mask, iterations=box + 1) if np.any(mask) else mask
        smooth[near_mask] = 0.0

        for i, j in skimage.feature.peak_local_max(
            smooth,
            num_peaks=max_peaks,
            min_distance=min_pix,
            threshold_abs=n_sigma * sigma / 3,
            exclude_border=border,
        ):
            if residual[i, j] < n_sigma * sigma:
                continue

            window = (slice(i - box, i + box + 1), slice(j - box, j + box + 1))
            w = np.clip(residual[window], 0.0, None)

            if w.sum() <= 0:
                continue

            ii, jj = np.mgrid[window]

            coords.append((bank, (w * ii).sum() / w.sum(), (w * jj).sum() / w.sum()))
            heights.append(w.sum())
            snr.append(residual[i, j] / sigma)

    return (
        np.array(coords, dtype=float).reshape(-1, 3),
        np.array(heights),
        np.array(snr),
    )


def destripe(residual, mask):
    """
    Remove line artefacts by subtracting the median of every row and then
    of every column of the residual (masked pixels ignored).
    """
    r = np.where(mask, np.nan, residual)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        residual = residual - np.nan_to_num(np.nanmedian(r, axis=1))[:, None]
        r = np.where(mask, np.nan, residual)
        residual = residual - np.nan_to_num(np.nanmedian(r, axis=0))[None, :]

    residual[mask] = 0.0

    return residual


def find_peaks_matched(
    images,
    z_min=5.0,
    spot_sigma=1.0,
    destripe_lines=True,
    edge=1,
    border=3,
    min_pix=3,
    box=2,
    background_size=15,
    max_peaks=400,
):
    """
    Peaks detected with a matched (Gaussian) filter on the
    background-subtracted image.

    The residual is convolved with a normalised Gaussian of the spot size
    and divided by its noise, sigma sqrt(sum w^2), so the threshold applies
    to the summed signal of a spot rather than to its brightest pixel. This
    finds faint spots spread over several pixels, which dominate at high
    scattering angles. Optional destriping removes line artefacts that a
    median-filter background leaves behind.

    Parameters
    ----------
    images : ndarray
        Detector counts with shape (n_banks, nx, ny).
    z_min : float, optional
        Detection threshold on the filtered significance.
    spot_sigma : float, optional
        Gaussian width of the matched filter in pixels.
    destripe_lines : bool, optional
        Subtract row and column medians from the residual.
    edge : int, optional
        Pixels excluded next to masked pixels.
    border : int, optional
        Border excluded from the noise estimate.
    min_pix : int, optional
        Minimum separation of peaks in pixels.
    box : int, optional
        Half-width of the centroid window.
    background_size : int, optional
        Median-filter size for the background.
    max_peaks : int, optional
        Maximum peaks per bank.

    Returns
    -------
    coords, heights, snr
        As :func:`find_peaks_local`; ``snr`` is the filtered significance.
    """
    kernel = np.zeros((8 * int(np.ceil(spot_sigma)) + 1,) * 2)
    kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = 1.0
    kernel = scipy.ndimage.gaussian_filter(kernel, spot_sigma)
    noise_gain = np.sqrt(np.sum(kernel**2))

    coords, heights, snr = [], [], []

    for bank, image in enumerate(images):
        image = np.asarray(image, dtype=float)

        if not np.any(image > 0):
            continue

        residual, sigma, mask = background_residual(image, background_size, border)

        if destripe_lines:
            residual = destripe(residual, mask)
            inner = residual[border:-border, border:-border][~mask[border:-border, border:-border]]
            if inner.size == 0:
                continue
            sigma = 1.4826 * np.median(np.abs(inner - np.median(inner))) + 1e-9

        z = scipy.ndimage.gaussian_filter(residual, spot_sigma) / (sigma * noise_gain)

        if np.any(mask) and edge:
            z[scipy.ndimage.binary_dilation(mask, iterations=edge)] = 0.0

        for i, j in skimage.feature.peak_local_max(
            z,
            num_peaks=max_peaks,
            min_distance=min_pix,
            threshold_abs=z_min,
            exclude_border=edge + 1,
        ):
            window = (
                slice(max(i - box, 0), i + box + 1),
                slice(max(j - box, 0), j + box + 1),
            )
            w = np.clip(residual[window], 0.0, None)

            if w.sum() <= 0:
                continue

            ii, jj = np.mgrid[window]

            coords.append((bank, (w * ii).sum() / w.sum(), (w * jj).sum() / w.sum()))
            heights.append(w.sum())
            snr.append(z[i, j])

    return (
        np.array(coords, dtype=float).reshape(-1, 3),
        np.array(heights),
        np.array(snr),
    )


def interpolate_positions(positions, coords):
    """
    Detector positions at (possibly fractional) pixel coordinates.

    Parameters
    ----------
    positions : ndarray
        Pixel-centre positions with shape (n_banks, nx, ny, 3).
    coords : ndarray
        (bank, i, j) with shape (N, 3).

    Returns
    -------
    xyz : ndarray
        Bilinearly interpolated positions with shape (N, 3).
    """
    coords = np.asarray(coords, dtype=float)

    bank = coords[:, 0].astype(int)
    _, nx, ny, _ = positions.shape

    i = np.clip(coords[:, 1], 0, nx - 1)
    j = np.clip(coords[:, 2], 0, ny - 1)

    i0 = np.minimum(np.floor(i).astype(int), nx - 2)
    j0 = np.minimum(np.floor(j).astype(int), ny - 2)

    fi = (i - i0)[:, None]
    fj = (j - j0)[:, None]

    return (
        (1 - fi) * (1 - fj) * positions[bank, i0, j0]
        + fi * (1 - fj) * positions[bank, i0 + 1, j0]
        + (1 - fi) * fj * positions[bank, i0, j0 + 1]
        + fi * fj * positions[bank, i0 + 1, j0 + 1]
    )


def scattering_directions(xyz, sample=(0.0, 0.0, 0.0), beam=(0.0, 0.0, 1.0)):
    """
    kf_hat - ki_hat for detector positions.

    Parameters
    ----------
    xyz : ndarray
        Detector positions with shape (N, 3).
    sample : array_like, optional
        Sample position.
    beam : array_like, optional
        Incident beam direction.

    Returns
    -------
    kf_ki_dir : ndarray
        Shape (N, 3); the norm is 2 sin(theta).
    """
    kf = np.asarray(xyz, dtype=float) - np.asarray(sample, dtype=float)
    kf /= np.linalg.norm(kf, axis=1)[:, None]

    ki = np.asarray(beam, dtype=float)
    ki = ki / np.linalg.norm(ki)

    return kf - ki
