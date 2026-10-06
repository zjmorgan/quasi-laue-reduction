"""
Inspection plots for peak finding and indexing.
"""

import numpy as np


def significance_map(image, background_size=15, border=8):
    """
    Background-subtracted image in units of its robust noise level.

    This is the quantity :func:`peaks.find_peaks_local` thresholds.
    """
    from .peaks import background_residual

    residual, sigma, _ = background_residual(image, background_size, border)
    return residual / sigma


def plot_peak_panels(
    images,
    coords,
    filename,
    indexed=None,
    predicted=None,
    banks=None,
    n_sigma=8.0,
    title="",
    background_size=15,
    border=8,
):
    """
    One PDF page per panel: counts with peaks, and the significance map.

    Parameters
    ----------
    images : ndarray
        Detector counts with shape (n_banks, nx, ny).
    coords : ndarray
        (bank, i, j) of found peaks with shape (N, 3); i, j may be
        fractional.
    filename : str
        Output PDF.
    indexed : ndarray, optional
        Boolean mask (N,) of indexed peaks; indexed peaks are drawn green,
        unindexed red. Without it all peaks are red.
    predicted : ndarray, optional
        (bank, i, j) of predicted reflections, drawn as blue crosses.
    banks : sequence, optional
        Panels to plot; default every panel with counts.
    n_sigma : float, optional
        Detection threshold drawn as a contour on the significance map.
    title : str, optional
        Prefix for page titles.
    background_size, border : int, optional
        As in :func:`peaks.find_peaks_local`.

    Returns
    -------
    n_pages : int
        Number of pages written.
    """
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Rectangle

    coords = np.asarray(coords, dtype=float).reshape(-1, 3)

    if indexed is None:
        indexed = np.zeros(len(coords), dtype=bool)
    indexed = np.asarray(indexed, dtype=bool)

    if predicted is not None:
        predicted = np.asarray(predicted, dtype=float).reshape(-1, 3)

    if banks is None:
        banks = [b for b in range(len(images)) if np.any(images[b] > 0)]

    pages = 0

    with PdfPages(filename) as pdf:
        for b in banks:
            image = np.asarray(images[b], dtype=float)
            sig = significance_map(image, background_size, border)

            on = coords[:, 0] == b
            found, ok = coords[on], indexed[on]

            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.6), layout="constrained")

            positive = image[image > 0]
            vmin, vmax = (np.percentile(positive, [5, 99.8]) if positive.size else (0, 1))

            ax0.imshow(
                image.T,
                origin="lower",
                cmap="binary",
                norm=PowerNorm(0.5, vmin=vmin, vmax=vmax),
                interpolation="nearest",
            )
            ax0.set_title("counts (square-root scale)")

            im1 = ax1.imshow(sig.T, origin="lower", cmap="viridis", vmin=-3, vmax=3 * n_sigma, interpolation="nearest")
            ax1.contour(sig.T, levels=[n_sigma], colors="w", linewidths=0.5)
            ax1.set_title(f"(counts - background) / sigma; contour at {n_sigma:g}")
            fig.colorbar(im1, ax=ax1, shrink=0.8)

            for ax in (ax0, ax1):
                ax.add_patch(
                    Rectangle(
                        (border - 0.5, border - 0.5),
                        image.shape[0] - 2 * border,
                        image.shape[1] - 2 * border,
                        edgecolor="orange",
                        facecolor="none",
                        linestyle="--",
                        linewidth=0.6,
                    )
                )

                if predicted is not None:
                    p = predicted[predicted[:, 0] == b]
                    ax.scatter(p[:, 1], p[:, 2], marker="+", s=25, color="tab:blue", linewidths=0.7)

                ax.scatter(found[~ok, 1], found[~ok, 2], s=90, facecolor="none", edgecolor="red", linewidths=0.8)
                ax.scatter(found[ok, 1], found[ok, 2], s=90, facecolor="none", edgecolor="lime", linewidths=0.9)

                ax.set_xlim(-0.5, image.shape[0] - 0.5)
                ax.set_ylim(-0.5, image.shape[1] - 0.5)
                ax.set_xlabel("i [pixel]")
                ax.set_ylabel("j [pixel]")

            n_pred = 0 if predicted is None else int(np.sum(predicted[:, 0] == b))
            fig.suptitle(
                f"{title} bank {b}: {len(found)} peaks ({int(ok.sum())} indexed), {n_pred} predicted; "
                f"total counts {image.sum():.3g}"
            )

            pdf.savefig(fig)
            plt.close(fig)
            pages += 1

    return pages
