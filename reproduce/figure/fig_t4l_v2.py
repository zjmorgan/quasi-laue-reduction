"""IMAGINE-X T4 lysozyme (CG4D run 1816) figure: detector view, significance curve, structure."""

import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, PercentFormatter

sys.path.insert(0, _os.path.join(_REPO, "src"))
from quasi_laue_reduction.peaks import background_residual

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 14,
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": " ".join(
            [
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{libertine}",
                r"\usepackage[libertine]{newtxmath}",
                r"\usepackage[scientific-notation=true]{siunitx}",
                r"\usepackage{tensor}",
                r"\usepackage{physics}",
                r"\usepackage{mhchem}",
            ]
        ),
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "axes.xmargin": 0.00,
        "axes.ymargin": 0.0,
    }
)

MS = (FIGURE_DIR + "/")
OUT = sys.argv[1] if len(sys.argv) > 1 else "imaginex.t4l.pdf"
STRUCTURE = sys.argv[2] if len(sys.argv) > 2 else MS + "image1.png"

d = np.load("t4l_1816.npz")
m = np.load("t4l_1816_matched.npz")
images, P = d["images"], d["positions"]
coords, hkl = m["coords"], m["hkl"]
indexed = np.any(hkl != 0, axis=1)

kfp = P / np.linalg.norm(P, axis=-1, keepdims=True)
gamma = np.rad2deg(np.arctan2(kfp[..., 0], kfp[..., 2]))
nu = np.rad2deg(np.arcsin(kfp[..., 1]))

sig = np.zeros_like(images)
for b in range(len(images)):
    if images[b].any():
        res, s, mask = background_residual(images[b])
        sig[b] = np.where(mask, 0, res / s)
live = [b for b in range(len(images)) if images[b].any()]


def angles(c):
    b = c[:, 0].astype(int)
    i, j = np.clip(c[:, 1], 0, 126.999), np.clip(c[:, 2], 0, 126.999)
    i0, j0 = i.astype(int), j.astype(int)
    fi, fj = (i - i0)[:, None], (j - j0)[:, None]
    p = (P[b, i0, j0] * (1 - fi) * (1 - fj) + P[b, i0 + 1, j0] * fi * (1 - fj)
         + P[b, i0, j0 + 1] * (1 - fi) * fj + P[b, i0 + 1, j0 + 1] * fi * fj)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return np.rad2deg(np.arctan2(p[:, 0], p[:, 2])), np.rad2deg(np.arcsin(p[:, 1]))


g_pk, n_pk = angles(coords)

# per-peak angular error to the nearest in-band candidate (degrees)
err = m["angle_error"]
ERR_CMAP = plt.get_cmap("viridis")
ERR_NORM = matplotlib.colors.Normalize(vmin=0.0, vmax=0.5, clip=True)

fig = plt.figure(figsize=(8.0, 6.7), layout="constrained")
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0)
top, bottom = fig.subfigures(2, 1, height_ratios=[1.75, 1])
# forward array left, back array right: gamma decreases continuously left to right
ax_f, ax_b = top.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [102, 82], "wspace": 0.0})
ax_c, ax_s = bottom.subplots(1, 2, gridspec_kw={"width_ratios": [1.9, 1]})

kw = dict(s=0.15, cmap="binary", vmin=0, vmax=6, rasterized=True, linewidths=0)
for ax in (ax_b, ax_f):
    for b in live:
        ax.scatter(gamma[b].ravel(), nu[b].ravel(), c=sig[b].ravel(), **kw)
    ax.scatter(g_pk, n_pk, s=14, ec=ERR_CMAP(ERR_NORM(err)), fc="none", lw=0.6, zorder=10)
    ax.set_aspect(1)
    ax.minorticks_on()
    ax.xaxis.set_major_formatter(FormatStrFormatter(r"$%d^\circ$"))
    ax.yaxis.set_major_formatter(FormatStrFormatter(r"$%d^\circ$"))
    ax.set_xlabel(r"$\gamma$")
ax_b.set_xlim(-78, -160)
ax_f.set_xlim(102, 0)
ax_f.set_ylim(-57, 57)
ax_f.set_yticks([-40, -20, 0, 20, 40, 60])
ax_f.set_ylabel(r"$\nu$")
cax = ax_b.inset_axes([1.03, 0.0, 0.035, 1.0])   # follows the drawn panel exactly
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=ERR_NORM, cmap=ERR_CMAP), cax=cax)
cb.set_label("angular error")
cb.ax.yaxis.set_major_formatter(FormatStrFormatter(r"$%.1f^\circ$"))
cb.minorticks_on()
ax_f.set_anchor("E")
ax_b.set_anchor("W")
ax_b.tick_params(left=False, which="both")

# ---- (b) indexed fraction of the search peaks against tolerance
tt = np.concatenate([[0], m["tols"]])
n = int(m["informative"].sum())
obs = np.concatenate([[0], m["obs"]]) / n
ch = np.concatenate([[0], m["chance"]])
ax_c.set_prop_cycle(color=plt.get_cmap("Set2").colors)
(line_r,) = ax_c.plot(tt, ch / n, ls="--", label=r"random ($\pm3\sigma$)")
ax_c.fill_between(tt, (ch - 3 * np.sqrt(ch)) / n, (ch + 3 * np.sqrt(ch)) / n, color=line_r.get_color(), alpha=0.25, lw=0)
ax_c.plot(tt, obs, "o-", ms=3, label="solution")
ax_c.set_xlabel("angular tolerance")
ax_c.xaxis.set_major_formatter(FormatStrFormatter(r"$%.1f^\circ$"))
ax_c.set_ylabel("indexed")
ax_c.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax_c.set_xlim(0, 1)
ax_c.set_ylim(0, 1.02)
ax_c.minorticks_on()
ax_c.legend(frameon=False, fontsize=12, loc="lower right")

# ---- (c) structure, to the right of (b)
ax_s.set_axis_off()
if os.path.exists(STRUCTURE):
    img = plt.imread(STRUCTURE)
    if img.shape[-1] == 4:   # trim transparent margins
        rows, cols = np.flatnonzero(img[..., 3].max(axis=1) > 0), np.flatnonzero(img[..., 3].max(axis=0) > 0)
        img = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    ax_s.imshow(img)
    ax_s.set_anchor("C")

    # crystal axes (projection), as in the mesolite and garnet structures:
    # a to the right, b at 120 deg toward the upper left, c mostly out of
    # the page with a short projection slightly down and toward a
    H, W = img.shape[:2]
    L = 0.12 * W
    ax_s.set_xlim(-0.09 * W, W - 0.5)   # room for the axes beside the structure
    ax_s.set_ylim(H - 0.5, -0.5)
    o = np.array([0.02 * W, 0.30 * H])          # image pixels, y down
    axes_2d = {
        "a": ("r", np.array([L, 0.0])),
        "b": ("g", L * np.array([np.cos(np.deg2rad(120)), -np.sin(np.deg2rad(120))])),
        "c": ("b", 0.5 * L * np.array([0.55, 0.85])),
    }
    for name, (color, v) in axes_2d.items():
        ax_s.annotate("", xy=o + v, xytext=o, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                                              mutation_scale=9, shrinkA=0, shrinkB=0), zorder=50,
                      annotation_clip=False)
        u = v / np.linalg.norm(v)
        ax_s.text(*(o + v + 0.055 * W * u), rf"${name}$", ha="center", va="center", zorder=50, clip_on=False)

# panel labels, as in the mesolite and garnet figures
# (a) outside the frame, above its top-left corner, as in the MANDI figure
ax_f.text(-0.24, 1.015, "(a)", transform=ax_f.transAxes, va="bottom", ha="left", zorder=20)
ax_c.text(0.015, 0.95, "(b)", transform=ax_c.transAxes, va="top", ha="left", zorder=20)
ax_s.text(-0.16, 0.97, "(c)", transform=ax_s.transAxes, va="top", ha="left", zorder=40)

# align the left spine of (b) with the left spine of (a)
fig.canvas.draw()
fig.canvas.draw()
inv_b = bottom.transSubfigure.inverted()
(ax0, _), _ = inv_b.transform(ax_f.get_window_extent().get_points())
(cx0, cy0), (cx1, cy1) = inv_b.transform(ax_c.get_window_extent().get_points())
fig.set_layout_engine("none")
ax_c.set_position([ax0, cy0, cx1 - ax0, cy1 - cy0])

fig.savefig(OUT, dpi=400, bbox_inches="tight", pad_inches=0.02)
print("peaks", len(coords), "indexed", int(indexed.sum()), "search", n, "tol95", float(m["tol95"]),
      "structure", os.path.exists(STRUCTURE))
