"""Wall time of one DE search (T4 lysozyme, run 1816) against worker processes."""
import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import json, sys
import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 14, "text.usetex": True, "pgf.rcfonts": False,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": " ".join([r"\usepackage[T1]{fontenc}", r"\usepackage{libertine}",
                              r"\usepackage[libertine]{newtxmath}", r"\usepackage[scientific-notation=true]{siunitx}",
                              r"\usepackage{tensor}", r"\usepackage{physics}", r"\usepackage{mhchem}"]),
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "axes.spines.top": True, "axes.spines.right": True, "axes.spines.bottom": True, "axes.spines.left": True,
    "axes.xmargin": 0.00, "axes.ymargin": 0.0,
})
OUT = sys.argv[1] if len(sys.argv) > 1 else "timing.workers.pdf"
rec = [json.loads(l) for l in open(_os.path.join(_REPO, "benchmarks", "results", "search.jsonl")) if l.strip()]
# latest measurement per configuration (the serial scan was rerun with the machine awake)
rec = list({(x["variant"], x["n_proc"]): x for x in rec}.values())
curves = {"legacy": (r"candidate scan, SciPy pool", "o"),
          "tree_scipy": (r"$k$-d tree, SciPy pool", "s"),
          "current": (r"$k$-d tree, persistent pool", "^")}

fig, ax = plt.subplots(figsize=(5.0, 4.0), layout="constrained")
ax.set_prop_cycle(color=plt.get_cmap("Set2").colors)
model = lambda n, ts, tp, c: ts + tp / n + c * n      # Amdahl plus per-worker overhead
fits = {}
nn = np.geomspace(1, 32, 200)
for key, (label, marker) in curves.items():
    r = sorted((x["n_proc"], x["seconds"]) for x in rec if x["variant"] == key)
    if len(r) < 4:          # benchmark not (yet) complete for this variant
        next(ax._get_lines.prop_cycler) if hasattr(ax._get_lines, "prop_cycler") else None
        continue
    n, t = np.array(r).T
    # SciPy with one worker evaluates serially, without a pool; for the k-d tree
    # that serial point lacks the communication cost, so the fit starts at 2
    f = n >= 2 if key == "tree_scipy" else np.ones(len(n), bool)
    p, cov = curve_fit(model, n[f], t[f], p0=[0.0, t[f][0], t[f][-1] / n[f][-1]], bounds=(0, np.inf), sigma=t[f])
    fits[key] = (p, np.sqrt(np.diag(cov)))
    color = plt.get_cmap("Set2")(list(curves).index(key))
    # one legend entry per marker/line pair
    ax.plot(n, t, marker=marker, ls="none", ms=6, color=color)
    nf = nn[nn >= n[f].min()]
    ax.plot(nf, model(nf, *p), "-", lw=1.5, color=color)
    if key == "tree_scipy" and 1 in n:   # switch from serial evaluation to the pool
        ax.plot([1, n[f].min()], [t[n == 1][0], model(n[f].min(), *p)], ":", lw=1.2, color=color)
    ax.plot([], [], marker=marker, ls="-", lw=1.5, ms=6, color=color, label=label)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32]))
ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax.xaxis.set_minor_locator(FixedLocator([n for n in range(1, 33) if n not in (1, 2, 4, 8, 16, 32)]))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlim(0.85, 38)
ax.set_ylim(2, 30000)
ax.set_xlabel("worker processes")
ax.set_ylabel("wall time [s]")
ax.legend(frameon=False, fontsize=12, loc="center right", bbox_to_anchor=(1.0, 0.42))
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
for k, (p, e) in fits.items():
    print(k, "Ts %.2f+-%.2f Tp %.1f+-%.1f c %.3f+-%.3f frac %.3f nopt %.1f" % (p[0], e[0], p[1], e[1], p[2], e[2], p[1] / (p[0] + p[1]), np.sqrt(p[1] / p[2])))
t = {k: {x["n_proc"]: x["seconds"] for x in rec if x["variant"] == k} for k in ("legacy", "current")}
print("speed-up serial scan -> tree 32:", round(t["legacy"][1] / t["current"][32]), "| tree serial/32:", round(t["current"][1] / t["current"][32], 1),
      "| scan best", min(t["legacy"].values()), "at", min(t["legacy"], key=t["legacy"].get))
