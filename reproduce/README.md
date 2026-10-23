# Reproducing the paper results

Scripts and reference outputs behind the numbers in *Automatic indexing of
wide-band quasi-Laue neutron diffraction patterns of large unit cells*.

## Setup

- The package itself is in `src/`; every script puts it on the path, so no
  installation is needed beyond the dependencies in `environment.yml`.
  Scripts that load raw data or predict peaks also need Mantid.
- `QLR_WORK` sets the working directory for inputs and outputs (default:
  `<repo>/work`, which is git-ignored). Intermediate arrays (`*.npz`) are
  written there and are not tracked.
- `QLR_FIGURES` sets where figure scripts write PDFs (default: current
  directory). `PYTHON` sets the interpreter used by the shell drivers.
- Raw data are read from the ORNL file systems (`/HFIR/CG4D/...`,
  `/SNS/MANDI/...`), proposals IPTS-8776, 34720 and 37331.

Python scripts load their inputs by relative path, so run each from the
subdirectory of `$QLR_WORK` named in the table below, e.g.

```bash
mkdir -p work/figure && cd work/figure
python ../../reproduce/figure/prep_t4l_matched.py
```

Shell drivers (`*.sh`) can be run from anywhere.

## Inputs produced by preparation scripts

| File (in `$QLR_WORK/...`) | Produced by |
|---|---|
| `mandi/mandi_garnet.npz` | `mandi/extract_mandi.py` |
| `peaks_1816_detcal.npz`, `U_1816_detcal_seeds.npy` (80 K cell) | `t4l/t4l_verify.py` |
| `U_1816_<a>_<c>.npy` (trial room-temperature cells) | `t4l/t4l_cells.py`, `t4l/t4l_cells2.py` |
| `images_1816.npz` | `t4l/t4l_overlay.py` |
| `figure/t4l_1816.npz` | `figure/prep_t4l.py` |
| `figure/t4l_1816_matched.npz` | `figure/prep_t4l_matched.py`, then `figure/add_angle_error.py` |
| `robust/mesolite_peaks.npz` | `robust/mesolite_peaks.py` |
| `mesolite_ip/images_078.npz` | `mesolite_ip/index_tif.py` |

## Paper results

`results/` holds the outputs of the runs reported in the paper.

| Paper | Script (run in) | Reference output |
|---|---|---|
| Table 1, Sec. 3.8: detector calibration on garnet | `scripts/calibrate_geometry.py` (see its docstring) | `results/garnet_calibration.log`, `results/garnet/calibration.json` |
| Sec. 3.8: DetCal check and mesolite validation | `calibration/calibration_check.sh` | `results/garnet/check_detcal.jsonl`, `results/validate_mesolite*.log`, `results/mesolite/validation.json` |
| Sec. 3.6: $10^5$ random orientations | `null/null_mc.py 100000` (`null/`) | `results/null/null_mc.log` |
| Sec. 3.6: null searches, production acceptance rule | `null/null_accept.py <out.jsonl>` (`null/`) | `results/null/null_accept.jsonl` |
| (earlier single-restart null searches) | `null/null_search.py` (`null/`) | `results/null/null_search.jsonl` |
| Table 2, Sec. 4.1: mesolite on the IMAGINE image plate | `mesolite_ip/index_tif.py` (`mesolite_ip/`) | printed |
| Sec. 4.1, 4.3: ten seeds, mesolite and T4 lysozyme | `robust/seeds.py` (`robust/`) | `results/robust/seeds.jsonl` |
| Table 2, Sec. 4.2: garnet on MANDI, 1--3 Å | `mandi/index_mandi_1-3.py`, `mandi/harmonic_check_1-3.py` (`mandi/`) | `results/mandi/index_mandi_1-3.jsonl` |
| Sec. 4.2: garnet restricted to 1--2 Å | `mandi/index_mandi.py`, `mandi/harmonic_check.py` (`mandi/`) | `results/mandi/index_mandi.jsonl` |
| Sec. 4.2: spurious peaks | `robust/spurious_mandi.py` (`robust/`) | `results/robust/spurious_mandi.jsonl` |
| Table 2, Sec. 4.3: T4 lysozyme run 1816 (229/132, $-13.8$; 127/53, $-17.4$) | stored solution in `figure/t4l_1816_matched.npz`; search timing from `figure/search_auto.py` (`figure/`) | `results/figure/search_auto.log` |
| Sec. 4.3, Discussion: all peaks vs informative peaks ($-7.9$ vs $-13.3$) | `figure/search_matched.py`, `figure/search_auto.py` (`figure/`) | `results/figure/search_*.log` |
| Sec. 3.6, 4.3, Discussion: chance fractions (two thirds, 38%) | `figure/chance_cut.py` (`figure/`) | printed |
| Sec. 4.3: 80 K cell, nominal vs calibrated geometry | `t4l/t4l_detcal.sh` | `results/t4l_detcal.log` |
| Sec. 4.3: 80 K cell, four seeds | `t4l/t4l_verify.sh` | `results/t4l_verify.log` |
| Sec. 4.3: trial room-temperature cells | `t4l/t4l_cells.py`, `t4l/t4l_cells2.py` (any) | `results/t4l_cells*.log` |
| Sec. 4.3: refined $c/a$ | `t4l/t4l_rt_check.py` (any) | printed |
| Sec. 4.3: tolerance curve, 95% at $0.7^\circ$, low-angle and edge peaks | `figure/tol_curve.py`, `figure/tol95.py`, `figure/unindexed*.py` (`figure/`) | printed |
| Fig. T4 lysozyme | `figure/fig_t4l_v2.py` (`figure/`) | PDF |
| Sec. 4.4, Fig. timing | `benchmarks/bench_search.py`, `benchmarks/bench_search_tree_scipy.py` (driver `benchmarks/queue_tree_scipy.sh`), `figure/fig_timing.py` | `results/benchmarks/search*.jsonl`, `results/bench_*.log` |
| Appendix D: optimizer comparison | `benchmarks/bench_optimizer.py` (driver `benchmarks/queue_optimizer.sh`) | `results/benchmarks/optimizer.jsonl`, `results/bench_optimizer.log` |

Scripts were copied from the working directory in which they were run, with
absolute paths replaced by `QLR_WORK`, `QLR_FIGURES` and the repository root.

## Known gap

- **Table 3 (rotation series, runs 1818--1826).** The script that predicted,
  refined and scored these runs was not preserved, and their images are not
  in the working directory. It still has to be recovered or rewritten.
