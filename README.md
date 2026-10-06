# quasi-laue-reduction

Reduction of quasi-Laue single-crystal neutron diffraction images (no
time-of-flight): peak finding, orientation (UB) determination with known
lattice constants, lattice refinement, peak prediction and 2D integration.
Built on [Mantid](https://www.mantidproject.org) for instrument geometry and
peak workspaces; the optimizer itself needs only NumPy and SciPy.

Supported inputs:

* autoreduced "lite" event files from the IMAGINE detector on HFIR CG4D
  (`CG4D_<run>.lite.nxs.h5`), and
* CG4D image-plate TIFF images (instrument definition included).

## Installation

Mantid is distributed through conda:

```bash
conda env create -f environment.yml
conda activate quasi-laue-reduction
```

or, in an existing environment that already has Mantid:

```bash
pip install -e .
```

## Usage

Command line:

```bash
quasi-laue --lite /HFIR/CG4D/IPTS-37331/shared/autoreduce/CG4D_1438.lite.nxs.h5 \
    --cell 18.40 56.65 6.54 90 90 90 --centering F --band 2.8 4.6 \
    --lattice Orthorhombic --refine --integrate \
    --save-ub CG4D_1438.mat --save-peaks CG4D_1438.integrate
```

In Python or Mantid Workbench (see `examples/`):

```python
from quasi_laue_reduction.loading import load_lite
from quasi_laue_reduction.workflow import QuasiLaue

info = load_lite(filename, ws="data")

ql = QuasiLaue("data", pixel_shape=info["pixel_shape"])
ql.find_peaks(method="local", n_sigma=8)
ql.find_UB(a, b, c, alpha, beta, gamma, wavelength_band, centering="F", n_proc=-1, seed=1)
ql.refine_UB("Orthorhombic", wavelength_band)
ql.predict_peaks(wavelength_band, d_min=2.0)
ql.integrate_peaks()
```

## Method

A peak at a pixel fixes only the direction of its scattering vector,
`d = kf_hat - ki_hat` with `|d| = 2 sin(theta)`; its wavelength is unknown
within the band `[lambda_min, lambda_max]`. For a trial orientation `U` the
candidate reflections of peak *i* are the HKLs in the shell
`|B hkl| in [|d_i|/lambda_max, |d_i|/lambda_min]`, and the best candidate is
the one whose direction `U B hkl` is closest to `d_i`.

1. **Peak finding.** Per-bank median-filter background, robust (MAD) noise,
   local maxima above `n_sigma`, residual-weighted sub-pixel centroids.
2. **Global search.** Differential evolution over a uniform parameterisation
   of SO(3), minimising `sum_i w_i min(angle_i/tol, cap)^2 - bonus * sum_{indexed} w_i`
   on a coarse subset of strong, unambiguous, angularly distinct peaks, at a
   search tolerance (default 1 degree) wider than the residuals. Centred
   lattices are searched in their primitive cell.
3. **Reassignment.** Alternating HKL assignment and weighted Wahba (Kabsch)
   refinement of `U`, on the subset and on all peaks at the search
   tolerance, then on all peaks at the final tolerance (default 0.35
   degrees); the result is expressed in the conventional cell with
   centering-allowed HKLs.
4. **Significance and restarts.** Each solution gets a p-value: the number
   of peaks a random orientation indexes within `tol` is close to Poisson
   with mean `sum_i p_i`, `p_i = 1 - exp(-N_i tol^2 / 4)`, `N_i` the distinct
   candidate directions of peak *i* (checked against random orientations).
   Up to three restarts are run and the most significant kept, stopping at
   `log10 p < -5`.
5. **Refinement.** Constrained least squares of length ratios, angles and
   orientation on the assigned reflections; `a` is held fixed, since with
   unknown wavelengths the absolute scale is undetermined.
6. **Integration.** Rotated 2D Gaussian plus background per peak; shapes of
   strong peaks are smoothed across each panel and imposed on a second pass.

### Objective and search reliability

The number of candidate directions per peak, and so the chance of a
spurious match, varies by orders of magnitude: for synthetic 99-peak T4
lysozyme data the median is 7,700 per peak at 2.8-4.6 A and 33,000 at
2-10 A (up to 340,000), and at 0.35 degrees a random orientation indexes 15
and 31 peaks respectively, as the Poisson model above predicts. With 0.2
degree noise the true orientation indexes 68/99 at 2-10 A, so the signal is
there, but a single DE search at the full band finds it for only 2-3 of 4
seeds (`benchmarks/bench_cost.py`); widening the search tolerance to 1
degree helped, and scoring every peak by -log p_i ("significance" score)
ranked solutions correctly but did not find the basin more often. Failed
searches are easy to recognise: recovered orientations had log10 p of -7.7
to -9.1, failed ones -0.8 to -2.4. Hence the restarts with p-value
selection. Restarts must be compared by this count-based p-value: a
tolerance-free sum of log p_i over all peaks preferred a false mesolite
solution (log10 p = -1.4) over the true one (-21.6).

### Nearest-direction indexing

Evaluating one orientation used to scan every candidate in every peak's
shell (about 14,000 per peak for a 61 x 61 x 97 A cell), which is O(N_peaks
N_candidates) per evaluation. Two observations make it much cheaper:

* harmonics `n hkl` are collinear, so each shell reduces to one
  representative (the lowest order) per reciprocal-lattice row; and
* the best candidate is the nearest neighbour of `U^T d_i` among the unit
  directions of the shell, and chord distance is monotone in angle.

All shells are stacked in one 4-D KD-tree, with peak *i*'s candidates lifted
to `(ghat, 4 i)`; a query at `(U^T d_i, 4 i)` can only return a candidate of
peak *i*. One vectorised query per evaluation replaces the scan, with the
same argmax (harmonic ties now resolved deterministically to the lowest
order).

### Parallel differential evolution

SciPy's `workers=N` pickles the objective, a bound method carrying the whole
HKL table, with every task. `ChunkedPoolMap` hands the prepared optimizer to
each worker once when the pool starts and sends only the trial vectors, one
chunk per worker per generation.

## Performance

IPTS-37331 run 1816 (80-peak coarse subset, 13,600 candidates per peak),
same DE seed and settings (population 180, 300 generations) in every row;
all rows reach the identical solution (`benchmarks/bench_search.py`).
Intel i9-13950HX (32 threads, WSL2), NumPy 2.1, SciPy 1.16, Mantid 6.16.

| configuration | DE wall time | speed-up |
|---|---:|---:|
| legacy indexer, serial | 580 s | 1x |
| legacy indexer, SciPy `workers=8` | 112 s | 5.2x |
| legacy indexer, SciPy `workers=32` | 200 s | 2.9x |
| KD-tree indexer, serial | 22.1 s | 26x |
| KD-tree indexer, persistent pool, 8 workers | 4.8 s | 121x |
| KD-tree indexer, persistent pool, 32 workers | 4.3 s | 135x |

One objective evaluation (`benchmarks/bench_indexer.py`): 7.6 ms legacy,
2.4 ms per-peak scan of the harmonic-reduced shells, 0.27 ms KD-tree (28x);
objective values agree to 1e-12. With SciPy's `workers`, 32 workers are
slower than 8 because each task re-pickles the optimizer; the persistent
pool removes that overhead, and beyond ~8 workers the per-generation
synchronisation dominates.

Loading the 4.3 GB lite file of run 1816 with `LoadNexus` took 1,853 s over
the HFIR file system; the workflow only needs counts per pixel, which
`load_lite` sums directly from the HDF5 event weights (33 s for the 0.6 GB
file of run 1438).

**Search size** (`benchmarks/bench_recovery.py`). On synthetic 99-peak
T4 lysozyme data (P3221, 61.5 / 95.9 A, detector coverage 10-80 degrees
2-theta) random orientations index a median of 15 peaks (99.9th
percentile 25). The previous default search (population 60 x 3, 300
generations) recovered the true orientation for 0 of 3 seeds (18-25
indexed, 1.4-20 degrees off); population 300 x 3 with 1000 generations
recovered it for 3 of 3 (90/99 indexed, 0.014 degrees from truth) in 46 s
each on 16 workers. Small centred cells behave the same way (see the
tests). The larger search is now the default and is affordable only because
of the faster indexer.

Reproduce with the scripts in `benchmarks/` (results go to
`benchmarks/results/`):

```bash
python benchmarks/bench_indexer.py [--peaks peaks.npz]
python benchmarks/bench_search.py --n-proc 1 8 32
python benchmarks/bench_recovery.py --sizes 60x300 300x1000
python benchmarks/bench_loading.py /HFIR/CG4D/IPTS-37331/shared/autoreduce/CG4D_1816.lite.nxs.h5
```

`benchmarks/legacy_optimize_refine_ub.py` is the unmodified previous
optimizer, kept as the baseline.

## Test cases

| run | sample | status |
|---|---|---|
| synthetic | T4 lysozyme cell | recovered for every seed, 90/99 indexed, 0.014 degrees from truth |
| IPTS-37331 / 2022 | garnet, Ia-3d, a = 11.93 A | indexed: 29/46 peaks at 0.35 degrees (random: 5), seeds agree to 0.1 degrees, median residual 0.24 degrees, band 2.0-4.6 A |
| IPTS-37331 / 1438 | mesolite, Fdd2, 18.40 x 56.65 x 6.54 A | indexed at 2-10 A: 24/57 at 0.35 degrees (1.3 expected by chance, log10 p = -21.6), 33/57 at 1 degree |
| IPTS-37331 / 1816 | T4 lysozyme, P3221 | not indexed at 2.8-4.6, 2.0-4.6 or 2-10 A: no orientation separates from random |

The incident band is logged as enum PVs in the raw NeXus files:
`CG4D:CS:Optics:LambdaMin:RBV` (0 = 2.0 A, 1 = 2.78 A, 2 = 3.3 A) and
`CG4D:CS:Optics:LambdaMax:RBV` (0 = 3.0 A, 1 = 4.0 A, 2 = 4.5 A, 3 = open,
10 A). Runs 1379, 1438, 1816 and 2022 all record 2.0 A to open, and the
indexed wavelengths agree (garnet from 2.08 A, mesolite up to 9.5 A). The
lite files keep only the enum index, not the labels.

All three runs share the IMAGINE geometry embedded in the lite files. The
lite-to-raw pixel mapping was verified directly (see below), and garnet
indexes with residuals of ~0.2 degrees, so the geometry is broadly right;
the remaining ~0.2 degree residual is still above the ~0.08 degrees expected
from the 0.95 mm pixels at 0.35 m. A 61 x 61 x 97 A cell has so many
reflections per shell that random orientations already index most peaks at
this tolerance, so 1816 needs either a tighter geometry (sample offset and
panel corrections fitted on garnet) or confirmation of the sample and cell.

## Tests

```bash
pytest -m "not slow"   # ~1.5 min; the Mantid end-to-end test is skipped without Mantid
pytest -m slow         # full-size protein-cell recovery
```

## Changes from the previous scripts

* The optimizer, peak finding, loading and workflow are one package
  (`src/quasi_laue_reduction`); the old scripts are in `legacy/`. The old
  top-level `reduction_workflow.py` shadowed Mantid's own
  `reduction_workflow` package and broke its SANS plugins when run from the
  repository directory.
* Indexing: KD-tree nearest-direction lookup (exact), deterministic
  harmonic tie-breaking, and the HKL table now grows when a later call needs
  a wider q range (previously a table prepared on a subset silently
  truncated the shells of the other peaks unless `force=True`).
* The A-, B- and C-centering matrices were left-handed (negative
  determinant), giving improper "rotations" and a transform that
  `TransformHKL` rejects; all are now right-handed. Centred cells are
  converted to the conventional setting directly (no `TransformHKL`), and
  refinement uses the conventional lattice system with the centering
  reflection condition.
* `PredictPeaks` reflection-condition names were misspelled for R
  centering.
* Integration: the integrated intensity was `2 pi A s1 s2 - B`, subtracting
  a per-pixel background level from an integrated area; it is now
  `2 pi A s1 s2` (the background is fitted separately). The ROI doubled
  cumulatively from panel to panel, and rows of unmatched peaks were not
  deleted.
* Peak lookup by detector ID is a sorted search instead of `list.index`
  over all pixels.
* HKL sign convention: peak vectors were built as `kf - ki`, while Mantid's
  default `Q.convention` is `Inelastic` (`Q = ki - kf`), so every assigned
  HKL was the Friedel mate of what `IndexPeaks`/`PredictPeaks` and exported
  peak files expect (orientation and predicted positions were unaffected,
  since the reflection set is centrosymmetric). `QuasiLaue` now sets
  `Q.convention = Crystallography` (`Q = kf - ki`, matching the peak
  vectors); on garnet (run 2022) Mantid's `IndexPeaks` reproduces all
  29 assigned HKLs and Mantid's Q agrees with `2 pi UB hkl` in length and
  direction. There is no 2 pi discrepancy: UB is in the `|UB hkl| = 1/d`
  convention on both sides.
* The lite files place each 4 x 4-grouped pixel 1.5 raw pixels (0.36 mm per
  panel axis, 0.5 mm total) from the centroid of the raw pixels it sums,
  because the grouped panel kept the full-resolution `xstart`/`ystart`.
  Verified against the raw NeXus geometry and the `GroupDetectors` pattern
  in the lite file's history (constant offset, no transposition or flip).
  `load_lite` corrects it (`correct_pixel_centres=True`); it is still worth
  fixing where the lite files are produced.
* Refinement holds `a` fixed by default: with unknown wavelengths the
  absolute cell scale is undetermined (all lengths and wavelengths can be
  scaled together), and a free scale drifted with a zero reported error.
* The incident band for the CG4D runs checked reaches down to ~2.1 A: on
  garnet (run 2022, Ia-3d, a = 11.93 A) 2.8-4.6 A indexes 13/46 peaks
  (median residual 2.9 degrees), 2.0-4.6 A indexes 29/46 (0.24 degrees).
