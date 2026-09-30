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
   on a coarse subset of strong, unambiguous, angularly distinct peaks.
   Centred lattices are searched in their primitive cell.
3. **Reassignment.** Alternating HKL assignment and weighted Wahba (Kabsch)
   refinement of `U`, first on the subset and then on all peaks; the result
   is expressed in the conventional cell with centering-allowed HKLs.
4. **Refinement.** Constrained least squares of lattice constants and
   orientation on the assigned reflections.
5. **Integration.** Rotated 2D Gaussian plus background per peak; shapes of
   strong peaks are smoothed across each panel and imposed on a second pass.

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
| IPTS-37331 / 1438 | mesolite, Fdd2, 18.40 x 56.65 x 6.54 A | consistent orientation across seeds, well above chance, but residuals of 0.5-3 degrees |
| IPTS-37331 / 1816 | T4 lysozyme, P3221 | not indexed: no orientation separates from random |

Both real runs share the IMAGINE geometry embedded in the lite files. The
mesolite residuals (expected ~0.08 degrees from the 0.95 mm pixels at
0.35 m) point to a systematic geometry error of a few millimetres, such as a
sample offset or panel placement; a small-cell pattern tolerates that, a
61 x 61 x 97 A cell does not. Transposed or mirrored pixel orderings within
the panels were ruled out. The next step is to fit a sample offset (and, if
needed, panel corrections) on run 1438 and apply it to 1816.

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
* The lite pixel centres written by the current lite recipe are offset by
  1.5 raw pixels (0.36 mm) because the full-resolution `xstart`/`ystart` were
  kept for the 4 x 4-grouped panel; worth fixing where the lite files are
  produced.
