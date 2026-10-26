# quasi-laue-reduction

Reduction of quasi-Laue single-crystal neutron diffraction images (no
time-of-flight): peak finding, orientation (UB) determination with known
lattice constants, lattice refinement, peak prediction and 2D integration.
Built on [Mantid](https://www.mantidproject.org) for instrument geometry and
peak workspaces; the optimizer itself needs only NumPy and SciPy.

The method and its validation are described in Morgan *et al.*, *Automatic
indexing of wide-band quasi-Laue neutron diffraction patterns of large unit
cells* (submitted). The scripts that reproduce every result in the paper are in
[`reproduce/`](reproduce/README.md).

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
from quasi_laue_reduction.loading import calibration_file, load_lite
from quasi_laue_reduction.workflow import QuasiLaue

info = load_lite(filename, ws="data", detcal=calibration_file())  # IMAGINE-X geometry

ql = QuasiLaue("data", pixel_shape=info["pixel_shape"])
ql.find_peaks(method="matched", z_min=5)
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

1. **Peak finding.** Per-camera median-filter background, robust (MAD) noise,
   and either local maxima above `n_sigma` or a matched filter (`z >= z_min`),
   with residual-weighted sub-pixel centroids.
2. **Informative peaks.** A peak whose chance of being matched by a random
   orientation, `1 - exp(-N_i tol^2 / 4)` for `N_i` candidates, exceeds 0.7 is
   left out of the search and scoring and indexed only at the end.
3. **Global search.** Differential evolution over a uniform parameterisation
   of SO(3), minimising `sum_i w_i min(angle_i/tol, cap)^2 - bonus * sum_{indexed} w_i`
   on a subset of up to 80 strong, unambiguous, angularly distinct peaks, at a
   search tolerance (default 1 degree) wider than the residuals. The weights
   `w_i ~ sqrt(I_i / N_i)` are capped at twice their mean. Centred lattices
   are searched in their primitive cell. Each trial orientation is evaluated
   with one query of a 4-D k-d tree in which every peak's candidate
   directions occupy their own layer, by a persistent pool of workers that
   holds the tree.
4. **Reassignment.** Alternating HKL assignment and weighted Wahba (Kabsch)
   refinement of `U`, on the subset and on all informative peaks at the
   search tolerance, then at the final tolerance (default 0.35 degrees).
5. **Chance-match score and restarts.** The number of peaks a random
   orientation indexes within `tol` is approximated as Poisson with mean
   `sum_i p_i`; a solution is scored by the Poisson tail `p`, taking the more
   significant of the values at the final and search tolerances. Up to three
   restarts are run, stopping at `log10 p < -5`; if none reaches it, the most
   significant is returned and flagged as not indexed
   (`significance["accepted"]`).
6. **Refinement.** Constrained least squares of length ratios, angles and
   orientation on the assigned reflections; one length is held fixed, since
   with unknown wavelengths the absolute scale is undetermined.
7. **Integration.** Rotated 2D Gaussian plus background per peak; shapes of
   strong peaks are smoothed across each panel and imposed on a second pass.

## Detector calibration

The IMAGINE-X geometry calibrated on garnet (IPTS-37331) is packaged as ISAW
DetCal files (`src/quasi_laue_reduction/calibration/`) and is applied with
`load_lite(..., detcal=calibration_file())` or `quasi-laue --detcal default`.
Indexing of protein cells over the full 2-10 A band needs it.
`scripts/calibrate_geometry.py` refits the geometry for a new calibration
series.

The incident band is logged as enum PVs in the raw NeXus files:
`CG4D:CS:Optics:LambdaMin:RBV` (0 = 2.0 A, 1 = 2.78 A, 2 = 3.3 A) and
`CG4D:CS:Optics:LambdaMax:RBV` (0 = 3.0 A, 1 = 4.0 A, 2 = 4.5 A, 3 = open,
10 A). Runs 1379, 1438, 1816 and 2022 all record 2.0 A to open, and the
indexed wavelengths agree (garnet from 2.08 A, mesolite up to 9.5 A). The
lite files keep only the enum index, not the labels.

## Results and benchmarks

Indexing results, the comparison with time-of-flight indexing on MANDI, the
rotation series of T4 lysozyme, the null searches, the optimizer comparison
and the timing benchmarks are reported in the paper; `reproduce/README.md`
maps each of them to its script and reference output. The benchmark scripts
are in `benchmarks/`; `benchmarks/baseline_scan.py` is the previous
optimizer, kept unchanged as the candidate-scan baseline of the timing
comparison.

## Tests

```bash
pytest -m "not slow"   # ~1.5 min; the Mantid end-to-end test is skipped without Mantid
pytest -m slow         # full-size protein-cell recovery
```

## Changes from the previous scripts

The package replaces the earlier CG4D reduction scripts (available in the git
history). Besides the k-d tree indexer, persistent worker pool and chance-match
score, it fixes several defects of those scripts: left-handed A-, B- and
C-centering matrices, truncated candidate shells when the HKL table was
prepared on a subset of peaks, misspelled reflection conditions for R
centering in `PredictPeaks`, an integrated intensity that subtracted a
per-pixel background from an integrated area, and a top-level
`reduction_workflow.py` that shadowed Mantid's own package.
