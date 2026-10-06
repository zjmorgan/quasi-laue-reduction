"""
End-to-end workflow on a small two-panel test instrument (requires Mantid).
"""

import numpy as np
import pytest
import scipy.spatial

mantid = pytest.importorskip("mantid")

from mantid.simpleapi import CreateWorkspace, LoadInstrument, PreprocessDetectorsToMD, mtd  # noqa: E402

from quasi_laue_reduction.optimize import centering_filter, primitive_cell  # noqa: E402
from quasi_laue_reduction.simulate import simulate_laue  # noqa: E402
from quasi_laue_reduction.workflow import QuasiLaue  # noqa: E402

pytestmark = pytest.mark.mantid

N = 128
PITCH = 0.00095


def panel(name, idstart, t, p, rot, r=0.25):
    return f"""
  <component type="panel" idstart="{idstart}" idfillbyfirst="y" idstepbyrow="{N}" name="{name}">
    <location r="{r}" t="{t}" p="{p}" rot="{rot}" axis-x="0" axis-y="1" axis-z="0"/>
  </component>"""


IDF = f"""<?xml version="1.0" encoding="UTF-8"?>
<instrument xmlns="http://www.mantidproject.org/IDF/1.0" name="TESTLAUE" valid-from="2000-01-01 00:00:00">
  <defaults>
    <length unit="metre"/>
    <angle unit="degree"/>
    <reference-frame>
      <along-beam axis="z"/>
      <pointing-up axis="y"/>
      <handedness val="right"/>
    </reference-frame>
  </defaults>
  <component type="moderator"><location z="-3.0"/></component>
  <type name="moderator" is="Source"/>
  <component type="sample-position"><location/></component>
  <type name="sample-position" is="SamplePos"/>
  {panel("bank1", 0, 50, 0, 50)}
  {panel("bank2", N * N, 90, 180, -90)}
  {panel("bank3", 2 * N * N, 90, 0, 90)}
  {panel("bank4", 3 * N * N, 130, 180, -130)}
  <type name="panel" is="rectangular_detector" type="pixel"
        xpixels="{N}" xstart="{-(N - 1) / 2 * PITCH}" xstep="{PITCH}"
        ypixels="{N}" ystart="{-(N - 1) / 2 * PITCH}" ystep="{PITCH}"/>
  <type is="detector" name="pixel">
    <cuboid id="pixel-shape">
      <left-front-bottom-point x="-0.0005" y="-0.0005" z="0.0"/>
      <left-front-top-point x="-0.0005" y="0.0005" z="0.0"/>
      <left-back-bottom-point x="-0.0005" y="-0.0005" z="-0.001"/>
      <right-front-bottom-point x="0.0005" y="-0.0005" z="0.0"/>
    </cuboid>
    <algebra val="pixel-shape"/>
  </type>
</instrument>
"""

CONV = (14.0, 18.0, 22.0, 90.0, 90.0, 90.0)
NB = 4
WL = (2.0, 4.5)
X_TRUE = [0.35, 0.62, 0.48]


def make_workspace(ws, counts):
    n = NB * N * N
    CreateWorkspace(OutputWorkspace=ws, DataX=np.tile([0.0, 1.0], n), DataY=counts, NSpec=n, UnitX="TOF")
    LoadInstrument(Workspace=ws, InstrumentXML=IDF, InstrumentName="TESTLAUE", RewriteSpectraMap=True)


@pytest.fixture(scope="module")
def laue_workspace():
    make_workspace("geom", np.zeros(NB * N * N))
    PreprocessDetectorsToMD(InputWorkspace="geom", OutputWorkspace="geom_det")
    t = mtd["geom_det"]
    L2, tt, az = (np.array(t.column(c)) for c in ("L2", "TwoTheta", "Azimuthal"))
    kf_pix = np.column_stack([np.sin(tt) * np.cos(az), np.sin(tt) * np.sin(az), np.cos(tt)])

    prim = primitive_cell(*CONV, "C")
    kf_ki, _, _ = simulate_laue(
        prim, X_TRUE, WL, d_min=1.0, n_peaks=80, detector_directions=kf_pix, coverage_deg=0.2, noise_deg=0.0, rng=0
    )

    _, pix = scipy.spatial.cKDTree(kf_pix).query(kf_ki + [0, 0, 1.0])

    rng = np.random.default_rng(1)
    images = np.full((NB, N, N), 20.0)
    ii, jj = np.mgrid[0:N, 0:N]
    for p in pix:
        b, i, j = np.unravel_index(p, images.shape)
        images[b] += 2000 * np.exp(-0.5 * ((ii - i) ** 2 + (jj - j) ** 2) / 1.0**2)
    images = rng.poisson(images).astype(float)

    make_workspace("data", images.ravel())
    return "data", len(pix)


def test_full_workflow(laue_workspace):
    ws, n_true = laue_workspace

    ql = QuasiLaue(ws, pixel_shape=(N, N))
    ql.find_peaks(method="local", n_sigma=10)

    # spots within the 8-pixel border exclusion or overlapping are not found
    assert len(ql.kf_ki_dir) >= 0.7 * n_true

    num = ql.find_UB(*CONV, WL, centering="C", n_proc=-1, seed=1)
    assert num >= 0.9 * len(ql.heights)

    peaks = mtd[ws + "_peaks"]
    hkl = np.array([p.getHKL() for p in peaks])
    assert np.all(np.any(hkl != 0, axis=1))
    assert np.allclose(hkl, np.round(hkl))
    assert np.all(centering_filter("C")(np.round(hkl).astype(int)))

    ql.refine_UB("Orthorhombic", WL)
    assert ql.opt.a == pytest.approx(CONV[0], rel=5e-3)
    assert ql.opt.b == pytest.approx(CONV[1], rel=5e-3)
    assert ql.opt.c == pytest.approx(CONV[2], rel=5e-3)

    ql.predict_peaks(WL, d_min=2.0)
    assert mtd[ws + "_peaks"].getNumberPeaks() > 0

    ql.integrate_peaks(roi_pixels=6)
    intens = np.array([p.getIntensity() for p in mtd[ws + "_peaks"]])
    assert np.mean(intens > 0) > 0.5


def test_detcal_round_trip(laue_workspace, tmp_path):
    from scipy.spatial.transform import Rotation

    from quasi_laue_reduction.detcal import apply_detcal, write_detcal

    make_workspace("nominal", np.zeros(NB * N * N))
    pos0 = QuasiLaue("nominal", pixel_shape=(N, N)).positions

    target = pos0.copy()
    target[0] += [0.004, -0.002, 0.001]
    c = target[1].reshape(-1, 3).mean(0)
    target[1] = c + Rotation.from_rotvec([0.01, 0.02, -0.005]).apply(target[1].reshape(-1, 3) - c).reshape(N, N, 3)

    fn = str(tmp_path / "test.DetCal")
    write_detcal(fn, target, range(1, NB + 1))

    make_workspace("moved", np.zeros(NB * N * N))
    assert apply_detcal("moved", fn, (N, N)) == NB
    pos1 = QuasiLaue("moved", pixel_shape=(N, N)).positions

    assert np.abs(pos1 - target).max() < 2e-6
