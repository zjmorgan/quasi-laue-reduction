"""
Loading detector images into Mantid workspaces.

Two sources are supported:

* Autoreduced "lite" event NeXus files (IMAGINE geometry on CG4D). The
  workflow only uses the total counts per pixel, so the event weights are
  summed straight from the HDF5 file and a one-bin Workspace2D is built
  with the instrument definition embedded in the file. This avoids
  ``LoadNexus`` of the full event list, which dominates run time for large
  files read over network storage.
* CG4D image-plate TIFF images, mapped pixel-by-pixel onto the packaged
  ``CG4D_Definition.xml``.
"""

import re
from importlib import resources

import h5py
import numpy as np


def instrument_definition(name):
    """
    Path to an instrument definition packaged with this module.
    """
    return str(resources.files("quasi_laue_reduction") / "instruments" / name)


def calibration_file(name="IMAGINE_garnet_IPTS-37331_2026-10-01", resolution="lite"):
    """
    Path to a packaged DetCal calibration.

    Parameters
    ----------
    name : str, optional
        Calibration name (see ``calibration/*.json`` for provenance).
    resolution : str, optional
        ``"lite"`` (128 x 128 panels) or ``"full"`` (512 x 512).
    """
    return str(resources.files("quasi_laue_reduction") / "calibration" / f"{name}_{resolution}.DetCal")


def pixel_shape_from_idf(xml):
    """
    (xpixels, ypixels) of the rectangular panels in an IDF string.
    """
    xs = set(re.findall(r'xpixels="(\d+)"', xml))
    ys = set(re.findall(r'ypixels="(\d+)"', xml))

    if len(xs) != 1 or len(ys) != 1:
        raise ValueError("Expected a single rectangular panel type in the IDF.")

    return int(xs.pop()), int(ys.pop())


def grouping_size(pattern):
    """
    Side length g of the square g x g pixel groups in a GroupingPattern.
    """
    first = pattern.split(",", 1)[0]
    n = len(first.split("+"))
    g = int(round(np.sqrt(n)))
    return g if g * g == n else 1


def correct_grouped_panel_origin(xml, group):
    """
    Move grouped-panel pixel centres to the centres of their pixel groups.

    A g x g-grouped rectangular panel written with the full-resolution
    ``xstart``/``ystart`` (the centre of the first raw pixel) places every
    grouped pixel (g - 1) / 2 raw pixels off the centroid of the pixels it
    sums: 1.5 raw pixels (0.36 mm) for the IMAGINE 4 x 4 lite files. The
    start is shifted only when it still equals the raw value,
    -N g / 2 * step / g.

    Returns
    -------
    xml : str
        Corrected IDF.
    corrected : bool
        Whether a panel was changed.
    """
    if group <= 1:
        return xml, False

    corrected = False

    def fix(m):
        nonlocal corrected
        axis, n, start, step = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
        raw_step = step / group
        if not np.isclose(start, -n * group / 2 * raw_step, rtol=0, atol=1e-9):
            return m.group(0)
        corrected = True
        new = start + (group - 1) / 2 * raw_step
        return f'{axis}pixels="{n}" {axis}start="{new!r}" {axis}step="{step!r}"'

    pattern = r'([xy])pixels="(\d+)"\s+\1start="([-\d.eE+]+)"\s+\1step="([-\d.eE+]+)"'
    xml = re.sub(pattern, fix, xml)

    return xml, corrected


def read_lite(filename):
    """
    Per-pixel counts and metadata from a lite event NeXus file.

    Parameters
    ----------
    filename : str
        Lite ``.nxs.h5`` file written by ``SaveNexus`` of an EventWorkspace.

    Returns
    -------
    info : dict
        ``counts`` (per spectrum, summed event weights), ``idf`` (embedded
        instrument XML), ``instrument``, ``pixel_shape``, ``run_number``,
        ``title``, ``goniometer`` (3x3 rotation, identity if absent) and
        ``group`` (pixel grouping side length from the processing history,
        1 if none).
    """
    with h5py.File(filename, "r") as f:
        entry = f["mantid_workspace_1"]
        events = entry["event_workspace"]

        indices = events["indices"][()]
        weights = events["weight"][()]

        idf = entry["instrument/instrument_xml/data"][()][0]
        instrument = entry["instrument/name"][()][0]

        logs = entry["logs"]

        def text(value):
            return value.decode() if isinstance(value, bytes) else str(value)

        run_number = text(logs["run_number/value"][()][0]) if "run_number" in logs else ""
        title = text(entry["title"][()][0]) if "title" in entry else ""

        R = np.eye(3)
        if "goniometer/rotation_matrix" in logs:
            R = np.asarray(logs["goniometer/rotation_matrix"][()], dtype=float).reshape(3, 3)

        group = 1
        for key in entry.get("process", {}):
            if key.startswith("MantidAlgorithm"):
                record = text(entry["process"][key]["data"][()][0])
                if record.startswith("Algorithm: GroupDetectors"):
                    m = re.search(r"GroupingPattern, Value: ([0-9+,]+)", record)
                    if m:
                        group = grouping_size(m.group(1))

    counts = np.zeros(len(indices) - 1)
    nonempty = np.flatnonzero(np.diff(indices) > 0)
    counts[nonempty] = np.add.reduceat(weights.astype(np.float64), indices[nonempty])

    idf = text(idf)

    return {
        "counts": counts,
        "idf": idf,
        "instrument": text(instrument),
        "pixel_shape": pixel_shape_from_idf(idf),
        "run_number": run_number,
        "title": title,
        "goniometer": R,
        "group": group,
    }


def _counts_workspace(ws, counts, idf_xml=None, idf_file=None, instrument=None):
    from mantid.simpleapi import CreateWorkspace, LoadInstrument

    n = len(counts)

    CreateWorkspace(
        OutputWorkspace=ws,
        DataX=np.tile([0.0, 1.0], n),
        DataY=np.asarray(counts, dtype=float),
        DataE=np.sqrt(np.maximum(counts, 0.0)),
        NSpec=n,
        UnitX="TOF",
    )

    if idf_xml is not None:
        LoadInstrument(
            Workspace=ws,
            InstrumentXML=idf_xml,
            InstrumentName=instrument,
            RewriteSpectraMap=True,
        )
    else:
        LoadInstrument(Workspace=ws, Filename=idf_file, RewriteSpectraMap=True)


def load_lite(filename, ws="data", correct_pixel_centres=True, detcal=None):
    """
    One-bin Workspace2D of per-pixel counts from a lite event NeXus file.

    Parameters
    ----------
    filename : str
        Lite event NeXus file.
    ws : str, optional
        Output workspace name.
    correct_pixel_centres : bool, optional
        Apply :func:`correct_grouped_panel_origin` to the embedded IDF.
    detcal : str, optional
        DetCal calibration applied after loading (see
        :func:`detcal.apply_detcal`).

    Returns
    -------
    info : dict
        See :func:`read_lite` (without the counts array).
    """
    from mantid.simpleapi import AddSampleLog, mtd

    info = read_lite(filename)

    info["pixel_centres_corrected"] = False
    if correct_pixel_centres:
        info["idf"], info["pixel_centres_corrected"] = correct_grouped_panel_origin(info["idf"], info["group"])

    _counts_workspace(ws, info["counts"], idf_xml=info["idf"], instrument=info["instrument"])

    if detcal is not None:
        from .detcal import apply_detcal

        apply_detcal(ws, detcal, info["pixel_shape"])
        AddSampleLog(Workspace=ws, LogName="DetCal", LogText=detcal)

    if not np.allclose(info["goniometer"], np.eye(3)):
        mtd[ws].run().getGoniometer().setR(info["goniometer"])

    AddSampleLog(Workspace=ws, LogName="run_number", LogText=info["run_number"])
    AddSampleLog(Workspace=ws, LogName="Filename", LogText=filename)

    del info["counts"]

    return info


def load_cg4d_image(filename, ws="data", idf=None):
    """
    Workspace from a CG4D image-plate TIFF.

    The image is mapped column-major onto the detector IDs of
    ``CG4D_Definition.xml`` (5000 x 1800 pixels).
    """
    from PIL import Image
    from mantid.simpleapi import AddSampleLog

    idf = instrument_definition("CG4D_Definition.xml") if idf is None else idf

    counts = np.array(Image.open(filename), dtype=float).flatten(order="F")

    _counts_workspace(ws, counts, idf_file=idf)

    AddSampleLog(Workspace=ws, LogName="image", LogText=filename)

    return {"pixel_shape": (5000, 1800), "instrument": "CG4D"}
