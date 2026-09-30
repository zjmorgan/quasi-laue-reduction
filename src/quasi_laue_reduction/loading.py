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


def pixel_shape_from_idf(xml):
    """
    (xpixels, ypixels) of the rectangular panels in an IDF string.
    """
    xs = set(re.findall(r'xpixels="(\d+)"', xml))
    ys = set(re.findall(r'ypixels="(\d+)"', xml))

    if len(xs) != 1 or len(ys) != 1:
        raise ValueError("Expected a single rectangular panel type in the IDF.")

    return int(xs.pop()), int(ys.pop())


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
        ``title`` and ``goniometer`` (3x3 rotation, identity if absent).
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


def load_lite(filename, ws="data"):
    """
    One-bin Workspace2D of per-pixel counts from a lite event NeXus file.

    Returns
    -------
    info : dict
        See :func:`read_lite` (without the counts array).
    """
    from mantid.simpleapi import AddSampleLog, mtd

    info = read_lite(filename)

    _counts_workspace(ws, info["counts"], idf_xml=info["idf"], instrument=info["instrument"])

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
