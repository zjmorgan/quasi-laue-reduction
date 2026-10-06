"""
ISAW DetCal files from (calibrated) rectangular-panel geometry.

A DetCal entry describes a panel by its centre, two in-plane unit vectors
(``base`` along increasing column index, ``up`` along increasing row
index), its size and pixel counts, in centimetres, plus the source
distance L1. Mantid applies it with ``LoadIsawDetCal``.
"""

import time

import numpy as np


def panel_frames(positions):
    """
    Centre, base and up vectors, width and height of each panel.

    Parameters
    ----------
    positions : ndarray
        Pixel-centre positions with shape (n_banks, n_cols, n_rows, 3),
        first pixel index along the panel x (column) direction.

    Returns
    -------
    frames : dict
        ``centre``, ``base``, ``up`` with shape (n_banks, 3) and ``width``,
        ``height`` with shape (n_banks,), in metres.
    """
    positions = np.asarray(positions, dtype=float)
    _, nc, nr, _ = positions.shape

    centre = positions.mean(axis=(1, 2))

    base = positions[:, -1, :, :].mean(axis=1) - positions[:, 0, :, :].mean(axis=1)
    up = positions[:, :, -1, :].mean(axis=1) - positions[:, :, 0, :].mean(axis=1)

    width = np.linalg.norm(base, axis=1) * nc / (nc - 1)
    height = np.linalg.norm(up, axis=1) * nr / (nr - 1)

    base /= np.linalg.norm(base, axis=1)[:, None]
    up -= np.sum(up * base, axis=1)[:, None] * base
    up /= np.linalg.norm(up, axis=1)[:, None]

    return {"centre": centre, "base": base, "up": up, "width": width, "height": height}


def write_detcal(filename, positions, bank_numbers, l1=3.0, t0=0.0, pixels=None, depth=0.002, comment=""):
    """
    Write an ISAW DetCal file.

    Parameters
    ----------
    filename : str
        Output path.
    positions : ndarray
        Pixel-centre positions, (n_banks, n_cols, n_rows, 3), metres, with
        the sample at the origin.
    bank_numbers : sequence of int
        Detector numbers, e.g. 11 for ``bank11``.
    l1 : float, optional
        Source-sample distance in metres.
    t0 : float, optional
        Time-zero shift in microseconds.
    pixels : tuple, optional
        (n_cols, n_rows) to write, e.g. (512, 512) for the full-resolution
        instrument; defaults to the shape of ``positions``. The panel
        frame does not depend on the pixelation.
    depth : float, optional
        Panel depth in metres.
    comment : str, optional
        Extra comment line.
    """
    f = panel_frames(positions)
    nc, nr = positions.shape[1:3] if pixels is None else pixels

    lines = [
        "# NEW CALIBRATION FILE FORMAT (in NeXus/SNS coordinates):",
        "# Lengths are in centimeters.",
        "# Base and up give directions of unit vectors for a local",
        "# x,y coordinate system on the face of the detector.",
        "#",
        f"# Written by quasi_laue_reduction.detcal {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if comment:
        lines.append(f"# {comment}")
    lines += [
        "#",
        "6         L1     T0_SHIFT",
        f"7 {100 * l1:11.4f} {t0:12.3f}",
        "4 DETNUM  NROWS  NCOLS   WIDTH   HEIGHT   DEPTH   DETD   CenterX   CenterY   CenterZ    BaseX    BaseY    BaseZ      UpX      UpY      UpZ",
    ]

    for k, num in enumerate(bank_numbers):
        c = 100 * f["centre"][k]
        b, u = f["base"][k], f["up"][k]
        lines.append(
            f"5 {int(num):6d} {nr:6d} {nc:6d} {100 * f['width'][k]:7.4f} {100 * f['height'][k]:7.4f} {100 * depth:7.4f} "
            f"{np.linalg.norm(c):8.4f} {c[0]:9.4f} {c[1]:9.4f} {c[2]:9.4f} "
            f"{b[0]:8.5f} {b[1]:8.5f} {b[2]:8.5f} {u[0]:8.5f} {u[1]:8.5f} {u[2]:8.5f}"
        )

    with open(filename, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def read_detcal(filename):
    """
    Panel records of a DetCal file.

    Returns
    -------
    l1 : float
        Source-sample distance in metres.
    panels : dict
        bank number -> dict with ``centre``, ``base``, ``up`` (metres,
        unit vectors), ``nrows``, ``ncols``, ``width``, ``height``.
    """
    l1, panels = None, {}
    for line in open(filename):
        f = line.split()
        if not f or f[0] == "#":
            continue
        if f[0] == "7":
            l1 = float(f[1]) / 100
        elif f[0] == "5":
            v = [float(x) for x in f[1:]]
            panels[int(v[0])] = {
                "nrows": int(v[1]),
                "ncols": int(v[2]),
                "width": v[3] / 100,
                "height": v[4] / 100,
                "centre": np.array(v[7:10]) / 100,
                "base": np.array(v[10:13]),
                "up": np.array(v[13:16]),
            }
    return l1, panels


def apply_detcal(ws, filename, pixel_shape):
    """
    Move and rotate rectangular panels of a workspace to a DetCal file.

    Mantid's ``LoadIsawDetCal`` expects each ``bankNN`` component to be the
    rectangular detector itself; in the IMAGINE definition the detector is
    an unnamed ``panel`` inside a ``bankNN`` assembly, and
    ``LoadIsawDetCal`` silently changes nothing. Here each panel is rotated
    from its current (base, up, normal) frame to the file's, then
    translated so its pixel centroid lands on the file's centre.

    Parameters
    ----------
    ws : str
        Workspace with the instrument.
    filename : str
        DetCal file.
    pixel_shape : tuple
        Pixels per panel (n_cols, n_rows).

    Returns
    -------
    n_banks : int
        Number of panels moved.
    """
    from mantid.kernel import Quat, V3D
    from mantid.simpleapi import mtd

    _, panels = read_detcal(filename)

    workspace = mtd[ws]
    ci = workspace.componentInfo()
    di = workspace.detectorInfo()

    nc, nr = pixel_shape
    moved = 0

    for num, rec in panels.items():
        try:
            bank = int(ci.indexOfAny(f"bank{num}"))
        except Exception:
            continue

        # the rectangular detector: the bank itself or its single child
        kids = [int(k) for k in ci.children(bank)]
        panel = kids[0] if len(kids) == 1 and len(ci.children(kids[0])) > 0 else bank

        # detector indices follow detector IDs, which fill columns then rows
        dets = np.sort(np.array(ci.detectorsInSubtree(panel), dtype=int))
        pos = np.array([di.position(int(d)) for d in dets]).reshape(nc, nr, 3)
        cur = panel_frames(pos[None])

        old = np.column_stack([cur["base"][0], cur["up"][0], np.cross(cur["base"][0], cur["up"][0])])
        new = np.column_stack([rec["base"], rec["up"], np.cross(rec["base"], rec["up"])])
        R = new @ old.T

        q = Quat(V3D(*R[:, 0]), V3D(*R[:, 1]), V3D(*R[:, 2]))
        ci.setRotation(int(panel), q * ci.rotation(int(panel)))

        pos = np.array([di.position(int(d)) for d in dets])
        shift = rec["centre"] - pos.mean(axis=0)
        p = np.array(ci.position(int(panel))) + shift
        ci.setPosition(int(panel), V3D(*p))
        moved += 1

    return moved
