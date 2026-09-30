"""
Loading a lite event file: Mantid LoadNexus of the full event list vs
summing event weights per pixel straight from HDF5.
"""

import argparse
import os
import time

from common import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("filename")
    p.add_argument("--skip-loadnexus", action="store_true")
    args = p.parse_args()

    rec = {"file": args.filename, "size_GB": os.path.getsize(args.filename) / 1e9}

    from quasi_laue_reduction.loading import load_lite

    t = time.perf_counter()
    load_lite(args.filename, ws="fast")
    rec["load_lite_s"] = time.perf_counter() - t

    if not args.skip_loadnexus:
        from mantid.simpleapi import LoadNexus

        t = time.perf_counter()
        LoadNexus(Filename=args.filename, OutputWorkspace="events")
        rec["LoadNexus_s"] = time.perf_counter() - t

    save("loading", rec)


if __name__ == "__main__":
    main()
