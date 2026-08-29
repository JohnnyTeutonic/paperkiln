#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sync/verify the paperkiln-fetch package's vendored extractor.

    python tools/sync_fetch_pkg.py          # copy papers/fetch.py into the pkg
    python tools/sync_fetch_pkg.py --check  # exit 1 if the two have drifted

Source of truth is papers/fetch.py. The pip package vendors it verbatim
(same file, module name _fetch) so `pip install paperkiln-fetch` needs
no repo checkout and no build step. Same discipline as sync_vendor.sh:
one owner, one copy direction, drift is an error not a merge.
"""
import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "papers", "fetch.py")
DST = os.path.join(REPO, "paperkiln_fetch", "src", "paperkiln_fetch",
                   "_fetch.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    with open(SRC, encoding="utf-8") as f:
        src = f.read()
    dst = ""
    if os.path.exists(DST):
        with open(DST, encoding="utf-8") as f:
            dst = f.read()
    if args.check:
        if src != dst:
            print("DRIFT: papers/fetch.py != paperkiln_fetch/_fetch.py — "
                  "run python tools/sync_fetch_pkg.py")
            return 1
        print("in sync")
        return 0
    shutil.copyfile(SRC, DST)
    print(f"synced {SRC} -> {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
