#!/usr/bin/env python3
"""A31: the identity of a score file (lane PROTOCOL.md section 50; round-2 reviews, 13 Nov 2026). Written beside every final and
sweep score file on the pod: the sha256 of the weights scored, of the request file, of the mtstudio binary, and of the spec the
scoring read, with the spec itself. The scorer at home requires the weights' hash to equal the home copy's and the request
file's to equal the registered pin. Claude Opus 5.5, 13 Nov 2026.

    python3 tools/colab_chat/a31_bind.py --weights W --requests R --bin B --spec S --out score.json
"""
import argparse
import hashlib
import json
import os
import sys


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in ("--weights", "--requests", "--bin", "--spec", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    out = {"weights": os.path.basename(a.weights), "weights_sha256": sha(a.weights), "requests": os.path.basename(a.requests),
           "requests_sha256": sha(a.requests), "binary_sha256": sha(a.bin), "spec_sha256": sha(a.spec), "spec": json.load(open(a.spec))}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
