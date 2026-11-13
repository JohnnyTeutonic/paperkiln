#!/usr/bin/env python3
"""A31: the identity of a score file (lane PROTOCOL.md section 50; launch reviews of 13 Nov 2026). Run on the pod AFTER a successful
`mtstudio score`, it writes score.json beside the scores: the sha256 of the scores file itself, of the weights scored, of the
request file, of the mtstudio binary and of the spec the scoring read (with the spec), and, when the job's directory holds one, of
its GGUF (the same training output as the weights, so the guards' models are bound to the scored weights). The scorer at home
requires the scores', weights' and GGUF's hashes to equal the home copies', the request file's to equal the registered pin, the spec
to equal the registered one, and one binary across the deciding scores. Claude Opus 5.5, 13 Nov 2026.

    python3 tools/colab_chat/a31_bind.py --scores S --weights W --requests R --bin B --spec SPEC [--gguf G] --out score.json
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
    for k in ("--scores", "--weights", "--requests", "--bin", "--spec", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--gguf")
    a = ap.parse_args()
    if not os.path.getsize(a.scores):
        sys.exit("the scores file is empty: nothing is bound")
    out = {"scores_sha256": sha(a.scores), "weights": os.path.basename(a.weights), "weights_sha256": sha(a.weights),
           "requests": os.path.basename(a.requests), "requests_sha256": sha(a.requests), "binary_sha256": sha(a.bin),
           "spec_sha256": sha(a.spec), "spec": json.load(open(a.spec))}
    if a.gguf and os.path.exists(a.gguf):
        out["gguf_sha256"] = sha(a.gguf)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
