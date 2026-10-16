#!/usr/bin/env python3
"""TinyChat corpora: synthetic, generated, nothing downloaded.

    python corpus/fetchers/tinychat.py v2_1500     # quickstart "quick" preset
    python corpus/fetchers/tinychat.py v2_4000     # quickstart "better" preset
    python corpus/fetchers/tinychat.py v1_1500     # the transformer_cpp original
    python corpus/fetchers/tinychat.py all

The "pinned source" of a generated corpus is the generator itself:
tools/get_tinychat_data.py (v1) and tools/tinychat_v2.py (v2), with the
seed and dialogue count fixed per corpus id. A change to either generator
that alters a byte fails here loudly, which is the point: it would need a
new corpus id, and earlier results would no longer reproduce.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

ROOT = os.path.dirname(C.CORPUS_DIR)
GEN = os.path.join(ROOT, "tools", "get_tinychat_data.py")

# corpus id -> (generator version, dialogues, seed)
TARGETS = {
    "v1_1500": (1, 1500, 7),
    "v2_1500": (2, 1500, 11),
    "v2_4000": (2, 4000, 11),
}


def fetch(target, pin=False):
    version, n, seed = TARGETS[target]
    cid = f"tinychat_{target}"
    names = ["tinychat.txt", "tinychat_vocab.gguf"]
    if C.already_done(cid, names):
        print(f"{cid}: present and verified")
        return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, GEN, "--version", str(version), "--dialogues", str(n),
                        "--seed", str(seed), "--out-dir", tmp], check=True, stdout=subprocess.DEVNULL)
        outs = []
        for sub, name in (("data", "tinychat.txt"), ("releases", "tinychat_vocab.gguf")):
            dst = C.out_path(cid, name)
            shutil.copyfile(os.path.join(tmp, sub, name), dst)
            outs.append(dst)
    C.finish(cid, outs, f"tools/get_tinychat_data.py --version {version} --dialogues {n} --seed {seed}",
             pin=pin)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate pinned TinyChat corpora.")
    ap.add_argument("target", choices=sorted(TARGETS) + ["all"])
    ap.add_argument("--pin", action="store_true",
                    help="first generation of a new corpus id only: record its hashes")
    a = ap.parse_args(argv)
    try:
        for t in (sorted(TARGETS) if a.target == "all" else [a.target]):
            fetch(t, pin=a.pin)
    except C.HashMismatch as e:
        C.die(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
