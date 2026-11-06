#!/usr/bin/env python3
"""The stronger base's data pins (AI_ML/tiny_chat_benchmarks/stronger_base_spec.md, section 12). Claude Opus 5.5, 7 Nov 2026.

    python3 tools/colab_chat/tc_b1_pins.py DIR stage0|b1
    python3 tools/colab_chat/tc_b1_pins.py DIR sources          # before the mix: the source files alone

Reads tools/colab_chat/tc_b1_pins.json: every source file's sha256 (from Stage 1, Colab CPU), S-full's outputs that the first
command reproduces, and, once Stage 0 has run, mix_b1's two .ids files. Hashes DIR's files, writes DIR/tc_b1_pins_observed.json
and exits non-zero on any difference. Stage 0 may run with the mix pins null, and records them; B refuses without them.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check(pins, d, mode):
    """(differences, observed). mode sources stops before the mix's pins."""
    bad, seen = [], {}
    for rel, want in sorted(pins["files"].items()):
        p = os.path.join(d, rel)
        got = sha256(p) if os.path.exists(p) else None
        seen[rel] = got
        if got != want:
            bad.append(f"{rel}: {got} against {want}")
    for rel, want in sorted({} if mode == "sources" else pins["mix"].items()):
        p = os.path.join(d, rel)
        got = sha256(p) if os.path.exists(p) else None
        seen[rel] = got
        if got is None:
            bad.append(f"{rel}: missing")
        elif want is None:
            if mode != "stage0":
                bad.append(f"{rel}: no Stage 0 pin; B refuses")
        elif got != want:
            bad.append(f"{rel}: {got} against Stage 0's {want}")
    return bad, seen


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    d, mode = a[0], a[1]
    if mode not in ("stage0", "b1", "sources"):
        raise SystemExit("mode is stage0, b1 or sources")
    pins = json.load(open(os.path.join(HERE, "tc_b1_pins.json")))
    bad, seen = check(pins, d, mode)
    tmp = os.path.join(d, "tc_b1_pins_observed.json.tmp")
    with open(tmp, "w") as f:
        json.dump({"mode": mode, "observed": seen, "differences": bad}, f, indent=1)
    os.replace(tmp, os.path.join(d, "tc_b1_pins_observed.json"))
    if bad:
        raise SystemExit("not the pinned bytes:\n" + "\n".join(bad))
    print(json.dumps({"pins": "equal", "mode": mode, "files": len(seen)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
