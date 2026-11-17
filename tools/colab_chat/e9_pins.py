#!/usr/bin/env python3
"""E9's pins: a D arm's corpus files, checked as base 2's arms are, without touching base 2's pins.

    python3 tools/colab_chat/e9_pins.py DATA_DIR arm-D2 --upload /content/chat/upload --out OUT/pins.json
    python3 tools/colab_chat/e9_pins.py adopt-ledger STAGE_RECEIPT [--e9-pins e9_pins.json]

base2_pins.py and base2_pins.json are imported and read, never edited, so every ladder arm's pins job stays byte for byte
what it was. E9's own groups (D2, D3, D4, D4ops: `<ARM>.main.ids` and `<ARM>.tail.ids`) live in e9_pins.json; the panels
and the vocabulary are base 2's pinned ones. A mode `arm-<ARM>` checks the arm's two id files, base 2's panels and the
vocabulary, with base2_pins.check (the same rule: a missing file or a null pin refuses). `adopt-ledger` writes an arm's
two hashes from its build's stage receipt (e9_<ARM>_stage3.json: the published set's manifest, read back from the
store), once its corpus is built and published.

Claude Opus 5.5 (a fork of the coordinating session), 18 Nov 2026.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import base2_pins as P  # noqa: E402  (base 2's pins tool, unchanged)

E9_PINS = os.path.join(HERE, "e9_pins.json")
ARMS = ("D2", "D3", "D4", "D4ops")


def modes():
    return {f"arm-{a}": ([(a, "dir"), ("panels", "dir"), ("vocab", "upload")], False) for a in ARMS}


def merged(base2_path=None, e9_path=None):
    pins = json.load(open(P.pins_path(base2_path), encoding="utf-8"))
    e9 = json.load(open(e9_path or E9_PINS, encoding="utf-8"))
    for a in ARMS:
        if a in pins:
            raise SystemExit(f"base 2's pins already hold a group {a}; E9's would shadow it")
        pins[a] = e9[a]
    return pins


def _find_sha(x, name):
    """The sha256 the stage ledger records for a file called `name`, wherever it sits in the ledger."""
    if isinstance(x, dict):
        if name in x and isinstance(x[name], dict) and isinstance(x[name].get("sha256"), str):
            return x[name]["sha256"]
        for v in x.values():
            got = _find_sha(v, name)
            if got:
                return got
    elif isinstance(x, list):
        for v in x:
            got = _find_sha(v, name)
            if got:
                return got
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("mode")
    ap.add_argument("--upload", default="/content/chat/upload")
    ap.add_argument("--pins")
    ap.add_argument("--e9-pins")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.dir == "adopt-ledger":
        led = json.load(open(a.mode, encoding="utf-8"))
        path = a.e9_pins or E9_PINS
        e9 = json.load(open(path, encoding="utf-8"))
        arm = next((x for x in ARMS if _find_sha(led, f"{x}.main.ids")), None)
        if arm is None:
            raise SystemExit("the ledger names no E9 arm's id files")
        for part in ("main", "tail"):
            got = _find_sha(led, f"{arm}.{part}.ids")
            if not got or len(got) != 64:
                raise SystemExit(f"the ledger has no hash for {arm}.{part}.ids")
            if e9[arm][f"{arm}.{part}.ids"] not in (None, got):
                raise SystemExit(f"{arm}.{part}.ids is pinned to other bytes; nothing overwritten")
            e9[arm][f"{arm}.{part}.ids"] = got
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(e9, indent=1) + "\n")
        print(json.dumps({"adopted": arm}))
        return 0
    m = modes()
    if a.mode not in m:
        raise SystemExit(f"mode is one of {', '.join(sorted(m))}")
    pins = merged(a.pins, a.e9_pins)
    saved = dict(P.MODES)
    try:
        P.MODES.update(m)                     # in this process only
        bad, seen = P.check(pins, a.dir, a.mode, a.upload)
    finally:
        P.MODES.clear()
        P.MODES.update(saved)
    rec = {a.mode: {"observed": seen, "differences": bad}}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
    if bad:
        raise SystemExit("not the pinned bytes:\n" + "\n".join(bad))
    print(json.dumps({"pins": "equal", "mode": a.mode, "files": len(seen)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
