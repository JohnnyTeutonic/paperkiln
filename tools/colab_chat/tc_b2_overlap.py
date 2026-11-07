#!/usr/bin/env python3
"""Codex's counts-only overlap scan of the final mix, and the gate on its reviewed receipt (PROTOCOL.md section 30). Claude Opus 5.5, 7 Nov 2026.

    python3 tools/colab_chat/tc_b2_overlap.py scan --data /content/chat/data --packet KEYED.json --out OUT
    python3 tools/colab_chat/tc_b2_overlap.py check --data /content/chat/data --decision DECISION.json --out OUT

scan ties each part's text to its ids through the data job's own ledgers, so the text scanned is the text the ids came from, then runs
codex_overlap_worker.py unchanged. It reports counts and passes no judgement: the worker's receipt says COMPLETE_REQUIRES_CODEX_REVIEW.
check runs once Codex's reviewed decision reaches the VM (jobs_vm.py holds it on `promoted_by`): the decision must name both final mix
hashes, or training does not start. Keyed fingerprints only; no target text or case id is read or written here.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pins():
    return json.load(open(os.path.join(HERE, "tc_b1_pins.json")))["mix"]


def manifest(d):
    led = json.load(open(os.path.join(d, "mix_b1.ledger.json")))["parts"]
    want = pins()
    rows = []
    for part in ("main", "tail"):
        text, ids = os.path.join(d, f"mix_b1.{part}.txt"), os.path.join(d, f"mix_b1.{part}.ids")
        t, i = sha256(text), sha256(ids)
        if t != led[part]["sha256"] or i != want[f"mix_b1.{part}.ids"]:
            raise SystemExit(f"{part}: the text or ids is not the data job's")
        rows.append({"name": part, "text": text, "text_sha256": t, "ids": ids, "ids_sha256": i})
    return rows


def scan(a):
    os.makedirs(a.out, exist_ok=True)
    m = os.path.join(a.out, "manifest.json")
    if not os.path.exists(m):
        with open(m + ".tmp", "w") as f:
            json.dump(manifest(a.data), f, indent=1)
        os.replace(m + ".tmp", m)
    out = os.path.join(a.out, "overlap_receipt.json")
    if os.path.exists(out):
        print(f"{out}: present")
        return 0
    return subprocess.run([sys.executable, os.path.join(HERE, "codex_overlap_worker.py"), "--packet", a.packet, "--manifest", m, "--out", out]).returncode


def check(a):
    d = json.load(open(a.decision))
    want = pins()
    bad = [k for k, v in want.items() if d.get("mix", {}).get(k) != v]
    if d.get("decisions", {}).get("b2", {}).get("go") is not True:
        bad.append("no go")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "overlap_check.json"), "w") as f:
        json.dump({"mix": want, "decision_sha256": sha256(a.decision), "bad": bad}, f, indent=1)
    if bad:
        raise SystemExit(f"the decision does not cover this mix: {bad}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--data", required=True)
    p.add_argument("--packet", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check")
    p.add_argument("--data", required=True)
    p.add_argument("--decision", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return scan(a) if a.cmd == "scan" else check(a)


if __name__ == "__main__":
    sys.exit(main())
