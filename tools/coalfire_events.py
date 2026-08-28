#!/usr/bin/env python3
"""coalfire.cpp -> events.jsonl adapter (docs/ECOSYSTEM.md workstream C1).

coalfire's trainer writes a human log ("Step 40 | ... | Loss: 5.1234 |
PPL: ..."). This sidecar translates it into the paperkiln events.jsonl
contract, which buys a coalfire run the whole contract-2 toolchain for
free: `mtstudio serve <out>` tails it in the studio dashboard,
tools/atlas_extract.py turns it into an Atlas row, and mtsweep can
aggregate it.

    python tools/coalfire_events.py train.log --out /tmp/coalfire_run
    python tools/coalfire_events.py train.log --out DIR --follow   # live
    python tools/coalfire_events.py --selftest

Sidecar BY DESIGN (not a patch to coalfire's train.cpp): the adapter
ships while coalfire's tree carries in-flight work, and native emission
can replace it later without changing any consumer — that is the point
of contracts. Unknown log lines are ignored, never guessed at.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

STEP_RE = re.compile(
    r"^Step\s+(\d+)\s+\|.*?\|\s*Loss:\s*([0-9.]+)\s*\|\s*PPL:\s*([0-9.eE+]+)")
VAL_AT_RE = re.compile(r"^\*\*\* Validation at step (\d+) \*\*\*")
VAL_LOSS_RE = re.compile(
    r"^Val Loss:\s*([0-9.]+)\s*\|\s*Val Perplexity:\s*([0-9.eE+]+)")
BEST_RE = re.compile(r"^Best Validation Loss:\s*([0-9.]+)")
EARLY_RE = re.compile(r"\[no improvement (\d+)/(\d+)\]")


class Translator:
    def __init__(self, emit, name="coalfire-run"):
        self.emit = emit
        self.name = name
        self.started = False
        self.last_step = 0
        self.pending_val_step = None
        self.done = False

    def _start(self):
        if not self.started:
            self.started = True
            self.emit({"event": "start", "name": self.name, "engine": "coalfire"})

    def line(self, raw):
        s = raw.strip()
        if not s:
            return
        m = STEP_RE.match(s)
        if m:
            self._start()
            self.last_step = int(m.group(1))
            self.emit({"event": "step", "step": self.last_step,
                       "loss": float(m.group(2)), "ppl": float(m.group(3))})
            return
        m = VAL_AT_RE.match(s)
        if m:
            self.pending_val_step = int(m.group(1))
            return
        m = VAL_LOSS_RE.match(s)
        if m:
            self._start()
            step = self.pending_val_step or self.last_step
            self.pending_val_step = None
            ev = {"event": "eval", "step": step, "val_loss": float(m.group(1))}
            em = EARLY_RE.search(s)
            if em and em.group(1) == em.group(2):
                self.emit(ev)
                self.emit({"event": "early_stop", "step": step})
                return
            self.emit(ev)
            return
        m = BEST_RE.match(s)
        if m and not self.done:
            self.done = True
            self.emit({"event": "done", "best_val": float(m.group(1)),
                       "final_step": self.last_step})


def translate(lines, out_path, name):
    written = []

    def emit(ev):
        written.append(ev)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")

    tr = Translator(emit, name)
    for line in lines:
        tr.line(line)
    return tr, written


SAMPLE = """\
Epochs: 3
Step 40 | Batch [40/512 = 7.8%] | Loss: 5.1234 | PPL: 167.94 | GPU: 3.20/8.00GB | Time: 412.0ms | ETA: 3m14s
Step 80 | Batch [80/512 = 15.6%] | Loss: 4.6110 | PPL: 100.59 | GPU: 3.20/8.00GB | Time: 405.2ms | Sequences: 8 | ETA: 2m58s

*** Validation at step 80 ***
Val Loss: 4.7001 | Val Perplexity: 109.96 [NEW BEST!]
Step 120 | Batch [120/512 = 23.4%] | Loss: 4.3007 | PPL: 73.75 | GPU: 3.20/8.00GB | Time: 401.9ms | ETA: 2m41s
Val Loss: 4.7immaterial garbage line that must be ignored
*** Validation ***
Val Loss: 4.8120 | Val Perplexity: 122.97 [no improvement 2/2]
Best Validation Loss: 4.7001
"""


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "events.jsonl")
        _, evs = translate(SAMPLE.splitlines(), out, "selftest")
        kinds = [e["event"] for e in evs]
        assert kinds == ["start", "step", "step", "eval", "step", "eval",
                         "early_stop", "done"], kinds
        assert evs[1] == {"event": "step", "step": 40, "loss": 5.1234,
                          "ppl": 167.94}
        assert evs[3]["step"] == 80 and evs[3]["val_loss"] == 4.7001
        # the second eval had no "at step" header: falls back to last step
        assert evs[5]["step"] == 120 and evs[5]["val_loss"] == 4.8120
        assert evs[7] == {"event": "done", "best_val": 4.7001,
                          "final_step": 120}
        # the file is valid JSONL and atlas_extract can feature it
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        assert len(rows) == 8
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from atlas_extract import behavioural_features
        feats = behavioural_features(rows)
        assert feats["best_val"] == 4.7001
        assert feats["final_train_loss"] > 0
    print("SELFTEST OK: start/step/eval/early_stop/done from verbatim "
          "coalfire log shapes; garbage ignored; atlas_extract features "
          f"the result (best_val={4.7001})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--out", help="output DIR (events.jsonl inside)")
    ap.add_argument("--name")
    ap.add_argument("--follow", action="store_true",
                    help="keep tailing the log (live dashboard mode)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.log or not args.out:
        ap.error("need LOG and --out DIR (or --selftest)")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "events.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)
    name = args.name or os.path.splitext(os.path.basename(args.log))[0]

    def emit(ev):
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")

    tr = Translator(emit, name)
    with open(args.log, encoding="utf-8", errors="replace") as f:
        for line in f:
            tr.line(line)
        while args.follow and not tr.done:
            line = f.readline()
            if line:
                tr.line(line)
            else:
                time.sleep(1.0)
    print(f"events -> {out_path}  (serve live: ./mtstudio serve {args.out} 8080)")


if __name__ == "__main__":
    main()
