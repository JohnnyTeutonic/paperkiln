#!/usr/bin/env python3
"""W169's two halts on the VM (the author's research tree, AI_ML/chat_screens_claude/base2/w169/REGISTRATION.md section 4a).
Claude Opus 5.5, 26 Nov 2026. Each runs as a light job with "halt_on_failure", so an exit 1 halts the campaign: the runner
then stops the session and bills nothing more.

    python3 tools/colab_chat/w169_guard_vm.py loss --chat /content/chat --job pt_base2_W169 --out {out} [--interval 30]
        Reads the trainer's own event stream (<chat>/<job>/events.jsonl) as it grows. Exit 1 at the first `step` event whose
        loss is missing, null or not finite (the trainer writes a NaN loss as null). Exit 0 when the job's STATUS file reads
        FINISHED. A trainer FAILED is the training job's own halt; this one then exits 0.
    python3 tools/colab_chat/w169_guard_vm.py diverge --rung <chat>/rung_W169_s10000/rung.json --out {out}
        Exit 1 if the rung's FineWeb panel is missing, not finite, or above N's own step-10,000 FineWeb panel
        (N_FINEWEB_10K, from C:/ml_artifacts/base2/arm_N/rung_N_s10000/rung.json) by more than MARGIN nats.
Both write OUT/guard.json with what they read.
"""
import argparse
import json
import math
import os
import sys
import time

N_FINEWEB_10K = 3.750180445611477   # arm N's step-10,000 rung, panels.fineweb (weights fcc483bd...)
MARGIN = 0.10


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def scan(path, pos):
    """New complete lines of `path` from byte `pos`: (step events whose loss is not finite, last step seen, new pos)."""
    bad, last = [], None
    try:
        with open(path, "rb") as f:
            f.seek(pos)
            data = f.read()
    except OSError:
        return bad, last, pos
    end = data.rfind(b"\n")
    if end < 0:
        return bad, last, pos
    for ln in data[:end].split(b"\n"):
        if b'"step"' not in ln:
            continue
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if not isinstance(e, dict) or e.get("event") != "step":
            continue
        last = e.get("step")
        if not finite(e.get("loss")):
            bad.append({"step": e.get("step"), "loss": e.get("loss")})
    return bad, last, pos + end + 1


def write(out, rec):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "guard.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)


def loss(a):
    events = os.path.join(a.chat, a.job, "events.jsonl")
    pos, last, t0 = 0, None, time.time()
    while True:
        bad, step, pos = scan(events, pos)
        last = step if step is not None else last
        if bad:
            write(a.out, {"halt": "NONFINITE_LOSS", "first": bad[0], "count": len(bad), "at": time.time()})
            print(f"NONFINITE_LOSS at step {bad[0]['step']}: halting", flush=True)
            return 1
        try:
            status = open(os.path.join(a.chat, f"STATUS_{a.job}")).read().strip()
        except OSError:
            status = "NONE"
        if status in ("FINISHED", "FAILED"):
            write(a.out, {"halt": None, "status": status, "last_step": last, "seconds": time.time() - t0})
            return 0
        time.sleep(a.interval)


def diverge(a):
    try:
        r = json.load(open(a.rung))
        fw = (r.get("panels") or {}).get("fineweb")
    except (OSError, ValueError):
        fw = None
    rec = {"fineweb": fw, "n_fineweb_10k": N_FINEWEB_10K, "margin": MARGIN}
    if not finite(fw) or fw > N_FINEWEB_10K + MARGIN:
        write(a.out, {**rec, "halt": "DIVERGED_AT_10K"})
        print(f"DIVERGED_AT_10K: FineWeb {fw} against N's {N_FINEWEB_10K:.4f} + {MARGIN}", flush=True)
        return 1
    write(a.out, {**rec, "halt": None, "minus_n": fw - N_FINEWEB_10K})
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("loss")
    p.add_argument("--chat", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=30)
    q = sub.add_parser("diverge")
    q.add_argument("--rung", required=True)
    q.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return loss(a) if a.mode == "loss" else diverge(a)


if __name__ == "__main__":
    sys.exit(main())
