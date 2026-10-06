#!/usr/bin/env python3
"""Turn `mtstudio run`'s JSON event stream into a progress line a person can read.

    ./build/mtstudio run spec.json | python3 tools/train_progress.py spec.json

Non-JSON lines pass through untouched. Step events become one line per
`--every` steps (step, loss, steps per second, time left); evaluation and
end-of-run events are printed as they arrive. The full stream is still in
the run's events.jsonl.
"""
from __future__ import annotations

import json
import sys
import time


def main(argv):
    total, every = 0, 50
    args = list(argv[1:])
    if "--every" in args:
        i = args.index("--every"); every = int(args[i + 1]); del args[i:i + 2]
    if args:
        try:
            with open(args[0], encoding="utf-8") as f:
                total = int(json.load(f).get("train", {}).get("steps", 0))
        except (OSError, ValueError):
            pass
    start = time.time()
    for line in sys.stdin:
        s = line.strip()
        if not s.startswith("{"):
            print(line, end="", flush=True)
            continue
        try:
            ev = json.loads(s)
        except ValueError:
            print(line, end="", flush=True)
            continue
        kind = ev.get("event")
        if kind == "step":
            step = int(ev.get("step", 0))
            if step % every and step != total:
                continue
            rate = step / max(1e-9, time.time() - start)
            left = f", about {(total - step) / rate / 60:.0f} min left" if total and rate > 0 else ""
            of = f"/{total}" if total else ""
            print(f"  step {step}{of}  loss {ev.get('loss', float('nan')):.3f}  ({rate:.1f} steps/s{left})", flush=True)
        elif kind == "eval":
            val = ev.get("val_loss", ev.get("loss"))
            print(f"  validation loss {val:.3f} at step {ev.get('step', '?')}" if isinstance(val, (int, float))
                  else f"  evaluation: {s}", flush=True)
        elif kind in ("early_stop", "done", "finish", "export", "end"):
            print(f"  {kind}: " + ", ".join(f"{k}={v}" for k, v in ev.items() if k != "event"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
