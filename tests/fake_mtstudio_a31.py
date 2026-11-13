#!/usr/bin/env python3
"""A stand-in for mtstudio in A31's scheduler simulation (tests/test_a31_schedule.py). Claude Opus 5.5, 13 Nov 2026. It writes what
jobs_vm.py, the gate, the early readouts, the selection and the scorer read, quickly, and fails on request.

  run SPEC          events (steps, evals every eval_every, twin records when the spec has train.twin_loss), state.txt and
                    model.safetensors at every checkpoint_every (pausing there so the manager can snapshot), then the final
                    <name>.safetensors, spec.json and result.json. Exits 1 at step 200 when the spec's name is in FAKE_FAIL.
                    Held-out SFT loss 4.0 + 0.1 * lambda; FAKE_HIGH_VAL names jobs whose loss at step 256 is 5.0.
  score DIR --in REQ --out OUT ...   one record per request, {"i", "sum", "logprobs"}, own-context rows scored higher.
"""
import json
import os
import sys
import time


def run(spec_path):
    s = json.load(open(spec_path))
    name, out = s.get("name", "mtstudio"), s["out_dir"]
    os.makedirs(out, exist_ok=True)
    tr = s["train"]
    steps, ck, ev_every = tr["steps"], tr["checkpoint_every"], tr["eval_every"]
    tl = tr.get("twin_loss")
    lam = tl["lambda"] if tl else 0.0
    fail = name in os.environ.get("FAKE_FAIL", "").split(",")
    high = name in os.environ.get("FAKE_HIGH_VAL", "").split(",")
    slow = name in os.environ.get("FAKE_SLOW", "").split(",")   # trains slowly past step 256, so a health stop finds it alive
    json.dump(s, open(os.path.join(out, "spec.json"), "w"))
    with open(os.path.join(out, "events.jsonl"), "a") as f:
        for step in range(1, steps + 1):
            if fail and step == 200:
                return 1
            e = {"event": "step", "step": step, "loss": 3.0 - 0.5 * step / steps, "grad_norm": 3.0 + (0.5 if tl else 0.0),
                 "sft_rows_hash": step}
            if tl:
                e["twin"] = {"loss": 0.5, "d_a": 0.1, "d_b": 0.2, "d_n": 0.05, "active": 0.9, "len": 100}
            f.write(json.dumps(e) + "\n")
            if step % ev_every == 0 or step == steps:
                v = 5.0 if high and step == 256 else 4.0 + 0.1 * lam
                f.write(json.dumps({"event": "eval", "step": step, "val_loss": v}) + "\n")
            f.flush()
            if step % ck == 0:
                open(os.path.join(out, "model.safetensors"), "w").write(f"weights {name} {step}")
                open(os.path.join(out, "state.txt"), "w").write(f"{step}\n")
                time.sleep(float(os.environ.get("FAKE_CKPT_PAUSE", "0.6")))
            if slow and step > 256:
                time.sleep(0.1)
    open(os.path.join(out, f"{name}.safetensors"), "w").write(f"weights {name} final")
    if "gguf" in (s.get("export") or {}).get("formats", []):
        open(os.path.join(out, f"{name}.gguf"), "w").write(f"gguf {name} final")
    json.dump({"final_step": steps, "steps_requested": steps, "name": name}, open(os.path.join(out, "result.json"), "w"))
    return 0


def score(args):
    req = args[args.index("--in") + 1]
    out = args[args.index("--out") + 1]
    n = sum(1 for x in open(req) if x.strip())
    with open(out, "w") as f:
        for i in range(n):
            k = i % 5
            s = -2.0 if k in (0, 2, 4) else -3.0
            f.write(json.dumps({"i": i, "sum": s, "logprobs": [s / 2, s / 2], "over_ctx": False}) + "\n")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    sys.exit(run(sys.argv[2]) if cmd == "run" else score(sys.argv[2:]) if cmd == "score" else 2)
