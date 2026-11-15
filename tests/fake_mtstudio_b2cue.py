#!/usr/bin/env python3
"""A stand-in for mtstudio in A32's scheduler simulation (tests/test_b2cue_schedule.py). Claude Opus 5.5, 15 Nov 2026, from A31's
tests/fake_mtstudio_a31.py. It writes what jobs_vm.py, the gate, the early readouts, the watch, the selection and the binding read,
quickly, and fails on request.

  run SPEC   events (steps; evals every eval_every and at train.eval_at; twin records, with answer_mask when the spec asks), state.txt
             and model.safetensors at every checkpoint_every (pausing so the manager can snapshot), then <name>.safetensors, spec.json
             and result.json. Exits 1 at step 200 when the spec's name is in FAKE_FAIL. Held-out SFT loss 4.0 + 0.1 * lambda; FAKE_HIGH_VAL
             names jobs whose loss at step 256 is 5.0; FAKE_NAN names jobs whose step-20 SFT loss is NaN (written as null).
  score DIR --in REQ --out OUT ...   one record per request: {"i", "sum", "logprobs", "continuation_ids", "over_ctx"}; own-context rows
             (AA, BB) score -2 + 0.1 * lambda over two tokens, crossed rows -3, the nuisance row -2.05 + 0.1 * lambda (so the
             nuisance shift is lambda-free and dec^D rises with lambda; lambda from DIR/spec.json).
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
    at = set(tr.get("eval_at", []))
    tl = tr.get("twin_loss")
    lam = tl["lambda"] if tl else 0.0
    env = lambda k: os.environ.get(k, "").split(",")  # noqa: E731
    fail, high, slow, nan = name in env("FAKE_FAIL"), name in env("FAKE_HIGH_VAL"), name in env("FAKE_SLOW"), name in env("FAKE_NAN")
    json.dump(s, open(os.path.join(out, "spec.json"), "w"))
    with open(os.path.join(out, "events.jsonl"), "a") as f:
        for step in range(1, steps + 1):
            if fail and step == 200:
                return 1
            loss = None if (nan and step == 20) else 3.0 - 0.5 * step / steps
            e = {"event": "step", "step": step, "loss": loss, "grad_norm": 3.0 + (0.5 if tl else 0.0), "sft_rows_hash": step}
            if tl:
                e["twin"] = {"loss": 0.5, "d_a": 0.1, "d_b": 0.2, "d_n": 0.05, "active": 0.9, "len": 100}
                if tl.get("answer_mask"):
                    e["twin"]["answer_mask"] = True
            f.write(json.dumps(e) + "\n")
            if step % ev_every == 0 or step == steps or step in at:
                v = 5.0 if high and step == 256 else 4.0 + 0.1 * lam
                f.write(json.dumps({"event": "eval", "step": step, "val_loss": v}) + "\n")
            f.flush()
            if step % ck == 0:
                open(os.path.join(out, "model.safetensors"), "w").write(f"weights {name} {step}")
                open(os.path.join(out, "state.txt"), "w").write(f"{step}\n")
                time.sleep(float(os.environ.get("FAKE_CKPT_PAUSE", "0.3")))
            if (slow or nan) and step > 20:
                time.sleep(0.02)
    open(os.path.join(out, f"{name}.safetensors"), "w").write(f"weights {name} final")
    if "gguf" in (s.get("export") or {}).get("formats", []):
        open(os.path.join(out, f"{name}.gguf"), "w").write(f"gguf {name} final")
    json.dump({"final_step": steps, "steps_requested": steps, "name": name}, open(os.path.join(out, "result.json"), "w"))
    return 0


def score(args):
    d = args[0]
    req = args[args.index("--in") + 1]
    out = args[args.index("--out") + 1]
    lam = 0.0
    try:
        tl = json.load(open(os.path.join(d, "spec.json")))["train"].get("twin_loss")
        lam = tl["lambda"] if tl else 0.0
    except (OSError, ValueError, KeyError):
        pass
    n = sum(1 for x in open(req) if x.strip())
    with open(out, "w") as f:
        for i in range(n):
            k = i % 5
            s = -2.0 + 0.1 * lam if k in (0, 2) else -2.05 + 0.1 * lam if k == 4 else -3.0
            ids = [7, 8] if k in (0, 1, 4) else [9, 8]
            f.write(json.dumps({"i": i, "sum": s, "logprobs": [s / 2, s / 2], "continuation_ids": ids, "over_ctx": False}) + "\n")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    sys.exit(run(sys.argv[2]) if cmd == "run" else score(sys.argv[2:]) if cmd == "score" else 2)
