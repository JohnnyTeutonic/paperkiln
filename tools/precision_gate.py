#!/usr/bin/env python3
"""Gates for train.precision and train.attention_kernel (24 Oct 2026), on real text.

    python3 tools/precision_gate.py run --new BIN [--old OLD_BIN] --text TEXT --vocab VOCAB.gguf --out DIR
        [--shape 20m_L8 | --custom D,L,H,DFF] [--batch B] [--accum A] [--T T] [--steps N] [--text-bytes N] [--cpu]

Runs mtstudio on the same spec (seeded; one eval at the end; safetensors + F32 GGUF export) as
legs, and writes DIR/gate.json:
  off_bitwise   OLD_BIN vs BIN, the spec without the new keys: every step's loss and grad norm, the
                final validation loss and the exported files' sha256 identical (the registered arms
                reproduce from HEAD with precision off). Only with --old.
  off_explicit  BIN with "precision": "fp32", "attention_kernel": "tiled" vs BIN without the keys.
  repeat        BIN twice: run-to-run determinism (a precondition of the two above on a GPU).
  flash         attention_kernel flash vs tiled; bf16: precision bf16 vs fp32; bf16_flash: both.
                Per-step losses compared raw and as a 20-step moving average, the final
                validation loss, and seconds per step.
Tolerances (stated before any run): bitwise legs 0 differences; flash moving average within 0.5%
relative and final validation within 0.5%; bf16 and bf16_flash moving average within 2% and final
validation within 1%. Every run must record its setting (model event, result.json, GGUF key
paperkiln.train.precision for bf16). --cpu runs the host path (MICROTORCH_DEVICE=cpu); otherwise
the production device environment (tools/colab_chat/jobs_vm.py ENV).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step_profile  # noqa: E402

TOL = {"flash": (5e-3, 5e-3), "bf16": (2e-2, 1e-2), "bf16_flash": (2e-2, 1e-2)}  # (moving average, final val)
LEGS = {"flash": {"attention_kernel": "flash"}, "bf16": {"precision": "bf16"},
        "bf16_flash": {"precision": "bf16", "attention_kernel": "flash"}}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gguf_has_key(path, key):
    with open(path, "rb") as f:
        head = f.read(1 << 22)
    return key.encode() in head


def run(binary, spec, run_dir, env):
    os.makedirs(os.path.dirname(run_dir.rstrip("/")) or ".", exist_ok=True)
    spec = dict(spec, out_dir=run_dir)
    sp = run_dir.rstrip("/") + "_spec.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    t_first = t_last = None
    s_first = s_last = None
    with open(run_dir.rstrip("/") + ".log", "w", encoding="utf-8") as log:
        p = subprocess.Popen([binary, "run", sp], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             bufsize=1, env=env)
        for line in p.stdout:
            log.write(line)
            if line.startswith('{"event":"step"'):
                try:
                    st = json.loads(line)["step"]
                except (ValueError, KeyError):
                    continue
                now = time.monotonic()
                if st >= 10 and t_first is None:
                    t_first, s_first = now, st
                t_last, s_last = now, st
        rc = p.wait()
    evs = step_profile.events(run_dir)
    steps = [e for e in evs if e.get("event") == "step"]
    evals = [e for e in evs if e.get("event") == "eval"]
    model = next((e for e in evs if e.get("event") == "model"), {})
    res = {}
    rp = os.path.join(run_dir, "result.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            res = json.load(f)
    files = {}
    for fn in sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else []:
        if fn.endswith((".safetensors", ".gguf")):
            files[fn] = sha(os.path.join(run_dir, fn))
    gguf = [os.path.join(run_dir, f) for f in files if f.endswith(".gguf")]
    return {"rc": rc, "loss": [e.get("loss") for e in steps], "grad_norm": [e.get("grad_norm") for e in steps],
            "val": evals[-1].get("val_loss") if evals else None, "files": files,
            "model": {k: model.get(k) for k in ("precision", "attention_kernel")},
            "result": {k: res.get(k) for k in ("precision", "attention_kernel")},
            "gguf_precision_key": bool(gguf) and gguf_has_key(gguf[0], "paperkiln.train.precision"),
            "s_per_step": (t_last - t_first) / (s_last - s_first) if t_first is not None and s_last > s_first else None}


def bitwise(a, b):
    diff = sum(x != y for x, y in zip(a["loss"], b["loss"])) + sum(x != y for x, y in zip(a["grad_norm"], b["grad_norm"]))
    diff += abs(len(a["loss"]) - len(b["loss"]))
    same_files = a["files"] == b["files"] and bool(a["files"])
    ok = a["rc"] == 0 and b["rc"] == 0 and diff == 0 and a["val"] == b["val"] and same_files
    return {"ok": ok, "steps_differ": diff, "val": [a["val"], b["val"]], "files_identical": same_files,
            "files": a["files"]}


def moving(xs, w=20):
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= w:
            acc -= xs[i - w]
        out.append(acc / min(i + 1, w))
    return out


def compare(leg, run_leg, base):
    ma_tol, val_tol = TOL[leg]
    a, b = run_leg["loss"], base["loss"]
    raw = max(abs(x - y) / abs(y) for x, y in zip(a, b)) if a and b else None
    ma = max(abs(x - y) / abs(y) for x, y in zip(moving(a), moving(b))) if a and b else None
    val = abs(run_leg["val"] - base["val"]) / abs(base["val"]) if run_leg["val"] and base["val"] else None
    want = LEGS[leg]
    recorded = all(run_leg["model"].get(k) == v and run_leg["result"].get(k) == v for k, v in want.items())
    if "precision" in want:
        recorded = recorded and run_leg["gguf_precision_key"]
    ok = (run_leg["rc"] == 0 and recorded and ma is not None and ma <= ma_tol and val is not None and val <= val_tol)
    return {"ok": ok, "recorded": recorded, "loss_raw_worst_rel": raw, "loss_ma20_worst_rel": ma,
            "val": [run_leg["val"], base["val"]], "val_rel": val, "tolerance": {"ma20": ma_tol, "val": val_tol},
            "first_last": [a[0], a[-1], b[0], b[-1]] if a and b else None,
            "s_per_step": [run_leg["s_per_step"], base["s_per_step"]],
            "speedup": base["s_per_step"] / run_leg["s_per_step"]
            if run_leg["s_per_step"] and base["s_per_step"] else None}


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    text = os.path.join(a.out, "text.txt")
    if not os.path.exists(text):
        with open(a.text, "rb") as f:
            raw = f.read(a.text_bytes)
        cut = raw.rfind(b"\n")  # whole lines, valid UTF-8
        with open(text + ".tmp", "wb") as f:
            f.write(raw[:cut + 1] if cut > 0 else raw)
        os.replace(text + ".tmp", text)
    if a.custom:
        d, layers, heads, dff = (int(x) for x in a.custom.split(","))
        arch = {"preset": "llama-tiny", "custom": {"d": d, "layers": layers, "heads": heads, "d_ff": dff}}
    else:
        arch = step_profile.shape_spec(a.shape)["arch"]
    spec = {"name": "gate", "arch": arch,
            "data": {"corpus": text, "vocab": a.vocab, "vocab_cap": 16384, "T": a.T, "max_tokens": 0},
            "train": {"steps": a.steps, "lr": a.lr, "batch": a.batch, "accum": a.accum, "clip": 1.0,
                      "eval_every": a.steps, "eval_windows": 16, "checkpoint_every": 10 ** 7, "gradmap_every": 0,
                      "seed": 1, "optimizer": "adamw",
                      "lr_schedule": {"warmup_steps": max(1, a.steps // 10), "decay": "cosine", "min_lr_frac": 0.1}},
            "export": {"formats": ["safetensors", "gguf"]}}
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICROTORCH_")}
    env.update({"MICROTORCH_DEVICE": "cpu", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")} if a.cpu
               else step_profile.ENV)
    gate = {"spec": spec, "env": {k: v for k, v in env.items() if k.startswith(("MICROTORCH_", "OMP_"))},
            "new": a.new, "old": a.old, "gpu": None if a.cpu else step_profile.gpu_name(), "legs": {}}
    R = os.path.join(a.out, "runs")
    base = run(a.new, spec, os.path.join(R, "new_default"), env)
    gate["base"] = {k: base[k] for k in ("rc", "val", "s_per_step")}
    if a.old:
        old = run(a.old, spec, os.path.join(R, "old_default"), env)
        gate["legs"]["off_bitwise"] = bitwise(base, old)
    explicit = dict(spec, train=dict(spec["train"], precision="fp32", attention_kernel="tiled"))
    gate["legs"]["off_explicit"] = bitwise(run(a.new, explicit, os.path.join(R, "new_explicit"), env), base)
    gate["legs"]["repeat"] = bitwise(run(a.new, spec, os.path.join(R, "new_repeat"), env), base)
    for leg, keys in LEGS.items():
        s = dict(spec, train=dict(spec["train"], **keys))
        gate["legs"][leg] = compare(leg, run(a.new, s, os.path.join(R, leg), env), base)
    gate["ok"] = all(v["ok"] for v in gate["legs"].values())
    with open(os.path.join(a.out, "gate.json"), "w", encoding="utf-8") as f:
        json.dump(gate, f, indent=1)
    for k, v in gate["legs"].items():
        print(k, json.dumps({x: v.get(x) for x in ("ok", "steps_differ", "files_identical", "loss_ma20_worst_rel",
                                                    "val_rel", "speedup") if x in v}))
    print("GATE", "PASS" if gate["ok"] else "FAIL")
    return 0 if gate["ok"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--new", required=True)
    r.add_argument("--old")
    r.add_argument("--text", required=True)
    r.add_argument("--vocab", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--shape", default="20m_L8")
    r.add_argument("--custom")
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--accum", type=int, default=2)
    r.add_argument("--T", type=int, default=512)
    r.add_argument("--steps", type=int, default=300)
    r.add_argument("--lr", type=float, default=0.002)
    r.add_argument("--text-bytes", type=int, default=20_000_000)
    r.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
