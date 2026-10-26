#!/usr/bin/env python3
"""Gate for train.fused_head (the fused LM head + cross-entropy, 26 Oct 2026), on real text.

    python3 tools/fused_head_gate.py run --new BIN [--old OLD_BIN] --text TEXT --vocab VOCAB.gguf --out DIR
        [--shape 20m_L8 | --custom D,L,H,DFF] [--batch B] [--accum A] [--T T] [--steps N] [--text-bytes N] [--cpu]

Runs mtstudio on one seeded spec (tools/precision_gate.py's: one eval at the end, safetensors + F32
GGUF export) as legs, and writes DIR/gate.json. The rule, stated before any run:
  off_bitwise   OLD_BIN vs BIN, the spec without the new keys: every step's loss and grad norm, the
                final validation loss and the exported files' sha256 identical (pre-registered arms,
                which run with the option off, train as before). Only with --old.
  off_explicit  BIN with "fused_head": false vs BIN without the key: identical as above.
  repeat        BIN twice: identical (run-to-run determinism, the precondition of the two above).
  fused         BIN with "fused_head": true vs BIN without: the model event and result.json record
                fused_head; the first step's loss identical on the host (--cpu: the op's loss is the
                unfused loss bit for bit there) and within 1e-6 relative on the device (cuBLAS); the
                20-step moving average of the training loss within 0.5% relative at every step and
                the final validation loss within 0.5% (the flash gate's tolerances: only the head
                weight's gradient changes summation order, so the runs part by rounding alone).
                Seconds per step for both, and their ratio.
--cpu runs the host path (MICROTORCH_DEVICE=cpu); otherwise the production device environment
(tools/colab_chat/jobs_vm.py ENV). Exit 0 when every leg passes.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precision_gate as pg  # noqa: E402
import step_profile  # noqa: E402

MA_TOL, VAL_TOL = 5e-3, 5e-3
FIRST_TOL_DEVICE = 1e-6


def run(binary, spec, run_dir, env):
    """precision_gate.run, plus what the run recorded about fused_head."""
    r = pg.run(binary, spec, run_dir, env)
    model = next((e for e in step_profile.events(run_dir) if e.get("event") == "model"), {})
    res = {}
    rp = os.path.join(run_dir, "result.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            res = json.load(f)
    r["fused_recorded"] = bool(model.get("fused_head")) and bool(res.get("fused_head"))
    r["fused_mentioned"] = "fused_head" in model or "fused_head" in res
    return r


def fused_leg(f, base, cpu):
    a, b = f["loss"], base["loss"]
    out = {"recorded": f["fused_recorded"], "rc": [f["rc"], base["rc"]]}
    if not a or not b or len(a) != len(b):
        out.update({"ok": False, "why": "missing or unequal step counts"})
        return out
    first_rel = abs(a[0] - b[0]) / abs(b[0])
    first_ok = a[0] == b[0] if cpu else first_rel <= FIRST_TOL_DEVICE
    ma = max(abs(x - y) / abs(y) for x, y in zip(pg.moving(a), pg.moving(b)))
    raw = max(abs(x - y) / abs(y) for x, y in zip(a, b))
    val = abs(f["val"] - base["val"]) / abs(base["val"]) if f["val"] and base["val"] else None
    sp = [f["s_per_step"], base["s_per_step"]]
    out.update({"first_loss": [a[0], b[0]], "first_rel": first_rel, "first_ok": first_ok,
                "loss_raw_worst_rel": raw, "loss_ma20_worst_rel": ma, "val": [f["val"], base["val"]], "val_rel": val,
                "tolerance": {"first": "bitwise" if cpu else FIRST_TOL_DEVICE, "ma20": MA_TOL, "val": VAL_TOL},
                "s_per_step": sp, "speedup": sp[1] / sp[0] if sp[0] and sp[1] else None})
    out["ok"] = bool(f["rc"] == 0 and base["rc"] == 0 and f["fused_recorded"] and first_ok and ma <= MA_TOL
                     and val is not None and val <= VAL_TOL)
    return out


def cmd_run(a):
    spec, env = pg.build_spec(a)
    gate = {"spec": spec, "env": {k: v for k, v in env.items() if k.startswith(("MICROTORCH_", "OMP_"))},
            "new": a.new, "old": a.old, "gpu": None if a.cpu else step_profile.gpu_name(), "rule": __doc__,
            "legs": {}}
    R = os.path.join(a.out, "runs")
    base = run(a.new, spec, os.path.join(R, "new_default"), env)
    gate["base"] = {k: base[k] for k in ("rc", "val", "s_per_step")}
    if base["fused_mentioned"]:
        gate["legs"]["default_silent"] = {"ok": False, "why": "the default run's events or result mention fused_head"}
    if a.old:
        old = run(a.old, spec, os.path.join(R, "old_default"), env)
        gate["legs"]["off_bitwise"] = pg.bitwise(base, old)
    explicit = dict(spec, train=dict(spec["train"], fused_head=False))
    gate["legs"]["off_explicit"] = pg.bitwise(run(a.new, explicit, os.path.join(R, "new_explicit"), env), base)
    gate["legs"]["repeat"] = pg.bitwise(run(a.new, spec, os.path.join(R, "new_repeat"), env), base)
    fused = dict(spec, train=dict(spec["train"], fused_head=True))
    gate["legs"]["fused"] = fused_leg(run(a.new, fused, os.path.join(R, "fused"), env), base, a.cpu)
    gate["ok"] = all(v["ok"] for v in gate["legs"].values())
    with open(os.path.join(a.out, "gate.json"), "w", encoding="utf-8") as f:
        json.dump(gate, f, indent=1)
    for k, v in gate["legs"].items():
        print(k, json.dumps({x: v.get(x) for x in ("ok", "steps_differ", "files_identical", "first_ok",
                                                    "loss_ma20_worst_rel", "val_rel", "speedup") if x in v}))
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
