#!/usr/bin/env python3
"""A31's device gate (lane PROTOCOL.md section 50), run on the pod before any arm. Claude Opus 5.5, 13 Nov 2026.

One step of the gate spec (S71's spec at batch 2 with a twin loss on two groups) three times from the same weights:
  cuda_twin   the pod's CUDA environment (jobs_vm.py's ENV, which this job inherits)
  cpu_twin    MICROTORCH_DEVICE=cpu, every other MICROTORCH_ variable removed
  cuda_plain  the same spec without the twin loss, on CUDA
Passes when step 1's SFT loss of cuda_plain equals cuda_twin's bit for bit (the twin loss leaves the SFT batch alone on the
device too), and cuda_twin agrees with cpu_twin: SFT loss and twin loss within 1% (relative), the per-token shifts d_a, d_b
and d_n within 0.02 nats. The twin loss's device path is otherwise untested (no local GPU).

    python3 tools/colab_chat/a31_gate.py --out DIR --spec GATE_SPEC.json [--bin MTSTUDIO]
"""
import argparse
import json
import os
import subprocess
import sys

TOL_REL, TOL_D = 0.01, 0.02


def run(binary, spec, out, env):
    s = json.loads(json.dumps(spec))
    s["out_dir"] = out
    os.makedirs(out, exist_ok=True)
    p = out + "_spec.json"
    with open(p, "w") as f:
        json.dump(s, f, indent=1)
    with open(out + ".log", "w") as log:
        rc = subprocess.run([binary, "run", p], stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        return None
    ev = [json.loads(x) for x in open(os.path.join(out, "events.jsonl")) if x.strip()]
    return next(e for e in ev if e.get("event") == "step" and e.get("step") == 1)


def compare(ct, cc, cp):
    """The gate's verdict from the three step-1 events (None where a run failed)."""
    fails = []
    if ct is None or cc is None or cp is None:
        return ["a run failed"]
    if cp["loss"] != ct["loss"]:
        fails.append(f"SFT loss with and without the twin loss differ on CUDA: {cp['loss']!r} {ct['loss']!r}")

    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-6)
    if rel(ct["loss"], cc["loss"]) > TOL_REL:
        fails.append(f"SFT loss CUDA {ct['loss']} against CPU {cc['loss']}")
    if rel(ct["twin"]["loss"], cc["twin"]["loss"]) > TOL_REL and abs(ct["twin"]["loss"] - cc["twin"]["loss"]) > 1e-3:
        fails.append(f"twin loss CUDA {ct['twin']['loss']} against CPU {cc['twin']['loss']}")
    for k in ("d_a", "d_b", "d_n"):
        if abs(ct["twin"][k] - cc["twin"][k]) > TOL_D:
            fails.append(f"{k} CUDA {ct['twin'][k]} against CPU {cc['twin'][k]}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--bin", default="/content/pk/build_cuda/mtstudio")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    plain = json.loads(json.dumps(spec))
    plain["data"].pop("twin_examples")
    plain["train"].pop("twin_loss")
    cpu_env = {k: v for k, v in os.environ.items() if not k.startswith("MICROTORCH_")}
    cpu_env.update({"MICROTORCH_DEVICE": "cpu", "OMP_NUM_THREADS": str(os.cpu_count() or 4)})
    ct = run(a.bin, spec, os.path.join(a.out, "cuda_twin"), dict(os.environ))
    cp = run(a.bin, plain, os.path.join(a.out, "cuda_plain"), dict(os.environ))
    cc = run(a.bin, spec, os.path.join(a.out, "cpu_twin"), cpu_env)
    fails = compare(ct, cc, cp)
    out = {"producer": "a31_gate.py", "cuda_twin": ct, "cpu_twin": cc, "cuda_plain": cp, "tolerances": {"relative": TOL_REL, "shift_nats": TOL_D},
           "env": {k: v for k, v in os.environ.items() if k.startswith(("MICROTORCH_", "OMP_"))}, "fails": fails, "pass": not fails}
    with open(os.path.join(a.out, "gate.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"pass": not fails, "fails": fails}))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
