#!/usr/bin/env python3
"""A31's device gate (lane PROTOCOL.md section 50), run on the pod before any arm. Claude Opus 5.5, 13 Nov 2026; revised after the
launch reviews of 13 Nov 2026 (the gradient, not only the forward; fail-closed on missing or non-finite values; timeouts).

Two steps of the gate spec (S71's spec at batch 2, a twin loss on two groups, the full learning rate from step 1) three times from
the same weights:
  cuda_twin   the pod's CUDA environment (jobs_vm.py's ENV, which this job inherits)
  cpu_twin    MICROTORCH_DEVICE=cpu, every other MICROTORCH_ variable removed
  cuda_plain  the same spec without the twin loss, on CUDA
Passes only when every compared value is present and finite (nlohmann writes NaN as null, which fails), and:
  - step 1's SFT loss of cuda_plain equals cuda_twin's bit for bit (the twin term leaves the SFT batch alone on the device);
  - step 1, cuda_twin against cpu_twin: SFT loss, twin loss and grad_norm within 1% (relative), d_a, d_b, d_n within 0.02 nats;
  - the twin term's own gradient: grad_norm(twin) - grad_norm(plain) on CUDA within 5% (relative) of the same difference on the CPU
    (cpu_plain is run for it), and that difference is not 0;
  - step 2 (which reads the weights the first update wrote), cuda_twin against cpu_twin: SFT loss, twin loss, d_a, d_b, d_n as at
    step 1, and grad_norm within 1%.
Each run has a 30-minute timeout.

    python3 tools/colab_chat/a31_gate.py --out DIR --spec GATE_SPEC.json [--bin MTSTUDIO]
"""
import argparse
import json
import math
import os
import subprocess
import sys

TOL_REL, TOL_D, TOL_TWIN_GRAD = 0.01, 0.02, 0.05
TIMEOUT_S = 1800


def run(binary, spec, out, env):
    """The step events of one run, {step: event}, or None when it failed or timed out."""
    s = json.loads(json.dumps(spec))
    s["out_dir"] = out
    os.makedirs(out, exist_ok=True)
    p = out + "_spec.json"
    with open(p, "w") as f:
        json.dump(s, f, indent=1)
    try:
        with open(out + ".log", "w") as log:
            rc = subprocess.run([binary, "run", p], stdout=log, stderr=subprocess.STDOUT, env=env, timeout=TIMEOUT_S).returncode
    except subprocess.TimeoutExpired:
        return None
    if rc != 0:
        return None
    ev = [json.loads(x) for x in open(os.path.join(out, "events.jsonl")) if x.strip()]
    return {e["step"]: e for e in ev if e.get("event") == "step"}


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def compare(ct, cc, cp, cpc):
    """The gate's failures from the four runs' step events ({step: event}; None where a run failed)."""
    if ct is None or cc is None or cp is None or cpc is None:
        return ["a run failed or timed out"]
    fails = []
    for name, r, twin in (("cuda_twin", ct, True), ("cpu_twin", cc, True), ("cuda_plain", cp, False), ("cpu_plain", cpc, False)):
        for step in (1, 2):
            e = r.get(step)
            if e is None:
                fails.append(f"{name}: no step {step}")
                continue
            vals = [e.get("loss"), e.get("grad_norm")]
            if twin:
                t = e.get("twin") or {}
                vals += [t.get(k) for k in ("loss", "d_a", "d_b", "d_n")]
            if not all(num(v) for v in vals):
                fails.append(f"{name} step {step}: a value is missing or not finite")
    if fails:
        return fails

    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-6)
    if cp[1]["loss"] != ct[1]["loss"]:
        fails.append(f"step 1 SFT loss with and without the twin term differ on CUDA: {cp[1]['loss']!r} {ct[1]['loss']!r}")
    for step in (1, 2):
        a, b = ct[step], cc[step]
        if rel(a["loss"], b["loss"]) > TOL_REL:
            fails.append(f"step {step} SFT loss CUDA {a['loss']} against CPU {b['loss']}")
        if rel(a["grad_norm"], b["grad_norm"]) > TOL_REL:
            fails.append(f"step {step} grad_norm CUDA {a['grad_norm']} against CPU {b['grad_norm']}")
        if rel(a["twin"]["loss"], b["twin"]["loss"]) > TOL_REL and abs(a["twin"]["loss"] - b["twin"]["loss"]) > 1e-3:
            fails.append(f"step {step} twin loss CUDA {a['twin']['loss']} against CPU {b['twin']['loss']}")
        for k in ("d_a", "d_b", "d_n"):
            if abs(a["twin"][k] - b["twin"][k]) > TOL_D:
                fails.append(f"step {step} {k} CUDA {a['twin'][k]} against CPU {b['twin'][k]}")
    g_cuda = ct[1]["grad_norm"] - cp[1]["grad_norm"]
    g_cpu = cc[1]["grad_norm"] - cpc[1]["grad_norm"]
    if g_cpu == 0 or g_cuda == 0:
        fails.append("the twin term changes no gradient norm")
    elif abs(g_cuda - g_cpu) > TOL_TWIN_GRAD * abs(g_cpu):
        fails.append(f"the twin term's gradient-norm change CUDA {g_cuda} against CPU {g_cpu}")
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
    cpc = run(a.bin, plain, os.path.join(a.out, "cpu_plain"), cpu_env)
    fails = compare(ct, cc, cp, cpc)
    out = {"producer": "a31_gate.py", "cuda_twin": ct, "cpu_twin": cc, "cuda_plain": cp, "cpu_plain": cpc,
           "tolerances": {"relative": TOL_REL, "shift_nats": TOL_D, "twin_grad_relative": TOL_TWIN_GRAD},
           "env": {k: v for k, v in os.environ.items() if k.startswith(("MICROTORCH_", "OMP_"))}, "fails": fails, "pass": not fails}
    with open(os.path.join(a.out, "gate.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"pass": not fails, "fails": fails}))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
