#!/usr/bin/env python3
"""A34's device gate (the lane's cue_selective/a34/REGISTRATION.md), run on the pod before any training job. Claude Opus 5.5,
15 Nov 2026. A31's gate (a31_gate.py, imported unchanged) on A34's gate spec, which adds the own-context floor (weight 1) to A31's,
with references a34_gate_ref.jsonl: the start weights' own-context terms, scored at home by `mtstudio score` on the CPU, less 0.5
at knot 0 and plus 0.5 at knot 1. Passes only when A31's gate passes on this spec (its plain runs drop the floor's keys as well) and:
  - tests/test_twin_device.cpp passed with its floor legs (two "floor:" lines; the test itself compares host and device);
  - at step 1 (the start weights, references 0.5 below) on CUDA and on the CPU: the floor is inactive (active 0, loss 0), and
    |gap_absmax - 0.5| <= PARITY: the trainer's own-context terms equal the scorer's to PARITY nats a token;
  - at step 2 (references 0.5 above): the floor is active on every side, on both devices, and its loss agrees within 1%.

    python3 tools/colab_chat/a34_gate.py --out DIR --spec GATE_SPEC.json [--bin MTSTUDIO] [--device-test TEST_TWIN_DEVICE]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a31_gate as G  # noqa: E402

PARITY = 5e-3


def floor_checks(ct, cc, dev_log):
    fails = []
    if dev_log is None or dev_log.count("floor:") < 2:
        fails.append("the device test did not run its floor legs")
    if ct is None or cc is None:
        return fails + ["a floor run failed"]
    fl = {}
    for name, r in (("cuda_twin", ct), ("cpu_twin", cc)):
        for step in (1, 2):
            f = ((r.get(step) or {}).get("twin") or {}).get("floor")
            if not isinstance(f, dict) or not all(G.num(f.get(k)) for k in ("loss", "active", "gap", "gap_absmax")):
                fails.append(f"{name} step {step}: no finite floor record")
                continue
            fl[(name, step)] = f
    if fails:
        return fails
    for name in ("cuda_twin", "cpu_twin"):
        f1, f2 = fl[(name, 1)], fl[(name, 2)]
        if f1["active"] != 0 or f1["loss"] != 0:
            fails.append(f"{name} step 1: the floor is active below the start's own terms ({f1})")
        if abs(f1["gap_absmax"] - 0.5) > PARITY:
            fails.append(f"{name} step 1: trainer and scorer differ by {abs(f1['gap_absmax'] - 0.5):.2e} > {PARITY}")
        if f2["active"] != 1:
            fails.append(f"{name} step 2: the floor is not active on every side ({f2})")
    a, b = fl[("cuda_twin", 2)]["loss"], fl[("cpu_twin", 2)]["loss"]
    if abs(a - b) > G.TOL_REL * max(abs(b), 1e-6):
        fails.append(f"step 2 floor loss CUDA {a} against CPU {b}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--bin", default="/content/pk/build_cuda/mtstudio")
    ap.add_argument("--device-test", default="/content/pk/build_cuda/test_twin_device")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec = json.load(open(a.spec))
    plain = json.loads(json.dumps(spec))
    for k in ("twin_examples", "twin_floor_ref"):
        plain["data"].pop(k)
    plain["train"].pop("twin_loss")
    cpu_env = {k: v for k, v in os.environ.items() if not k.startswith("MICROTORCH_")}
    cpu_env.update({"MICROTORCH_DEVICE": "cpu", "OMP_NUM_THREADS": str(len(os.sched_getaffinity(0)))})
    dev_log_p = os.path.join(a.out, "twin_device.log")
    dev = G.device_test(a.device_test, dev_log_p)
    dev_log = open(dev_log_p).read() if os.path.exists(dev_log_p) else None
    secs, t0 = {}, time.time()
    runs = {}
    for name, s, env in (("cuda_twin", spec, dict(os.environ)), ("cuda_plain", plain, dict(os.environ)),
                         ("cpu_twin", spec, cpu_env), ("cpu_plain", plain, cpu_env)):
        runs[name] = G.run(a.bin, s, os.path.join(a.out, name), env)
        secs[name], t0 = time.time() - t0, time.time()
    ct, cp, cc, cpc = runs["cuda_twin"], runs["cuda_plain"], runs["cpu_twin"], runs["cpu_plain"]
    fails = ([] if dev["pass"] else [f"twin gradient on the device: {dev['why']}"]) + G.compare(ct, cc, cp, cpc)
    fails += floor_checks(ct, cc, dev_log)
    out = {"producer": "a34_gate.py (a31_gate.py's checks and the floor's)", "device_test": dev, "seconds": secs, **runs,
           "tolerances": {"relative": G.TOL_REL, "shift_nats": G.TOL_D, "twin_grad_relative": G.TOL_TWIN_GRAD, "parity": PARITY},
           "env": {k: v for k, v in os.environ.items() if k.startswith(("MICROTORCH_", "OMP_"))}, "fails": fails, "pass": not fails}
    with open(os.path.join(a.out, "gate.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"pass": not fails, "fails": fails}))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
