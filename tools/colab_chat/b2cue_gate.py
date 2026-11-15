#!/usr/bin/env python3
"""A32's device gate (b2cue; REGISTRATION_DRAFT.md, frozen at eef513ad), run on the pod before any job of a campaign. Claude Opus 5.5,
15 Nov 2026. A31's gate (a31_gate.py, unchanged and imported) at each of the campaign's shapes, and two legs more:

  1. tests/test_twin_device.cpp under the pod's CUDA environment (its answer-mask legs included): must pass with the device op set ON.
  2. per shape, A31's four runs (cuda_twin, cpu_twin, cuda_plain, cpu_plain: two steps of the gate spec, batch 1, lambda = mu = 1) and
     a31_gate.compare's checks unchanged;
  3. per shape, the answer-token margin (arm NA): cuda_am and cpu_am, the gate spec with train.twin_loss.answer_mask on the gate
     twins' masks. Passes when every value is present and finite, step 1's SFT loss equals cuda_plain's bit for bit, and at steps 1 and
     2 CUDA and CPU agree on SFT loss, twin loss and grad_norm within 1% and on d_a, d_b, d_n within 0.02 nats (A31's tolerances);
  4. per shape, one full-batch leg: the arm spec at its own batch (16), two steps on CUDA from the same weights, which must finish with
     finite values (the memory of a 1,024-step arm at that shape: an out-of-memory fails here, not mid-campaign).
Each run has A31's 60-minute timeout.

    python3 tools/colab_chat/b2cue_gate.py --out DIR --spec GATE_D1.json [--spec GATE_A.json] [--bin MTSTUDIO] [--device-test T]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a31_gate as G  # noqa: E402  (run, device_test, compare, num, TOL_REL, TOL_D: A31's, unchanged)


def compare_am(am, amc, cp):
    if am is None or amc is None or cp is None:
        return ["an answer-mask run failed or timed out"]
    fails = []
    for name, r in (("cuda_am", am), ("cpu_am", amc)):
        for step in (1, 2):
            e = r.get(step)
            if e is None:
                fails.append(f"{name}: no step {step}")
                continue
            t = e.get("twin") or {}
            if not all(G.num(v) for v in [e.get("loss"), e.get("grad_norm")] + [t.get(k) for k in ("loss", "d_a", "d_b", "d_n")]):
                fails.append(f"{name} step {step}: a value is missing or not finite")
            if t.get("answer_mask") is not True:
                fails.append(f"{name} step {step}: the twin record does not say answer_mask")
    if fails:
        return fails

    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-6)
    if am[1]["loss"] != cp[1]["loss"]:
        fails.append("step 1 SFT loss with the answer-token margin differs from the plain run's on CUDA")
    for step in (1, 2):
        a, b = am[step], amc[step]
        for k in ("loss", "grad_norm"):
            if rel(a[k], b[k]) > G.TOL_REL:
                fails.append(f"answer mask step {step} {k} CUDA {a[k]} against CPU {b[k]}")
        if rel(a["twin"]["loss"], b["twin"]["loss"]) > G.TOL_REL:
            fails.append(f"answer mask step {step} twin loss CUDA {a['twin']['loss']} against CPU {b['twin']['loss']}")
        for k in ("d_a", "d_b", "d_n"):
            if abs(a["twin"][k] - b["twin"][k]) > G.TOL_D:
                fails.append(f"answer mask step {step} {k} CUDA {a['twin'][k]} against CPU {b['twin'][k]}")
    return fails


def full_batch_spec(spec):
    s = json.loads(json.dumps(spec))
    s["train"]["batch"] = 16
    s["data"]["twin_examples"] = spec["data"]["twin_examples"]
    s["train"]["twin_loss"]["groups"] = 4
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", action="append", required=True)
    ap.add_argument("--bin", default="/content/pk/build_cuda/mtstudio")
    ap.add_argument("--device-test", default="/content/pk/build_cuda/test_twin_device")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cpu_env = {k: v for k, v in os.environ.items() if not k.startswith("MICROTORCH_")}
    cpu_env.update({"MICROTORCH_DEVICE": "cpu", "OMP_NUM_THREADS": str(len(os.sched_getaffinity(0)))})
    dev = G.device_test(a.device_test, os.path.join(a.out, "twin_device.log"))
    fails = [] if dev["pass"] else [f"twin gradient on the device: {dev['why']}"]
    out = {"producer": "b2cue_gate.py", "device_test": dev, "shapes": {}}
    for path in a.spec:
        spec = json.load(open(path))
        tag = spec.get("name", os.path.basename(path))
        plain = json.loads(json.dumps(spec))
        plain["data"].pop("twin_examples")
        plain["train"].pop("twin_loss")
        am_spec = json.loads(json.dumps(spec))
        am_spec["train"]["twin_loss"]["answer_mask"] = True
        secs, t0, r = {}, time.time(), {}
        for leg, sp, env in (("cuda_twin", spec, dict(os.environ)), ("cuda_plain", plain, dict(os.environ)),
                             ("cpu_twin", spec, cpu_env), ("cpu_plain", plain, cpu_env),
                             ("cuda_am", am_spec, dict(os.environ)), ("cpu_am", am_spec, cpu_env),
                             ("cuda_full_batch", full_batch_spec(spec), dict(os.environ))):
            r[leg] = G.run(a.bin, sp, os.path.join(a.out, f"{tag}_{leg}"), env)
            secs[leg], t0 = time.time() - t0, time.time()
        f = [f"{tag}: {x}" for x in G.compare(r["cuda_twin"], r["cpu_twin"], r["cuda_plain"], r["cpu_plain"])]
        f += [f"{tag}: {x}" for x in compare_am(r["cuda_am"], r["cpu_am"], r["cuda_plain"])]
        fb = r["cuda_full_batch"]
        if fb is None or not all(G.num((fb.get(s) or {}).get("loss")) and G.num((fb.get(s) or {}).get("grad_norm")) for s in (1, 2)):
            f.append(f"{tag}: the full-batch leg failed or gave a non-finite value (memory or numerics at batch 16)")
        fails += f
        out["shapes"][tag] = {"seconds": secs, "runs": r, "fails": f}
    out.update({"tolerances": {"relative": G.TOL_REL, "shift_nats": G.TOL_D, "twin_grad_relative": G.TOL_TWIN_GRAD},
                "env": {k: v for k, v in os.environ.items() if k.startswith(("MICROTORCH_", "OMP_"))}, "fails": fails, "pass": not fails})
    with open(os.path.join(a.out, "gate.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"pass": not fails, "fails": fails}))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
