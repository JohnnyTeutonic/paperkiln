"""Pre-registered analysis for registry #0001 pilot (Highway Networks).

WRITTEN PRE-DATA for the full contrast (the residual lane had finished
its runs when this was committed, but NO lane had been read or compared;
the decision rules below are fixed verbatim by PREREGISTRATION.md,
committed before any run):

  ESTIMATION FRAMING: no committed direction, no one-tailed licence.
  All tests paired two-tailed t, df=4, crit 2.776.
  Q1 (primary):    Delta1 = bestval(HIGHWAY) - bestval(RESIDUAL),
                   read at BOTH 400 and 1200 steps (neither privileged).
  Q2 (secondary,   Delta2 = bestval(PLAIN) - bestval(RESIDUAL),
      descriptive) reported as estimate with CI.
  Mechanism observable E[T]: per-layer transform-gate activation — the
      event stream does not record it (verified pre-data: step events
      carry loss/grad_norm only), so per the prereg it is OMITTED WITH
      THIS NOTE rather than patched in mid-run.
  Guards: every run's model event must match its lane exactly
      (d=128, layers=2, attention=exact, residual knob, d_ff 512/384/512,
      gate_bias_init -2 on highway) and the param accounting must land
      on the documented +256 highway mismatch with plain == residual.

Reads ~/registry_0001_out/<lane>/runs/run_*/ (override: REG0001_OUT).
--bank copies each run's events.jsonl + result.json into receipts/ here
(the volatility rule: receipts live in the repo, not in a home dir).
"""
import glob
import json
import os
import shutil
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("REG0001_OUT", os.path.expanduser("~/registry_0001_out"))
SEEDS = [1, 2, 3, 4, 5]
LANES = ("residual", "highway", "plain")
CRIT = 2.776  # two-tailed, df=4 — the only licensed threshold

EXPECT = {
    "residual": {"residual": "residual", "d_ff": 512},
    "highway": {"residual": "highway", "d_ff": 384, "gate_bias_init": -2.0},
    "plain": {"residual": "plain", "d_ff": 512},
}


def load_run(lane, seed):
    pats = glob.glob(os.path.join(ROOT, lane, "runs", f"run_*_s{seed}"))
    assert len(pats) == 1, f"{lane} s{seed}: {len(pats)} run dirs"
    d = pats[0]
    result = json.load(open(os.path.join(d, "result.json")))
    evals, model = [], None
    for line in open(os.path.join(d, "events.jsonl")):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") == "eval":
            evals.append((ev["step"], ev["val_loss"]))
        elif ev.get("event") == "model":
            model = ev
    assert evals, f"{lane} s{seed}: no eval events"
    assert model is not None, f"{lane} s{seed}: no model event"
    assert model.get("d") == 128 and model.get("layers") == 2, \
        f"{lane} s{seed}: dims {model.get('d')}x{model.get('layers')}"
    assert model.get("attention") == "exact", \
        f"{lane} s{seed}: attention={model.get('attention')}"
    for k, v in EXPECT[lane].items():
        assert model.get(k) == v, f"{lane} s{seed}: {k}={model.get(k)} != {v}"
    return {"result": result, "evals": sorted(evals), "dir": d,
            "params": model.get("params")}


def bestval_by(run, b):
    vals = [v for s, v in run["evals"] if s <= b]
    assert vals, f"no evals at or before step {b}"
    return min(vals)


def tstat(xs):
    m, sd = st.fmean(xs), st.stdev(xs)
    se = sd / len(xs) ** 0.5
    return m, se, m / se


runs = {(lane, s): load_run(lane, s) for lane in LANES for s in SEEDS}

# Param accounting guard (prereg table): highway = residual + 256,
# plain = residual exactly.
pr = runs[("residual", 1)]["params"]
ph = runs[("highway", 1)]["params"]
pp = runs[("plain", 1)]["params"]
assert ph - pr == 256, f"highway mismatch {ph - pr} != +256"
assert pp == pr, f"plain params {pp} != residual {pr}"

print("=" * 72)
print("registry #0001 PILOT — PRE-REGISTERED ANALYSIS (estimation framing)")
print(f"params: residual {pr}  highway {ph} (+{ph - pr}, documented)  "
      f"plain {pp}")
print("=" * 72)

for budget in (400, 1200):
    d1 = [bestval_by(runs[("highway", s)], budget)
          - bestval_by(runs[("residual", s)], budget) for s in SEEDS]
    m, se, t = tstat(d1)
    lo, hi = m - CRIT * se, m + CRIT * se
    verdict = ("resolvable" if abs(t) > CRIT else
               "no resolvable difference at this protocol and scale")
    print(f"\nQ1 Delta1(HIGHWAY - RESIDUAL) at {budget} steps:")
    print(f"  per-seed {[f'{x:+.4f}' for x in d1]}")
    print(f"  mean {m:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  t={t:.2f} "
          f"(df=4, crit {CRIT})")
    print(f"  reading: {verdict}"
          + (f"; sign {'+' if m > 0 else '-'} "
             f"({'residual' if m > 0 else 'highway'} better)"
             if abs(t) > CRIT else ""))

for budget in (400, 1200):
    d2 = [bestval_by(runs[("plain", s)], budget)
          - bestval_by(runs[("residual", s)], budget) for s in SEEDS]
    m, se, t = tstat(d2)
    print(f"\nQ2 Delta2(PLAIN - RESIDUAL) at {budget} steps (descriptive):")
    print(f"  per-seed {[f'{x:+.4f}' for x in d2]}")
    print(f"  mean {m:+.5f}  95% CI [{m - CRIT * se:+.5f}, {m + CRIT * se:+.5f}]"
          f"  t={t:.2f}")

print("\nMechanism observable E[T]: OMITTED — the event stream does not "
      "record gate activations (verified pre-data); per the prereg this "
      "is noted, not patched mid-run. A gate-logging cell would be its "
      "own registration.")

print("\n" + "-" * 72)
print("THREAT CHECK: stall/early-stop diagnostics + trace cross-check")
for lane in LANES:
    fs = [runs[(lane, s)]["result"].get("final_step") for s in SEEDS]
    es = [runs[(lane, s)]["result"].get("early_stopped") for s in SEEDS]
    print(f"  {lane:>8}: final_step {fs}  early_stopped {es}")
bad = 0
for lane in LANES:
    for s in SEEDS:
        a = bestval_by(runs[(lane, s)], 1200)
        b = runs[(lane, s)]["result"].get("best_val")
        if b is None or abs(a - b) >= 1e-4:
            bad += 1
            print(f"  MISMATCH {lane} s{s}: trace {a:.4f} vs result {b}")
print("  trace vs result.json best_val: "
      + ("all agree" if bad == 0 else f"{bad} MISMATCHES"))

if "--bank" in sys.argv:
    dst_root = os.path.join(HERE, "receipts")
    for (lane, s), r in runs.items():
        dst = os.path.join(dst_root, f"{lane}_s{s}")
        os.makedirs(dst, exist_ok=True)
        for f in ("events.jsonl", "result.json"):
            shutil.copy2(os.path.join(r["dir"], f), os.path.join(dst, f))
    print(f"\nreceipts banked to {dst_root}")
