"""Pre-registered analysis for sparse_s1_scale Rung B (width-only, Amendment 1).

WRITTEN AND COMMITTED BEFORE ANY RESULT WAS RETRIEVED. Decision rules
are fixed by PREREGRISTRATION.md as amended (licence anchor 9c7c8b3e):

  H-SCALE-SHRINK (committed direction): within-run paired
      shrink_s = Delta_s(1200) - Delta_s(400) > 0,
      Delta_s(b) = bestval_by_b(swa,s) - bestval_by_b(exact,s).
      Paired t over 5 seeds; ONE-TAILED crit 2.132 (df=4) licensed by
      the pre-data amendment commit; two-tailed 2.776 reported too.
  D-SIGNS (descriptive only): signs of Delta(400) and Delta(1200).
  Threat check: early-stop / stall diagnostics per lane.
  Amendment guard: every run must record d=256, layers=2 in its model
      event or the analysis refuses to run.

Local-execution layout (the EXECUTION NOTE moved Rung B to local CPU;
one mtsweep with both lanes as factor cells): all runs live under
  ~/rungB_out/runs/run_*_c*_s*/{events.jsonl,result.json}
and each run's LANE is read from its own model event's `attention`
field — never inferred from the directory name. (MECHANICAL loader
adaptation for the local layout, 13 Aug 2026, made before any contrast
was read; decision rules below are untouched from the pre-data commit.)
Override the root with SPARSE_S1_ROOT.
"""
import glob
import json
import os
import re
import statistics as st

ROOT = os.environ.get("SPARSE_S1_ROOT", os.path.expanduser("~/rungB_out"))
SEEDS = [1, 2, 3, 4, 5]
LANES = ("exact", "swa")
ONE_TAILED_CRIT = 2.132   # df=4, alpha=.05, direction committed pre-data
TWO_TAILED_CRIT = 2.776


def _read_run(d):
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
    assert evals, f"{d}: no eval events"
    assert model is not None, f"{d}: no model event"
    # Amendment 1 guard: width-only rung, d=256 at 2 layers.
    d_val = model.get("d", model.get("d_model"))
    layers = model.get("layers", model.get("num_layers"))
    assert d_val == 256, f"{d}: d={d_val} != 256"
    assert layers == 2, f"{d}: layers={layers} != 2"
    return model, {"result": result, "evals": sorted(evals)}


def load_all():
    dirs = sorted(glob.glob(os.path.join(ROOT, "runs", "run_*")))
    assert dirs, f"no runs under {ROOT}/runs"
    table = {}
    for d in dirs:
        model, run = _read_run(d)
        lane = model.get("attention")
        seed = model.get("seed")
        assert lane in LANES, f"{d}: attention={lane}"
        assert (lane, seed) not in table, f"duplicate {lane} s{seed}"
        table[(lane, seed)] = run
    for lane in LANES:
        for s in SEEDS:
            assert (lane, s) in table, f"missing {lane} s{s}"
    return table


def bestval_by(run, b):
    vals = [v for s, v in run["evals"] if s <= b]
    assert vals, f"no evals at or before step {b}"
    return min(vals)


runs = load_all()

d400 = [bestval_by(runs[("swa", s)], 400) - bestval_by(runs[("exact", s)], 400)
        for s in SEEDS]
d1200 = [bestval_by(runs[("swa", s)], 1200) - bestval_by(runs[("exact", s)], 1200)
         for s in SEEDS]
shrink = [d1200[i] - d400[i] for i in range(len(SEEDS))]


def tstat(xs):
    m, sd = st.fmean(xs), st.stdev(xs)
    return m, sd, m / (sd / len(xs) ** 0.5)


print("=" * 72)
print("sparse_s1_scale RUNG B PRE-REGISTERED ANALYSIS")
print("(width-only per Amendment 1; licence anchor 9c7c8b3e)")
print("=" * 72)
m4, _, t4 = tstat(d400)
m12, _, t12 = tstat(d1200)
print(f"\nDelta(400)  = {m4:+.5f} (t={t4:.2f})  per-seed "
      f"{[f'{x:+.4f}' for x in d400]}")
print(f"Delta(1200) = {m12:+.5f} (t={t12:.2f})  per-seed "
      f"{[f'{x:+.4f}' for x in d1200]}")

ms, _, ts = tstat(shrink)
print("\n" + "-" * 72)
print("H-SCALE-SHRINK: Delta(1200) - Delta(400) > 0 (direction committed)")
print(f"  per-seed shrink {[f'{x:+.4f}' for x in shrink]}")
print(f"  mean {ms:+.5f}  t = {ts:.2f}  df = 4  "
      f"one-tailed crit {ONE_TAILED_CRIT}  two-tailed {TWO_TAILED_CRIT}")
if ms > 0 and ts > ONE_TAILED_CRIT:
    two = "ALSO two-tailed" if ts > TWO_TAILED_CRIT else "one-tailed only"
    print(f"  VERDICT: SUPPORTED at Rung B ({two}) -- the budget effect "
          "recurs at d=256")
elif ms <= 0 and abs(ts) > ONE_TAILED_CRIT:
    print("  VERDICT: FALSIFIED IN DIRECTION -- budget-conditionality is "
          "scoped to d=128; record the scale boundary per prereg")
else:
    print("  VERDICT: INSIDE NOISE -- inconclusive; report as such, no "
          "trend claim is licensed")

print("\nD-SIGNS (descriptive, no committed direction):")
print(f"  Delta(400) < 0 (early sparse advantage): "
      f"{sum(1 for x in d400 if x < 0)}/5 seeds, mean {m4:+.5f}")
print(f"  Delta(1200) > 0 (late exact advantage): "
      f"{sum(1 for x in d1200 if x > 0)}/5 seeds, mean {m12:+.5f}")

print("\n" + "-" * 72)
print("THREAT CHECK: stall/early-stop diagnostics")
for lane in LANES:
    fs = [runs[(lane, s)]["result"].get("final_step") for s in SEEDS]
    bv = [round(runs[(lane, s)]["result"].get("best_val", -1), 4)
          for s in SEEDS]
    print(f"  {lane:>5}: final_step {fs}  best_val {bv}")

print("\nCross-check: trace-derived bestval_by(1200) vs result.json best_val")
for lane in LANES:
    for s in SEEDS:
        a = bestval_by(runs[(lane, s)], 1200)
        b = runs[(lane, s)]["result"].get("best_val")
        flag = "" if b is None or abs(a - b) < 1e-4 else "  <-- MISMATCH"
        print(f"  {lane}_s{s}: trace {a:.4f}  result {b}{flag}")
