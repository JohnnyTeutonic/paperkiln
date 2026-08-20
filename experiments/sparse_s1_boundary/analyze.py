"""Pre-registered analysis for sparse_s1_boundary: locating B*(256).

WRITTEN AND COMMITTED WITH THE PRE-REGISTRATION, BEFORE ANY RUN.
Decision rules fixed by PREREGISTRATION.md (licence anchor = the
commit that introduced this file):

  H-SHRINK-CONT (committed direction, one-tailed 2.132 licensed):
      within-seed paired shrink2_s = Delta_s(3600) - Delta_s(1200) > 0
  H-CROSS vs H-NO-CROSS (no committed direction, TWO-TAILED 2.776 only):
      the sign of Delta(3600). Positive+clearing => B*(256) in
      (1200, 3600]. Negative+clearing => B*(256) > 3600. Neither =>
      undetermined; claim nothing.
  D-SHAPE (descriptive): Delta at every 400-step slice; no test.
  Threat 1 (regime): overfit onset per lane (best_val achieved >= 3
      evals before the cutoff, in >= 3 seeds => scope the statement).
  Threat 3 (comparability): sign of Delta_s(1200) recomputed HERE must
      match Rung B's stored per-seed values on >= 4/5 seeds:
      Rung B Delta_s(1200) signs: all five negative.

Guard: every run's model event must record d=256, layers=2; lanes read
from model events, never directory names. Layout: $SPARSE_S1_ROOT
(default ~/boundary_out)/runs/run_*/{events.jsonl,result.json}.
"""
import glob
import json
import os
import statistics as st

ROOT = os.environ.get("SPARSE_S1_ROOT", os.path.expanduser("~/boundary_out"))
SEEDS = [1, 2, 3, 4, 5]
LANES = ("exact", "swa")
ONE_TAILED_CRIT = 2.132
TWO_TAILED_CRIT = 2.776
CUTOFF = 3600
# Rung B's stored per-seed Delta(1200) signs (RESULTS.md): all negative.
RUNGB_D1200_NEG = 5


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
        lane, seed = model.get("attention"), model.get("seed")
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


def tstat(xs):
    m, sd = st.fmean(xs), st.stdev(xs)
    return m, sd, m / (sd / len(xs) ** 0.5)


runs = load_all()

def delta(b):
    return [bestval_by(runs[("swa", s)], b) - bestval_by(runs[("exact", s)], b)
            for s in SEEDS]

d1200 = delta(1200)
d3600 = delta(CUTOFF)
shrink2 = [d3600[i] - d1200[i] for i in range(len(SEEDS))]

print("=" * 72)
print("sparse_s1_boundary PRE-REGISTERED ANALYSIS -- locating B*(256)")
print("=" * 72)

# --- Threat 3: comparability with Rung B ------------------------------
neg = sum(1 for x in d1200 if x < 0)
print(f"\nThreat 3 (comparability): Delta_s(1200) negatives here = {neg}/5 "
      f"(Rung B: {RUNGB_D1200_NEG}/5)")
if abs(neg - RUNGB_D1200_NEG) > 1:
    print("  VOID: sign pattern disagrees with Rung B on >1 seed -- "
          "different effective protocol; NO claim licensed. Stop.")
    raise SystemExit(1)

# --- Threat 1: regime check -------------------------------------------
overfit_lanes = []
for lane in LANES:
    n_early = 0
    for s in SEEDS:
        evs = runs[(lane, s)]["evals"]
        best_step = min(evs, key=lambda e: e[1])[0]
        if best_step <= CUTOFF - 3 * 100:
            n_early += 1
    print(f"Threat 1 (regime) {lane}: best_val >=3 evals before cutoff in "
          f"{n_early}/5 seeds")
    if n_early >= 3:
        overfit_lanes.append(lane)
scope_note = (" [SCOPED: overfit onset in " + ",".join(overfit_lanes) + "]"
              if overfit_lanes else "")

# --- D-SHAPE ----------------------------------------------------------
print("\nD-SHAPE (descriptive): mean Delta by budget slice")
for b in range(400, CUTOFF + 1, 400):
    ds = delta(b)
    print(f"  b={b:4}: {st.fmean(ds):+.5f}  per-seed "
          f"{[f'{x:+.4f}' for x in ds]}")

# --- H-SHRINK-CONT ----------------------------------------------------
ms, _, ts = tstat(shrink2)
print("\n" + "-" * 72)
print("H-SHRINK-CONT: Delta(3600) - Delta(1200) > 0 (direction committed)")
print(f"  per-seed {[f'{x:+.4f}' for x in shrink2]}")
print(f"  mean {ms:+.5f}  t = {ts:.2f}  df = 4  one-tailed crit "
      f"{ONE_TAILED_CRIT}  two-tailed {TWO_TAILED_CRIT}")
if ms > 0 and ts > ONE_TAILED_CRIT:
    two = "ALSO two-tailed" if ts > TWO_TAILED_CRIT else "one-tailed only"
    print(f"  VERDICT: SUPPORTED ({two}){scope_note}")
elif ms <= 0 and abs(ts) > ONE_TAILED_CRIT:
    print(f"  VERDICT: FALSIFIED IN DIRECTION{scope_note}")
else:
    print(f"  VERDICT: NOT ESTABLISHED{scope_note}")

# --- H-CROSS ----------------------------------------------------------
m36, _, t36 = tstat(d3600)
print("\n" + "-" * 72)
print("H-CROSS vs H-NO-CROSS: sign of Delta(3600) -- TWO-TAILED ONLY")
print(f"  Delta(3600) = {m36:+.5f}  t = {t36:.2f}  two-tailed crit "
      f"{TWO_TAILED_CRIT}  per-seed {[f'{x:+.4f}' for x in d3600]}")
if m36 > 0 and t36 > TWO_TAILED_CRIT:
    print(f"  VERDICT: H-CROSS -- B*(256) in (1200, 3600]{scope_note}")
elif m36 < 0 and abs(t36) > TWO_TAILED_CRIT:
    print(f"  VERDICT: H-NO-CROSS -- B*(256) > 3600; compatible with both "
          f"'arrives later' and 'never at this width'{scope_note}")
else:
    print(f"  VERDICT: sign undetermined at 3600; no B* claim{scope_note}")

print("\nScope: TinyStories slice, gpt2-nano family, d=256, L=2, w=64+sink,")
print("T=256, 5 paired seeds, CPU numerics, house protocol. B*(d) is a")
print("property of THIS protocol; the general claim remains budget-reporting")
print("discipline.")
