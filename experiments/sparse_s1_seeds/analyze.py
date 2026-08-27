"""Pre-registered analysis for sparse_s1_seeds: B*(256) point or distribution.

WRITTEN AND COMMITTED WITH THE PRE-REGISTRATION, BEFORE ANY RUN.
Decision rules fixed by PREREGISTRATION.md (licence anchor = the
commit that introduced this file):

  H-CROSS-15 (no committed direction, TWO-TAILED 2.145, df=14):
      sign of Delta(3600) over the POOLED 15 seeds.
  H-SHRINK-4 (committed direction, one-tailed 1.833, df=9; NEW seeds
      {6..15} ONLY): shrink2_s = Delta_s(3600) - Delta_s(1200) > 0.
  V-DISPERSION (pre-committed descriptive criterion, no t-test):
      pooled b=3600 mean/SD/sign-split + per-seed persistent-crossing
      budget b0_s. "Distribution" reading adopted iff |mean| < SD AND
      minority sign >= 4/15.
  D-SHAPE (descriptive): pooled mean Delta at every 400-step slice.
  Threat 1 (protocol-drift / pooling licence): NEW seeds' Delta_s(1200)
      negative on >= 7/10, else pooling VOID -> new-10-only analyses.
  Threat 2 (regime): per lane over 15 seeds, best_val >= 3 evals before
      3600 in >= 9/15 -> scope statements.
  Guard: model events must record d=256, layers=2; lanes read from
      model events, never directory names.

Layout: new seeds under $SPARSE_SEEDS_ROOT (default ~/seeds_out)/runs,
boundary seeds under $SPARSE_S1_ROOT (default ~/boundary_out)/runs.
"""
import glob
import json
import os
import statistics as st

NEW_ROOT = os.environ.get("SPARSE_SEEDS_ROOT", os.path.expanduser("~/seeds_out"))
OLD_ROOT = os.environ.get("SPARSE_S1_ROOT", os.path.expanduser("~/boundary_out"))
SEEDS_NEW = list(range(6, 16))
SEEDS_OLD = [1, 2, 3, 4, 5]
LANES = ("exact", "swa")
CUTOFF = 3600
CRIT_CROSS_15 = 2.145   # two-tailed, df=14
CRIT_SHRINK_1T = 1.833  # one-tailed, df=9
CRIT_SHRINK_2T = 2.262  # two-tailed, df=9


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


def load_root(root, expect_seeds):
    dirs = sorted(glob.glob(os.path.join(root, "runs", "run_*")))
    assert dirs, f"no runs under {root}/runs"
    table = {}
    for d in dirs:
        model, run = _read_run(d)
        lane, seed = model.get("attention"), model.get("seed")
        if seed not in expect_seeds:
            continue
        assert lane in LANES, f"{d}: attention={lane}"
        assert (lane, seed) not in table, f"duplicate {lane} s{seed}"
        table[(lane, seed)] = run
    for lane in LANES:
        for s in expect_seeds:
            assert (lane, s) in table, f"missing {lane} s{s} under {root}"
    return table


def bestval_by(run, b):
    vals = [v for s, v in run["evals"] if s <= b]
    assert vals, f"no evals at or before step {b}"
    return min(vals)


def tstat(xs):
    m, sd = st.fmean(xs), st.stdev(xs)
    return m, sd, m / (sd / len(xs) ** 0.5)


runs = load_root(NEW_ROOT, SEEDS_NEW)
runs.update(load_root(OLD_ROOT, SEEDS_OLD))


def delta(b, seeds):
    return [bestval_by(runs[("swa", s)], b) - bestval_by(runs[("exact", s)], b)
            for s in seeds]


print("=" * 72)
print("sparse_s1_seeds PRE-REGISTERED ANALYSIS -- B*(256): point or "
      "distribution")
print("=" * 72)

# --- Threat 1: protocol-drift guard (pooling licence) -----------------
d1200_new = delta(1200, SEEDS_NEW)
neg_new = sum(1 for x in d1200_new if x < 0)
print(f"\nThreat 1 (pooling licence): NEW-seed Delta_s(1200) negatives = "
      f"{neg_new}/10 (boundary + Rung B precedent: 5/5)")
if neg_new >= 7:
    POOL = SEEDS_OLD + SEEDS_NEW
    print("  pooling LICENSED: analyses on n=15")
else:
    POOL = SEEDS_NEW
    print("  pooling VOID: sign pattern drifts from the 5/5 precedent; "
          "analyses on the NEW 10 alone; the discrepancy is a finding.")
n = len(POOL)
df = n - 1
crit_cross = CRIT_CROSS_15 if n == 15 else CRIT_SHRINK_2T  # df=9 -> 2.262

# --- Threat 2: regime check -------------------------------------------
overfit_lanes = []
for lane in LANES:
    n_early = sum(
        1 for s in POOL
        if min(runs[(lane, s)]["evals"], key=lambda e: e[1])[0]
        <= CUTOFF - 3 * 100)
    print(f"Threat 2 (regime) {lane}: best_val >=3 evals before cutoff in "
          f"{n_early}/{n} seeds")
    if n_early >= (9 if n == 15 else 6):
        overfit_lanes.append(lane)
scope_note = (" [SCOPED: overfit onset in " + ",".join(overfit_lanes) + "]"
              if overfit_lanes else "")

# --- D-SHAPE ----------------------------------------------------------
print(f"\nD-SHAPE (descriptive, n={n}): mean Delta by budget slice")
for b in range(400, CUTOFF + 1, 400):
    ds = delta(b, POOL)
    print(f"  b={b:4}: {st.fmean(ds):+.5f}  (SD {st.stdev(ds):.4f})")

# --- H-SHRINK-4 (NEW seeds only, always) ------------------------------
d3600_new = delta(CUTOFF, SEEDS_NEW)
shrink2 = [d3600_new[i] - d1200_new[i] for i in range(len(SEEDS_NEW))]
ms, _, ts = tstat(shrink2)
print("\n" + "-" * 72)
print("H-SHRINK-4: Delta(3600) - Delta(1200) > 0, NEW seeds {6..15} only "
      "(direction committed pre-data for the FOURTH time)")
print(f"  per-seed {[f'{x:+.4f}' for x in shrink2]}")
print(f"  mean {ms:+.5f}  t = {ts:.2f}  df = 9  one-tailed crit "
      f"{CRIT_SHRINK_1T}  two-tailed {CRIT_SHRINK_2T}")
if ms > 0 and ts > CRIT_SHRINK_1T:
    two = "ALSO two-tailed" if ts > CRIT_SHRINK_2T else "one-tailed only"
    print(f"  VERDICT: SUPPORTED ({two}) -- fourth replication{scope_note}")
elif ms <= 0 and abs(ts) > CRIT_SHRINK_1T:
    print(f"  VERDICT: FALSIFIED IN DIRECTION{scope_note}")
else:
    print(f"  VERDICT: NOT ESTABLISHED{scope_note}")

# --- H-CROSS-15 -------------------------------------------------------
d36 = delta(CUTOFF, POOL)
m36, sd36, t36 = tstat(d36)
print("\n" + "-" * 72)
print(f"H-CROSS-{n}: sign of Delta(3600), n={n} -- TWO-TAILED ONLY")
print(f"  Delta(3600) = {m36:+.5f}  SD {sd36:.4f}  t = {t36:.2f}  "
      f"two-tailed crit {crit_cross}")
if m36 > 0 and t36 > crit_cross:
    print(f"  VERDICT: H-CROSS -- B*(256) in (1200, 3600]{scope_note}")
elif m36 < 0 and abs(t36) > crit_cross:
    print(f"  VERDICT: H-NO-CROSS -- B*(256) > 3600{scope_note}")
else:
    print(f"  VERDICT: sign undetermined at n={n}; V-DISPERSION carries "
          f"the report{scope_note}")

# --- V-DISPERSION -----------------------------------------------------
pos = sum(1 for x in d36 if x > 0)
minority = min(pos, n - pos)
print("\n" + "-" * 72)
print("V-DISPERSION (pre-committed descriptive criterion)")
print(f"  Delta(3600): mean {m36:+.5f}  SD {sd36:.4f}  sign split "
      f"{pos}+/{n - pos}-  minority {minority}")
b0s = {}
for s in POOL:
    slices = list(range(400, CUTOFF + 1, 400))
    dd = {b: delta(b, [s])[0] for b in slices}
    b0 = None
    for b in slices:
        if all(dd[bb] > 0 for bb in slices if bb >= b):
            b0 = b
            break
    b0s[s] = b0
print("  persistent-crossing budget b0_s per seed:")
print("   " + "  ".join(f"s{s}:{b0s[s] if b0s[s] else 'none'}"
                        for s in POOL))
never = sum(1 for v in b0s.values() if v is None)
print(f"  seeds never crossing by 3600: {never}/{n}")
if abs(m36) < sd36 and minority >= (4 if n == 15 else 3):
    print("  READING ADOPTED: B*(256) is a DISTRIBUTION at this protocol's "
          "resolution -- the crossover budget varies across seeds by more "
          "than its mean offset.")
else:
    print("  READING NOT ADOPTED: point-reading stands; any H-CROSS "
          "failure is attributed to power, not dispersion.")

print(f"\nScope: TinyStories slice, gpt2-nano family, d=256, L=2, w=64+sink,")
print(f"T=256, n={n} paired seeds, CPU numerics, house protocol. B*(d) is a")
print("property of THIS protocol; the general claim remains budget-reporting")
print("discipline.")
