# Decisions — why things are the way they are

*Short records of judgement calls, so they are neither re-litigated by
the next person nor silently reversed. One heading per decision: what
was chosen, what was rejected, and what would change the answer.*

Most of these were made with reasoning that otherwise lives only in
commit messages, which nobody reads a month later.

---

## D1. The events.jsonl contract is the interface, not the trainer

**Chosen:** every consumer (studio, sweeps, Atlas extraction, the seed
lottery, every pre-registered analysis) reads `events.jsonl` and nothing
else. A run whose events file is complete is fully analysable even if
the trainer that produced it no longer exists.

**Why:** it lets contributions arrive from engines and hardware we will
never own. `tools/coalfire_events.py` is the existence proof — a ~100
line sidecar translating a foreign trainer's human log buys that trainer
the entire toolchain without patching it.

**Would change if:** a consumer needed something the contract cannot
express. Add a field (consumers must ignore unknown keys); do not add a
second interface.

---

## D2. Receipts must name their own binary

**Chosen:** `mtsweep` writes `provenance.json` — commit, dirty flag,
binary sha256 — beside every `result.json`. `--require-clean` refuses to
run from a dirty tree, which is the setting for any pre-registered
experiment.

**Why:** found the hard way. A full night of Colab sweeps ran a CPU-only
`mtstudio` because the CUDA wiring existed only in an uncommitted
working copy, and nothing in any receipt could have revealed it. A
registry whose receipts cannot name their binary is one silent rebuild
from unreproducible.

**Known debt:** the S1e receipts predate this and carry a note saying so.
The numbers stand; the reproducibility does not.

---

## D3. Test the analysis before the data exists

**Chosen:** every `analyze.py` ships with a `smoke_analyze.py` that
fabricates the run layout and drives each decision rule through all of
its branches, including the positive ones.

**Why:** it has caught three errors that would each have produced a
wrong published verdict — a gate that could not read lanes at all; a
rule that would have called a FIRST crossing a second (reporting a
theorem falsified on data that never falsified it); and a primary
hypothesis never shown capable of firing positively. None of these are
visible once real data is in. You simply get a verdict.

**The sharpest version:** a decision rule that can never fire is worse
than one that is wrong, because it reports "not adopted" whatever the
world does.

---

## D4. A licensed rule may be amended only pre-data, and only in writing

**Chosen:** the commit introducing `PREREGISTRATION.md` + `analyze.py`
is the licence anchor. Changing a rule afterwards requires a dated
**Amendment** section in that file carrying the reasoning — never a
silent edit. Lesser changes that alter no rule go in
"Execution clarifications".

**Live example:** transfer_s1 Amendment 1. The bridge gate demanded
per-seed sign agreement at 3600 steps; since a backend change behaves
like a reseed, and S1e measured p(+) = 0.60, two independent draws agree
with probability 0.52 and P(>=4/5) = 0.21. **The gate would have halted
the study four times in five on a perfect engine.** It was replaced —
pre-data, at 0/10 runs — with an early-step numerics criterion that is
STRICTER about what the bridge exists to test.

**The guard that matters:** loosening a gate is the amendment most
deserving of suspicion. If you must, make the replacement stricter on
the thing the check was for, and show both directions still work.

---

## D5. Ground truth is read, never recalled

**Chosen:** in `papers/flavor_bench.py`, every truth-set entry is taken
from the fetched paper source, and a field a paper does not state is
**omitted** rather than inferred from a sibling model.

**Why:** an omitted field costs the benchmark a little coverage; a
guessed one poisons it. InternLM2's norm/activation are left blank
though it is RMSNorm/SwiGLU in fact, because its paper states them only
in a related-work sentence about LLaMA — scoring that as InternLM2's own
choice would reward exactly the confusion the Megatron case penalises.

---

## D6. Known failures are registered, not hidden or excused

**Chosen:** `KNOWN_WRONG` holds each wrong assertion with its diagnosis
and a named fix. The gate still fails on any NEW wrong assertion, and an
entry that stops reproducing is reported as **FIXED** with an
instruction to delete it.

**Why:** growing the truth set from 26 to 40 papers broke a scorer that
had looked flawless. A benchmark reporting zero errors is reporting that
its test set is too easy. The "zero wrong assertions" claim was
corrected everywhere it appeared rather than quietly dropped.

---

## D7. The banked CPU cohort is the bridge, not the S arm

**Chosen:** re-run arm S in full on CUDA (~6 T4-hours) and use the
banked 15-seed CPU cohort as the numerics-bridge reference instead.

**Why:** reusing it as the S arm would put CPU-S against CUDA-M inside
the *primary* comparison, confounding width with backend. As a bridge
reference it does more work: it converts "did the engine change
anything?" from an assumption into a measurement against pre-registered
prior data.

**Would change if:** Jonathan would rather have the six hours. Listed in
[`../open/FOR_JONATHAN.md`](../open/FOR_JONATHAN.md).

---

## D8. Colab: the relay unit is one run, and the binary is cached

**Chosen:** `tools/colab_transfer_runner.py` relays a run directory at a
time and pushes completed runs back on relaunch, so mtsweep skips them.
The CUDA build — the real cost of a relaunch — happens once and the
binary is cached locally.

**Why:** supervisor.py's admission rule, applied literally: a reclaim
must cost minutes, not the run. Reclaims were observed mid-build twice
in one night. Caching turns a reclaim from ~10 minutes into ~1.

**Related:** `tools/colab_reap_orphans.py` kills unnamed sessions, the
one zombie class supervisor.py's registry structurally cannot reach —
`stop` authenticates on the control plane, so a scratch config carrying
only the endpoint is enough.
