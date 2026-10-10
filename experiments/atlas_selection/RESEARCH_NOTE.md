# What the decision audit adds

28 September 2026. Post-hoc reading of `results_v2.json`, verified by replay.

The source ranking's failure has a measurable decision consequence in the
existing transfer_s2 panel. At step 3600, every one of the 924 six/six seed
splits shortlists **swa16s1 and swa32s1** when k=2. Target calibration always
selects swa32s1. Its held-out loss exceeds exact attention by 0.118044 and the
expected uniformly random two-candidate shortlist by 0.097774. These are
validation-loss differences, not percentage changes.

The two-candidate policy has a mean gap of 0.124105 to the best lane on the
held-out panel. The entire gap is exclusion: calibration always chooses the
better of the two available lanes on the held-out seeds as well. More accurate
calibration *within that shortlist* cannot repair the discarded alternatives.
This is an algebraic description of these archived decisions, not a new causal
or population claim.

At k=3, exact attention enters 914 of 924 shortlists and is selected whenever
present. The remaining selections are swa128s1 (9) and swa64s0 (1). Mean excess
loss versus exact falls to 0.000161. Direct selection from all six target lanes
has a small positive mean excess versus exact (0.002587), illustrating that a
larger candidate set also introduces selection noise. The tiny k=3 advantage
over direct selection is not evidence of general policy superiority.

The all-twelve-seed endpoint means independently explain the pattern:

| Lane | Source width 256 | Target width 512 |
|---|---:|---:|
| exact | 3.231771 | 3.228542 |
| swa16s1 | 3.211812 | 3.487161 |
| swa32s1 | 3.205886 | 3.346586 |
| swa64s1 | 3.256960 | 3.263104 |
| swa128s1 | 3.241961 | 3.225650 |
| swa64s0 | 3.261278 | 3.273300 |

These endpoint means explain the result; they are **not** used to choose within
the held-out analysis. No new training or unseen corpus was evaluated.

## Consequence for the next experiment

The useful research target is **whether ATLAS can identify when pruning is
unsafe**, with acquisition cost included. Simply calibrating a source top-two
shortlist fails here because the relevant choices are gone before calibration
starts. All nine source checkpoints yield a positive mean top-two loss contrast
against random shortlisting at the final target endpoint.

Keeping winners from several training positions is only a candidate remedy.
The present source winners are narrow windows, so temporal diversity alone
may preserve the same mistake. A prospective comparison should include an
exact-anchored shortlist and a structurally diverse shortlist alongside temporal
diversity, source top-k, uniform random, and target-only allocation. Selecting
exact as an anchor here is informed by these results and must be evaluated on
fresh tasks/panels before making a general claim.

The desired output is a policy with measured loss and compute cost, or a clear
boundary showing that this fingerprint cannot justify pruning. Both outcomes
advance ATLAS's proposed role in guiding experiments. The current contribution
is the executable evaluation layer and a concrete failure case; the proposed
remedy remains untested.

## Verification and provenance

- 13 synthetic tests passed on Windows Python; Ruff passed.
- All 144 input receipts passed the scoped reader and have recorded SHA-256s.
- Replay of v2 matched source/protocol hashes and all 108 cells exactly.
- `results_v1.json` was produced during a stalled terminal session. It is retained;
  v2 adds progress logging to the source, with identical numerical analysis.
  Use v2 for replay against the current source.
- No changes to the parent's preregistrations, frozen analysis, or findings registry.
- WSL initially worked, then returned HCS_E_CONNECTION_TIMEOUT. Native Python
  completed the analysis; no WSL restart, new Colab session or GPU charge was needed.
