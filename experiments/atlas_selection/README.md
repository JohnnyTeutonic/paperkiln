# ATLAS decision audit: does a proxy help choose?

Contribution started 28 September 2026. **Post-hoc archival analysis**, designed
with the transfer_s2 findings already known. Neither a preregistration nor a
new independent replication. Frozen transfer studies and their claims are
untouched. The intended contribution is a decision-evaluation layer for ATLAS.

## Research question

The transfer study asks whether comparison signs survive a width change.
This analysis asks what happens when someone **acts** on a small-width panel:
does its shortlist improve a target-width choice over an equally sized random
shortlist? How much error arises from excluding useful candidates, and how much
from noisy selection among retained candidates?

A poor overall ranking can still retain a useful candidate. Conversely, a
mostly correct ranking can make an expensive mistake at its top. Decision
consequences therefore complement, rather than replace, the frozen transfer
analysis. This is an existing family of questions in multi-fidelity optimisation
and NAS; we make **no novelty claim for regret, shortlisting or cross-validation**.
The local research opportunity is a receipt-backed study of how this decision
changes with training position in a panel already known to change its ranking.

## Exact design

`protocol.json` records all choices. Read all 72 S and 72 M runs from transfer_s2
(six attention lanes, twelve paired seeds, widths 256 and 512; selected learning
rates 0.00025 and 0.0005). Do not use L or exploratory Lx as if they had six lanes.

Enumerate all 924 ways of assigning six seeds to selection/calibration and the
remaining six to scoring. Withhold the **same seed IDs at both widths**, preventing
selection from using the source version of a held-out target seed. For each split:

1. Rank the six source lanes by mean validation loss on the selection seeds.
2. Retain the first k lanes, k in {1, 2, 3, 6}. Resolve ties by manifest order.
3. Select the lowest mean target loss within this shortlist on calibration seeds.
4. Score that chosen lane on the six held-out target seeds.

At k=1, step 3 needs no target observations. At k=6, the result must equal direct
target selection, providing a built-in control.

Evaluate every source step in {400, 800, ..., 3600} and target endpoint in
{1200, 2400, 3600}: 108 cells. The reference cell is source 3600, target 3600,
k=2; it was designated before running this new script but **after** the parent
results were known. All cells are reported. No search for the best source step
is evaluated as though that step had been selected in advance. These are
fixed-step decisions, not matched-loss comparisons.

Comparators:

- Always choose exact attention (no search).
- Direct target selection among all six lanes using the same calibration seeds.
- Uniform random size-k shortlist, followed by the same target calibration rule.
  Compute its expectation exactly over all combinations; no Monte Carlo noise.

The informed shortlist has additional source acquisition cost. Counts are
reported separately as source and target run-steps. They are **not** FLOPs,
elapsed time, or a cost-matched demonstration. Historical rate selection costs
are sunk and excluded for every policy. A future budgeted study must include
them if it repeats rate selection.

## Estimands and interpretation

For a split, let H_j be target held-out mean loss, A the source shortlist, and
c its target-calibration-selected candidate. Compute:

    excess_vs_exact = H_c - H_exact
    hindsight_gap  = H_c - min_j H_j
    exclusion_gap  = min_(j in A) H_j - min_j H_j
    calibration_gap = H_c - min_(j in A) H_j

The last two are nonnegative and sum exactly to hindsight_gap. They distinguish
discarding good choices from failing to choose them. The held-out minimum is
an optimistic finite-panel oracle; this is **not unbiased population simple
regret**. Paired excess loss versus exact, random shortlisting, and direct
target selection are the main practical contrasts; they can be negative.

Each policy's choice is independent of its scoring seeds conditional on the
fixed archival study. However, every run uses the same validation corpus, and
the learning rates and analysis direction were chosen using prior information.
This evaluates seed generalisation, not unseen data, unseen architecture
families, or prospective discovery.

The 924 splits overlap heavily and contain only twelve distinct seeds.
`split_p05`/`split_p95` are **descriptive split-sensitivity percentiles**, not
confidence bounds. No t-test treating splits as independent is permitted.
Random-policy percentiles describe the split-wise *expected* random policy,
not individual random shortlist realisations.

## Reproduce

From this directory, Windows Python or WSL Python (standard library only):

```sh
python3 -B -m pytest -q
ruff check selection.py test_selection.py
python3 -B selection.py run --output new_report.json
python3 -B selection.py verify results_v2.json
```

`run` refuses an existing output. `verify` recomputes every cell and compares
the complete report, including source-code and protocol hashes and all 144
input hashes. Source files are expected to remain available for replay.
The JSON contains the complete protocol, ordered split digest, selection and
inclusion counts, all baseline summaries, and every cell's budget accounting.
The loader rejects missing/extra/duplicate seeds or lanes, missing evaluations,
non-finite losses, inconsistent resume metadata, conflicting duplicate evals,
incomplete runs, and out-of-scope models. Lane identity comes from model events.

Banked outputs: `results_v2.json` (replayable current report), `RESULTS.md`
(generated tables), `RESEARCH_NOTE.md` (interpretation), and
`selection_audit.png` / `.pdf` (static figure). `results_v1.json` is retained
from the first execution; its numerical analysis equals v2, whose only source
change was progress logging. Run `render_report.py results_v2.json --output
new_results.md` to render another Markdown copy. `plot_results.py` requires
matplotlib and refuses to overwrite its two figure files.

## Literature anchors checked for this contribution

- [Abdelfattah et al., Zero-Cost Proxies for Lightweight NAS (2021)](https://arxiv.org/abs/2101.08134):
  evaluates ranking proxies and their use within search. Establishes that
  evaluating a proxy through downstream selection is not a new research theme.
- [Sen, Kandasamy and Shakkottai, Multi-Fidelity Black-Box Optimization with
  Hierarchical Partitions (2018)](https://proceedings.mlr.press/v80/sen18a.html):
  develops multi-fidelity optimisation with simple-regret guarantees. Our
  finite-panel decomposition claims no such population guarantee.
- [Pfisterer et al., YAHPO Gym (2022)](https://proceedings.mlr.press/v188/pfisterer22a.html):
  benchmarks multi-fidelity optimisation and discusses how benchmark design
  can affect conclusions about optimisation methods. A single six-lane panel
  cannot establish general search-policy superiority.

## Next research step

The highest-upside extension is a **prospective test of when to trust pruning**:
whether a source panel's changing winners across training positions predicts
that its shortlist will exclude the target winner. A position-diverse shortlist
(retain winners from different source positions) is a candidate policy, with
top-k, random, and exact-anchored shortlists as controls. This is a proposal,
not a claim that diversity works.

Use this archive for development only. Freeze the policy and analysis before
fresh runs; compare on a new panel or corpus with new seeds. Charge all source
checkpoints, target calibration, and rate selection. Measure held-out excess
loss at matched total compute, including a target-only policy allowed to spend
the saved source compute. Report elimination error and calibration error
separately. A failure of position diversity would still constrain the kind of
information ATLAS needs before it can safely guide architecture search.
