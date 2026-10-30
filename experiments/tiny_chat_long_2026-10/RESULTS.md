# Carry-12 registered core — 30 October 2026

The complete seven-cell core has reported seeds 1, 2 and 3. The unchanged registered
analysis ran through the existing watchdog finalizer, with `partial: false`, no missing
cells and every validity gate passing. Independent verification rehashed all 36 frozen
sources, 55 inputs and three published reports without a mismatch. The exact
[analysis and completion receipts](receipts/core_2026-10-30/) preserve the source and
input inventory, command, output hashes and independent audit. Earlier partial analyses
remain historical records.

The clear registered result is **HT-SFT: adding the disjoint human-written template SFT
improved Carry-12's carry metric by 9.16 percentage points over B**. All three paired
differences were positive (10.29, 8.26 and 8.92 points). This is a result on this small
model, training recipe and benchmark, not a state-of-the-art or general dialogue claim.
The secondary trained/untrained family analyses do not resolve whether that improvement
transfers to structures absent from the templates.

## Cell means and the metric

Y12 is self-history forced-choice accuracy minus twin accuracy over recall, correction,
topic, load and knowledge. A less negative value means a smaller history cost relative
to each model's own twin accuracy; raw accuracy is therefore reported alongside it.
All entries below are percentages or percentage-point differences, averaged over three
seeds. W512 was registered for gold-history scoring, so it has no self-history Y12.

| Cell | Self accuracy | Twin accuracy | Y12 | Gold-history Y12 |
|---|---:|---:|---:|---:|
| W512, original context 512 | — | 49.61 | — | −5.99 |
| B, context 1024 | 42.90 | 52.37 | −9.47 | −8.48 |
| D, long human SFT weighting | 42.83 | 52.13 | −9.30 | −8.12 |
| TS, explicit template SFT | 57.22 | 57.54 | −0.32 | −0.58 |
| S-gist, per-turn state | 31.84 | 52.76 | −20.92 | −19.42 |
| P16, plain depth-16 control | 45.53 | 54.30 | −8.77 | −7.51 |
| X, Attention Residuals | 46.23 | 52.10 | −5.87 | −4.11 |

The validity gates require mean twin accuracy ≥35% and twin minus question-alone
accuracy ≥10 points over context families. Every cell passes. W512 reproduced E1's
pretraining and SFT weight hashes in all three seeds, with zero FineWeb panel difference.

## Frozen decisions

These are the unchanged analysis's registered verdicts, not judgments made from the
table's rounded means. SUPPORTED requires the registered direction, a two-sided 5%
paired t test and the registered effect-size threshold; equivalence uses a 90% interval.
The intervals below are the analysis's 90% intervals in percentage points.

| Contrast | Mean | 90% interval | Registered verdict |
|---|---:|---:|---|
| HT: B − W512, gold-history Y12 | −2.49 | [−7.44, 2.46] | NOT_RESOLVED |
| HD: D − B | 0.17 | [−3.26, 3.60] | NOT_RESOLVED |
| HT-SFT: TS − B | 9.16 | [7.41, 10.90] | SUPPORTED |
| HS1: S-gist − B, predicted negative | −11.45 | [−19.73, −3.16] | NOT_RESOLVED |
| HX-main: X − P16 | 2.89 | [−5.53, 11.32] | NOT_RESOLVED |

S-gist is lower than B in every seed (−13.33, −15.14 and −5.87 points), but its
absolute t statistic is 4.035, below the frozen 4.303 cutoff. Its 90% interval does
not replace that registered test. The analysis flags a five-seed extension; **S-full
does not trigger**, because HS1 is not SUPPORTED. No conditional paid campaign was
launched from this result.

HX's two distance tests are also NOT_RESOLVED: the far-minus-near recall interaction
is −0.14 points, and the correction-distance slope is −0.20 points per turn. The
registered joint H-gap is consequently NOT_RESOLVED. X's FineWeb panel loss is lower
than P16's by 0.05883 nats (the secondary test is SUPPORTED), but this does not establish
a carry benefit. Its isoplot residual is unresolved and extrapolates beyond the two
plain cells' loss range in every seed; the analysis flags those limitations.

## What the TS result does and does not establish

TS improves both absolute self accuracy (57.22% versus B's 42.90%) and the carry metric;
the result is not merely a reduction in twin accuracy. The secondary gains in trained
families (recall/correction: 10.08 points) and untrained families (topic/load: 7.92 points)
are individually NOT_RESOLVED, as is their difference. All three seeds have positive
gains in each pool, and the shared decision helper emits extension flags for both.
Section 9 limits the registered extension rule to primary contrasts: those secondary
flags do not authorize additional campaigns or establish transfer.

The checks beside HT-SFT do not show supported degradation: Carry-5's difference is
+5.11 points but unresolved; SFT dialogue loss changes by −0.01838 nats and is unresolved;
FineWeb loss changes by −0.01188 nats, contradicting the registered worsening direction.
Descriptive chat checks are mixed: clean stops are unchanged on chat and 2.45 points
higher on chat2, while repetition rises by about 2 points and distinct bigrams fall by
0.76/0.91 points. QA's overall accuracy is unchanged at 16.67%. Instruction compliance
remains near zero (B 0%, TS 0.46%). These are narrow diagnostics, not evidence that TS
has become a broadly useful conversational model.

## Remaining work at the three-seed core receipt

The three-seed core is complete. HS1 alone meets section 9's primary-contrast extension
rule (NOT_RESOLVED and one sign in all three seeds): it requires B and S-gist at seeds
4 and 5, once. The shared helper also flags secondary comparisons, but those are outside
that launch rule. Exact flags and per-seed values remain in the JSON; follow-up scope
and spending require review before launch. Stage L0 on the registered 150M pick and the
matched five-seed size extension are separate campaigns and were still running when
this core receipt was written.


## Registered HS1 extension: all five matched seeds

Completed 30 October 2026. Only B and S-gist gained seeds 4 and 5, as section 9
required after the unresolved, same-sign three-seed contrast. The complete seven-cell
three-seed core above remains unchanged; the five-seed report is a separate registered
HS1 contrast, not a five-seed reanalysis of the other arms.

**HS1 is now SUPPORTED:** S-gist minus B is -12.668 percentage points of Y12, with
90% CI [-16.353, -8.983], paired t=-7.329 against the unchanged df=4 cutoff 2.776
and the three-point SESOI. The five differences are -13.33, -15.14, -5.87, -14.87
and -14.13 points. Both validity gates pass.

**History-parity limitation:** a subsequent complete receipt audit found six
nonidentical full-history argmax checks among 174,240 recorded reply checks. The
engine spot checks passed, but that does not mean every full-history check passed.
B seed1 has three failures, B seed3 one, S-gist seed1 one and S-gist seed4 one;
all are distinct primary-family items, before the target turn, with EOS stops and
no dropped histories or server errors. The saved boolean flags do not retain the
scorer argmax vectors or logit margins, so a near-tie/rounding explanation is not
established. The numerical registered verdict above is preserved as computed;
it must not be described as clearing complete greedy-history parity. The
[exact affected-item audit](receipts/hs1_five_seed_20261030/history_argmax_audit.json)
records this protocol limitation. S-full launch remains blocked on review of
this limitation as well as funding; no threshold or history has been changed.

A separate [posthoc sensitivity bound](receipts/hs1_five_seed_20261030/parity_sensitivity.json)
allows each of those six affected primary-item self-history credits to vary anywhere
in [0,1], holding all other items and twin credits fixed. The primary denominator
is 1,380 per run, and the bound includes the registered four-decimal pool rounding.
The paired mean stays between -12.684 and -12.596 points (at most a 0.072-point
change); a conservative sample-SD bound gives |t| at least 7.008, still above 2.776.
Thus those six bounded item-credit changes alone cannot reverse the numerical rule.
This does not establish the mismatch cause, bound unrecorded defects, or repair the
historical parity violation. Canonical observations and analysis remain unchanged.
Replay uses the adjacent pinned public receipts and the retained raw artifacts:
`python audit_parity_sensitivity.py --art-root PATH` (optional `--output NEW.json`).
The receipt pins the report, affected-item audit, raw-item inventory, completion,
every used item/summary file and replay-source hashes; no temporary file is required.

| Five-seed cell | Self accuracy | Twin accuracy | Y12 | Gold-history Y12 |
|---|---:|---:|---:|---:|
| B | 43.014% | 52.522% | -9.508 points | -8.536 points |
| S-gist | 31.116% | 53.292% | -22.176 points | -20.696 points |

The tested per-turn state arrangement reduces self-history performance while twin
accuracy remains similar. This comparison does not locate the cost between inserting
state slots and imposing the state mask. The registration's **S-full condition is now
true**: that control is needed to separate those explanations. It has not been launched;
its exact recipe and spending await review. No additional seed extension is allowed
for this contrast. The broader long-carry programme is therefore not complete.

The watchdog finalizer published at 13:53:00 UTC. Independent verification rehashed all
56 frozen sources, 69 inputs and three reports, checked exactly ten cell/seed summaries
and recomputed the paired result without importing the registered statistical helpers.
An additional replay of all 14,520 saved items recomputed 72,600 forced-choice credits
from their four raw log-prob sums, frozen gold index and overflow flag. These reproduce
the frozen summaries' four-decimal pool rounding; the registered contrast uses those
unchanged summary values. This checks recorded score arithmetic, not fresh model forwards.
All four extension campaigns are home, with their pods and volumes removed.

S-gist seed 5 required an operational dependency repair and resumed a legacy data cache.
The original v1 finalizer halt is preserved. Before its post-training outcomes, v2 froze
a narrowly bound exception for that exact resumed-ledger hash and independently checked
source/decontamination/output proof; the statistical helpers and rules stayed unchanged.
The historical builder blob is retained rather than substituted with the later cache fix.
See [environment evidence](receipts/hs1_environment_2026-10-30/) and the exact input/source
inventory in the [five-seed receipts](receipts/hs1_five_seed_20261030/).

Canonical [analysis](receipts/hs1_five_seed_20261030/analysis_hs1.json),
[completion manifest](receipts/hs1_five_seed_20261030/completion.json),
[independent audit](receipts/hs1_five_seed_20261030/independent_audit.json),
[raw-credit audit](receipts/hs1_five_seed_20261030/raw_credit_audit.json) and
[home proof](receipts/hs1_five_seed_20261030/home_receipt.json) are retained here.
The single raw-artifact home remains `C:/ml_artifacts/tiny_chat/`.


## Stage L0 on the original 150M pick

The last pending 150M pick evaluation is complete and home, with no training or
model reselection. This is **one descriptive pick**, at its original context 512,
not another five-seed comparison or a new confirmatory scaling result. The frozen
Stage L0 command scored all 1,452 Carry-12 items in gold, near, twin, no-context,
self and near-self modes; greedy history ends at the target and uses max-new 24.

Its validity gate passes: primary twin accuracy is 48.623%, and context-family
twin minus no-context accuracy is 23.788 points. Gold accuracy on the 1,320 context
items is 42.121%, against 46.818% with near history. Thus the recorded distance
cost D=gold-near is **-4.697 points**, normal-approximation 95% CI
[-7.069,-2.325] and 90% CI [-6.688,-2.706]. The frozen helper labels it SUPPORTED
against its negative-direction, three-point rule. For this additional 150M pick
that verdict is descriptive: it does not extend the originally named winner tests.
Items share templates and values, so these item-level intervals may be too narrow;
near history changes length as well as distance.

Descriptively, primary self accuracy is 43.12%, twin accuracy 48.62%, and the
frozen rounded-pool Y12 is -5.50 points. These values do not establish a general
model-quality or state-of-the-art comparison. The model is the unchanged original
150M/depth-8 Carry-5 pick, not a pick made using these L0 results.

Independent verification rehashed its original weights, GGUF, spec and vocabulary,
bound the job to source pin `82c86c96bd0086ae47e35ab31370b5fb98ae28ed` and checked
10 selected registered source/data blobs. Replaying all 8,712 forced-choice credits
from saved raw sums reproduces D, its intervals and the gate. The recorded generation
gate passes all 22,956 reply checks, and the 20-item spot check matches all 404 replies;
this audit checks those retained receipts, not fresh model forwards. The campaign is
home, with pod and volume gone and provider absence checked. Its cumulative cost was
$6.973 under the $10 recovery cap, including the original interrupted attempt.

The original partial run was lost in the documented volume/sweeper incident. This
receipt is the explicitly authorized recomputation of the same frozen recipe and
inputs; original failure and recovery records remain unchanged in
[incident evidence](receipts/l0_volume_loss_20261030/). No lost partial scores were
presented as recovered results.

Exact [L0 receipt](receipts/l0_150m_20261030/l0.json),
[full summary](receipts/l0_150m_20261030/summary.json),
[independent source/input/raw-credit/home audit](receipts/l0_150m_20261030/independent_audit.json)
and [raw-credit replay](receipts/l0_150m_20261030/raw_credit_audit.json) are retained.
Raw generation and scored-item records have one home under
`C:/ml_artifacts/tiny_chat/l0_colab/l0_pick_e1_150m_L8/pick_e1_150m_L8/`.

### Conditional S-full packet prepared, unlaunched

The offline packet is **READY_FOR_FUNDING_REVIEW_WITH_RUNTIME_DATA_GATE**, not authorised
for an unconditional launch. [Exact commands, hashes and limits](receipts/sfull_prospective_20261031/readiness.json)
are bound to archive `7edb0ca1b252bd9332094558a478d924568f8e52`, whose exact parent is the
frozen `3dc818b2476b0957b1337b4c9af22474b76350b5`. Its only delta is twelve new paths:
three S-full seed jobs, the L8 causal-mask spec and operational proof/verifier/boot files.
Existing scientific source, registration, helpers, core results and five-seed HS1 stay unchanged.
The spec clones S-gist L8 (d768, h12, d_ff2240, k4), changing only turn_state to causal;
it does not use the unsupported Sfull/L16 fallback in make_jobs.

The three initial seeds run **serially**, each with its own $7 cumulative cap and 23.5-hour
operational clock: **$21 total**, expected **18–27 GPU-hours combined**, approximately $11–16
before incidental storage. The combined estimate is not squeezed into one 23.5-hour campaign.
Any later continuation requires explicit funding and remaining-budget review; no cap increase
or new arm is automatic. The proposal preserves a $20 reserve; a top-up offer is not credited funds.

The thirteen original human-source corpora and five finished data products are unavailable
locally. Their expected byte hashes, the exact reference ledger, pinned vocabulary and
`tokenizers==0.23.3` are specified. A future frozen rebuild must match every source, product,
evaluation and decontamination hash **before pretraining**; mismatch blocks training, with no
fallback or automatic rerun. Offline readiness does not claim those absent data bytes verified.

The separate prospective HS0 finalizer retains the unchanged registered statistical helpers,
initial seeds1–3 and full historical parity disclosure. New S-full raw checks must all pass,
with each item's flag count matching its registered target turn; six historical HS1 failures
remain a review blocker, not a waived threshold. The once-only smoke check is WAITING on thirty
missing inputs and produced no analysis. The [offline validation](receipts/sfull_prospective_20261031/offline_validation.json)
records 35 passing tests and the frozen source inventory. All proposed entries remain disabled;
no compute was rented and no S-full training or result exists.
