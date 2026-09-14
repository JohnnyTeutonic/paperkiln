# Paper spine: a pre-registered positive, its pre-registered refutation (14 Sep 2026)

*Target: TMLR (double-blind, any length justified by content, TMLR LaTeX
style, up to 100 MB anonymised supplementary, reproducibility
certification available). Author's decision 14 Sep 2026: a methodology
paper that records the journey; the result is what the journey caught.
This is the spine: every section with its content, its numbers and the
receipt each number comes from. Prose comes after the L arm lands and
the literature scoping (`LITERATURE.md`) is in. Commonwealth spelling
throughout; TMLR does not impose US spelling.*

## Working titles

1. *The Confounder Both Arms Shared: A Pre-Registered Transfer Result and
   Its Pre-Registered Refutation*
2. *Does the Ranking Transfer? Pre-Registration, Falsifiers and Seeds in a
   Small-Scale Attention Panel*
3. *What a Tiny Panel Can License: A Two-Study Pre-Registered Test of
   Cross-Width Transfer, and How It Failed*

## Abstract (draft, about 180 words)

Small-scale panels are routinely used to rank architectural choices
before committing compute at scale. We report a pre-registered two-study
test of whether such a ranking transfers across model width, and we
report it in the order it happened. Study 1 compared six attention
variants (exact attention and sliding-window attention with four window
sizes and an optional sink token) at widths 256 and 512 under a learning
rate held fixed across widths, and found the sign of every one of the
fifteen pairwise comparisons preserved (concordance 1.000, twelve seeds,
bootstrap band [0.867, 1.000]) at the one validation-loss milestone both
widths reached. Study 2, pre-registered with a falsifier before any run,
set the learning rate per width by a mechanical rule; the loss curves
then aligned across all five milestones, the concordance fell to 0.627
(band [0.467, 0.667]) and the Spearman correlation of effect sizes across
widths was -0.60: the ranking reversed. The positive result was produced
by the confounder the two widths shared. We identify which protocol
elements caught it (matched-loss milestones, a committed foil, twelve
seeds with bootstrap bands, a falsifier named in advance) and which one
misbehaved (a fallback clause in the rate-selection rule), and we
release every run's receipt.

## Contributions (the list a referee scans)

1. A pre-registered positive transfer result with its scope stated
   exactly (one reached milestone), and the pre-registered diagnosis of
   why the scope was that narrow (fixed learning rate mistunes the larger
   width; the sign pattern is itself rate-sensitive).
2. A pre-registered sequel whose falsifier fired: with curves aligned by
   a rule fixed in advance, sign concordance 0.627 and effect-size
   correlation -0.60 across widths.
3. The methodological account: each protocol element and the specific
   way the positive would have survived without it. The paper's
   argument is that all of them were necessary.
4. An honest failure of the pre-registration itself (the fallback clause
   at d = 1024) and the protocol's handling of it, dated.
5. Receipts: every run's event log, the analysis script frozen at the
   licence commit, the amendment history, and the runner that made the
   study reproducible across Colab pre-emption (bit-identical
   checkpoint-resume).

## The spine, section by section

### 1. Why this question, and why this way

- The practice: screen mechanisms at small width, extrapolate the
  ranking. State it neutrally; cite the proxy-transfer literature
  (LITERATURE.md group 2) and the tuning-changes-rankings literature
  (group 3).
- The measurement: the comparison graph. Six lanes, fifteen edges, each
  oriented at a training position by seed-majority sign of the
  validation-loss difference. Two layers: orientation (which wins) and
  magnitude (by how much, when the crossover comes). Define once.
- The question: does orientation transfer across width when position on
  the curve is matched by loss rather than by step?
- The design decision that makes the paper: pre-register, with a named
  falsifier, and report in order.

### 2. Protocol (shared by both studies)

- Family: gpt2-nano, two layers, T = 256, batch 4, 3600 steps,
  TinyStories slice, chat7b vocabulary capped at 4096; widths 256, 512,
  1024 (heads 8, 16, 32). L4 GPU throughout, no mixing (clarification 5).
- Lanes: exact; swa w in {16, 32, 64, 128} with sink; swa w = 64 without
  sink. Twelve seeds (21 to 32) at S and M; L three seeds in study 1,
  twelve in study 2.
- Matched milestones: the S-arm exact-lane per-seed median validation
  loss at slices 800, 1600, 2400, 3200, 3600. Reachability printed by
  the analysis (clarification 8).
- Hypotheses and thresholds, fixed before any run: F1 sign-pattern
  concordance at matched milestones, adopted iff >= 0.75 with
  seed-bootstrap 2.5th percentile > 0.50; H-SCALAR Spearman |rho| < 0.5
  as the committed foil; F2 shape classes and F3 crossover distributions
  descriptive; robustness variant at fixed steps reported, expected to
  fail.
- Guards: regime check per lane; drift guard on the exact-vs-swa64s1
  edge at step 1200; bridge gate (numerics); lanes read from model
  events; deterministic F2 tie-break (clarification 7).
- Receipts: `transfer_s1/PREREGISTRATION.md` (amendment 1,
  clarifications 1 to 8, all dated), `transfer_s2/PREREGISTRATION.md`
  (licence anchor 90791ed), `analyze.py` frozen at the anchor.

### 3. Study 1: the positive, and its scope

Numbers (receipt `transfer_s1/receipts/L/ANALYSIS_SML_20260907b.txt`):

- Milestone band at lr 1e-3: S reaches all five (12/12); M reaches
  800 only (12/12, then 0/12); L (3 seeds) reaches none. M bottoms at
  median best val 3.72 against S 3.35.
- F1 primary: 1.000 over 15 cells; band [0.867, 1.000]. Adopted.
- F1 robustness (fixed steps): 0.681. Not adopted, as predicted.
- H-SCALAR rho = -0.082: foil condition met.
- F2 modal agreement 9/15; F3 S crossed 11/12 (median b0 2000), M 4/12.
- Threat 2: S 49/72, M 31/72, L 0/6. Threat 3 at M 12/12.
- Fixed-step trend: M vs L 1.000 over 9 cells; S vs L 0.444.

The licensed sentence, as written on 7 Sep: at the one matched position
both arms reach, all fifteen edge signs agree and scalars do not
transfer. The narrowness is a protocol property (fixed rate), stated as
such. Registry rows T1-structure-transfers and T1-fixed-lr-mistunes.

The pre-registered Threat 4 cell (M at 5e-4, seeds 21 to 23, receipt
`receipts/M_lr/THREAT4_lr_sensitivity_20260907.txt`): M reaches all five
milestones and trains below S; the exact-vs-swa64s1 sign agrees between
rates on 16/27 fixed slices. The sign pattern is rate-sensitive at fixed
width. This is the diagnosis that motivates study 2 and is the paper's
first "the protocol flagged its own flaw" moment.

### 4. Study 2: pre-registration, rule, falsifier, result

- Stage 1 rule (fixed before any run): per width, exact lane, three
  seeds, the grid; lr* = lowest median best val among grid points
  passing the regime check in >= 2/3 seeds; else latest median best-val
  step; ties to the larger rate. Results (`RESULTS_STAGE1.md`): lr*(256)
  = 2.5e-4 (1e-3 gave 3.369, 2.5e-4 3.228; 5e-4 lowest but regime-failed
  2/3); lr*(512) = 5e-4 (3.2247, 3/3 pass); lr*(1024) = 5e-4 by the
  fallback clause (no rate passes; see section 6).
- Predictions written before running: F1 >= 0.75 at every milestone;
  |rho| < 0.5. Falsifier: F1 < 0.75 at any later milestone makes the
  paper the receipts-backed protocol-noise outcome.
- Result (receipt `receipts/ANALYSIS_stage2_SM_20260914.txt`):
  - Band: S 12/12 at four milestones, 8/12 at 3600; M 12/12 at all
    five. The alignment the design was for.
  - F1 primary 0.627 over 75 cells, band [0.467, 0.667]. Not adopted.
    Robustness 0.630 over 135 cells, band [0.585, 0.719].
  - H-SCALAR rho = -0.600. The ranking reverses.
  - F2 5/15. F3 S crossed 9/12 (b0 1600), M 11/12 (b0 3200).
  - Threat 2: S 47/72, M 19/72. Threat 3 at M 5/12 (guard calibrated on
    the fixed-rate protocol; reported, not used).
- The sentence: with the curves aligned, neither orientation nor
  magnitude carries from d = 256 to d = 512 in this family. The study-1
  concordance is explained by the shared rate regime.

### 5. Why it was caught: the methodological argument

One subsection per element, each with the counterfactual "without this,
the positive ships":

- **Matched loss, not matched steps.** Without it, study 1's robustness
  variant (0.681) would have been the primary and the narrowness
  invisible; or, worse, a step-matched positive with no scope line.
- **The milestone band printed by the analysis** (clarification 8).
  Without it, "fifteen cells" reads as fifteen independent tests rather
  than fifteen edges at one position.
- **A committed foil** (H-SCALAR). Without it, rho = -0.60 is a footnote
  instead of the sharpest figure; with it, the reversal is licensed as a
  reportable outcome and the script's own warning forbids letting the
  threshold speak for it.
- **Twelve seeds with bootstrap bands.** Post-hoc demonstration to add
  (descriptive, labelled): subsample the banked twelve seeds to three,
  many times, and tabulate how often F1 would have cleared 0.75 in study
  2 and how wide the study-1 band would have been. This is the "seeds
  matter" figure and costs no compute.
- **A falsifier named in advance.** Without it, study 2's 0.627 becomes
  "partial transfer" and the paper still claims something. With it, the
  outcome was named before the run and the claim is the negative.
- **The Threat 4 cell, pre-registered as descriptive.** Without it, the
  narrowness has no diagnosis and no sequel.
- **Reporting in order.** The paper's form is itself the method.

### 6. The protocol's own failure, dated

The stage-1 fallback clause ("latest median best-val step") was written
for the case where nothing has converged and prefers the rate still
improving. At d = 1024 every rate overfits inside 3600 steps and the
clause prefers 5e-4 (median best val 3.63) over 2.5e-4 and 1.25e-4
(3.15 to 3.16), because the slow rate peaks last. A rule fixed in
advance behaved badly in a regime it did not anticipate. The paper
reports: the rule as applied, the note written the night it fired
(`RESULTS_STAGE1.md`, 14 Sep 00:25), the author's decision (pending:
amend with a dated amendment before the L arm is read, or keep), and
the L-arm consequence either way. This section is the paper's claim to
candour and should be short.

### 7. The L arm (pending)

Twelve seeds, two lanes, at lr*(1024) as decided in section 6. Trend
point: it cannot rescue F1, which failed at S vs M. Report its band, its
edge sign against S and M, and its scalars. If the author amends the
rule, the L arm re-runs at the amended rate and both are reported.

### 8. What a tiny panel can license

The positive sentence that survives: nothing about orientation across
width, in this family, at this range, once the curves are aligned. The
practical sentence: a small-width screen can rank backwards (rho -0.60),
so a screen without a matched-position check and a cross-width
calibration is not evidence. Scope as declared; no muP claim; the
selection rule's favouring of the exact lane disclosed.

### 9. Limits

One family, two layers, one corpus slice, three widths, a rate grid of
three or four points, a selection rule that favours one lane, a 16x
model that overfits the slice. Each is a reason the numbers are
properties of this protocol; none is a reason the methodological
argument does not carry.

## Figures and tables (planned)

- Fig 1: the comparison graph at each milestone, S and M side by side,
  study 1 and study 2 (edge orientation as arrow direction; agreement
  coloured). This is the paper in one figure.
- Fig 2: validation-loss curves per arm with the milestones marked;
  study 1 (M never reaches them) against study 2 (M reaches all).
- Fig 3: effect sizes per edge, S against M, study 1 and study 2 (the
  rho = -0.60 scatter).
- Fig 4: the seeds-matter subsampling figure (section 5).
- Table 1: protocol summary. Table 2: stage-1 selection tables (three
  widths). Table 3: all F1, H-SCALAR, F2, F3 and guard numbers, both
  studies. Table 4: amendment and clarification history, dated.

## Post-hoc results in hand (14 Sep, receipt `receipts/POSTHOC_20260914.txt`)

- Per-milestone F1, study 2: 0.933, 0.733, 0.667, 0.467, 0.333 at
  800 ... 3600. Monotone decay to below chance. Study 1's one milestone
  (800) is where study 2 agrees 14/15. **This is Fig 1's second panel
  and the paper's central mechanism: agreement early, reversal late; a
  fixed rate froze the comparison at the agreeing position.**
- Seeds-matter: a 3-seed study 2 clears 0.75 in 16 % of draws; a 3-seed
  study 1 clears it in 97 %. Study 1's flaw was position, not noise.
- Per-edge signs at 800: identical across studies except one tie.

## Post-hoc analyses to add (all descriptive, all labelled post-hoc)

1. Per-milestone F1 for study 2 (the frozen script reports the
   aggregate). A separate script, not a change to `analyze.py`.
2. The seeds-matter subsampling (section 5).
3. Per-edge sign table: which edges flipped between study 1 and study 2
   at the 800 milestone (the one both studies share).

## What is banked and where

- `transfer_s1/`: PREREGISTRATION.md, RESULTS_M.md, RESULTS_L.md,
  PAPER_SKETCH.md (the pre-refutation framing, kept as history),
  receipts/{S,M,L,M_lr,bridge}.
- `transfer_s2/`: PREREGISTRATION.md, RESULTS_STAGE1.md,
  RESULTS_STAGE2.md, receipts/{lr_S,lr_M,lr_L,s2_S,s2_M},
  ANALYSIS_stage2_SM_20260914.txt, analyze.py (frozen), select_lr.py.
- `atlas/findings.jsonl`: rows T1-structure-transfers and
  T1-fixed-lr-mistunes (status to be revised by the author; the sequel
  supersedes the first).
- Runner: `tools/colab_transfer_runner.py`, `tools/colab_adopt.py`,
  `tools/mtsweep.py --shard`, bit-identical checkpoint-resume
  (`tools/test_resume.sh`); a methods appendix, not a section.

## Order of work

1. LITERATURE.md in (agent running, 14 Sep).
2. L arm lands; author's decision on the fallback clause; L read.
3. Post-hoc analyses 1 to 3 (scripts under `transfer_s2/posthoc/`).
4. Figures from receipts.
5. Prose, TMLR style, double-blind: no author names, no repository
   names that identify the author, receipts as anonymised supplementary.
6. Hostile read against the author's rules before submission.
