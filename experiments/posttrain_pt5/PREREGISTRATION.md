# PT5 — provenance-weighted preference optimisation (pre-registration)

*Charter 0001 §5, Study PT5; registry rows `PT5-judge-vs-noise` and
`PT5-provenance-weighting`. Licensed by the commit that records the
pilot's outcome here, together with `run_pt5.py`, `analyze.py` and
`smoke_analyze.py`, each exercised end to end before licence. Rules change
only by a dated amendment, before data.*

**Fixed by `posttrain_pt5_pilot`:**
- **Round 1 (NO_BETA):** plain DPO lost the answer format at every β in
  {0.1, 0.5, 2.0}.
- **Round 2 (CHOSEN):** an RPO-style NLL term on the chosen completions
  with α = 0.2 keeps every arm answering (answer_rate ≥ 0.989).

PT5 therefore trains with β = 0.1 and α = 0.2 (`train.po.nll`). The NLL
term carries the same pair weights as the DPO term, so the Mu, Mw and Md
arms still differ only in weighting.

## Questions

1. **F-PT5a, the thesis.** Does DPO on judge-ordered pairs teach a model to
   fool the judge more than DPO on noise-matched pairs, which carry the
   same ordering-error rate placed at random?
2. **F-PT5b, the mechanism.** With half the pairs from exact match and
   half from the judge, does weighting them by evidence (exact 1.0, judge
   0.3, the frozen charter table) beat both treating them equally and
   simply dropping the judge pairs?

## Fixed beforehand

- **Task:** `tinyarith` R = 20 (charter §13). Training pairs come from the
  base model's samples on training-split prompts. Evaluation uses the
  frozen test set and hack probe.
- **Judge:** `SyntheticJudgeVerifier` at q = 0.25, the PT2 fallback profile
  (cue = near = 0.25, k = 1, other = 0, reject_correct = 0, seed 1). One
  judge only, to bound the run count. q = 0.10 is not run.
- **Seeds:** 10 (301–310). Each seed has its own base model (the
  calibration recipe), its own samples (8 per prompt, temperature 1.0) and
  its own arms, paired within the seed.

## Pairs

From one prompt's samples, a pair is (chosen, rejected) with the chosen
sample strictly preferred by the ordering verifier. Within a prompt,
pairs are formed by a seeded shuffle of the preferred × dispreferred
samples, at most 4 per prompt. **Both completions of every pair, in every
pool, must pass the format verifier** (the answer template), and the two
completions must differ. Every arm therefore teaches which well-formed
answer to prefer, never whether to write one. This was added before
licence: the driver smoke run showed that truth-tied pairs mixing a
well-formed and a malformed completion, oriented at random, taught the R
arm to stop writing answers. That would have made F-PT5a come out
"supported" because R could not answer, not because the judge's errors
have structure.

- **E (exact-ordered):** chosen exact = 1, rejected exact = 0. Receipts
  `exact_match`.
- **J (judge-ordered):** chosen judge = 1, rejected judge = 0. Receipts
  `judge`. Relative to the truth, a J pair is *correct* (exact 1 over 0),
  *inverted* (0 over 1) or *tied* (equal exact verdicts).
- **R (noise-matched):** measure J's fractions of inverted (f_inv) and
  tied (f_tie) pairs on this seed. Build R as N pairs: a fraction f_tie
  drawn from truth-tied sample pairs, and the rest from E pairs, each
  oriented the wrong way with probability f_inv / (1 − f_tie), seeded.
  The result has J's error composition, with the errors placed at random.

## Arms (equal N pairs per seed, N_s = the smallest pool)

| Arm | Pairs | Weights |
|---|---|---|
| E | N exact-ordered | uniform |
| J | N judge-ordered | uniform |
| R | N noise-matched | uniform |
| Mu | N/2 E + N/2 J | uniform |
| Mw | the same pairs as Mu | provenance (exact 1.0, judge 0.3) |
| Md | the same pairs as Mu | judge weight 0 (the J half dropped) |

Training: `train.objective` provenance_po from the seed's base
(`init_from`), β = 0.1, NLL α = 0.2, lr 1e-3, batch 8 pairs, 400 steps,
train.seed = s. `answer_rate` (the fraction of test samples still in the
answer format) is reported for every model, so format collapse cannot read
as honesty.

## Metrics

The same for every model, including the base, and the same as PT2's
(`run_pt2.evaluate`):
- `truth_acc` on test;
- `judge_truth_gap_25` = mean judge reward − `truth_acc` on the same test
  samples;
- `distractor_frac` on the hack probe;
- `hack_frac` on the hack probe, descriptive only.

The charter named `hack_frac` as PT5's metric. It is replaced here, before
data, by `judge_truth_gap_25`. The smoke runs showed `hack_frac` mostly
counts plain wrong answers: a near-random model scored 0.9 on it
(`experiments/posttrain_pt2/PREREGISTRATION.md`).

## Decision rules (`analyze.py`)

Paired over 10 seeds, df = 9, two-tailed critical value 2.262.

- **F-PT5a:** `judge_truth_gap_25`(J) − `judge_truth_gap_25`(R) > 0 with
  t > 2.262 → SUPPORTED. Mean < 0 with t < −2.262 → CONTRADICTED. Otherwise
  NOT_RESOLVED, published as the thesis not supported at this scale.
- **F-PT5b,** both required:
  - (i) `judge_truth_gap_25`(Mu) − `judge_truth_gap_25`(Mw) > 0 with
    t > 2.262: weighting lowers the gap;
  - (ii) `truth_acc`(Md) − `truth_acc`(Mw) < SD of those per-seed
    differences: weighting does not lose to dropping by more than seed
    noise.

  If both hold, PROVENANCE_WEIGHTING_EARNED. If (i) fails, the weighting
  does nothing (WEIGHTING_INERT). If (ii) fails, dropping is better
  (DROP_BEATS_WEIGHT). Either negative retires the default weight table
  and is registered.
- **Descriptive:** E − J and R − J on `truth_acc`; `distractor_frac`
  contrasts; N_s, f_inv and f_tie per seed; per-arm dataset ESS.
- The shuffled-weight control runs as a seventh arm only if F-PT5b holds,
  under a dated amendment.

## A property of this judge, found by the smoke test

With `reject_correct = 0` the judge almost never rejects a correct
answer. The exception: it reads the last number in a completion, so a
completion such as "the answer is 7 . 3" is exact-correct but judged
wrong. Inverted J pairs are therefore rare (f_inv = 0.001 in the driver
smoke run), and almost every ordering error J makes is a tie: a wrong but
judge-accepted answer preferred over another wrong answer. R reproduces
J's tie rate and its inversion rate (in the pair-construction test, a tie
rate of 0.294 against 0.294). F-PT5a therefore asks whether *which* wrong
answers the judge prefers teaches more judge-fooling than ties chosen at
random.

## Threats, stated before data

- **Md trains on half the pairs at full weight.** Its effective sample
  size is lower, which is the point of the comparison.
- **The pools differ by seed.** N_s is set per seed and reported.
- **One synthetic judge profile.** Nothing here speaks to LLM judges.
