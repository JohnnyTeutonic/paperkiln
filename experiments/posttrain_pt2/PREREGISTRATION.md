# PT2 — filter provenance: does judge error act like noise? (pre-registration)

*Charter 0001 §5, Study PT2; registry row `PT2-filter-provenance`. **DRAFT,
not licensed.** It is licensed by the commit that fills in R from
`posttrain_calib` (the one blank below) together with `run_pt2.py`,
`analyze.py` and `smoke_analyze.py`. Rules change only by a dated
amendment, before data.*

## Question

Rejection sampling filters a model's own samples and fine-tunes on the
kept ones. When the filter is a judge that errs, does the fine-tuned model
learn to fool that judge more than a model filtered with **random** errors
at the same rates?

- If it does, judge error has exploitable structure, and evidence
  provenance matters beyond label accuracy (the charter's thesis, §2).
- If it does not, judge labels behave like label noise at this scale. That
  is a published negative.

## Fixed beforehand

- **Task:** `tinyarith`, level **R = ___** (set by `posttrain_calib`).
  Sampling prompts are the level's training-split prompts, one per
  training fact. Evaluation uses the frozen test set and hack probe of the
  same level.
- **Judges:** the charter §12.3 fallback, because no real-judge
  calibration has been run. `SyntheticJudgeVerifier` with
  `accept_wrong_with_cue = accept_wrong_near = q`, `near_k = 1`,
  `accept_wrong_other = 0`, `reject_correct = 0`, seed 1, for
  q ∈ {0.10, 0.25}. The scope of any finding is "synthetic judges with this
  error profile", never "LLM judges".
- **Seeds:** 10 (101–110). Each seed has its own base model, its own
  samples and its own arms. Arms are paired within a seed.

## Procedure, per seed s (`run_pt2.py`)

1. **Base.** The `posttrain_calib` recipe at level R (llama-tiny, T=64,
   batch 8, lr 3e-3, 3000 steps, objective lm), train.seed = s.
2. **Samples.** 8 per sampling prompt, temperature 1.0, sampling seed s,
   at most 8 new tokens. Each sample is scored once by exact_match, J10
   and J25.
3. **Arms.** Each arm is a set of accepted samples:
   - **E:** exact_match = 1.
   - **Jq:** judge q accepts.
   - **Rq:** the noise-matched arm. On this seed's samples, measure judge
     q's false-accept rate FA (accepts among exact-wrong) and its
     false-reject rate FR (rejects among exact-right). Then accept each
     exact-wrong sample with probability FA and each exact-right sample
     with probability 1 − FR, by an RNG seeded (s, q). This gives the same
     confusion matrix as Jq, with errors falling at random rather than by
     the judge's rules.
   - **X:** accept each sample with probability equal to J25's overall
     acceptance rate (the floor).
4. **Equal size.** Every arm of seed s is subsampled, seeded, to
   K_s = the smallest arm's size, so arms differ in which samples they
   hold, not in how many.
5. **SFT.** `train.objective` rejection_sft from the seed's base
   (`train.init_from`): 600 steps, lr 1e-3, batch 8, train.seed = s.
6. **Evaluation**, the same for every model, including the base:
   8 samples per test prompt and 8 per hack-probe prompt, temperature 1.0,
   seed 17. Metrics:
   - `truth_acc` (exact match on test);
   - `judge_truth_gap_q` = mean Jq reward − `truth_acc` on the same test
     samples, for each q;
   - `hack_frac` = the fraction of hack-probe samples that pass the
     format verifier and fail exact match;
   - `distractor_frac` = the fraction of hack-probe samples answering
     a + b + d, the sum that wrongly adds the distractor quantity. This is
     the probe's specific hack signature. `hack_frac` also counts plain
     wrong answers: in the plumbing run, a 60-step near-random model
     scored 0.9 on it.

## Hypotheses and decision rules (`analyze.py`)

Every contrast is a paired difference over the 10 seeds. Reported for
each: mean, SE, t (df = 9), the per-seed values and the sign count.
Two-tailed critical value 2.262.

- **H1, the primary (the thesis at this step), for each q:**
  `judge_truth_gap_q`(Jq) − `judge_truth_gap_q`(Rq) > 0.
  - **Supported:** mean > 0 and t > 2.262.
  - **Contradicted:** mean < 0 and t < −2.262.
  - **Otherwise:** not resolved at 10 seeds. Published as such, and as
    the falsifier F-PT2 firing for that q.
  - The two q values are separate tests and are reported separately.
    No correction is applied: each is a pre-registered primary for its
    own judge.
- **H2 (secondary):** `truth_acc`(E) − `truth_acc`(Jq) > 0 for each q.
  Correct labels help.
- **H3 (secondary):** `truth_acc`(Rq) vs `truth_acc`(Jq), two-sided:
  does structured error cost more accuracy than random error at the same
  rates?
- **Descriptive, no decision:** `hack_frac` and `distractor_frac` for
  every arm against the base; the E − X contrast (filtering at all against a random floor);
  K_s; FA and FR per seed and q.

## Threats, stated before data

- **Equal size is not equal information.** Arms keep the same number of
  samples but different mixtures of right and wrong. That difference is
  the treatment.
- **The judge's FA can differ across seeds**, because it depends on the
  base model's samples. Rq is matched within each seed, so the pairing
  absorbs it.
- **Self-consistency.** These are small synthetic models and synthetic
  judges, and nothing here speaks to LLM judges or real tasks.
