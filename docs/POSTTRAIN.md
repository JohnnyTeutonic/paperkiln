# Post-training a paperkiln model

paperkiln trains a model from scratch. This page covers what comes after:
- supervised fine-tuning on completions;
- rejection sampling with a verifier;
- reinforcement learning with verifiable rewards (GRPO);
- preference optimisation weighted by where each label came from.

Every piece is a readable forward pass with a hand-derived backward on the
same tape as pre-training. There is no PyTorch and no trl.

Why it is built this way, and the studies that test it, are in
[`decisions/0001-posttrain-charter.md`](decisions/0001-posttrain-charter.md).
The short version: **every reward carries a receipt**, and the question
worth answering is whether *where a label came from* changes what a model
learns.

## The task the examples use

`tinyarith` is arithmetic word problems in the TinyStories world, at three
difficulty levels. Its numbers are single tokens in its own vocabulary.

```bash
python corpus/fetchers/tinyarith.py      # corpora + frozen dev/test/hack-probe sets, hash-verified
```

A 4,096-word model of about a million parameters gets roughly 39% of
unseen R = 20 facts right after the base recipe
(`experiments/posttrain_calib/`). That leaves room for post-training to
help or hurt, which is why R = 20 is the level the studies use.

## 1. Verifiers: no reward without a receipt

`include/microtorch/posttrain/verifier.hpp`. A `VerifierResult` cannot be
built without a rationale, an evidence kind (`exact_match`, `unit_test`,
`format`, `judge`, `human`) and an evidence URI. The constructor throws,
in tests too.

```bash
build/posttrain_verify exact_match --refs refs.jsonl < records.jsonl > scored.jsonl
build/posttrain_verify format < records.jsonl
build/posttrain_verify synthetic_judge --refs refs.jsonl --cue 0.25 --near 0.25 < records.jsonl
build/posttrain_verify llm_judge --cache verdicts.jsonl --judge-id MODEL --rubric-version V < records.jsonl
```

Each input line keeps its fields and gains `reward`, `rationale`,
`evidence_kind`, `evidence_uri`, `audited` and `verifier`.

- **The synthetic judge** has a declared, auditable bias. It reads the
  *last* number in a completion, accepts near misses and answers with
  confident phrasing at set rates, and names the rule that fired in every
  rationale.
- **A real judge** is reached only through
  `tools/posttrain/judge_sidecar.py`. It reads one key from a designated
  env file, observes a hard call cap, and keeps an append-only cache. A
  malformed reply is logged as a failure, never stored as a verdict.

## 2. Sampling and rejection sampling

```bash
build/mtstudio sample-set spec.json --prompts corpus/eval/tinyarith_r20_dev_v1.jsonl \
    --out samples.jsonl --n 8 --temp 1.0 --seed 17

python tools/posttrain/rejection_sample.py --mtstudio build/mtstudio --verify build/posttrain_verify \
    --spec spec.json --prompts corpus/eval/tinyarith_r20_dev_v1.jsonl \
    --verifier exact_match --n 8 --keep top1 --out-dir runs/rs
```

- **Sampling is pure temperature** over the whole vocabulary, with no top-k
  and no banned tokens. Truncation would make the samples off-policy
  without saying so.
- **Records replay.** Each carries its token ids and log-probabilities and
  can be replayed from its seed.
- **Rejection keeps everything.** It writes every record with an
  `accepted` flag and the checkpoint's sha256.
- **Auditing by hand:**
  ```bash
  python tools/posttrain/audit_sample.py draw runs/rs/scored.jsonl --n 40 --out-dir runs/audit
  # edit runs/audit/audit.jsonl: audited, audit_reward, audit_rationale, annotator
  python tools/posttrain/audit_sample.py merge runs/rs/scored.jsonl runs/audit/audit.jsonl --out-dir runs/audit
  ```

## 3. The objectives

All four start from a trained base (`train.init_from`). The reference
model for KL and DPO terms is that base, frozen.

| `train.objective` | Data | What it optimises |
|---|---|---|
| `sft` | `data.sft_examples`: `{prompt, completion}` | next-token loss on completion tokens only (`ops::cross_entropy_masked`) |
| `rejection_sft` | rejection-sampler records | the same, on accepted records that carry a full receipt; human-labelled ones need `train.allow_human_labels` |
| `grpo` | `data.grpo_prompts` (+ `grpo_eval`): `{prompt, answer, uri}` | group-relative policy gradient plus exact KL to the reference (`ops::policy_gradient_kl`) |
| `provenance_po` | `data.pref_pairs`: `{prompt, chosen, rejected, *_evidence}` | DPO with each pair weighted by its evidence (`ops::dpo_weighted`) |

**GRPO** (`train.grpo.*`):
- `group_size`, `prompts_per_step`, `max_new`, `beta`, `temperature`,
  `std_norm`;
- `reward`: `exact_match`, `format` or `synthetic_judge`;
- `advantage`: `shuffled` is the built-in falsifier;
- `reward_shift`: a deliberately buggy key, for testing the monitor.

Every step logs reward, exact-match truth, `hack_frac` (passes format,
fails truth), `reward_hack_frac` (the reward accepts, truth rejects), exact
KL to the reference, entropy and flat groups.

**Provenance weights** (`train.po.*`):
- the frozen table: exact match, unit test and human 1.0; format 0.5;
  judge 0.3; unattributed 0;
- `weight_mode`: `provenance`, `uniform`, or `shuffled` (the falsifier,
  which keeps the weight sum and ESS);
- `weights` overrides the table;
- each step logs Kish effective sample size, weight sum, mean margin and
  preference accuracy.

## How the pieces are checked

- **Finite-difference gradchecks:** every new loss, in
  `tests/test_posttrain_losses.cpp`.
- **Parity anchors:**
  - an all-ones mask is bit-identical to plain cross-entropy;
  - {0,1} DPO weights equal DPO on the kept pairs bit for bit;
  - the KL to itself is zero;
  - GRPO on a two-action bandit lands on the analytic KL-regularised
    optimum, to 2e-7.
- **The pre-training path is unchanged:** it is bit-identical to a banked
  Atlas receipt over 100 steps (`experiments/posttrain_bridge/`).

## The studies

Each is pre-registered beside its driver, with the analysis frozen at the
licence commit.

| Study | Question | Status |
|---|---|---|
| `posttrain_calib` | which difficulty leaves headroom | done: R = 20 |
| `posttrain_pt2` | does judge-filtered rejection SFT fool the judge more than noise-filtered? | running |
| `posttrain_pt4` | does the reward-hacking monitor see a hack built on purpose, and see it close? | licensed, queued |
| `posttrain_pt5` | judge vs noise-matched preference pairs; does provenance weighting earn its keep? | draft |

Results go to `atlas/findings.jsonl` whatever they say.
