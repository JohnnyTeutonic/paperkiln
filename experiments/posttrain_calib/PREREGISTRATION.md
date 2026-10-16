# posttrain_calib — choosing the tinyarith level (pre-registration)

*Charter 0001 §3.2, "Task difficulty". The commit that adds this file,
`sweep.json`, `measure.py`, `analyze.py` and `smoke_analyze.py` is the
licence anchor (D4). Rules change only by a dated amendment here, before
data.*

## Question

At which `tinyarith` level R does a base model, trained by the fixed
recipe below, answer unseen arithmetic facts correctly often enough to
leave room for post-training to help or hurt? The charter's band is a
mean dev `truth_acc` in **[0.25, 0.60]**. Below it, RL has almost no
correct samples to learn from. Above it, there is little headroom to
measure.

These runs are calibration only. Under the charter, they are excluded from
every confirmatory analysis, and they produce no registry row of their own.
The chosen R is recorded in the charter and in each study that uses it.

## Design

- Levels R ∈ {10, 20, 50}: corpora `tinyarith_r{R}_v1` (each with its own
  explicit vocabulary), as pinned in `corpus/hashes/`.
- Seeds {1, 2, 3}. Nine runs in all.
- Recipe, the same for every run: llama-tiny (d=128, L=2, 4 heads),
  T=64, batch 8, AdamW, lr 3e-3, 3000 steps, `train.objective` lm on the
  level's `train.txt`, no early stopping. That is the `sweep.json` base.
- Venue: WSL CPU, the Release build of the licence commit, run through
  `tools/mtsweep.py --require-clean --jobs 3`.

## Measurement (`measure.py`)

For each run's exported checkpoint:
- `mtstudio sample-set` on the level's frozen dev set
  (`corpus/eval/tinyarith_r{R}_dev_v1.jsonl`): 8 samples per prompt,
  temperature 1.0, seed 17, at most 8 new tokens;
- scored by `posttrain_verify exact_match`.

`truth_acc` = the mean exact-match reward over all samples of the run.
Recorded descriptively, with no decision attached: the default
`synthetic_judge` reward on the same samples, and the format-pass and
truth-fail fraction.

## Decision rule (`analyze.py`)

1. For each level, report the mean `truth_acc` over its three seeds, the
   per-seed values and the seed SD.
2. **Choose the smallest R whose mean lies in [0.25, 0.60].** Smaller R is
   preferred because it gives the cheapest runs at an adequate difficulty.
3. If no level qualifies:
   - **All below 0.25:** outcome `TOO_HARD`. Pre-declared next step: one
     re-run of the whole design at 6000 steps (a dated execution
     clarification, not a new rule), then this same rule. If the outcome
     repeats, halt and write an amendment before any study is licensed.
   - **All above 0.60:** outcome `TOO_EASY`. Halt; the generator needs
     larger operands (a new corpus version), recorded as an amendment.
   - **Otherwise** (the band falls between two levels, one below 0.25
     and one above 0.60): outcome `GAP`. Halt and write an amendment. No
     new level is invented after seeing data without one.
4. A missing or duplicated (level, seed) measurement makes the analysis
   refuse to run. Each measurement records the sha256 of the checkpoint it
   sampled and of the dev set it used.

## Known limits, stated before data

- Three seeds bound the seed spread coarsely. The band is wide on
  purpose, so a level near its edge may be chosen by noise. The chosen
  level's per-seed values are published with it.
- Dev sets are small at R=10 (18 facts), and the R=10 corpus is small (604
  examples), so 3000 steps is many epochs there. That is part of what is
  being calibrated, not a defect.
