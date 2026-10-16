# posttrain_smoke — the Phase 2 pipeline, end to end, once

**What this is:** one tiny model taken through every Phase 2 step on real
data, to show the pipeline runs and its receipts are complete. **What it is
not:** a study. It has one seed, a short run, and no pre-registration, and
it is not a registry row. The calibration pilot (charter 0001 §3.2) and PT2
are the studies.

- Model: llama-tiny (d=128, L=2), objective `lm` on `tinyarith_r20_v1`
  `train.txt`, 1500 steps, seed 1, CPU (WSL). Spec: `smoke_r20.json`
  (`smoke_train.sh`). Final held-out loss 0.670.
- Sampling: `mtstudio sample-set` on the 67 frozen dev prompts
  (`corpus/eval/tinyarith_r20_dev_v1.jsonl`), 8 samples each, temperature
  1.0, seed 17. Scored with `exact_match` and, separately, the default
  `synthetic_judge`, keeping the top 1 per prompt (`smoke_rs.sh`).

| check | result |
|---|---|
| completions in the trained template, stopped at end-of-text | 536 / 536 |
| sampling replayed with the same seed | byte-identical |
| records carrying checkpoint, spec and prompt-file sha256 | all |
| audit draw | 12 items over 4 strata (`receipts/audit_sheet.md`) |
| exact-match reward, mean over samples (unseen facts) | 0.194 |
| synthetic-judge reward, mean over samples | 0.388 |
| judge accepts, exact match rejects | 104 / 536 |

Read the last three rows as a description of this one model, not as an
effect. They show what the studies need: the base model is imperfect on
unseen facts, and the declared judge passes wrong answers often enough to
give the provenance thesis something to measure.
