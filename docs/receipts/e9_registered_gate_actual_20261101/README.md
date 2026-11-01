# E9 actual run: scientific negative, not setup failure

The VM completed both models' measurements and failed the final registered receipt validator. The unchanged checker independently reproduces the failure. This is not a timeout, a download/build failure, a nonfinite-probe refusal, or evidence that the manager's `FINISHED` status means scientific success. The execution receipt remains `gate_pass=false`, `process_exit_code=1`, `complete_marker=false`.

The canonical ZIP is untouched at `/mnt/c/ml_artifacts/kvq/ember-e9-gate-20261101/e9_gate/out.zip`: 246,062 bytes, SHA256 `409ca8fb79354a051b23e5b2cd8c7984e0a3baf6b76ea644b623424e5b010fc4`. All 112 returned files pass CRC and safe extraction checks. The returned staging manifest matches the admitted V3 manifest, and the exact archived checker/reference inputs are hash-bound in the JSON audit. Raw float dumps were intentionally removed by the frozen script; the retained complete dump hashes and byte-comparison records agree, but this audit cannot rerun deleted-array comparisons.

## Reached and passed

- G1: both models' base/new default and deterministic byte comparisons, plus deterministic repeat controls.
- G2: both models' f16 short/long deterministic repeats. All 14 G1/G2 comparisons report identical bytes, corroborated by the retained full dump hashes and 26 integrity receipts.
- G3: f32 and f16 CPU/device controls for both models pass their registered thresholds, all 288 top-1 positions and all 12 greedy continuations. This is the new A1/k=2 gate, not reversal of the historical strict 1e-4 f16 negative.
- G4: CUDA-f16 versus CUDA-f32 quality passes both models' registered limits.
- Both book/chat teacher-forced decode legs complete all 8,192 positions for both models. The Qwen2.5 8,064-context/128-step benchmark also executes.

## Exact failing predicate

The final checker requires every teacher-forcing decode metric to be no larger in magnitude than the corresponding stored historical prefill observation. Both models fail that conjunction. All twelve `max_rel` comparisons pass; ten of twelve `mean_rel` comparisons and three of twelve absolute `ppl_rel_diff` comparisons fail.

| Model | Input | Context | Metric | Decode | Historical prefill |
|---|---|---:|---|---:|---:|
| qwen3 | book | 4096 | ppl_rel_diff | 3.884434461e-05 | -2.094232715e-05 |
| qwen3 | book | 8192 | mean_rel | 3.647960441e-05 | 3.64588209e-05 |
| qwen3 | chat | 2048 | mean_rel | 3.727599889e-05 | 3.693721518e-05 |
| qwen3 | chat | 4096 | mean_rel | 3.864398587e-05 | 3.842736665e-05 |
| qwen3 | chat | 8192 | mean_rel | 3.961496795e-05 | 3.942886321e-05 |
| qwen25 | book | 2048 | mean_rel | 5.276878044e-05 | 5.243678482e-05 |
| qwen25 | book | 4096 | mean_rel | 4.97796796e-05 | 4.951398585e-05 |
| qwen25 | book | 8192 | mean_rel | 4.955684292e-05 | 4.916622026e-05 |
| qwen25 | book | 8192 | ppl_rel_diff | 5.560230211e-05 | 4.117734224e-05 |
| qwen25 | chat | 2048 | mean_rel | 4.714097192e-05 | 4.680994551e-05 |
| qwen25 | chat | 2048 | ppl_rel_diff | -6.339400939e-05 | 1.954380717e-06 |
| qwen25 | chat | 4096 | mean_rel | 4.746413659e-05 | 4.715594409e-05 |
| qwen25 | chat | 8192 | mean_rel | 4.437713528e-05 | 4.402705711e-05 |

The final VM log consequently ends `FAILED: receipt validation`; missing `GATE_DONE` and process exit 1 are expected consequences of that scientific rejection. No earlier measured leg is relabelled as failed, and the overall result remains negative.

The mean-error excesses are small in absolute terms. The Qwen2.5 chat/2048 perplexity-relative difference is about -6.34e-5 versus a historical prefill observation near +1.95e-6; its ratio is large partly because the historical value is close to zero. Neither observation waives the registered predicate. Significance, benign rounding, or a particular kernel defect cannot be established from these aggregates alone.

## What follows

No concrete engine defect or justified production fix is demonstrated. A separately approved, prospective matched-source/current-device prefill-versus-decode diagnostic could distinguish present path differences from historical source/runtime effects. It must retain this negative, avoid selecting outcomes, and must not replace the frozen historical ceiling or loosen a criterion retroactively. No such run was launched by this audit.

The [JSON audit](independent_failure_audit.json) retains all source/reference hashes, exact comparisons, failed cells, G3/G4 values and benchmark receipt. The [frozen checker result](frozen_checker_result.json) is recomputed, not copied from the VM verdict. `audit_e9_actual_failure.py` records the integrity/replay procedure; existing evidence is refused. This audit used one CPU budget slot, with no GPU, provider, configuration, engine, canonical-receipt or Claude-message action.
