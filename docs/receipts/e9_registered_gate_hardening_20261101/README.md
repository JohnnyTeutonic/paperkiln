# E9 registered-gate diagnostic hardening

This is prelaunch engineering evidence, not an E9 device result. The original Claude packet and registration remain preserved. The baseline is public Ember `3ada969788f4d5ea65f5b81b29c22c30b7f25c97`, including ordered-reduction defaults. Only the newly added diagnostic quality probe and its transport wrapper were hardened; engine arithmetic, models, prompt generator, registered phases and A1/k=2 scientific thresholds are unchanged. Historical strict-gate negatives remain negatives.

## Actual CPU checks

The [V3 probe](e9_quality.v3.cpp) rejects nonfinite logits in either operand, incomplete/empty/oversize forward shapes, invalid row access and overflowed float row ranges before those values can silently reduce to a passing maximum. It also rejects nonfinite derived and final errors. The ordinary finite metric retains its original float-range arithmetic. The [actual shared-code C++ test](test_e9_quality_checks_v3.cpp) passes 20 negative cases, 256 finite comparisons bit-identical to the [original aggregation](e9_quality.original.cpp), a constant-reference case and the known relative error 0.125. This was a CPU-only fixture with no model or GPU call. The complete production tool translation unit also passed a CPU syntax check; no CUDA build was claimed by these tests.

The [V2 delta](e9_quality_hardening_v2.diff) and [additional V3 range refusal](e9_quality_v3_vs_v2.diff) are separate. Original and V2 archives stay in their canonical local homes; no model or source archive is duplicated in this compact receipt. [Readiness](readiness_v3.json) and the [final admission review](final_admission_review_v3.json) bind every final upload and predecessor hash.

The [wrapper](e9_runpod_job.v2.py) refuses existing positive, negative, partial, phase, log and terminal evidence before writing outputs, and uses a durable exclusive attempt claim to prevent concurrent or retained-volume retries. The normal jobs manager creates only the empty job directory; its PID, status and log are outside it. The [actual wrapper tests](test_e9_wrapper_v2.py) pass 15 cases, including byte-exact preservation, fresh setup failure, stub negative/positive transport receipts, second-attempt refusal and exactly one concurrent claimant. Stub positives are unit tests, not scientific successes. The [original overwrite reproduction](original_wrapper_preservation_negative.json) remains preserved.

## Reviewed admission

The [exact root guard](guarded_runner.py), [reviewed launch](launch_review.json), [guard tests](test_e9_admission_guard_20261101.py), [actual ownership probe](owner_probe.json) and [admitted registration](REGISTRATION_as_admitted.md) are retained unchanged. The root owns activation, serialized GPU use, fixed deadlines and cleanup. This evidence package neither activates a driver nor changes configuration. The operational source files use their original absolute local paths; they are preserved for audit, not standalone public launch instructions.

Scientific closure still requires both `gate_summary.pass` and `execution.gate_pass` to be true for the unchanged registered matrix, including both models, G1-G4, teacher-forced decode and the Qwen2.5 8064-position benchmark. Transport completion alone is not a scientific pass. This folder makes no claim about the outcome of that remote run.

## Reproduction

From this directory, the shared-code CPU tests can be rebuilt without a model:

```
g++ -std=c++17 -O0 test_e9_quality_checks_v3.cpp -o /tmp/test_e9_quality_checks_v3
/tmp/test_e9_quality_checks_v3
```

On the original preparation machine, run the preserved wrapper and guard unit sources with Python; they use temporary stub content and never run a real GPU/model workload. Their original versioned local paths are explicit in the source. Exact artifact hashes, source checks, test scope and the one-attempt contract are recorded in [the public compact manifest](compact_manifest.json).

Private provider resource/account preflight, credentials, model weights, source tarballs, compiled binaries and watchdog activation snapshots are intentionally excluded. Public launch bounds are not account balances.
