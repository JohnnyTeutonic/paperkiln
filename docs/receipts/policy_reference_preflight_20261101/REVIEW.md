Independent prospective review, 1 November 2026.

V2 is ready for root serial admission only. No CUDA execution, provider action, configuration change or feature/source edit was performed by this reviewer.

The public ops.hpp contract accepts constant reference log-probabilities as a Matrix without requiring host-only storage. The CLI producer log_softmax_rows already materializes its input and returns a host-created matrix; this is a separate consumer-boundary correction. The production change is precisely materialize(ref_logp) plus the host-read check before forward reads and by-value tape capture, with no arithmetic change.

The prospective CUDA harness creates one FP32 identity GEMM reference, checks that it is deferred, poisons only stale host storage, and reports the caller reference after the operation captures its tape copy. Old mixed and KL-only results must disagree with correct loss and every-element gradient comparison, while agreeing with the independent all-nine-reference prediction. Old policy-only and all fixed controls must remain correct. The construction isolates the reference Matrix consumer and does not establish historical CLI training invalidity, residency, speed or quality.

V1 review found a negative profiler counter could cancel an extra GEMM in the validator. V2 rejects every noninteger or negative counter before aggregation. The V1 finding and original manifest remain preserved. All 63 optimized parser tests pass, including the added negative-counter regressions. Six fresh native CPU reruns pass complete loss and gradient replay; strict CPU CUDA refusal and wrong-manifest refusal were independently reproduced. Original CPU binary hashes match their recorded build identities.

All 964 V2 archived source files exactly match the declared source Git commit and manifest inventory. Relative to V1 only parser and test source change; production, numeric oracle, fixture values, bounds and wrapper are unchanged. Protocol updates only record the pre-outcome hardening and test count. The manifest, source identity, exact inventory proof and mutation results are in v2_readiness.json.

The proposed gate remains unlaunched and requires root-controlled serial admission, final controller/deadline/landing registration, and actual L4 outcomes. No blocking source or scientific-oracle issue remains in the reviewed V2 packet.
