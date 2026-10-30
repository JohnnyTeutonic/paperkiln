# HS1 seed-5 operational repair and exact ledger exception

S-gist seed 5 initially failed before training because its VM lacked `tokenizers`.
The failed log and statuses were preserved. Installing version 0.23.3, matching the
already working B seed-4 VM, left the frozen source and vocabulary hashes unchanged.
The old remote coordinator had exited; only the failed data/PT/post markers were
reset. The same frozen boot resumed the same pod and volume. CUDA and recipe gates
were retained. Data completed at 06:01:19 UTC and pretraining began normally.

The original frozen data builder reconstructs its ledger on each invocation. Its
cached split branch omits `human`, `steps.decontam` and `steps.human_fetch`; consequently
the resumed ledger lacked these fields. The v1 finalizer correctly halted. Its
HALTED marker/state and both original ledger byte streams are preserved here, and
the original artifact ledgers were not changed.

The read-only audit independently hashed all thirteen retained source dialogue files,
all five actual training/SFT/panel outputs, both evaluation sets and the retained
decontamination/fetch reports. All scientific input hashes and sizes match the original
S-gist reference. The retained decontamination report's `summary` exactly matches the
reference ledger, including rules and counts. Every other normalized ledger field
already matches. Fetch stdout is an operational report, not an additional data source.

`finalize_hs1_extension_v2.py` permits only this exact resumed ledger SHA256, this exact
reference ledger SHA256 and this exact audit proof SHA256, with explicit source,
output, evaluation and decontamination checks. It does not weaken the general ledger
normalizer, edit the frozen statistical helpers, change training or alter the original
three-seed results. New v2 state/output paths preserve the v1 halt.

The sole source-inventory substitution is the historical builder artifact here, whose
bytes hash to `48df84d69ef7e309b3eee1701653584625a426bca0ea6a0b214cba82d2eef465` and
equal the Git blob at frozen job pin `7baf6d95f79b9cc543705e9b510bcbfeb5c18a63`.
The analysis does not execute the builder. Freezing this historical artifact rather
than the live builder allows a separately reviewed future cache-provenance fix without
falsely attributing that new source to the historical data. All statistical helper and
other source checks remain unchanged.
