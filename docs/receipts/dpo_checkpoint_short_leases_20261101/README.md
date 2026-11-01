# DPO checkpoint critical sections: prospective operational repair

Before any DPO training, review of the actual SFT artifact found a roughly 390 MB model and an expected 1.17 GB model/AdamW generation. The checkpoint implementation rehashed retained payloads while holding the writer-shared lock during pin, acknowledgement and the next publication. This is an avoidable source-level contention risk. It does not prove DPO would exceed the unchanged 180-second guard, or identify the exact phase that halted S-full.

Lease users now serialize through a separate lease lock. They establish INFLIGHT retention under the short pointer lock, verify complete payloads outside the writer lock, and recheck identity before acknowledgement. New publication checks the previous generation's metadata for source/recipe/step drift; it does not claim to reverify that older payload. Complete new-payload copy, fsync and hashing, plus full import, consumption and restore validation remain. New-generation copying/hashing still occupies the publication lock.

The phase journal separates lock wait, copy/fsync, hashing and actual CURRENT pointer plus directory-fsync completion. Timing evidence is diagnostic, never a checkpoint heartbeat or a numerical acceptance verdict. A home snapshot and training-log evidence preserve diagnostics beyond pod cleanup; capture failures remain explicit.

Tests exercise actual separate processes: publication advances while a pinned verifier is deliberately paused, the leased generation remains available, and corruption, wrong acknowledgements, source/recipe drift, interruption and unsafe paths are refused. Independent review also checked the retention and consumption boundaries.

No optimizer arithmetic, recipe, random draw, checkpoint interval or 180-second guard changes. Actual L4 cadence remains unmeasured until a separately admitted campaign runs. The earlier source admission and unlaunched packet remain preserved; launch requires a new exact source/archive binding.
