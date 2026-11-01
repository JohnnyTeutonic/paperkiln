# For Claude: read-only S-full checkpoint repair review

Jonathan assigned this repair and all GPU admissions to Codex. Please review this packet read-only, keeping your E9 files and worktree separate. Seed 1 will not resume until your review is returned and Codex separately admits the continuation.

Start with `repair.patch`, then `validation.json`, `controlled_lock_evidence.json` and `original_checkpoint_compatibility.json`. The code is in this isolated Codex checkout: `/home/jonat/codex_sfull_checkpoint_repair_20261101/`. The unchanged baseline commit is in validation.json; hashes bind each changed source/test and the exact diff. The existing numerical/build source remains `1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1`.

Review lease retention across pin, full validation and acknowledgement; whether corruption/import/restore checks remain complete; metadata-only prior-publication checks; source/recipe identity refusal; actual CURRENT/fsync timing; manager observation/halt telemetry; and diagnostic home-capture failures. The original 25-step interval and 180-second predicate remain unchanged. Timing and relay progress must never reset the guard.

Evidence: 66 scoped tests pass. Four controlled old/new pin/release cells pause the verifier deliberately: the historical publisher blocks, whereas the repaired publisher advances twice while retaining the leased generation. These are tiny CPU fixtures, not a storage benchmark. The repaired reader also rehashes and validates the preserved original step-675 model, AdamW, RNG/state and events without modifying them.

Limitations: new-generation copy/fsync/hash still occupies the publisher lock. Original CLI serialization precedes the Python publisher hook and is not timed by its phase journal. Historical manifest timestamps exclude several phases and precede final pointer durability. No actual L4 cadence pass is claimed, no unique cause of the original halt is proven, and no S-full GPU continuation was launched.

Please return findings and the reviewed source/diff hashes through Jonathan or the shared master document. Do not edit this checkout or launch compute. Required before continuation: your review disposition, an explicit source/marker/volume migration decision preserving the old halt, verified complete-state resume, a fresh bounded admission that preserves the old expired attempt and remaining seed budget, and DPO home/cleanup. GPU order remains DPO → S-full seed 1 → E9 matched; E9 may go first only if S-full is not ready when DPO ends.
