# S-full V2 operational readiness — UNLAUNCHED

This packet prepares three serial S-full seeds under $7 per seed/$21 total.
Root admission and source publication remain required. No provider calls,
watchdog activation or numerical outcome collection were performed here.
Existing V1 FAILED/HALTED records remain intact. Existing B/Sgist models are
reused; no baseline retraining or automatic HS1 follow-up is authorised here.
Six historical near-tie parity disagreements remain an interpretation
qualification; the user GO does not convert them into a clean parity result.

The original numerical launch is retained at 7edb0ca1. New commit
1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1 changes checkpoint I/O only:
the original mtstudio body is restored byte-for-byte by reversing one include
and the post-state-close publisher call. Arithmetic and RNG statements are
unchanged. The original source/finalizer files are not edited or relocated.
The reviewed input V2 verifier is used without changing the frozen data recipe.
The new finalizer explicitly binds new provenance and exact complete runtime
PT/SFT specs; its scientific computation and parity checks delegate unchanged
to the preserved original finalizer. Its source freeze includes both old and
new launch pins and absolute original paths.

Checkpoint and relay settings become25 steps before observing S-full outcomes.
A synchronous checkpoint publisher closes model/AdamW/state and snapshots the
flushed events prefix into a validated generation, fsyncs files/directory and
atomically replaces CURRENT. Publication blocks the next optimizer update.
RNG reconstruction is source-bound std::mt19937 with seed123+1000003*recipe_seed
and step*batch*accum draws. A separate owned remote manager monitors only actual
training process folders; absence of a new committed persistent-volume
checkpoint for over180 seconds stops its own process group and records a
preserved failure marker. Production timing is not established by tiny CPU
fixtures; a cadence failure is a failed run, not permission to relax the bound.

Reset protection depends on the retained persistent volume. Home backups can
lag and do not have a three-minute network requirement. An immutable INFLIGHT
pin lets CURRENT advance during slow full model+AdamW+state+events transfer;
only current, inflight and one temporary generation are retained. Disk space
is checked before copies. Home fallback restores all optimizer state and the
source-bound RNG reconstruction, never cold AdamW. A complete newer remote
generation takes priority over an older home upload; incomplete/mixed/symlinked
bundles refuse. Record actual home fallback age if the provider loses a volume.
The normal runner remains owner; no custom local controller or new watchdog
is introduced. Existing RunPod transport enforces time/spend limits during
blocking transfer subprocesses by polling every2 seconds, not only main ticks.

CPU proof: full original core plus amended CLI built successfully. Actual tiny
Llama LM and SFT interrupted at step4, copied the entire committed bundle home
and back into a fresh output folder, then resumed to8 with byte-identical final
weights, AdamW and state versus uninterrupted runs. See
actual_cpu_resume_continuity_v4.json and raw_home. This is continuity evidence,
not actual L4 cadence, training quality or speed evidence. Checkpoint transport
negative tests cover corruption, missing optimizer, mixed/source drift,
manifest symlink, cold-flat fallback and remote-newer priority.

Fresh runner states must be under /home/jonat/sfull_v2_states_20261101 with
basenames pk_c12_sfull_r1_runner through r3_runner, matching the unchanged
finalizer lookup. Artifact homes remain /mnt/c/ml_artifacts/tiny_chat/c12_Sfull_rN
(which were absent at preflight). No original HALTED marker is removed.
Estimated total training duration18–27 GPU hours is historical planning only.
The standard runner's max-hours23.5 and $7 limit apply separately to each seed;
root must admit seeds serially and enforce the $21 aggregate including retries.
