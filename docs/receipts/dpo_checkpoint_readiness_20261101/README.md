# DPO full-state checkpoint readiness — UNLAUNCHED

This is a separately named DPO operational contract. Frozen S-full sources,
active campaign sources and provider/watchdog configuration were not changed.
The original d642 CLI is restored byte-for-byte by reversing only the new
include and prepare/publish checkpoint-I/O calls (receipt alongside this file).
No optimizer, arithmetic, RNG or training-loop statement changes are claimed.
Fresh training from an SFT init_from checkpoint remains valid when its output
folder has no numeric partial checkpoint. Resume requires a complete validated
atomic model+AdamW+state+events generation; partial flat output refuses cold AdamW.

DPO_FULL_STATE_CHECKPOINT_V1 binds source commit, complete exact runtime spec,
dataset manifest and all runtime vocab/pairs/initial-weight hashes. Every immutable
generation carries the contract SHA. Publisher source/recipe/contract drift,
missing optimizer, mixed generations, symlinks, regressing steps and older-home
replacement of newer remote state are refused. CURRENT publication is fsynced
and atomic. INFLIGHT preserves a relay generation while training advances;
only current/inflight/temporary generations are retained, with disk preflight.

The normal runner uses explicit dpo_full_checkpoint_v1 opt-in only. It uploads
full home generations before boot, retains incomplete transfer progress, and
requires the complete final optimizer/state generation before marking DPO home.
The separately named boot consumes fixed uploaded DPO_RUNTIME_JOBS_V1.json;
whole jobs SHA and checkpoint contract SHA are in independent dpo_binding.json
to avoid a circular content hash. The original root-admitted VM source archive
preflight executes before build/training; an exact runtime spec and preflight
training dependency are checked by the new manager too.

A separate owned VM manager starts normal jobs_vm in its own process group and
monitors the actual admitted training executable/output folder. Checkpoint_every10
is prospectively fixed; failure to observe an advancing durable volume generation
within180 seconds while the training process is active stops only that group and
preserves a halt marker. This includes initial reference preparation: there is no
claim its first checkpoint can already meet180 seconds. Actual L4 cadence remains
unmeasured and is a required operational launch check. Full home backup may lag;
reset protection depends on the retained persistent volume. Existing RunPod
blocking-transfer budget/deadline polling remains in force.

Actual CPU proof used a real tiny SFT initial checkpoint followed by provenance_po
with uniform weighting, beta.1, NLL.2, FP32, batch8, seed1, total20 steps and fixed
checkpoint10. Uninterrupted versus interrupted at10/full-home/fresh-output/resume20
had byte-identical weights, AdamW and state, with160 reconstructed RNG draws.
An actual partial-flat missing-optimizer run refused before training. Raw paths,
binary/source hashes and scope are in actual_cpu_continuity_v2.json. This tiny
fixture does not establish L4 cadence, training quality or throughput. The earlier
LM-initialized engineering fixture is retained and explicitly distinguished.

Builder integration must include checkpoint contract and runtime jobs in the
upload manifest and bind both hashes in dpo_binding.json. No eligible dataset,
paid run or scientific outcome was manufactured by this engineering work.
