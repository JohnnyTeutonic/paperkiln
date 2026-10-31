# Preference downstream packet: prepared, not activated

No dataset was materialized, no model or candidate reply was loaded for analysis, no provider/key accessor was called and no GPU job was launched. Existing winner inputs were read only to hash them. The active V7 wrapper and its provider/parser dependencies are unchanged.

## Reviewable components

- `materialize_preference_v7.py` requires the exact root adoption record and live reconciled 6,036-ranking eligible gate. It runs the unchanged both-order agreement and 1.5-times word-length filters, requires actual sampled stop fields and never filters for chosen replies that finished. It retains file order, re-encoding context refusals and the original last-five-percent validation split. Publication is immutable and atomic.
- `build_dpo_instance_v7.py` creates no training job until the final dataset, winner checkpoint, vocabulary, source commit, exact Git tar archive and root source admission are bound. It fixes `provenance_po`, uniform weights, beta 0.1, chosen NLL 0.2, AdamW, learning rate 0.000002, batch eight, FP32, warmup 20 and cosine minimum fraction 0.1. Steps are ceiling of twice the training-pair count divided by eight. Selection is the final checkpoint. Standard `jobs_vm` preflight, training, parity and paired evaluation jobs are serial. The preflight verifies all archived tracked files against the actual checkout, every uploaded input and the exact spec. Winner GGUF aliases are links, not duplicate uploads. A separate unbound Ember archive override is refused so all existing gate scripts use the same frozen default builder commit.
- `heldout_sampling_receipt_v1.py` captures and independently replays all 300 original source-filtered item dispositions, every actual generation request and raw response, original index seeds and explicit skips. It cannot refill skipped prompts. `heldout_sample_vm_v1.py` is a draft owned-server capture wrapper; its local ownership/cleanup boundary has an offline fake-process test.
- `heldout_gate_runner_v1.py` verifies prospectively adopted operational source and source-data hashes, the complete sample replay, model/hash summary, fresh held-out namespace and final prior-ledger accounting. It preserves one GiB memory admission, serial calls, both orders, no retries and fail-stop unknowns or failed rankings. The existing frozen held-out core enforces 600 calls, a $25 stage allowance and $200 cumulative costs including all prior actual costs and unknown ceilings. No held-out verdict can be converted into automatic model adoption.
- `build_heldout_instance_v1.py` binds final sample receipts and their model summary to the prospectively adopted policy after eligible training rankings are complete. It performs no dispatch or credential access.
- `assess_preference_gates_v1.py` replays raw protocol failures, repeats, all 60 QA outcomes, both 40-item parity legs and complete reconciled API side-by-side judgements. It checks common Ember and checkpoint identities and rejects missing or duplicated raw evaluation coverage. All five unchanged criteria must pass; a negative retains the SFT model. It produces a verdict, never a replacement action.

## Existing inputs

`existing_input_pins.json` identifies the actual winner safetensors, its trained model specification, its GGUF used as the exact vocabulary, and the registered validation/train split files. The winner GGUF matches the candidate sampling receipt. No second model copy or download was created. These are input hash bindings, not a new CPU/GPU model equivalence proof.

## Root records required before execution

Dataset adoption is the existing root record at `/home/jonat/api_v7_root_admission_20261101/dataset_adoption.json`, verified against the exact proposed dataset and V7 protocol hashes and fixed policy fields. No freeform authorisation text is interpreted as a permission input.

Before any DPO outcome, a root held-out adoption record must have kind `ROOT_ADOPTED_API_HELDOUT_POLICY_V1`, adopted true, before_dpo_outcomes true, the frozen held-out policy file hash, `operational_source_sha256` equal to the full mapping in `operational_source_freeze.json`, and `heldout_source_data_sha256` equal to its vocabulary/validation/train mapping. This freezes source rules, decontamination inputs, original prompt universe and seeds before model outcomes.

After eligible judging, pair materialization and the root's chosen KL merge/source work, the root source admission has kind `ROOT_DPO_SOURCE_FREEZE_V1`, ready true, source_commit, source_files_sha256, source_archive_sha256, dataset_manifest_sha256, model_spec_sha256, checkpoint_sha256, vocab_sha256, heldout_input_pins_sha256, heldout_adoption_sha256 and ember_commit. Its source archive must be the exact uncompressed Git archive of that commit. Required source files include actual DPO consumers and the gate/preflight scripts. This ordering is a scheduling and source-freeze decision; the separate public KL-reference Matrix issue is not established as a runtime DPO blocker. Actual DPO uses frozen reference pair margins in `dpo_weighted`; chosen NLL uses a newly created host zero matrix and zero KL coefficients.

## Exact offline entry points

Run substantial audits under the workspace CPU-budget wrapper, using one ember core. The following are source entry points, not launch authorisations:

```
python3 tools/posttrain/materialize_preference_v7.py --protocol PROTOCOL --adoption ROOT_DATASET_ADOPTION --vocab WINNER_GGUF --out-dir NEW_DATASET_DIRECTORY
python3 tools/posttrain/build_dpo_instance_v7.py --dataset-dir DATASET --base-spec WINNER_SFT_SPEC --checkpoint WINNER_SAFETENSORS --vocab WINNER_GGUF --source-root CLEAN_FROZEN_SOURCE --source-archive EXACT_GIT_TAR --source-admission ROOT_SOURCE_ADMISSION --heldout-inputs existing_input_pins.json --heldout-adoption ROOT_PROSPECTIVE_HELDOUT_ADOPTION --out-dir NEW_PACKET_DIRECTORY
python3 tools/posttrain/build_heldout_instance_v1.py --policy FROZEN_HELDOUT_POLICY --adoption ROOT_PROSPECTIVE_HELDOUT_ADOPTION --sampling COMPLETE_SAMPLING_RECEIPT --sampling-summary MODEL_AND_DISPOSITION_SUMMARY --requests ELIGIBLE_REQUESTS --vocab WINNER_GGUF --val-split REGISTERED_VAL --train-split REGISTERED_TRAIN --training-protocol V7_PROTOCOL --env-file DESIGNATED_RUNTIME_ENV --state-dir NEW_HELDOUT_STATE --out NEW_INSTANCE_JSON
python3 tools/posttrain/assess_preference_gates_v1.py --evaluation-dir UNPACKED_EVALUATIONS --parity-dir PARITY_OUTPUT --training-binding DPO_BINDING --heldout-instance HELDOUT_INSTANCE --heldout-state HELDOUT_STATE --vocab WINNER_GGUF --out NEW_VERDICT_JSON
```

Only root may activate GPU work through the existing watchdog and serialize it with other GPU campaigns. Only root may activate the held-out API runner after its exact instance and accounting migration are independently admitted.

## Remaining readiness limits

The whole completion chain is not launch-ready. Eligible API judging must finish before the dataset and actual training source/weights/pairs can be bound. The prospective held-out adoption and final root source/paid admission records do not yet exist in this packet.

The draft single-job held-out sampler still lacks its own durable, request-bound, fsynced resumable journal with original global indices. That journal must checkpoint within about three minutes on the persistent RunPod volume and resume safely from remote or home evidence, or sampling must use separately admitted bounded chunks. A full home pull every three minutes is not required. The draft sampler is deliberately written to `draft_heldout_sampling_job.json` and excluded from the standard training/evaluation jobs file. Do not activate it until that real resume contract is repaired and independently reviewed. Existing training checkpoint relay and resume remain provided by the standard runner. Its ten-step checkpoint interval requires a wall-time check against the three-minute bound in root's launch admission.

These scripts have offline tests, including source/data tamper boundaries, but no new actual-device lifecycle or numerical proof is claimed. Root must independently review the prospective packet, verify the final exact source and inputs, and bind the paid operational instance. Preserve all original failed and negative evidence.
