# DPO V3 held-out intake: offline independent review

Status: preparation only, not a judging job or admission. No provider calls or credential-file reads. Frozen source is Paperkiln b444700195625570130913646bb5ad86c364f4c4. Source-pinned details and exact future commands are in `source_audit.json` and `intake_plan.json`.

## Intake blockers

1. Returned V3 sampling outputs are not yet present. Require all four: sampling_receipt.json, sampling_summary.json, sampling.journal.jsonl, requests.jsonl. Do not create their contents or fill missing prompts.
2. Original adoption's consumer map has three older hashes: DPO checkpoint core, checkpoint transport and training-instance builder. V3 also binds materialize_preference_v7.py, absent from the older consumer map. The frozen policy and all 26 judge/sampler/parser operational hashes match exactly. These are already reviewed operational/layout corrections, not new scientific criteria. Bind this linkage prospectively in a supplementary root receipt before outcome judging; preserve the old adoption unchanged. A deliberately unadopted proposal is supplied.
3. Final accounting and model/output integrity require actual returned evidence. Freeze/replay final old training/prior ledgers and legacy unknown ceilings once, then bind the exact returned sampling universe, final DPO/reference models, common Ember build, source, and fresh held-out instance. No API activation before that root review.

## Important enforcement boundary

`build_heldout_instance_v1.py` checks exact original adoption and operational sources, performs training readiness/accounting work and invokes full sampling replay before writing an immutable instance. It does not inspect credentials. `heldout_gate_runner_v1.load_inputs()` enforces policy, judge profile, source/data hashes, complete 300-item journal, model/request binding and prior ledger hashes. Neither enforces the older adoption's `consumer_source_sha256` map against V3; root must explicitly cover the supplemental linkage, rather than infer the builder will reject it.

## Limits and verdicts

Use unchanged two planner orders for every eligible request, no refill of the original first 300 source-filtered items. Context eligibility was prospectively 288 plus 12 over-context; final eligible sample count can be lower due to unchanged identical/invalid-UTF8 skips, not replacement. The ledger's binding caps stage spend at $25 and attempts at twice actual eligible request count, hence at most 600. Global actual costs plus retained unknown ceilings and original legacy costs stay within $200; batches are at most 24, per-call reserve $0.25, retries zero. A failed/unknown ranking halts and preserves evidence; it is not an exclusion or tie.

Held-out STOPPED completion means complete judging, not adoption. The frozen assessor recomputes protocol <=10%, non-increasing repeats, simple QA no worse than three of 60, CPU and CUDA parity 40/40 across all fields, and judged DPO share and Wilson lower bound both >0.5. All five must pass. It never automatically replaces pk_sft_v1. Preserve negative verdicts.

## Commands and validation

Use the exact argv arrays in `intake_plan.json` only when preconditions are fulfilled. Builder is offline, inspector replays without dispatch, campaign is admitted through the existing watchdog only. Evaluation raw files are packed in dpo_eval/chat_eval_results.tar.gz; safe extraction and archive/model/hash review are needed before the final assessor. Do not run the eventual campaign CLI manually or invent an instance hash.

64 offline tests passed across held-out API accounting, campaign failure/restart, sampling server cleanup, durable sampling journal, and five-gate assessment. Tests use fake requests/providers and temporary files; they do not establish readiness of unreturned V3 outputs. Original adoption, frozen source and generated V3 instance are unchanged.
