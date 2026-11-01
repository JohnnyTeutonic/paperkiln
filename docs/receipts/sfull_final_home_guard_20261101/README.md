# S-full final checkpoint home guard

Prospective continuation correction. The runner now relays full checkpoints for FINISHED as well as RUNNING S-full jobs, and refuses to mark a finished job home until the exact final step has passed complete payload and source/recipe validation. Previously the final fetch could bypass the full optimizer checkpoint relay. No completed S-full observation is reclassified; seed 1 remains historically halted at step 675.

The numerical source, 25-step interval and 180-second guard are unchanged. This is separate from Claude's approval of the earlier publication-lock repair and needs a bounded follow-up review before admission.

Validation: 85 tests passed across sfull_checkpoint_v2, sfull_checkpoint_transport_v2, sfull_operational_v2 and dpo_checkpoint_transport_v1. The actual relay regression imports a final step 10000 over home step 675, restores all model/AdamW/RNG/event payloads, and tests successful, failed and partial diagnostic downloads. Separate tests reject corrupted final optimizer data and mismatched source identity. Tests use small synthetic payloads; they do not establish real L4 checkpoint latency.
