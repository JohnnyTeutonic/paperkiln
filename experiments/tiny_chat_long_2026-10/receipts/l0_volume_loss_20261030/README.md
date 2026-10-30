# L0 partial-output loss, 30 October 2026

This is an operational failure receipt, not a completed L0 result. No L0 score or scientific conclusion is available from this run.

The original L0 pod `lvxo04p2pncblo` reached its cumulative $3 cap and stopped at 06:25:23 UTC. Its persistent volume `y867xwx6ny` in EU-RO-1 held all 1,452 generated responses and at least 214 completed score chunks (last directly observed at 06:17 UTC). The exact final chunk count was not recovered. Completion-only output fetching had not relayed these partial artifacts locally.

A watchdog replacement under the approved cumulative $4 ceiling encountered no L4 stock in EU-RO-1. The transport automatically moved the campaign to new volume `vnwyd6e9jp` in EUR-IS-1, recording the original as retired. At 06:33:49 UTC, the replacement driver was gracefully stopped to avoid recomputation and preserve the original volume. Its new pod `qv7vvy5l25kmst` terminated; both volumes still existed in the immediate API snapshot. The watchdog entry was then disabled by the parent agent.

At 06:35:46 UTC the independent scheduled RunPod sweeper deleted the original volume. Its rule accepted `retired=True` as sufficient for deletion when the owning driver was dead, even though the campaign was not home. The preserved sweep excerpt explicitly records this deletion; the state ledger records timestamp 1793342146.9735975 and the subsequent API list omits the original volume. Thus stopping the replacement protected against transport-create deletion but exposed a second unsafe sweeper deletion path. No public API recovery or replacement run has been attempted in this receipt.

The new volume remains preserved, but had not completed preparation and is not evidence that the original generated responses or score chunks survived. The local L0 artifact directory was empty at inspection. The original scientific inputs, source pin, and local upload artifacts remain intact; the lost material was this run's remote partial outputs. No additional funds are authorized by this receipt and the existing cumulative $4 limit remains unchanged.

The accompanying state and logs are byte copies captured after deletion. `runpod_sweep_observed.py` records the unsafe deletion rule at inspection. `volumes_after.json` is a timestamped provider API response; `SHA256.json` hashes these archived files. The immediate pre-deletion API observation was reported to the parent during rescue but was not itself saved as a standalone API receipt. All times above are UTC; log timestamps use the host's UTC+11 clock.
