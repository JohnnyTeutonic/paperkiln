# In flight right now

*The living page. If you are picking this up cold, read this first, then
[`BACKLOG.md`](BACKLOG.md). Update this file when state changes — a
stale NOW.md is worse than none.*

**Last updated: 31 Aug 2026, ~19:00 AEST.**

## Running

**transfer_s1 bridge arm** — 10 runs (exact + swa64s1 x seeds 1-5) at
d=256 on a Colab T4. **Running with DEFER_DOWNLOADS OFF** — the
first attempt died at step 1 in all ten cells with heap corruption;
see BACKLOG 4b. `res` (ops + residency) is validated and converges
identically, so the study runs on it, driven by `tools/colab_transfer_runner.py`
(session `tr-bridge`, local out `/mnt/c/ml_artifacts/transfer/bridge`).

Check it with:
```
wsl -e bash -lc "tail -3 /mnt/c/ml_artifacts/transfer/bridge_driver.log; pgrep -af colab_transfer_runner"
```

**Do not launch a second driver for an arm that already has one.** The
runner stops and recreates its session on launch, so a duplicate kills
the live run.

## What happens when the bridge finishes

1. Copy receipts into `experiments/transfer_s1/receipts/bridge/`.
2. `python tools/validate_events.py <receipts>/*_events.jsonl`
3. `python3 experiments/transfer_s1/analyze.py --bridge /mnt/c/ml_artifacts/transfer/bridge --banked ~/boundary_out`
4. **Gate PASS** → launch arm S (72 runs). **Gate FAIL** → the study
   HALTS and the discrepancy is written up as its own finding. That
   outcome is not a disaster; see `BACKLOG.md`, "if the gate fails".
5. Order is fixed by the pre-registration: **bridge → S → M → L.**

## Watch for

- **~10 s/step instead of ~0.3** means a CPU-only binary got built
  again. Stop and investigate rather than burning units — this exact
  failure cost a night on 30 Aug. Root cause then: the CUDA wiring lived
  in an uncommitted working copy while Colab clones master.
- Orphaned Colab sessions:
  `python3 tools/colab_reap_orphans.py --kill --keep tr-bridge,tr-S,tr-M,tr-L`

## Compute

~631 units as of 31 Aug. The whole transfer study is priced at ~32
units (≈5%). Units are not the constraint; wall clock and VM reclaims
are. Parallelism across sessions costs the same total.

## Not running, deliberately

- **sparse_s1_longbudget** — pre-registered (12000 steps, 10 seeds) and
  ready, but queued behind the transfer arms so the two do not compete
  for sessions.
