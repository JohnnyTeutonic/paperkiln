# In flight right now

*The living page. If you are picking this up cold, read this first, then
[`BACKLOG.md`](BACKLOG.md). Update this file when state changes — a
stale NOW.md is worse than none.*

**Last updated: 1 Sep 2026, ~01:20 AEST.**

## Running

**transfer_s1 bridge arm** — 10 runs (exact + swa64s1 x seeds 1-5) at
d=256 on a Colab **L4, 4 cells concurrent**, driven by
`tools/colab_transfer_runner.py --gpu L4 --jobs 4 --omp 2`
(session `tr-bridge`, local out `/mnt/c/ml_artifacts/transfer/bridge`).

**Why L4 and not T4.** The T4 runtime has 2 vCPU and was CPU-bound
(load 2.1 while the GPU idled at 11%) — ~52 min per run against a
~52 min reclaim interval. Since only COMPLETED runs are banked, runs
kept dying just short of the line and the arm sat at 0/10 for two
hours. The L4 has 12 vCPU; 4 concurrent cells give 3.4x throughput at
~41 min per run, which fits inside a vm lifetime. Details and the
scaling table: execution clarification 5 in the pre-registration.

The GPU change is numerics-relevant (reduction order — see
`../../atlas/THEOREM_CROSSING.md`), so **every arm runs on L4** and no
partially-completed T4 work was kept.

**Running on the device op set** (`MICROTORCH_DEVICE_OPS=1`) since
`697e281` fixed the leak that had forced gemm-only (BACKLOG 4c: `In`'s
`owned` flag clobbered by member initialization order, so nothing was
ever freed). Re-measured flat over 400 steps and 1.31x faster than
gemm-only. Observed pace on the live arm: **~1.2 s/step, GPU memory
flat at ~361 MiB** — so roughly 70 min per run, ~12 h for the arm.

Deferral (`MICROTORCH_DEFER_DOWNLOADS`) stays OFF — BACKLOG 4b is still
open and still crashes mtstudio at step 1.

**The first attempt (19:41–22:24) was abandoned, not lost to a bug.**
On gemm-only it ran at 7.5 s/step and decelerating, i.e. ~7 h per run
against a 1–2 h reclaim interval, so no run could ever finish and be
banked. It was stopped deliberately. That deceleration was very likely
self-inflicted — heavy memory probes were run on the same GPU as the
live arm. Do not probe a live arm; bring up a second session instead.

Check it with:
```
wsl -e bash -lc "tail -3 /mnt/c/ml_artifacts/transfer/bridge_driver.log; pgrep -af colab_transfer_runner"
```

**Do not launch a second driver for an arm that already has one.** The
runner stops and recreates its session on launch, so a duplicate kills
the live run.

Note that `bridge_probe.py` in the session scratchpad reported "no
processes" twice while the arm was in fact healthy (4 processes, repo,
binary, sweep log all present on a direct query). Trust a direct
`colab exec` query over that script, and trust the driver's own
`sweep_alive()` — which was right both times.

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
