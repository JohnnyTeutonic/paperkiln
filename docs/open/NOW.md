# In flight right now

*The living page. If you are picking this up cold, read this first, then
[`BACKLOG.md`](BACKLOG.md). Update this file when state changes — a
stale NOW.md is worse than none.*

**Last updated: 4 Sep 2026, ~16:00 AEST.**

## NOTHING IS RUNNING

No drivers, no Colab sessions. This is deliberate, not a stall.

## Done

- **Bridge gate: PASS.** 10/10 banked and validated. Worst early-step
  relative val-loss difference 3.24e-04 against the pre-registered
  1e-03 threshold. Receipts and `GATE_OUTPUT.txt` in
  `../../experiments/transfer_s1/receipts/bridge/`.
- **Arm S: COMPLETE.** 72/72 banked and validated, 12 seeds x 6 lanes,
  balanced 12 per lane. Receipts in `receipts/S/`. Commit `56b49b0`.

## BLOCKED: arms M and L, on a hard 60-minute Colab session cap

**The finding.** Every Colab session ends at 60-61 minutes, on every
GPU tier, regardless of load. Measured from the CLI's own history
(`~/.config/colab-cli/history/tr-*.jsonl`) across 23 arm-S sessions:
62, 61, 60, 60, 62, 60, 62, 61, 61, 63, 70, 61, 62, 62 minutes. The
bridge arm's T4 sessions show the same: 60, 60, 61, 60, 61.

This is **not** ours to fix:

- The CLI's keep-alive daemon IS spawned by `colab new` and IS running
  (verified: process alive, `keep_alive_started` logged for all 23
  sessions, and **zero** `keep_alive_error` events).
- The termination reason is always `pruned`, which in
  `colab_cli/common.py` means the endpoint has disappeared from the
  server's active assignments. Colab drops the assignment; the CLI
  merely notices.
- It is account-wide, not GPU-specific. T4 and L4 prune identically.

**Consequence.** Any run longer than ~55 min can never complete, because
only finished runs are banked. Arm S succeeded because a 4-cell wave
took ~52 min. Arm M at d=512 measured **1.23 s/step single-cell (74
min/run)** and **1.66 s/step at 4 cells (100 min/run)**. No `--jobs`
value and no GPU tier helps. Arm L at d=1024 is worse again.

**Do not restart arm M as configured.** It will bank zero runs and burn
units. It was stopped for exactly this reason on 2 Sep.

## Checkpoint/resume: DONE end to end, proven on CUDA (4 Sep 2026)

**Green-lit by Jonathan 4 Sep; implemented, relayed and proven the same day.
The `HALTED` sentinels on M and L are lifted. M was launched at 17:35 on a
T4 by mistake (runner default `--gpu`; PREREGISTRATION clarification 5
requires L4, no mixing), stopped at 18:30 with its T4 work discarded, and
relaunched on L4 with 4 cells. The runner default is now L4.**

`tools/mtstudio.cpp` writes three files per checkpoint: `model.safetensors`,
`optim.safetensors` (every AdamW m/v matrix and every Muon momentum buffer),
and `state.txt` (line 1 the step, as before; line 2 JSON with the AdamW
timesteps and early-stopping state). Resume restores all of it, trims
`events.jsonl` back to the checkpoint step (a killed run has logged past
it), and the `resume` event reports `"optimizer": "restored"`, or `"cold"`
for an old checkpoint without `optim.safetensors`, never silently. The RNG
needs no saving: the batch stream is replayed exactly (accum*batch draws
per step, fast-forwarded on resume) and nothing in the tree uses dropout.

`tools/colab_transfer_runner.py` relays every unfinished cell's latest
complete checkpoint home every `--partial-every` seconds (write-completeness
guard: state.txt still for 30 s and newer than both safetensors), pushes
finished markers plus partials up on the next session, and mtstudio resumes
mid-run. Uploads over 32 MB go up in retried 32 MB chunks with a byte-count
check (a single 111 MB upload died with an SSL EOF on the first probe). The
binary cache is keyed by GPU and a fingerprint of the build inputs at
origin/master, so a source change costs one rebuild rather than shipping a
stale binary forever.

**Proof, CPU:** `tools/test_resume.sh`, ALL PASS for `adamw` and `muon`;
emulated kill (checkpoint at 20, logged to 30), bit-identical losses at
steps 21 to 40, byte-identical final weights, no duplicate step lines;
negative control (optimizer state deleted) diverges.

**Proof, CUDA (the one that matters):** `experiments/transfer_s1/sweep_probe.json`,
an M-shaped cell (d=512, 16 heads, exact, seed 21, 900 steps, checkpoint
every 100) run twice on Colab T4: once uninterrupted, once with `colab stop`
by hand after the step-100 checkpoint had been relayed home, the driver
recreating the VM, pushing the partial and resuming. Result: `resume` event
`optimizer=restored` at step 100; **800 of 800 post-resume step losses and
all nine evals identical to the float** (max |dloss| = 0). A resumed run is
the same run. Receipts: `/mnt/c/ml_artifacts/transfer/probe_ref/` and
`probe/`, drivers' logs beside them.

**Arm configuration:** `sweep_M.json` `checkpoint_every` 200, `sweep_L.json`
100; PREREGISTRATION.md execution clarification 6 records it as a config
change made before any M or L receipt existed. GPU: L4 for every arm
(clarification 5); the T4 ceiling re-measured on 4 Sep is ~1.4 s/step
aggregate at any concurrency, so T4 is a probe machine only.

*(Historical description of the gap, kept for the record:)*

## The only route to M and L: checkpoint/resume

`mtstudio` ALREADY has the skeleton. `tools/mtstudio.cpp`:

- line 548 saves every `ckpt_every` steps
- lines 444-453 read `out_dir/state.txt` for `start_step`, load
  `model.safetensors`, and emit a `resume` event
- the sweeps disable it with `checkpoint_every: 1000000`

**Two gaps make it unusable for THIS study as it stands.** The save
lambda (line 464) writes model weights and a step number, nothing else:

1. **Optimizer state is not saved.** AdamW's per-parameter `m` and `v`
   moments and the bias-correction timestep `t` are lost, so a resumed
   run restarts the optimizer cold and follows a different trajectory.
2. **RNG state is not saved.** `std::mt19937 rng(123 + 1000003u *
   s.seed)` is reconstructed at startup, so a resumed run replays the
   batch sequence from step 0 while the weights sit at step N. That
   silently breaks the pre-registered guarantee that seed varies both
   init and batch composition.

For a study whose subject is whether numerical details change
conclusions, both are disqualifying.

**The work:** extend `save`/resume to cover AdamW moments and RNG
state, then prove it — run 200 steps uninterrupted, run 100 + resume +
100, and assert the loss traces are bit-identical. The test is the
deliverable, not the feature. Awaiting Jonathan's go-ahead as of
2 Sep 07:10.

## Historical note on the GPU choice

Arms ran on **L4** (execution clarification 5). The original reason was
that the T4 runtime has 2 vCPU and was CPU-bound; the L4's 12 vCPU let
4 cells run concurrently at ~52 min per wave. The GPU choice is
numerics-relevant (reduction order, see
`../../atlas/THEOREM_CROSSING.md`), so every arm carrying a claim uses
L4 and no partially-completed T4 work was kept. **Note the earlier
belief that T4 sessions lived 2h43m was wrong** — that was three
consecutive 60-minute sessions with the run restarting each time, which
is why it never finished.

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
