# Chunked (parallel-in-time) selective-scan backward: protocol

Claude Opus 5.5, 3 Nov 2026. Fixed before any GPU result. Branch `claude/mamba-chunked-backward`. Design, recurrences
and the tolerance argument: `BACKWARD_DESIGN.md`. It runs only after independent review, from the reviewed commit.

## Claim and scope

`MICROTORCH_SCAN_CHUNK=L` (L >= 1) runs the selective-scan backward as a chunked scan
(`src/selective_scan_chunked_backward.cuh`), as it already runs the forward chunked. Unset or 0, the ordered kernels
run unchanged, and so do the default and the CPU path. The claim is limited to the backward kernels on CUDA:
opt-in; correct within the frozen selective-scan bound against binary64 and against the ordered backward;
deterministic; the default path's device outputs and gradients bit-identical to master's; and its kernel time
measured against the ordered backward at the forward protocol's Mamba-130m and Mamba-370m-like shapes. It is not a
model-speed, training-speed or quality claim, and it changes no default.

The chunked backward does not reproduce the ordered backward's bits, by design (`BACKWARD_DESIGN.md`, rounding). Its
contract is the bound. With one chunk per sequence it equals the ordered backward exactly (dA and dD for one
sequence); the host test checks this, and gate 4 checks it on the device.

## Gates (all required for "correct")

1. **Host tests**, locally (`receipts/host_test_local.json`, `receipts/host_test_bwd_local.json`) and again on the
   VM at the run's commit: the forward's 173 cases (`tests/test_selective_scan_chunked.cpp`, unchanged) and the
   backward's 173 (`tests/test_selective_scan_chunked_backward.cpp`, the checks in `BACKWARD_DESIGN.md`).
   Prerequisites.
2. **Provenance.** Exactly one NVIDIA L4. The image's own CUDA toolkit, recorded (`nvcc.txt`, `cuda_dirs.txt`). No
   toolkit is installed: run r1-20261103 built this repository with the image's CUDA 13.0.88 (its archived
   `configure.log`), and this branch compiles with CUDA 12.6 locally. Prerequisite.
3. **Frozen CUDA gate, unchanged.** The 31 Oct gate's configure, build and
   `test_selective_scan_cuda --require-cuda`, then its independent replay (`tools/replay_selective_scan.py`).
   - It runs three times: `MICROTORCH_SCAN_CHUNK` unset (the control, a prerequisite), 2 and 3.
   - At L = 2 and 3 the op now runs both the chunked forward and the chunked backward. Lengths 3, 4 and 9 (L = 2)
     and 4 and 9 (L = 3) span several chunks, with short tails; the 2-row mask cases are one chunk at L = 2 and
     test the one-chunk path through the op. So this gate checks the chunked backward through the op, its
     residency modes and every gradient mask, on these shapes only; it is not multi-chunk coverage of all 220 cases.
   - All 220 cases and the replay must pass in each run. Its bound and metric are the frozen ones,
     |v - ref| / (1 + |ref|) < 3e-5, on sums of at most 18 rows.
   - Path selection: a malformed value (`" 2"`) must be refused on the device path.
   - The replay's numpy is 2.2.6 from a binary wheel (`--only-binary=:all:`). The image's Python was 3.13 on 3 Nov,
     for which numpy 1.26.4, the forward protocol's pin, has no wheel (r1 built it from source). The replay passes
     under numpy 2.2.6 on the CPU path's 220 cases (worst 1.57e-7).
4. **Long-sequence device check** (`tools/scan_bwd_bench.cu`, device-resident buffers, the kernels called directly,
   the ordered forward's states and an upstream gradient as inputs, all six gradients requested). For every shape
   and L:
   - the ordered backward and the chunked backward within 3e-5 of the binary64 backward
     (`tests/selective_scan_backward_reference.hpp`) under |v - ref| / max(1, mag), and the chunked within 3e-5 of
     the ordered under the same metric; any nonfinite value fails;
   - every output written (pre-filled with NaN); repeats bit-identical;
   - one chunk per sequence (L = sl and sl + 5): dx, ddt, dB, dC bit-identical to the ordered kernels on the device,
     and dA and dD as well at batch 1;
   - the chunked forward's states (same L) into the chunked backward within 3e-5 of binary64.
5. **Default path identical to master.** The base commit `d0390e39` (master when the branch was cut) is built in
   the same way, and its frozen test runs with the variable unset. Its 220 case lines (inputs, every requested
   gradient and y, each at %.9g, which round-trips binary32 exactly) must equal the branch's control run's, byte for
   byte.

The 3e-5 bound is a fixed pass/fail criterion for these fixtures (`BACKWARD_DESIGN.md` gives the argument), not a
general error bound. A failure of any gate blocks the claim. The ordered path stays the only supported one, the
failure is preserved, and the bound is not relaxed after the result.

## Performance (reported, not a gate)

Median of 20 CUDA-event timings after 3 warm-ups, ordered and chunked alternating within each repetition, for the
backward kernels only, all six gradients requested (a training step's case). The time is the elapsed default-stream
time: two launches for the ordered backward, eight for the chunked, idle gaps between them included. It excludes
the scratch allocation that the op makes on every backward. Shapes (sl, batch, D, N), as the forward protocol:
(512, 1, 1536, 16), (2048, 1, 1536, 16), (2048, 4, 1536, 16), (4096, 1, 2048, 16), (1024, 8, 768, 16),
(64, 16, 1536, 16). L in {32, 64, 128, 256}. The ratio ordered/chunked is reported for every cell; no cell is
dropped or rerun. Outcomes:
- If chunked is slower at every L for a shape, that is a valid negative for that shape.
- A default change, or a model-level or training speed claim, needs its own protocol.

## Budget and procedure

- **Command.** From a paperkiln checkout's root in a foreground WSL process, after the reviewed commit is pushed
  (the VM fetches `vm_run.sh` at that commit):
  `python3 experiments/mamba_parallel_scan_2026-11-03/drive.py FULL_COMMIT_SHA RUN_ID backward`, with RUN_ID
  `b1-YYYYMMDD`.
- **One Colab L4 session**, AGENTS.md path C, named `claude-scan-RUN_ID`. Cap: that one session, at most 90
  minutes from the driver's start, within Claude's Colab lane (stack page: one L4 at a time, at most 100 compute
  units a week). Expected about 20-30 minutes: two CUDA builds of the library (the whole of r1 took 6 minutes), the
  host binary64 references (single-threaded, about 2.5 GB of RAM at the largest shape) and the timings. Device
  memory peaks near 3.3 GB at (2048, 4, 1536, 16).
- **Deadlines** from the driver's start: polling ends at 60 minutes, retrieval at 75, shutdown attempts at 85, the
  final inventory at 90. Every external call has a timeout capped by its deadline, timed-out children are killed,
  and a call past its deadline is not started.
- **Session ownership and shutdown** as the forward protocol: the preflight needs a successful session listing,
  orphan listing and adopt listing, and refuses if the name is taken; creation is reconciled against the listing
  whatever the creation call returned; shutdown counts as done only after an authoritative listing that shows
  neither our name, nor our endpoint, nor any endpoint new since before the creation call. The CLI's `[colab]`
  banner rows are not sessions. Otherwise the driver reports unresolved cleanup and exits 3.
- **Launch and stop hardening** (from the Gemma packet, ember `experiments/hf_cuda_2026-10-30/gemma_colab_20261103/`):
  - the launch writes `vm_run.sh` to its own temporary file and renames it into place, so a delayed duplicate
    launch never rewrites a script a running bash is reading;
  - `vm_run.sh` claims `/content/run_RUN_ID` with an atomic `mkdir`, so a second launch of the same run exits
    before doing anything;
  - a launch without acknowledgement is reconciled by a read-only probe (run directory or PID file), and is retried
    once only if the script is shown absent; an unknown state is never relaunched;
  - read-only and idempotent calls (probe, pack, download, stop) are retried; each retrieval extracts into its own
    staging folder, and only a complete extraction is used;
  - `vm_run.sh` publishes its PID once its traps are set, and runs every gate command in its own process group;
    if polling ends without DONE, the driver checks that the PID is still the script and sends SIGTERM; the script
    stops the running command's group (TERM every second for up to 60 s, then KILL), writes `stop.json` and exits
    through the trap that writes `status.json` and DONE. An unconfirmed stop, or a group that outlived a command,
    makes the exit at least 2.
- **The VM** checks out the full commit SHA, and the base as a detached worktree; it verifies both HEADs and clean
  trees before and after the builds, and works in the fresh run directory. It never modifies the source. It writes
  `status.json` atomically (every gate's own exit status, the incoming shell status, the aggregate final status,
  protocol, run id, commit and HEAD); only if that write succeeds does the `DONE` marker follow, on every exit path
  once the run directory exists. A failed prerequisite (provenance, checkout, configure, build, either host test,
  numpy, the ordered control) stops further paid work.
- **Validation.** The driver validates `status.json`: protocol `backward`, run, commit and HEAD; exactly the 17
  required gates (`provenance`, `checkout`, `configure`, `build`, `host_test`, `host_test_bwd`, `numpy`,
  `strict_unset`, `replay_unset`, `strict_2`, `replay_2`, `strict_3`, `replay_3`, `switch_refuses_malformed`,
  `build_base`, `default_identical`, `scan_bwd_bench`), all integers equal to 0; aggregate exit 0. A validated success
  also needs the matching `DONE` marker to have been seen; a failed log write makes the exit at least 2. Exit codes:
  0 validated, 1 experiment failed (receipts kept), 2 transport, receipt, status-identity or stop-confirmation
  failure, 3 cleanup unresolved. Test hooks are refused unless the test harness flag is set.
- **Outputs** come home to `/mnt/c/ml_artifacts/mamba_parallel_scan_20261103/RUN_ID/`. After the VM is gone, the
  top-level files of `out/` are copied to `receipts/RUN_ID/` here with `archive.sha256` hashing every retrieved file,
  then the driver log.
- **Tests.** `test_drive.py` (mock Colab CLI, 19 tests) and `test_vm_run.py` (stub tools, 16 tests) test that
  bookkeeping locally, for both protocols, including the lost-acknowledgement, absent-launch, unfinished-run,
  unconfirmed-stop and second-launch paths.
- The run is not repeated to improve a result. An operational failure (session lost, build failure) is recorded
  as such and may be rerun once, unchanged, under a new RUN_ID.
