# Chunked (parallel-in-time) selective-scan forward: protocol

Claude Opus 5.5, 3 Nov 2026. Fixed before any GPU result. Branch `claude/mamba-parallel-scan`.

## Claim and scope

`MICROTORCH_SCAN_CHUNK=L` (L >= 1) runs the selective-scan forward as a chunked scan
(`src/selective_scan_chunked.cuh`). Unset or 0, the ordered kernels run unchanged, and so do the default, the CPU path
and the backward. The claim is limited to the forward kernel on CUDA. It is opt-in, its correctness holds within
the frozen selective-scan bound, and its kernel time is measured against the ordered kernel at Mamba-130m and
Mamba-370m-like shapes. It is not a model-speed, training-speed or quality claim, and it changes no default.

The chunked path does not reproduce the ordered path's bits. It differs only through the chunk starts, by design.
So its contract is the binary64 bound, not bit identity with the ordered scan. One chunk per sequence (L >= sl)
equals the ordered recurrence exactly; the CPU proof checks this.

## Gates (all required for "correct")

1. **Host test** (`tests/test_selective_scan_chunked.cpp`, through the host emulation built from the device
   functions). It establishes the algorithm, indexing and chunk boundaries on the host, not device numerics: the
   host uses `std::exp`/`std::log1p` and volatile-rounded float operations, the device CUDA `expf`/`log1pf` and
   `__fmul_rn`/`__fadd_rn`.
   - 173 cases: sequence lengths 1-4096 around chunk edges, L in {1, 2, 16, 64, 128, 256}, softplus on and off,
     skip on and off, mixed, long-memory and fast-decay regimes, one and several sequences.
   - y and every state within 3e-5 of an independent binary64 recurrence, under the metric
     |v - ref| / max(1, |ref|). That is stricter, by up to a factor of 2, than the frozen gate's |v - ref| / (1 + |ref|).
   - One chunk per sequence (L = sl, sl + 1 and 4 sl + 7) bit-identical to the host ordered recurrence.
   - Repeats bit-identical, measured.
   - It runs locally before the GPU run and again on the VM at the run's commit.
2. **Frozen CUDA gate, unchanged.** The 31 Oct gate's configure, build and
   `test_selective_scan_cuda --require-cuda`, then its independent replay (`tools/replay_selective_scan.py`).
   - It runs three times: `MICROTORCH_SCAN_CHUNK` unset (the control, a prerequisite), 2 and 3.
   - At L = 2 the lengths 3, 4 and 9 span several chunks with short tails; at L = 3, lengths 4 and 9 do. Most
     mask cases have length 2 and are a single chunk. So this gate checks the chunked forward feeding the ordered
     backward, through the op and its residency modes, on those shapes only; it is not multi-chunk coverage of all
     220 cases.
   - All 220 cases and the replay must pass in each run.
   - Path selection: a malformed value (`" 2"`) must be refused on the device path.
3. **Long-sequence device check** (`tools/scan_bench.cu`, device-resident buffers, the kernels called directly).
   - For the ordered kernel and every L: y and every state within 3e-5 of binary64 (same metric as gate 1); any
     nonfinite value fails.
   - All state slots written: pre-filled with NaN, so an unwritten slot fails the check.
   - Repeats bit-identical.
   - One chunk per sequence (L = sl and sl + 5) bit-identical to the ordered kernel on the device.

The 3e-5 bound is a fixed pass/fail criterion for these fixtures, not a general error bound for arbitrary
4096-step inputs. A failure of any gate blocks the claim. The ordered path stays the only supported one, the
failure is preserved, and the bound is not relaxed after the result.

## Performance (reported, not a gate)

Median of 20 CUDA-event timings after 3 warm-ups, ordered and chunked alternating within each repetition, for the
forward kernel(s) only. The time is the elapsed default-stream time: for the chunked path it spans its five
launches, including any idle gaps between them. It excludes the chunk-buffer allocation that the op itself makes
on every forward. Shapes (sl, batch, D, N):
(512, 1, 1536, 16), (2048, 1, 1536, 16), (2048, 4, 1536, 16), (4096, 1, 2048, 16), (1024, 8, 768, 16),
(64, 16, 1536, 16). L in {32, 64, 128, 256}. The ratio ordered/chunked is reported for every cell; no cell is
dropped or rerun. Outcomes:
- If chunked is slower at every L for a shape, that is a valid negative for that shape.
- A default change, or a model-level speed claim, needs its own protocol.

## Budget and procedure

- One Colab L4 session, AGENTS.md path C, through `drive.py`, run in a foreground WSL process. The session is
  named `claude-scan-RUN_ID`.
  - The preflight needs a successful session listing, orphan listing and adopt listing, and refuses if the
    name is taken.
  - Creation is reconciled against the listing whatever the creation call returned.
  - Shutdown counts as done only after an authoritative listing that shows neither our name, nor our endpoint,
    nor any endpoint that is new since before the creation call. A VM whose name the CLI lost appears as `[?]`;
    it is not stopped unless it is ours by endpoint, but it blocks certification. Otherwise the driver reports
    unresolved cleanup and exits 3. It ends with a session and orphan inventory.
  - Logging and cleanup never raise. Signals request cancellation, which is observed at controlled points.
- Deadlines from the driver's start: polling ends at 60 minutes, retrieval at 75, shutdown attempts at 85, the
  final inventory at 90. Every external
  call has a timeout capped by its deadline, timed-out children are killed, and a call past its deadline is not
  started. Expected about 30 minutes. Within Claude's Colab lane envelope (stack page).
- The VM checks out a full commit SHA, verifies HEAD and a clean tree before and after the build, and works in a
  fresh per-run directory. It never modifies the source.
  - It writes `status.json` atomically: every gate's own exit status, the script's exit status, run id, commit
    and HEAD. The `DONE` marker follows it on every exit path after argument validation.
  - A failed prerequisite (checkout, configure, build, host test, ordered control) stops further paid work.
- Outputs come home to `/mnt/c/ml_artifacts/mamba_parallel_scan_20261103/RUN_ID/`. Receipts are copied to
  `receipts/RUN_ID/` here before any analysis, and the driver log is copied after cleanup.
- The driver validates `status.json`: run, commit and HEAD; exactly the 13 required gates, all 0; VM exit 0.
  Exit codes: 0 validated, 1 experiment failed (receipts kept), 2 transport, receipt or status-identity failure,
  3 cleanup unresolved. Test hooks are refused unless the test harness flag is set.
- `test_drive.py` (mock Colab CLI) and `test_vm_run.py` (stub tools) test that bookkeeping locally. The local
  host-test receipt is `receipts/host_test_local.json`.
- The run is not repeated to improve a result. An operational failure (session lost, build failure) is recorded
  as such and may be rerun once, unchanged.
