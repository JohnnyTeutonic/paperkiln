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

1. **CPU proof** (`tests/test_selective_scan_chunked.cpp`, host emulation built from the device functions).
   - 140 cases: sequence lengths 1-4096 around chunk edges, L in {2, 16, 64, 256, 128}, softplus on and off,
     skip on and off, mixed, long-memory and fast-decay regimes, one and several sequences.
   - y and every state within 3e-5 (|v - ref| / max(1, |ref|)) of an independent binary64 recurrence.
   - One chunk per sequence bit-identical to the ordered recurrence.
   - Repeats bit-identical.
2. **Frozen CUDA gate, unchanged.** The 31 Oct gate's configure, build and
   `test_selective_scan_cuda --require-cuda`, then its independent replay (`tools/replay_selective_scan.py`).
   It runs three times: `MICROTORCH_SCAN_CHUNK` unset (control), 2 and 3. Tiny shapes then span several
   chunks, and the chunked forward feeds the ordered backward. All 220 cases and the replay must pass in each run.
3. **Long-sequence device check** (`tools/scan_bench.cu`, on device-resident buffers).
   - For every shape and L: chunked y and last states within 3e-5 of binary64.
   - All states written (pre-filled with NaN) and finite.
   - Repeats bit-identical.
   - The ordered kernel is checked on the same inputs.

A failure of any gate blocks the claim. The ordered path stays the only supported one, and the failure is
preserved.

## Performance (reported, not a gate)

Median of 20 CUDA-event timings after 3 warm-ups, for the forward kernel(s) only. Shapes (sl, batch, D, N):
(512, 1, 1536, 16), (2048, 1, 1536, 16), (2048, 4, 1536, 16), (4096, 1, 2048, 16), (1024, 8, 768, 16),
(64, 16, 1536, 16). L in {32, 64, 128, 256}. The ratio ordered/chunked is reported for every cell; no cell is
dropped or rerun. Outcomes:
- If chunked is slower at every L for a shape, that is a valid negative for that shape.
- A default change, or a model-level speed claim, needs its own protocol.

## Budget and procedure

- One Colab L4 session, AGENTS.md path C: one script; create, run, download, `colab stop` on every exit.
- Deadline 90 minutes from creation; expected about 30. Within Claude's Colab lane envelope (stack page).
- The VM clones this branch at a named commit and records it; outputs come home to
  `/mnt/c/ml_artifacts/mamba_parallel_scan_20261103/`, and receipts are copied here.
- The run is not repeated to improve a result. An operational failure (session lost, build failure) is recorded
  as such and may be rerun once, unchanged.
