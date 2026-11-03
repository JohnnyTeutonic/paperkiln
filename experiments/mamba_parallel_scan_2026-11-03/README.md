# Chunked (parallel-in-time) selective-scan forward: result

Claude Opus 5.5, 3 Nov 2026. Protocol: `PROTOCOL.md`, fixed before the run.
- **Review:** Codex methods review, four rounds; rounds 1-3 DO NOT RUN, round 4 RUN. The reviews are kept in the
  research workspace (`AI_ML/mamba_scan_claude/reviews/`), not in this repository.
- **Run:** `r1-20261103`, one Colab L4 (NVIDIA L4, driver 580.82.07, CUDA 13.0), commit `b2ddc596`, 00:51-00:57 UTC+11.
- **Driver:** exit 0. `status.json` matched the run, commit and HEAD, and all 13 gates were 0. Cleanup was verified:
  no session, and no endpoint new since creation, remained listed.
- **Receipts:** `receipts/r1-20261103/`.

## Gates: all pass

| Gate | Result |
|---|---|
| 1. Host test, locally (`receipts/host_test_local.json`) and on the VM | 173/173 cases; worst chunked 9.85e-7, ordered 1.08e-6 (bound 3e-5); one chunk bit-identical to ordered 519/519; repeats bit-identical 173/173 |
| 2. Frozen 220-case CUDA gate and independent replay, with `MICROTORCH_SCAN_CHUNK` unset, 2, 3 | 220/220 each, `actual_cuda` and `devcheck` true; replay passes in all three (worst normalised 1.6e-7, bound 3e-5) |
| Path selection | Malformed `" 2"` refused with the parser's message. With L = 2 and L = 3, four cases' outputs differ bit for bit from the ordered run, all of them multi-chunk. The remaining multi-chunk cases are bit-identical: on these tiny shapes the chunk-start arithmetic rounds exactly. |
| 3. Long-sequence device check (`scan_bench.jsonl`, 24 rows) | every row ok: chunked y and every state within 1.12e-6 of binary64 (ordered: 1.00e-6); every state slot rewritten; repeats bit-identical; one chunk (L = sl, sl + 5) bit-identical to the ordered kernel at every shape |

## Timings (median of 20, alternating; forward kernel(s) only, device-resident buffers)

| sl | batch | D | ordered ms | chunked ms, best L | speedup range over L = 32-256 |
|---:|---:|---:|---:|---:|---|
| 512 | 1 | 1536 | 3.05 | 0.41 (L 128) | 6.4-7.4x |
| 2048 | 1 | 1536 | 12.28 | 2.20 (L 128) | 5.4-5.6x |
| 2048 | 4 | 1536 | 12.98 | 8.88 (L 256) | 1.39-1.47x |
| 4096 | 1 | 2048 | 24.72 | 5.89 (L 256) | 4.0-4.2x |
| 1024 | 8 | 768 | 5.35-5.62 | 4.40 (L 128) | 1.17-1.28x |
| 64 | 16 | 1536 | 1.21 | 1.06 (L 128) | 1.14-1.15x |

The chunked path was faster in every one of the 24 cells. The gain is largest for single long sequences and
shrinks with batch, because the ordered kernel's parallelism is batch x D threads, while the chunked one adds
chunks x N.

## What this establishes, and what it does not

- **Established:** an opt-in chunked forward for the selective scan on CUDA. It is correct within the frozen
  bound on the frozen gate and on Mamba-130m and Mamba-370m-like shapes, deterministic, bit-identical to the
  ordered kernel when one chunk covers the sequence, and faster as a kernel in every measured cell.
- **Not established:** model-level or training speed (these are component timings); the backward (in r1 it was
  ordered and consumed the chunked states); other GPUs.
- **Default:** unchanged and ordered. Making the chunked path the default, or claiming model speed, needs its own
  protocol (PROTOCOL.md).
- **Follow-ups:** the chunked backward's GPU run and the end-to-end prefill measurement are done (below).
- **Choosing L automatically:** `MICROTORCH_SCAN_CHUNK=auto` selects L = 128 (`src/scan_chunk_switch.hpp`). The
  measured timings leave no room for a shape-dependent rule:
  - forward (r1): across the six cells and L = 32-256, a fixed L = 128 is within 0.2% of each cell's best L;
  - backward (b1): the same holds within 2%;
  - end-to-end prefill (p2): L = 64 and 128 are within 1.5% of each other everywhere.

  `auto` is the value 128 exactly, so it runs the code those gates passed at L = 128 (`tests/test_scan_chunk_switch.cpp`).
  The default stays ordered.
- **Driver note:** the Colab CLI prints `[colab]` banner rows in `sessions`. In r1 the driver read them as session
  rows, which was harmless, since they were in the pre-creation snapshot and constant. The driver skips them.

## Chunked backward: result (run `b1-20261103`, one Colab L4, 3 Nov 2026)

- **Design:** `BACKWARD_DESIGN.md` (recurrences, stored against recomputed, rounding, tolerance). Code:
  `src/selective_scan_chunked_backward.cuh`, opt-in through the same `MICROTORCH_SCAN_CHUNK`; the ordered default is
  unchanged.
- **Review:** Codex methods review, one round, RUN (kept in the research workspace, not here).
- **Gates:** all 17 of `PROTOCOL_BACKWARD.md` pass (`receipts/b1-20261103/status.json`):
  - the host tests;
  - the frozen 220-case gate and replay at L unset, 2 and 3;
  - the malformed switch value refused;
  - the device backward bench;
  - the default path identical to master `d0390e39`.
  The build used the image's CUDA 13.0.88; NumPy as recorded in `numpy_install.log`.
- **Device bench** (`receipts/b1-20261103/scan_bwd_bench.jsonl`, 24 rows, all gradients, median of 20 alternating runs):
  - **Accuracy:** every row is within 2.9e-7 of binary64 under the magnitude metric (bound 3e-5), for both the
    ordered and the chunked backward.
  - **Bit identity and determinism:** with one chunk per sequence the chunked backward equals the ordered one
    bit for bit, every output is written, and repeats are identical.
  - **Speed:** at the best chunk length the chunked backward is 3.4x faster than the ordered one at 512 tokens,
    3.65x at 2048, 2.96x at 4096, 3.8x at batch 4 of 2048, 6.9x at batch 8 of 1024 and 3.8x at batch 16 of 64.
    These are kernel timings, not training speed.
- **What a pass means:** the error is small relative to the expanded expression (the metric scales by the
  binary64 sum of absolute terms, including inside the recurrences), not small relative error in each gradient.
- **Forward mode on this commit:** with the switch set, the forward protocol now also runs the new backward, so it
  is not an unchanged reproduction of the forward run r1.
- **Host evidence:** `receipts/host_test_bwd_local.json` (173/173 cases; chunked within 2.51e-7 of binary64 and
  2.82e-7 of the ordered backward) and `receipts/default_cpu_local.txt` (the CPU default identical to master's).
- **Running it:** `vm_run.sh` and `drive.py` take a protocol argument: `forward` (the default, PROTOCOL.md's gate
  set), `backward` or `prefill`.

## End-to-end prefill: result (run `p2-20261103`, one Colab L4, 3 Nov 2026)

- **Protocol:** `PROTOCOL_PREFILL.md`, fixed before the run. It measures the whole Mamba-1 model's prefill
  (`parity::MambaLM`, FP32, logits at every position), ordered against chunked scan at L = 64 and 128, at two
  shapes (`d768x24`, Mamba-130m's width and depth; `d384x12`), five cases and two GEMM backends.
- **Review:** Codex methods review, four rounds; round 2 DO NOT RUN, rounds 3 and 4 RUN (kept in the research
  workspace, not here).
- **Runs:**
  - `p1-20261103` stopped at the resources gate before checkout, on Colab's cgroup-namespace layout (about 3
    L4-minutes; `receipts/p1-20261103/`).
  - `p2-20261103`, commit `f2abc6c1`, is the revised packet's only attempt. It ran 13:29-14:04 UTC+11 with driver
    exit 0. `status.json` matched the run, commit and HEAD, and all 13 gates were 0. Cleanup was certified.
    Receipts: `receipts/p2-20261103/`.
- **Tool:** `tools/mamba_prefill_bench.cpp` (modes gate, time, profile and calibrate), with its host test
  `tests/test_mamba_prefill_bench.cpp`. Host evidence: `receipts/prefill_host_test_local.json` and
  `receipts/prefill_calibration_local.json` (the tolerance's CPU calibration).

### Gates: all pass

- **Host test**, on the VM: one chunk equals ordered bit for bit. The chunked emulation is within 7.6e-7 (logits)
  and 3.9e-7 (scan) of ordered. Both broken scans exceed the bound: dropped chunk starts by 4.9e-4 and 2.5e-3,
  carrying state across sequences by 0.14.
- **The two correctness gates**, tiled and cuBLAS: 72 of 72 checks each. They cover:
  - the CPU and device replicas;
  - CPU agreement at all three shapes;
  - repeats and one chunk, bit-identical;
  - every chunked variant differing from ordered yet within the bound;
  - the timed path matching the recorded one;
  - the malformed switch refused.
- **Worst errors** (max |a - ref| / rms(ref), over logits and every block's scan output):
  - chunked against ordered: 2.2e-5 at `d768x24` and 8.2e-6 at `d384x12` (bound 1e-3);
  - device against CPU: 2.6e-5 (bound 1e-3), and 1.7e-6 at the tiny shape (bound 1e-4).

### Prefill latency (median of 15, alternating; whole forward, host work included)

L = 64 and L = 128 are within 1.5% of each other in every cell; the table shows L = 128. "Scan op" is the profiled
`selective_scan_fwd` time summed over the layers (cuBLAS run; the tiled run agrees within 2% or 0.2 ms). "Outside device ops"
is 1 - (summed device-op time) / wall, at L = 128 in the profile.

| Shape | tokens x batch | tiled: ordered -> L128 ms | cuBLAS: ordered -> L128 ms | scan op, ms | wall saving / scan saving (tiled, cuBLAS) | outside device ops (tiled, cuBLAS) |
|---|---|---|---|---|---|---|
| `d768x24` | 512 x 1 | 318.5 -> 264.7 (1.20x) | 269.7 -> 223.1 (1.21x) | 62.6 -> 10.4 (6.0x) | 1.01, 0.89 | 72%, 89% |
| `d768x24` | 1024 x 1 | 593.6 -> 484.3 (1.23x) | 469.6 -> 357.9 (1.31x) | 135.9 -> 26.3 (5.2x) | 0.99, 1.02 | 68%, 86% |
| `d768x24` | 2048 x 1 | 1157.1 -> 924.1 (1.25x) | 908.6 -> 669.7 (1.36x) | 284.7 -> 52.6 (5.4x) | 1.00, 1.03 | 66%, 85% |
| `d768x24` | 4096 x 1 | 2262.8 -> 1785.8 (1.27x) | 1789.1 -> 1311.0 (1.36x) | 581.3 -> 106.6 (5.5x) | 1.00, 1.01 | 60%, 84% |
| `d768x24` | 2048 x 4 | 4058.9 -> 3965.6 (1.02x) | 3120.8 -> 3039.1 (1.03x) | 301.7 -> 214.5 (1.4x) | 1.03, 0.94 | 66%, 86% |
| `d384x12` | 512 x 1 | 100.8 -> 74.1 (1.36x) | 88.7 -> 60.7 (1.46x) | 29.8 -> 1.6 (18.6x) | 0.95, 0.99 | 74%, 90% |
| `d384x12` | 1024 x 1 | 173.0 -> 126.0 (1.37x) | 145.4 -> 98.1 (1.48x) | 63.2 -> 5.2 (12.3x) | 0.81, 0.81 | 69%, 88% |
| `d384x12` | 2048 x 1 | 346.7 -> 220.4 (1.57x) | 290.3 -> 165.8 (1.75x) | 135.2 -> 13.2 (10.3x) | 1.01, 1.02 | 64%, 85% |
| `d384x12` | 4096 x 1 | 763.0 -> 505.8 (1.51x) | 640.3 -> 384.9 (1.66x) | 279.2 -> 26.3 (10.6x) | 1.01, 1.01 | 67%, 87% |
| `d384x12` | 2048 x 4 | 1095.2 -> 1004.8 (1.09x) | 858.9 -> 768.2 (1.12x) | 144.5 -> 53.2 (2.7x) | 0.98, 0.99 | 61%, 87% |

- **The chunked scan's gain reaches the prefill in full.** The wall-time saving (timing run) is 0.81-1.03 times
  the scan's device-time saving (profile), and 0.94-1.03 times in 17 of the 20 cells. The prefill ratio is therefore set by the
  scan's share of the ordered prefill: 1.20-1.76x for single sequences, 1.02-1.12x at batch 4. The scan op itself
  is 5.2-6.0x faster at `d768x24` and 10-19x at `d384x12`, where the ordered kernel has half the threads.
- **Most of the prefill is outside the device ops.** With the chunked scan, 60-74% of the wall time (tiled) and
  84-90% (cuBLAS) is the residual. This mixes host work, transfers, allocation and gaps, and the profile does not
  split it further. The protocol lists the host work in this forward: `dt_proj`'s bias added on the host with its
  round trip, A = -exp(A_log) on the host in every layer, and the embedding table uploaded on every forward. cuBLAS
  removes most of the GEMM time (the head falls from 197 to 20 ms at `d768x24` x 4096), which is why its residual
  share is larger. The profiled walls are within 11% of the timing medians.
- **Memory, measured:**
  - Device in use at most 5.3 GB, and the pool's peak 2.3 GB (estimate about 5 GB).
  - Host peak RSS 8.5 GB, in the gates (estimate about 9 GB); 1.1-1.2 GB in timing and profiles.
  - Step times: build 70 s, gates 316-329 s, timing 396-510 s, profiles 169-211 s.

### What this establishes, and what it does not

- **Established:** on one L4 in FP32, paperkiln's Mamba-1 prefill with the opt-in chunked scan is
  1.20-1.36x faster at Mamba-130m's shape and 1.36-1.76x at `d384x12` for single sequences of 512-4096 tokens, and
  1.02-1.12x at batch 4 of 2048. It stays within the frozen bound, and one chunk is bit-identical to ordered.
- **Not established:** training or decode speed; other GPUs, precisions or engines; what the residual consists of.
- **Default:** unchanged and ordered; making the chunked path the default needs its own protocol.
- **Where the next prefill gain is:** the residual, not the scan. Moving `dt_proj`'s bias add and A = -exp(A_log)
  to the device, and keeping the embedding table resident, are the named candidates. Each needs its own measurement.
