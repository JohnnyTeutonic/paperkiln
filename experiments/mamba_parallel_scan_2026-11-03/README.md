# Chunked (parallel-in-time) selective-scan forward: result

Claude Opus 5.5, 3 Nov 2026. Protocol: `PROTOCOL.md`, fixed before the run.
- **Review:** Codex methods review, four rounds; rounds 1-3 DO NOT RUN, round 4 RUN. All four in `receipts/`.
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
- **Not established:** model-level or training speed (these are component timings); the backward (it stays
  ordered and consumes the chunked states); other GPUs.
- **Default:** unchanged and ordered. Making the chunked path the default, or claiming model speed, needs its own
  protocol (PROTOCOL.md).
- **Open follow-ups:** a chunked backward; an end-to-end Mamba prefill measurement; choosing L automatically
  (128-256 was best or near best at every shape here).
- **Driver note:** the Colab CLI prints `[colab]` banner rows in `sessions`. The driver read them as session
  rows, which was harmless here, since they were in the pre-creation snapshot and constant. A later driver should
  skip `[colab]` rows.
