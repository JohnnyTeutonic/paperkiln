# C2 CUDA leg: result

Claude Opus 5.5, 3 Nov 2026. Protocol: `PROTOCOL.md`. Run `c2-20261103`, one Colab L4 (NVIDIA L4, driver
580.82.07, CUDA 13.0.88), packet commit `4fd869a6`, coalfire `605685e8`. It ran from 15:27:55 to 15:31:10 UTC+11
(about 3.5 L4-minutes). Driver exit 0: `status.json` matched the run, the commit, HEAD and coalfire's HEAD, and all
12 gates were 0. Cleanup was certified. Receipts: `receipts/c2-20261103/`.

## Result: C2's pin holds on coalfire's CUDA build

| Leg, against the float64 referee | max rel. error | median | argmax |
|---|---:|---:|---|
| coalfire CUDA build (L4, FP32) | 7.0e-7 | 4.2e-7 | identical in all 48 rows |
| coalfire CPU build (C2 receipt) | 5.6e-7 | 3.4e-7 | identical |
| paperkiln CPU (C2 receipt) | 5.2e-7 | 3.5e-7 | identical |

- The bound is C2's frozen 1e-4 (max |a - ref| / max(1, max |ref row|)).
- CUDA against coalfire's CPU build: 7.5e-7. CUDA against paperkiln: 8.4e-7. Both identical in argmax.
- The repeat is bit-identical to the first run, and both equal c1's logits file (sha256 `de52befb…`), so coalfire's
  CUDA forward is deterministic here across two VMs.
- **Device evidence:**
  - every dump reported `NVIDIA L4`, compute capability 8.9;
  - with no device visible, the tool refused (exit 3);
  - the tool links the image's cudart 13.0.96 and cuBLAS 13.1.1.3 (`ldd.txt`, `toolchain.txt`);
  - the GEMMs are cublasSgemm in the default math mode, with `NVIDIA_TF32_OVERRIDE=0`.

**What this establishes:** at coalfire `605685e8`, the CUDA build's eval-mode `forward_batch`, on C2's after-fix
checkpoint (2 layers, hidden 128, MHA, untied head) and 48 ids, on one L4 in FP32, computes the model its export
describes. It is within C2's bound of the independent float64 forward, as its CPU build and paperkiln are. With C2,
the tiny llama-family checkpoint now agrees across coalfire CPU, coalfire CUDA and paperkiln.

**Not established:**
- training or backward parity;
- other attention paths, shapes, precisions or devices;
- GQA or a tied head (coalfire implements neither);
- that coalfire's CUDA build exits normally.

## Finding: coalfire's CUDA teardown crashes on Linux too

The first run, `c1-20261103` (about 8 L4-minutes), wrote the same logits and then died with SIGSEGV in the process's
CUDA teardown, after the model's destructor. coalfire's `train.cpp` documents this teardown as unsafe (there, on
Windows) and fast-exits around it. c1 shows a teardown crash on Linux consistent with that, without a backtrace to
establish the cause. The revision for c2 adopts the same fast exit after the logits are written and checked, and
changes nothing else (`PROTOCOL.md`, Runs).

The fix belongs in coalfire, as its own item:
- free the static CUDA pools and handles in a defined order, or stop running CUDA frees from static destructors;
- in `TransformerConfig`, give `use_fp16` and `use_gradient_checkpointing` initialisers (both are indeterminate
  today, and constructing the layers reads `use_fp16`).

## Reviews

Codex methods review, three rounds, all RUN (kept in the research workspace, `AI_ML/coalfire_c2_claude/reviews/`).
Round 3's non-blocking points are applied after the run, as text and tests only:
- the protocol's wording on what was fixed before any GPU result;
- the tool's list of changes;
- the teardown cause;
- c1's diagnostic repeat fields;
- the shutdown marker's meaning;
- separate marker tests.
