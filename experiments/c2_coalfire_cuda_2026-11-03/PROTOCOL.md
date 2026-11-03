# C2 CUDA leg: coalfire's CUDA build on the C2 checkpoint: protocol

Claude Opus 5.5, 3 Nov 2026. Fixed before any GPU result. Branch `claude/c2-coalfire-cuda`, cut from paperkiln
master `93bf21d2`. It runs only after independent review, from the reviewed commit.

## Question and scope

C2 (`docs/receipts/c2_coalfire_parity_20261103/`, paperkiln PR #6) pinned coalfire's CPU forward and paperkiln's
`nn::Llama` to an independent float64 forward of the same tiny llama-family checkpoint. With coalfire's FFN-bias fix
(coalfire.cpp #32), both are within 5.6e-7 of it. Its stated gap is that coalfire's CUDA path was not run. The
question here is whether coalfire's CUDA build computes the same model as its CPU build and its export: does its
forward on the same checkpoint and ids stay within C2's frozen bound of the float64 referee?

Scope: one tiny model (2 layers, hidden 128, 4 heads, FFN 512, vocabulary 2,275; multi-head attention, untied head),
48 positions, one L4, FP32. It is a correctness pin, not a speed measurement. The GQA, tied-head leg stays out of
scope: coalfire implements neither (reference §10).

## What runs

- **Model and inputs:** C2's after-fix artifacts, unchanged. These are the format-3 checkpoint, the HF export
  (`config.json`, `model.safetensors`), the 48 ids and both CPU legs' logits (`coalfire.f32`, `paperkiln.f32`). They
  are uploaded and checked against `inputs.sha256`, which is the C2 receipt's `after_fix/sha256.txt`
  (`test_drive.py` checks the two agree). The tokenizer's two vocabulary files are rebuilt on the VM from coalfire's
  `data/training_pairs.txt`, as C2's `run.sh` built them, and checked against `corpus.sha256`. That file has been
  unchanged since February 2025 (blob `fbbe6ce0` at both C2's commit and the pin).
- **Engine:** coalfire.cpp `605685e8` (main after #38), built with `-DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=89`
  with the image's own toolkit (recorded, not installed).
- **Tool:** `coalfire_dump.cpp`, C2's dump tool, with three changes:
  - it reports the build's CUDA state first: `use_cuda`, the device count, name and compute capability;
  - a CUDA build without a visible device exits 3 instead of running;
  - it sets the config's two uninitialised flags (`use_fp16`, `use_gradient_checkpointing`) to false. It is compiled with exactly the flags coalfire's build gave `src/transformer.cpp`, taken from
  `compile_commands.json` by `dump_flags.py`, because the headers change under `USE_CUDA`. It runs `forward_batch`
  in eval mode, as C2 did.
- **What runs on the device** in this build, by construction (`USE_CUDA` is compile-time; there is no runtime
  fallback):
  - the FFN (SwiGLU kernel with cuBLAS GEMMs);
  - attention (cuBLAS GEMMs, CUDA softmax);
  - the head (`cuda::matmul`, cublasSgemm).

  RMSNorm stays on the CPU (its CUDA path is disabled in the source), as do the embedding gather and the residual
  adds. The GEMMs are `cublasSgemm` in the default math mode, which is FP32, and every dump also runs with
  `NVIDIA_TF32_OVERRIDE=0`. The internals of the batched-attention and SwiGLU launchers are coalfire's. The excerpts
  reviewed show their call sites and compile-time selection; the device run is the evidence for the rest.
- **The FP16 flag.** coalfire's `TransformerConfig` gives `use_fp16` and `use_gradient_checkpointing` no
  initialiser, a latent defect. Constructing the layers reads `use_fp16`: `MultiHeadAttention` stores it, which is
  undefined behaviour while it is indeterminate, though no forward branch selects on it. Only backward paths branch
  on it. C2's CPU tool inherited the read. This tool sets both flags to false, and the CPU control was rerun with
  that change.
- **Referee:** C2's `compare.py`, imported unchanged by `compare_cuda.py`.

## Gates (all required; `vm_c2.sh`, in order; a failure stops the run)

1. `provenance`: exactly one NVIDIA L4. `nvcc --version` and the CUDA directories are recorded.
2. `checkout`: this packet at the run's commit and coalfire at `605685e8`, each verified at HEAD with a clean tree.
3. `inputs`: the uploaded set is exactly `inputs.sha256`'s names, and every hash matches.
4. `corpus`: the rebuilt vocabulary files match `corpus.sha256`.
5. `configure`, `build`: coalfire's CUDA configure and `transformer_core`. Neither tree is modified.
6. `dump_build`: the dump tool is built with the library's flags, passed one argument per line. They must include
   `-DUSE_CUDA`, and the tool is linked to the toolkit's cudart and cuBLAS (rpath to the toolkit beside `nvcc`). The
   compile database, the resolved `nvcc`, compiler and library paths (`toolchain.txt`) and the tool's `ldd` are kept.
7. `device_required`, a negative control: with `CUDA_VISIBLE_DEVICES` empty, the tool exits 3, reports
   `"devices": 0` and writes no logits.
8. `dump_cuda`, `dump_cuda_repeat`: two runs. Each must exit 0, report `"use_cuda": true, "devices": 1,
   "device": "NVIDIA L4", "cc": "8.9"`, and print `{"logits_written": true}` and `{"cuda_shutdown": "ok"}`
   (from c2 on; see Runs).
9. `numpy`: the image's NumPy, recorded (nothing is installed).
10. `compare`: `compare_cuda.py`. It must find:
    - the CUDA logits: 48 x 2,275 finite values, every row within the frozen bound of the referee
      (max |a - ref| / max(1, max |ref row|) <= 1e-4), and the referee's argmax in every row;
    - the referee, rerun on the VM, still passing both CPU legs of the C2 receipt.

**Reported, not gated:**
- the repeat's own scores, and whether its file bytes equal the first run's (the gate is on the first run only; the
  repeat must only run and report the L4);
- the CUDA logits against coalfire's CPU build and against paperkiln;
- the CUDA leg's maximum and median error.

The bound is C2's, fixed before C2's first run, and it is not changed here. The CPU legs were within 5.6e-7.

**Outcomes.**
- **A pass** establishes this, and no more: at coalfire `605685e8`, the CUDA build's eval-mode `forward_batch`, on
  C2's checkpoint and 48 ids on one L4 in FP32, is within C2's bound of the float64 referee, with its argmax in
  every row. It says nothing about training or backward parity, other attention paths, other shapes, other
  precisions or other devices.
- **A completed dump that fails the comparison** is a parity discrepancy to diagnose, with C2's sub-op method. It
  does not by itself locate the cause, and the bound is not relaxed.
- **A provenance, input, build, transport or referee-environment failure** is operational.

No rerun to improve a result: an operational failure (session lost, build failure, CLI hang) is recorded and may be
rerun once, unchanged, under a new RUN_ID.

## Local evidence (before any GPU)

- **CPU control** (`receipts/cpu_control_local.txt`): the modified dump tool, as committed, built against a CPU
  build of coalfire `605685e8`, reproduces C2's `coalfire.f32` byte for byte (sha256 `a4806858…`). This covers the
  tool's changes and coalfire's merges since C2 (#33, #37, #38) for this model.
- **CUDA compile** (no GPU): coalfire `605685e8` builds with CUDA 12.6 for sm_89 in about 2.5 minutes on 4 cores. The
  dump tool compiles with the library's flags (`-DCUDA_AVAILABLE -DUSE_CUDA ...`) and links cudart and cuBLAS.
- **Comparison:** `compare_cuda.py` on C2's own files reproduces C2's figures (coalfire 5.647e-7, paperkiln
  5.160e-7, against each other 7.86e-7).
- **Tests:**
  - `test_drive.py` (mock Colab CLI, 26 tests): the prefill driver's tests, adapted, plus the upload, its
    retries, coalfire's HEAD and the pins.
  - `test_vm_c2.py` (stub tools, 14 tests): every gate's stop and the stop handling.
  - `test_compare_cuda.py` (synthetic tiny Llama, 4 tests).
- **Review:** Codex methods review round 1, RUN with no blocking findings. Its five non-blocking points are applied:
  - the FP16 flag's description corrected, and both uninitialised flags set;
  - flags passed one per line;
  - repeat identity on bytes;
  - the toolchain evidence kept;
  - the outcome wording narrowed.

  Round 2 reviews these changes.

## Budget and procedure

- **Command.** From a paperkiln checkout's root, in a foreground WSL process, after the reviewed commit is pushed (the
  VM fetches `vm_c2.sh` at that commit):
  `python3 experiments/c2_coalfire_cuda_2026-11-03/drive.py FULL_COMMIT_SHA RUN_ID`, with RUN_ID `c1-YYYYMMDD`.
- **Cap.** One Colab L4 session, AGENTS.md path C, named `claude-c2cuda-RUN_ID`, within Claude's Colab lane (one L4
  at a time, at most 100 compute units a week). At most 42 minutes from the driver's start.
- **Expected time:** about 10-15 minutes, made up of:
  - session creation and the launch, 3-5 minutes;
  - uploading about 9.3 MB, 1-2 minutes at the home uplink;
  - two clones and coalfire's CUDA build, 3-5 minutes;
  - dumps and comparison, under a minute.
- **Deadlines** from the driver's start: polling ends at 25 minutes, workload-stop attempts at 29, retrieval at 33,
  shutdown attempts at 38, the final inventory at 42.
- **Driver.** The Mamba prefill packet's driver (run `p2-20261103`), with this protocol's gate set, deadlines, paths
  and session name, plus the Gemma packet's upload step. The local inputs are checked against `inputs.sha256` before
  any session exists, and the upload's mkdir and every file are tried up to three times. Validation also requires
  `coalfire_head` = `605685e8`. Session ownership, the reconciled launch, PID-based stop with process groups,
  retrieval before shutdown and authoritative shutdown certification are unchanged from it. Exit codes:
  - 0 validated;
  - 1 experiment failed (receipts kept);
  - 2 transport, receipt, status-identity or stop-confirmation failure;
  - 3 cleanup unresolved.
- **Outputs** come home to `/mnt/c/ml_artifacts/c2_coalfire_cuda_20261103/RUN_ID/`. The top-level files of `out/`
  are copied to `receipts/RUN_ID/` with `archive.sha256`, and then the driver log.

## Runs

- **`c1-20261103`** (commit `2a7370c5`, reviewed rounds 1-2; receipts in `receipts/c1-20261103/`): failed at
  `dump_cuda` with exit 139. It ran from 15:13:36 to 15:21:58 UTC+11 (about 8 L4-minutes); the driver exited 1 and
  cleanup was certified.
  - **What passed:** provenance, checkout, inputs, corpus, configure, build (27 s), the dump build and the
    no-device control.
  - **The failure:** the CUDA dump reported the L4, loaded the checkpoint, ran the forward and wrote all 48 x
    2,275 logits (436,800 bytes). It printed its final line, then died with SIGSEGV after "Transformer destructor
    called", in the process's CUDA teardown. `dump_cuda_repeat`, `numpy` and `compare` did not run.
  - **A known coalfire defect.** coalfire's own `src/train.cpp` documents that its component destructors
    (`CudaMatrix` `cudaFree`) and the cudart/cuBLAS atexit handlers "intermittently hit an access violation during
    the CUDA runtime's teardown". It describes this as a Windows problem, and the trainer fast-exits with
    `std::_Exit(0)` after `cuda::cleanup_loss_resources()` and `cuda::cleanup_cuda()`. c1 shows the same crash on
    Linux (L4, driver 580.82.07, CUDA 13.0.88).
  - **Seen before this revision, diagnostic only:** c1's logits file was scored at home with `compare_cuda.py`
    (`receipts/c1-20261103/diagnostic_compare_home.json`). It is within 7.0e-7 of the float64 referee (median
    4.2e-7), with the referee's argmax in every row, 7.5e-7 from coalfire's CPU logits and 8.4e-7 from
    paperkiln's. This is not a protocol pass: the gate failed, and the file came from a process that then crashed.
- **The revision for `c2-20261103`** is a reviewed operational revision following c1, not an unchanged rerun. It
  was written after c1's diagnostic score was known. It changes the process's exit only, never the model, the
  inputs, the forward, the bound or the comparison:
  - **The dump tool** closes the logits file and checks it, and prints `{"logits_written": true}`. A CUDA build
    then takes coalfire's own shutdown path from `train.cpp`: synchronise, `cuda::cleanup_loss_resources()`,
    `cuda::cleanup_cuda()`, flush, `std::_Exit`. It prints `{"cuda_shutdown": "ok"}`, or exits 4 if the
    synchronisation reports an error.
  - **`vm_c2.sh`** requires both lines in each dump's log (status 93 otherwise). It now captures the tool's `ldd`
    right after the build, and records resolved paths (`readlink -f`) in `toolchain.txt`; these are round 2's two
    non-blocking points.
  - **The teardown crash itself** stays a recorded coalfire finding outside this protocol's gate. Diagnosing it
    would need its own run, with a backtrace.
  - **Checks:** the CPU control was rerun with the revised tool, still byte-identical to C2's `coalfire.f32`; the
    revised CUDA tool compiles and links. `test_vm_c2.py` gains a test that a dump without both lines fails (14
    tests).
