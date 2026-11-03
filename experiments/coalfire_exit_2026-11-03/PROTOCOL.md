# coalfire exit check: does the CUDA build exit cleanly after the LanguageModelHead fix? Protocol

Claude Opus 5.5, 3 Nov 2026. Fixed before any GPU result. It runs only after independent review, from the reviewed
commit. One item of Jonathan's closed list (the coalfire CUDA teardown crash at exit).

## Question

C2's CUDA run `c1-20261103` wrote its logits and then died with SIGSEGV right after `~Transformer`'s body
(`experiments/c2_coalfire_cuda_2026-11-03/`). coalfire's `train.cpp` fast-exits around an "intermittent" teardown
crash. The proposed cause is in `LanguageModelHead`:
- `compute_stream` and `cublas_handle` are never assigned in `lm_head.cpp`, and the header gave them no
  initialiser;
- `~LanguageModelHead` passes each to `cudaStreamDestroy` and `cublasDestroy` when it is non-null, so it destroyed
  whatever bytes were there.

The fix, coalfire branch `claude/lmhead-cuda-member-init` at `72b1f750`, initialises both to `nullptr`. Does the CUDA
build now exit cleanly through every destructor and atexit handler, and did main crash there?

## What runs (`vm_exit.sh`)

- **Two coalfire CUDA builds:**
  - base: main `642de84c`, after #39;
  - fix: `72b1f750`, base plus the two initialisers.

  Each is built with `-DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=89` and the image's toolkit. Each gets the C2 CUDA
  leg's dump tool, built with that build's own flags.
- **The workload:** C2's after-fix checkpoint and 48 ids, uploaded and hash-checked as in the C2 CUDA leg (the same
  `inputs.sha256` and `corpus.sha256`).
- **The tool:** `coalfire_dump.cpp`, the C2 CUDA leg's, plus one switch. With `COALFIRE_DUMP_NORMAL_EXIT=1` it prints
  `{"exit": "normal"}` after writing the logits and returns from `main`, so every destructor and atexit handler
  runs. That is the teardown under test.
- **Attempts:** 10 per build, base first. Each runs in its own process, bounded by `timeout -s KILL 120`, with
  `NVIDIA_TF32_OVERRIDE=0`. Each appends one line to `exit_TAG.jsonl`:
  - the exit status;
  - whether the L4 device line, the written marker and the normal-exit marker appeared;
  - the logits' sha256, and whether it equals the C2 CUDA leg's `de52befb…`.

  `summary.json` counts them.

## Gates (all required)

1. `provenance`: exactly one NVIDIA L4; toolkit recorded.
2. `checkout`: this packet at the run's commit, and coalfire at base and at fix, each at HEAD with a clean tree.
3. `inputs`, `corpus`: as the C2 CUDA leg.
4. `configure`, `build`: both trees, unmodified by the builds.
5. `dump_build`: both tools built with their library's flags, each with `-DUSE_CUDA`.
6. `device_required`: the fix tool refuses (exit 3) with no device visible.
7. `exit_base`: all 10 base attempts ran and were recorded. Their exit statuses are the measurement, not gated.
8. `exit_fix`: all 10 fix attempts must:
   - exit 0;
   - report the L4;
   - print the written and normal-exit markers;
   - produce logits byte-identical to `de52befb…`.

## Outcomes, stated before the run

- **Fix confirmed:** `exit_fix` passes, and at least one base attempt ends non-zero after its normal-exit marker
  (a crash in teardown). That shows that, on this workload, the two initialisers are what separate a crashing exit
  from a clean one.
- **Clean exit, cause not shown:** `exit_fix` passes and every base attempt exits 0. The fix exits cleanly, but this
  run does not show that it removes a crash. The fix is still merged, since an indeterminate handle passed to
  `cublasDestroy` is a defect either way, and the record says the cause was not reproduced.
- **Fix fails:** a fix attempt crashes, hangs or differs. The teardown has another cause; it is diagnosed
  separately, and nothing is relaxed.

Out of scope:
- other static-destruction hazards in coalfire, such as `g_batched_attn` destroying its own cuBLAS handle at static
  destruction;
- `train.cpp`'s fast exit, which stays;
- other workloads, devices and training.

## Local evidence (no GPU)

- **The defect, by inspection:** `compute_stream` and `cublas_handle` are assigned nowhere in `lm_head.cpp` and are
  released in the destructor.
- **Builds:** the tool builds against a CUDA 12.6 build of coalfire, compile only. The normal-exit switch is under
  `USE_CUDA`, so CPU builds are unchanged.
- **Tests:** `test_vm_exit.py` (stub tools, 12 tests: crash, clean, hang, mismatch, marker and checkout cases, and
  stopping); `test_drive.py` (the C2 CUDA leg's driver tests, adapted, 26 tests).

## Budget and procedure

- **Command:** `python3 experiments/coalfire_exit_2026-11-03/drive.py FULL_COMMIT_SHA RUN_ID`, from a paperkiln
  checkout's root, in a foreground WSL process, after the reviewed commit is pushed. RUN_ID is `x1-YYYYMMDD`.
- **Cap and lane:** one Colab L4, session `claude-cfexit-RUN_ID`, within Claude's Colab lane. The cap is 42 minutes
  from the driver's start.
- **Deadlines:** polling ends at 25 minutes, stop attempts at 29, retrieval at 33, shutdown at 38, inventory at 42.
- **Expected time:** about 10 minutes: session and uploads 4-6, two builds about 1, 21 forwards under a minute.
- **Driver:** the C2 CUDA leg's, with this protocol's gate set and paths. Validation requires both coalfire HEADs.
  Exit codes are as there. Receipts go to `receipts/RUN_ID/`.
- **Reruns:** none to improve a result. An operational failure may be rerun once, unchanged, under a new RUN_ID.

## Runs

None yet.
