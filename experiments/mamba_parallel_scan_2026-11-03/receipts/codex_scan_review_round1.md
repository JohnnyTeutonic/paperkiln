<!-- codex_review
date: 2026-11-03 00:27
model: Codex config default
preset: -
minutes: 2.6
input-dir: C:/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad/scanreview/input
context: C:/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad/scanreview/context
prompt: Review this change before it is run on a rented GPU. It adds an opt-in chunked (parallel-in-time) forward for the selective scan (Mamba-1's state recurrence) to the paperkiln C++/CUDA trainer. The inputs are the commit, the new header, the CPU proof, the GPU benchmark, the protocol and the Colab VM and driver scripts. The context holds the existing ordered kernels, the frozen 220-case CUDA test and its replay, the scan section of cuda_ops.cu after the change, and the repository's Colab rules.  Find defects that would make the run wrong, wasteful or misleading. In particular: 1. Correctness of the chunked algorithm and its indexing. Check the state layout S[(q*sl + q + t + 1) * D*N + c*N + n],    the per-sequence zero state, the last short chunk, sl not a multiple of L, L >= sl, L = 1, batch > 1, and the    readout's dependence on all final_states having been written. Check whether the delta written by k_delta is    what the ordered backward expects, and whether the states feeding the unchanged ordered backward are complete. 2. Whether the host emulation really exercises the same code paths as the device. Name every place where host    and device differ (exp, log1p, rounding) and whether the CPU proof's claims survive that difference. 3. Whether the gates are adequate and not circular. Is the frozen 220-case gate meaningful for the chunked path    with L = 2 and 3? Is the 3e-5 bound appropriate at sl = 4096? Does scan_bench check enough of the states to    support "all states written and finite" and the error claim? (It checks y, the last state of each sequence    and finiteness of all states.) 4. The timing method in scan_bench (what is and is not timed, the warm-up, events on the default stream, chunk    buffers allocated outside the timed region) and whether the protocol's reporting avoids overclaiming. 5. The Colab scripts against the repository's Colab rules: the session stopped on every exit path, timeouts,    polling, outputs brought home, the DONE marker on failure, and any way the VM could run the wrong commit. 6. Anything in the env-variable switch (MICROTORCH_SCAN_CHUNK) that could silently select a path, or leave the    default path changed.  Give a verdict: RUN, or DO NOT RUN with the blocking findings. List each finding with file and line, a concrete failure scenario, and a fix. Separate blocking from non-blocking findings. Do not restate the design. 
-->
# Verdict: DO NOT RUN

The indexing and launch ordering look correct for ordinary, valid dimensions. The blockers are in the run controls, commit pinning, environment parsing, and validation’s ability to report success reliably.

Line references below use the supplied source files and repository paths.

## Blocking findings

### 1. The driver does not enforce the deadline or establish that the session stopped

**Files:** `experiments/mamba_parallel_scan_2026-11-03/drive.sh:13–14,34–49`; `PROTOCOL.md:48`

**Failure scenario:** The 85-minute timer starts only after creation, endpoint lookup, and launch. Those operations can already consume roughly 15 minutes. A polling iteration can then overrun the deadline, followed by up to 15 minutes for packing and downloading. This is not a 90-minute deadline from creation.

The exit trap attempts one name-based stop, suppresses its result, and prints “session … stopped” even if the command times out or fails. If the session has become an orphan, the name may no longer identify the live VM. That can leave paid compute running while falsely reporting cleanup.

**Fix:** Start an absolute deadline before `colab new`, reserve time for retrieval and shutdown, and cap each operation by the remaining budget. In cleanup, check the stop result and verify the endpoint disappeared; recover/adopt the known endpoint if necessary. Report unresolved cleanup explicitly rather than printing unconditional success. Add explicit signal handling as well.

### 2. Required Colab preflight and durable result handling are missing

**Files:** `drive.sh:7–14,25,44–49`; repository rules §§2, 6, 7

**Failure scenario:**

- The fixed name `claude-scan` is used without checking existing sessions or orphans. A failed creation followed by the trap could attempt to stop a pre-existing session with that name.
- Launch failure exits before attempting to retrieve available diagnostics.
- Packing or download failure is ignored; the script still announces a results directory.
- `OUT` can point into temporary storage or OneDrive, and receipts are never copied into the experiment.
- There is no final session/orphan inventory.

These omissions violate the supplied Colab rules and can lose the only evidence from a paid run.

**Fix:** Perform the prescribed preflight before creation, check ownership/name collisions, validate a durable output destination, and retain a local driver log. Make diagnostic retrieval part of cleanup where possible; check packing, download, and extraction results. Copy receipts before analysis and perform the required final inventory. Run the driver in a foreground WSL process whose lifetime is maintained.

### 3. Completion is confused with success, and the CPU prerequisite is never executed

**Files:** `vm_run.sh:8,20–34`; `drive.sh:25–43`; `PROTOCOL.md:19–36`

**Failure scenario:** A strict test or replay fails, but execution continues through all later gates and timings. The final successful `sha256sum` can make the VM script exit successfully. `DONE` appears on both success and failure, and the driver checks only its existence. Several failures never create `FAILED`.

Also, the VM builds only `test_selective_scan_cuda`; it neither builds nor runs `test_selective_scan_chunked`. No supplied driver verifies a previously recorded CPU pass.

**Fix:** Run and preserve the CPU gate before renting the GPU, or build and run it explicitly on the VM. Capture each gate’s status and produce an aggregate machine-readable result with a nonzero exit on failure. Keep `DONE` as a completion marker, but write the final status before it and have the driver inspect that status. Stop further paid work on a prerequisite failure, unless diagnostic continuation is explicitly intended and labelled.

### 4. The requested source is not securely pinned, and stale directories can produce false completion

**Files:** `drive.sh:7,19–21,25`; `vm_run.sh:7,14–16`

**Failure scenario:** `COMMIT` is not restricted to an immutable full SHA. Passing a branch name allows the raw-script fetch and subsequent checkout to resolve at different commits if the branch moves.

On a reused VM, `git clone` can fail because `/content/src` already exists, yet the script proceeds. Checkout can preserve local modifications, while `commit.txt` records only HEAD. An old `/content/scan_out/DONE` can cause immediate completion detection and retrieval of stale or partial results.

**Fix:** Require a full commit SHA, use it for both fetch and detached checkout, and verify HEAD equals it. Fail on clone errors and require a clean source tree. Use fresh, unique source/build/output directories and a run identifier in the completion record. Validate that identifier and commit after download.

### 5. The environment parser accepts invalid counts that can break geometry

**Files:** `src/cuda_ops.cu:2127–2133`; `src/selective_scan_chunked.cuh:65–66,76–79`

**Failure scenario:** `strtoull` accepts leading whitespace and a sign, and overflow is not checked. For example, `" -1"` bypasses `v[0] == '-'` and can yield `ULLONG_MAX`; an overflowing decimal string can yield the same value.

On a 64-bit host, with `sl=9` and `L=SIZE_MAX`, `(sl + L - 1) / L` wraps and yields zero chunks. The path then launches zero-sized work and cannot produce valid states. Even a valid, extremely large positive `L` can trigger this overflow, contrary to the advertised `L >= sl` behavior. Whitespace-only input also silently selects the ordered path.

**Fix:** Require decimal digits only, check conversion overflow and `size_t` range, and document empty-string behavior. Compute ceiling division without addition overflow, for example `1 + (sl - 1) / L` after validating positive operands. Clamp `L` to `sl` before chunk arithmetic and check allocation-size products.

### 6. The benchmark can accept nonfinite readout output

**Files:** `tools/scan_bench.cu:90–95,182–191`

**Failure scenario:** `reference_errors` updates errors with `std::max(current, error)`. When `error` is NaN, this can retain `current`, potentially leaving `ey == 0`. The explicit finiteness check covers states, not `y`. A faulty readout producing repeatable NaNs with finite states can therefore pass both the error gate and bit-repeat check.

The ordered output has the same NaN-blind error calculation and no explicit finiteness gate.

**Fix:** Reject nonfinite outputs and references explicitly, or map every nonfinite error to infinity, as the CPU test already does. Check all ordered states too. Make each row’s `ok` include ordered correctness, rather than reserving that failure for the final summary.

## Non-blocking findings and limits

### 7. Long-sequence state accuracy is not established

**Files:** `tools/scan_bench.cu:70–96,183–191`; `PROTOCOL.md:8–15,29–32`

The NaN prefill followed by finiteness testing **does support “all allocated state slots were written and finite”** for the first chunked invocation. It does not establish that every state is accurate.

Only the last state of each sequence is compared directly with binary64. Intermediate state errors can cancel in the readout’s sum over `n`, and later decay can erase them before the final state. `chunked_vs_ordered_states` is diagnostic only: it is neither independently referenced nor gated.

**Fix:** Compare every state with the binary64 state as the reference loop advances, including initial zeros. This needs no full double state-history allocation. Otherwise restrict the claim to readouts, final states, and state finiteness.

### 8. The CPU test establishes host behavior, not device numerical identity

**Files:** `src/selective_scan_chunked.cuh:30–58,195–209`; `tests/test_selective_scan_chunked.cpp:85–112,160–165`; `PROTOCOL.md:14–15,19`

The host/device differences are:

| Operation or execution | Host | Device |
|---|---|---|
| Recurrence exponential, in local and final scans | `std::exp(float)` | CUDA `expf` |
| Softplus exponential | `std::exp(float)` | CUDA `expf` |
| Softplus logarithm | `std::log1p(float)` | CUDA `log1pf` |
| Multiply/add rounding | Native operations stored through `volatile float` | Explicit `__fmul_rn` / `__fadd_rn` |
| Execution | Sequential phase loops | Grid-stride kernels with concurrent work items |

The host float overloads are not binary64 exponentials. Their library results need not match CUDA’s. The volatile operations prevent the intended multiply/add contraction, but do not independently specify the host floating-point environment.

The binary64 oracle is independent; its comparisons are meaningful evidence for those host fixtures. The one-chunk comparator shares the header’s arithmetic helpers, so its identity check establishes host operation-order equivalence, not actual CUDA bit identity. Repeat identity likewise does not prove device determinism.

**Fix:** Call this a host test rather than a numerical proof. Add actual device one-chunk identity and repeat checks.

Coverage also omits multi-row `L=1`, tests the identity claim at `L=sl` rather than strictly greater values, and hardcodes `"repeat_bit_identical": true` even when repetition caused failure. Add those cases and report the measured repeat result.

### 9. The frozen gate is useful but does not certify long sequences or independently prove path selection

**Files:** `vm_run.sh:23–28`; `context/test_selective_scan_cuda.cpp:run/main`; `PROTOCOL.md:25–28`

The gate is meaningful:

- At `L=2`, lengths 3, 4, and 9 exercise multiple chunks and short tails.
- At `L=3`, lengths 4 and 9 exercise multiple chunks.
- It exercises batching, softplus thresholds, requested gradients, and host/device handoffs.
- The independent replay remains active under `python3 -O`: it uses `need`, not assertions.

However, most mask cases have `sl=2`, so they use one chunk at both settings. Profiling proves a CUDA forward ran, not which forward implementation ran. A broken switch that always selected ordered execution could pass.

**Fix:** Preserve the frozen gate unchanged, but add separate path-selection evidence and supplementary chunked device edge cases. Do not describe all 220 cases as multi-chunk coverage.

### 10. The fixed bound is a test criterion, not a general 4096-step guarantee

**Files:** `PROTOCOL.md:8–23`; `tests/test_selective_scan_chunked.cpp:139–150`; `tools/scan_bench.cu:60–65,158–160`

There is no basis here to declare `3e-5` inherently inappropriate at `sl=4096`; retaining it as a preregistered pass/fail threshold is reasonable. There is also no error analysis proving it for arbitrary 4096-step inputs. Long-memory and cancellation-sensitive inputs can accumulate numerical error.

The new tests use `max(1, |ref|)`, whereas the frozen CUDA gate/replay use `1 + |ref|`. The new denominator is stricter, by up to a factor of two, so “the same frozen bound” is imprecise.

**Fix:** State the exact metric and fixture-limited scope. Preserve failures rather than relaxing the threshold after observing results.

### 11. Timing is valid for the stated component, with reporting qualifications

**Files:** `tools/scan_bench.cu:135–151,172–176,193`; `PROTOCOL.md:38–44`

Events and kernels use the same default stream, so the timing brackets the ordered kernel or all five chunked kernels, including delta generation. Three warm-ups are followed by synchronization. Transfers, allocation, validation, and event creation are outside the measured interval.

For the multi-kernel path, event time can include GPU idle gaps between host submissions. It is therefore elapsed stream time, not a sum of isolated kernel execution durations.

The production path allocates three chunk buffers per forward; the benchmark excludes those allocations. That is consistent with the component-only protocol, which appropriately avoids training/model speed claims.

**Fix:** Preserve that qualification in results. Optionally interleave timing order to reduce clock/order bias, report dispersion, and validate `reps > 0`; zero repetitions currently index an empty vector. For 20 samples, the implementation reports the upper middle sample rather than the average of the two middle samples.

## Correctness checks with no defect found

For positive, non-overflowing geometry:

- State writes and readout agree on  
  `S[(q*sl + q + t + 1)*D*N + c*N + n]`.
- Each sequence has its own zero state, written by chunk zero.
- Ceiling chunking and the clipped endpoint handle short tails and `sl % L != 0`.
- `L >= sl` gives one chunk; `L=1` has valid indexing, though its numerical coverage is incomplete.
- Batch indexing keeps sequences separate.
- Separate, same-stream launches guarantee `k_final` completes before `k_readout` begins; no block-level synchronization is missing.
- `k_delta` writes the transformed delta expected by ordered backward, with the same threshold and CUDA expression.
- All state slots needed by unchanged backward are produced. Their values are approximate chunked states, so gradient agreement remains an empirical contract.
- With the variable unset or exactly `"0"`, the original ordered launch remains selected. No default-path numerical change is apparent.