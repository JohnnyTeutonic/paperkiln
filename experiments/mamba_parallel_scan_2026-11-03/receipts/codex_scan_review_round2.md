<!-- codex_review
date: 2026-11-03 00:34
model: Codex config default
preset: -
minutes: 2.7
input-dir: C:/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad/scanreview/input2
context: C:/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad/scanreview/context
prompt: Round 2 of the review of an opt-in chunked selective-scan forward, before it runs on a rented GPU. The context holds your round-1 review (DO NOT RUN, six blocking findings) and the ordered kernels, the frozen CUDA test, its replay, the scan section of cuda_ops.cu now, and the repository's Colab rules. The inputs are the fixes as one patch and the current versions of every changed file.  1. For each round-1 finding (1-11), say whether the fix resolves it, citing file and line, or what remains. 2. Look for defects the fixes introduced. In particular:    - vm_run.sh's gate bookkeeping: an associative array, `fail`, the `finish` trap and the 13-gate count;    - drive.sh's deadline arithmetic (`cap`, `left`), cleanup on every exit path, the `exec > >(tee ...)`      redirection under a trap, and the retrieval checks;    - scan_bench's buffer sizing for L in {32, 64, 128, 256}, sl and sl + 5, and whether every printed row and      the summary can be trusted;    - scan_chunk()'s parser and the clamped geometry. 3. Give a verdict: RUN, or DO NOT RUN with the blocking findings. For each finding give file, line, a concrete    failure scenario and a fix. Keep blocking and non-blocking findings separate. 
-->
# Verdict: DO NOT RUN

The scan changes address the main numerical-validation defects. The benchmark’s revised buffer allocation is sufficient for every requested length. The remaining blockers concern ownership and shutdown of paid sessions, deadline enforcement, and reliable reporting of failed runs.

This is a static review of the supplied material only; no commands were run. Line references below refer to the current supplied files.

## Blocking findings

### B1. Session creation can leave an unowned, running VM

**File:** `drive.sh:63–65` (`colab new`, followed by `CREATED=1`); cleanup at `drive.sh:37`

**Failure scenario:** Colab allocates the VM, but the CLI times out or loses its response. The creation command returns nonzero, so the driver exits before setting `CREATED=1`. Cleanup skips stopping or recovering the newly allocated session. A signal between successful creation and the assignment has the same ownership gap.

The preflight also treats a failed session-list command as “no name collision”: without `pipefail`, the pipeline reports only whether `grep` found a match.

**Fix:** Require a successful, captured preflight listing before proceeding. Record creation intent before calling `new`, and reconcile the session inventory after every creation outcome, including timeout and signals. Use a unique run-specific name and retain endpoint ownership evidence so reconciliation cannot stop someone else’s session.

### B2. Cleanup can falsely certify shutdown

**File:** `drive.sh:39–47`, especially the negated `sessions | grep` pipeline at line 42

**Failure scenario:** `colab stop` fails, and `colab sessions` then times out, fails authentication, or returns an error with no matching session text. `grep` returns nonzero; the negation succeeds; cleanup prints `CLEANUP OK` while the GPU may still be running.

Endpoint matching also uses a regular expression rather than an exact, parsed endpoint match.

**Fix:** Capture the listing and its exit status separately. Only a successful authoritative listing that excludes the owned name and endpoint can establish disappearance. Treat listing errors as unknown, retry recovery, and ultimately report unresolved cleanup with a nonzero driver result. Match endpoints literally or parse structured output.

### B3. The advertised absolute deadline is still unenforced

**File:** `drive.sh:26–28,30–51,83–99`

**Failure scenario:** A polling iteration begins immediately before `POLL_END`. Its sleep, refresh, listing, adoption, and poll can consume roughly 13 minutes, leaving almost none of the nominal 15-minute reserve. After the deadline, `left()` returns **1**, permitting every subsequent operation to start.

Cleanup then uses uncapped timeouts. Three complete stop/list/adopt attempts can consume approximately 21½ minutes, followed by up to seven minutes of inventory. Local extraction and copying have no deadline either. Moreover, plain `timeout` does not impose a final kill deadline on a child that ignores SIGTERM.

**Fix:** Use separate work, retrieval, and shutdown deadlines. Recheck the applicable deadline before each operation; return zero/expired rather than a synthetic remaining second. Reserve shutdown time before retrieval starts. Cap cleanup and inventory too, and use a bounded escalation such as `timeout --kill-after=...`. Report explicitly if shutdown cannot be established within budget.

The `cap()` arithmetic itself is valid; its use of the wrong deadline, the expired-time behavior, and the uncapped paths are the defects.

### B4. Failure receipts and exit statuses still cannot reliably establish success

**Files:** `drive.sh:69–71,102–114`; `vm_run.sh:12–28,66–68`

**Failure scenarios:**

- A launch error immediately exits without attempting diagnostic retrieval, although the remote process may already have started.
- Missing or unreadable `status.json` explicitly exits the checker with **0**.
- A run/commit mismatch only prints a warning.
- `pass:false`, missing gates, and failed gates do not make the driver fail.
- Failed receipt creation/copying does not change `ok`.
- A failed benchmark can be followed by successful `sha256sum`, leaving `vm_run.sh` with a successful process exit.
- `finish()` does not incorporate the original shell exit status. A later unexpected failure can coexist with thirteen zero gate entries.

Thus “retrieval ok,” process success, and successful experiment validation remain different things without enforced distinctions.

**Fix:** Separate transport success, receipt-copy success, and experiment success. Validate the status schema, exact required gate names and statuses, run/commit/HEAD, and completion marker; reject missing or inconsistent data. Preserve valid failed-run receipts, but return a nonzero result for a failed experiment. Capture the VM’s original exit status and combine it with gate validation. Attempt bounded diagnostic retrieval after launch failure as well as normal polling completion.

### B5. Logging failure can interrupt cleanup

**File:** `drive.sh:23`, signal handler at `drive.sh:55`, cleanup output at `drive.sh:46–50`

**Failure scenario:** `tee` exits because its output destination fails or its downstream terminal closes. The driver’s stdout remains a pipe with no reader. A later shell write can receive SIGPIPE. Cleanup itself writes through that same failed pipe, so it can terminate before completing retries or final inventory. The signal handler also writes before requesting exit.

Neither the process-substitution setup nor the trap supervises `tee` or provides an independent cleanup logging channel.

**Fix:** Give cleanup a reliable output descriptor independent of the tee pipe, preserve and handle SIGPIPE, and make the signal handler request cleanup without first depending on that pipe. Supervise or explicitly wait for the logger, and propagate logging failure. Copy the final driver log only after cleanup and logger flushing; the current receipt copy omits the subsequent cleanup evidence.

## Round-1 findings: disposition

| # | Status | Evidence and remaining work |
|---|---|---|
| **1 — Deadline and shutdown** | **Partially resolved; still blocking** | `drive.sh:25–28` starts an absolute clock before creation, and lines 35–55 add cleanup and signal handling. B1–B3 and B5 prevent the promised guarantee. |
| **2 — Preflight and durable results** | **Partially resolved; still blocking** | Lines 19–23 choose durable per-run storage and logging; lines 30–34 provide inventories; lines 94–105 check archive operations and copy receipts. Missing launch-failure retrieval and unenforced receipt/status checks remain under B4. The prescribed `colab_adopt.py --list` preflight is also absent. |
| **3 — Completion versus success; host prerequisite** | **Partially resolved; still blocking** | `vm_run.sh:44–49` builds and runs the host test. Lines 53–60 stop after a failed ordered control/replay. Lines 12–28 write aggregate status before DONE. Final process status and driver validation remain defective under B4. No supplied receipt establishes the protocol’s separate local host-test prerequisite. |
| **4 — Commit pinning and stale runs** | **Substantially resolved** | `drive.sh:13–14` validates SHA/run syntax; `vm_run.sh:29–39` refuses reuse, checks clone/checkout, verifies HEAD and cleanliness. Driver polling checks the run-specific marker at lines 74–76. Downloaded identity validation remains unenforced under B4. |
| **5 — Parser and geometry** | **Resolved for the supplied CUDA target and valid geometry** | `cuda_ops.cu:2128–2140` accepts digits only and rejects values above \(2^{31}\). `selective_scan_chunked.cuh:68–71` clamps L and uses subtraction-based ceiling division. See the geometry discussion below. |
| **6 — NaN-blind benchmark acceptance** | **Resolved** | `scan_bench.cu:79–82` maps nonfinite errors to infinity. Lines 90–106 check readouts and every state; line 247 includes ordered correctness in each row. Failed rows can still contain invalid JSON numbers; see N1. |
| **7 — Intermediate-state accuracy** | **Resolved** | `scan_bench.cu:92–103` independently checks initial zeros and every subsequent state against the binary64 recurrence. |
| **8 — Host/device claims and coverage** | **Resolved with a reporting qualification** | Header lines 15–17 and test lines 1–8 correctly limit the host claim. Test lines 142–153 add multi-row L=1 coverage; lines 174–183 check three one-chunk lengths; lines 189–193 report measured repeats. Benchmark lines 220–228 add actual device identity checks, and lines 237–242 check device repeats. |
| **9 — Frozen coverage and path selection** | **Partially resolved; non-blocking limitation** | `PROTOCOL.md:31–39` accurately describes tiny-shape coverage. `vm_run.sh:62–63` adds malformed-value rejection, but accepting any nonzero exit does not prove the parser caused it or that positive values select chunked execution. See N2. |
| **10 — Bound and scope** | **Resolved** | `PROTOCOL.md:23–25,48–50` states the precise metric, its relationship to the frozen metric, fixture-limited scope, and preservation of failures. |
| **11 — Timing qualifications** | **Resolved** | `scan_bench.cu:151–184` alternates order, warms both paths, validates positive repetition counts, and computes the conventional even-sample median. `PROTOCOL.md:54–58` discloses stream gaps and excluded allocation cost. Dispersion remains optional. |

## Specific checks requested

### `vm_run.sh`: associative array, `fail`, trap, and gate count

There is **no apparent associative-array indexing defect** in `G[$1]=${2:-1}` or `G[strict_$L]=$?`: `declare -A G` makes these string keys.

The successful path has exactly **13 entries**:

- checkout, configure, build, host_test, numpy: **5**
- strict and replay for unset, 2, and 3: **6**
- malformed-switch rejection: **1**
- scan_bench: **1**

Successful `clone` and `bench_build` are not separately recorded. Their failure entries are nonzero, so a failed benchmark build reaching thirteen entries does **not** produce a pass.

However:

- `fail` records a generic **1**, not the failed command’s actual exit status, despite the protocol’s promise of exit statuses.
- Counting entries is weaker than validating the exact required key set.
- `finish` needs the original exit status and checked, preferably atomic status/marker writes.
- The trap is installed only after argument validation, reuse refusal, and directory creation. “Every exit path” is therefore overstated.
- `RUN` is not independently validated on the VM; the supplied driver validates it, but direct invocation can introduce unsafe paths and malformed JSON.
- Initializing the array explicitly with `declare -A G=()` avoids empty/unset-array issues under `set -u` on Bash versions where empty-array length expansion is problematic.

### `scan_bench`: buffers and row trust

**The allocation fix is sufficient.** At `scan_bench.cu:205–207`, each scratch buffer holds

\[
\text{batch}\times\left\lceil sl/32\right\rceil\times D\times N
\]

floats. For all supplied shapes, L=32 requires the most entries. L=64, 128, and 256 require no more; clamped L=sl and sl+5 each require exactly `batch*D*N`.

For `sl=64`, the chunk counts are **2, 1, 1, 1**, and both identity lengths have one chunk. There is no undersized-buffer or overlapping-live-allocation defect here.

Every completed row’s `ok` combines ordered accuracy, both one-chunk identity checks, repeat identity, and chunked readout/all-state accuracy. The summary accumulates all 24 rows. CUDA failures exit before a success summary. The numeric gate logic is therefore sound, subject to the failed-row JSON issue below and the driver actually enforcing the result.

### Parser and clamped geometry

On the supplied 64-bit CUDA platform, the parser’s largest multiplication before rejection is about \(2^{31}\times10+9\), safely below `size_t` overflow. Signs, whitespace, and nondigits are rejected; empty/unset and all-zero decimal strings select ordered execution. Exactly \(2^{31}\) is accepted and then clamped.

For valid `sl>=1`, clamping makes L=1, L=sl, and L>sl well-defined and removes the original ceiling-division overflow. `geom()` still relies on its documented positive-sl precondition. General allocation-product overflow checks remain absent, but I found no new geometry defect affecting the supplied fixtures.

## Non-blocking findings

### N1. Failed numerical rows are not valid JSON

**File:** `scan_bench.cu:79–82,253–262`; similarly the host test’s failure/summary printing

**Scenario:** A NaN output correctly becomes `INFINITY`, fails the gate, and is printed with `%g` as `inf`. JSON does not permit that token, so downstream JSONL parsing can lose the failure evidence.

**Fix:** Print `null` or a quoted nonfinite label plus an explicit failure field. Preserve finite numbers as numbers.

### N2. The malformed-switch gate can pass for an unrelated crash

**File:** `vm_run.sh:62–63`

**Scenario:** The malformed invocation fails because of a CUDA/runtime error rather than parser rejection; the gate records success. A future switch regression that rejects malformed values but routes all positive values to ordered execution would also escape this check.

**Fix:** Require the expected parser diagnostic or a dedicated exception result, and add explicit positive-path evidence without altering the frozen test.

### N3. Some procedure and provenance claims remain stronger than the implementation

**Files:** `drive.sh:30–34,60–61`; `vm_run.sh:39–40`; `PROTOCOL.md:66–74`

The required adoption-list preflight is missing. The clean-tree check precedes the `sed` mutation of tracked shell files, so it does not establish that the subsequently used tree remains clean. Neither issue demonstrates wrong scan numerics in these inputs.

**Fix:** Add and check the full preflight. Normalize only files that require it and record/recheck any resulting source differences. Preserve the local host-test receipt before renting the GPU.

**The run should remain blocked until B1–B5 are fixed.**