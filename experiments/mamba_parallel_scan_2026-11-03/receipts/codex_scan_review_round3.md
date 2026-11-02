<!-- codex_review
date: 2026-11-03 00:43
model: Codex config default
preset: -
minutes: 2.5
input-dir: C:/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad/scanreview/input3
context: C:/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad/scanreview/context
prompt: Round 3 of the review of an opt-in chunked selective-scan forward, before one paid Colab L4 run. The context holds your round-1 and round-2 reviews (both DO NOT RUN), the ordered kernels, the frozen CUDA test and replay, the scan section of cuda_ops.cu and the repository's Colab rules. The inputs are the round-2 fixes as one patch and the current driver (now Python, drive.py), VM script, their local tests (test_drive.py against a mock Colab CLI, test_vm_run.py with stub tools; all 14 pass), the protocol, the benchmark and the host test.  1. For each round-2 blocking finding (B1-B5) and non-blocking finding (N1-N3), say whether it is resolved, citing    file and line, or what remains. 2. Look for defects the rewrite introduced. In particular:    - drive.py's deadline handling;    - its signal handling (raising from a handler, ignoring signals during cleanup);    - the creation reconciliation;    - the listing parse (`[name] endpoint ...`);    - tar extraction safety;    - the exit-code logic;    - the test-only environment overrides (could they be set by accident in a real run?);    - vm_run.sh's finish() and exit-status handling. 3. Verdict: RUN, or DO NOT RUN with the blocking findings. For each finding give file, line, a concrete failure    scenario and a fix. Keep blocking and non-blocking findings separate. Do not block on hypothetical issues    outside this run's scope. 
-->
# Verdict: DO NOT RUN

Two blockers remain: creation reconciliation can still falsely certify that an allocated VM is gone, and a logging or cleanup exception can terminate shutdown prematurely. The numerical changes do not introduce an apparent blocker for the supplied fixtures.

This is a static review of all supplied material; no files were opened or commands run. Line references refer to the supplied source, excluding patch headers.

## Round-2 findings: disposition

| Finding | Disposition | Evidence |
|---|---|---|
| **B1 — Creation ownership** | **Partially resolved; still blocking** | `drive.py:96–119` requires successful preflight, uses a run-specific name, records intent before creation, and reconciles after a returned failure or timeout. However, reconciliation only discovers the requested **name**. An allocation whose name/key is lost can remain running while cleanup certifies success; see R3-B1. |
| **B2 — False cleanup certification** | **Partially resolved; still blocking** | `drive.py:79–90,211–233` distinguishes a failed listing from an empty listing and compares known endpoints literally. But an unknown endpoint is explicitly exempted from the disappearance check. This leaves the B1 failure scenario open. |
| **B3 — Deadline enforcement** | **Substantially resolved for subprocess calls; incomplete overall** | `drive.py:51–52,63–77` establishes separate monotonic deadlines, refuses expired calls, and uses subprocess timeouts. Poll sleeps and cleanup sleeps are capped. Final inventory deliberately uses `shutdown_end + 300` (`251–255`), and extraction/copying remain synchronous without deadlines (`160–185`). See R3-N1. |
| **B4 — Receipts and exit statuses** | **Substantially resolved, with remaining inconsistencies** | `drive.py:142–208,240–265` attempts retrieval after launch failure, checks copying, validates identity and exact gate names, and returns failure for failed gates. `vm_run.sh:19–37,40–84` records actual gate statuses and aggregates failure. Its recorded `exit_status` can still disagree with its final process status; see R3-N2. Signal interruption also skips retrieval. |
| **B5 — Logging interrupts cleanup** | **Partially resolved; still blocking** | `drive.py:57–61` catches terminal write failures, eliminating the original tee/SIGPIPE dependency. File writes and flushes at `55–56` remain unguarded, including during cleanup. Exceptions from cleanup itself also escape the finalizer; see R3-B2. |
| **N1 — Invalid failure JSON** | **Resolved** | `scan_bench.cu:79–86,255–266` and `test_selective_scan_chunked.cpp:133–140,191–202` serialize nonfinite metrics as `null`, while the numerical checks still fail them. |
| **N2 — Malformed-switch evidence** | **Resolved for the unrelated-crash defect; positive-path limitation remains** | `vm_run.sh:72–78` now requires both nonzero status and the parser diagnostic. It still does not independently demonstrate that valid positive values select chunked execution through the op. The supplied parser/dispatch inspection supports that behavior; this remains non-blocking. |
| **N3 — Procedure/provenance overclaims** | **Substantially resolved** | Adoption-list preflight is checked at `drive.py:96–102`; source mutation was removed; clean-tree checks appear at `vm_run.sh:45,52`. The patch supplies the local host-test receipt. The “every exit path after argument validation” claim remains too broad because directory refusal/creation failure precede trap installation (`vm_run.sh:15–16,39`). |

## Blocking findings

### R3-B1. Unknown-endpoint creation can still leave a paid orphan while reporting cleanup success

**Files:** `drive.py:104–116,211–224`

**Failure scenario:** `colab new` allocates a VM but loses its response/session key—the orphan failure mode explicitly described by the repository rules. Subsequent listings show `[?] endpoint ...` rather than `[claude-scan-RUN_ID] endpoint ...`.

`create()` never captures that endpoint. After its three unsuccessful name searches, cleanup attempts a name-based stop. The successful listing contains no requested name, and `self.endpoint is None`, so line 223 accepts the listing as proof of shutdown. The orphan remains running.

Setting `intent` before `new` fixes the original assignment gap, but does not resolve an allocation that never becomes discoverable under its requested name.

**Fix:** Preserve the pre-creation endpoint inventory and reconcile post-creation allocations, including orphan inventory, after every creation outcome. Recover and stop an endpoint only when ownership is established. When an allocation was attempted but its outcome/endpoint cannot be established, return **cleanup unresolved / exit 3**, rather than treating name absence as proof of disappearance. Do not stop unrelated orphans.

Add a mock case where creation allocates an endpoint that subsequently appears only as `[?]`.

### R3-B2. Exceptions in logging or cleanup can still prevent shutdown

**Files:** `drive.py:53–61,211–233,243–265`

**Failure scenario:** The artifact volume becomes full during the run. A log write or flush raises `OSError`. The exception handler calls `log()` again, which raises again. The `finally` block starts cleanup, but its first `call()` logs the stop result through the same failed file. That exception aborts cleanup before verification or retries. The program exits without a reliable cleanup result.

Similarly, an exception starting a cleanup command—for example, `OSError` from `subprocess.run`—escapes the unprotected finalizer. Ignoring signals does not protect against these exceptions.

**Fix:** Make cleanup logging non-throwing, with a best-effort fallback. Separate command execution from logging so a completed stop remains usable even if recording it fails. Catch failures around each cleanup attempt, continue bounded reconciliation, and conservatively return exit 3 unless disappearance was verified. Preserve cleanup-result precedence over receipt/logging failures.

Add tests that inject file-write failure and a cleanup command-start exception.

## Non-blocking findings

### R3-N1. The deadline contract is stronger than the implementation

**Files:** `drive.py:112–116,160–185,251–255`; `PROTOCOL.md`, Budget and procedure

The ordinary external calls now have correctly capped timeouts. However:

- Final inventory can run until **95 minutes**, not 90.
- Creation reconciliation sleeps are not capped by the work deadline.
- Archive inspection/extraction and receipt copying can delay entry into shutdown.
- `subprocess.run` kills the direct timed-out child; it does not establish process-tree termination.

These are real qualifications, but the supplied small receipt archive does not justify blocking this run on an oversized or hostile archive scenario.

**Fix:** Use the shutdown deadline for final inventory, cap reconciliation sleeps, and place bounded local processing before the shutdown reserve expires. Document or implement process-group termination if the CLI/helpers spawn persistent children.

### R3-N2. VM `exit_status` is not always the script’s actual exit status

**File:** `vm_run.sh:19–24,26–37,84`

**Failure scenario:** `strict_2` fails, later diagnostics continue, and `scan_bench` succeeds. The final test at line 84 returns zero. `finish()` records `"exit_status": 0`, finds a failed gate, sets `pass:false`, and exits **1**.

The driver correctly rejects this run, so this cannot produce a false successful experiment. The receipt nevertheless contradicts its stated meaning.

Also, failure to publish `status.json` does not prevent publishing `DONE`; if all gates passed, publication failure does not necessarily change the process exit.

**Fix:** Compute the final aggregate process status before writing the receipt, and record that value. If useful, record the incoming shell status separately. Publish `DONE` only after successful status publication, and fail if either publication fails.

### R3-N3. Signal handling should request cancellation instead of raising asynchronously

**Files:** `drive.py:243–249,276–282`

A signal during the protected work phase generally reaches `finally`, and ignoring subsequent signals protects ordinary cleanup retries. However, raising `Stop` can interrupt Python operations at arbitrary points, including subprocess setup or exception logging. Another signal can also arrive while the finalizer is installing the three ignore handlers.

A signal during work bypasses diagnostic retrieval and goes directly to shutdown.

**Fix:** Have the handler set a cancellation flag. Observe it at controlled boundaries, explicitly terminate/reap an active child, and attempt bounded diagnostic retrieval before cleanup when feasible. Make cleanup entry idempotent and protected independently of handler installation.

### R3-N4. Parsing and validation should fail closed on malformed successful responses

**Files:** `drive.py:79–90,187–208`

The supplied `[name] endpoint ...` format is parsed correctly, including `[?]`. There is no demonstrated format mismatch here. But a successful response with unrecognized session rows becomes `{}`, and duplicate names—including multiple `?` rows—overwrite earlier endpoints.

Likewise, syntactically valid JSON with the wrong structure can raise rather than return a validation result.

**Fix:** Preserve session rows/endpoints independently of names, reject ambiguous listing responses, and validate status object/container/value types explicitly.

### R3-N5. Test overrides are ungated, but the supplied tests do not leak them

**Files:** `drive.py:29–35,49`; `vm_run.sh:13`; `test_drive.py:72–79`; `test_vm_run.py:75`

An inherited `DRIVE_TEST_*` variable can redirect commands/storage or change deadlines during a real invocation. `VM_ROOT` similarly redirects remote outputs away from the driver’s expected `/content` location.

However, these tests pass overrides in child-specific environment dictionaries; running them does **not** export those variables into the invoking shell. Accidental persistence is possible through manual exports or launch configuration, not caused by the shown tests.

**Fix:** Reject test overrides in the production entry point, and inject test configuration through an explicit test harness/configuration object.

## Other requested checks

The tar checks reject absolute paths, `..`, and symbolic/hard links for the specified WSL execution. They do not restrict extraction to `run_RUN_ID/out/` or reject special-file members. Restricting members to regular files/directories beneath that prefix would improve safety, but a hostile archive is outside this run’s demonstrated scope and is not an additional blocker.

The driver’s normal exit-code precedence is correct: unresolved cleanup wins, then transport/receipt failure, then experiment failure. A wrong commit returning 1 is conservative, although classifying identity/schema failures as validation failure 2 would better match the documented distinction.

The benchmark’s scratch allocation remains sufficient, and its row/summary gate logic remains sound. The JSON changes preserve temporary-string lifetime through each `printf` call. No new numerical blocker is apparent.

**Fix R3-B1 and R3-B2 before the paid run.**