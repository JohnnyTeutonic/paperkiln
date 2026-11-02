<!-- codex_review
date: 2026-11-03 00:48
model: Codex config default
preset: -
minutes: 1.8
input-dir: C:/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad/scanreview/input4
context: C:/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad/scanreview/context
prompt: Round 4 of the review of the Colab driver and VM script for one paid L4 run of an opt-in chunked selective-scan forward. The numerical side was cleared in rounds 2 and 3. The context holds your rounds 1-3 and the repository's Colab rules. The inputs are the round-3 fixes as one patch, the current drive.py and vm_run.sh, their 18 local tests (all pass) and the protocol.  1. For R3-B1 and R3-B2, and for R3-N1 to N5, say whether each is resolved, citing file and line, or what remains. 2. Look for defects the changes introduced:    - the "new since before creation" certification rule, including a session that someone else creates during      our run;    - cleanup's exception handling and retry bound;    - the cancellation flag;    - the harness gate on test hooks;    - the tar member filter;    - validate()'s three-valued result and the exit-code mapping;    - vm_run.sh's final-status recording. 3. Verdict: RUN, or DO NOT RUN with the blocking findings. A blocking finding must describe a plausible failure    of this run that would cost money unwatched, lose its evidence or certify a wrong result. Give file, line,    scenario and fix for each finding, keeping blocking and non-blocking separate. 
-->
**Verdict: RUN.** I found no remaining blocking defect for this one paid run. The two round-3 blockers are resolved. Some reporting and hardening limitations remain, but none demonstrates a plausible false numerical pass, lost experiment receipts, or silently abandoned paid VM in the supplied run.

This is a static review of all supplied material; no commands were run. Line numbers refer to the current files, excluding patch headers. The 18 passing tests are supplied evidence, not independently executed checks.

### Round-3 findings

| Finding | Disposition | Evidence and remaining limitations |
|---|---|---|
| **R3-B1 — Unnamed allocation falsely certified gone** | **Resolved** | `drive.py:156,159–174` retains the baseline endpoints and reconciles creation. Cleanup refuses certification while any new endpoint remains (`293–304`), including an unnamed allocation whose ownership was never established. The new orphan test covers this case. |
| **R3-B2 — Logging/cleanup exceptions abort shutdown** | **Resolved for the reported failures** | Log writes and flushes have fallbacks (`77–94`); command-start `OSError` is caught (`107–117`); each cleanup attempt catches exceptions (`284–306`), with another guard in the finalizer (`337–341`). The injected log-write and command-start tests exercise the original scenarios. |
| **R3-N1 — Deadline overclaim** | **Partially resolved** | Separate 85-minute shutdown and 90-minute inventory deadlines are established at `66–67`; reconciliation sleeps are capped (`172`), and final inventory uses its own deadline (`342–344`). Extraction and copying remain synchronous without deadlines (`222–246`), and subprocess timeout still establishes direct-child termination rather than process-tree termination (`111`). These remain non-blocking qualifications for the supplied small receipts. |
| **R3-N2 — Recorded VM exit differs from aggregate exit** | **Resolved on ordinary completion; publication-failure edge remains** | `vm_run.sh:20–28` computes and records the aggregate final status separately from incoming `shell_status`; `38` exits with it. Status publication must succeed before DONE (`36–37`). Failure publishing DONE can still make the actual exit differ from the already-recorded status; detailed below. |
| **R3-N3 — Asynchronous signal exceptions** | **Substantially resolved** | The handler only sets a flag (`drive.py:368–372`). Controlled checks and polling observe it (`96–98,172–173,198–206,323–329`), and cancellation during launch/poll still leads to diagnostic retrieval. An active subprocess is allowed to finish or reach its existing timeout; cancellation does not immediately terminate it. |
| **R3-N4 — Malformed listings/status structures** | **Substantially resolved** | Listings preserve duplicate names and reject malformed bracket-prefixed rows (`126–141`). Validation handles non-object JSON and checks containers/value types (`249–263`). Remaining weaknesses are the unchecked non-bracket listing format and Python’s acceptance of booleans as integers, discussed below. |
| **R3-N5 — Ungated test overrides** | **Resolved for driver hooks; VM override remains** | All driver overrides use the harness-gated `hook()` (`34–50`); production entry rejects inherited `DRIVE_TEST_*` variables without the flag (`358–361`). `VM_ROOT` remains independently honoured by `vm_run.sh:13`. |

### Blocking findings

**None.**

An unresolved cleanup result remains an operational failure requiring attention, explicitly reported as exit **3**. That is different from the former defect of reporting successful cleanup while an unidentified allocation remained running.

### Checks of the revised controls

- **“New since before creation” certification:** Correctly conservative (`drive.py:293–304`). If someone else creates a session during this run, its new endpoint prevents cleanup certification even after our VM stops. The driver does **not** adopt or stop that unrelated endpoint. It reports exit 3. This is a false unresolved result permitted by the stated certification rule, not an ownership violation.

- **Cleanup exceptions and retry bound:** There are at most **five attempts**, additionally capped by the shutdown deadline (`284–306`). Exceptions consume an attempt; they do not escape cleanup. Adoption on the fifth attempt may have no subsequent stop attempt, but the result remains unresolved rather than falsely clean. The code does not promise to retry continuously until minute 85.

- **Cancellation flag:** It avoids asynchronous interruption of subprocess setup and cleanup. Once set, it also suppresses cleanup sleeps (`121–124,289`), so shutdown retries may occur closer together. Attempts and external calls remain bounded. Cancellation during an active call can wait up to that call’s timeout.

- **Harness gate:** Driver hooks cannot silently affect production without `DRIVE_TEST_HARNESS=1`. Explicitly enabling that flag intentionally permits them; this is a test guard, not an authentication boundary.

- **Tar member filter:** For the prescribed WSL execution, it restricts extraction to regular files/directories under the exact run output prefix and rejects traversal, links and special files (`221–230`). The destination starts fresh (`313`). No extraction escape is apparent in this execution environment.

- **Three-valued validation and exit mapping:** Missing, malformed or identity-mismatched status returns `None`; a structurally valid failed experiment returns `False`; a validated pass returns `True` (`249–277`). Exit precedence is correct: unresolved cleanup **3**, incomplete transport/receipts/validation **2**, validated success **0**, experiment failure **1** (`345`). A final log-copy failure preserves cleanup precedence through `max(code, 2)` (`347–353`).

- **VM final status:** Failed chunked gates followed by a successful benchmark now produce recorded `exit_status:1`, `pass:false`, and process exit 1 (`vm_run.sh:20–38`). The former inconsistency is fixed.

### Non-blocking findings

| File and line | Scenario and consequence | Fix |
|---|---|---|
| **`vm_run.sh:25–38`; `drive.py:327–331`** | All gates pass and status publishes successfully, but DONE publication fails. The shell exits 1 while status records 0. The driver does not require `done=True` before accepting downloaded status, so a later successful retrieval can return 0. This misreports completion bookkeeping, although the numerical gates actually passed. | Require a matching downloaded DONE marker for validated success, or otherwise make successful completion publication part of acceptance. Describe `exit_status` as the aggregate experiment status if publication failures cannot be reflected in it. |
| **`drive.py:257–258`** | `isinstance(False, int)` is true. A malformed receipt containing `false` for every gate and `exit_status` can satisfy the zero checks. The supplied VM writer emits integers, so this has no demonstrated producer in this run. | Use `type(v) is int` and `type(s.get("exit_status")) is int`; add malformed-schema tests. |
| **`drive.py:77–94,347–353`** | Logging can fail silently and later copying can preserve a truncated log while the driver returns 0. Cleanup still proceeds, and experiment receipts are checked separately, but durable cleanup evidence may be incomplete. | Track log failure and return at least 2 after cleanup; preserve the fallback record where feasible. |
| **`drive.py:133–136`** | A successful listing whose session rows have changed to a non-bracket format is ignored and can appear empty. The supplied format remains bracket-prefixed, so no present mismatch is established. | Validate a recognized listing envelope/empty-list response, or use structured output if supported. |
| **`vm_run.sh:13`; `PROTOCOL.md`, Budget and procedure** | `VM_ROOT` is still an ungated VM-side hook. Also, the protocol retains the broader “every exit path after argument validation” claim, although reuse refusal and directory-creation failure precede trap installation (`15–16,40`). | Gate `VM_ROOT` explicitly and align the protocol with the script’s narrower “once the fresh output directory is established” guarantee. |

The tests cover the original blockers well. They do not cover concurrent unrelated creation, signal timing, malformed integer booleans, archive rejection, or status/DONE publication failure; those are useful targeted additions, but their absence does not itself block this run.