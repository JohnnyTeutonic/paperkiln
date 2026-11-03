# coalfire exit check: result

Claude Opus 5.5, 3 Nov 2026. Protocol: `PROTOCOL.md`. Run `x1-20261103`, one Colab L4, packet commit `e6c238ef`
(code as reviewed in Codex round 2, RUN; the protocol's precedence text was added after it). It ran from 17:02:05 to
17:05:46 UTC+11, about 4 L4-minutes. Driver exit 0, all 10 gates 0, cleanup certified. Receipts:
`receipts/x1-20261103/`.

## Outcome: base teardown failure reproduced; fix passed 10/10

| Build | attempts | exit 0 | teardown failures | before teardown | timeouts | logits = C2's |
|---|---:|---:|---:|---:|---:|---:|
| coalfire main `642de84c` | 10 | 0 | 10 (SIGSEGV, exit 139) | 0 | 0 | 10 |
| fix `72b1f750` | 10 | 10 | 0 | 0 | 0 | 10 |

- **On main,** every attempt wrote its logits, printed the normal-exit marker and returned from `main`. It then died
  with SIGSEGV after "Transformer destructor called"; `timeout` reported a core dump.
- **With the two `LanguageModelHead` members initialised to `nullptr`,** every attempt exited 0 through the same
  destructors and atexit handlers.
- **All 20 logits files** are byte-identical to the C2 CUDA leg's (`de52befb…`). The fix changes nothing in the
  forward.

## What this does and does not establish

- **Established, on this workload and this VM:** main's CUDA build fails in teardown on every attempt here, and the
  fix build exited cleanly in all 10. That is 0/10 observed failures, which bounds the per-attempt failure
  probability below about 26% (one-sided, 95%); it is not a reliability figure.
- **Consistent with the proposed cause, not a proof of it.** The two members were never assigned and were passed,
  indeterminate, to `cudaStreamDestroy` and `cublasDestroy`. The run does not locate the SIGSEGV (no backtrace).
  The fix build also runs `g_batched_attn`'s static teardown, which here completed without a failure.
- **Not covered:** other workloads, training runs, other devices. `train.cpp` keeps its fast exit, which can now be
  revisited separately.

The fix is coalfire.cpp PR #40 (`claude/lmhead-cuda-member-init`).
