# Backlog — ordered

*What is outstanding, roughly in the order it should be done. The
long-arc vision lives in [`../../ROADMAP.md`](../../ROADMAP.md); this
file is the near-term queue. Anything finished moves to the CHANGELOG
and comes out of here.*

## 1. Finish the transfer study (the flagship)

Pre-registered at anchor `3fa55ae`. Order fixed: **bridge → S → M → L.**

- [ ] bridge gate verdict (in flight — see [`NOW.md`](NOW.md))
- [ ] arm S — d=256, 12 seeds, 6 lanes, 72 runs, ~6 T4-hours
- [ ] arm M — d=512, 12 seeds, 6 lanes, 72 runs, ~13 T4-hours
- [ ] arm L — d=1024, 3 seeds, 2 lanes, 6 runs, ~3.6 T4-hours
      (preliminary by design; carries no inference)
- [ ] run `analyze.py` exactly as committed, write `RESULTS.md`, bank
      receipts, add the registry row

**If the gate fails**, the study halts and the finding is: *the compute
backend changes the conclusion.* That slots into the existing arc — S1c
says budget changes the sign, S1e says seed changes the sign, this would
say venue does too. Three axes on which a single-number ablation claim
is under-specified. Write it up rather than treating it as a setback.

## 2. sparse_s1_longbudget — the theorem's falsifier

Pre-registered, ready, queued. 12000 steps, 10 seeds, both lanes,
~5 GPU-hours. Tests whether a SECOND crossing appears once the larger
class overfits — which would falsify assumption (iii) of
[`../../atlas/THEOREM_CROSSING.md`](../../atlas/THEOREM_CROSSING.md).
Condition and consequence are registered separately, so all four cells
of the outcome table are informative.

## 3. Extractor bugs — three, each with a named one-line fix

All in `papers/fetch.py`, all registered in
`papers/flavor_bench.py::KNOWN_WRONG` with diagnoses. The benchmark
gate still fails on any NEW wrong assertion, and reports a fixed entry
as FIXED so the list cannot rot into an excuse list.

- [ ] **Attributed adoption** (Megatron-LM). A paper stating its own
      flavor only as a property of its ancestors loses to a contrasted
      alternative in a bare declarative clause. Fix: inheritance
      outranks third-party attribution. Narrowed by the Qwen2 positive
      control — attribution *with* a first-person verb already works, so
      the failing case is syntactically well-defined.
- [ ] **Future-work mention** (Cerebras-GPT). "Worth exploring in future
      work" read as adoption. Fix: future-work mentions should VETO,
      exactly as explicit rejections already do. Second gap in the same
      paper: "X-like architecture" does not register as an inheritance
      cue.
- [ ] **Compound-name shadowing** (LaMDA). "gated-GELU" IS GeGLU; bare
      `GELU` matched inside it. Fix: longest-match-wins over the flavor
      lattice plus a `gated-X → XGLU` normalisation.

## 4. Extractor benchmark — grow past 40

40 papers, grouped AUROC 0.905, CI-band target met. Growth is
mechanical but slow by design: **every entry is read off the fetched
source, never recalled, and a field a paper does not state is omitted
rather than inferred.** Next candidates that scanned clean but were not
yet added: StarCoder, ELECTRA, UL2, BigBird, Chinchilla (several state
no flavors at all — correctly skipped).

## 4b. DEFER_DOWNLOADS crashes mtstudio (open defect, found 31 Aug 2026)

**Symptom.** With `MICROTORCH_DEFER_DOWNLOADS=1`, `mtstudio run` dies at
step 1 with `malloc(): unsorted double linked list corrupted` — SIGSEGV
on the first cell, SIGABRT on the rest. All ten bridge cells died this
way. It is the dying-temporary class for the third time: a deferred
value-cache entry outliving its host buffer, so `step_end()`'s
`materialize_all()` writes freed memory.

**Measured on the VM at the study's real shape** (d=256, T=256, L=2,
vocab 4096, batch 4), 30 steps:

| config | time | result |
|---|---|---|
| `res` — ops + residency | 20.9s | clean, loss 4.962438106536865 |
| `ops` — device ops only | 26.6s | clean, identical loss |
| `gpu` — gemm only | 38.3s | clean, identical loss |
| `b2` — full defer | 2.1s | **crash at step 1** |

**Why the 285-check suite missed it.** B2.3's validation exercised
`test_cuda_ops` and `bench_b2`. **mtstudio's own training loop has never
run under deferral** — it has its own model construction, eval, gradmap
and export paths, none of which any leg touches. Same shape of blind
spot as the tiny-test-shapes one, one level up: we validated the library
and not the application.

**Mitigation in place:** `tools/colab_transfer_runner.py` runs the study
with deferral OFF. `res` is a validated configuration converging
identically, and keeps most of the speedup.

**To fix:** find the deferred temporary in the mtstudio step path that
dies inside the window. The four existing enforcement points
(`~Variable`, consumed non-leaf grads in `backward()`, the rvalue
`accumulate`, and `ops::cached`) do not cover raw `Matrix` locals that
never pass through any of them. Then add a leg that runs a real
mtstudio-shaped model end to end under defer, because the absence of one
is what let this through.

## 4c. ~~The device op set leaks device memory~~ — FIXED 31 Aug 2026 (`697e281`)

**Root cause: C++ member initialization order.** `In` (the operand
wrapper in `src/cuda_ops.cu`) declared `float* d` before `bool owned`,
and initialized `d` in the mem-init list by calling
`vc_operand(h, n, owned)` — which sets `owned` through an out-parameter.
Members initialize in **declaration** order, so `d` was initialized
first (setting `owned = true`), and then `owned` ran its own default
member initializer and was reset to `false`. `~In` tests `owned` before
freeing, so **it never freed anything**: every operand of every device
op leaked. No `-Wreorder` warning fires, because only one member appears
in the init list. The fix assigns in the constructor body instead.

Re-measured on a T4 at the study's exact shape after the fix:

| config | before | after |
|---|---|---|
| `ops` — device op set | OOM at step 95, 156 MiB/step | **400 steps, FLAT at 173 MiB, 0.84 s/step** |
| `gpu` — gemm only | 200 steps flat, 1.10 s/step | unchanged |

The op set is now both stable and **1.31x faster** than gemm-only, so
`tools/colab_transfer_runner.py` runs the study on it again.

**The missing check now exists.** `test_cuda_ops` leg 8 runs 200
composed tapes and asserts device memory has not grown past a warmup
baseline (observed: 0.0 MiB). `device::device_bytes_in_use()` was added
for it. Note what the diagnosis cost: reading every allocation site
found nothing, because every *free* path was correct — the bug was in
the flag those frees test. The measurement (memory vs step, at two
vocab sizes) is what localised it, and the vocab-independent bulk is
what said "every operand" rather than "the logits buffers".

<details>
<summary>Original report (kept for the record)</summary>

**Symptom.** `MICROTORCH_DEVICE_OPS=1` OOMs a real training run at
**step ~95**: `mtstudio: CUDA malloc: out of memory`. Measured at the
transfer study's shape (d=256, T=256, L=2, vocab 4096, batch 4) on a
16 GB T4, 400-step probe:

| config | outcome |
|---|---|
| `gpu` — gemm only | **reached 400 steps clean** |
| `ops` — + device op set | OOM at step **95** |
| `res` — + residency | OOM at step **96** |
| `b2` — + deferral | heap corruption at step 1 (see 4b) |

Residency and deferral are not implicated in the leak; turning the op
set on is what does it. Where they survive, all four configs converge
to identical losses, so this is a resource bug and not a numerics one.

**Why nothing caught it — the third coverage hole of the same family.**
Every CUDA validation is too SHORT:

- `bench_b2` — 3 warmup + 6 timed = **9 steps**
- `test_step_residency` — **50 steps**
- `test_cuda_ops` legs — single ops and short composed tapes

The leak kills at ~95. **Not one of the 285 checks runs long enough to
reach it.** First it was test shapes too small, then an application path
(mtstudio) never exercised, now durations too short. The pattern is the
lesson: a suite that only tests small, short, library-level cases
certifies small, short, library-level correctness.

**Consequence for a banked claim — state this plainly.** The B2 adoption
gate's 21x/30.5x was measured over 9 steps with deferral on. Those
numbers are real for 9 steps, but that configuration **cannot complete a
training run**. The config the study actually runs on is gemm-only,
which is correct but slower. Phase B's speedup is measured; its
endurance is not. See the correction note in `../CUDA_PHASE_B2.md`.

**To fix:** find the per-call device allocation in `src/cuda_ops.cu`
that is not released — the `In`/`Out`/`DBuf`/`IBuf` wrappers are RAII, so
suspect a path that returns `owned=false` for a buffer nobody owns, or
a cache insert with no eviction. Then add an ENDURANCE leg: a few
hundred steps with the op set live, asserting device memory is flat.
That leg is the thing whose absence allowed this.

*(The guess was half right — it was indeed a path returning
`owned=false` for a buffer somebody owned. It just wasn't returning it;
the constructor was overwriting it afterwards.)*

</details>

## 5. CUDA — what is left

Phase B is complete and adopted (21× at d=256, 30.5× at d=512). Open:

- [ ] Wire `MT_DEVCHECK_HOST_READ` at actual call sites. The macro and
      `devcheck_host_read` exist; nothing calls them. A DEVCHECK build
      would turn a silent stale read into a loud assert.
- [ ] Coalesced loads in the transposed GEMM paths (stated non-goal
      during B2, correctness first — now fair game).

## 6. Housekeeping

- [ ] `paperkiln-fetch` and `paperkiln` are not yet claimed on PyPI. The
      wheel builds; only Jonathan can upload.
- [ ] `tools/sync_fetch_pkg.py --check` reports drift while
      `papers/fetch.py` has uncommitted changes. Expected, not a defect.
- [ ] The repo carries a large CRLF-only diff surface on Windows
      checkouts (~180 files show modified with zero real changes;
      `git diff --ignore-cr-at-eol` is empty). Cosmetic, but it makes
      `git status` useless at a glance. A `.gitattributes` pass would
      settle it.
