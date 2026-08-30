# CUDA Phase B2 — training-step residency (design)

*Status: B2.0 VALIDATED on Colab T4, 13 Aug 2026 — full
gradcheck/nn/lora/resident-parity suites plus test_step_residency all
green: the CUDA training pin matched the CPU reference to max weight
diff 2.33e-07 over 8 AdamW steps (bound: 1e-4), and the staleness
probe confirmed the epoch-scope contract on hardware (a host poke
between windows is seen by the device). B1 was validated 12 Aug — see
CUDA_PHASE_B.md. B2.1-B2.3 below remain open; B2 is the actual Rung C
unlock: the training step's bytes stay on the device; the host sees
loss scalars and explicit materializations (checkpoints, eval samples)
only.*

## Why B1 cannot be stretched into B2

Three structural reasons, found in the seam, not assumed:

1. **Params mutate every step.** A B1 table refresh after each Adam
   step is a full H2D upload of every parameter — the traffic B2
   exists to remove.
2. **Backward creates untrackable operands.** The matmul grad closures
   call `Matrix::transpose()` on the host
   (`ops.cpp`: `device::matmul(self->grad, b->data.transpose())`).
   Each call constructs a *fresh host Matrix*, so a pointer-keyed
   table can never see it twice. No cache policy fixes this; the GEMM
   itself must accept transpose flags.
3. **Host-pointer keying is unsafe for activations.** Params live for
   the process; activations die with the tape, and freed host memory
   is recycled *within a step*. A pointer+dims table hit on a recycled
   address is silently wrong. B1's keying is correct for B1 precisely
   because residency there is explicit and parameter-only.

## Design

### 1. Device state is owned by the Variable, not a side table

`Variable` (our tape, `autograd.hpp` — the vendored Matrix stays
untouched) gains one opaque member:

```cpp
DevState* dev = nullptr;   // POD forward-declared; null in CPU builds
```

`DevState` = `{float* data; float* grad; bool data_dev_valid,
data_host_valid, grad_dev_valid, grad_host_valid;}`. The Variable
destructor releases the buffers through a function pointer registered
at CUDA init (no CUDA includes in the header; CPU builds carry a null
pointer and zero overhead). Ownership by the Variable kills the
aliasing class by construction: the buffer dies exactly when the
tensor dies, and no recycled host address can inherit stale state.

### 2. Validity flags maintained only at mutation sites

The contract that keeps the no-silent-staleness rule: during a
training step, the ONLY mutators of Var data/grad are tape ops,
`accumulate()`, `zero_grad()`, and the optimizer — each updates the
flags as it writes. Everything outside the step (checkpoint save,
studio event emission, sampling) calls `device::materialize(v)` /
`materialize_grad(v)` first. Debug builds (`MICROTORCH_DEVCHECK`)
assert `host_valid` on host reads, so a missed materialize is a loud
assert, not a wrong number.

Master switch: `device::set_step_residency(bool)`, env
`MICROTORCH_STEP_RESIDENCY=1`. Off = today's B1/Phase A behaviour,
bit-for-bit.

### 3. GEMM with transpose flags

`gemm(A, opA, B, opB, C)` — NN/NT/TN as index-math variants of the
audited 32x32 tiled kernel (no materialized transposes on either
side of the seam). This deletes the host `.transpose()` temporaries
from matmul backward and both attention nodes' backwards.

### 4. The on-device op set (self-contained, canonical formulas)

Same zero-dependency rule as B1: our own kernels in `src/cuda_*.cu`,
no cuBLAS, one canonical entry point per op (the Phase 0 audit's
duplicate-kernel landmine stays behind the seam). Needed by the
parity/flex training step:

- elementwise: add, sub, mul, scale, axpy (grad accumulate), fill
- sigmoid fwd/bwd (highway gates), GELU fwd/bwd — using the CORRECT
  derivative (the vendored CPU formula is the audited ~13%-error bug;
  our CPU path and the vendored CUDA kernel agree, and B2 matches
  them)
- softmax rowwise fwd/bwd (`dS = S .* (dP − rowsum(dP .* S))`)
- layernorm fwd/bwd, rmsnorm fwd/bwd
- embedding gather fwd / scatter-add bwd
- cross-entropy fwd (downloads ONE scalar) + fused grad
- AdamW update (m, v resident; bias correction + decoupled decay,
  matching `nn.cpp` exactly), SGD+momentum likewise

Attention stays COMPOSED (matmul + softmax + elementwise), per the
Phase 0 verdict — the fused vendored kernels are shape-bound and stay
behind. The per-head row slices in the fused tape nodes are offset
pointers into contiguous row-major buffers, so composition works on
device without copies; that audit is part of B2.1, stated here so it
is not discovered as a surprise.

### 5. What the host sees per step

Loss scalar (one float D2H). Nothing else. `zero_grad` = device
memset. Eval/sampling/checkpoint paths call materialize explicitly —
eval already runs under B1 semantics today and keeps doing so.

### Memory budget (stated, not hoped)

Rung C shape d=512, T=512, L=4: params ~13M floats (~50 MB), peak
activations + grads + Adam state comfortably under 1 GB. T4 = 16 GB.
Not a constraint until far above this ladder.

## Staging (each stage lands green or does not land)

- **B2.0** Variable DevState plumbing + gemm transpose variants,
  **write-through and epoch-scoped**: results always land in host
  storage (host data is never stale in B2.0 — no validity flags are
  live yet), and cached device operands are trusted only inside a
  `step_begin()`/`step_end()` window. Host mutations (optimizer, the
  gradcheck suite's FD pokes, checkpoint load) all fall between
  windows, so the staleness class is killed by SCOPE rather than by
  discipline — code that never opens a window gets today's behaviour
  unchanged. Result caching is deliberately deferred: in the real
  models every matmul→matmul chain passes through a host-side op
  (GELU, softmax, norm) until B2.1 moves those, so B2.0's win is
  operand dedup (params upload once per window instead of once per
  use) plus the in-kernel transposes. The full §2 flag contract with
  DEVCHECK asserts activates in B2.1 when downloads are deferred.
- **B2.1a — T4-VALIDATED 21 Aug 2026** (receipts:
  docs/receipts_b21a_t4_20260821.txt; Tesla T4, CC 7.5). Kernel parity:
  elementwise bitwise 0.00e+00, activations <= 2.4e-07, rowwise worst
  3.8e-06 (layernorm dgamma, column-sum order); composed-tape OFF vs ON:
  loss bitwise 0.00e+00, all leaf grads <= 2.3e-10. gradcheck (27 ok)
  and nn (22 ok, on the T4) reran green WITH the op set live — same FD
  oracle. The run also caught and fixed a latent repo defect: three
  CMake-referenced test files were never committed (paperkiln cbbed8b)
  and coalfire's remote CMakeLists referenced the pre-31-Jul
  train_wikitext (coalfire 9c0b023) — fresh clones of either repo had
  been failing at CMake generate for weeks while local builds passed.** the op set
  itself: elementwise add/sub/mul/scale/axpy/fill, sigmoid fwd/bwd, GELU
  fwd/bwd (the CORRECT derivative), softmax rowwise fwd/bwd, layernorm
  fwd/bwd, rmsnorm fwd/bwd — src/cuda_ops.cu, one canonical entry per
  op, gated by MICROTORCH_DEVICE_OPS=1 + device==CUDA, write-through
  (host never stale, B2.0's contract). Tape call sites in ops.cpp fall
  through to their own loops, so CPU numerics are untouched. BOTH
  attention nodes' matmuls now route through the transpose-flag gemm
  (the host .transpose() temporaries in their forwards and backwards
  are deleted — the design's section 3 debt, paid). New gate:
  tests/test_cuda_ops.cpp (kernel parity vs the CPU formulas at 1e-6
  elementwise / 1e-5 rowwise, plus a composed-tape OFF-vs-ON leg), and
  colab_cuda_validate.sh now reruns gradcheck/nn with the op set live.
- **B2.1b — T4-VALIDATED 29 Aug 2026** (receipts:
  docs/receipts_b21b_t4_20260829.txt; Tesla T4, CC 7.5, CUDA 12.8).
  Deferred downloads behind MICROTORCH_DEFER_DOWNLOADS=1: value cache
  in cuda_resident.cu holds device-fresh outputs+grads (epoch-stamped,
  stale-flagged), gemm + all 15 devops entries defer their D2H and
  chain on-device; materialize()/step_end() are the download
  boundaries; accumulate() is the grad choke point. THE HEADLINE:
  defer-vs-write-through tape diffs all EXACTLY 0.00e+00 — deferral
  changes scheduling, never values. Full suites green WITH deferral
  live: test_cuda_ops legs 1-3, test_step_residency (CUDA pin 6.89e-08,
  between-window poke visible), gradcheck 29/29, nn 22/22. Original
  bullet's DevState-grows-grads design superseded by the value cache
  (checklist item 1); slice audit closed (item 5: slicing copies;
  recycled-address class killed by epoch guards).

  **B2.1b working checklist (opened 28 Aug 2026; core landed
  29 Aug 2026 — design note below):**
  1. [x] Validity state — DESIGN AMENDED in implementation: instead of
     growing `DevState` with per-Variable `d_grad`+flags, B2.1b adds a
     VALUE CACHE (`g_vcache` in cuda_resident.cu): device-fresh op
     outputs AND grads keyed by host data pointer, epoch-stamped, with
     a `stale` flag = the validity contract. Param `DevState` slots
     stay as B2.0 built them (params never go host-stale in B2.1b —
     the optimizer host-mutates between windows). One mechanism covers
     activations, intermediates, and grads uniformly.
  2. [x] Deferral behind `MICROTORCH_DEFER_DOWNLOADS=1` (env read in
     device.cpp set_from_env; setter `set_defer_downloads`). Active
     only inside a step window (killed-by-scope, as B2.0). gemm and
     all 15 devops entry points route outputs through
     `detail::vc_output` (deferred: buffer retained, no D2H) and
     inputs through `detail::vc_operand` / `window_operand`
     (stale-value hit → no H2D re-upload). In-place axpy preloads the
     buffer unless the cache already holds a fresh copy. Aux per-row
     stat vectors stay write-through by design (small; host-consumed).
  3. [x] `materialize(m)` / `materialize_all()` are the download
     boundaries; `step_end()` materializes all when defer is on (the
     optimizer/eval/checkpoint still read host between windows), so
     cross-window staleness cannot exist. `host_stale(m)` exposed.
  4. [x] Staleness enforcement wired 29 Aug — by MATERIALIZE-AT-ENTRY
     rather than assert-only: `Variable::accumulate()` materializes its
     input (the choke point that keeps every grad host-authoritative
     until B2.3 moves accumulation on-device), every CPU-formula tape
     op materializes the Matrix inputs it reads (22 sites in ops.cpp:
     add/sub/mul/add_bias/mean/scale/transpose/slice/concat/embedding-
     free ops/CE/mul_row/mul_col/silu/rope/kimi/ssm/rms_row/add_scalar/
     dropout/relu/inplace_unary), and fused/swa attention materialize
     their gemm scores + ds before the host mask/softmax loops (moving
     that fused masked softmax on-device is B2.2 territory). The
     devops-routed ops need nothing: their fall-throughs can only run
     with devops off, and defer_active() is COUPLED to
     device_ops_enabled() for exactly that reason. DEVCHECK's
     `devcheck_host_read` + macro remain available as belt-and-braces
     under `-DMICROTORCH_DEVCHECK`.
  5. [x] Aliasing/recycling audit CLOSED 29 Aug. Findings: (a) current
     tape slicing (slice_cols/concat_cols) COPIES, so no offset-pointer
     aliasing reaches the value cache; kimi's rows_of offset views stay
     on the host-only path (inputs materialized at entry). (b) The real
     hazard was RECYCLED HOST ADDRESSES inheriting stale entries across
     windows — closed by epoch-guarding vcache hits (a stale entry is
     honoured only within the window that stamped it), plus the
     accumulate() choke ensuring within-window temporaries are
     materialized before they can die stale.
  6. [x] tests/test_cuda_ops.cpp leg 3 added: staleness-contract units
     (stale inside window; chained op consumes the stale value
     on-device; materialized at step_end; deferred chain == write-
     through at 1e-6) plus the composed tape under deferral vs plain
     ops-ON at leg-2 tolerances. Local checks: both .cu TUs compile
     clean under nvcc 12.6 + MSVC; CPU side g++ -fsyntax-only clean.
  7. [ ] T4 validation via colab_cuda_validate.sh per the contract
     below; receipts into docs/, phase doc + ROADMAP updated. THIS IS
     THE ONLY OPEN ITEM — B2.1b is code-complete pending hardware.
  Constraint for this pass: do not touch tools/mtstudio.cpp or
  tools/parity_model.hpp (uncommitted WIP present, 28 Aug).
- **B2.2** embedding + CE: the forward touches host only at the loss
  scalar.
  IN PROGRESS 30 Aug — the attention half is code-complete: fused
  (causal/block) and swa (window+sinks) masked softmax moved on-device
  (`k_attn_masked_softmax` / `k_swa_masked_softmax` + shared
  `k_attn_softmax_bwd`; one block per row, visible ranges computed
  exactly as the ops.cpp host loops compute them, masked entries hard
  zeros). All four former forced-materialize sites in ops.cpp now try
  the devops path first and keep the host loop as fallback + reference
  semantics; the backward fallbacks materialize A defensively. In-place
  via the vcache `Out(need_current)` seam, so under deferral the [T,T]
  scores/weights never cross the bus inside a step — this was the
  biggest remaining forced materialize. `test_cuda_ops` leg 4: three
  flavors x {devops-off reference, devops-on, defer} at leg-2
  tolerances. Local checks: both .cu TUs compile clean (nvcc 12.6,
  `-DMICROTORCH_CUDA`), CPU side g++ -fsyntax-only clean. T4 validation
  rides the next colab_cuda_validate.sh run (legs already rerun
  test_cuda_ops under DEFER=1, so leg 4 is covered with no script
  change).
  CODE-COMPLETE 30 Aug (same day, second increment): embedding + CE
  moved on-device. `k_embed_gather` (ids bounds-checked host-side, ids
  as an IBuf int upload; the forward's first activation is born
  resident; backward scatter-add stays host until B2.3),
  `k_ce_fwd` + `k_vec_sum` (softmax + per-row nll + on-device sum — the
  [R,vocab] logits never come home and the host receives ONE float,
  which is the B2.2 contract sentence verbatim; P cached through the
  vcache for the backward under the host op's (P - onehot)/N contract,
  1e-12 clamp included), `k_ce_bwd` ((P - onehot) * g; the gradient
  first touches host at accumulate(), B2.3's choke point). Host paths
  retained as fallback + reference; `test_cuda_ops` leg 5 composes
  embedding -> CE exactly as a model does and pins loss + scatter-add
  table grad across {off, on, defer}. Both TUs nvcc 12.6 clean; CPU
  g++ -fsyntax-only clean.
  **T4-VALIDATED 30 Aug** (receipts docs/receipts_b22_t4_20260830.txt,
  commit 5bcf0d1 = code-identical child of the 0457af9 fix): 12/12
  suites, 281 checks, 0 fails on Tesla T4 / CUDA 12.8.93 — legs 4+5
  green in BOTH test_cuda_ops invocations (DEVICE_OPS=1, and
  +STEP_RESIDENCY+DEFER_DOWNLOADS) with identical numbers: worst leg-4
  diff 3.725e-09 (swa), leg-5 ce loss 2.384e-07 / table grad 1.397e-09;
  gradcheck 29/29 and nn 22/22 under all three env configs.
  The road here banked a real bug class: the first T4 run SIGABRTed at
  leg 4 with glibc heap corruption — a DEFERRED TEMPORARY DYING STALE
  (closure-local ds freed while its vcache entry was still stale;
  step_end()'s materialize_all() then D2H'd into the freed pointer).
  Fixed by the new `device::discard(m)` primitive (drop the entry, no
  download) wired into both attention backwards (ds + A, A also on the
  v-only early return) and CE's backward (dl + P) — which doubles as an
  optimization: A and P skip their pointless step_end downloads.
  Confirmed both directions on one VM: pre-fix binary rc=134 at leg 4,
  post-fix all legs green under MALLOC_CHECK_=3 MALLOC_PERTURB_=85
  (docs/receipts_b22_mcheck_t4_20260830.txt). RULE FOR EVERY FUTURE
  DEVOPS PATH: any deferred output that dies before step_end() MUST be
  discarded or materialized — never left stale.
- **B2.3** AdamW/SGD on device; optimizer state uploads once at
  construction; checkpoint = explicit materialize sweep. Full step
  resident. Staged like B2.1a -> B2.1b:
  1. [x] **B2.3a (30 Aug): write-through parity seam.** `k_adamw_step`
     / `k_sgd_step` (bias corrections c1/c2 computed host-side so the
     per-element math is the nn.cpp formula verbatim); SGD::step and
     AdamW::step try the devops path per param, host loops stay as
     fallback + reference. State round-trips the bus each step — slower
     than host BY DESIGN at this stage; the point is a parity-testable
     seam. Residency-transparent: host state after a device step is
     identical to the host loop's, so the existing mutate->re-upload
     contract is untouched. test_cuda_ops leg 6 drives both real
     optimizers over 5-step trajectories with fresh grads per step
     (state must track across steps, not one update). Both TUs nvcc
     clean; CPU syntax clean; T4 pending next validate run.
  2. [x] **B2.3b (30 Aug): persistent device optimizer state.** m+v
     (AdamW, one owned buffer: m block then v block) and vel (SGD) live
     on device — allocated ZEROED on the first step (= the host init,
     so trajectories match), owned via shared_ptr with
     opt_state_free as the deleter, NEVER the pointer-keyed value cache
     (the B2.2 lifetime rule applied by design). Per step: only g
     uploads and p round-trips (host-authoritative until B2.3c) — the
     2x-state bus traffic of B2.3a is gone. nullptr from opt_state_new
     (CPU build / devops off) pins the host path for the whole run;
     a mid-run MICROTORCH_DEVICE_OPS toggle with device state present
     throws rather than silently forking the trajectory (loud-failure
     rule). mtstudio does not serialize optimizer moments (verified),
     so no checkpoint sync is required yet; a materialize accessor is
     B2.3c-adjacent work if resume-with-moments ever lands. leg 6
     covers the persistent path (fresh optimizer per drive, 5-step
     trajectories vs host). p updated in its RESIDENT slot moves to
     B2.3c with device-side accumulate.
  3. [x] **B2.3c: device-side accumulate.** Variable::accumulate keeps
     grad on-device (devops axpy: vcache Out(need_current) on grad, In
     on the incoming g — a stale hit, no download) inside the step
     window; the materialize choke moves from every accumulate to the
     clip/checkpoint boundary. clip_grad_norm needs a device norm
     reduction (or materializes once per param).
     GROUNDWORK LANDED 30 Aug: ~Variable now discards its data and grad
     value-cache entries, making the B2.2 lifetime rule STRUCTURAL — no
     entry can outlive its host buffer through a dead Variable, and no
     dangling cache key can greet a recycled address. (Also closes the
     pre-existing dangling-non-stale-key exposure for dead activations.)
     AUDIT DONE + SWITCH FLIPPED 30 Aug (two passes, 27 sites):
     pass 1 caught the 12 direct `self->grad(...)` element-read
     closures (add_bias/mean/slice/concat/embedding — which also
     materializes t->grad against the tied-weight case — CE, mul_row,
     mul_col, rmsnorm fallback, norm_rows, relu, inplace_unary); pass 2
     caught the forms the first grep missed — copies (`dx = self->grad`
     in gelu/sigmoid fallbacks, silu, rope, dropout), hadamards (mul),
     scalar multiplies (sub, scale), transpose, layernorm fallback,
     kimi's rows_of slicing, ssm's dY alias. Every site now
     materializes at entry (no-op unless stale). accumulate() itself:
     when either side is device-fresh, devops::axpy through the value
     cache — no download; else the host add with the B2.1b choke,
     bit-identical. backward() discards each non-leaf's grad entry
     right after its backward_fn runs (fully consumed by topo order),
     so step_end downloads param grads ONLY — the materialize choke
     has moved from every accumulate to the step boundary, which is
     what B2.3c is. Local checks green (CPU syntax, both TUs); T4
     validation is item 4.
  4. [x] **T4 VALIDATION PASSED 30 Aug 20:52** (receipts
     docs/receipts_b23_t4_20260830.txt, commit d588198): 12/12 suites,
     285 checks — legs 4/5/6 numerically IDENTICAL between DEVICE_OPS=1
     and the fully-deferred config (leg 6 optimizer trajectories
     1.192e-07 vs host), gradcheck 29/29 + nn 22/22 under all three
     configs, training pin 6.89e-08.
     THE ROAD THERE — the dying-temporary class, round two: the first
     gate run on e38f710 crashed rc=134 (glibc heap corruption,
     deterministic 2/2 fresh VMs). Root cause: pre-B2.3c, accumulate()
     materialized every incoming g, so gradient TEMPORARIES (gemm
     expression results, devops dX locals) died host-fresh; the axpy
     path consumed them by stale-hit and left their entries stale, and
     step_end() then wrote each dead temp's freed host buffer — the
     B2.2 class through a new door. Fix (d588198): rvalue
     accumulate(Matrix&&) overload — expression args bind
     automatically, 29 named locals std::move'd, the temp SELF-DISCARDS
     on consumption; lvalue pass-throughs (accumulate(self->grad)) keep
     their entries for later stale-hit/materialize readers. Confirmed
     both directions: pre-fix deterministic crash; post-fix clean under
     MALLOC_CHECK_=3 in both configs (receipts
     docs/receipts_b23_mcheck_t4_20260830.txt) and the full gate above.
     STANDING RULE (now enforced three ways): a deferred entry must
     never outlive its host buffer — ~Variable discards data+grad,
     backward() discards consumed non-leaf grads, and rvalue accumulate
     discards temporaries.
  5. [x] **ADOPTION GATE PASSED 31 Aug 2026 — B2 IS ADOPTED FOR RUNG C**
     (receipts docs/receipts_b2gate_t4_20260831.txt, commit e9da26d,
     Tesla T4, ParityLM T=512 L=4, 3 warmup + 6 timed steps).

     | d | cpu ms/step | b2 ms/step | speedup | final loss cpu / b2 |
     |---|---|---|---|---|
     | 256 | 6530.22 | 308.88 | **21.1x** | 5.2485 / 5.2485 |
     | 512 | 21785.61 | 714.86 | **30.5x** | 4.9382 / 4.9382 |

     Engine ladder at d=256 (ms/step): cpu 6530 -> gpu 1095 -> ops 499
     -> res 574 -> b2 309. The final losses are IDENTICAL at print
     precision across all five engines at d=256 and across cpu/b2 at
     d=512 — the gate times the same computation, which is the property
     the first run lacked. Deferral verified live in the same run: the
     b2 probe reports logits_stale=1 (activations genuinely staying on
     device) with loss_dev == loss_host == the healthy engines' value
     and matching row statistics. The speedup GROWS with width
     (21x -> 30x), which is the expected direction: bigger matmuls
     amortize the fixed per-step overhead the CPU path cannot.
     Consequence: Rung C (d=512, T=512) runs on CUDA. A CPU cell that
     took ~6 hours takes ~12 minutes.
     **FIRST BENCH RUN EXPOSED A CORRECTNESS BUG THE WHOLE SUITE
     MISSED (30 Aug 2026).** The gate is a measurement, but it was the
     first end-to-end training run at REAL shapes (vocab 4096, d=256/512,
     T=512, L=4). Its B2 arm clocked 19-25x faster than CPU while sitting
     at loss EXACTLY ln(4096) = 8.3178 — uniform logits, no learning —
     at every width, depth (even L=0), and step. No timing was banked.
     Bisection by engine ladder (cpu / gpu / ops / res / b2, shared seeds
     and data, per-step losses): cpu, gpu, ops and res agree step-for-step
     to +-1e-4 and reach identical final loss; ONLY deferral breaks, and
     it breaks the first forward. A probe printing device-CE vs host-CE on
     materialized logits plus row stats gave the signature: logits
     row0 min = max = mean = 0.0 with stale=1 — the device buffer itself
     held zeros.
     ROOT CAUSE (`window_operand`, cuda_resident.cu): the value-cache
     stale-hit check sat INSIDE the slotless branch. Every Variable's
     data passes a slot (ops call gemm with `&var->dev`), so under
     deferral a gemm consuming a deferred activation took the slot path,
     skipped the cache, and uploaded the untouched host buffer — zeros —
     into its DevState. The function's own comment already stated the
     rule ("a stale hit is mandatory — host is behind"); it was simply
     implemented in one branch of two. Fix (42725a7): the stale hit is
     UNIVERSAL and outranks both the slot and the B1 table.
     WHY 285 CHECKS PASSED ANYWAY — the blind spot, now closed: legs 2-3
     compose tapes with NO matmul after a device op, every unit leg
     materializes between stages, and every test shape is tiny. The bug
     needs exactly "deferred activation -> slotted gemm, no intervening
     materialize". Leg 7 (e9da26d) is that shape, twice
     (layernorm->matmul->mean and embedding->matmul->CE), pinning loss
     and both grads defer vs write-through.
     STANDING LESSON: a numerics suite that only runs at test shapes
     certifies numerics at test shapes. The adoption benchmark doubles as
     the first realistic-shape correctness probe and should be treated as
     part of the gate, not as a postscript to it.

## Validation contract

Local: compile-only (nvcc), as always — no local GPU execution.
On T4, per stage:
1. Full gradcheck/nn/lora suites under `MICROTORCH_DEVICE=cuda` with
   `MICROTORCH_STEP_RESIDENCY=1` — the same FD oracle that gates every
   other path.
2. `test_step_residency`: N=50 training steps, CPU vs B2, identical
   seeds/data. NOT bitwise — device reduction order differs; the pin
   is per-step loss within rtol 1e-4 early, plus final-weights max
   abs diff bound. Bitwise claims are reserved for same-backend pins
   (the SWA/highway precedent).
3. DEVCHECK build runs the studio smoke: any host read of a
   device-valid tensor asserts.

## Adoption criterion (the Rung C gate, stated before any numbers)

Benchmark = wall-clock per training step, d=256 and d=512 at T=512,
CPU AVX vs B2, on T4. B2 is adopted for Rung C iff it wins at d=512.
If AVX still wins there, that is a *finding* — Rung C runs on CPU,
the result goes in this file, and the GPU path waits for bigger dims
without shame.

## Non-goals (B2)

Streams/overlap, multi-GPU, fp16/mixed precision, fused attention
kernels, inference-path changes. Each is a separate decision after B2
measures.
