# Chunked (parallel-in-time) selective-scan backward: design

Claude Opus 5.5, 3 Nov 2026. Branch `claude/mamba-chunked-backward`. Code: `src/selective_scan_chunked_backward.cuh`
(dispatch in `src/cuda_ops.cu`), host test `tests/test_selective_scan_chunked_backward.cpp`, binary64 reference
`tests/selective_scan_backward_reference.hpp`, device check `tools/scan_bwd_bench.cu`. GPU run: `PROTOCOL_BACKWARD.md`.

## What the ordered backward computes

Per sequence q, channel c and state n, the forward is s_t = a_t s_{t-1} + b_t from s = 0, with
a_t = exp(Δ_tc A_cn), b_t = B_tn x_tc Δ_tc and y_tc = Σ_n C_tn s_t + D_c x_tc. Δ is softplus(dt) when softplus is on
and dt ≤ 20, else dt. Given the upstream gradient dy, the backward is:

| quantity | definition | kind |
|---|---|---|
| adjoint | g_t = dy_tc C_tn + G_{t+1}, G_{t+1} = g_{t+1} a_{t+1}, G = 0 after the last row | linear recurrence, backward in time |
| dpre_t | g_t s_{t-1} a_t (the gradient of Δ A) | local |
| dA_cn | Σ over sequences and rows of dpre_t Δ_tc | reduction over time |
| dΔ_tc | Σ_n dpre_t A_cn + x_tc Σ_n g_t B_tn | reduction over n |
| dx_tc | Δ_tc Σ_n g_t B_tn + dy_tc D_c | reduction over n |
| ddt_tc | dΔ_tc σ(dt_tc) with softplus and dt ≤ 20, else dΔ_tc | local |
| dB_tn, dC_tn | Σ_c g_t x_tc Δ_tc and Σ_c dy_tc s_t | reductions over c |
| dD_c | Σ over sequences and rows of dy_tc x_tc | reduction over time |

These are all the ordered backward produces (`k_selective_scan_bwd_channels` and `k_selective_scan_bwd_BC`, and the
CPU path in `ops::selective_scan`). Only g is a recurrence; given g and the forward's states, everything else is
local or a reduction. The ordered kernel runs one thread per channel, which walks every sequence backward in turn:
D threads in all, the batch serial inside each thread (the ordered forward at least runs batch × D).

## The chunked backward

The adjoint recurrence g_t = e_t + a_{t+1} g_{t+1} (e_t = dy_tc C_tn) has the forward's form with time reversed, so
the forward's chunks (L rows; the last of a sequence may be shorter) carry over. For chunk k with rows [r0, r1) and
incoming G_in (the G that the following chunk passes back, 0 for the last chunk):

    G_out(k) = g_{r0} a_{r0} = Gloc_k + P_k G_in(k),   P_k = Π_{t in [r0, r1)} a_t,

where Gloc_k is the chunk's own backward scan from G = 0. This is exact, since g is linear in G_in. The steps, each
a kernel and each an `SS_HD` function that the host emulation also calls:

| step | work items | what it does |
|---|---|---|
| 1 local | batch · nchunk · D · N | backward scan of the chunk from G = 0: Gloc_k and P_k |
| 2 carry | batch · D · N | chunks last to first: G_in(k-1) = P_k G_in(k) + Gloc_k (the forward's carry, reversed) |
| 3 final | batch · nchunk · D · N | rescan from G_in(k): g_t for every row into the history (the ordered kernel's `g_history` layout) and the chunk's partial sum of dA's terms, rows descending |
| dD part | batch · nchunk · D | the chunk's partial sum of dy x, rows descending |
| 4 reduce | D · N and D | dA and dD from their partials: sequences ascending, chunks last to first, the order in which the ordered kernel visits rows |
| 5 cell | T · D | dΔ, dx, ddt from the history, ascending n: the ordered kernel's arithmetic |
| 6 B/C | T · N | dB, dC, ascending c: the ordered BC kernel's arithmetic |

An output that is not wanted is not computed. If only dC and dD are wanted, no adjoint is computed at all. The
default (variable unset or 0) runs the ordered kernels: their launch code moved unchanged into the `else` branch of
`if (scan_chunk())`, and `selective_scan_kernels.cuh` is untouched. The CPU path never reads the variable.

## Stored against recomputed

- **Read from the forward:** every state S (including each sequence's zero start) and Δ, as the ordered backward
  does. Nothing of the forward is recomputed.
- **Recomputed:** a_t = exp(Δ A), with the forward's exact expression, in steps 1, 3 and 5 (three evaluations per
  (t, c, n) against the ordered kernel's one); e_t = dy C in steps 1 and 3; dpre in steps 3 (for dA) and 5 (for dΔ).
  So the backward does about three times the ordered kernel's exponentials, all of them in parallel work items.
- **Stored by the backward:** the history g (T · D · N floats, the size of S) whenever an adjoint is needed (the
  ordered kernel stores it only for dB); the chunk ends, products and starts and the dA partials (batch · nchunk · D · N
  each, so T · D · N / L each when L divides sl); the dD partials (batch · nchunk · D).
- **Not fused:** step 5 could be folded into step 3 only with a reduction over n across threads, which would change
  the ordered summation order. It stays separate so that, given g, dx and ddt are the ordered arithmetic.

## Rounding: what matches the ordered backward bit for bit, and what does not

Every multiply and add is explicitly rounded (`__fmul_rn`/`__fadd_rn` on the device, volatile binary32 on the host;
no contraction), so results are deterministic. Given the same incoming G, step 3's operations per row are the
ordered kernel's, and given the same g, steps 5 and 6 are the ordered kernels' arithmetic. Rounding therefore
differs from the ordered backward only in two places:

1. the chunk starts G_in(k) of step 2, computed as P_k G_in + Gloc_k instead of along the sequential chain (this
   changes g, and through it every output but dC and dD);
2. the grouping of dA and dD: per-chunk partial sums, then a sum over chunks, instead of one running sum.

With one chunk per sequence (L ≥ sl) every G_in is 0, so g, dx, ddt, dB and dC are the ordered kernel's bits. dA and
dD are too for a single sequence (one partial, summed from +0 exactly as the running sum is). With several
sequences the ordered running sum crosses sequence boundaries and the chunked one restarts per sequence: a
regrouping, within the bound below. The host emulation uses `std::exp` and `std::log1p` where the device uses
`expf` (at most 2 ulp) and `log1pf`; the division in ddt is IEEE on both (nvcc's default `-prec-div=true`;
paperkiln passes no fast-math flag to device code). So the host test establishes the algorithm, indexing and chunk
boundaries, and the device's numerics are checked on the GPU.

## Why a tolerance, and which

Both backward paths evaluate the same exact sums of products in different association. A floating-point evaluation
of a sum of products has error at most γ_K = K u / (1 - K u) times the same expression evaluated in absolute values,
K being the longest chain of rounded operations and u = 2^-24 (Higham, *Accuracy and Stability of Numerical
Algorithms*, 2nd ed., chapter 3). So each path is within γ_K · mag of the exact value and the two are within
2 γ_K · mag of each other, where mag is that absolute-value evaluation. Neither is closer to the truth by
construction, so bit identity is not the contract except where the association is the same (one chunk, above).

**Metric.** |v - ref| / max(1, mag), with ref and mag from an independent binary64 forward and backward
(`tests/selective_scan_backward_reference.hpp`): mag replaces every term by its magnitude (|e_t|, |b_t| and the
adjoint and state recurrences in absolute values; a_t > 0 already). mag ≥ |ref|, with equality for sign-definite
sums, so this is the forward's metric |v - ref| / max(1, |ref|) except where a sum cancels. The forward's metric is
not usable here: dA and dD sum T terms of both signs; where they cancel to |ref| < 1 it demands an absolute error
of 3e-5 from a sum whose terms total about T/4, which correct rounding alone exceeds at T = 4096. Measured on the
host test: under the forward's metric the ordered backward itself reaches 5.5e-5 (chunked 5.7e-5); under the
magnitude metric both reach 2.5e-7.

**Bound.** 3e-5, the frozen selective-scan bound and the forward's. The worst-case γ_K for K ≈ 3 × 4096 is about
7e-4 and does not bind; rounding errors accumulate statistically as about √K u ≈ 7e-6 at 4096 rows, which the bound
covers by a factor of four. The same bound is used for the chunked backward against binary64, the ordered backward
against binary64 (a check of the reference and the metric), and the chunked against the ordered backward (the
triangle inequality would allow twice the bound; the single bound is the stricter gate). The binary64 reference is
itself checked against binary64 central differences (|fd - ref| / (1 + |ref|) ≤ 1e-6).

## Host evidence (3 Nov 2026, this commit)

`tests/test_selective_scan_chunked_backward.cpp`, the forward test's 173 shapes (sequence lengths 1-4096 around
chunk edges, L in {1, 2, 16, 64, 128, 256}, softplus and skip on and off, mixed, long-memory and fast-decay regimes,
one, two and three sequences, D up to 8 and N up to 16), each with an upstream gradient:
`receipts/host_test_bwd_local.json`.

| check | result |
|---|---|
| host emulation of the ordered kernels = the CPU op's backward, bit for bit | 173/173 |
| chunked vs binary64 (bound 3e-5) | worst 2.51e-7 |
| ordered vs binary64 | worst 2.51e-7 |
| chunked vs ordered | worst 2.82e-7 |
| chunked forward's states into the chunked backward, vs binary64 | worst 3.71e-7 |
| one chunk per sequence (L = sl, sl + 1, 4 sl + 7) bit-identical to ordered | 2106/2106 checks; 1008 regrouped dA/dD checks (several sequences) within the bound |
| repeats bit-identical; every output written (NaN-prefilled) | 173/173; 173/173 |
| each output computed alone = the same output computed with all others | 765/765 |
| binary64 reference vs central differences (2514 entries) | worst 2.11e-10 (bound 1e-6) |

A mutation check (not committed) confirmed the test can fail: dropping the carry, reading the state one row late
in step 3, or dropping the chunk product fails 113, 152 and 100 of the 173 cases; summing the dA partials with
chunks ascending (a pure regrouping) passes, as it should.

The default path: the frozen 220-case test on the CPU path gives 220 case lines (inputs, gradients and y at %.9g,
which round-trips binary32 exactly) identical to master's (`d0390e39`), and identical again with
`MICROTORCH_SCAN_CHUNK=2` set: `receipts/default_cpu_local.txt`. The device default is gate 5 of
`PROTOCOL_BACKWARD.md`.

## Expected performance (to be measured, not claimed)

The ordered backward has D-way parallelism; the chunked one batch · nchunk · D · N in its scans and T · D or T · N in
its readouts, at the cost of three exponentials per (t, c, n) and an always-written history. The gain should be at
least the forward's for single sequences and larger with batch, since the ordered backward does not parallelise
over sequences at all. `PROTOCOL_BACKWARD.md` measures it.
