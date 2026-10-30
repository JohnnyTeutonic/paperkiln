#pragma once
// Op-level forward + grad_fn pairs over the PHASE0 op set. Forwards call
// the canonical transformer_core entry points (primitives.hpp); backwards
// are the audited manual formulas. This is the op subset that the tape's
// first gate (tests/test_gradcheck.cpp) verifies; the rest of the audited
// set (layernorm, CE loss, embedding) lands in phase 1b with Modules.
#include <cstdint>

#include "microtorch/autograd.hpp"

namespace microtorch {
namespace ops {

Var matmul(const Var& a, const Var& b);  // [m,k]x[k,n]; AVX2 forward,
                                         // dA = dC B^T, dB = A^T dC
Var add(const Var& a, const Var& b);     // same shape
Var sub(const Var& a, const Var& b);
Var mul(const Var& a, const Var& b);       // hadamard
Var add_bias(const Var& x, const Var& b);  // b: [1,n] broadcast over rows;
                                           // db = column-sum of upstream
Var gelu(const Var& x);                    // tanh approximation; OWN correct
                                           // derivative (see primitives.hpp
                                           // on the transformer_core bug)
Var gelu_erf(const Var& x);                // the exact GELU (Hendrycks and
                                           // Gimpel 2016): 0.5 x (1 + erf(x /
                                           // sqrt 2)), the IR's gelu approx
                                           // "none"; host, f32 erff (IR_SPEC
                                           // 5.8); dx = dy (Phi(x) + x phi(x))
Var softmax_row(const Var& x);             // dX = S .* (dY - rowsum(dY .* S))
Var mean(const Var& x);                    // -> [1,1]

// ---- phase 1b additions (the layer zoo's dependencies) ----
Var scale(const Var& x, float s);                    // x * s
Var transpose(const Var& x);                         // dX = dY^T
Var slice_cols(const Var& x, size_t j0, size_t j1);  // [r, j1-j0); backward
                                                     // scatters into the range
Var concat_cols(const std::vector<Var>& xs);         // inverse of slice_cols
Var layernorm(const Var& x, const Var& gamma, const Var& beta,
              float eps = 1e-5f);  // row-wise; gamma/beta [1, d]
Var embedding(const Var& table, const std::vector<int>& ids);
// gather rows; backward
// scatter-adds into table.grad
// WITHOUT a dense temp (the
// table can be [50k, d])
Var cross_entropy(const Var& logits, const std::vector<int>& targets);
// mean NLL over rows -> [1,1];
// dlogits = (softmax-onehot)/N,
// the loss_kernels.cu contract
Var policy_gradient_kl(const Var& logits, const std::vector<int>& targets, const std::vector<float>& w,
                       const std::vector<float>& k, const Matrix& ref_logp);
// GRPO's loss on the tape (charter 0001, Phase 4), summed over rows:
//   sum_i [ -w_i * log p_i(target_i) + k_i * KL(p_i || ref_i) ]
// with p_i = softmax(logits_i) and ref_logp the reference model's
// log-probabilities ([rows, vocab], constant). The caller folds advantage,
// length normalisation and the KL coefficient into w and k (zero on prompt
// and padding rows). Hand-derived backward, per row:
//   w_i (p - onehot) + k_i p (log p - ref - KL_i).
Var fused_attention_blocked(const Var& q, const Var& k, const Var& v, float scale, size_t seq_len, bool causal);
// Batched attention one sequence block at a time: O(B * seq_len^2) instead
// of O((B * seq_len)^2). fused_attention dispatches to it only when
// blocked_attention_enabled() (env MICROTORCH_ATTN_BLOCKED=1).
bool blocked_attention_enabled();
// Causal-or-not multi-head attention over a fused [T, 3d] qkv (q | k | v,
// head h in columns h dk .. (h + 1) dk of each part) -> [T, d], heads
// concatenated: exactly concat_cols of fused_attention per head on column
// slices, which is what runs off the device. On the device with blocked
// attention live (sl < T) it is one op over all heads (no slices), the
// same floats.
Var attention_heads(const Var& qkv, size_t heads, float scale, size_t seq_len, bool causal);
// The attention kernel of attention_heads and attention_heads_turn_state
// (24 Oct 2026; mtstudio's train.attention_kernel, process-wide, default
// Tiled). Tiled is the code above and below, unchanged. Flash runs
// attention_heads_flash for every head width up to 64
// (devops::flash_attention_supported; wider heads keep Tiled): an online
// softmax that stores the per-row log-sum-exp instead of the [H T, seq_len]
// weights, and a backward that recomputes them (FlashAttention-2). Same
// formulas, another summation order: equal to the tiled path to float
// rounding, not bit for bit.
enum class AttnKernel { Tiled, Flash };
void set_attention_kernel(AttnKernel k);
AttnKernel attention_kernel();
// The Flash op itself. mode 0: full attention within each seq_len block
// (seq_len 0 = one sequence), 1: causal, 2: turn_state with one code per row
// (code non-null exactly for mode 2). On CUDA with the device op set live
// the kernels of src/cuda_flash.cu; otherwise host loops over the same
// tiles (32 keys per online-softmax update), the reference the device legs
// and the gradchecks pin. Backward per row i of each head, visible keys j:
//   P_ij = exp(scale q_i . k_j - L_i), D_i = dy_i . y_i,
//   dS_ij = P_ij (dy_i . v_j - D_i), dv_j += P_ij dy_i,
//   dq_i += scale dS_ij k_j, dk_j += scale dS_ij q_i.
Var attention_heads_flash(const Var& qkv, size_t heads, float scale, size_t seq_len, int mode,
                          const std::vector<int>* code);
// ---- per-turn state tokens (ir.hpp attention mask "turn_state", 24 Oct 2026) ----
// The mask's per-position code, a function of the ids alone: code = 2 s + c
// with c = 1 for a state id (id >= vocab) and 0 for text, and s the number
// of COMPLETE state blocks strictly before the position, a complete block
// being k consecutive ids vocab, vocab + 1, .., vocab + k - 1 in that order.
// Counted afresh in every seq_len block of ids (0 = one sequence). A state
// id outside a complete block (a window that starts inside one) is a state
// position of the current segment.
std::vector<int> turn_state_codes(const std::vector<int>& ids, size_t vocab, size_t k, size_t seq_len = 0);
// The incremental form: feed ids one at a time from (seg, run) = (0, 0);
// returns the position's code and advances the tracker.
int turn_state_step(int id, size_t vocab, size_t k, int& seg, int& run);
// Whether key code cj is visible to query code ci (key at or before the
// query): the same segment, or a state position of an earlier one.
inline bool turn_state_visible(int ci, int cj) { return (cj & 1) != 0 || (cj >> 1) == (ci >> 1); }
// attention_heads under the turn_state mask: query i of a sequence block
// sees key j <= i iff turn_state_visible(code[i], code[j]); code has one
// entry per row (turn_state_codes over the same seq_len). Per head, row by
// row, over the visible keys in ascending position: dot product summed
// over the head's columns, times scale, max over the visible keys,
// exp(s - max), the sum, divided, then the weighted values summed in
// ascending key order (forward_cached_rows' arithmetic). Host loops, or
// with the device op set on CUDA the blocked all-heads kernels with a
// turn_state softmax (equal to rounding). Weights are kept block-local [H
// T, seq_len] (head-major) for the backward; masked entries are zeros, so
// the backward is the causal one. A code vector with no state position
// still runs this op (callers that want plain causal attention bit for bit
// call attention_heads).
Var attention_heads_turn_state(const Var& qkv, size_t heads, float scale, size_t seq_len,
                               const std::vector<int>& code);
Var dpo_weighted(const Var& logits, const std::vector<int>& targets, const std::vector<int>& row_pair,
                 const std::vector<int8_t>& row_sign, const std::vector<double>& ref_margin,
                 const std::vector<float>& pair_w, float beta);
// Provenance-weighted DPO (charter 0001, Phase 5). Row r belongs to pair
// row_pair[r] (-1 = ignored, e.g. prompt or padding) with sign +1 (a chosen
// completion token) or -1 (a rejected one). Per pair,
//   margin_p = sum_r sign_r log p_r(target_r) - ref_margin_p
// where ref_margin_p is the same quantity under the frozen reference. Loss:
//   sum_p w_p softplus(-beta margin_p) / sum_p w_p   ->  [1,1].
// Pairs with w = 0 contribute nothing, so weights in {0,1} equal plain DPO
// on the kept pairs. Throws if every weight is 0.
Var cross_entropy_masked(const Var& logits, const std::vector<int>& targets, const std::vector<uint8_t>& mask);
// SFT loss (charter 0001, Phase 3): mean NLL over rows with mask 1 only;
// masked rows get zero gradient. All ones IS cross_entropy (bit-identical
// on every backend). Otherwise, on the host, equal bit for bit to
// cross_entropy on the kept rows; with the device op set live, a device
// path (devops::ce_fwd_masked/ce_bwd_masked) whose gradient equals device
// cross_entropy on the kept rows bit for bit and whose loss matches the
// host within fp32 rounding (the sum's order differs). Throws if every
// row is masked.

// ---- the fused LM head + cross-entropy (LOCAL_LLM_STACK.md section 11 item 5.1, 26 Oct 2026) ----
// The head's weight as the model stores it: InOut is nn::Linear's W [d, vocab]
// (logits = h W); VocabMajor is a tied token embedding E [vocab, d] (logits =
// h E^T, the llama family's default head).
enum class HeadLayout { InOut, VocabMajor };
// mean NLL of softmax(h W + b) at the targets over the kept rows -> [1, 1],
// without ever holding the [rows, vocab] logits: the rows go through in
// chunks of chunk_rows (0 = linear_ce_chunk_rows), each chunk's logits are
// computed, turned into its loss and its gradient in place, and dropped. The
// gradients are computed in the forward (Liger-Kernel's FusedLinearCrossEntropy,
// Hsu et al. 2024, arXiv 2410.10989: chunk over tokens, eager gradients, three
// GEMMs per chunk), so the backward only hands them out; like Cut
// Cross-Entropy (Wijmans et al. 2024, arXiv 2411.09009) it never materialises
// the logits, but it keeps every gradient entry (no CCE gradient filtering).
//   mask: empty = every row (ops::cross_entropy), else 0/1 per row
//         (ops::cross_entropy_masked: SFT, loss policy all/final/each, state
//         tokens, padding); a masked row's target is never read and its
//         gradient is exactly zero. Throws if every row is masked.
//   bias: [1, vocab] or null.
//   upstream: the gradient the caller's graph will send into the loss (mtstudio:
//         1 / accum). The eager gradients use g = upstream / kept, as the
//         unfused backward does; if the backward then brings another value,
//         the stored gradients are rescaled by its ratio (one more rounding).
// Numerics against the unfused path (matmul [+ add_bias] -> cross_entropy[_masked]):
//   host: per row the same GEMM, softmax, clamp and (P - onehot) g as the
//     unfused ops, the loss summed in double in ascending row order. Loss, dh
//     and db are bit-identical whenever host_gemm_rows_free holds for the
//     logits GEMM (rows, vocab, d) and the dh GEMM (rows, d, vocab) at both
//     the full row count and the chunk; dW (dE when tied) is summed over the
//     rows in ascending order, one rounded product and add per row, and
//     matches the vendored blocked GEMM's order only to float rounding. Every
//     result is the same at every chunk size and OpenMP thread count.
//   device (devops::linear_ce): the unfused kernels' own CE kernels on each
//     chunk, the GEMMs through the step-resident path; with the tiled GEMM
//     (MICROTORCH_CUBLAS unset) every output is the unfused device path's bit
//     for bit, dW's chunks continuing one FMA chain per element; under cuBLAS
//     to float rounding. Not yet run on a GPU.
Var linear_cross_entropy(const Var& h, const Var& w, HeadLayout layout, const Var& bias,
                         const std::vector<int>& targets, const std::vector<uint8_t>& mask, float upstream = 1.0f,
                         size_t chunk_rows = 0);
// The default chunk: Liger's rule, the logits chunk about the size of the
// hidden states (next power of two >= rows / ceil(vocab / d)), at least 32
// rows (the device GEMM's tile), at most rows.
size_t linear_ce_chunk_rows(size_t rows, size_t d, size_t vocab);
// True when the vendored blocked GEMM (matmul_optimized: halve M, N and K
// together until all are <= 128) computes every row of an [M, K] x [K, N]
// product with the summation order it would give that row alone, that is
// when the M dimension never forces a split. Then a chunk of rows is the
// full product's rows bit for bit.
bool host_gemm_rows_free(size_t M, size_t N, size_t K);

// ---- phase 2a additions (DiT's dependencies) ----
Var mul_row(const Var& x, const Var& r);  // x .* broadcast row r [1, cols];
                                          // dr = colsum(dY .* x) -- the
                                          // multiplicative twin of add_bias
Var silu(const Var& x);                   // x * sigmoid(x) (DiT's nonlinearity
                                          // for the conditioning MLP)
Var swiglu(const Var& gate, const Var& up);  // mul(silu(gate), up), the same
                                             // bits; on the device one op that
                                             // recomputes silu(gate) in backward
                                             // instead of keeping it

// ---- phase 2b additions (Llama-family dependencies) ----
// The epsilon rmsnorm adds to mean(x^2), on host and device alike (the IR
// export states it from here).
constexpr float kRmsNormEps = 1e-5f;
Var rmsnorm(const Var& x, const Var& w);  // RMS normalization: x / RMS(x) * w
                                          // (no bias, no mean centering);
                                          // w: [1, d], learned scale per feature

// ---- quantisation-aware training (BitNet b1.58; LOCAL_LLM_WORK R3) ----
// Fake quantisers with straight-through estimators: the forward returns the
// quantised values in float, the backward passes the gradient through
// unchanged (dX = dY). Formulas are those of the BitNet b1.58 authors'
// reference code ("The Era of 1-bit LLMs: Training Tips, Code and FAQ"):
//   ternary_ste:   s = 1 / max(mean|W|, 1e-5) over the whole tensor;
//                  y = clamp(round(W s), -1, 1) / s   (absmean ternary)
//   act_quant_ste: per row, s = 127 / max(max_j |x_j|, 1e-5);
//                  y = clamp(round(x s), -128, 127) / s   (absmax int8)
// round is round-half-to-even (std::nearbyint; rintf on the device), as
// torch.round. mean|W| accumulates in double (on the device too, in a
// different order: equal to double rounding). bits: 2..8 for act_quant_ste
// (127 and -128 become 2^(bits-1) - 1 and -2^(bits-1)).
Var ternary_ste(const Var& w);
Var act_quant_ste(const Var& x, int bits = 8);
// The values alone (no tape), for export and tests: the ternary codes in
// {-1, 0, 1} and the scale s, so ternary_ste(W) = codes / s.
float ternary_scale(const Matrix& w);  // s above
// RoPE pairing (keramos IR_SPEC 5.14): Interleaved rotates adjacent pairs
// (x[2j], x[2j+1]) within a head (paperkiln's own models, llama.cpp NORM);
// Neox rotates (x[j], x[j + n_rot/2]) (llama.cpp NEOX, HF rotate_half;
// Qwen, Gemma, GPT-NeoX).
enum class RopePairing { Interleaved, Neox };
Var apply_rope(const Var& qk, const std::vector<int>& pos, float theta_base, size_t head_dim,
               bool all_heads, RopePairing pairing = RopePairing::Interleaved, size_t n_rot = 0);
// Rotary embeddings to query/key rows.
// qk: [T, 3*d] (fused qkv); returns
// [T, 3*d]. Pair j of each head (by
// `pairing`) rotates by
// pos * theta^(-2j/n_rot).
// all_heads: every head of q and k
// (d/head_dim of them); false: head 0
// only (legacy, arch.rope_heads=first).
// n_rot (0 = head_dim; even, <= head_dim):
// only each head's first n_rot dims
// rotate, the rest pass through (partial
// rotary). The defaults are the original
// op, the same floats.

// ---- third-party graph ops (keramos thirdparty profile; LOCAL_LLM_STACK.md section 11 item 4.4) ----
// Per-group RMSNorm (IR_SPEC 5.6 rmsnorm `group`): each consecutive `group`
// columns of a row normalised on their own, x / RMS * w with the weight w
// [1, group] shared by every group (per-head q/k norm: Qwen3, Gemma 3).
// group == cols(x) is ops::rmsnorm. Same epsilon (kRmsNormEps) and order of
// operations as rmsnorm, per group: the squares summed in ascending column,
// divided by group, 1/sqrt(. + eps), then x * inv * w. Backward per group as
// rmsnorm's; dw sums every (row, group) in row-major order. On CUDA the
// rmsnorm kernels over the same buffer read as [rows * groups, group].
Var rmsnorm_group(const Var& x, const Var& w, size_t group, float eps = kRmsNormEps);
// Grouped-query attention's key/value sharing (IR_SPEC 5.15): x [T, heads *
// head_dim] -> [T, heads * group * head_dim], output head h a copy of input
// head h / group. Backward: input head k's gradient is the sum of output
// heads k * group .. k * group + group - 1, added in that order (host and
// device alike, so bit for bit). attention_heads over q and the repeated k
// and v is query head h reading kv head h / group, the reference's grouped
// attention.
Var repeat_heads(const Var& x, size_t heads, size_t group, size_t head_dim);
// Mamba-1's depthwise causal convolution (IR_SPEC 6.6): x [T, C], w [C, K]
// (K >= 2), bias [1, C] or null. out[t, c] = sum over k ascending of w[c, k]
// * xx[t + k, c], xx = K - 1 zero rows then x (each sequence of seq_len rows
// starts from a zero state; 0 = one sequence), then + bias[c] on the finished
// sum; each product and sum a rounded f32 operation, so the numpy
// reference's floats. Backward: dx[u] = sum_k dy[u - k + K - 1] w[k] (rows
// inside the sequence), dw[c, k] = sum_t dy[t, c] xx[t + k, c], dbias = the
// column sums. Host only (the device copies come home).
Var conv1d_causal(const Var& x, const Var& w, const Var& bias, size_t seq_len = 0);
// Mamba-1's selective scan (IR_SPEC 6.7, ggml_ssm_scan): x, dt [T, D], B, C
// [T, N], A [D, N], Dskip [1, D] or null. Per sequence of seq_len rows (0 =
// one), from s = 0: delta = softplus(dt) (dt <= 20 ? log1p(exp(dt)) : dt; dt
// itself when !dt_softplus), s[c, n] = s[c, n] * exp(delta A[c, n]) + B[t, n]
// (x[t, c] delta), y[t, c] = sum over n ascending of s[c, n] C[t, n], then +
// Dskip[c] x[t, c]. Backward through time over the kept states (T + 1 of
// them per sequence, [D, N] each); A, B, C, Dskip, x and dt all get
// gradients. Host only.
Var selective_scan(const Var& x, const Var& dt, const Var& B, const Var& C, const Var& A, const Var& Dskip,
                   bool dt_softplus = true, size_t seq_len = 0);

// ---- state-space scan (Mamba/S4, phase 3c completion) ----
Var ssm_scan(const Var& u, const Var& A, const Var& B, const Var& C, const Var& D);
// u [T,n], A [n,n], B [n,1],
// C [1,n], D [1,1]:
//   s_t = A s_{t-1} + B .* u_t
//   y_t = C .* s_t + D * u_t
// Full BPTT backward (reverse
// recurrence dS_t = C.*dY_t +
// A^T dS_{t+1}), FD-gradchecked.
// Sequential scan; the op API is
// associative-scan-shaped so a
// parallel executor can slot in
// behind the same signature.

// ---- sparse-attention research ops (docs/SPARSE_ATTENTION.md V1) ----
Var mul_col(const Var& x, const Var& c);       // x .* broadcast column c [rows, 1];
                                               // dc = rowsum(dY .* x) -- the
                                               // per-ROW twin of mul_row; the op
                                               // that makes gated path-blending
                                               // differentiable
Var rms_row(const Var& x, float eps = 1e-8f);  // per-row sqrt(mean(x^2)) -> [T, 1];
                                               // the surprise magnitude
Var sigmoid(const Var& x);                     // elementwise logistic
Var add_scalar(const Var& x, float s);         // x + s (for affine gate logits)

// ---- mixture-of-experts dispatch (nn::MoEFFN, 19 Oct 2026) ----
// A top-k MoE FFN is COMPOSED on the tape from these index ops plus the
// existing Linear/silu/mul/mul_col/softmax_row, rather than fused into one
// moe_ffn op: gather and scatter are each other's adjoints, so the only new
// gradient math is index bookkeeping, and every expert reuses the audited
// SwiGLU path (and its device seams) unchanged. Gradients reach the expert
// weights through the experts, x through gather_rows, and the router
// through the gate values picked by gather_elems (the top-k INDICES are
// constants; selection is not differentiable). All FD-checked in
// tests/test_moe.cpp. With the device op set on (CUDA) each has a kernel
// that keeps its result resident in the step window, the scatters
// deterministic (no atomics) and equal to the host loops bit for bit
// (tests/test_moe_device.cpp).
Var gather_rows(const Var& x, const std::vector<int>& idx);
// out[i, :] = x[idx[i], :]   ([n, cols]);   dx[idx[i], :] += dY[i, :]
// (repeated indices accumulate; the backward adds into x's gradient in
// place rather than through a dense [rows(x), cols] temporary). idx must
// be non-empty and in range.
Var scatter_add_rows(const Var& x, const std::vector<int>& idx, size_t n_rows);
// out [n_rows, cols], out[idx[i], :] += x[i, :] (rows never named stay
// zero; a row named once is a COPY, so a one-to-one scatter is
// bit-exact);   dx[i, :] = dY[idx[i], :]. Adjoint of gather_rows.
Var scatter_add_rows(const std::vector<Var>& xs, const std::vector<std::vector<int>>& idx, size_t n_rows);
// The sum of the scatters of xs[e] by idx[e], as ONE op: a destination
// row adds its sources in input order, then row order (the first copies).
// The MoE combine: one [n_rows, cols] output where a chain of per-expert
// scatters and adds would build E of them (and their gradients).
Var gather_elems(const Var& x, const std::vector<int>& rows, const std::vector<int>& cols,
                 size_t out_cols);
// Pick entries x(rows[i], cols[i]) into an [n/out_cols, out_cols] matrix,
// row-major (n = rows.size()); dx(rows[i], cols[i]) += dY_i. Picks a
// router's top-k logits or probabilities ([T, k] with out_cols = k) and an
// expert's gate column ([n_e, 1]).
Var logsumexp_row(const Var& x);
// [T, C] -> [T, 1], max-shifted log(sum_j exp x_ij);  dX = dY .* softmax(X).
// The router z-loss is mean(logsumexp_row(logits)^2).

// ---- training utilities ----
Var dropout(const Var& x, float p, unsigned long long seed);
// inverted dropout: keep with prob
// 1-p, scale kept by 1/(1-p) so
// eval needs no rescaling. The mask
// is regenerated from `seed` in
// backward, so nothing is stored.
// p==0 returns x unchanged.

// Global grad-norm clip over a parameter list (PyTorch's
// clip_grad_norm_). Returns the pre-clip norm. Not a tape op: it mutates
// .grad in place between backward() and step().
float clip_grad_norm(const std::vector<Var>& params, float max_norm);

// Additive attention mask for a stacked mini-batch: rows are B sequences
// of seq_len each (seq_len==0 means one sequence spanning all rows).
// Entry (i,j) is 0 where i may attend j and -1e9 otherwise. Cross-block
// entries are ALWAYS -1e9 (sequences in a batch never see each other);
// within a block, j>i is masked when causal, open when not. Plain
// no-grad constant, shared by every attention implementation.
Matrix attention_mask(size_t rows, size_t seq_len, bool causal);

// ---- safe in-place elementwise ops (performance-triage gap 4) ----
// Each op transforms x's buffer IN PLACE and chains x's existing
// backward: the node's gradient is first converted through f' — which
// for these ops is computable from the OUTPUT alone — and then the
// original backward runs. No new tape node, no new allocation; the
// activation that would have been stored simply never exists.
//
// The eligibility rule is mathematical, not stylistic: an op may be
// in-place only if f'(x) is expressible in f(x). relu/sigmoid/scale
// qualify; gelu and silu need their INPUT and are excluded on purpose.
//
// CONTRACT (the caller's side of the bargain, PyTorch's `_` convention):
// (1) x must have NO other consumer — another op's backward would read
//     the mutated buffer and be silently wrong; and
// (2) x must not be the OUTPUT of an op whose backward reads its own
//     output (softmax_row, sigmoid, fused_attention) — their closures
//     would see f(x) where they stored x. The blessed pattern is the
//     one that matters in practice: in-place on a fresh matmul/add
//     result, e.g. relu_(matmul(h, W)) — those backwards read only
//     their parents. Returns x itself.
Var relu_(const Var& x);            // y = max(x,0);      dx = dy .* (y > 0)
Var sigmoid_(const Var& x);         // y = 1/(1+e^-x);    dx = dy .* y .* (1-y)
Var scale_(const Var& x, float s);  // y = s*x;    dx = s * dy
Var relu(const Var& x);             // out-of-place sibling (parity + FD anchor)

// Fused scaled-dot-product attention for one head:
//   Y = softmax(scale * Q K^T + mask(seq_len, causal)) V
// as ONE tape node. The two GEMMs stay on device::matmul (AVX/CUDA);
// the scale+mask+softmax collapse into a single in-place pass, so the
// mask matrix is never materialized and exactly one [T,T] buffer (the
// attention weights, needed for backward) survives the forward —
// against s, s+mask and softmax(s) as separate tape nodes on the
// composed path. Backward is the hand-derived softmax-attention
// gradient; FD-checked in test_gradcheck.
Var fused_attention(const Var& q, const Var& k, const Var& v, float scale, size_t seq_len,
                    bool causal);

// Sliding-window causal attention + attention sinks (sparse phase S1
// baseline): query i sees the first `sinks` positions of its block plus
// the trailing `window` positions. window >= seq_len with sinks=0 is
// BITWISE-identical to fused_attention causal (the equivalence pin,
// test_swa). Backward mirrors fused_attention's; FD-checked.
Var swa_attention(const Var& q, const Var& k, const Var& v, float scale, size_t window,
                  size_t sinks, size_t seq_len = 0);

// ---- phase 3a additions (Novel architectures: Kimi Linear) ----
Var kimi_attention(const Var& q, const Var& k, const Var& v, bool causal,
                   size_t seq_len = 0);  // O(n*d²) linear-time attention
                                         // Feature map: φ(x)=elu(x)+1
                                         // Numerator: Σ φ(k_i)⊗v_i (cumsum)
                                         // Denominator: Σ φ(k_i) (normalization)
                                         // Output: φ(q)·numerator/denominator
                                         // Causal: if true, only attend to past
                                         // seq_len > 0: rows are B stacked
                                         // sequences; the prefix sums RESET at
                                         // every block boundary (the verified
                                         // kernel runs per block)

// ---- runtime-defined ops' primitives (ops::expr, microtorch/expr.hpp) ----
// Host ops, f32, one rounding per element (no fused multiply-add). The
// binary ones take operands of one shape. FD-checked in tests/test_expr.cpp.
Var exp(const Var& x);   // dx = dy .* y
Var log(const Var& x);   // dx = dy ./ x
Var sqrt(const Var& x);  // dx = dy .* 0.5 / y
Var tanh(const Var& x);  // dx = dy .* (1 - y^2)
Var erf(const Var& x);   // dx = dy .* 2/sqrt(pi) exp(-x^2)
Var abs(const Var& x);   // dx = dy .* sign(x), 0 at 0
Var div(const Var& a, const Var& b);  // a ./ b;  da = dy ./ b, db = -dy .* y ./ b
Var pow(const Var& a, const Var& b);  // powf;  da = dy .* b a^(b-1) (0 where b = 0),
                                      // db = dy .* y log(a) (0 where y = 0)
Var maximum(const Var& a, const Var& b);  // a where a >= b, else b; the gradient
Var minimum(const Var& a, const Var& b);  // follows the chosen operand (min: a <= b)
Var where(const Var& c, const Var& a, const Var& b);  // a where c > 0, else b;
                                                      // nothing reaches c
Var rowsum(const Var& x);  // [T, w] -> [T, 1], ascending column order;
                           // dx = dy broadcast along the row
Var rowmax(const Var& x);  // [T, w] -> [T, 1]; the gradient goes to the
                           // first largest entry of the row
Var broadcast_cols(const Var& c, size_t n);  // [T, 1] -> [T, n] copies;
                                             // dc = rowsum(dy)

// ---- T5's relative position bias (LOCAL_LLM_STACK 11 item 4.2; the flex family's
// position "t5_relative") ----
// T5's causal bucketing of the distance n = query position - key position >= 0 (HF
// _relative_position_bucket with bidirectional=False): n itself below num_buckets / 2, then
// logarithmically spaced up to max_distance, num_buckets - 1 from there. The table has
// max_distance + 1 entries; entry max_distance covers every larger distance. Computed once in
// binary64 (the table, not the logarithm, is what a model and an IR carry).
std::vector<int> t5_position_buckets(size_t num_buckets, size_t max_distance);
// One head of causal softmax attention with T5's additive bias: score(i, j) = (q_i . k_j) * scale
// + table(buckets[min(i - j, D - 1)], head), the product rounded, then the sum; keys after the
// query (and outside its seq_len block) are masked to exactly 0. q, k [T, dk], v [T, dv], table
// [num_buckets, heads] (HF's and llama.cpp's attn_rel_b layout), shared by every layer. Backward:
// the softmax-attention gradient plus d table(b, head) = the sum of d score over the pairs in
// bucket b. Host loops; FD-checked in test_flex.
Var relpos_attention(const Var& q, const Var& k, const Var& v, const Var& table, size_t head,
                     const std::vector<int>& buckets, float scale, size_t seq_len = 0);

// ---- Shaw-style relative positions (Shaw, Uszkoreit and Vaswani 2018, arXiv 1803.02155, eqs. 3-4; the
// flex family's position "shaw_relative"; keramos IR_SPEC 5.15 attention.relative_kv, revision 1.4) ----
// One head of causal softmax attention with learned relative key and value vectors, by the clipped
// distance n = min(i - j, max_distance) between the query row i and the key row j (inside its seq_len
// block): score(i, j) = ((q_i . k_j) + (q_i . rk[n])) * scale, each dot summed over the columns in
// ascending order and rounded, then the sum, then the product; out_i = (sum_j a_ij v_j) + (sum_j a_ij
// rv[n]), each sum over ascending j, then the addition (no rv: the first sum alone). rk [max_distance + 1,
// W] and rv [max_distance + 1, Wv] (null = keys only) hold one row per distance; this head reads columns
// col0 .. col0 + dk of rk (col0 .. col0 + dv of rv): col0 = 0 for vectors shared by every head (W = dk),
// head * dk for one set per head (W = heads * dk). Keys after the query are masked to exactly 0.
// Backward: the softmax-attention gradient through k_j + rk[n] and v_j + rv[n], with d rk[n] and d rv[n]
// the sums over the pairs at distance n. Host loops; FD-checked in tests/test_shaw_attention.cpp.
Var shaw_attention(const Var& q, const Var& k, const Var& v, const Var& rk, const Var& rv, size_t col0,
                   size_t max_distance, float scale, size_t seq_len = 0);

}  // namespace ops
}  // namespace microtorch
