#pragma once
// Op-level forward + grad_fn pairs over the PHASE0 op set. Forwards call
// the canonical transformer_core entry points (primitives.hpp); backwards
// are the audited manual formulas. This is the op subset that the tape's
// first gate (tests/test_gradcheck.cpp) verifies; the rest of the audited
// set (layernorm, CE loss, embedding) lands in phase 1b with Modules.
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

// ---- phase 2a additions (DiT's dependencies) ----
Var mul_row(const Var& x, const Var& r);  // x .* broadcast row r [1, cols];
                                          // dr = colsum(dY .* x) -- the
                                          // multiplicative twin of add_bias
Var silu(const Var& x);                   // x * sigmoid(x) (DiT's nonlinearity
                                          // for the conditioning MLP)

// ---- phase 2b additions (Llama-family dependencies) ----
Var rmsnorm(const Var& x, const Var& w);  // RMS normalization: x / RMS(x) * w
                                          // (no bias, no mean centering);
                                          // w: [1, d], learned scale per feature
Var apply_rope(const Var& qk, const std::vector<int>& pos, float theta_base,
               size_t head_dim);  // Rotary embeddings to query/key rows
                                  // qk: [T, 3*d] (fused qkv);
                                  // returns [T, 3*d] with RoPE applied
                                  // to q and k head_dim subspaces

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

}  // namespace ops
}  // namespace microtorch
