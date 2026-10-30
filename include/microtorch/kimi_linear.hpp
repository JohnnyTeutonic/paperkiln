#pragma once
// Legacy-named channelwise ELU+1 normalised mixer, not the published Kimi Linear/KDA architecture.
// For each feature j: y[t,j] = phi(q[t,j]) * sum_s(phi(k[s,j])*v[s,j]) / max(sum_s phi(k[s,j]),
// 1e-8). The causal form sums s<=t; the non-causal form sums the whole independent sequence. This
// is O(T*d), without cross-feature QK products. Float exp(x)-1+1 can cancel for very negative x;
// backward uses the ordinary smooth ELU derivative, not a derivative of floating-point rounding.

#include <cmath>
#include <memory>
#include "microtorch/primitives.hpp"

namespace microtorch::kimi {

// Linear-time attention forward + backward
// Input shapes: q, k, v all [seq, head_dim], one independent sequence.
// Output: [seq, head_dim]; ops::kimi_attention splits stacked batches into blocks.
//
// Mechanism:
// 1. Feature map: transform queries and keys through feature function φ
// 2. Numerator: Σ φ(k_i) * v_i (cumulative sum over sequence)
// 3. Denominator: Σ φ(k_i) (normalization)
// 4. Output: φ(q) * (numerator / denominator)
//
// This avoids materializing the [seq, seq] attention matrix entirely.
class KimiLinearAttention {
public:
    // Initialize with head dimension
    explicit KimiLinearAttention(size_t head_dim) : head_dim_(head_dim) {}

    // Forward: linear-time attention
    // q, k, v: [seq, head_dim]
    // causal: if true, use causal mask (tokens can only attend to past)
    // Returns: mixer output [seq, head_dim]
    Matrix forward(const Matrix& q, const Matrix& k, const Matrix& v, bool causal = true);

    // Backward: compute gradients w.r.t. q, k, v
    // causal must match forward (default true preserves existing callers).
    // grad_out: gradient from upstream [seq, head_dim]
    // q, k, v, attention_out: forward pass activations
    // Returns: (grad_q, grad_k, grad_v)
    std::tuple<Matrix, Matrix, Matrix> backward(const Matrix& grad_out, const Matrix& q,
                                                const Matrix& k, const Matrix& v,
                                                const Matrix& attention_out,
                                                bool causal = true) const;

private:
    size_t head_dim_;

    // Feature map: elu(x) + 1 (always positive, smooth gradients)
    // This is the key difference from standard attention
    // Maintains expressiveness while enabling linear complexity
    static Matrix feature_map(const Matrix& x);
    static Matrix feature_map_grad(const Matrix& x);

    // Cumulative sum: [seq, head_dim] -> [seq, head_dim]
    // cumsum[t, :] = sum(values[0:t+1, :], axis=0)
    static Matrix cumsum(const Matrix& x);

    // Causal mask: cumsum only up to current position
    // (prevents tokens from attending to future)
    static Matrix cumsum_causal(const Matrix& x);

    // Elementwise division: safer than direct /
    // Handles near-zero denominators
    static Matrix safe_divide(const Matrix& numerator, const Matrix& denominator,
                              float eps = 1e-8f);
};

}  // namespace microtorch::kimi
