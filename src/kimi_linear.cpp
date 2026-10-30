#include "microtorch/kimi_linear.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace microtorch::kimi {

// Feature map: elu(x) + 1
// - Always positive (enables normalization without masking negatives)
// - Smooth gradients (elu has derivative elu'(x) = 1 or exp(x))
// - Proven expressiveness in linear attention literature
Matrix KimiLinearAttention::feature_map(const Matrix& x) {
    // elu(x) = x if x > 0, else α(exp(x) - 1)
    // Standard: α = 1
    Matrix result(x);  // Copy constructor
    for (size_t i = 0; i < result.rows(); ++i) {
        for (size_t j = 0; j < result.cols(); ++j) {
            float val = result(i, j);
            if (val <= 0.f) {
                result(i, j) = std::exp(val) - 1.f;  // elu for val <= 0
            }
            // val > 0: result(i, j) = val (already set)
            result(i, j) += 1.f;  // +1 to ensure > 0
        }
    }
    return result;
}

// Gradient of feature map w.r.t. input
// d(elu(x) + 1)/dx = 1 if x > 0, else exp(x)
Matrix KimiLinearAttention::feature_map_grad(const Matrix& x) {
    Matrix grad = Matrix(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            float val = x(i, j);
            grad(i, j) = (val > 0.f) ? 1.f : std::exp(val);
        }
    }
    return grad;
}

// Cumulative sum along sequence dimension
// cumsum[t, :] = sum(x[0:t+1, :], axis=0)
Matrix KimiLinearAttention::cumsum(const Matrix& x) {
    Matrix result = Matrix(x.rows(), x.cols());
    // Initialize first row
    for (size_t j = 0; j < x.cols(); ++j) {
        result(0, j) = x(0, j);
    }
    // Cumulative sum: each row adds previous row
    for (size_t i = 1; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            result(i, j) = result(i - 1, j) + x(i, j);
        }
    }
    return result;
}

// Cumulative sum with causal mask (only past + current)
// Same as cumsum for auto-regressive (decoder) case
Matrix KimiLinearAttention::cumsum_causal(const Matrix& x) {
    return cumsum(x);  // In causal case, cumsum IS the causal operation
}

// Safe division: numerator / denominator with epsilon handling
// Avoids division by zero, handles near-zero cases
Matrix KimiLinearAttention::safe_divide(const Matrix& numerator, const Matrix& denominator,
                                        float eps) {
    Matrix result = Matrix(numerator.rows(), numerator.cols());
    for (size_t i = 0; i < numerator.rows(); ++i) {
        for (size_t j = 0; j < numerator.cols(); ++j) {
            float denom = denominator(i, j);
            // Clamp denominator to avoid division by very small numbers
            float safe_denom = std::max(std::abs(denom), eps);
            if (denom < 0.f) safe_denom = -safe_denom;
            result(i, j) = numerator(i, j) / safe_denom;
        }
    }
    return result;
}

// Forward: Linear-time attention
// Complexity: O(n * d) for this channelwise mixer; no cross-feature products.
// vs O(n² * d) for standard attention
Matrix KimiLinearAttention::forward(const Matrix& q, const Matrix& k, const Matrix& v,
                                    bool causal) {
    // Verify shapes: [batch*heads, seq, head_dim]
    // After attention ops, q/k/v should all be [seq, head_dim] per batch*head
    if (q.rows() == 0 || head_dim_ == 0 || q.rows() != k.rows() || k.rows() != v.rows()) {
        throw std::runtime_error("KimiLinear: seq_len mismatch in q, k, v");
    }
    if (q.cols() != k.cols() || k.cols() != head_dim_ || v.cols() != head_dim_) {
        throw std::runtime_error("KimiLinear: head_dim mismatch in q, k");
    }

    size_t seq_len = q.rows();

    // Step 1: Feature maps
    Matrix phi_q = feature_map(q);  // [seq, head_dim]
    Matrix phi_k = feature_map(k);  // [seq, head_dim]

    // Step 2+3: Numerator / denominator accumulation.
    //   causal:  prefix sums  -- position t sees keys/values 0..t
    //   full:    totals       -- every position sees the whole sequence
    // (The 2026-07-30 Debug-suite audit found `causal` was ignored and both
    // branches returned the causal result; test_kimi_causal_masking now
    // pins the difference.)
    Matrix numerator = Matrix(seq_len, head_dim_);
    Matrix denominator = Matrix(seq_len, head_dim_);
    if (causal) {
        for (size_t j = 0; j < head_dim_; ++j) {
            numerator(0, j) = phi_k(0, j) * v(0, j);
        }
        for (size_t t = 1; t < seq_len; ++t) {
            for (size_t j = 0; j < head_dim_; ++j) {
                numerator(t, j) = numerator(t - 1, j) + phi_k(t, j) * v(t, j);
            }
        }
        denominator = cumsum(phi_k);
    } else {
        for (size_t j = 0; j < head_dim_; ++j) {
            float num_total = 0.f, den_total = 0.f;
            for (size_t t = 0; t < seq_len; ++t) {
                num_total += phi_k(t, j) * v(t, j);
                den_total += phi_k(t, j);
            }
            for (size_t t = 0; t < seq_len; ++t) {
                numerator(t, j) = num_total;
                denominator(t, j) = den_total;
            }
        }
    }

    // Step 4: Output
    // output[t, :] = φ(q[t, :]) * numerator[t, :] / denominator[t, :]
    Matrix output = Matrix(seq_len, head_dim_);
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t j = 0; j < head_dim_; ++j) {
            float norm_denom = std::max(denominator(t, j), 1e-8f);
            output(t, j) = phi_q(t, j) * numerator(t, j) / norm_denom;
        }
    }

    return output;
}

// Backward: Compute gradients
// This is complex due to the cumulative sums and divisions
// We use reverse-mode AD: propagate grad_out backward through each operation
std::tuple<Matrix, Matrix, Matrix> KimiLinearAttention::backward(const Matrix& grad_out,
                                                                 const Matrix& q, const Matrix& k,
                                                                 const Matrix& v,
                                                                 const Matrix& attention_out,
                                                                 bool causal) const {
    if (q.rows() == 0 || head_dim_ == 0 || q.rows() != k.rows() || q.rows() != v.rows() ||
        q.cols() != head_dim_ || k.cols() != head_dim_ || v.cols() != head_dim_ ||
        grad_out.rows() != q.rows() || grad_out.cols() != head_dim_ ||
        attention_out.rows() != q.rows() || attention_out.cols() != head_dim_) {
        throw std::runtime_error("KimiLinear backward: shape mismatch or empty sequence");
    }
    size_t seq_len = q.rows();

    // Recompute forward activations (needed for backward)
    Matrix phi_q = feature_map(q);
    Matrix phi_k = feature_map(k);
    Matrix phi_q_grad = feature_map_grad(q);
    Matrix phi_k_grad = feature_map_grad(k);

    if (!causal) {
        Matrix grad_q(seq_len, head_dim_), grad_k(seq_len, head_dim_), grad_v(seq_len, head_dim_);
        for (size_t j = 0; j < head_dim_; ++j) {
            float numerator = 0.f, denominator = 0.f;
            for (size_t t = 0; t < seq_len; ++t) {
                numerator += phi_k(t, j) * v(t, j);
                denominator += phi_k(t, j);
            }
            const float denom = std::max(denominator, 1e-8f);
            float grad_numerator = 0.f, grad_denominator = 0.f;
            for (size_t t = 0; t < seq_len; ++t) {
                grad_q(t, j) = grad_out(t, j) * numerator / denom * phi_q_grad(t, j);
                grad_numerator += grad_out(t, j) * phi_q(t, j) / denom;
                // The clamp is constant below its boundary; equality uses the zero subgradient.
                if (denominator > 1e-8f)
                    grad_denominator -= grad_out(t, j) * phi_q(t, j) * numerator / (denom * denom);
            }
            for (size_t t = 0; t < seq_len; ++t) {
                grad_k(t, j) = (grad_numerator * v(t, j) + grad_denominator) * phi_k_grad(t, j);
                grad_v(t, j) = grad_numerator * phi_k(t, j);
            }
        }
        return std::make_tuple(grad_q, grad_k, grad_v);
    }

    // Recompute cumulative sums
    Matrix numerator = Matrix(seq_len, head_dim_);
    for (size_t j = 0; j < head_dim_; ++j) {
        numerator(0, j) = phi_k(0, j) * v(0, j);
    }
    for (size_t t = 1; t < seq_len; ++t) {
        for (size_t j = 0; j < head_dim_; ++j) {
            numerator(t, j) = numerator(t - 1, j) + phi_k(t, j) * v(t, j);
        }
    }

    Matrix denominator = cumsum(phi_k);

    // Gradients w.r.t. q, k, v
    Matrix grad_q = Matrix(seq_len, head_dim_);
    Matrix grad_k = Matrix(seq_len, head_dim_);
    Matrix grad_v = Matrix(seq_len, head_dim_);

    // Gradient of output = φ(q) * numerator / denominator
    // d(output)/d(φ(q)) = numerator / denominator
    // d(output)/d(numerator) = φ(q) / denominator
    // d(output)/d(denominator) = -φ(q) * numerator / (denominator²)

    // Backward through division and multiplication
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t j = 0; j < head_dim_; ++j) {
            float denom = std::max(denominator(t, j), 1e-8f);
            float denom_sq = denom * denom;

            // grad w.r.t. φ(q)
            float grad_phi_q = grad_out(t, j) * numerator(t, j) / denom;

            // grad w.r.t. φ(q) -> grad w.r.t. q
            grad_q(t, j) = grad_phi_q * phi_q_grad(t, j);

            // grad w.r.t. numerator will be accumulated from all positions
            // (backward through cumsum)
        }
    }

    // Backward through numerator and denominator (cumsum operations)
    // Each prefix depends on all previous keys; the suffix accumulation below
    // propagates every query contribution back to each affected key/value.

    Matrix grad_numerator = Matrix(seq_len, head_dim_);
    Matrix grad_denominator = Matrix(seq_len, head_dim_);

    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t j = 0; j < head_dim_; ++j) {
            float denom = std::max(denominator(t, j), 1e-8f);
            float denom_sq = denom * denom;

            // d(output[t,j])/d(numerator[t,j]) = φ(q[t,j]) / denom
            grad_numerator(t, j) = grad_out(t, j) * phi_q(t, j) / denom;

            // d(output[t,j])/d(denom[t,j]) = -φ(q[t,j]) * num / denom²
            grad_denominator(t, j) = denominator(t, j) > 1e-8f ? -grad_out(t, j) * phi_q(t, j) *
                                                                     numerator(t, j) / denom_sq
                                                               : 0.f;
        }
    }

    // Backward through cumsum operations.
    // cumsum_backward: if y = cumsum(x), then grad_x[t] = sum(grad_y[t:]),
    // so the suffix from t+1 must fold into position t BEFORE position t's
    // gradient is consumed. (The original ordering used the pre-suffix
    // value -- correct only at the last position; the SRD module-level FD
    // gradcheck caught the 30% composite error, 2026-07-30.)
    for (size_t t = seq_len - 1; t < seq_len; --t) {  // Reverse iteration
        for (size_t j = 0; j < head_dim_; ++j) {
            // Fold in future positions first (suffix sum).
            if (t + 1 < seq_len) {
                grad_denominator(t, j) += grad_denominator(t + 1, j);
                grad_numerator(t, j) += grad_numerator(t + 1, j);
            }

            // grad_k contribution from denominator cumsum
            grad_k(t, j) += grad_denominator(t, j) * phi_k_grad(t, j);

            // grad_v contribution from numerator
            grad_v(t, j) += grad_numerator(t, j) * phi_k(t, j);

            // grad_k contribution from numerator (φ(k_t) * v_t term)
            grad_k(t, j) += grad_numerator(t, j) * phi_k_grad(t, j) * v(t, j);
        }
    }

    return std::make_tuple(grad_q, grad_k, grad_v);
}

}  // namespace microtorch::kimi
