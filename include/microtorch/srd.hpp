#pragma once
// V1: Surprise-Routed Density attention (docs/SPARSE_ATTENTION.md section 2).
//
// Per-query density routed by prediction residual instead of attention-
// affinity scores (the NSA/MoBA family's router). A small predictor learns
// what "routine" hidden states look like; its residual sets a gate
//
//     g[t] = sigmoid(GATE_SCALE * rms(x[t] - predict(x)[t]) + GATE_BIAS)
//
// and each query blends two attention paths computed from ONE shared qkv
// projection (so the mechanism difference is isolated):
//
//     out[t] = g[t] * ExactAttention[t] + (1 - g[t]) * KimiLinear[t]
//
// Training is soft and fully differentiable end to end -- gradients reach
// the predictor through the gate (rms_row -> sigmoid -> mul_col are all
// gradchecked tape ops), and both paths through the blend. Inference
// hardens the gate (g > tau -> exact for that query), giving
// O(rho n^2 d + n d^2) with rho = surprise rate.
//
// mean(gate()) is exposed for an auxiliary sparsity loss: total_loss =
// task_loss + lambda * mean_gate prices density.
#include "microtorch/cerebellum.hpp"
#include "microtorch/nn.hpp"

namespace microtorch {
namespace nn {

class SurpriseRoutedAttention : public Module {
public:
    static constexpr float GATE_SCALE = 4.0f, GATE_BIAS = -1.0f;

    SurpriseRoutedAttention(size_t d, size_t n_heads, unsigned seed = 0);

    // seq_len > 0: x rows are B stacked sequences; both attention paths
    // are block-isolated. The router is row-wise and batch-agnostic.
    // (Falsifier caveat: shuffle_predictor permutes across the whole
    // stack; research falsifier runs use single-sequence forwards.)
    Var forward(const Var& x, size_t seq_len = 0) const;

    // The gate from the LAST forward, as a tape Var [T, 1]: feed
    // ops::mean(gate()) into the loss to price density. Empty before the
    // first forward.
    Var gate() const { return last_gate_; }

    std::shared_ptr<Linear> c_attn, c_proj;  // shared by both paths
    std::shared_ptr<cerebellum::RoutinePredictor> predictor;
    size_t H, dk;

    // FALSIFIER (docs/SPARSE_ATTENTION.md protocol): when set, the predictor
    // sees a row-permuted copy of x, so the gate keeps its distribution
    // but carries no information aligned to the query. If training
    // quality holds under this, the gate was not using surprise.
    bool shuffle_predictor = false;
    // Inference-time gate override for the matched-density controls
    // (experiments/SRD_PREREG_R2.md P5): when non-empty and sized [T], it replaces
    // the router's gate verbatim (1 = exact path, 0 = linear path), so
    // the SAME trained weights can be evaluated under SRD-chosen,
    // random, or positional gating at equal density. Empty = normal.
    mutable std::vector<float> forced_gate;

private:
    mutable Var last_gate_;
    mutable unsigned long long shuffle_ctr_ = 0;
};

}  // namespace nn
}  // namespace microtorch
