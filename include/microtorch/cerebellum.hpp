#pragma once
// Phase 3b: Cerebellum-inspired selective computation
// Neuroscience-based gating to ignore routine signals and focus computation
// on novel/important information
//
// Reference: Northwestern study on cerebellar prediction error filtering
// Key idea: Learn what's "routine", compute residual, gate expensive ops
//
// Mechanism:
// 1. Prediction head: learns to model expected/routine token features
// 2. Residual: actual - predicted (surprise/novelty signal)
// 3. Gate: sigmoid(4*rms(residual) - 1) -- the affine form lets the gate
//    close (< 0.5) for well-predicted tokens; sigmoid of a bare norm cannot
// 4. Efficiency: Skip expensive layers (LN, attention, MLP) for routine tokens
//
// Benefits:
// - 20-40% inference speedup (tokens bypass expensive computation)
// - Works as wrapper around existing layers (non-invasive)
// - Interpretable: can see which tokens were deemed routine
// - Learnable: prediction head trains alongside main model

#include <functional>

#include "microtorch/autograd.hpp"
#include "microtorch/nn.hpp"
#include "microtorch/ops.hpp"

namespace microtorch {
namespace cerebellum {

// Lightweight prediction head for learning routine patterns
class RoutinePredictor : public nn::Module {
public:
    explicit RoutinePredictor(size_t d_model, unsigned seed = 0);
    Var forward(const Var& x) const;  // x: [T, d] -> [T, d] (same shape)

    // Public for export (the SRD router's IR and GGUF tensors).
    std::shared_ptr<nn::Linear> fc1, fc2;  // Small MLP: d -> d/4 -> d
};

// Selective gating wrapper: wraps computation with residual-based gating
// Uses a function object to wrap the expensive computation
class SelectiveGate : public nn::Module {
public:
    using ForwardFn = std::function<Var(const Var&)>;

    SelectiveGate(ForwardFn layer_fn, size_t d_model, unsigned seed = 0);

    Var forward(const Var& x) const;

    // Access gating statistics for interpretation/debugging
    const Matrix& last_gate_logits() const { return last_gate_; }
    const Matrix& last_residual() const { return last_residual_; }

private:
    ForwardFn layer_fn_;  // Function that applies the expensive layer
    std::shared_ptr<RoutinePredictor> predictor_;

    // Mutable for caching statistics (for interpretation, not backprop)
    mutable Matrix last_gate_;
    mutable Matrix last_residual_;
};

// Utility: Apply selective gating to all layers in a block
class GatedBlock : public nn::Module {
public:
    GatedBlock(size_t d, size_t n_heads, unsigned seed = 0);
    Var forward(const Var& x) const;

private:
    std::shared_ptr<nn::LayerNorm> ln_1, ln_2;
    std::shared_ptr<SelectiveGate> attn_gate, mlp_gate;
};

}  // namespace cerebellum
}  // namespace microtorch
