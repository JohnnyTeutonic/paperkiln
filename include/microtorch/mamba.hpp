#pragma once
// Phase 3c: Mamba State-Space Model
// Alternative to transformers: an RNN-like recurrence trained through time
//
// Reference: Mamba paper (2024), S4 family foundation
// Core equation: dx/dt = A·x + B·u
// Discrete form: x[t+1] = A·x[t] + B·u[t]
//
// Key advantages:
// - O(n) inference complexity (vs O(n²) for standard attention)
// - O(1) memory per step (no attention matrix)
// - Training through time by a sequential scan (no parallel scan yet)
// - Better efficiency on long sequences
//
// Structure:
// 1. Input projection (D_in → D_state)
// 2. Learnable A, B matrices (state-space parameters)
// 3. Gating mechanism (output modulation)
// 4. Output projection (D_state → D_out)
//
// This is the foundation-only version; full Mamba includes:
// - Selective S4 (input-dependent A, B)
// - Complex numbers in state-space
// - Hardware-efficient scan algorithms

#include "microtorch/autograd.hpp"
#include "microtorch/nn.hpp"

namespace microtorch {
namespace mamba {

// S4 Layer: Simplified state-space model
// Baseline for Mamba: x[t+1] = A·x[t] + B·u[t], y[t] = C·x[t] + D·u[t]
class S4Layer : public nn::Module {
public:
    // d_model: input/output dimension
    // d_state: state vector dimension (typically d_model * expansion_factor)
    S4Layer(size_t d_model, size_t d_state = 64, unsigned seed = 0);

    // Forward: process sequence
    // Input: [T, d_model]
    // Output: [T, d_model]
    Var forward(const Var& x) const;

    // State dimensions
    size_t d_model() const { return d_model_; }
    size_t d_state() const { return d_state_; }

private:
    size_t d_model_, d_state_;

    // State-space matrices
    std::shared_ptr<nn::Linear> proj_in;   // d_model → d_state
    std::shared_ptr<nn::Linear> proj_out;  // d_state → d_model
    Var A, B, C, D;                        // State-space parameters

    // Gating
    std::shared_ptr<nn::Linear> gate_proj;  // d_model → d_model
};

// MambaBlock: Full Mamba block with normalization
class MambaBlock : public nn::Module {
public:
    MambaBlock(size_t d_model, size_t d_state = 64, unsigned seed = 0);

    // Forward: Pre-LN normalization + S4 + output gate
    Var forward(const Var& x) const;

private:
    std::shared_ptr<nn::LayerNorm> norm;
    std::shared_ptr<S4Layer> s4;
    std::shared_ptr<nn::Linear> out_gate;
};

// MambaModel: Stack of Mamba blocks (baseline for comparison)
class MambaModel : public nn::Module {
public:
    MambaModel(size_t vocab_size, size_t d_model, size_t n_layers, size_t d_state = 64,
               unsigned seed = 0);

    Var forward(const Var& x) const;  // [T, d_model]

private:
    std::shared_ptr<nn::Embedding> embed;
    std::vector<std::shared_ptr<MambaBlock>> blocks;
    std::shared_ptr<nn::LayerNorm> norm_final;
    std::shared_ptr<nn::Linear> lm_head;
};

}  // namespace mamba
}  // namespace microtorch
