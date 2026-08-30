#pragma once
// The autograd tape -- docs/DESIGN.md phase 1a, the one genuinely new component.
//
// Reverse-mode at op granularity (settled in docs/history/PHASE0_KERNEL_AUDIT.md
// section 6): a Variable owns a Matrix plus, when it was produced by an op
// under grad, the closure that scatters its gradient to its parents.
// backward() topologically sorts the tape from a scalar root and runs each
// closure once. Ownership is one-directional -- children hold shared_ptrs
// to parents, never the reverse -- so the graph is a DAG of shared_ptrs
// with no cycles to leak.
#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include "microtorch/device_cache.hpp"
#include "microtorch/primitives.hpp"

namespace microtorch {

class Variable;
using Var = std::shared_ptr<Variable>;

namespace detail {
extern std::atomic<size_t> g_live_vars;  // diagnostic; see live_variables()
}

class Variable {
public:
    Matrix data;
    Matrix grad;  // sized+zeroed on first accumulate()
    bool requires_grad = false;

    // Tape node; empty for leaves. backward_fn reads this->grad and
    // accumulates into parents' grads. It captures `this` raw -- safe
    // because the closure is a member of this Variable -- and the parents
    // as shared_ptrs, which is what keeps the upstream graph alive.
    std::vector<Var> parents;
    std::function<void()> backward_fn;

    // Phase B2 (docs/CUDA_PHASE_B2.md): device-resident copy of `data`,
    // OWNED by this Variable so the buffer dies exactly when the tensor
    // dies (no recycled-host-pointer aliasing). nullptr until a
    // step-resident gemm caches here; always nullptr in CPU builds.
    device::DevState* dev = nullptr;

    explicit Variable(Matrix d, bool rg = false) : data(std::move(d)), requires_grad(rg) {
        ++detail::g_live_vars;
    }
    ~Variable() {
        // B2.3c groundwork: a value-cache entry must never outlive its
        // host buffer (the B2.2 lifetime rule made STRUCTURAL — with
        // this, step_end()'s materialize_all can never write into freed
        // memory through a dead Variable, and no dangling cache key can
        // greet a recycled address). No-op on CPU builds.
        device::discard(data);
        device::discard(grad);
        device::release_devstate(dev);
        --detail::g_live_vars;
    }
    Variable(const Variable&) = delete;
    Variable& operator=(const Variable&) = delete;

    bool is_leaf() const { return parents.empty(); }
    void accumulate(const Matrix& g);  // grad += g (sizing on first use)
};

// Diagnostic: Variables currently alive (tape nodes + leaves). The memory
// receipt for activation checkpointing, and the raw signal a future
// fit-to-memory budgeter needs.
size_t live_variables();

inline Var make_var(Matrix data, bool requires_grad = false) {
    return std::make_shared<Variable>(std::move(data), requires_grad);
}

// Reverse pass from a scalar root ([1,1] -- asserted, because seeding a
// non-scalar with ones silently computes a sum-vector-Jacobian the caller
// probably did not mean).
void backward(const Var& root);

void zero_grad(const std::vector<Var>& vars);

// Activation checkpointing (rematerialization). Runs fn(x) WITHOUT
// recording the inner tape — only x and the output survive the forward.
// On the backward pass the segment is re-run under grad on a detached
// copy of x, the output gradient is pushed through that fresh subgraph
// (parameters captured inside fn accumulate their grads exactly as in
// the uncheckpointed case), and x receives its input gradient. Cost: one
// extra forward per segment; saving: the segment's intermediate
// activations never outlive the forward pass.
//
// fn must be RECOMPUTATION-DETERMINISTIC: calling it twice on the same
// input must run the same ops on the same values. nn::Dropout draws a
// fresh seed per forward and therefore must not appear inside a
// checkpointed segment.
Var checkpoint(const std::function<Var(const Var&)>& fn, const Var& x);

// no_grad scope. Ops record no tape nodes while one of these is alive.
bool grad_enabled();
class NoGrad {
public:
    NoGrad();
    ~NoGrad();
    NoGrad(const NoGrad&) = delete;
    NoGrad& operator=(const NoGrad&) = delete;

private:
    bool prev_;
};

}  // namespace microtorch
