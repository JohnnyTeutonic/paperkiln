#include "microtorch/autograd.hpp"
#include "microtorch/device_cache.hpp"

#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace microtorch {

namespace detail {
std::atomic<size_t> g_live_vars{0};
}

size_t live_variables() {
    return detail::g_live_vars.load();
}

void Variable::accumulate(const Matrix& g) {
    if (grad.rows() == 0) {
        grad = Matrix(data.rows(), data.cols());  // zero-filled by ctor
    }
    if (g.rows() != grad.rows() || g.cols() != grad.cols()) {
        throw std::runtime_error("accumulate: gradient shape mismatch");
    }
    // B2.3c: when either side is device-fresh (a deferred gemm gradient,
    // or a grad already accumulated on-device this window), stay on
    // device — axpy through the value cache, no download. Every backward
    // that host-reads a grad materializes at entry (the 12-site audit),
    // and step_end() materializes what the optimizer/clip read between
    // windows, so host reads remain safe. Otherwise the host add runs
    // with the B2.1b materialize choke, bit-identical to before.
    const bool device_side = device::host_stale(g) || device::host_stale(grad);
    if (!device_side || !device::devops::axpy(grad, 1.0f, g)) {
        device::materialize(grad);
        device::materialize(g);
        grad += g;
    }
}

void Variable::accumulate(Matrix&& g) {
    accumulate(static_cast<const Matrix&>(g));
    device::discard(g);  // the temp dies with this statement — its
                         // deferred entry must not outlive it
}

namespace {

// Post-order DFS over parents. Iterative, because a deep tape (15k-step
// training graphs are the eventual customer) must not be bounded by the C++
// call stack.
void topo(const Var& root, std::vector<Variable*>& order) {
    std::unordered_set<Variable*> seen;
    std::vector<std::pair<Variable*, size_t>> stack{{root.get(), 0}};
    seen.insert(root.get());
    while (!stack.empty()) {
        auto& [node, next] = stack.back();
        if (next < node->parents.size()) {
            Variable* p = node->parents[next++].get();
            if (seen.insert(p).second) stack.emplace_back(p, 0);
        } else {
            order.push_back(node);
            stack.pop_back();
        }
    }
}

thread_local bool g_grad_enabled = true;

}  // namespace

void backward(const Var& root) {
    if (root->data.rows() != 1 || root->data.cols() != 1) {
        throw std::runtime_error("backward: root must be a [1,1] scalar (compose with mean/sum)");
    }
    Matrix seed(1, 1);
    seed(0, 0) = 1.0f;
    root->accumulate(std::move(seed));

    std::vector<Variable*> order;  // post-order: leaves first
    topo(root, order);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {  // root first
        if ((*it)->backward_fn && (*it)->grad.rows() != 0) {
            (*it)->backward_fn();
            // B2.3c: a non-leaf's grad has now been fully consumed (topo
            // order guarantees every consumer accumulated into it before
            // its own backward ran). Drop any deferred copy so step_end
            // neither downloads a dead intermediate nor can ever write a
            // freed buffer through one. Leaves (params) keep theirs —
            // the optimizer reads them after the window closes.
            if (!(*it)->is_leaf()) device::discard((*it)->grad);
        }
    }
}

Var checkpoint(const std::function<Var(const Var&)>& fn, const Var& x) {
    if (!grad_enabled()) return fn(x);  // eval: nothing to rematerialize

    Matrix out_data;
    {
        NoGrad ng;
        out_data = fn(x)->data;  // inner tape never exists
    }
    // requires_grad is set unconditionally: fn may capture parameters the
    // segment's single recorded edge (x) cannot see, and under training
    // they need the backward to fire.
    Var out = make_var(std::move(out_data), true);
    out->parents = {x};
    Variable* self = out.get();
    auto fn_copy = fn;
    out->backward_fn = [self, fn_copy]() {
        const Var& xin = self->parents[0];
        // Recompute the segment on a detached leaf copy of the input.
        Var x2 = make_var(xin->data, /*requires_grad=*/true);
        Var y2 = fn_copy(x2);
        if (y2->data.rows() != self->grad.rows() || y2->data.cols() != self->grad.cols()) {
            throw std::runtime_error("checkpoint: recomputed output shape mismatch");
        }
        y2->accumulate(self->grad);
        // Backprop through the fresh subgraph only. Parameters inside fn
        // are shared with the real model, so their grads land in place.
        std::vector<Variable*> order;
        topo(y2, order);
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            if ((*it)->backward_fn && (*it)->grad.rows() != 0) (*it)->backward_fn();
        }
        if (xin->requires_grad && x2->grad.rows() != 0) xin->accumulate(x2->grad);
    };
    return out;
}

void zero_grad(const std::vector<Var>& vars) {
    for (const auto& v : vars) {
        if (v->grad.rows() != 0) v->grad.fill(0.0f);
    }
}

bool grad_enabled() {
    return g_grad_enabled;
}

NoGrad::NoGrad() : prev_(g_grad_enabled) {
    g_grad_enabled = false;
}
NoGrad::~NoGrad() {
    g_grad_enabled = prev_;
}

}  // namespace microtorch
