#include "microtorch/autograd.hpp"
#include "microtorch/device_cache.hpp"

#include <stdexcept>
#include <unordered_set>

namespace microtorch {

namespace detail {
std::atomic<size_t> g_live_vars{0};
}

size_t live_variables() {
    return detail::g_live_vars.load();
}

void Variable::accumulate(const Matrix& g) {
    // B2.1b: g may be a deferred device-fresh result (a gemm gradient);
    // grad += g is host arithmetic, so download first. This makes grads
    // host-authoritative for the whole step — full grad deferral is
    // B2.3's device-side accumulate.
    device::materialize(g);
    if (grad.rows() == 0) {
        grad = Matrix(data.rows(), data.cols());  // zero-filled by ctor
    }
    if (g.rows() != grad.rows() || g.cols() != grad.cols()) {
        throw std::runtime_error("accumulate: gradient shape mismatch");
    }
    grad += g;
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
    root->accumulate(seed);

    std::vector<Variable*> order;  // post-order: leaves first
    topo(root, order);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {  // root first
        if ((*it)->backward_fn && (*it)->grad.rows() != 0) {
            (*it)->backward_fn();
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
