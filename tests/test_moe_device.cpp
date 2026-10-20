// The mixture-of-experts dispatch ops on the device: ops::gather_rows,
// scatter_add_rows (one input and the multi-input combine), gather_elems,
// logsumexp_row and mul_col with the device op set on (CUDA build,
// MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1) against the host loops,
// forward and backward, on random inputs with repeated indices (scatters
// that must add in the host's order), write-through and under deferred
// downloads: every output and gradient bit for bit, including logsumexp
// (the host's serial double sum) and mul_col's dc (a serial per-row sum,
// unfused like the host build). Then a whole MoEFFN, top-1 raw and top-2
// renormalised and raw, with router logits full of ties (identical router
// columns and an all-zero row: ties go to the lower expert index): the same
// routing, and outputs and gradients to rounding (the gate softmax is the
// existing device softmax_row, a parallel reduction). On a CPU-only build
// (or with the op set off) the legs run the host path against itself.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/moe.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

namespace {

Matrix randn(size_t r, size_t c, unsigned seed, float scale = 1.0f) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, scale);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(g);
    return m;
}

bool on_device() { return device::device_ops_enabled() && device::get() == device::Device::CUDA; }

struct Diff {
    double worst = 0.0;  // max |a - b| / max |b| over the tensor
    size_t mismatched = 0, total = 0;
};

void cmp(const Matrix& a, const Matrix& b, Diff& d) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols());
    double scale = 0.0;
    for (size_t i = 0; i < b.rows(); ++i)
        for (size_t j = 0; j < b.cols(); ++j) scale = std::max(scale, (double)std::abs(b(i, j)));
    for (size_t i = 0; i < b.rows(); ++i)
        for (size_t j = 0; j < b.cols(); ++j) {
            const float x = a(i, j), y = b(i, j);
            d.worst = std::max(d.worst, std::abs((double)x - y) / (scale + 1e-30));
            d.mismatched += std::memcmp(&x, &y, sizeof x) != 0;
            ++d.total;
        }
}

// Run fn with the op set on or off, optionally inside a deferred step
// window (step_end is the materialize boundary), and restore the switch.
template <class F>
void with_ops(bool dev, bool windowed, F fn) {
    device::set_device_ops(dev);
    if (windowed) {
        device::set_step_residency(true);
        device::set_defer_downloads(true);
        device::step_begin();
    }
    fn();
    if (windowed) {
        device::step_end();
        device::set_defer_downloads(false);
        device::set_step_residency(false);
    }
    device::set_device_ops(true);
}

std::vector<int> rand_idx(size_t n, int hi, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> u(0, hi - 1);
    std::vector<int> v(n);
    for (int& i : v) i = u(g);
    return v;
}

// Each op alone, forward and backward (a random cotangent through
// mean(out .* G)), device against host.
void test_ops(bool windowed) {
    const size_t T = 1024, d = 384, E = 8;
    const Matrix X = randn(T, d, 11), L = randn(T, E, 12, 3.0f), Cc = randn(T, 1, 13);
    // Repeated indices on purpose: gathers whose backward must add, a
    // scatter with several sources per row and untouched rows.
    const std::vector<int> gi = rand_idx(1500, static_cast<int>(T), 14);
    const std::vector<int> si = rand_idx(T, 700, 15), si2 = rand_idx(300, 700, 19);
    std::vector<int> er = rand_idx(3000, static_cast<int>(T), 16), ec = rand_idx(3000, static_cast<int>(E), 17);
    struct Res {
        Matrix g, s, e, l, m, gx, gl, gc, gs, s2, gs2;
    };
    auto run = [&](bool dev, bool win) {
        Res r;
        with_ops(dev, win, [&] {
            Var x = make_var(X, true), lg = make_var(L, true), c = make_var(Cc, true);
            Var sx = make_var(randn(T, d, 18), true);
            Var g = ops::gather_rows(x, gi);                  // [1500, d]
            Var s = ops::scatter_add_rows(sx, si, 700);       // [700, d]
            Var sy = make_var(randn(300, d, 20), true);
            Var s2 = ops::scatter_add_rows({sx, sy}, {si, si2}, 700);  // the multi-input combine
            Var e = ops::gather_elems(lg, er, ec, 3);         // [1000, 3]
            Var l = ops::logsumexp_row(lg);                   // [T, 1]
            Var m = ops::mul_col(x, c);                       // [T, d]
            Var loss = ops::mean(ops::mul(g, make_var(randn(1500, d, 21))));
            loss = ops::add(loss, ops::mean(ops::mul(s, make_var(randn(700, d, 22)))));
            loss = ops::add(loss, ops::mean(ops::mul(s2, make_var(randn(700, d, 26)))));
            loss = ops::add(loss, ops::mean(ops::mul(e, make_var(randn(1000, 3, 23)))));
            loss = ops::add(loss, ops::mean(ops::mul(l, make_var(randn(T, 1, 24)))));
            loss = ops::add(loss, ops::mean(ops::mul(m, make_var(randn(T, d, 25)))));
            backward(loss);
            device::materialize_all();
            r = {g->data, s->data, e->data, l->data, m->data, x->grad, lg->grad, c->grad, sx->grad, s2->data, sy->grad};
        });
        return r;
    };
    const Res a = run(true, windowed), b = run(false, false);
    Diff exact, lse, dlg, dc;
    cmp(a.g, b.g, exact), cmp(a.s, b.s, exact), cmp(a.e, b.e, exact), cmp(a.m, b.m, exact);
    cmp(a.gx, b.gx, exact), cmp(a.gs, b.gs, exact), cmp(a.s2, b.s2, exact), cmp(a.gs2, b.gs2, exact);
    cmp(a.l, b.l, lse), cmp(a.gl, b.gl, dlg), cmp(a.gc, b.gc, dc);
    std::printf("  ops%s: gathers/scatters/mul_col fwd + dX: %zu/%zu bits differ (worst %.3e)\n",
                windowed ? " under deferred downloads" : "", exact.mismatched, exact.total, exact.worst);
    std::printf("    logsumexp fwd worst %.3e (%zu/%zu bits differ); logits grad (lse + gather_elems) worst %.3e "
                "(%zu/%zu); mul_col dc worst %.3e (%zu/%zu)\n",
                lse.worst, lse.mismatched, lse.total, dlg.worst, dlg.mismatched, dlg.total, dc.worst, dc.mismatched,
                dc.total);
    CHECK(exact.mismatched == 0);
    CHECK(lse.mismatched == 0 && dlg.mismatched == 0 && dc.mismatched == 0);
}

// A whole MoE FFN with ties in the router logits. Router columns 1 and 3
// are identical (every token ties those two experts), and token 0's input
// row is zero (all E logits tie at 0: it must go to experts 0..k-1).
void test_moe_layer(size_t k, bool renorm, bool windowed) {
    const size_t T = 512, d = 64, dff = 96, E = 8;
    nn::MoEConfig mc;
    mc.experts = E, mc.top_k = k, mc.renormalize = renorm;
    nn::MoEFFN m(d, dff, mc, 7);
    Matrix& W = m.router->W->data;  // [d, E]
    W = randn(d, E, 31, 0.5f);
    for (size_t i = 0; i < d; ++i) W(i, 3) = W(i, 1);
    for (const auto& p : m.parameters())
        if (p.get() != m.router->W.get()) p->data = randn(p->data.rows(), p->data.cols(), 40 + p->data.cols(), 0.2f);
    Matrix X = randn(T, d, 32);
    for (size_t j = 0; j < d; ++j) X(0, j) = 0.0f;
    const Matrix G = randn(T, d, 33);
    struct Res {
        Matrix y, gx, lb, z;
        std::vector<Matrix> gp;
        std::vector<std::vector<double>> frac;
    };
    auto run = [&](bool dev, bool win) {
        Res r;
        with_ops(dev, win, [&] {
            for (const auto& p : m.parameters()) p->grad = Matrix();
            Var x = make_var(X, true);
            nn::MoEAux aux;
            Var y = m.forward(x, &aux);
            Var loss = ops::add(ops::mean(ops::mul(y, make_var(G))), ops::add(ops::scale(aux.lb, 0.01f), ops::scale(aux.z, 0.001f)));
            backward(loss);
            device::materialize_all();
            r.y = y->data, r.gx = x->grad, r.lb = aux.lb->data, r.z = aux.z->data, r.frac = aux.frac;
            for (const auto& p : m.parameters()) r.gp.push_back(p->grad);
        });
        return r;
    };
    const Res a = run(true, windowed), b = run(false, false);
    CHECK(a.frac == b.frac);
    Diff dy, dg, dp, aux;
    cmp(a.y, b.y, dy), cmp(a.gx, b.gx, dg), cmp(a.lb, b.lb, aux), cmp(a.z, b.z, aux);
    CHECK(a.gp.size() == b.gp.size());
    for (size_t i = 0; i < a.gp.size(); ++i)
        if (b.gp[i].rows() != 0) cmp(a.gp[i], b.gp[i], dp);
    // Selection is the host's (from the downloaded logits) on both paths, so
    // equal fractions per layer and expert are the expected outcome; the
    // tie rule is checked below for k = 1.
    std::printf("  MoEFFN E=8 k=%zu %s%s: routing identical; y worst %.3e (%zu/%zu bits differ), dX worst %.3e "
                "(%zu/%zu), params worst %.3e (%zu/%zu), aux worst %.3e\n",
                k, renorm ? "renorm" : "raw", windowed ? " under deferred downloads" : "", dy.worst, dy.mismatched,
                dy.total, dg.worst, dg.mismatched, dg.total, dp.worst, dp.mismatched, dp.total, aux.worst);
    // Not bit for bit: the gates come from softmax_row, whose device kernel
    // (an existing op) reduces in parallel. Everything downstream of the
    // gates is exact, so the output differs by the gates' rounding only.
    CHECK(dy.worst < 1e-6 && dg.worst < 1e-5 && dp.worst < 1e-5 && aux.worst < 1e-6);
    // Expert 3 duplicates expert 1's logit and ranks after it on the tie:
    // with k = 1 it can never be chosen.
    if (k == 1) CHECK(a.frac[0][3] == 0.0);
}

}  // namespace

int main() {
    device::set_from_env();
    std::printf("MoE dispatch ops: device vs host (%s)\n",
                on_device() ? "device op set ON" : "host vs host: CPU build or op set off");
    test_ops(false);
    test_ops(true);
    for (bool win : {false, true}) {
        test_moe_layer(1, false, win);
        test_moe_layer(2, true, win);
        test_moe_layer(2, false, win);
    }
    std::printf("\n[PASS] MoE device ops\n");
    return 0;
}
