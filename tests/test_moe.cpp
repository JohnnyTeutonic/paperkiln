// Mixture-of-experts FFN (moe.hpp) receipts:
//   1. the dispatch ops (gather_rows, scatter_add_rows, gather_elems,
//      logsumexp_row) against central finite differences;
//   2. MoEFFN FD gradcheck w.r.t. x, router and every expert weight, for
//      top-1 raw (Switch), top-2 renormalised (Mixtral) and top-2 raw;
//   3. dense equivalence: E = 1, k = 1 gives the dense SwiGLU's output and
//      gradients bit for bit, at the FFN and at the whole-model level;
//   4. load-balance and z-loss values on a hand-computed case, and their
//      gradients by FD;
//   5. sparse upcycling: identical experts + any router + renormalised
//      gates reproduce the dense model up to float rounding;
//   6. parameter names and dense-model invariance (no moe = no change).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/llama.hpp"
#include "microtorch/moe.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

namespace {

Matrix randn(size_t r, size_t c, unsigned seed, float scale = 1.0f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, scale);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(gen);
    return m;
}

// Worst relative error of analytic vs central-difference gradients over
// every entry of `leaf`. `same_routing` (optional) is evaluated at each
// perturbed point; entries whose perturbation flips a top-k choice are a
// discontinuity of the loss, not a gradient error, and are skipped and
// counted in *skipped.
double fd_check(const std::function<double()>& f, const Var& leaf, const Matrix& analytic,
                const std::function<bool()>& same_routing = nullptr, size_t* skipped = nullptr,
                float h = 1e-2f) {
    double worst = 0.0;
    for (size_t i = 0; i < leaf->data.rows(); ++i)
        for (size_t j = 0; j < leaf->data.cols(); ++j) {
            const float keep = leaf->data(i, j);
            NoGrad ng;
            leaf->data(i, j) = keep + h;
            const double up = f();
            const bool ok_up = !same_routing || same_routing();
            leaf->data(i, j) = keep - h;
            const double dn = f();
            const bool ok_dn = !same_routing || same_routing();
            leaf->data(i, j) = keep;
            if (!(ok_up && ok_dn)) {
                if (skipped) ++*skipped;
                continue;
            }
            const double fd = (up - dn) / (2.0 * h);
            const double a = analytic.rows() ? analytic(i, j) : 0.0;
            worst = std::max(worst, std::abs(a - fd) / (1.0 + std::max(std::abs(a), std::abs(fd))));
        }
    return worst;
}

const double TOL = 5e-3;  // float32 forward, h = 1e-2 central differences (repo convention)

bool bit_equal(const Matrix& a, const Matrix& b, size_t* mismatches = nullptr) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    size_t bad = 0;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j) {
            const float x = a(i, j), y = b(i, j);
            if (std::memcmp(&x, &y, sizeof(float)) != 0) ++bad;
        }
    if (mismatches) *mismatches = bad;
    return bad == 0;
}

// Values equal under ==: identical except possibly the sign of a zero.
bool value_equal(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j)
            if (!(a(i, j) == b(i, j))) return false;
    return true;
}

double max_abs_diff(const Matrix& a, const Matrix& b) {
    double w = 0.0;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j) w = std::max(w, (double)std::abs(a(i, j) - b(i, j)));
    return w;
}

void test_dispatch_ops() {
    std::printf("=== dispatch ops vs finite differences ===\n");
    Var x = make_var(randn(5, 3, 1), true);
    const std::vector<int> gi = {4, 0, 4, 2};  // repeated row: gradients must add
    const std::vector<int> si = {1, 1, 0, 3};  // repeated target row: outputs add
    const std::vector<int> er = {0, 0, 3, 4, 4, 1}, ec = {2, 2, 0, 1, 0, 2};
    const Var M1 = make_var(randn(4, 3, 2)), M2 = make_var(randn(4, 3, 3)), M3 = make_var(randn(3, 2, 4)),
              M4 = make_var(randn(5, 1, 5));
    auto build = [&]() {
        Var g = ops::gather_rows(x, gi);                     // [4,3]
        Var s = ops::scatter_add_rows(ops::mul(g, M1), si, 4);  // [4,3]
        Var e = ops::gather_elems(x, er, ec, 2);             // [3,2]
        Var l = ops::logsumexp_row(x);                       // [5,1]
        Var a = ops::add(ops::mean(ops::mul(s, M2)), ops::mean(ops::mul(e, M3)));
        return ops::add(a, ops::mean(ops::mul(l, M4)));
    };
    backward(build());
    const double w = fd_check([&] { return (double)build()->data(0, 0); }, x, x->grad);
    std::printf("  gather_rows/scatter_add_rows/gather_elems/logsumexp_row dX: worst rel err %.3e\n", w);
    CHECK(w < TOL);

    // scatter_add_rows backward alone w.r.t. its input (gathers dY rows).
    Var y = make_var(randn(4, 3, 6), true);
    auto build2 = [&]() { return ops::mean(ops::mul(ops::scatter_add_rows(y, si, 4), M2)); };
    backward(build2());
    const double w2 = fd_check([&] { return (double)build2()->data(0, 0); }, y, y->grad);
    std::printf("  scatter_add_rows dX: worst rel err %.3e\n", w2);
    CHECK(w2 < TOL);

    // logsumexp value against a direct double computation, large logits.
    Var big = make_var(Matrix(1, 3));
    big->data(0, 0) = 100.0f, big->data(0, 1) = 99.0f, big->data(0, 2) = -50.0f;
    const double expect = 100.0 + std::log(1.0 + std::exp(-1.0) + std::exp(-150.0));
    const double got = ops::logsumexp_row(big)->data(0, 0);
    std::printf("  logsumexp([100,99,-50]) = %.6f (expect %.6f)\n", got, expect);
    CHECK(std::abs(got - expect) < 1e-4);
}

// Routing signature: the top-k sets of every token (via the aux record's
// expert fractions is too coarse, so recompute from the router directly).
std::vector<int> topk_sets(const nn::MoEFFN& m, const Var& x) {
    NoGrad ng;
    Var lg = m.router->forward(x);
    std::vector<int> sig;
    const size_t E = m.cfg.experts;
    for (size_t t = 0; t < lg->data.rows(); ++t) {
        std::vector<int> o(E);
        for (size_t e = 0; e < E; ++e) o[e] = (int)e;
        std::partial_sort(o.begin(), o.begin() + m.cfg.top_k, o.end(), [&](int a, int b) {
            return lg->data(t, a) > lg->data(t, b) || (lg->data(t, a) == lg->data(t, b) && a < b);
        });
        std::sort(o.begin(), o.begin() + m.cfg.top_k);
        sig.insert(sig.end(), o.begin(), o.begin() + m.cfg.top_k);
    }
    return sig;
}

void test_moe_gradcheck(size_t k, bool renorm) {
    const size_t T = 7, d = 6, dff = 5, E = 4;
    nn::MoEConfig mc;
    mc.experts = E, mc.top_k = k, mc.renormalize = renorm;
    nn::MoEFFN m(d, dff, mc, 11);
    // Larger-than-default weights so the FFN output and the routing margins
    // are well above float noise at h = 1e-2.
    m.router->W->data = randn(d, E, 21, 0.6f);
    for (size_t e = 0; e < E; ++e) {
        m.experts[e]->gate_proj->W->data = randn(d, dff, 30 + 3 * e, 0.5f);
        m.experts[e]->up_proj->W->data = randn(d, dff, 31 + 3 * e, 0.5f);
        m.experts[e]->down_proj->W->data = randn(dff, d, 32 + 3 * e, 0.5f);
    }
    Var x = make_var(randn(T, d, 40), true);
    const Var Mk = make_var(randn(T, d, 41));
    auto build = [&]() { return ops::mean(ops::mul(m.forward(x), Mk)); };
    const std::vector<int> sig0 = topk_sets(m, x);
    {
        nn::MoEAux aux;
        NoGrad ng;
        m.forward(x, &aux);
        std::printf("  [k=%zu %s] routing fractions:", k, renorm ? "renorm" : "raw");
        for (double f : aux.frac[0]) std::printf(" %.3f", f);
        std::printf("\n");
        size_t used = 0;
        for (double f : aux.frac[0]) used += f > 0;
        CHECK(used >= 2);  // the check must exercise more than one expert
    }
    zero_grad(m.parameters());
    backward(build());
    auto f = [&]() { return (double)build()->data(0, 0); };
    auto same = [&]() { return topk_sets(m, x) == sig0; };
    size_t skipped = 0;
    double wx = fd_check(f, x, x->grad, same, &skipped);
    double wr = fd_check(f, m.router->W, m.router->W->grad, same, &skipped);
    double we = 0.0;
    for (size_t e = 0; e < E; ++e)
        for (const Var& p : {m.experts[e]->gate_proj->W, m.experts[e]->up_proj->W, m.experts[e]->down_proj->W})
            we = std::max(we, fd_check(f, p, p->grad, same, &skipped));
    std::printf("  [k=%zu %s] FD worst rel err: dx %.3e  drouter %.3e  dexperts %.3e  (%zu entries skipped: routing flip)\n",
                k, renorm ? "renorm" : "raw", wx, wr, we, skipped);
    CHECK(wx < TOL && wr < TOL && we < TOL);
    CHECK(skipped < 5);
    // The router must actually receive task gradient (k = 1 renorm aside).
    double rnorm = 0.0;
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < E; ++j) rnorm += std::abs(m.router->W->grad(i, j));
    std::printf("  [k=%zu %s] |d router| = %.3e\n", k, renorm ? "renorm" : "raw", rnorm);
    CHECK(rnorm > 1e-4);
}

void test_dense_equivalence_ffn(bool renorm) {
    const size_t T = 9, d = 8, dff = 12;
    nn::MoEConfig mc;
    mc.experts = 1, mc.top_k = 1, mc.renormalize = renorm;
    nn::MoEFFN m(d, dff, mc, 3);
    // The dense FFN: same weights, the LlamaBlock expression.
    nn::Linear g(d, dff, false, 100), u(d, dff, false, 101), dn(dff, d, false, 102);
    g.W->data = m.experts[0]->gate_proj->W->data;
    u.W->data = m.experts[0]->up_proj->W->data;
    dn.W->data = m.experts[0]->down_proj->W->data;
    const Matrix X = randn(T, d, 7), Mk = randn(T, d, 8);
    Var x1 = make_var(X, true), x2 = make_var(X, true);
    Var y1 = m.forward(x1);
    Var y2 = dn.forward(ops::mul(ops::silu(g.forward(x2)), u.forward(x2)));
    backward(ops::mean(ops::mul(y1, make_var(Mk))));
    backward(ops::mean(ops::mul(y2, make_var(Mk))));
    size_t bad = 0;
    CHECK(bit_equal(y1->data, y2->data, &bad));
    const bool gx = value_equal(x1->grad, x2->grad);
    const bool gg = bit_equal(m.experts[0]->gate_proj->W->grad, g.W->grad);
    const bool gu = bit_equal(m.experts[0]->up_proj->W->grad, u.W->grad);
    const bool gd = bit_equal(m.experts[0]->down_proj->W->grad, dn.W->grad);
    double rmax = 0.0;
    for (size_t i = 0; i < d; ++i) rmax = std::max(rmax, (double)std::abs(m.router->W->grad(i, 0)));
    std::printf("  [E=1 k=1 %s] FFN output bit-identical; grads: x %s, gate %s, up %s, down %s; |d router| max %.1e\n",
                renorm ? "renorm" : "raw", gx ? "equal" : "DIFF", gg ? "bit-identical" : "DIFF",
                gu ? "bit-identical" : "DIFF", gd ? "bit-identical" : "DIFF", rmax);
    CHECK(gx && gg && gu && gd);
    CHECK(rmax == 0.0);  // one expert: the gate is the constant 1
}

nn::LlamaConfig tiny_cfg() {
    nn::LlamaConfig c;
    c.vocab = 50, c.d = 16, c.n_layers = 2, c.n_heads = 2, c.d_ff = 24, c.n_ctx = 16;
    return c;
}

std::vector<std::string> router_names(const nn::Llama& m) {
    std::vector<std::string> r;
    for (const auto& [n, p] : m.named_parameters())
        if (n.find("mlp.router.") != std::string::npos) r.push_back(n);
    return r;
}

void test_dense_equivalence_model(bool renorm) {
    nn::LlamaConfig dc = tiny_cfg();
    nn::Llama dense(dc, 5);
    nn::LlamaConfig mcfg = dc;
    mcfg.moe.experts = 1, mcfg.moe.top_k = 1, mcfg.moe.renormalize = renorm;
    nn::Llama moe(mcfg, 5);
    std::vector<std::string> rk;
    moe.load_state_dict(nn::upcycle_state_dict(dense.state_dict(), 1, &rk), true, rk);
    CHECK(rk.size() == dc.n_layers);
    const std::vector<int> ids = {3, 9, 9, 41, 0, 17, 22, 5, 30, 2, 8, 13};
    const std::vector<int> tgt = {9, 9, 41, 0, 17, 22, 5, 30, 2, 8, 13, 4};
    Var ld = dense.forward(ids, 6), lm = moe.forward(ids, 6);
    size_t bad = 0;
    const bool out_eq = bit_equal(ld->data, lm->data, &bad);
    backward(ops::cross_entropy(ld, tgt));
    backward(ops::cross_entropy(lm, tgt));
    // Every dense parameter's gradient against its MoE counterpart.
    auto mp = moe.named_parameters();
    std::map<std::string, Var> by(mp.begin(), mp.end());
    size_t params = 0, bit_same = 0, value_same = 0;
    for (const auto& [n, p] : dense.named_parameters()) {
        std::string mn = n;
        const auto at = mn.find("mlp.");
        if (at != std::string::npos) mn.insert(at + 4, "experts.0.");
        CHECK(by.count(mn) == 1);
        ++params;
        bit_same += bit_equal(p->grad, by[mn]->grad);
        value_same += value_equal(p->grad, by[mn]->grad);
    }
    std::printf("  [E=1 k=1 %s] model logits bit-identical: %s (%zu mismatches); grads bit-identical %zu/%zu, "
                "value-equal %zu/%zu\n",
                renorm ? "renorm" : "raw", out_eq ? "yes" : "NO", bad, bit_same, params, value_same, params);
    CHECK(out_eq);
    CHECK(value_same == params);
}

void test_from_scratch_init() {
    // Expert 0 takes the dense FFN's seeds: an E-expert model's expert 0
    // starts equal to the dense model's FFN, and every non-FFN tensor is
    // initialised identically.
    nn::LlamaConfig dc = tiny_cfg();
    nn::Llama dense(dc, 9);
    nn::LlamaConfig mcfg = dc;
    mcfg.moe.experts = 3, mcfg.moe.top_k = 2;
    nn::Llama moe(mcfg, 9);
    auto ds = dense.state_dict(), ms = moe.state_dict();
    size_t same = 0;
    for (const auto& [n, m] : ds) {
        std::string mn = n;
        const auto at = mn.find("mlp.");
        if (at != std::string::npos) mn.insert(at + 4, "experts.0.");
        same += ms.count(mn) && bit_equal(m, ms[mn]);
    }
    std::printf("  from-scratch E=3: %zu/%zu dense tensors equal their MoE counterpart (expert 0)\n", same, ds.size());
    CHECK(same == ds.size());
    CHECK(!bit_equal(ms["layers.0.mlp.experts.0.gate_proj.weight"], ms["layers.0.mlp.experts.1.gate_proj.weight"]));
}

void test_aux_losses() {
    std::printf("=== auxiliary losses: hand-computed values + FD ===\n");
    // d = 2, E = 2, k = 1 raw. Router W = [[2,0],[0,1]]; tokens e1,e1,e2,e1.
    // logits rows: [2,0] (x3) and [0,1]; top-1: expert 0 (x3), expert 1.
    nn::MoEConfig mc;
    mc.experts = 2, mc.top_k = 1, mc.renormalize = false;
    nn::MoEFFN m(2, 3, mc, 1);
    m.router->W->data = Matrix(2, 2);
    m.router->W->data(0, 0) = 2.0f, m.router->W->data(1, 1) = 1.0f;
    Var x = make_var(Matrix(4, 2), true);
    x->data(0, 0) = x->data(1, 0) = x->data(3, 0) = 1.0f;
    x->data(2, 1) = 1.0f;
    nn::MoEAux aux;
    m.forward(x, &aux);
    const double e = std::exp(1.0), e2 = std::exp(2.0);
    const double p0a = e2 / (e2 + 1), p0b = 1 / (1 + e);  // P[t,0] for the two token kinds
    const double P0 = (3 * p0a + p0b) / 4, P1 = 1 - P0;
    const double lb_expect = 2 * (0.75 * P0 + 0.25 * P1);
    const double la = std::log(e2 + 1), lb_ = std::log(1 + e);
    const double z_expect = (3 * la * la + lb_ * lb_) / 4;
    std::printf("  f = [%.3f %.3f] (expect [0.750 0.250])\n", aux.frac[0][0], aux.frac[0][1]);
    std::printf("  load balance %.7f (expect %.7f), z-loss %.7f (expect %.7f)\n", aux.lb->data(0, 0), lb_expect,
                aux.z->data(0, 0), z_expect);
    CHECK(std::abs(aux.frac[0][0] - 0.75) < 1e-12 && std::abs(aux.frac[0][1] - 0.25) < 1e-12);
    CHECK(std::abs(aux.lb->data(0, 0) - lb_expect) < 1e-6);
    CHECK(std::abs(aux.z->data(0, 0) - z_expect) < 1e-5);
    // Uniform routing gives lb = 1 exactly: zero router -> P = 1/2, and
    // the tie rule sends everything to expert 0 (f = [1, 0]) -> 2*(1*0.5) = 1.
    {
        nn::MoEFFN u(2, 3, mc, 1);
        u.router->W->data = Matrix(2, 2);
        nn::MoEAux a2;
        u.forward(x, &a2);
        std::printf("  zero router: lb %.7f (expect 1), z %.7f (expect log(2)^2 = %.7f)\n", a2.lb->data(0, 0),
                    a2.z->data(0, 0), std::log(2.0) * std::log(2.0));
        CHECK(std::abs(a2.lb->data(0, 0) - 1.0) < 1e-6);
        CHECK(std::abs(a2.z->data(0, 0) - std::log(2.0) * std::log(2.0)) < 1e-6);
    }
    // Gradients of each loss by FD, on a random case with E = 4, k = 2.
    nn::MoEConfig mc2;
    mc2.experts = 4, mc2.top_k = 2, mc2.renormalize = true;
    nn::MoEFFN m2(5, 4, mc2, 2);
    m2.router->W->data = randn(5, 4, 50, 0.7f);
    Var x2 = make_var(randn(8, 5, 51), true);
    const std::vector<int> sig0 = topk_sets(m2, x2);
    auto same = [&]() { return topk_sets(m2, x2) == sig0; };
    for (int which = 0; which < 2; ++which) {
        auto build = [&]() {
            nn::MoEAux a;
            m2.forward(x2, &a);
            return which == 0 ? a.lb : a.z;
        };
        zero_grad(m2.parameters());
        x2->grad = Matrix();
        backward(build());
        size_t skipped = 0;
        auto f = [&]() { return (double)build()->data(0, 0); };
        const double wr = fd_check(f, m2.router->W, m2.router->W->grad, same, &skipped);
        const double wx = fd_check(f, x2, x2->grad, same, &skipped);
        std::printf("  %s: FD worst rel err drouter %.3e, dx %.3e (%zu skipped)\n",
                    which == 0 ? "load balance" : "z-loss", wr, wx, skipped);
        CHECK(wr < TOL && wx < TOL && skipped < 5);
        // The aux losses never touch the experts.
        CHECK(m2.experts[0]->gate_proj->W->grad.rows() == 0 ||
              max_abs_diff(m2.experts[0]->gate_proj->W->grad, Matrix(5, 4)) == 0.0);
    }
}

void test_upcycling() {
    std::printf("=== sparse upcycling: identical experts reproduce the dense model ===\n");
    nn::LlamaConfig dc = tiny_cfg();
    nn::Llama dense(dc, 13);
    // Give the dense FFN non-trivial weights (the 0.02 init makes it tiny).
    for (const auto& [n, p] : dense.named_parameters())
        if (n.find("mlp.") != std::string::npos) p->data = randn(p->data.rows(), p->data.cols(), 77 + n.size(), 0.3f);
    const std::vector<int> ids = {1, 4, 9, 16, 25, 36, 49, 0, 7, 14, 21, 28};
    Matrix ref;
    {
        NoGrad ng;
        ref = dense.forward(ids, 6)->data;
    }
    double scale = 0.0;
    for (size_t i = 0; i < ref.rows(); ++i)
        for (size_t j = 0; j < ref.cols(); ++j) scale = std::max(scale, (double)std::abs(ref(i, j)));
    for (size_t k : {1, 2}) {
        nn::LlamaConfig mcfg = dc;
        mcfg.moe.experts = 4, mcfg.moe.top_k = k, mcfg.moe.renormalize = true;
        nn::Llama moe(mcfg, 99);  // different seed: every non-loaded tensor differs
        std::vector<std::string> rk;
        moe.load_state_dict(nn::upcycle_state_dict(dense.state_dict(), 4, &rk), true, rk);
        CHECK(rk == router_names(moe));
        // An arbitrary, large router: tokens spread over the experts.
        for (const auto& [n, p] : moe.named_parameters())
            if (n.find("mlp.router.") != std::string::npos) p->data = randn(p->data.rows(), p->data.cols(), 5 + n.size(), 2.0f);
        NoGrad ng;
        nn::MoEAux aux;
        const Matrix got = moe.forward(ids, 6, &aux)->data;
        size_t used = 0;
        for (const auto& fr : aux.frac)
            for (double f : fr) used += f > 0;
        const double err = max_abs_diff(got, ref) / scale;
        std::printf("  E=4 k=%zu renorm, random router: max |logit diff| / max |logit| = %.3e (%zu expert-layer slots used of %zu)\n",
                    k, err, used, 4 * dc.n_layers);
        CHECK(err < 1e-5);
        CHECK(used >= 4);
    }
    // The raw (Switch) convention scales the FFN by p and is NOT the dense
    // model at init -- recorded so the choice for an upcycling run is clear.
    {
        nn::LlamaConfig mcfg = dc;
        mcfg.moe.experts = 4, mcfg.moe.top_k = 1, mcfg.moe.renormalize = false;
        nn::Llama moe(mcfg, 99);
        std::vector<std::string> rk;
        moe.load_state_dict(nn::upcycle_state_dict(dense.state_dict(), 4, &rk), true, rk);
        NoGrad ng;
        const double err = max_abs_diff(moe.forward(ids, 6)->data, ref) / scale;
        std::printf("  E=4 k=1 raw (Switch): relative logit diff %.3e -- not dense-equivalent, by construction\n", err);
        CHECK(err > 1e-4);
    }
}

void test_names_and_dense_untouched() {
    std::printf("=== names; dense models untouched ===\n");
    nn::LlamaConfig c = tiny_cfg();
    nn::Llama d1(c, 4);
    auto sd = d1.state_dict();
    CHECK(sd.count("layers.0.mlp.gate_proj.weight") == 1 && sd.count("layers.1.mlp.down_proj.weight") == 1);
    for (const auto& [n, m] : sd) CHECK(n.find("router") == std::string::npos && n.find("experts") == std::string::npos);
    CHECK(d1.blocks[0]->moe == nullptr && d1.blocks[0]->gate_proj != nullptr);
    nn::LlamaConfig mc = c;
    mc.moe.experts = 3, mc.moe.top_k = 2;
    nn::Llama m(mc, 4);
    auto ms = m.state_dict();
    const char* expect[] = {"layers.0.mlp.router.weight", "layers.0.mlp.experts.0.gate_proj.weight",
                            "layers.0.mlp.experts.2.up_proj.weight", "layers.1.mlp.experts.1.down_proj.weight"};
    for (const char* k : expect) CHECK(ms.count(k) == 1);
    CHECK(ms.count("layers.0.mlp.gate_proj.weight") == 0);
    CHECK(ms["layers.0.mlp.router.weight"].rows() == c.d && ms["layers.0.mlp.router.weight"].cols() == 3);
    // Parameter count: dense + (E-1) FFNs + routers.
    const size_t ffn = 3 * c.d * c.d_ff;
    CHECK(m.parameter_count() == d1.parameter_count() + c.n_layers * (2 * ffn + c.d * 3));
    std::printf("  dense: %zu params, no router/expert keys; E=3: %zu params, HF-style expert names\n",
                d1.parameter_count(), m.parameter_count());
}

void test_short_training() {
    std::printf("=== short training with auxiliary losses ===\n");
    nn::LlamaConfig c = tiny_cfg();
    c.moe.experts = 4, c.moe.top_k = 2;
    nn::Llama model(c, 42);
    model.train();
    nn::AdamW opt(model.parameters(), 3e-3f);
    std::mt19937 drng(123);
    float first = 0, last = 0;
    double ent = 0;
    for (int step = 0; step < 40; ++step) {
        std::vector<int> x(12), y(12);
        for (size_t t = 0; t < x.size(); ++t) {
            x[t] = drng() % c.vocab;
            y[t] = (x[t] + 1) % static_cast<int>(c.vocab);
        }
        nn::MoEAux aux;
        Var task = ops::cross_entropy(model.forward(x, 0, &aux), y);
        Var loss = ops::add(task, ops::add(ops::scale(aux.lb, 0.01f / aux.layers), ops::scale(aux.z, 0.001f / aux.layers)));
        opt.zero_grad();
        backward(loss);
        ops::clip_grad_norm(model.parameters(), 1.0f);
        opt.step();
        if (step == 0) first = task->data(0, 0);
        last = task->data(0, 0);
        ent = 0;
        for (const auto& f : aux.frac) ent += nn::routing_entropy(f) / aux.frac.size();
    }
    std::printf("  task loss %.4f -> %.4f over 40 steps; final routing entropy %.3f (1 = balanced)\n", first, last, ent);
    CHECK(last < first - 0.3f);
    CHECK(ent > 0.5);
}

}  // namespace

int main() {
    device::set_from_env();
    std::printf("microtorch MoE FFN tests\n");
    test_dispatch_ops();
    std::printf("=== MoEFFN gradcheck (x, router, experts) ===\n");
    test_moe_gradcheck(1, false);
    test_moe_gradcheck(2, true);
    test_moe_gradcheck(2, false);
    std::printf("=== dense equivalence (E = 1, k = 1) ===\n");
    test_dense_equivalence_ffn(false);
    test_dense_equivalence_ffn(true);
    test_dense_equivalence_model(false);
    test_dense_equivalence_model(true);
    test_from_scratch_init();
    test_aux_losses();
    test_upcycling();
    test_names_and_dense_untouched();
    test_short_training();
    std::printf("\n[PASS] all MoE tests\n");
    return 0;
}
