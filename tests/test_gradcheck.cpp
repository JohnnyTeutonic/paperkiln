// Phase-1a gate: every op's analytic gradient against central finite
// differences, plus the structural properties a tape must have. Mirrors the
// discipline of the Python references (kda/attn_res/muon): named checks,
// assert the property, print the measured number.
//
// Also serves as the audit's verification pass: it MEASURES the
// transformer_core CPU gelu-derivative bug (primitives.hpp) instead of
// taking it on faith, and pins matmul_optimized against the naive matmul.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "microtorch/device.hpp"
#include "microtorch/ops.hpp"

using microtorch::make_var;
using microtorch::Var;
namespace ops = microtorch::ops;

namespace {

int g_failures = 0;

void check(bool ok, const char* label, double measured) {
    std::printf("  [%s] %-52s %.3e\n", ok ? "ok" : "FAIL", label, measured);
    if (!ok) ++g_failures;
}

Matrix randn(size_t r, size_t c, unsigned seed, float scale = 1.0f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, scale);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(gen);
    return m;
}

// Central finite differences on a scalar-valued rebuild of the graph. The
// forward runs under NoGrad so FD never contaminates the tape under test.
double fd_vs_analytic(const std::function<float()>& forward, Var leaf, const Matrix& analytic,
                      float h = 1e-2f) {
    double worst = 0.0;
    for (size_t i = 0; i < leaf->data.rows(); ++i) {
        for (size_t j = 0; j < leaf->data.cols(); ++j) {
            const float keep = leaf->data(i, j);
            microtorch::NoGrad ng;
            leaf->data(i, j) = keep + h;
            const float up = forward();
            leaf->data(i, j) = keep - h;
            const float dn = forward();
            leaf->data(i, j) = keep;
            const double fd = (static_cast<double>(up) - dn) / (2.0 * h);
            const double a = analytic(i, j);
            const double err = std::abs(a - fd) / (1.0 + std::max(std::abs(a), std::abs(fd)));
            worst = std::max(worst, err);
        }
    }
    return worst;
}

}  // namespace

int main() {
    microtorch::device::set_from_env();
    std::printf("microtorch phase-1a gradcheck\n\n");

    // ---- matmul_optimized parity vs the naive path (audit 1a promise) ----
    {
        Matrix A = randn(7, 13, 1), B = randn(13, 9, 2);
        Matrix C1 = matmul_optimized(A, B);
        Matrix C2 = A * B;  // naive operator*
        double worst = 0.0;
        for (size_t i = 0; i < C1.rows(); ++i)
            for (size_t j = 0; j < C1.cols(); ++j)
                worst = std::max(worst, static_cast<double>(std::abs(C1(i, j) - C2(i, j))));
        check(worst < 1e-4, "matmul_optimized == naive matmul", worst);
    }

    // ---- full composed model: X -> linear -> gelu -> linear -> softmax ----
    // Loss = mean(softmax_row(gelu(X W1 + b) W2) .* M), M a fixed mask, so
    // the softmax backward sees a non-uniform upstream gradient.
    Var X = make_var(randn(4, 3, 10), true);
    Var W1 = make_var(randn(3, 5, 11, 0.7f), true);
    Var b = make_var(randn(1, 5, 12, 0.3f), true);
    Var W2 = make_var(randn(5, 3, 13, 0.7f), true);
    Var M = make_var(randn(4, 3, 14));  // no grad: constant mask

    auto build = [&]() -> Var {
        Var h = ops::gelu(ops::add_bias(ops::matmul(X, W1), b));
        Var s = ops::softmax_row(ops::matmul(h, W2));
        return ops::mean(ops::mul(s, M));
    };
    auto forward = [&]() -> float { return build()->data(0, 0); };

    Var loss = build();
    microtorch::backward(loss);

    const double TOL = 5e-3;  // float32 forward + h=1e-2 central differences
    check(fd_vs_analytic(forward, X, X->grad) < TOL, "dX vs finite differences",
          fd_vs_analytic(forward, X, X->grad));
    check(fd_vs_analytic(forward, W1, W1->grad) < TOL, "dW1 vs finite differences",
          fd_vs_analytic(forward, W1, W1->grad));
    check(fd_vs_analytic(forward, b, b->grad) < TOL, "db vs finite differences",
          fd_vs_analytic(forward, b, b->grad));
    check(fd_vs_analytic(forward, W2, W2->grad) < TOL, "dW2 vs finite differences",
          fd_vs_analytic(forward, W2, W2->grad));

    // ---- diamond: the same leaf reaches the root twice; grads must ADD ----
    {
        Var a = make_var(randn(3, 3, 20), true);
        Var y = ops::mean(ops::mul(a, a));  // dy/da = 2a/N exactly
        microtorch::backward(y);
        double worst = 0.0;
        const float n = 9.0f;
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                worst = std::max(
                    worst, static_cast<double>(std::abs(a->grad(i, j) - 2.0f * a->data(i, j) / n)));
        check(worst < 1e-6, "diamond accumulation: d(mean(a.*a)) == 2a/N", worst);
    }

    // ---- sub: both branches of the subtraction get the right sign ----
    {
        Var p = make_var(randn(2, 4, 21), true);
        Var q = make_var(randn(2, 4, 22), true);
        Var y = ops::mean(ops::mul(ops::sub(p, q), ops::sub(p, q)));
        microtorch::backward(y);
        double worst = 0.0;
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 4; ++j) {
                float want = 2.0f * (p->data(i, j) - q->data(i, j)) / 8.0f;
                worst = std::max(worst, static_cast<double>(std::abs(p->grad(i, j) - want) +
                                                            std::abs(q->grad(i, j) + want)));
            }
        check(worst < 1e-6, "sub: d(mean((p-q)^2)) signs on both branches", worst);
    }

    // ---- the transformer_core CPU gelu-derivative bug, MEASURED ----
    // Analytic-vs-FD for our op must pass; the legacy formula against the
    // same FD must fail by a margin. This doubles as a canary: if
    // apply_gelu_derivative is ever fixed upstream, the second assertion
    // trips and tells us to reconsider wrapping it.
    {
        Var x = make_var(randn(1, 64, 30), true);
        auto gforward = [&]() -> float { return ops::mean(ops::gelu(x))->data(0, 0); };
        Var gy = ops::mean(ops::gelu(x));
        microtorch::backward(gy);
        double ours = fd_vs_analytic(gforward, x, x->grad);
        check(ours < TOL, "gelu backward (microtorch) vs FD", ours);

        Matrix legacy(1, 64, 1.0f / 64.0f);     // upstream grad of mean
        legacy.apply_gelu_derivative(x->data);  // the buggy in-place multiply
        double bug = 0.0;
        for (size_t j = 0; j < 64; ++j)
            bug = std::max(bug, static_cast<double>(std::abs(legacy(0, j) - x->grad(0, j))));
        check(bug > 1e-4, "legacy apply_gelu_derivative disagrees (canary)", bug);
    }

    // ---- phase-2a ops: mul_row (both branches) and silu vs FD ----
    {
        Var x = make_var(randn(3, 5, 40), true);
        Var r = make_var(randn(1, 5, 41), true);
        auto f = [&]() -> float { return ops::mean(ops::mul_row(ops::silu(x), r))->data(0, 0); };
        microtorch::backward(ops::mean(ops::mul_row(ops::silu(x), r)));
        check(fd_vs_analytic(f, x, x->grad) < TOL, "silu+mul_row: dx vs FD",
              fd_vs_analytic(f, x, x->grad));
        check(fd_vs_analytic(f, r, r->grad) < TOL, "mul_row: drow vs FD",
              fd_vs_analytic(f, r, r->grad));
    }

    // ---- phase-2b ops: rmsnorm and apply_rope vs FD ----
    {
        Var x = make_var(randn(4, 8, 60), true);
        Var w = make_var(randn(1, 8, 61), true);
        auto f = [&]() -> float { return ops::mean(ops::rmsnorm(x, w))->data(0, 0); };
        microtorch::backward(ops::mean(ops::rmsnorm(x, w)));
        check(fd_vs_analytic(f, x, x->grad) < TOL, "rmsnorm: dx vs FD",
              fd_vs_analytic(f, x, x->grad));
        check(fd_vs_analytic(f, w, w->grad) < TOL, "rmsnorm: dw vs FD",
              fd_vs_analytic(f, w, w->grad));
    }
    {
        Var qk = make_var(randn(2, 12, 62), true);  // [T=2, d*3=12], one head
        std::vector<int> pos = {0, 1};
        auto f = [&]() -> float {
            return ops::mean(ops::apply_rope(qk, pos, 10000.0f, 4, true))->data(0, 0);
        };
        microtorch::backward(ops::mean(ops::apply_rope(qk, pos, 10000.0f, 4, true)));
        check(fd_vs_analytic(f, qk, qk->grad) < TOL, "apply_rope: dqk vs FD",
              fd_vs_analytic(f, qk, qk->grad));
    }
    // Per-head RoPE at H > 1 (d=12, head_dim 4: three heads), both the
    // every-head rotation and the legacy head-0-only one. A fixed random
    // weighting makes the loss sensitive to every rotated column (a plain
    // mean lets errors in the two halves of a pair partly cancel).
    for (const bool all_heads : {true, false}) {
        Var qk = make_var(randn(3, 36, all_heads ? 63 : 64), true);  // [T=3, 3d]
        Var wgt = make_var(randn(3, 36, 65), false);
        const std::vector<int> pos = {0, 3, 7};
        auto f = [&]() -> float {
            return ops::mean(ops::mul(ops::apply_rope(qk, pos, 10000.0f, 4, all_heads), wgt))
                ->data(0, 0);
        };
        microtorch::backward(
            ops::mean(ops::mul(ops::apply_rope(qk, pos, 10000.0f, 4, all_heads), wgt)));
        const double e = fd_vs_analytic(f, qk, qk->grad);
        check(e < TOL, all_heads ? "apply_rope H=3 all heads: dqk vs FD"
                                 : "apply_rope H=3 first head: dqk vs FD",
              e);
    }

    // ---- fused_attention: FD on q,k,v + parity vs the composed path ----
    {
        const double TOL = 5e-3;
        const size_t T = 8, dk = 4;
        const float sc = 1.0f / std::sqrt(static_cast<float>(dk));
        Var q = make_var(randn(T, dk, 61, 0.5f), true);
        Var k = make_var(randn(T, dk, 62, 0.5f), true);
        Var v = make_var(randn(T, dk, 63, 0.5f), true);
        struct Cfg {
            const char* name;
            size_t sl;
            bool causal;
        };
        for (const Cfg c : {Cfg{"causal", 0, true}, Cfg{"causal+batch", T / 2, true},
                            Cfg{"bidir+batch", T / 2, false}}) {
            auto f = [&] {
                return ops::mean(ops::fused_attention(q, k, v, sc, c.sl, c.causal))->data(0, 0);
            };
            microtorch::zero_grad({q, k, v});
            microtorch::backward(ops::mean(ops::fused_attention(q, k, v, sc, c.sl, c.causal)));
            char label[80];
            std::snprintf(label, sizeof label, "fused_attention(%s): dq FD", c.name);
            check(fd_vs_analytic(f, q, q->grad) < TOL, label, fd_vs_analytic(f, q, q->grad));
            std::snprintf(label, sizeof label, "fused_attention(%s): dk FD", c.name);
            check(fd_vs_analytic(f, k, k->grad) < TOL, label, fd_vs_analytic(f, k, k->grad));
            std::snprintf(label, sizeof label, "fused_attention(%s): dv FD", c.name);
            check(fd_vs_analytic(f, v, v->grad) < TOL, label, fd_vs_analytic(f, v, v->grad));
            // Forward parity against the composed mask+softmax path.
            microtorch::NoGrad ng;
            Var composed = ops::matmul(
                ops::softmax_row(ops::add(ops::scale(ops::matmul(q, ops::transpose(k)), sc),
                                          make_var(ops::attention_mask(T, c.sl, c.causal)))),
                v);
            Var fused = ops::fused_attention(q, k, v, sc, c.sl, c.causal);
            double worst = 0.0;
            for (size_t i = 0; i < T; ++i)
                for (size_t j = 0; j < dk; ++j)
                    worst = std::max(worst, static_cast<double>(std::abs(composed->data(i, j) -
                                                                         fused->data(i, j))));
            std::snprintf(label, sizeof label, "fused_attention(%s): forward parity", c.name);
            check(worst < 1e-5, label, worst);
        }
    }

    // ---- NoGrad records nothing; requires_grad=false leaves stay clean ----
    {
        microtorch::NoGrad ng;
        Var y = ops::mean(ops::gelu(ops::matmul(X, W1)));
        check(y->is_leaf() && !y->requires_grad, "NoGrad: no tape recorded", 0.0);
    }
    check(M->grad.rows() == 0, "constant leaf never accumulated a grad", 0.0);

    // ---- determinism: zero_grad + rebuild reproduces identical grads ----
    {
        Matrix first = X->grad;
        microtorch::zero_grad({X, W1, b, W2});
        microtorch::backward(build());
        double worst = 0.0;
        for (size_t i = 0; i < first.rows(); ++i)
            for (size_t j = 0; j < first.cols(); ++j)
                worst = std::max(worst, static_cast<double>(std::abs(first(i, j) - X->grad(i, j))));
        check(worst == 0.0, "zero_grad + rebuild: bit-identical grads", worst);
    }

    if (g_failures == 0) {
        std::printf("\nALL MICROTORCH GRADCHECKS PASSED\n");
        return 0;
    }
    std::printf("\n%d FAILURE(S)\n", g_failures);
    return 1;
}
