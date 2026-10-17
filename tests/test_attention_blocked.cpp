// Batched attention computed block by block (ops::fused_attention with
// seq_len < rows). It must equal running each sequence through the
// single-sequence path, bit for bit, forward and backward, and pass a
// finite-difference gradcheck.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "check.hpp"
#include "microtorch/ops.hpp"

using microtorch::make_var;
using microtorch::Var;
namespace ops = microtorch::ops;

namespace {

Matrix randn(size_t r, size_t c, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(g);
    return m;
}

Matrix rows(const Matrix& m, size_t r0, size_t n) {
    Matrix o(n, m.cols());
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m.cols(); ++j) o(i, j) = m(r0 + i, j);
    return o;
}

bool same(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

void test_equals_per_sequence(bool causal) {
    // B = 4 so the per-block weights below scale by exactly 0.25: mean() divides
    // by the element count, which is B times larger for the batched call, and a
    // power-of-two rescaling keeps every rounding identical.
    const size_t B = 4, sl = 8, d = 5;
    const Matrix Q = randn(B * sl, d, 1), K = randn(B * sl, d, 2), V = randn(B * sl, d, 3), G = randn(B * sl, d, 4);
    Var q = make_var(Q, true), k = make_var(K, true), v = make_var(V, true);
    Var y = ops::fused_attention_blocked(q, k, v, 0.4f, sl, causal);
    microtorch::backward(ops::mean(ops::mul(y, make_var(G))));
    for (size_t b = 0; b < B; ++b) {
        Var qb = make_var(rows(Q, b * sl, sl), true), kb = make_var(rows(K, b * sl, sl), true),
            vb = make_var(rows(V, b * sl, sl), true);
        Var yb = ops::fused_attention(qb, kb, vb, 0.4f, 0, causal);
        Matrix Gb = rows(G, b * sl, sl);
        for (size_t i = 0; i < sl; ++i)
            for (size_t j = 0; j < d; ++j) Gb(i, j) *= 0.25f;
        microtorch::backward(ops::mean(ops::mul(yb, make_var(Gb))));
        for (size_t i = 0; i < sl; ++i)
            for (size_t j = 0; j < d; ++j) {
                CHECK(same(y->data(b * sl + i, j), yb->data(i, j)));
                CHECK(same(q->grad(b * sl + i, j), qb->grad(i, j)));
                CHECK(same(k->grad(b * sl + i, j), kb->grad(i, j)));
                CHECK(same(v->grad(b * sl + i, j), vb->grad(i, j)));
            }
    }
    std::printf("  blocked == per-sequence, bit for bit (causal=%d): output and all three gradients\n", causal);
}

void test_gradcheck() {
    const size_t B = 2, sl = 4, d = 3;
    Var q = make_var(randn(B * sl, d, 5), true), k = make_var(randn(B * sl, d, 6), true),
        v = make_var(randn(B * sl, d, 7), true);
    const Var W = make_var(randn(B * sl, d, 8));
    auto f = [&]() { return ops::mean(ops::mul(ops::fused_attention_blocked(q, k, v, 0.5f, sl, true), W))->data(0, 0); };
    microtorch::backward(ops::mean(ops::mul(ops::fused_attention_blocked(q, k, v, 0.5f, sl, true), W)));
    double worst = 0.0;
    for (Var leaf : {q, k, v})
        for (size_t i = 0; i < leaf->data.rows(); ++i)
            for (size_t j = 0; j < leaf->data.cols(); ++j) {
                const float keep = leaf->data(i, j), h = 1e-2f;
                microtorch::NoGrad ng;
                leaf->data(i, j) = keep + h;
                const double up = f();
                leaf->data(i, j) = keep - h;
                const double dn = f();
                leaf->data(i, j) = keep;
                const double fd = (up - dn) / (2.0 * h), a = leaf->grad(i, j);
                worst = std::max(worst, std::abs(a - fd) / (1.0 + std::max(std::abs(a), std::abs(fd))));
            }
    std::printf("  FD gradcheck, blocked attention: worst relative error %.3e\n", worst);
    CHECK(worst < 5e-3);
}

}  // namespace

int main() {
    std::printf("blocked attention\n");
    test_equals_per_sequence(true);
    test_equals_per_sequence(false);
    test_gradcheck();
    std::printf("test_attention_blocked: all checks passed\n");
    return 0;
}
