// Batched attention computed block by block (ops::fused_attention with
// seq_len < rows). It must equal running each sequence through the
// single-sequence path, bit for bit, forward and backward, and pass a
// finite-difference gradcheck. With the device op set on (CUDA build,
// MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1) the blocked path runs in
// kernels that sum in a different order, so equality is to rounding, and a
// further leg pins the kernels to the host loops at the chat model's shape.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
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

bool on_device() {
    return microtorch::device::device_ops_enabled() && microtorch::device::get() == microtorch::device::Device::CUDA;
}

// Bit equality on the host path; on the device path, equality to rounding.
bool same(float a, float b) {
    if (on_device()) return std::abs(a - b) <= 1e-5f * (1.0f + std::abs(b));
    return std::memcmp(&a, &b, sizeof a) == 0;
}

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
    std::printf("  blocked == per-sequence, %s (causal=%d): output and all three gradients\n",
                on_device() ? "to rounding (device)" : "bit for bit", causal);
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

// Device kernels against the host loops at the chat model's shape (B = 4
// sequences of 256, head dim 64): output and all three gradients.
void test_device_matches_host(bool causal) {
    const size_t B = 4, sl = 256, d = 64;
    const Matrix Q = randn(B * sl, d, 11), K = randn(B * sl, d, 12), V = randn(B * sl, d, 13), G = randn(B * sl, d, 14);
    auto run = [&](bool dev, Matrix& y, Matrix& gq, Matrix& gk, Matrix& gv) {
        microtorch::device::set_device_ops(dev);
        Var q = make_var(Q, true), k = make_var(K, true), v = make_var(V, true);
        Var out = ops::fused_attention_blocked(q, k, v, 0.125f, sl, causal);
        microtorch::backward(ops::mean(ops::mul(out, make_var(G))));
        y = out->data, gq = q->grad, gk = k->grad, gv = v->grad;
    };
    Matrix y1, q1, k1, v1, y0, q0, k0, v0;
    run(true, y1, q1, k1, v1);
    run(false, y0, q0, k0, v0);
    microtorch::device::set_device_ops(true);
    double worst = 0.0;
    auto cmp = [&](const Matrix& a, const Matrix& b) {
        double scale = 0.0;
        for (size_t i = 0; i < b.rows(); ++i)
            for (size_t j = 0; j < b.cols(); ++j) scale = std::max(scale, (double)std::abs(b(i, j)));
        for (size_t i = 0; i < b.rows(); ++i)
            for (size_t j = 0; j < b.cols(); ++j)
                worst = std::max(worst, std::abs((double)a(i, j) - b(i, j)) / (scale + 1e-30));
    };
    cmp(y1, y0), cmp(q1, q0), cmp(k1, k0), cmp(v1, v0);
    std::printf("  device kernels vs host loops, B=4 sl=256 d=64 (causal=%d): worst error %.3e of each tensor's max\n",
                causal, worst);
    CHECK(worst < 1e-5);
}

// ops::attention_heads (all heads of a fused [T, 3d] qkv in one op on the
// device) against the composition it replaces: slices, fused_attention per
// head, concat. Bit for bit on either path, output and the qkv gradient,
// at shapes off the kernels' 64-wide tiles.
void test_heads_equal_per_head(bool causal) {
    const size_t B = 3, sl = 70, H = 3, dk = 20, d = H * dk, T = B * sl;
    const Matrix QKV = randn(T, 3 * d, 21), G = randn(T, d, 22);
    const float sc = 0.2f;
    Var a = make_var(QKV, true);
    Var ya = ops::attention_heads(a, H, sc, sl, causal);
    microtorch::backward(ops::mean(ops::mul(ya, make_var(G))));
    Var b = make_var(QKV, true);
    std::vector<Var> hs;
    for (size_t h = 0; h < H; ++h)
        hs.push_back(ops::fused_attention(ops::slice_cols(b, h * dk, (h + 1) * dk),
                                          ops::slice_cols(b, d + h * dk, d + (h + 1) * dk),
                                          ops::slice_cols(b, 2 * d + h * dk, 2 * d + (h + 1) * dk), sc, sl, causal));
    Var yb = ops::concat_cols(hs);
    microtorch::backward(ops::mean(ops::mul(yb, make_var(G))));
    microtorch::device::materialize(ya->data);
    microtorch::device::materialize(yb->data);
    microtorch::device::materialize(a->grad);
    microtorch::device::materialize(b->grad);
    CHECK(std::memcmp(ya->data.get_data(), yb->data.get_data(), T * d * sizeof(float)) == 0);
    CHECK(std::memcmp(a->grad.get_data(), b->grad.get_data(), T * 3 * d * sizeof(float)) == 0);
    std::printf("  attention_heads == per-head slices + concat, bit for bit%s (causal=%d)\n",
                on_device() ? " (device kernels)" : "", causal);
}

}  // namespace

int main() {
    microtorch::device::set_from_env();  // MICROTORCH_DEVICE / _DEVICE_OPS select the kernels
    std::printf("blocked attention%s\n", on_device() ? " (device op set)" : "");
    test_equals_per_sequence(true);
    test_equals_per_sequence(false);
    test_gradcheck();
    test_heads_equal_per_head(true);
    test_heads_equal_per_head(false);
    if (on_device()) {
        test_device_matches_host(true);
        test_device_matches_host(false);
    }
    std::printf("test_attention_blocked: all checks passed\n");
    return 0;
}
