// RoPE head coverage (spec arch.rope_heads):
//   1. "first" (all_heads=false) is bit-for-bit the pre-fix apply_rope,
//      pinned against a verbatim copy of that loop;
//   2. "all" equals the single-head rotation applied to each head on its
//      own, so head 0 is unchanged and every other head gets the same
//      frequency basis;
//   3. the relative-position property per head: under "all", the score
//      q_m . k_n of EVERY head depends only on m - n; under "first" it does
//      for head 0 and not for head 1 (the defect, measured);
//   4. nn::Llama: one head -> the two settings agree bit-for-bit; several
//      heads -> they differ, and rope_all_heads defaults to true.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/llama.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

namespace {

Matrix randn(size_t r, size_t c, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(gen);
    return m;
}

// The forward loop of ops::apply_rope as it stood before arch.rope_heads
// (src/ops.cpp, commit 0151ba3), copied verbatim: rotates columns
// [0, head_dim) of q and of k only.
Matrix legacy_rope(const Matrix& in, const std::vector<int>& pos, float theta_base,
                   size_t head_dim) {
    const size_t T = in.rows(), d = in.cols() / 3;
    Matrix out = in;
    for (size_t i = 0; i < T; ++i) {
        const float m = static_cast<float>(pos[i]);
        for (size_t start = 0; start < 2 * d; start += d) {
            for (size_t dim = 0; dim < head_dim; dim += 2) {
                const float inv_freq =
                    1.0f / std::pow(theta_base, static_cast<float>(dim) / head_dim);
                const float theta = m * inv_freq;
                const float cos_t = std::cos(theta);
                const float sin_t = std::sin(theta);
                const float x0 = out(i, start + dim);
                const float x1 = out(i, start + dim + 1);
                out(i, start + dim) = x0 * cos_t - x1 * sin_t;
                out(i, start + dim + 1) = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    return out;
}

// The legacy backward (inverse rotation of head 0's columns).
Matrix legacy_rope_grad(const Matrix& g, const std::vector<int>& pos, float theta_base,
                        size_t head_dim) {
    std::vector<int> neg(pos.size());
    for (size_t i = 0; i < pos.size(); ++i) neg[i] = -pos[i];
    return legacy_rope(g, neg, theta_base, head_dim);
}

bool bit_equal(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j)
            if (a(i, j) != b(i, j)) return false;
    return true;
}

}  // namespace

int main() {
    device::set_from_env();
    const size_t T = 9, H = 4, dk = 8, d = H * dk;
    const float theta = 10000.0f;
    std::vector<int> pos(T);
    for (size_t i = 0; i < T; ++i) pos[i] = static_cast<int>(3 * i + 1);
    const Matrix x = randn(T, 3 * d, 11);

    printf("=== first == pre-fix apply_rope, bit for bit ===\n");
    {
        Var in = make_var(x, true);
        Var out = ops::apply_rope(in, pos, theta, dk, /*all_heads=*/false);
        CHECK(bit_equal(out->data, legacy_rope(x, pos, theta, dk)));
        // Backward: an arbitrary upstream gradient through a weighted mean;
        // the input gradient must be the legacy inverse rotation of the
        // gradient that reached the output.
        const Matrix w = randn(T, 3 * d, 12);
        backward(ops::mean(ops::mul(out, make_var(w, false))));
        CHECK(out->grad.rows() == T);
        CHECK(bit_equal(in->grad, legacy_rope_grad(out->grad, pos, theta, dk)));
        printf("  forward and backward identical to the legacy loop\n");
    }

    printf("=== all == the one-head rotation applied per head ===\n");
    Matrix all_out;
    {
        Var in = make_var(x, false);
        all_out = ops::apply_rope(in, pos, theta, dk, /*all_heads=*/true)->data;
        for (size_t h = 0; h < H; ++h) {
            Matrix one(T, 3 * dk);  // head h's q, k, v as a fused one-head qkv
            for (size_t i = 0; i < T; ++i)
                for (size_t part = 0; part < 3; ++part)
                    for (size_t c = 0; c < dk; ++c)
                        one(i, part * dk + c) = x(i, part * d + h * dk + c);
            const Matrix r = legacy_rope(one, pos, theta, dk);
            for (size_t i = 0; i < T; ++i)
                for (size_t part = 0; part < 3; ++part)
                    for (size_t c = 0; c < dk; ++c)
                        CHECK(all_out(i, part * d + h * dk + c) == r(i, part * dk + c));
        }
        // Head 0 is what "first" produced.
        const Matrix first = legacy_rope(x, pos, theta, dk);
        for (size_t i = 0; i < T; ++i)
            for (size_t part = 0; part < 3; ++part)
                for (size_t c = 0; c < dk; ++c)
                    CHECK(all_out(i, part * d + c) == first(i, part * d + c));
        printf("  %zu heads each rotated as head 0; v untouched\n", H);
    }

    printf("=== relative position, per head ===\n");
    {
        // One q row and one k row repeated at every position: a correct RoPE
        // makes q_m . k_n a function of (m - n) alone.
        const Matrix base = randn(1, 3 * d, 13);
        const size_t N = 12;
        Matrix rep(N, 3 * d);
        std::vector<int> p(N);
        for (size_t i = 0; i < N; ++i) {
            p[i] = static_cast<int>(i);
            for (size_t j = 0; j < 3 * d; ++j) rep(i, j) = base(0, j);
        }
        auto score = [&](const Matrix& r, size_t h, size_t m, size_t n) {
            double s = 0;
            for (size_t c = 0; c < dk; ++c) s += double(r(m, h * dk + c)) * r(n, d + h * dk + c);
            return s;
        };
        // max over heads h of |score(m, n) - score(m + 5, n + 5)|
        auto drift = [&](const Matrix& r, size_t h) {
            double worst = 0;
            for (size_t m = 0; m + 5 < N; ++m)
                for (size_t n = 0; n + 5 < N; ++n)
                    worst =
                        std::max(worst, std::abs(score(r, h, m, n) - score(r, h, m + 5, n + 5)));
            return worst;
        };
        auto range = [&](const Matrix& r, size_t h) {  // does the score vary with m - n at all?
            double lo = 1e30, hi = -1e30;
            for (size_t m = 0; m < N; ++m) {
                lo = std::min(lo, score(r, h, m, 0));
                hi = std::max(hi, score(r, h, m, 0));
            }
            return hi - lo;
        };
        const Matrix ra = ops::apply_rope(make_var(rep, false), p, theta, dk, true)->data;
        const Matrix rf = ops::apply_rope(make_var(rep, false), p, theta, dk, false)->data;
        for (size_t h = 0; h < H; ++h) {
            CHECK(drift(ra, h) < 1e-3);
            CHECK(range(ra, h) > 1e-2);
            printf("  all,   head %zu: shift drift %.2e, score range %.3f\n", h, drift(ra, h),
                   range(ra, h));
        }
        CHECK(drift(rf, 0) < 1e-3 && range(rf, 0) > 1e-2);
        CHECK(range(rf, 1) < 1e-9);  // no position signal at all in head 1
        printf("  first, head 1: score range %.2e (position-blind)\n", range(rf, 1));
    }

    printf("=== nn::Llama ===\n");
    {
        nn::LlamaConfig cfg;
        CHECK(cfg.rope_all_heads);  // new models rotate every head
        cfg.vocab = 64;
        cfg.d = 16;
        cfg.n_layers = 2;
        cfg.d_ff = 32;
        cfg.n_ctx = 16;
        const std::vector<int> ids = {3, 9, 27, 1, 44, 44, 7, 12};
        NoGrad ng;
        for (size_t heads : {size_t(1), size_t(4)}) {
            cfg.n_heads = heads;
            cfg.rope_all_heads = true;
            nn::Llama a(cfg, 5);
            cfg.rope_all_heads = false;
            nn::Llama f(cfg, 5);
            const bool same = bit_equal(a.forward(ids)->data, f.forward(ids)->data);
            CHECK(same == (heads == 1));
            printf("  H=%zu: all vs first logits %s\n", heads, same ? "identical" : "differ");
        }
    }
    printf("rope_heads: all checks passed\n");
    return 0;
}
