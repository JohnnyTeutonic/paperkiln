// Post-training losses against finite differences and their parity anchors
// (docs/decisions/0001-posttrain-charter.md §4). Phase 3: the SFT loss mask.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "microtorch/ops.hpp"

using microtorch::make_var;
using microtorch::Var;
namespace ops = microtorch::ops;

namespace {

Matrix randn(size_t r, size_t c, unsigned seed, float scale = 1.0f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, scale);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(gen);
    return m;
}

// Central differences in double over a float forward; relative error as in
// tests/test_gradcheck.cpp.
double fd_worst(const std::function<float()>& f, Var leaf, const Matrix& analytic, float h = 1e-2f) {
    double worst = 0.0;
    for (size_t i = 0; i < leaf->data.rows(); ++i)
        for (size_t j = 0; j < leaf->data.cols(); ++j) {
            const float keep = leaf->data(i, j);
            microtorch::NoGrad ng;
            leaf->data(i, j) = keep + h;
            const float up = f();
            leaf->data(i, j) = keep - h;
            const float dn = f();
            leaf->data(i, j) = keep;
            const double fd = (static_cast<double>(up) - dn) / (2.0 * h);
            const double a = analytic(i, j);
            worst = std::max(worst, std::abs(a - fd) / (1.0 + std::max(std::abs(a), std::abs(fd))));
        }
    return worst;
}

bool bit_equal(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

void test_all_ones_is_bit_identical() {
    const Matrix L = randn(9, 11, 1);
    const std::vector<int> t = {0, 3, 10, 5, 5, 7, 1, 2, 9};
    Var a = make_var(L, true), b = make_var(L, true);
    Var la = ops::cross_entropy(a, t);
    Var lb = ops::cross_entropy_masked(b, t, std::vector<uint8_t>(9, 1));
    CHECK(bit_equal(la->data(0, 0), lb->data(0, 0)));
    microtorch::backward(la);
    microtorch::backward(lb);
    for (size_t i = 0; i < 9; ++i)
        for (size_t j = 0; j < 11; ++j) CHECK(bit_equal(a->grad(i, j), b->grad(i, j)));
    std::printf("  all-ones mask: loss and gradient bit-identical to cross_entropy\n");
}

void test_equals_cross_entropy_on_kept_rows() {
    const Matrix L = randn(8, 7, 2);
    const std::vector<int> t = {1, 2, 3, 4, 5, 6, 0, 1};
    const std::vector<uint8_t> m = {0, 0, 0, 1, 1, 0, 1, 1};  // a prompt, then a completion
    Var full = make_var(L, true);
    Var lm = ops::cross_entropy_masked(full, t, m);
    std::vector<size_t> keep;
    for (size_t i = 0; i < m.size(); ++i)
        if (m[i]) keep.push_back(i);
    Matrix K(keep.size(), 7);
    std::vector<int> tk;
    for (size_t r = 0; r < keep.size(); ++r) {
        for (size_t j = 0; j < 7; ++j) K(r, j) = L(keep[r], j);
        tk.push_back(t[keep[r]]);
    }
    Var sub = make_var(K, true);
    Var ls = ops::cross_entropy(sub, tk);
    CHECK(bit_equal(lm->data(0, 0), ls->data(0, 0)));
    microtorch::backward(lm);
    microtorch::backward(ls);
    for (size_t r = 0; r < keep.size(); ++r)
        for (size_t j = 0; j < 7; ++j) CHECK(bit_equal(full->grad(keep[r], j), sub->grad(r, j)));
    for (size_t i = 0; i < m.size(); ++i)
        if (!m[i])
            for (size_t j = 0; j < 7; ++j) CHECK(full->grad(i, j) == 0.0f);
    std::printf("  masked: equals cross_entropy on the kept rows bit for bit; masked rows get zero gradient\n");
}

void test_gradcheck() {
    Var x = make_var(randn(6, 5, 3), true);
    const std::vector<int> t = {4, 0, 2, 2, 1, 3};
    const std::vector<uint8_t> m = {0, 1, 0, 1, 1, 0};
    auto f = [&]() { return ops::cross_entropy_masked(x, t, m)->data(0, 0); };
    Var l = ops::cross_entropy_masked(x, t, m);
    microtorch::backward(l);
    const double w = fd_worst(f, x, x->grad);
    std::printf("  FD gradcheck, masked CE: worst relative error %.3e\n", w);
    CHECK(w < 5e-3);
}

void test_rejects_bad_masks() {
    Var x = make_var(randn(3, 4, 4), true);
    auto throws = [&](const std::vector<uint8_t>& m, const std::vector<int>& t) {
        try {
            ops::cross_entropy_masked(x, t, m);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    CHECK(throws({0, 0, 0}, {1, 1, 1}));     // nothing to learn from
    CHECK(throws({1, 1}, {1, 1, 1}));        // wrong length
    CHECK(throws({1, 2, 0}, {1, 1, 1}));     // not binary
    CHECK(throws({1, 0, 0}, {9, 1, 1}));     // kept target out of range
    CHECK(!throws({1, 0, 0}, {1, 99, 99}));  // a masked row's target is never read
}

}  // namespace

int main() {
    std::printf("posttrain losses\n");
    test_all_ones_is_bit_identical();
    test_equals_cross_entropy_on_kept_rows();
    test_gradcheck();
    test_rejects_bad_masks();
    std::printf("test_posttrain_losses: all checks passed\n");
    return 0;
}
