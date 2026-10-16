// Post-training losses against finite differences and their parity anchors
// (docs/decisions/0001-posttrain-charter.md §4). Phase 3: the SFT loss mask.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/posttrain/grpo.hpp"

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

// ---- Phase 4: GRPO ---------------------------------------------------------

Matrix log_softmax_rows(const Matrix& L) {
    Matrix out(L.rows(), L.cols());
    for (size_t i = 0; i < L.rows(); ++i) {
        double mx = -1e300, z = 0.0;
        for (size_t j = 0; j < L.cols(); ++j) mx = std::max(mx, static_cast<double>(L(i, j)));
        for (size_t j = 0; j < L.cols(); ++j) z += std::exp(L(i, j) - mx);
        for (size_t j = 0; j < L.cols(); ++j) out(i, j) = static_cast<float>(L(i, j) - mx - std::log(z));
    }
    return out;
}

void test_pgkl_gradcheck() {
    Var x = make_var(randn(5, 6, 7), true);
    const std::vector<int> t = {1, 4, 0, 5, 2};
    const std::vector<float> w = {0.7f, -1.3f, 0.0f, 0.4f, 0.0f};  // signed advantages; a prompt row
    const std::vector<float> k = {0.2f, 0.2f, 0.0f, 0.5f, 0.3f};   // a KL-only row
    const Matrix ref = log_softmax_rows(randn(5, 6, 8));
    auto f = [&]() { return ops::policy_gradient_kl(x, t, w, k, ref)->data(0, 0); };
    microtorch::backward(ops::policy_gradient_kl(x, t, w, k, ref));
    const double worst = fd_worst(f, x, x->grad);
    std::printf("  FD gradcheck, policy_gradient_kl: worst relative error %.3e\n", worst);
    CHECK(worst < 5e-3);
    for (size_t j = 0; j < 6; ++j) CHECK(x->grad(2, j) == 0.0f);  // w = k = 0: no gradient
}

void test_pgkl_anchors() {
    // k = 0, w = 1, one row: the loss is the cross-entropy of that row.
    const Matrix L = randn(1, 9, 9);
    Var a = make_var(L, true);
    const float pg = ops::policy_gradient_kl(a, {3}, {1.0f}, {0.0f}, Matrix(1, 9))->data(0, 0);
    Var b = make_var(L, true);
    const float ce = ops::cross_entropy(b, {3})->data(0, 0);
    CHECK(std::fabs(pg - ce) < 1e-5f);
    // ref = the policy itself: KL is 0 and so is its gradient.
    Var c = make_var(L, true);
    Var l = ops::policy_gradient_kl(c, {3}, {0.0f}, {1.0f}, log_softmax_rows(L));
    CHECK(std::fabs(l->data(0, 0)) < 1e-6f);
    microtorch::backward(l);
    for (size_t j = 0; j < 9; ++j) CHECK(std::fabs(c->grad(0, j)) < 1e-6f);
    std::printf("  anchors: k=0 equals cross_entropy; KL to itself is 0 with zero gradient\n");
}

void test_bandit_reaches_the_analytic_optimum() {
    // Two actions, rewards r = (1, 0), reference pi_ref = (0.7, 0.3), beta.
    // The KL-regularised objective E_pi[r] - beta KL(pi || pi_ref) has the
    // closed-form optimum pi* proportional to pi_ref exp(r / beta). The
    // exact-expectation surrogate (every action weighted by its probability,
    // advantage = r - E_pi[r], std-normalisation off) has gradient -grad J,
    // so gradient descent on it must land on pi*.
    const double beta = 0.5, r0 = 1.0, r1 = 0.0;
    Matrix ref(1, 2);
    ref(0, 0) = static_cast<float>(std::log(0.7));
    ref(0, 1) = static_cast<float>(std::log(0.3));
    const double s0 = 0.7 * std::exp(r0 / beta), s1 = 0.3 * std::exp(r1 / beta);
    const double star = s0 / (s0 + s1);
    Var th = make_var(Matrix(1, 2), true);  // logits start at 0: pi = (0.5, 0.5)
    double p0 = 0.5;
    for (int step = 0; step < 4000; ++step) {
        const double e0 = std::exp(static_cast<double>(th->data(0, 0)));
        const double e1 = std::exp(static_cast<double>(th->data(0, 1)));
        p0 = e0 / (e0 + e1);
        const double mean = p0 * r0 + (1 - p0) * r1;
        Var loss = ops::add(ops::add(ops::policy_gradient_kl(th, {0}, {static_cast<float>(p0 * (r0 - mean))}, {0.0f}, ref),
                                     ops::policy_gradient_kl(th, {1}, {static_cast<float>((1 - p0) * (r1 - mean))}, {0.0f}, ref)),
                            ops::policy_gradient_kl(th, {0}, {0.0f}, {static_cast<float>(beta)}, ref));
        th->grad = Matrix(1, 2);
        microtorch::backward(loss);
        for (size_t j = 0; j < 2; ++j) th->data(0, j) -= 0.5f * th->grad(0, j);
    }
    std::printf("  bandit: pi(0) = %.6f, analytic optimum %.6f, |diff| %.2e\n", p0, star, std::fabs(p0 - star));
    CHECK(std::fabs(p0 - star) < 1e-3);
}

void test_group_advantages() {
    using namespace microtorch::posttrain;
    GroupStats st;
    // G = 1: every group is flat, every advantage 0.
    auto A = group_advantages({1.0f, 0.0f, 1.0f}, 1, true, &st);
    CHECK(st.flat_groups == 3 && A == std::vector<float>(3, 0.0f));
    // A known group: rewards (1, 0, 0, 1), mean 0.5, population SD 0.5.
    A = group_advantages({1, 0, 0, 1, 1, 1, 1, 1}, 4, true, &st, 0.0f);
    CHECK(st.groups == 2 && st.flat_groups == 1);
    CHECK(A[0] == 1.0f && A[1] == -1.0f && A[2] == -1.0f && A[3] == 1.0f && A[4] == 0.0f);
    A = group_advantages({1, 0, 0, 1}, 4, false, &st);
    CHECK(A[0] == 0.5f && A[1] == -0.5f);
    // Shuffled: the same values within each group, attached to other members.
    std::vector<float> r;
    for (int i = 0; i < 64; ++i) r.push_back(static_cast<float>(i % 3) / 2.0f);
    const auto N = group_advantages(r, 8, true, &st);
    const auto S = group_advantages(r, 8, true, &st, 1e-6f, AdvantageMode::Shuffled, 3);
    bool moved = false;
    for (size_t g = 0; g < r.size(); g += 8) {
        std::vector<float> a(N.begin() + g, N.begin() + g + 8), b(S.begin() + g, S.begin() + g + 8);
        moved = moved || a != b;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        CHECK(a == b);
    }
    CHECK(moved);
    bool threw = false;
    try {
        group_advantages({1.0f, -1.0f}, 2, true);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    std::printf("  group advantages: flat groups, known values, shuffle preserves each group's values\n");
}

// ---- Phase 5: provenance-weighted DPO ---------------------------------------

// Three pairs over 10 rows: pair 0 = rows 0-1 chosen, 2-3 rejected; pair 1 =
// rows 4 chosen, 5 rejected; pair 2 = rows 6-7 chosen, 8 rejected; row 9 is
// a prompt row (ignored).
struct DpoCase {
    std::vector<int> targets = {1, 2, 3, 0, 4, 4, 2, 1, 0, 3};
    std::vector<int> pair = {0, 0, 0, 0, 1, 1, 2, 2, 2, -1};
    std::vector<int8_t> sign = {1, 1, -1, -1, 1, -1, 1, 1, -1, 1};
    std::vector<double> ref = {0.3, -0.7, 0.1};
};

void test_dpo_gradcheck() {
    DpoCase c;
    Var x = make_var(randn(10, 5, 21), true);
    const std::vector<float> w = {1.0f, 0.3f, 0.5f};
    auto f = [&]() { return ops::dpo_weighted(x, c.targets, c.pair, c.sign, c.ref, w, 0.8f)->data(0, 0); };
    microtorch::backward(ops::dpo_weighted(x, c.targets, c.pair, c.sign, c.ref, w, 0.8f));
    const double worst = fd_worst(f, x, x->grad);
    std::printf("  FD gradcheck, dpo_weighted: worst relative error %.3e\n", worst);
    CHECK(worst < 5e-3);
    for (size_t j = 0; j < 5; ++j) CHECK(x->grad(9, j) == 0.0f);  // ignored row
}

void test_dpo_binary_weights_equal_dpo_on_kept_pairs() {
    DpoCase c;
    const Matrix L = randn(10, 5, 22);
    Var full = make_var(L, true);
    Var lf = ops::dpo_weighted(full, c.targets, c.pair, c.sign, c.ref, {1.0f, 0.0f, 1.0f}, 0.5f);
    // The same pairs 0 and 2 alone, renumbered 0 and 1, unweighted.
    const std::vector<size_t> rows = {0, 1, 2, 3, 6, 7, 8};
    Matrix K(rows.size(), 5);
    std::vector<int> t, pr;
    std::vector<int8_t> sg;
    for (size_t r = 0; r < rows.size(); ++r) {
        for (size_t j = 0; j < 5; ++j) K(r, j) = L(rows[r], j);
        t.push_back(c.targets[rows[r]]);
        pr.push_back(c.pair[rows[r]] == 0 ? 0 : 1);
        sg.push_back(c.sign[rows[r]]);
    }
    Var sub = make_var(K, true);
    Var ls = ops::dpo_weighted(sub, t, pr, sg, {c.ref[0], c.ref[2]}, {1.0f, 1.0f}, 0.5f);
    CHECK(bit_equal(lf->data(0, 0), ls->data(0, 0)));
    microtorch::backward(lf);
    microtorch::backward(ls);
    for (size_t r = 0; r < rows.size(); ++r)
        for (size_t j = 0; j < 5; ++j) CHECK(bit_equal(full->grad(rows[r], j), sub->grad(r, j)));
    for (size_t j = 0; j < 5; ++j) CHECK(full->grad(4, j) == 0.0f && full->grad(5, j) == 0.0f);
    std::printf("  dpo: weights {1,0,1} equal plain DPO on the kept pairs bit for bit; weight-0 rows get no gradient\n");
}

void test_dpo_at_the_reference() {
    // When the policy margin equals the reference margin, margin_p = 0:
    // the loss is log 2 and dL/dlogits_r = -(beta / (2 W)) w_p sign_r (onehot - p).
    DpoCase c;
    const Matrix L = randn(10, 5, 23);
    Var x = make_var(L, true);
    const Matrix lp = log_softmax_rows(L);
    std::vector<double> ref(3, 0.0);
    for (size_t r = 0; r < 10; ++r)
        if (c.pair[r] >= 0) ref[c.pair[r]] += c.sign[r] * static_cast<double>(lp(r, c.targets[r]));
    const float beta = 0.7f;
    const std::vector<float> w = {2.0f, 1.0f, 1.0f};
    Var l = ops::dpo_weighted(x, c.targets, c.pair, c.sign, ref, w, beta);
    CHECK(std::fabs(l->data(0, 0) - std::log(2.0)) < 1e-5);
    microtorch::backward(l);
    double worst = 0.0;
    for (size_t r = 0; r < 9; ++r)
        for (size_t j = 0; j < 5; ++j) {
            const double pr = std::exp(static_cast<double>(lp(r, j)));
            const double want = -(beta / (2.0 * 4.0)) * w[c.pair[r]] * c.sign[r] *
                                ((static_cast<int>(j) == c.targets[r] ? 1.0 : 0.0) - pr);
            worst = std::max(worst, std::fabs(want - x->grad(r, j)));
        }
    std::printf("  dpo at the reference: loss log 2, gradient matches the closed form (max err %.2e)\n", worst);
    CHECK(worst < 1e-6);
}

void test_dpo_rejects_bad_weights() {
    DpoCase c;
    Var x = make_var(randn(10, 5, 24), true);
    auto throws = [&](const std::vector<float>& w) {
        try {
            ops::dpo_weighted(x, c.targets, c.pair, c.sign, c.ref, w, 0.5f);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    CHECK(throws({0.0f, 0.0f, 0.0f}));
    CHECK(throws({1.0f, -0.1f, 1.0f}));
    CHECK(!throws({0.0f, 0.3f, 0.0f}));
}

}  // namespace

int main() {
    std::printf("posttrain losses\n");
    test_all_ones_is_bit_identical();
    test_equals_cross_entropy_on_kept_rows();
    test_gradcheck();
    test_rejects_bad_masks();
    test_pgkl_gradcheck();
    test_pgkl_anchors();
    test_bandit_reaches_the_analytic_optimum();
    test_group_advantages();
    test_dpo_gradcheck();
    test_dpo_binary_weights_equal_dpo_on_kept_pairs();
    test_dpo_at_the_reference();
    test_dpo_rejects_bad_weights();
    std::printf("test_posttrain_losses: all checks passed\n");
    return 0;
}
