// The cue-selective twin loss (include/microtorch/posttrain/twin.hpp; lane amendment A31) against a
// direct computation in double and against finite differences. Claude Opus 5.5, 13 Nov 2026.
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/posttrain/twin.hpp"

using microtorch::make_var;
using microtorch::Var;
namespace pt = microtorch::posttrain;

namespace {

Matrix randn(size_t r, size_t c, unsigned seed, float scale = 1.0f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, scale);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(gen);
    return m;
}

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

constexpr size_t L = 6, C = 7;

// G groups of 5 rows, each row L positions. Group g's reply yA has na[g] targets and yB nb[g]; a
// reply's targets sit at the end of its row's real length, which differs by row (contexts differ).
struct Case {
    pt::TwinTargets t;
    std::vector<std::vector<int>> ya, yb;
};

Case make_case(int per_group) {
    Case c;
    const std::vector<int> na = {2, 3, 1}, nb = {1, 2, 3}, len = {6, 5, 4, 6, 5};
    c.t.groups = na.size();
    c.t.per_group = per_group;
    c.t.rows = c.t.groups * per_group;
    std::mt19937 gen(5);
    for (size_t g = 0; g < c.t.groups; ++g) {
        std::vector<int> a, b;
        for (int i = 0; i < na[g]; ++i) a.push_back(static_cast<int>(gen() % C));
        for (int i = 0; i < nb[g]; ++i) b.push_back(static_cast<int>(gen() % C));
        c.ya.push_back(a);
        c.yb.push_back(b);
        c.t.n_a.push_back(static_cast<float>(na[g]));
        c.t.n_b.push_back(static_cast<float>(nb[g]));
        for (int j = 0; j < per_group; ++j) {
            const auto& y = (j == 0 || j == 1 || j == 4) ? a : b;
            const size_t r = g * per_group + j;
            const size_t end = static_cast<size_t>(len[(g + j) % len.size()]);
            for (size_t i = 0; i < y.size(); ++i) {
                c.t.pos.push_back(static_cast<int>(r * L + end - y.size() + i));
                c.t.tgt.push_back(y[i]);
                c.t.pos_row.push_back(static_cast<int>(r));
            }
        }
    }
    return c;
}

// The loss in double, straight from the definition.
double reference(const Matrix& x, const Case& c, double m, double lam, double mu, std::vector<double>* d_out = nullptr) {
    std::vector<double> lp(c.t.rows, 0.0);
    for (size_t i = 0; i < c.t.pos.size(); ++i) {
        const size_t r = c.t.pos[i];
        double mx = -1e300, z = 0;
        for (size_t j = 0; j < C; ++j) mx = std::max(mx, static_cast<double>(x(r, j)));
        for (size_t j = 0; j < C; ++j) z += std::exp(x(r, j) - mx);
        lp[c.t.pos_row[i]] += x(r, c.t.tgt[i]) - mx - std::log(z);
    }
    const size_t G = c.t.groups, k = c.t.per_group;
    double ha = 0, hb = 0, inv = 0;
    for (size_t g = 0; g < G; ++g) {
        const double dA = (lp[g * k] - lp[g * k + 1]) / c.t.n_a[g];
        const double dB = (lp[g * k + 2] - lp[g * k + 3]) / c.t.n_b[g];
        ha += std::max(0.0, m - dA);
        hb += std::max(0.0, m - dB);
        if (d_out) {
            d_out->push_back(dA);
            d_out->push_back(dB);
        }
        if (k == 5) {
            const double dN = (lp[g * k] - lp[g * k + 4]) / c.t.n_a[g];
            inv += std::abs(dN);
            if (d_out) d_out->push_back(dN);
        }
    }
    return lam * (ha / G + hb / G) + mu * inv / G;
}

void test_value_and_gradient() {
    for (int per_group : {4, 5}) {
        const Case c = make_case(per_group);
        const Matrix X = randn(c.t.rows * L, C, 11, 2.0f);
        std::vector<double> d;
        reference(X, c, 0.0, 1.0, 0.0, &d);
        // a margin far from every hinge's kink, so finite differences see one side; |dN| also clear of 0
        std::vector<double> dd;
        for (size_t i = 0; i < d.size(); ++i)
            if (per_group == 4 || i % 3 != 2) dd.push_back(d[i]);
        std::sort(dd.begin(), dd.end());
        double m = 0, gap = -1;
        for (size_t i = 0; i + 1 < dd.size(); ++i)
            if (dd[i + 1] - dd[i] > gap) {
                gap = dd[i + 1] - dd[i];
                m = 0.5 * (dd[i] + dd[i + 1]);
            }
        CHECK(gap > 0.2);
        if (per_group == 5)
            for (size_t i = 2; i < d.size(); i += 3) CHECK(std::abs(d[i]) > 0.1);
        const float lam = 0.7f, mu = per_group == 5 ? 0.3f : 0.0f;
        Var x = make_var(X, true);
        pt::TwinStats st;
        Var loss = pt::twin_loss(x, c.t, static_cast<float>(m), lam, mu, &st);
        const double want = reference(X, c, m, lam, mu);
        std::printf("  per_group %d: loss %.6f, reference %.6f, active %.2f\n", per_group, loss->data(0, 0), want,
                    st.active);
        CHECK(std::abs(loss->data(0, 0) - want) < 1e-5 * (1 + std::abs(want)));
        CHECK(std::abs(st.loss - want) < 1e-5 * (1 + std::abs(want)));
        CHECK(st.active > 0.0 && st.active < 1.0);  // the case exercises both sides of the hinge
        CHECK(st.has_n == (per_group == 5));
        microtorch::backward(loss);
        auto f = [&]() { return pt::twin_loss(x, c.t, static_cast<float>(m), lam, mu)->data(0, 0); };
        const double worst = fd_worst(f, x, x->grad);
        std::printf("  FD gradcheck, twin loss (per_group %d): worst relative error %.3e\n", per_group, worst);
        CHECK(worst < 5e-3);
        // rows holding no reply target get no gradient
        std::vector<bool> used(c.t.rows * L, false);
        for (int p : c.t.pos) used[p] = true;
        for (size_t r = 0; r < used.size(); ++r)
            if (!used[r])
                for (size_t j = 0; j < C; ++j) CHECK(x->grad(r, j) == 0.0f);
    }
}

void test_met_margin_gives_no_gradient() {
    const Case c = make_case(4);
    Var x = make_var(randn(c.t.rows * L, C, 12, 2.0f), true);
    pt::TwinStats st;
    Var loss = pt::twin_loss(x, c.t, -1000.0f, 1.0f, 0.0f, &st);
    CHECK(loss->data(0, 0) == 0.0f && st.active == 0.0);
    microtorch::backward(loss);
    for (size_t i = 0; i < x->grad.rows(); ++i)
        for (size_t j = 0; j < C; ++j) CHECK(x->grad(i, j) == 0.0f);
    std::printf("  a margin every twin meets: loss 0, gradient 0\n");
}

void test_per_token_scale() {
    // the same reply scored twice over (every target repeated) halves nothing: dA is per token
    const Case c = make_case(4);
    const Matrix X = randn(c.t.rows * L, C, 13, 2.0f);
    std::vector<double> d;
    reference(X, c, 0.0, 1.0, 0.0, &d);
    pt::TwinStats st;
    pt::twin_loss(make_var(X, true), c.t, 0.0f, 1.0f, 0.0f, &st);
    double a = 0;
    for (size_t g = 0; g < c.t.groups; ++g) a += d[2 * g];
    CHECK(std::abs(st.d_a - a / c.t.groups) < 1e-5);
    std::printf("  the logged decisive shift is the per-token mean\n");
}

void test_refusals() {
    const Case c4 = make_case(4), c5 = make_case(5);
    Var x = make_var(randn(c5.t.rows * L, C, 14), true);
    auto refuses = [](const std::function<void()>& f) {
        try {
            f();
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    CHECK(refuses([&] { pt::twin_loss(x, c5.t, 1.0f, 0.0f, 0.0f); }));   // both weights 0
    CHECK(refuses([&] { pt::twin_loss(x, c4.t, 1.0f, 1.0f, 0.5f); }));   // mu without the nuisance row
    CHECK(refuses([&] { pt::twin_loss(x, c5.t, 1.0f, -1.0f, 0.5f); }));  // negative weight
    pt::TwinTargets bad = c5.t;
    bad.pos.pop_back();  // the nuisance row of the last group loses a target
    bad.tgt.pop_back();
    bad.pos_row.pop_back();
    CHECK(refuses([&] { pt::twin_loss(x, bad, 1.0f, 1.0f, 0.5f); }));
    bad = c5.t;
    bad.n_a[0] += 1;  // a length that does not match the targets
    CHECK(refuses([&] { pt::twin_loss(x, bad, 1.0f, 1.0f, 0.5f); }));
    bad = c5.t;
    bad.pos[0] = static_cast<int>(c5.t.rows * L);  // out of range
    CHECK(refuses([&] { pt::twin_loss(x, bad, 1.0f, 1.0f, 0.5f); }));
    bad = c5.t;
    bad.pos[0] = static_cast<int>(L);  // row 0's target placed in row 1's block
    CHECK(refuses([&] { pt::twin_loss(x, bad, 1.0f, 1.0f, 0.5f); }));
    bad = c5.t;
    bad.tgt[0] = static_cast<int>(C);  // past the vocabulary
    CHECK(refuses([&] { pt::twin_loss(x, bad, 1.0f, 1.0f, 0.5f); }));
    bad = c5.t;
    bad.n_b[1] = std::numeric_limits<float>::infinity();
    CHECK(refuses([&] { pt::twin_loss(x, bad, 1.0f, 1.0f, 0.5f); }));
    CHECK(refuses([&] { pt::twin_loss(x, c5.t, std::numeric_limits<float>::quiet_NaN(), 1.0f, 0.5f); }));   // margin
    CHECK(refuses([&] { pt::twin_loss(x, c5.t, 1.0f, std::numeric_limits<float>::infinity(), 0.5f); }));    // lambda
    std::printf("  refusals: zero weights, mu without N, negative or non-finite weight or margin, unpaired targets, bad length,\n"
                "    bad position, a target outside its row's block, a target id past the vocabulary\n");
}

// A34's own-context floor (Claude Opus 5.5, 15 Nov 2026): the per-token own-context terms lp(row 0)/n_A and
// lp(row 2)/n_B of each group, in double.
std::vector<double> own_terms(const Matrix& x, const Case& c) {
    std::vector<double> lp(c.t.rows, 0.0);
    for (size_t i = 0; i < c.t.pos.size(); ++i) {
        const size_t r = c.t.pos[i];
        double mx = -1e300, z = 0;
        for (size_t j = 0; j < C; ++j) mx = std::max(mx, static_cast<double>(x(r, j)));
        for (size_t j = 0; j < C; ++j) z += std::exp(x(r, j) - mx);
        lp[c.t.pos_row[i]] += x(r, c.t.tgt[i]) - mx - std::log(z);
    }
    std::vector<double> o;
    for (size_t g = 0; g < c.t.groups; ++g) {
        o.push_back(lp[g * c.t.per_group] / c.t.n_a[g]);
        o.push_back(lp[g * c.t.per_group + 2] / c.t.n_b[g]);
    }
    return o;
}

void test_floor_value_and_gradient() {
    for (int per_group : {4, 5}) {
        Case c = make_case(per_group);
        const Matrix X = randn(c.t.rows * L, C, 21, 2.0f);
        const std::vector<double> own = own_terms(X, c);
        // references 0.4 above and 0.4 below the own terms by turns: both sides of the floor's hinge, each clear of its kink
        for (size_t g = 0; g < c.t.groups; ++g) {
            c.t.ref_a.push_back(static_cast<float>(own[2 * g] + ((g % 2) ? 0.4 : -0.4)));
            c.t.ref_b.push_back(static_cast<float>(own[2 * g + 1] + ((g % 2) ? -0.4 : 0.4)));
        }
        const float lam = 0.7f, mu = per_group == 5 ? 0.3f : 0.0f, fl = 1.3f, m = -1000.0f;  // margin met: the floor alone moves
        double fa = 0, fb = 0;
        for (size_t g = 0; g < c.t.groups; ++g) {
            fa += std::max(0.0, c.t.ref_a[g] - own[2 * g]);
            fb += std::max(0.0, c.t.ref_b[g] - own[2 * g + 1]);
        }
        const double want = reference(X, c, m, lam, mu) + fl * (fa + fb) / c.t.groups;
        Var x = make_var(X, true);
        pt::TwinStats st;
        Var loss = pt::twin_loss(x, c.t, m, lam, mu, &st, fl);
        std::printf("  floor, per_group %d: loss %.6f, reference %.6f, floor active %.2f, gap %+.3f\n", per_group,
                    loss->data(0, 0), want, st.floor_active, st.floor_gap);
        CHECK(std::abs(loss->data(0, 0) - want) < 1e-5 * (1 + std::abs(want)));
        CHECK(st.has_floor && std::abs(st.floor_loss - fl * (fa + fb) / c.t.groups) < 1e-5);
        CHECK(st.floor_active == 0.5);
        CHECK(std::abs(st.floor_gap) < 1e-5 && std::abs(st.floor_gap_absmax - 0.4) < 1e-5);
        microtorch::backward(loss);
        auto f = [&]() { return pt::twin_loss(x, c.t, m, lam, mu, nullptr, fl)->data(0, 0); };
        const double worst = fd_worst(f, x, x->grad);
        std::printf("  FD gradcheck, floor (per_group %d): worst relative error %.3e\n", per_group, worst);
        CHECK(worst < 5e-3);
        // only own-context rows (0 and 2) of groups below their reference carry gradient (rows 1, 3 and 4 none: margin met,
        // mu's term aside); with mu = 0, rows 1, 3 get exactly zero
        if (per_group == 4) {
            for (size_t i = 0; i < c.t.pos.size(); ++i) {
                const size_t r = c.t.pos_row[i], g = r / 4, j = r % 4;
                const bool below = (j == 0 && c.t.ref_a[g] > own[2 * g]) || (j == 2 && c.t.ref_b[g] > own[2 * g + 1]);
                double sum = 0;
                for (size_t q = 0; q < C; ++q) sum += std::abs(x->grad(c.t.pos[i], q));
                CHECK(below ? sum > 0 : sum == 0);
            }
        }
    }
}

void test_floor_zero_is_a31() {
    // floor 0, with references set or not, gives A31's loss and gradient bit for bit
    for (int per_group : {4, 5}) {
        Case c = make_case(per_group);
        const Matrix X = randn(c.t.rows * L, C, 22, 2.0f);
        const float lam = 0.3f, mu = per_group == 5 ? 0.3f : 0.0f, m = 1.0f;
        Var x0 = make_var(X, true);
        pt::TwinStats s0;
        Var l0 = pt::twin_loss(x0, c.t, m, lam, mu, &s0);
        microtorch::backward(l0);
        for (size_t g = 0; g < c.t.groups; ++g) {
            c.t.ref_a.push_back(5.0f);  // far above every own term: would bind if read
            c.t.ref_b.push_back(5.0f);
        }
        Var x1 = make_var(X, true);
        pt::TwinStats s1;
        Var l1 = pt::twin_loss(x1, c.t, m, lam, mu, &s1, 0.0f);
        microtorch::backward(l1);
        CHECK(l0->data(0, 0) == l1->data(0, 0));
        CHECK(!s1.has_floor && s0.d_a == s1.d_a && s0.d_b == s1.d_b && s0.active == s1.active);
        for (size_t i = 0; i < X.rows(); ++i)
            for (size_t j = 0; j < C; ++j) CHECK(x0->grad(i, j) == x1->grad(i, j));
    }
    std::printf("  floor 0: A31's loss and gradient bit for bit, references ignored\n");
}

void test_floor_refusals() {
    Case c = make_case(4);
    Var x = make_var(randn(c.t.rows * L, C, 23), true);
    auto refuses = [](const std::function<void()>& f) {
        try {
            f();
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    CHECK(refuses([&] { pt::twin_loss(x, c.t, 1.0f, 1.0f, 0.0f, nullptr, 1.0f); }));  // no references
    for (size_t g = 0; g < c.t.groups; ++g) {
        c.t.ref_a.push_back(-1.0f);
        c.t.ref_b.push_back(-1.0f);
    }
    CHECK(!refuses([&] { pt::twin_loss(x, c.t, 1.0f, 1.0f, 0.0f, nullptr, 1.0f); }));
    CHECK(refuses([&] { pt::twin_loss(x, c.t, 1.0f, 1.0f, 0.0f, nullptr, -1.0f); }));  // negative floor
    CHECK(refuses([&] { pt::twin_loss(x, c.t, 1.0f, 1.0f, 0.0f, nullptr, std::numeric_limits<float>::quiet_NaN()); }));
    Case bad = c;
    bad.t.ref_b.pop_back();  // one reference short
    CHECK(refuses([&] { pt::twin_loss(x, bad.t, 1.0f, 1.0f, 0.0f, nullptr, 1.0f); }));
    bad = c;
    bad.t.ref_a[1] = std::numeric_limits<float>::infinity();
    CHECK(refuses([&] { pt::twin_loss(x, bad.t, 1.0f, 1.0f, 0.0f, nullptr, 1.0f); }));
    std::printf("  floor refusals: no or short references, a non-finite reference, a negative or NaN floor\n");
}

}  // namespace

int main() {
    std::printf("twin loss\n");
    test_value_and_gradient();
    test_met_margin_gives_no_gradient();
    test_per_token_scale();
    test_refusals();
    test_floor_value_and_gradient();
    test_floor_zero_is_a31();
    test_floor_refusals();
    std::printf("test_twin_loss: all checks passed\n");
    return 0;
}
