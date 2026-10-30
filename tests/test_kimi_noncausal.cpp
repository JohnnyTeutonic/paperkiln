// Dense float64 oracle for the implemented channelwise mixer, NOT published Kimi/KDA.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <vector>
#include "check.hpp"
#include "microtorch/kimi_linear.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
using kimi::KimiLinearAttention;
using Vec = std::vector<double>;
struct Ref {
    Vec y, dq, dk, dv;
};
static double worst = 0, worst_fd = 0, worst_direction = 0;
static int components = 0, directions = 0;
static double phi(double x) {
    return x > 0 ? 1 + x : std::exp(x);
}
static double dphi(double x) {
    return x > 0 ? 1 : std::exp(x);
}
static double rounded_phi(double x) {
    float v = static_cast<float>(x);
    return v > 0 ? v + 1.f : (std::exp(v) - 1.f) + 1.f;
}
// Deliberately dense O(B*T*T*d): each query builds its own row of pair weights.
Ref dense(const Vec& q, const Vec& k, const Vec& v, const Vec& g, int B, int T, int D, bool causal,
          bool rounded = false) {
    Ref r;
    size_t n = q.size();
    r.y.resize(n);
    r.dq.resize(n);
    r.dk.resize(n);
    r.dv.resize(n);
    auto feature = [&](double x) { return rounded ? rounded_phi(x) : phi(x); };
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < D; ++j) {
                int i = (b * T + t) * D + j;
                double z = 0, num = 0;
                for (int s = 0; s < T; ++s)
                    if (!causal || s <= t) {
                        int a = (b * T + s) * D + j;
                        z += feature(k[a]);
                        num += feature(k[a]) * v[a];
                    }
                double den = std::max(z, 1e-8);
                r.y[i] = feature(q[i]) * num / den;
                r.dq[i] = g[i] * dphi(q[i]) * num / den;
                for (int s = 0; s < T; ++s)
                    if (!causal || s <= t) {
                        int a = (b * T + s) * D + j;
                        r.dk[a] += g[i] * feature(q[i]) * dphi(k[a]) *
                                   (v[a] / den - (z > 1e-8 ? num / (den * den) : 0));
                        r.dv[a] += g[i] * feature(q[i]) * feature(k[a]) / den;
                    }
            }
    return r;
}
Matrix matrix(const Vec& v, int D) {
    Matrix m(v.size() / D, D);
    for (size_t i = 0; i < v.size(); ++i) m(i / D, i % D) = v[i];
    return m;
}
Vec vector(const Matrix& m) {
    Vec v(m.rows() * m.cols());
    for (size_t i = 0; i < v.size(); ++i) v[i] = m(i / m.cols(), i % m.cols());
    return v;
}
void close(const Matrix& m, const Vec& v, double tol) {
    CHECK(m.rows() * m.cols() == v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        double error =
            std::abs(m(i / m.cols(), i % m.cols()) - v[i]) / std::max(1., std::abs(v[i]));
        worst = std::max(worst, error);
        CHECK(error < tol);
    }
}
double loss(const Ref& r, const Vec& g) {
    double s = 0;
    for (size_t i = 0; i < g.size(); ++i) s += r.y[i] * g[i];
    return s;
}
void run_case(int B, int T, int D, bool causal) {
    int N = B * T * D;
    Vec q(N), k(N), v(N), g(N);
    for (int i = 0; i < N; ++i) {
        q[i] = static_cast<float>(.9 * std::sin(i * 1.3 + .2));
        k[i] = static_cast<float>(1.1 * std::cos(i * .9 + .4));
        v[i] = static_cast<float>(std::sin(i * .7) - .2);
        g[i] = static_cast<float>(std::cos(i * .6 + .1));
    }
    Ref ref = dense(q, k, v, g, B, T, D, causal);
    // Every component against central differences of the independent double oracle.
    for (int which = 0; which < 3; ++which)
        for (int i = 0; i < N; ++i) {
            auto qp = q, kp = k, vp = v, qm = q, km = k, vm = v;
            Vec* plus[] = {&qp, &kp, &vp};
            Vec* minus[] = {&qm, &km, &vm};
            (*plus[which])[i] += 1e-5;
            (*minus[which])[i] -= 1e-5;
            double numeric = (loss(dense(qp, kp, vp, g, B, T, D, causal), g) -
                              loss(dense(qm, km, vm, g, B, T, D, causal), g)) /
                             2e-5;
            const Vec* grads[] = {&ref.dq, &ref.dk, &ref.dv};
            double error = std::abs(numeric - (*grads[which])[i]);
            worst_fd = std::max(worst_fd, error);
            CHECK(error < 2e-7);
            ++components;
        }
    auto a = make_var(matrix(q, D), true), b = make_var(matrix(k, D), true),
         c = make_var(matrix(v, D), true);
    auto y = ops::kimi_attention(a, b, c, causal, T);
    close(y->data, ref.y, 2e-6);
    backward(ops::mean(ops::mul(y, make_var(matrix(g, D)))));
    for (auto* grad : {&ref.dq, &ref.dk, &ref.dv})
        for (double& x : *grad) x /= N;
    close(a->grad, ref.dq, 3e-6);
    close(b->grad, ref.dk, 3e-6);
    close(c->grad, ref.dv, 3e-6);
    // Production float forward directional FD, all q/k/v moved together.
    for (int direction = 0; direction < 3; ++direction) {
        Vec qp = q, kp = k, vp = v, qm = q, km = k, vm = v;
        double dot = 0;
        Vec* plus[] = {&qp, &kp, &vp};
        Vec* minus[] = {&qm, &km, &vm};
        const Vec* grads[] = {&ref.dq, &ref.dk, &ref.dv};
        for (int w = 0; w < 3; ++w)
            for (int i = 0; i < N; ++i) {
                double d = std::sin((i + 1) * (w + 1) * (.37 + direction));
                (*plus[w])[i] += 1e-3 * d;
                (*minus[w])[i] -= 1e-3 * d;
                dot += (*grads[w])[i] * d;
            }
        auto eval = [&](const Vec& qa, const Vec& ka, const Vec& va) {
            NoGrad guard;
            auto o = ops::kimi_attention(make_var(matrix(qa, D)), make_var(matrix(ka, D)),
                                         make_var(matrix(va, D)), causal, T);
            return loss(Ref{vector(o->data), {}, {}, {}}, g) / N;
        };
        double numeric = (eval(qp, kp, vp) - eval(qm, km, vm)) / 2e-3;
        double error = std::abs(numeric - dot);
        worst_direction = std::max(worst_direction, error);
        CHECK(error < 3e-4);
        ++directions;
    }
    // Selective requires_grad and no cross-sequence leakage.
    for (int mask = 0; mask < 8; ++mask) {
        auto qa = make_var(matrix(q, D), mask & 1), ka = make_var(matrix(k, D), mask & 2),
             va = make_var(matrix(v, D), mask & 4);
        auto o = ops::kimi_attention(qa, ka, va, causal, T);
        if (mask) backward(ops::mean(ops::mul(o, make_var(matrix(g, D)))));
        std::array<Var, 3> inputs{qa, ka, va};
        const Vec* grads[] = {&ref.dq, &ref.dk, &ref.dv};
        for (int w = 0; w < 3; ++w) {
            if (mask & (1 << w))
                close(inputs[w]->grad, *grads[w], 3e-6);
            else
                CHECK(inputs[w]->grad.rows() == 0);
        }
    }
    if (B > 1) {
        Vec modified = v;
        for (int i = T * D; i < N; ++i) modified[i] += 10;
        NoGrad guard;
        auto o = ops::kimi_attention(make_var(matrix(q, D)), make_var(matrix(k, D)),
                                     make_var(matrix(modified, D)), causal, T);
        for (int i = 0; i < T * D; ++i) CHECK(o->data(i / D, i % D) == y->data(i / D, i % D));
    }
}
void metamorphic_noncausal() {
    Vec q{.2, -.7, .4, .1, .8, -.3}, k{-.2, .3, -.4, .9, .1, -.8}, v{2., -.3, 2., -.3, 2., -.3},
        g{.1, .2, -.3, .4, .5, -.6};
    KimiLinearAttention op(2);
    auto qm = matrix(q, 2), km = matrix(k, 2), vm = matrix(v, 2), gm = matrix(g, 2);
    auto out = op.forward(qm, km, vm, false);
    auto [dq, dk, dv] = op.backward(gm, qm, km, vm, out, false);
    for (int i = 0; i < 6; ++i) {
        CHECK(std::abs(out(i / 2, i % 2) - phi(q[i]) * v[i]) < 1e-6);
        CHECK(std::abs(dk(i / 2, i % 2)) < 1e-6);
    }
    v = {1., -.2, .7, 2., -1., .4};
    vm = matrix(v, 2);
    out = op.forward(qm, km, vm, false);
    Vec kp, vp;
    for (int row : {2, 0, 1})
        for (int j = 0; j < 2; ++j) {
            kp.push_back(k[row * 2 + j]);
            vp.push_back(v[row * 2 + j]);
        }
    close(op.forward(qm, matrix(kp, 2), matrix(vp, 2), false), vector(out), 1e-6);
}
void clamp_characterization(bool causal) {
    Vec q{.2, .5}, k{-20., -21.}, v{2., -3.}, g{.7, -.4};
    KimiLinearAttention op(1);
    auto qm = matrix(q, 1), km = matrix(k, 1), vm = matrix(v, 1), gm = matrix(g, 1);
    auto out = op.forward(qm, km, vm, causal);
    auto [dq, dk, dv] = op.backward(gm, qm, km, vm, out, causal);
    auto ref = dense(q, k, v, g, 1, 2, 1, causal, true);
    close(out, ref.y, 1e-7);
    close(dq, ref.dq, 1e-7);
    close(dk, ref.dk, 2e-6);
    close(dv, ref.dv, 1e-7);
    CHECK(out(0, 0) == 0 && out(1, 0) == 0);
    CHECK(std::abs(dk(0, 0)) > 0);
    // This characterises smooth-expression AD at rounded-zero phi, NOT float-forward FD
    // correctness.
    auto plus = km, minus = km;
    plus(0, 0) += .001f;
    minus(0, 0) -= .001f;
    CHECK(op.forward(qm, plus, vm, causal)(0, 0) == op.forward(qm, minus, vm, causal)(0, 0));
}
void refuses(std::function<void()> fn) {
    bool failed = false;
    try {
        fn();
    } catch (const std::runtime_error&) {
        failed = true;
    }
    CHECK(failed);
}
int main() {
    metamorphic_noncausal();
    for (bool causal : {false, true}) {
        run_case(1, 5, 3, causal);
        run_case(2, 3, 4, causal);
        run_case(3, 1, 2, causal);
        clamp_characterization(causal);
    }
    auto x = make_var(Matrix(3, 2), true);
    refuses([&] { ops::kimi_attention(x, x, x, false, 2); });
    auto empty = make_var(Matrix(), true);
    refuses([&] { ops::kimi_attention(empty, empty, empty, false, 0); });
    auto bad = make_var(Matrix(2, 2));
    refuses([&] { ops::kimi_attention(x, bad, x, false, 1); });
    auto narrow = make_var(Matrix(3, 1));
    refuses([&] { ops::kimi_attention(x, x, narrow, false, 1); });
    KimiLinearAttention direct(2);
    refuses([&] { direct.forward(Matrix(), Matrix(), Matrix(), false); });
    refuses([&] {
        direct.backward(Matrix(2, 2), Matrix(3, 2), Matrix(3, 2), Matrix(3, 2), Matrix(3, 2),
                        false);
    });
    std::printf(
        "PASS channelwise noncausal+causal dense64, %d component FD, %d production directional FD; "
        "worst normalized gradient/forward %.3g, FD %.3g, directional %.3g; selective "
        "grad/batches/clamp-cancellation/refusals\n",
        components, directions, worst, worst_fd, worst_direction);
}