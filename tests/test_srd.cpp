// V1 Surprise-Routed Density tests (docs/SPARSE_ATTENTION.md).
//
//   1. FD gradchecks for the router ops: mul_col, rms_row, sigmoid,
//      add_scalar (central differences, the phase-1a discipline)
//   2. Module-level FD gradcheck of the FULL SRD forward -- gradients must
//      be correct through BOTH paths and the gate
//   3. Routing behavior: surprising rows gate higher than routine rows
//   4. Blend algebra: g == 1 reproduces the exact path exactly
//   5. Trainability: AdamW steps reduce a toy loss; predictor gets grads;
//      aux mean-gate loss moves the gate
#include <cmath>
#include <cstdio>
#include <random>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/srd.hpp"

using namespace microtorch;

namespace {

Matrix randm(size_t r, size_t c, unsigned seed, float scale = 0.5f) {
    Matrix m(r, c);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(rng);
    return m;
}

// Central-difference check of d mean(f(x)) / d x against the tape.
template <typename F>
float fd_gradcheck(F f, Matrix x0, size_t probes, unsigned seed) {
    Var x = make_var(x0, true);
    Var loss = ops::mean(f(x));
    backward(loss);

    std::mt19937 rng(seed);
    const float eps = 1e-3f;
    float worst = 0;
    for (size_t p = 0; p < probes; ++p) {
        const size_t i = rng() % x0.rows(), j = rng() % x0.cols();
        float fp, fm;
        {
            NoGrad ng;
            Matrix xp = x0;
            xp(i, j) += eps;
            fp = ops::mean(f(make_var(xp)))->data(0, 0);
            Matrix xm = x0;
            xm(i, j) -= eps;
            fm = ops::mean(f(make_var(xm)))->data(0, 0);
        }
        const float fd = (fp - fm) / (2 * eps);
        const float an = x->grad(i, j);
        const float err = std::fabs(fd - an) / std::max({std::fabs(fd), std::fabs(an), 1e-4f});
        worst = std::max(worst, err);
    }
    return worst;
}

void test_op_gradchecks() {
    printf("=== router-op gradchecks ===\n");
    {
        Matrix c0 = randm(6, 1, 21, 0.8f);
        float e = fd_gradcheck([&](const Var& x) { return ops::mul_col(x, make_var(c0, false)); },
                               randm(6, 5, 20), 10, 1);
        printf("  mul_col (x path)  rel err %.2e\n", e);
        CHECK(e < 5e-3f);
    }
    {
        // c path: perturb the column, x fixed.
        Matrix x0 = randm(6, 5, 22);
        float e = fd_gradcheck([&](const Var& c) { return ops::mul_col(make_var(x0, false), c); },
                               randm(6, 1, 23, 0.8f), 6, 2);
        printf("  mul_col (c path)  rel err %.2e\n", e);
        CHECK(e < 5e-3f);
    }
    {
        float e =
            fd_gradcheck([](const Var& x) { return ops::rms_row(x); }, randm(5, 8, 24), 10, 3);
        printf("  rms_row           rel err %.2e\n", e);
        CHECK(e < 5e-3f);
    }
    {
        float e = fd_gradcheck([](const Var& x) { return ops::sigmoid(x); }, randm(5, 8, 25, 2.0f),
                               10, 4);
        printf("  sigmoid           rel err %.2e\n", e);
        CHECK(e < 5e-3f);
    }
    {
        float e = fd_gradcheck([](const Var& x) { return ops::add_scalar(x, 0.37f); },
                               randm(5, 8, 26), 10, 5);
        printf("  add_scalar        rel err %.2e\n", e);
        CHECK(e < 5e-3f);
    }
}

void test_module_gradcheck() {
    printf("=== full-module FD gradcheck (both paths + gate) ===\n");
    nn::SurpriseRoutedAttention srd(16, 2, 42);
    srd.train();
    float e = fd_gradcheck([&](const Var& x) { return srd.forward(x); }, randm(6, 16, 30), 12, 6);
    printf("  d mean(SRD(x))/dx rel err %.2e\n", e);
    // Composite fp32 module: looser tolerance than single ops.
    CHECK(e < 2e-2f);
}

void test_routing_behavior() {
    printf("=== routing: surprise raises the gate ===\n");
    nn::SurpriseRoutedAttention srd(32, 4, 7);
    srd.train();

    // Rows 0-5 routine (tiny values, predictor residual ~ 0); rows 6-7
    // surprising (large values).
    Matrix x(8, 32);
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 32; ++j) x(i, j) = (i < 6) ? 1e-3f : 0.9f;

    srd.forward(make_var(x));
    Var g = srd.gate();
    CHECK(g && g->data.rows() == 8 && g->data.cols() == 1);
    float routine_avg = 0, surprise_avg = 0;
    for (size_t i = 0; i < 6; ++i) routine_avg += g->data(i, 0) / 6.0f;
    for (size_t i = 6; i < 8; ++i) surprise_avg += g->data(i, 0) / 2.0f;
    printf("  routine gate %.3f | surprising gate %.3f\n", routine_avg, surprise_avg);
    CHECK(routine_avg < 0.5f);
    CHECK(surprise_avg > routine_avg + 0.15f);
}

void test_gate_one_is_exact() {
    printf("=== blend algebra: g == 1 -> exact path ===\n");
    Matrix a = randm(5, 8, 40), ones(5, 1, 1.0f), zeros(5, 1);
    Var blended = ops::add(ops::mul_col(make_var(a), make_var(ones)),
                           ops::mul_col(make_var(randm(5, 8, 41)), make_var(zeros)));
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 8; ++j) CHECK(blended->data(i, j) == a(i, j));
    printf("  mul_col blend reproduces the selected path bit-exactly\n");
}

void test_trainability() {
    printf("=== trainability + aux gate loss ===\n");
    nn::SurpriseRoutedAttention srd(16, 2, 11);
    srd.train();
    auto params = srd.parameters();
    CHECK(params.size() > 0);
    nn::AdamW opt(params, 3e-3f);

    Matrix x0 = randm(6, 16, 50);
    float first_loss = 0, last_loss = 0, first_gate = 0, last_gate = 0;
    for (int step = 0; step < 12; ++step) {
        Var out = srd.forward(make_var(x0));
        // Toy objective + density price: drive outputs to zero AND pay
        // for open gates (lambda = 0.1).
        Var task = ops::mean(ops::mul(out, out));
        Var loss = ops::add(task, ops::scale(ops::mean(srd.gate()), 0.1f));
        opt.zero_grad();
        backward(loss);

        // The predictor must receive gradient through the gate.
        if (step == 0) {
            bool pred_has_grad = false;
            for (const auto& [name, p] : srd.named_parameters()) {
                if (name.find("predictor") != std::string::npos && p->grad.rows() != 0) {
                    float s = 0;
                    for (size_t i = 0; i < p->grad.rows(); ++i)
                        for (size_t j = 0; j < p->grad.cols(); ++j) s += std::fabs(p->grad(i, j));
                    pred_has_grad = pred_has_grad || s > 0;
                }
            }
            CHECK(pred_has_grad);
            printf("  predictor receives gradient through the gate\n");
        }
        opt.step();

        const float l = loss->data(0, 0);
        const float gm = ops::mean(srd.gate())->data(0, 0);
        if (step == 0) {
            first_loss = l;
            first_gate = gm;
        }
        last_loss = l;
        last_gate = gm;
    }
    printf("  loss %.5f -> %.5f | mean gate %.3f -> %.3f\n", first_loss, last_loss, first_gate,
           last_gate);
    CHECK(last_loss < first_loss);
    CHECK(last_gate < first_gate);  // the density price closes gates
    CHECK(last_gate > 0.0f && last_gate < 1.0f);
}

}  // namespace

int main() {
    microtorch::device::set_from_env();
    test_op_gradchecks();
    test_module_gradcheck();
    test_routing_behavior();
    test_gate_one_is_exact();
    test_trainability();
    printf("\n[PASS] all SRD tests\n");
    return 0;
}
