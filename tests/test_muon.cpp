// Muon receipts, mirroring python/muon_reference.py's self-test suite (the
// correctness anchor for this port — TECH_TRANSFER item 3):
//   1. golden pin: newton_schulz5 against float64 values from a
//      dependency-free rerun of the reference algorithm
//   2. structural pin: an n_heads=1 optimizer step equals a hand-rolled
//      base-Muon step (momentum/nesterov/scale plumbing)
//   3. K3's stated motivation, measured: a 100x-loud head dominates the
//      full-matrix update; per-head equalizes update norms
//   4. training receipt: a matrices-only net actually learns on the tape
//   5. guards: vectors and non-divisible head counts are refused
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "microtorch/nn.hpp"
#include "microtorch/ops.hpp"

#include "check.hpp"

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

double fro(const Matrix& m) {
    double s = 0.0;
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j) s += static_cast<double>(m(i, j)) * m(i, j);
    return std::sqrt(s);
}

}  // namespace

int main() {
    // 1. golden pin ---------------------------------------------------------
    {
        Matrix G(8, 4);
        for (size_t i = 0; i < 8; ++i)
            for (size_t j = 0; j < 4; ++j)
                G(i, j) = static_cast<float>(std::sin(3.0 * i + 7.0 * j));
        Matrix O = nn::newton_schulz5(G, 5);
        struct Probe {
            size_t i, j;
            double want;
        };
        const Probe probes[] = {{0, 0, 0.2956676191},  {0, 3, 0.1073029152}, {3, 1, -0.0961735174},
                                {5, 2, -0.0580540669}, {7, 0, 0.4786074643}, {7, 3, -0.2501416451}};
        double worst = 0.0;
        for (const auto& p : probes) worst = std::max(worst, std::fabs(O(p.i, p.j) - p.want));
        const double dfro = std::fabs(fro(O) - 1.3580900639);
        std::printf("  [muon] golden probe max diff          %.3e\n", worst);
        std::printf("  [muon] golden frobenius diff          %.3e\n", dfro);
        CHECK(worst < 5e-3);
        CHECK(dfro < 5e-3);
    }

    // 2. structural pin: n_heads=1 == hand-rolled base Muon -----------------
    {
        Var W = make_var(randn(8, 24, 1), true);
        Matrix W0 = W->data;
        Matrix g = randn(8, 24, 2);
        W->grad = g;
        nn::Muon opt({W}, /*lr=*/0.1f);
        opt.step();

        // Hand-rolled: buf = 0.95*0 + g; upd = g + 0.95*buf; O = ns5(upd);
        // W -= 0.1 * sqrt(max(1, cols/rows)) * O.
        Matrix upd(8, 24);
        for (size_t i = 0; i < 8; ++i)
            for (size_t j = 0; j < 24; ++j) upd(i, j) = g(i, j) + 0.95f * g(i, j);
        Matrix O = nn::newton_schulz5(upd, 5);
        const float scale = std::sqrt(24.0f / 8.0f);
        double worst = 0.0;
        for (size_t i = 0; i < 8; ++i)
            for (size_t j = 0; j < 24; ++j)
                worst = std::max(worst, std::fabs(static_cast<double>(W->data(i, j)) -
                                                  (W0(i, j) - 0.1f * scale * O(i, j))));
        std::printf("  [muon] pin vs hand-rolled step        %.3e\n", worst);
        CHECK(worst <= 1e-7);
    }

    // 3. per-head equalization under a 100x-loud head -----------------------
    {
        const size_t H = 4, dk = 8, d = 16;
        Matrix g = randn(d, H * dk, 3);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < dk; ++j) g(i, j) *= 100.0f;  // loud head 0

        auto head_spread = [&](size_t n_heads) {
            Var W = make_var(Matrix(d, H * dk), true);
            W->grad = g;
            nn::Muon opt({W}, /*lr=*/1.0f, /*momentum=*/0.0f, /*nesterov=*/false,
                         /*ns_steps=*/5, n_heads);
            opt.step();
            double mn = 1e30, mx = 0.0;
            for (size_t h = 0; h < H; ++h) {
                double s = 0.0;
                for (size_t i = 0; i < d; ++i)
                    for (size_t j = 0; j < dk; ++j)
                        s += static_cast<double>(W->data(i, h * dk + j)) * W->data(i, h * dk + j);
                s = std::sqrt(s);
                mn = std::min(mn, s);
                mx = std::max(mx, s);
            }
            return mx / mn;
        };
        const double full = head_spread(1), per = head_spread(H);
        std::printf("  [muon] head update spread full=%.2fx per-head=%.2fx\n", full, per);
        // The K3 claim is comparative: per-head equalizes what full-matrix
        // leaves imbalanced (exact magnitudes depend on the random draw).
        CHECK(per < 1.1);
        CHECK(full > 1.3 * per);
    }

    // 4. training receipt: matrices-only net learns on the tape -------------
    {
        Var X = make_var(randn(64, 8, 10));
        Matrix ym(64, 2);
        for (size_t i = 0; i < 64; ++i) {
            ym(i, 0) = X->data(i, 0) * X->data(i, 1);
            ym(i, 1) = 0.5f * X->data(i, 2);
        }
        Var y = make_var(std::move(ym));
        Var W1 = make_var(randn(8, 16, 11, 0.5f), true);
        Var W2 = make_var(randn(16, 2, 12, 0.5f), true);
        nn::Muon opt({W1, W2}, /*lr=*/0.02f);
        float first = 0, last = 0;
        for (int s = 0; s < 200; ++s) {
            opt.zero_grad();
            Var pred = ops::matmul(ops::gelu(ops::matmul(X, W1)), W2);
            Var err = ops::sub(pred, y);
            Var loss = ops::mean(ops::mul(err, err));
            if (s == 0) first = loss->data(0, 0);
            last = loss->data(0, 0);
            backward(loss);
            opt.step();
        }
        std::printf("  [muon] training loss %.4f -> %.4f (200 steps)\n", first, last);
        CHECK(last < 0.5f * first);
    }

    // 5. guards -------------------------------------------------------------
    {
        bool threw = false;
        try {
            Var v = make_var(Matrix(1, 16), true);
            nn::Muon opt({v});
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        try {
            Var w = make_var(Matrix(4, 10), true);
            nn::Muon opt({w}, 0.02f, 0.95f, true, 5, /*n_heads=*/3);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
        std::printf("  [muon] guards: vectors + bad head counts refused\n");
    }

    std::printf("[PASS] all Muon tests\n");
    return 0;
}
