// Registry #0001 gate (Highway Networks, arXiv 1505.00387; dossier in
// registry/0001_highway_networks/ENTRY.md). What composition can get
// wrong here, checked:
//   1. gate-bias init: every gate bias element equals gate_bias_init
//      (default -2, custom value honoured)
//   2. parameter arithmetic: highway = residual + L * 2 * (d*d + d)
//   3. wiring is live: highway / plain / residual logits all differ
//   4. the default path is untouched: explicit residual="residual"
//      is bitwise the default config (the equivalence pin's guard)
//   5. gradients: FD spot-check through the gate weights AND the gate
//      bias (central differences vs backward()), grads reach the gates
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../tools/parity_model.hpp"
#include "check.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

namespace {
std::vector<int> ids_mod(size_t n, size_t vocab, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<int> v(n);
    for (auto& x : v) x = static_cast<int>(g() % vocab);
    return v;
}

parity::FlexConfig base_cfg(size_t V, size_t d, size_t H, size_t T) {
    parity::FlexConfig fc;
    fc.vocab = V;
    fc.d = d;
    fc.n_heads = H;
    fc.n_ctx = T;
    fc.n_layers = 2;
    fc.d_ff = 4 * d;
    return fc;
}

float max_abs_logit_diff(const Var& a, const Var& b) {
    float m = 0;
    for (size_t i = 0; i < a->data.rows(); ++i)
        for (size_t j = 0; j < a->data.cols(); ++j)
            m = std::max(m, std::fabs(a->data(i, j) - b->data(i, j)));
    return m;
}
}  // namespace

int main() {
    const size_t V = 31, d = 32, H = 4, T = 16;
    auto ids = ids_mod(T, V, 3);
    std::vector<int> y(ids.rbegin(), ids.rend());

    // 1. gate-bias init --------------------------------------------------
    {
        auto fc = base_cfg(V, d, H, T);
        fc.residual = "highway";
        parity::FlexLM m(fc, 7);
        size_t gates_seen = 0;
        for (const auto& [name, p] : m.named_parameters())
            if (name.find("gate_") != std::string::npos &&
                name.find(".b") != std::string::npos) {
                ++gates_seen;
                for (size_t j = 0; j < p->data.cols(); ++j)
                    CHECK(p->data(0, j) == -2.0f);
            }
        CHECK(gates_seen == 4);  // 2 layers x 2 sublayer gates

        fc.gate_bias_init = -1.0f;
        parity::FlexLM m2(fc, 7);
        for (const auto& [name, p] : m2.named_parameters())
            if (name.find("gate_") != std::string::npos &&
                name.find(".b") != std::string::npos)
                CHECK(p->data(0, 0) == -1.0f);
        std::printf("1. gate biases init to gate_bias_init (default -2, custom honoured)\n");
    }

    // 2 + 3 + 4. params, wiring, default guard ---------------------------
    {
        auto fc = base_cfg(V, d, H, T);
        parity::FlexLM residual(fc, 7);
        auto fc_explicit = fc;
        fc_explicit.residual = "residual";
        parity::FlexLM residual2(fc_explicit, 7);
        auto fc_hw = fc;
        fc_hw.residual = "highway";
        parity::FlexLM highway(fc_hw, 7);
        auto fc_pl = fc;
        fc_pl.residual = "plain";
        parity::FlexLM plain(fc_pl, 7);

        const size_t expect_extra = fc.n_layers * 2 * (d * d + d);
        CHECK(highway.parameter_count() ==
              residual.parameter_count() + expect_extra);
        CHECK(plain.parameter_count() == residual.parameter_count());

        Var lr = residual.forward(ids), lr2 = residual2.forward(ids),
            lh = highway.forward(ids), lp = plain.forward(ids);
        CHECK(max_abs_logit_diff(lr, lr2) == 0.0f);  // default untouched
        CHECK(max_abs_logit_diff(lr, lh) > 1e-4f);   // highway is live
        CHECK(max_abs_logit_diff(lr, lp) > 1e-4f);   // plain is live
        CHECK(max_abs_logit_diff(lh, lp) > 1e-4f);
        std::printf("2. params: +%zu as computed; 3./4. wiring live, default bitwise\n",
                    expect_extra);
    }

    // 5. FD gradcheck through the gates -----------------------------------
    {
        auto fc = base_cfg(V, d, H, T);
        fc.residual = "highway";
        parity::FlexLM m(fc, 13);

        auto loss_now = [&]() {
            return ops::cross_entropy(m.forward(ids), y)->data(0, 0);
        };
        Var loss = ops::cross_entropy(m.forward(ids), y);
        backward(loss);

        double gate_grad_mass = 0;
        int fd_checked = 0;
        const float eps = 5e-3f;
        for (const auto& [name, p] : m.named_parameters()) {
            if (name.find("layers.0.gate_attn") == std::string::npos) continue;
            for (size_t i = 0; i < p->grad.rows(); ++i)
                for (size_t j = 0; j < p->grad.cols(); ++j)
                    gate_grad_mass += std::fabs(p->grad(i, j));
            // spot FD on two entries per tensor (W and b both hit)
            for (int k = 0; k < 2; ++k) {
                const size_t i = (k * 7) % p->data.rows();
                const size_t j = (k * 13 + 5) % p->data.cols();
                const float w0 = p->data(i, j);
                p->data(i, j) = w0 + eps;
                const float lp_ = loss_now();
                p->data(i, j) = w0 - eps;
                const float lm_ = loss_now();
                p->data(i, j) = w0;
                const double fd = (double(lp_) - double(lm_)) / (2.0 * eps);
                const double an = p->grad(i, j);
                CHECK(std::fabs(fd - an) <= 5e-3 + 0.05 * std::fabs(fd));
                ++fd_checked;
            }
        }
        CHECK(gate_grad_mass > 0);
        CHECK(fd_checked == 4);  // 2 entries x {W, b}
        std::printf("5. FD gradcheck through gate W and b (4 spots), |g|=%.3g\n",
                    gate_grad_mass);
    }

    std::printf("all highway checks passed\n");
    return 0;
}
