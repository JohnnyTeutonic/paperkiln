// Deep SWA gate (ROADMAP 1a): sliding-window attention at any depth via
// the flex family. What composition can get wrong, checked:
//   1. THE SWA PIN: FlexLM(attention=swa, defaults, L=2, w=64, s=1)
//      reproduces ParityLM(SWA) logits BITWISE at the same seed —
//      exactly the guarantee the exact-attention pin gives, extended
//      to the swa path (same module name, same seed layout).
//   2. depth is real: L=4 builds distinct layers.N groups; grads reach
//      the DEEPEST block's attention through three swa layers above it.
//   3. FD spot-check through layer-0 attention weights at depth 4.
//   4. batch pin at depth: stacked [2T] forward at seq_len=T equals the
//      two single-sequence forwards row-for-row (window masking and
//      positions isolate correctly through depth).
//   5. knob independence: swa + highway compose (registry #0001 lanes
//      will want this eventually) — builds and runs.
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

parity::FlexConfig swa_cfg(size_t V, size_t d, size_t H, size_t T,
                           size_t layers) {
    parity::FlexConfig fc;
    fc.vocab = V;
    fc.d = d;
    fc.n_heads = H;
    fc.n_ctx = T;
    fc.n_layers = layers;
    fc.d_ff = 4 * d;
    fc.attention = "swa";
    fc.window = 8;   // small window so masking does real work at T=16
    fc.sinks = 1;
    return fc;
}
}  // namespace

int main() {
    const size_t V = 31, d = 32, H = 4, T = 16;
    auto ids = ids_mod(T, V, 3);
    std::vector<int> y(ids.rbegin(), ids.rend());

    // 1. the swa pin ------------------------------------------------------
    {
        parity::ParityLM ref(parity::AttnKind::SWA, V, d, H, T, 7,
                             /*swa_window=*/8, /*swa_sinks=*/1);
        auto fc = swa_cfg(V, d, H, T, 2);
        parity::FlexLM flex(fc, 7);
        CHECK(flex.parameter_count() == ref.parameter_count());
        Var a = ref.forward(ids), b = flex.forward(ids);
        for (size_t i = 0; i < a->data.rows(); ++i)
            for (size_t j = 0; j < a->data.cols(); ++j)
                CHECK(a->data(i, j) == b->data(i, j));  // bitwise
        std::printf("1. SWA pin: FlexLM(swa, defaults) == ParityLM(SWA) bitwise\n");
    }

    // 2. depth is real + grads reach the deepest block --------------------
    {
        auto fc = swa_cfg(V, d, H, T, 4);
        parity::FlexLM m(fc, 13);
        size_t deep_params = 0;
        for (const auto& [name, p] : m.named_parameters())
            if (name.find("layers.3.") != std::string::npos)
                deep_params += p->data.rows() * p->data.cols();
        CHECK(deep_params > 0);

        Var loss = ops::cross_entropy(m.forward(ids), y);
        backward(loss);
        double g = 0;
        for (const auto& [name, p] : m.named_parameters())
            if (name.find("layers.3.attn") != std::string::npos && p->grad.rows())
                for (size_t i = 0; i < p->grad.rows(); ++i)
                    for (size_t j = 0; j < p->grad.cols(); ++j)
                        g += std::fabs(p->grad(i, j));
        CHECK(g > 0);
        std::printf("2. depth-4 swa: layers.3 real (%zu params), |grad|=%.3g\n",
                    deep_params, g);
    }

    // 3. FD spot-check at depth 4 ------------------------------------------
    {
        auto fc = swa_cfg(V, d, H, T, 4);
        parity::FlexLM m(fc, 17);
        auto loss_now = [&]() {
            return ops::cross_entropy(m.forward(ids), y)->data(0, 0);
        };
        Var loss = ops::cross_entropy(m.forward(ids), y);
        backward(loss);
        const float eps = 5e-3f;
        int checked = 0;
        for (const auto& [name, p] : m.named_parameters()) {
            // the weight matrix under the first block's qkv projection
            // (suffix-agnostic: pick the 2-D tensor, skip the bias row)
            if (name.find("layers.0.attn.c_attn") == std::string::npos ||
                p->data.rows() <= 1)
                continue;
            for (int k = 0; k < 3; ++k) {
                const size_t i = (k * 11 + 2) % p->data.rows();
                const size_t j = (k * 17 + 3) % p->data.cols();
                const float w0 = p->data(i, j);
                p->data(i, j) = w0 + eps;
                const float lp = loss_now();
                p->data(i, j) = w0 - eps;
                const float lm = loss_now();
                p->data(i, j) = w0;
                const double fd = (double(lp) - double(lm)) / (2.0 * eps);
                CHECK(std::fabs(fd - p->grad(i, j)) <= 5e-3 + 0.05 * std::fabs(fd));
                ++checked;
            }
        }
        CHECK(checked == 3);
        std::printf("3. FD gradcheck through layers.0 swa attention (3 spots)\n");
    }

    // 4. batch pin at depth -------------------------------------------------
    {
        auto fc = swa_cfg(V, d, H, T, 4);
        parity::FlexLM m(fc, 21);
        auto s1 = ids_mod(T, V, 31), s2 = ids_mod(T, V, 32);
        std::vector<int> both = s1;
        both.insert(both.end(), s2.begin(), s2.end());
        Var a1 = m.forward(s1), a2 = m.forward(s2), ab = m.forward(both, T);
        float md = 0;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < V; ++j) {
                md = std::max(md, std::fabs(ab->data(i, j) - a1->data(i, j)));
                md = std::max(md, std::fabs(ab->data(T + i, j) - a2->data(i, j)));
            }
        CHECK(md == 0.0f);
        std::printf("4. depth-4 batch pin: stacked == singles (max diff %.2e)\n", md);
    }

    // 5. swa + highway compose ----------------------------------------------
    {
        auto fc = swa_cfg(V, d, H, T, 3);
        fc.residual = "highway";
        parity::FlexLM m(fc, 23);
        Var out = m.forward(ids);
        CHECK(out->data.rows() == T && out->data.cols() == V);
        std::printf("5. swa + highway at depth 3: builds and runs\n");
    }

    std::printf("all deep-swa checks passed\n");
    return 0;
}
