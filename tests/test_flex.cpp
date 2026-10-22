// FlexLM receipts — the paper-faithful decoder family (norm/activation/
// position/d_ff/depth all constructor-real, so a fetched arXiv config
// trains as WRITTEN, not approximated by the nearest preset):
//   1. THE EQUIVALENCE PIN: FlexLM(layernorm, gelu, learned, L=2,
//      d_ff=4d) reproduces ParityLM(EXACT) logits BIT-FOR-BIT at the
//      same seed — flex is a strict generalization, not a rewrite
//   2. sinusoidal table matches the Vaswani formula at spot positions,
//      and contributes zero parameters (learned - sinusoidal == n_ctx*d)
//   3. depth is real: L=6 builds 6 distinct "layers.N." param groups
//   4. rmsnorm + swiglu + sinusoidal trains: loss falls (composition of
//      FD-checked ops, exercised end-to-end)
//   5. relu path: grads reach the first block's fc through the ReLU
//   6. batch pin: stacked [2*T] forward at seq_len=T equals the two
//      single-sequence forwards row-for-row (mask isolation + positions
//      restart, sinusoidal included)
//   7. rope positions: no table; row 0 (position 0) equals the module
//      without RoPE bit for bit, later rows rotate; batch pin; whole-model
//      gradient check through the rotation
//   8. expr norms: layernorm written as an expr equals layernorm to float
//      rounding; whole-model gradient checks of it and of Primer's x(x - mu)
//      norm (with rope and squared ReLU)
#include <algorithm>
#include <cmath>
#include <string>
#include <cstdio>
#include <cstring>
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

// Whole-model gradient check: L = sum(logits .* R), R fixed, autograd vs
// central finite differences (f32 forwards, h below) on up to `per`
// entries of every parameter whose name contains one of `names`. Returns
// the worst |a - fd| / (1 + max(|a|, |fd|)) and counts the entries.
double model_gradcheck(parity::FlexLM& m, const std::vector<int>& ids, const std::vector<std::string>& names,
                       size_t per, size_t& checked) {
    std::mt19937 g(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Var probe = m.forward(ids);
    Matrix R(probe->data.rows(), probe->data.cols());
    for (size_t i = 0; i < R.rows(); ++i)
        for (size_t j = 0; j < R.cols(); ++j) R(i, j) = nd(g);
    const Var r = make_var(R);
    auto loss = [&]() {
        return ops::scale(ops::mean(ops::mul(m.forward(ids), r)), static_cast<float>(R.rows() * R.cols()));
    };
    for (const auto& [name, p] : m.named_parameters()) p->grad = Matrix();
    backward(loss());
    double worst = 0.0;
    checked = 0;
    for (const auto& [name, p] : m.named_parameters()) {
        bool want = false;
        for (const auto& n : names) want = want || name.find(n) != std::string::npos;
        if (!want) continue;
        const size_t total = p->data.rows() * p->data.cols();
        // h = 1e-2, or a tenth of the tensor's rms when that is smaller: the
        // 0.02-scale token table feeds a norm that is far from linear over
        // a step half its size.
        double ss = 0.0;
        for (size_t e = 0; e < total; ++e) ss += double(p->data(e / p->data.cols(), e % p->data.cols())) *
                                                 p->data(e / p->data.cols(), e % p->data.cols());
        const float h = static_cast<float>(std::min(1e-2, 0.1 * std::sqrt(ss / double(total))));
        const size_t stride = std::max<size_t>(1, total / per);
        for (size_t e = 0; e < total; e += stride) {
            const size_t i = e / p->data.cols(), j = e % p->data.cols();
            const float keep = p->data(i, j);
            NoGrad ng;
            p->data(i, j) = keep + h;
            const double up = loss()->data(0, 0);
            p->data(i, j) = keep - h;
            const double dn = loss()->data(0, 0);
            p->data(i, j) = keep;
            const double fd = (up - dn) / (2.0 * h), a = p->grad.rows() ? p->grad(i, j) : 0.0;
            const double rel = std::abs(a - fd) / (1.0 + std::max(std::abs(a), std::abs(fd)));
            if (rel > 2e-3) std::printf("   gradcheck %s[%zu,%zu]: autograd %.6g fd %.6g\n", name.c_str(), i, j, a, fd);
            worst = std::max(worst, rel);
            ++checked;
        }
    }
    return worst;
}

parity::FlexConfig small(size_t V, size_t d, size_t H, size_t T) {
    parity::FlexConfig fc;
    fc.vocab = V;
    fc.d = d;
    fc.n_heads = H;
    fc.n_ctx = T;
    fc.d_ff = 4 * d;
    return fc;
}
}  // namespace

int main() {
    const size_t V = 31, d = 32, H = 4, T = 16;

    // 1. equivalence pin ------------------------------------------------------
    {
        parity::ParityLM ref(parity::AttnKind::EXACT, V, d, H, T, 7);
        parity::FlexConfig fc;
        fc.vocab = V;
        fc.d = d;
        fc.n_heads = H;
        fc.n_ctx = T;
        fc.n_layers = 2;
        fc.d_ff = 4 * d;
        parity::FlexLM flex(fc, 7);
        CHECK(flex.parameter_count() == ref.parameter_count());
        auto ids = ids_mod(T, V, 3);
        Var a = ref.forward(ids), b = flex.forward(ids);
        for (size_t i = 0; i < a->data.rows(); ++i)
            for (size_t j = 0; j < a->data.cols(); ++j)
                CHECK(a->data(i, j) == b->data(i, j));  // bitwise, not approx
        std::printf("1. equivalence pin: FlexLM(defaults) == ParityLM(EXACT) bitwise\n");
    }

    // 2. sinusoidal table -----------------------------------------------------
    {
        parity::FlexConfig fc;
        fc.vocab = V;
        fc.d = d;
        fc.n_heads = H;
        fc.n_ctx = T;
        fc.d_ff = 4 * d;
        parity::FlexLM learned(fc, 7);
        fc.pos = "sinusoidal";
        parity::FlexLM sinu(fc, 7);
        CHECK(learned.parameter_count() - sinu.parameter_count() == T * d);
        // Vaswani: PE(p, 2i) = sin(p / 10000^(2i/d)), PE(p, 2i+1) = cos(same)
        for (size_t p : {size_t(0), size_t(5), T - 1})
            for (size_t i : {size_t(0), size_t(7), d - 1}) {
                const double angle = p / std::pow(10000.0, double(2 * (i / 2)) / double(d));
                const float want =
                    static_cast<float>(i % 2 == 0 ? std::sin(angle) : std::cos(angle));
                CHECK(std::fabs(sinu.sin_table(p, i) - want) < 1e-6f);
            }
        std::printf("2. sinusoidal: formula matches, zero parameters added\n");
    }

    // 3. depth is real --------------------------------------------------------
    {
        parity::FlexConfig fc;
        fc.vocab = V;
        fc.d = d;
        fc.n_heads = H;
        fc.n_ctx = T;
        fc.n_layers = 6;
        fc.d_ff = 4 * d;
        parity::FlexLM deep(fc, 7);
        int block_params = 0;
        for (const auto& [name, p] : deep.named_parameters())
            if (name.rfind("layers.5.", 0) == 0) ++block_params;
        CHECK(block_params > 0);
        parity::FlexConfig f2 = fc;
        f2.n_layers = 2;
        parity::FlexLM two(f2, 7);
        CHECK(deep.parameter_count() > two.parameter_count());
        std::printf("3. depth: layers.5 exists, params grow with L\n");
    }

    // 4. rmsnorm + swiglu + sinusoidal trains ---------------------------------
    {
        parity::FlexConfig fc;
        fc.vocab = V;
        fc.d = d;
        fc.n_heads = H;
        fc.n_ctx = T;
        fc.d_ff = 3 * d;
        fc.norm = "rmsnorm";
        fc.act = "swiglu";
        fc.pos = "sinusoidal";
        parity::FlexLM m(fc, 11);
        nn::AdamW opt(m.parameters(), 3e-3f);
        auto ids = ids_mod(T + 1, V, 5);
        std::vector<int> x(ids.begin(), ids.end() - 1), y(ids.begin() + 1, ids.end());
        float first = 0, last = 0;
        for (int step = 0; step < 30; ++step) {
            Var loss = ops::cross_entropy(m.forward(x), y);
            if (step == 0) first = loss->data(0, 0);
            last = loss->data(0, 0);
            opt.zero_grad();
            backward(loss);
            opt.step();
        }
        CHECK(last < first * 0.8f);
        std::printf("4. rmsnorm+swiglu+sinusoidal trains: %.3f -> %.3f\n", first, last);
    }

    // 5. relu path grads ------------------------------------------------------
    {
        parity::FlexConfig fc;
        fc.vocab = V;
        fc.d = d;
        fc.n_heads = H;
        fc.n_ctx = T;
        fc.d_ff = 4 * d;
        fc.act = "relu";
        parity::FlexLM m(fc, 13);
        auto ids = ids_mod(T, V, 9);
        std::vector<int> y(ids.rbegin(), ids.rend());
        Var loss = ops::cross_entropy(m.forward(ids), y);
        backward(loss);
        double g = 0;
        for (const auto& [name, p] : m.named_parameters())
            if (name.find("layers.0.mlp.c_fc") != std::string::npos && p->grad.rows())
                for (size_t i = 0; i < p->grad.rows(); ++i)
                    for (size_t j = 0; j < p->grad.cols(); ++j) g += std::fabs(p->grad(i, j));
        CHECK(g > 0);
        std::printf("5. relu path: grad reaches layers.0.mlp.c_fc (|g|=%.3g)\n", g);
    }

    // 6. batch pin ------------------------------------------------------------
    {
        parity::FlexConfig fc;
        fc.vocab = V;
        fc.d = d;
        fc.n_heads = H;
        fc.n_ctx = T;
        fc.d_ff = 4 * d;
        fc.norm = "rmsnorm";
        fc.pos = "sinusoidal";
        parity::FlexLM m(fc, 17);
        auto s1 = ids_mod(T, V, 21), s2 = ids_mod(T, V, 22);
        std::vector<int> both = s1;
        both.insert(both.end(), s2.begin(), s2.end());
        Var a1 = m.forward(s1), a2 = m.forward(s2), ab = m.forward(both, T);
        float md = 0;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < V; ++j) {
                md = std::max(md, std::fabs(ab->data(i, j) - a1->data(i, j)));
                md = std::max(md, std::fabs(ab->data(T + i, j) - a2->data(i, j)));
            }
        CHECK(md < 2e-5f);
        std::printf("6. batch pin: stacked == singles (max diff %.2e)\n", md);
    }

    // 7. rope positions --------------------------------------------------------
    {
        parity::FlexConfig fc = small(V, d, H, T);
        fc.pos = "rope";
        fc.norm = "rmsnorm";
        fc.act = "expr:relu(x)^2";
        parity::FlexLM m(fc, 19);
        // No position table: learned has T*d more parameters.
        parity::FlexConfig fl = fc;
        fl.pos = "learned";
        parity::FlexLM learned(fl, 19);
        CHECK(learned.parameter_count() - m.parameter_count() == T * d);
        // Row 0 sits at position 0, where the rotation is the identity, and
        // attends to itself only: bit for bit the module without RoPE. Later
        // rows rotate, so they differ.
        Matrix X(T, d);
        std::mt19937 g(5);
        // Large inputs, so the 0.02-scale projections give q.k far from 0
        // and the rotation visibly moves the attention weights.
        std::normal_distribution<float> nd(0.0f, 30.0f);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j) X(i, j) = nd(g);
        const Var x = make_var(X);
        const Var with = m.blocks[0]->attend(x, 0), without = m.blocks[0]->attn->forward(x, 0);
        float later = 0, mag = 0;
        for (size_t j = 0; j < d; ++j) {
            CHECK(with->data(0, j) == without->data(0, j));
            later = std::max(later, std::fabs(with->data(T - 1, j) - without->data(T - 1, j)));
            mag = std::max(mag, std::fabs(without->data(T - 1, j)));
        }
        later /= std::max(mag, 1e-30f);  // relative to the row's largest entry
        CHECK(later > 1e-2f);
        // Batch pin: positions restart at 0 in each stacked sequence.
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
        CHECK(md < 2e-5f);
        // Gradients through the rotation: qkv projections, the token table.
        size_t n = 0;
        const double worst = model_gradcheck(m, ids_mod(8, V, 33), {"attn.c_attn", "wte", "norm1"}, 24, n);
        CHECK(worst < 2e-3);
        std::printf("7. rope: no table (-%zu params), row 0 == no-rope bitwise, last row moves by %.2e of its range, "
                    "batch pin %.2e, gradcheck %zu entries worst rel %.2e\n",
                    T * d, later, md, n, worst);
    }

    // 8. expr norms -------------------------------------------------------------
    {
        // Layer norm written as an expr equals layernorm to float rounding
        // (gain 1 and bias 0 at init in both; every other seed is shared).
        const char* ln_expr = "(x - rowmean(x)) / sqrt(rowmean((x - rowmean(x))^2) + 1e-5)";
        parity::FlexConfig fc = small(V, d, H, T);
        parity::FlexLM ln(fc, 23);
        fc.norm = std::string("expr:") + ln_expr;
        parity::FlexLM ex(fc, 23);
        CHECK(ex.parameter_count() == ln.parameter_count());
        auto ids = ids_mod(T, V, 41);
        Var a = ln.forward(ids), b = ex.forward(ids);
        float md = 0, range = 0;
        for (size_t i = 0; i < a->data.rows(); ++i)
            for (size_t j = 0; j < a->data.cols(); ++j) {
                md = std::max(md, std::fabs(a->data(i, j) - b->data(i, j)));
                range = std::max(range, std::fabs(a->data(i, j)));
            }
        CHECK(md <= 1e-5f * std::max(1.0f, range));
        size_t n1 = 0;
        const double w1 = model_gradcheck(ex, ids_mod(8, V, 42), {"norm", "wte", "mlp.c_fc"}, 24, n1);
        CHECK(w1 < 2e-3);
        // Primer's modified layer norm, x(x - mu) in place of (x - mu)^2
        // (arXiv 2109.08668), with rope and squared ReLU beside it.
        const char* primer = "(x - rowmean(x)) / sqrt(abs(rowmean(x * (x - rowmean(x)))))";
        fc.norm = std::string("expr:") + primer;
        fc.pos = "rope";
        fc.act = "expr:relu(x)^2";
        parity::FlexLM pr(fc, 29);
        size_t n2 = 0;
        const double w2 = model_gradcheck(pr, ids_mod(8, V, 43), {"norm", "wte", "attn.c_attn"}, 24, n2);
        CHECK(w2 < 2e-3);
        std::printf("8. expr norm: layernorm as an expr == layernorm (max %.2e, logit range %.3g); gradcheck "
                    "layernorm-expr %zu entries worst rel %.2e, Primer's x(x - mu) %zu entries %.2e\n",
                    md, range, n1, w1, n2, w2);
    }

    std::printf("FLEX-OK 8/8\n");
    return 0;
}
