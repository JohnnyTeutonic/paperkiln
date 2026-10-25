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
//   9. expr inputs: a learned scalar (Swish-beta at init 1 equals x sigmoid(x)
//      bit for bit; DyT's tanh(alpha x) norm, scalar and per channel), a gated
//      unit (relu(a) * x over a gate projection); gradient checks; refusals
//  10. norm placements: every attention x FFN placement (pre, post, sandwich,
//      branch_post, none) equals the block's parts composed by hand; Primer's
//      pre / branch_post with its expr norm, gradient-checked
//  11. T5's relative position bias: the bucket table, ops::relpos_attention
//      against causal attention and finite differences; the flex model's
//      gradient check; the GGUF export refused (no keramos op yet)
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

    // 9. gated and parameterised exprs -------------------------------------------
    {
        using In = parity::FlexConfig::ExprInput;
        // A learned scalar at its init 1: Swish-beta x * sigmoid(a * x) equals x * sigmoid(x) bit for
        // bit (1 * x is x), one parameter per block more; every other seed is shared.
        parity::FlexConfig fc = small(V, d, H, T);
        fc.act = "expr:x * sigmoid(x)";
        parity::FlexLM plain_swish(fc, 51);
        fc.act = "expr:x * sigmoid(a * x)";
        fc.act_inputs = {In{"up"}, In{"scalar", 1.0f}};
        parity::FlexLM beta(fc, 51);
        CHECK(beta.parameter_count() == plain_swish.parameter_count() + fc.n_layers);
        auto ids = ids_mod(T, V, 52);
        Var a = plain_swish.forward(ids), b = beta.forward(ids);
        for (size_t i = 0; i < a->data.rows(); ++i)
            for (size_t j = 0; j < a->data.cols(); ++j) CHECK(a->data(i, j) == b->data(i, j));
        size_t n1 = 0;
        const double w1 = model_gradcheck(beta, ids_mod(8, V, 53), {"mlp.p_a", "mlp.c_fc"}, 24, n1);
        CHECK(w1 < 2e-3);
        // A gated unit: the GLU, sigmoid(a) * x over the up-projection x and a gate projection a (smooth,
        // so finite differences hold everywhere; relu(a) * x kinks where a gate crosses 0).
        fc.act = "expr:sigmoid(a) * x";
        fc.act_inputs = {In{"up"}, In{"gate"}};
        parity::FlexLM reglu(fc, 54);
        CHECK(reglu.blocks[0]->fc_gate != nullptr);
        CHECK(reglu.parameter_count() == plain_swish.parameter_count() + fc.n_layers * (d * fc.d_ff + fc.d_ff));
        size_t n2 = 0;
        const double w2 = model_gradcheck(reglu, ids_mod(8, V, 55), {"mlp.c_gate", "mlp.c_fc", "wte"}, 24, n2);
        CHECK(w2 < 2e-3);
        // DyT (arXiv 2503.10622): the norm tanh(alpha x) * g + b, alpha a learned scalar per norm (init
        // 0.5); and a per-channel learned vector inside the row expression.
        parity::FlexConfig fd = small(V, d, H, T);
        fd.norm = "expr:tanh(a * x)";
        fd.norm_inputs = {In{"row"}, In{"scalar", 0.5f}};
        parity::FlexLM dyt(fd, 56);
        CHECK(dyt.blocks[0]->ns[0].params.size() == 1 && dyt.blocks[0]->ns[0].params[0]->data.cols() == 1);
        CHECK(dyt.blocks[0]->ns[0].params[0]->data(0, 0) == 0.5f && dyt.fn.params.size() == 1);
        size_t n3 = 0;
        const double w3 =
            model_gradcheck(dyt, ids_mod(8, V, 57), {"norm1.p_a", "norm2.p_a", "norm.p_a", "wte"}, 24, n3);
        CHECK(w3 < 2e-3);
        fd.norm = "expr:tanh(x * a)";
        fd.norm_inputs = {In{"row"}, In{"channel", 0.5f}};
        parity::FlexLM dyt_c(fd, 56);
        CHECK(dyt_c.blocks[0]->ns[1].params[0]->data.cols() == d);
        size_t n4 = 0;
        const double w4 = model_gradcheck(dyt_c, ids_mod(8, V, 58), {"norm1.p_a", "norm.p_a"}, 24, n4);
        CHECK(w4 < 2e-3);
        // DyT on LLaMA: alpha starts at 0.8 in the attention sublayer's norm and 0.2 in the FFN's and the
        // head's; a learned scalar times the token embedding (an expr on the embed output).
        In alpha{"scalar", 0.8f};
        alpha.init_ffn = alpha.init_final = 0.2f;
        fd.norm = "expr:tanh(a * x)";
        fd.norm_inputs = {In{"row"}, alpha};
        fd.embed_expr = "expr:x * a";
        fd.embed_inputs = {In{"row"}, In{"scalar", 4.0f}};
        fd.act = "swiglu";
        fd.pos = "rope";
        parity::FlexLM dl(fd, 59);
        CHECK(dl.blocks[0]->ns[0].params[0]->data(0, 0) == 0.8f && dl.blocks[0]->ns[1].params[0]->data(0, 0) == 0.2f);
        CHECK(dl.fn.params[0]->data(0, 0) == 0.2f && dl.embed_params.size() == 1 && dl.embed_params[0]->data(0, 0) == 4.0f);
        size_t n5 = 0;
        const double w5 = model_gradcheck(dl, ids_mod(8, V, 60), {"embed.p_a", "norm1.p_a", "wte"}, 24, n5);
        CHECK(w5 < 2e-3);
        // Refused configurations, by name.
        auto refuses = [&](parity::FlexConfig c, const char* what) {
            bool threw = false;
            try {
                parity::FlexLM m(c, 1);
            } catch (const std::exception& e) {
                threw = std::string(e.what()).find(what) != std::string::npos;
                if (!threw) std::printf("   unexpected refusal: %s\n", e.what());
            }
            CHECK(threw);
        };
        parity::FlexConfig bad = small(V, d, H, T);
        bad.act = "expr:x * a";
        bad.act_inputs = {In{"up"}, In{"up"}};
        refuses(bad, "exactly once");
        bad.act_inputs = {In{"gate"}, In{"scalar"}};
        refuses(bad, "exactly once");
        bad.act = "relu";
        bad.act_inputs = {In{"up"}};
        refuses(bad, "need an expr activation");
        bad = small(V, d, H, T);
        bad.norm = "expr:x * a";
        bad.norm_inputs = {In{"row"}, In{"gate"}};
        refuses(bad, "unknown input");
        std::printf("9. expr inputs: Swish-beta at beta 1 == x sigmoid(x) bitwise, gradcheck %zu entries %.2e; "
                    "GLU as sigmoid(a) * x %zu entries %.2e; DyT tanh(alpha x) %zu entries %.2e, per channel "
                    "%zu entries %.2e; DyT-on-LLaMA (per-place alpha, embedding scalar) %zu entries %.2e; four bad "
                    "configurations refused\n",
                    n1, w1, n2, w2, n3, w3, n4, w4, n5, w5);
    }

    // 10. norm placements --------------------------------------------------------
    {
        // Each placement against the block's own parts composed by hand (bitwise), and a gradient check.
        const char* places[] = {"pre", "post", "sandwich", "branch_post", "none"};
        double worst = 0;
        size_t checked = 0;
        for (const char* pa : places)
            for (const char* pf : places) {
                parity::FlexConfig fc = small(V, d, H, T);
                fc.n_layers = 1;
                fc.place_attn = pa;
                fc.place_ffn = pf;
                parity::FlexLM m(fc, 61);
                const auto& blk = *m.blocks[0];
                auto ids = ids_mod(T, V, 62);
                std::vector<int> pos(ids.size());
                for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i);
                Var x = ops::add(m.wte->forward(ids), m.wpe->forward(pos));
                const std::string sa = pa, sf = pf;
                Var a = blk.attend(sa == "pre" || sa == "sandwich" ? blk.ns[0].ln->forward(x) : x, 0);
                if (sa == "sandwich") a = blk.ns[2].ln->forward(a);
                if (sa == "branch_post") a = blk.ns[0].ln->forward(a);
                Var h = ops::add(x, a);
                if (sa == "post") h = blk.ns[0].ln->forward(h);
                Var f = blk.mlp->forward(sf == "pre" || sf == "sandwich" ? blk.ns[1].ln->forward(h) : h);
                if (sf == "sandwich") f = blk.ns[3].ln->forward(f);
                if (sf == "branch_post") f = blk.ns[1].ln->forward(f);
                Var y = ops::add(h, f);
                if (sf == "post") y = blk.ns[1].ln->forward(y);
                const bool fin = !(sa == "post" && sf == "post");
                CHECK(m.fn.on() == fin && fc.has_final_norm() == fin);
                Var want = m.head->forward(fin ? m.ln_f_ln->forward(y) : y);
                Var got = m.forward(ids);
                for (size_t i = 0; i < got->data.rows(); ++i)
                    for (size_t j = 0; j < got->data.cols(); ++j) CHECK(got->data(i, j) == want->data(i, j));
                const int norms = (sa != "none") + (sf != "none") + (sa == "sandwich") + (sf == "sandwich");
                CHECK(int(blk.ns[0].on()) + int(blk.ns[1].on()) + int(blk.ns[2].on()) + int(blk.ns[3].on()) == norms);
                if (sa == sf || (sa == "pre" && sf == "branch_post")) {
                    size_t n = 0;
                    worst = std::max(worst, model_gradcheck(m, ids_mod(8, V, 63), {"ln_", "wte"}, 16, n));
                    checked += n;
                }
            }
        CHECK(worst < 2e-3);
        // rmsnorm and the expr norm take the same placements (Primer: pre attention, the FFN's norm after it).
        parity::FlexConfig fp = small(V, d, H, T);
        fp.norm = "expr:(x - rowmean(x)) / sqrt(abs(rowmean(x * (x - rowmean(x)))))";
        fp.act = "expr:relu(x)^2";
        fp.pos = "rope";
        fp.place_ffn = "branch_post";
        parity::FlexLM primer(fp, 64);
        size_t np = 0;
        const double wp = model_gradcheck(primer, ids_mod(8, V, 65), {"norm", "wte"}, 24, np);
        CHECK(wp < 2e-3);
        parity::FlexConfig bad = small(V, d, H, T);
        bad.place_ffn = "middle";
        bool threw = false;
        try {
            parity::FlexLM m(bad, 1);
        } catch (const std::exception& e) {
            threw = std::string(e.what()).find("norm_placement") != std::string::npos;
        }
        CHECK(threw);
        std::printf("10. placements: 25 attention x FFN placements == the block's parts composed by hand "
                    "(bitwise), final norm absent only for post/post; gradcheck %zu entries worst %.2e; "
                    "Primer's (expr norm, pre / branch_post, rope) %zu entries %.2e; an unknown placement refused\n",
                    checked, worst, np, wp);
    }

    // 11. T5's relative position bias --------------------------------------------
    {
        const std::vector<int> b = ops::t5_position_buckets(32, 128);
        CHECK(b.size() == 129);
        for (int i = 0; i < 16; ++i) CHECK(b[i] == i);
        CHECK(b[16] == 16 && b[18] == 16 && b[19] == 17 && b[127] == 31 && b[128] == 31);
        for (size_t i = 1; i < b.size(); ++i) CHECK(b[i] >= b[i - 1]);
        std::mt19937 g(71);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        auto rnd = [&](size_t r, size_t c, bool grad) {
            Matrix m(r, c);
            for (size_t i = 0; i < r; ++i)
                for (size_t j = 0; j < c; ++j) m(i, j) = nd(g);
            return make_var(m, grad);
        };
        const size_t Tq = 12, dh = 8, nb = 8, heads = 2;
        const std::vector<int> bk = ops::t5_position_buckets(nb, 16);
        Var q = rnd(Tq, dh, true), k = rnd(Tq, dh, true), v = rnd(Tq, dh, true), tb = rnd(nb, heads, true);
        // A zero table is causal attention; a constant added to every bucket changes nothing.
        Var zero = make_var(Matrix(nb, heads));
        Var fa = ops::fused_attention(q, k, v, 0.35f, 0, true);
        Var ra = ops::relpos_attention(q, k, v, zero, 1, bk, 0.35f);
        Matrix shifted = tb->data;
        for (size_t i = 0; i < nb; ++i) shifted(i, 1) += 3.0f;
        Var r1 = ops::relpos_attention(q, k, v, tb, 1, bk, 0.35f);
        Var r2 = ops::relpos_attention(q, k, v, make_var(shifted), 1, bk, 0.35f);
        float dz = 0, ds = 0;
        for (size_t i = 0; i < Tq; ++i)
            for (size_t j = 0; j < dh; ++j) {
                dz = std::max(dz, std::fabs(fa->data(i, j) - ra->data(i, j)));
                ds = std::max(ds, std::fabs(r1->data(i, j) - r2->data(i, j)));
            }
        CHECK(dz < 1e-5f && ds < 1e-5f);
        // The op's gradient against finite differences: q, k, v and the table, two sequences of 6.
        Var R = rnd(Tq, dh, false);
        const std::vector<Var> leaves = {q, k, v, tb};
        auto loss = [&]() {
            return ops::scale(ops::mean(ops::mul(ops::relpos_attention(q, k, v, tb, 1, bk, 0.35f, 6), R)),
                              static_cast<float>(Tq * dh));
        };
        for (const Var& l : leaves) l->grad = Matrix();
        backward(loss());
        double worst = 0;
        size_t n = 0;
        for (const Var& l : leaves)
            for (size_t i = 0; i < l->data.rows(); ++i)
                for (size_t j = 0; j < l->data.cols(); ++j) {
                    const float keep = l->data(i, j), h = 1e-2f;
                    NoGrad ng;
                    l->data(i, j) = keep + h;
                    const double up = loss()->data(0, 0);
                    l->data(i, j) = keep - h;
                    const double dn = loss()->data(0, 0);
                    l->data(i, j) = keep;
                    const double fdv = (up - dn) / (2.0 * h), av = l->grad.rows() ? l->grad(i, j) : 0.0;
                    worst = std::max(worst, std::abs(av - fdv) / (1.0 + std::max(std::abs(av), std::abs(fdv))));
                    ++n;
                }
        CHECK(worst < 2e-3);
        // Column 0 of the table belongs to head 0, which this loss never reads.
        for (size_t i = 0; i < nb; ++i) CHECK(tb->grad(i, 0) == 0.0f);
        // The flex family with position t5_relative: no position table, one bias table for every
        // layer, a whole-model gradient check, and the export refused until keramos has the op.
        parity::FlexConfig ft = small(V, d, H, T);
        ft.pos = "t5_relative";
        ft.rel_buckets = 8;
        ft.rel_max_distance = 16;
        parity::FlexLM t5(ft, 72);
        parity::FlexLM learned(small(V, d, H, T), 72);
        CHECK(t5.parameter_count() == learned.parameter_count() - T * d + 8 * H);
        size_t nm = 0;
        const double wm = model_gradcheck(t5, ids_mod(8, V, 73), {"rel_bias", "attn.c_attn"}, 24, nm);
        CHECK(wm < 2e-3);
        bool refused = false;
        try {
            (void)t5.family_export();
        } catch (const std::exception& e) {
            refused = std::string(e.what()).find("t5_relative") != std::string::npos;
        }
        CHECK(refused || std::getenv("PAPERKILN_IR_UNRATIFIED"));
        std::printf("11. T5 relative bias: buckets(32, 128) as HF's (16 exact, 31 from 127); zero table == causal "
                    "attention (%.2e), shift-invariant (%.2e); op gradcheck %zu entries worst %.2e; flex "
                    "t5_relative gradcheck %zu entries %.2e; GGUF export refused\n",
                    dz, ds, n, worst, nm, wm);
    }

    std::printf("FLEX-OK 11/11\n");
    return 0;
}
