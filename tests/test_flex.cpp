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
//  14. the q/k norm (qk_norm per_head, full): scale invariance per head and over the row, gradient
//  checks,
//      exported as rmsnorm after the split (group head_dim per head)
//  15. an expr on the logits (logits_expr): x*0.125 bit for bit, a soft cap's gradient check, the
//  expr node
//      after the head
//  16. the GELU's two forms: ops::gelu_erf against a binary64 statement of 0.5 x (1 + erf(x / sqrt
//  2)) and its
//      finite differences; flex "gelu_erf" differs from the tanh "gelu", passes a whole-model
//      gradient check and exports gelu {approx: "none"}, the tanh form {approx: "tanh"}; an embed
//      expr before the position table (embed_before_pos, the Transformer's sqrt(d_model)) equals
//      the token table scaled by hand, bit for bit, and exports embedding, expr, pos_embedding in
//      that order
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
        // layer, a whole-model gradient check, and the export's attention.position_bias (keramos 1.3).
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
        const auto fx5 = t5.family_export();
        CHECK(fx5.has_value() && fx5->ir["family_hint"] == "flex");
        size_t with_bias = 0;
        for (const auto& node : fx5->ir["block"]["nodes"])
            if (node["op"] == "attention") {
                CHECK(node["position_bias"]["weight"] == "rel_pos_bias.weight");
                CHECK(node["position_bias"]["buckets"].get<std::vector<int>>() == ops::t5_position_buckets(8, 16));
                ++with_bias;
            }
        CHECK(with_bias == 1);
        std::printf("11. T5 relative bias: buckets(32, 128) as HF's (16 exact, 31 from 127); zero table == causal "
                    "attention (%.2e), shift-invariant (%.2e); op gradcheck %zu entries worst %.2e; flex "
                    "t5_relative gradcheck %zu entries %.2e; exported as attention.position_bias\n",
                    dz, ds, n, worst, nm, wm);
    }

    // 12. Primer's MDHA (qkv_conv): a causal depthwise convolution on the fused projection ----------------
    {
        parity::FlexConfig mc = small(V, d, H, T);
        mc.qkv_conv = 3;
        parity::FlexLM m(mc, 72);
        parity::FlexLM base(small(V, d, H, T), 72);
        CHECK(m.parameter_count() == base.parameter_count() + mc.n_layers * 3 * d * 3);
        const std::vector<int> ids = ids_mod(T, V, 74);
        // The identity kernel (only the current row, weight 1) is the model without the convolution.
        for (auto& b : m.blocks) {
            Matrix& w = b->qkv_conv_w->data;
            for (size_t i = 0; i < w.rows(); ++i)
                for (size_t k = 0; k < w.cols(); ++k) w(i, k) = k + 1 == w.cols() ? 1.0f : 0.0f;
        }
        Var a = m.forward(ids), b = base.forward(ids);
        float did = 0;
        for (size_t i = 0; i < a->data.rows(); ++i)
            for (size_t j = 0; j < a->data.cols(); ++j) did = std::max(did, std::fabs(a->data(i, j) - b->data(i, j)));
        CHECK(did < 1e-5f);
        parity::FlexLM m2(mc, 75);  // its own random kernels again
        // Causal: the last token moves no earlier row; two stacked sequences equal their own forwards.
        std::vector<int> ids2 = ids;
        ids2.back() = (ids2.back() + 1) % static_cast<int>(V);
        Var c1 = m2.forward(ids), c2 = m2.forward(ids2);
        bool causal = true;
        for (size_t i = 0; i + 1 < T; ++i)
            for (size_t j = 0; j < V; ++j) causal = causal && c1->data(i, j) == c2->data(i, j);
        CHECK(causal);
        std::vector<int> two = ids;
        two.insert(two.end(), ids2.begin(), ids2.end());
        Var st = m2.forward(two, T);
        float dst = 0;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < V; ++j)
                dst = std::max({dst, std::fabs(st->data(i, j) - c1->data(i, j)),
                                std::fabs(st->data(T + i, j) - c2->data(i, j))});
        CHECK(dst < 1e-5f);
        size_t nm = 0;
        const double wm = model_gradcheck(m2, ids_mod(8, V, 76), {"qkv_conv", "attn.c_attn"}, 24, nm);
        CHECK(wm < 2e-3);
        const auto fx = m2.family_export();
        CHECK(fx.has_value() && fx->arch == "paperkiln");
        const auto& nodes = fx->ir["block"]["nodes"];
        size_t conv_at = 0, split_at = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i]["op"] == "conv1d_causal") {
                conv_at = i;
                CHECK(nodes[i]["in"][0] == "qkv" && nodes[i]["out"] == "qkv" && nodes[i]["state"] == "qkv_conv");
                CHECK(nodes[i]["channels"] == 3 * d && nodes[i]["kernel"] == 3 && nodes[i]["bias"].is_null());
                CHECK(nodes[i]["weight"] == "blk.{i}.attn_qkv_conv.weight");
            }
            if (nodes[i]["op"] == "split") split_at = i;
        }
        CHECK(conv_at > 0 && conv_at + 1 == split_at);
        const auto& sts = fx->ir["block"]["state"];
        CHECK(sts.size() == 2 && sts[1]["op"] == "conv1d_causal" && sts[1]["tensors"][0]["shape"][0] == 2 &&
              sts[1]["tensors"][0]["shape"][1] == 3 * d);
        size_t conv_tensors = 0;
        for (const auto& t : fx->tensors)
            if (t.name.find("attn_qkv_conv.weight") != std::string::npos) {
                CHECK(t.m.rows() == 3 * d && t.m.cols() == 3);
                ++conv_tensors;
            }
        CHECK(conv_tensors == mc.n_layers);
        std::printf("12. MDHA (qkv_conv 3): identity kernel = no convolution (%.2e); causal; stacked = alone "
                    "(%.2e); gradcheck %zu entries %.2e; exported as conv1d_causal over 3d before the split\n",
                    did, dst, nm, wm);
    }

    // 13. Shaw's relative positions (position shaw_relative) ------------------------------------------------
    {
        parity::FlexConfig sc = small(V, d, H, T);
        sc.pos = "shaw_relative";
        sc.rel_max_distance = 4;
        parity::FlexLM s(sc, 72);
        parity::FlexLM learned(small(V, d, H, T), 72);
        const size_t dkh = d / H;
        CHECK(s.parameter_count() == learned.parameter_count() - T * d + sc.n_layers * 2 * 5 * dkh);
        size_t nm = 0;
        const double wm = model_gradcheck(s, ids_mod(8, V, 77), {"rel_k", "rel_v", "attn.c_attn"}, 24, nm);
        CHECK(wm < 2e-3);
        // One set per head, keys only.
        parity::FlexConfig sp = sc;
        sp.rel_per_head = true;
        sp.rel_values = false;
        parity::FlexLM s2(sp, 78);
        size_t nm2 = 0;
        const double wm2 = model_gradcheck(s2, ids_mod(8, V, 79), {"rel_k", "attn.c_attn"}, 24, nm2);
        CHECK(wm2 < 2e-3);
        for (const auto* mm : {&s, &s2}) {
            const auto fx = mm->family_export();
            CHECK(fx.has_value() && fx->arch == "paperkiln");
            size_t with = 0;
            for (const auto& node : fx->ir["block"]["nodes"])
                if (node["op"] == "attention") {
                    const auto& r = node["relative_kv"];
                    CHECK(r["key"] == "blk.{i}.attn_rel_k.weight" && r["max_distance"] == 4);
                    CHECK(mm->cfg.rel_values ? r["value"] == "blk.{i}.attn_rel_v.weight" : r["value"].is_null());
                    ++with;
                }
            CHECK(with == 1);
            size_t tabs = 0;
            for (const auto& t : fx->tensors) {
                CHECK(t.name != "position_embd.weight");
                if (t.name.find("attn_rel_") != std::string::npos) {
                    CHECK(t.m.rows() == 5 && t.m.cols() == (mm->cfg.rel_per_head ? d : dkh));
                    ++tabs;
                }
            }
            CHECK(tabs == sc.n_layers * (mm->cfg.rel_values ? 2 : 1));
        }
        std::printf("13. Shaw relative positions (K 4): gradcheck shared k+v %zu entries %.2e, per-head keys only "
                    "%zu entries %.2e; exported as attention.relative_kv\n", nm, wm, nm2, wm2);
    }

    // 14. The q/k norm (qk_norm per_head and full): an RMSNorm on q and k after the projection -----------------
    {
        const size_t dkh = d / H;
        // Scale one head's query columns (weights and bias) of every block's fused projection by c.
        auto scale_q = [&](parity::FlexLM& m, size_t h0, size_t h1, float c) {
            for (auto& b : m.blocks) {
                nn::Linear& l = *b->attn->c_attn;
                for (size_t j = h0 * dkh; j < h1 * dkh; ++j) {
                    for (size_t i = 0; i < l.W->data.rows(); ++i) l.W->data(i, j) *= c;
                    if (l.b) l.b->data(0, j) *= c;
                }
            }
        };
        auto maxdiff = [](const Var& a, const Var& b) {
            float w = 0;
            for (size_t i = 0; i < a->data.rows(); ++i)
                for (size_t j = 0; j < a->data.cols(); ++j) w = std::max(w, std::fabs(a->data(i, j) - b->data(i, j)));
            return w;
        };
        const std::vector<int> ids = ids_mod(T, V, 81);
        float inv[2] = {0, 0}, head_only[2] = {0, 0};
        double wg[2] = {0, 0};
        size_t ng[2] = {0, 0};
        int k = 0;
        for (const char* mode : {"per_head", "full"}) {
            parity::FlexConfig qc = small(V, d, H, T);
            qc.norm = "rmsnorm";
            qc.pos = "rope";
            qc.qk_norm = mode;
            parity::FlexLM m(qc, 80);
            scale_q(m, 0, H, 10.0f);  // queries well above the norm's eps, so the invariance is to rounding
            parity::FlexConfig bc = qc;
            bc.qk_norm = "";
            parity::FlexLM base(bc, 80);
            const size_t w = k == 0 ? dkh : d;
            CHECK(m.parameter_count() == base.parameter_count() + qc.n_layers * 2 * w);
            // An RMSNorm row is invariant to a positive scale of its input (to eps): per head, scaling one head's
            // queries moves nothing; full width, scaling all of them moves nothing, one head's alone does.
            const Var y0 = m.forward(ids);
            scale_q(m, 0, k == 0 ? 1 : H, 3.0f);
            inv[k] = maxdiff(m.forward(ids), y0);
            parity::FlexLM m1(qc, 80);
            scale_q(m1, 0, H, 10.0f);
            scale_q(m1, 0, 1, 3.0f);
            head_only[k] = maxdiff(m1.forward(ids), y0);
            parity::FlexLM g(qc, 82);
            wg[k] = model_gradcheck(g, ids_mod(8, V, 83), {"q_norm", "k_norm", "attn.c_attn"}, 24, ng[k]);
            CHECK(wg[k] < 2e-3);
            // Exported: an rmsnorm on q and on k right after the split and before the rope node, group head_dim
            // per head, no group over the full width; weights [w].
            const auto fx = g.family_export();
            CHECK(fx.has_value() && fx->arch == "paperkiln");
            const auto& nodes = fx->ir["block"]["nodes"];
            size_t split_at = 0, rope_at = 0, norms = 0;
            for (size_t i = 0; i < nodes.size(); ++i) {
                if (nodes[i]["op"] == "split") split_at = i;
                if (nodes[i]["op"] == "rope") rope_at = i;
            }
            for (size_t i = split_at + 1; i < rope_at; ++i)
                if (nodes[i]["op"] == "rmsnorm") {
                    const std::string t = norms == 0 ? "q" : "k";
                    CHECK(nodes[i]["in"][0] == t && nodes[i]["out"] == t);
                    CHECK(nodes[i]["weight"] == "blk.{i}.attn_" + t + "_norm.weight");
                    CHECK(k == 0 ? nodes[i]["group"] == dkh : !nodes[i].contains("group"));
                    ++norms;
                }
            CHECK(norms == 2 && rope_at == split_at + 3);
            size_t tens = 0;
            for (const auto& t : fx->tensors)
                if (t.name.find("_norm.weight") != std::string::npos && t.name.find("attn_q") != std::string::npos) {
                    CHECK(t.m.rows() * t.m.cols() == w);
                    ++tens;
                }
            CHECK(tens == qc.n_layers);
            ++k;
        }
        // Invariant (to the eps and rounding): per head to its head's scale, full to the whole row's; one head's
        // scale moves the full-width model by far more.
        const float tol = 1e-4f;
        CHECK(inv[0] < tol && inv[1] < tol && head_only[0] < tol);
        CHECK(head_only[1] > 20.0f * std::max({inv[0], inv[1], head_only[0], 1e-7f}));
        std::printf("14. q/k norm: per head invariant to a head's query scale (%.2e), full invariant to the row's "
                    "(%.2e) and not to one head's (%.2e); gradchecks %zu entries %.2e, %zu entries %.2e; exported as "
                    "rmsnorm after the split (group head_dim per head)\n", head_only[0], inv[1], head_only[1], ng[0],
                    wg[0], ng[1], wg[1]);
    }

    // 15. An expr on the logits (logits_expr): PaLM's scale and Gemma 2's final soft cap -----------------------
    {
        parity::FlexConfig lc = small(V, d, H, T);
        lc.logits_expr = "expr:x*0.125";
        parity::FlexLM m(lc, 84);
        parity::FlexLM base(small(V, d, H, T), 84);
        CHECK(m.parameter_count() == base.parameter_count());
        const std::vector<int> ids = ids_mod(T, V, 85);
        const Var a = m.forward(ids), b = base.forward(ids);
        bool exact = true;
        for (size_t i = 0; i < a->data.rows(); ++i)
            for (size_t j = 0; j < a->data.cols(); ++j) exact = exact && a->data(i, j) == b->data(i, j) * 0.125f;
        CHECK(exact);  // a power-of-two scale: bit for bit
        lc.logits_expr = "expr:30*tanh(x/30)";
        parity::FlexLM cap(lc, 86);
        size_t n = 0;
        const double w = model_gradcheck(cap, ids_mod(8, V, 87), {"head", "wte"}, 24, n);
        CHECK(w < 2e-3);
        const auto fx = cap.family_export();
        CHECK(fx.has_value() && fx->arch == "paperkiln");
        const auto& hn = fx->ir["head"]["nodes"];
        CHECK(hn.back()["op"] == "expr" && hn.back()["in"][0] == "logits" && hn.back()["out"] == "logits");
        CHECK(!cap.gpt2_export_blocker().empty());
        std::printf("15. logits expr: x*0.125 = the head's logits * 0.125 bit for bit; 30*tanh(x/30) gradcheck %zu "
                    "entries %.2e; exported as an expr node after the head's linear\n", n, w);
    }

    // 16. The GELU's two forms and an embed expr before the position table
    // -------------------------------------
    {
        // The op against binary64 and its finite differences, on [-8, 8].
        Matrix xs(1, 161);
        for (size_t j = 0; j < xs.cols(); ++j) xs(0, j) = -8.0f + 0.1f * static_cast<float>(j);
        const Var x = make_var(xs, /*requires_grad=*/true);
        const Var y = ops::gelu_erf(x);
        double worst_fwd = 0.0;
        for (size_t j = 0; j < xs.cols(); ++j) {
            const double v = xs(0, j), want = 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
            worst_fwd =
                std::max(worst_fwd, std::abs(y->data(0, j) - want) / std::max(1.0, std::abs(want)));
        }
        CHECK(worst_fwd < 1e-6);
        backward(ops::scale(ops::mean(y), static_cast<float>(xs.cols())));  // dL/dy = 1
        double worst_op = 0.0;
        for (size_t j = 0; j < xs.cols(); ++j) {
            const double v = xs(0, j), h = 1e-4;
            auto g = [](double t) { return 0.5 * t * (1.0 + std::erf(t / std::sqrt(2.0))); };
            const double fd = (g(v + h) - g(v - h)) / (2.0 * h), a = x->grad(0, j);
            worst_op = std::max(worst_op, std::abs(a - fd) / (1.0 + std::abs(fd)));
        }
        CHECK(worst_op < 1e-5);
        // Not the tanh form: the two differ somewhere on the grid by more than f32 rounding.
        const Var yt = ops::gelu(make_var(xs));
        double gap = 0.0;
        for (size_t j = 0; j < xs.cols(); ++j)
            gap = std::max(gap, double(std::abs(yt->data(0, j) - y->data(0, j))));
        CHECK(gap > 1e-5);

        // The flex model with the exact form: the same parameters as the tanh form's (nn::MLP's
        // Linears and seeds), different logits, a whole-model gradient check, the IR's gelu with
        // approx "none".
        parity::FlexConfig ec = small(V, d, H, T);
        ec.pos = "sinusoidal";  // not the gpt2 flavour, so both forms export through family_export
        parity::FlexConfig tc = ec;
        ec.act = "gelu_erf";
        parity::FlexLM erf_m(ec, 91), tanh_m(tc, 91);
        CHECK(erf_m.parameter_count() == tanh_m.parameter_count());
        const std::vector<int> ids = ids_mod(T, V, 92);
        const Var la = erf_m.forward(ids), lb = tanh_m.forward(ids);
        double diff = 0.0;
        for (size_t i = 0; i < la->data.rows(); ++i)
            for (size_t j = 0; j < la->data.cols(); ++j)
                diff = std::max(diff, double(std::abs(la->data(i, j) - lb->data(i, j))));
        CHECK(diff > 0.0);
        size_t n = 0;
        const double w =
            model_gradcheck(erf_m, ids_mod(8, V, 93), {"mlp.c_fc", "mlp.c_proj", "wte"}, 24, n);
        CHECK(w < 2e-3);
        auto gelu_nodes = [](const parity::FlexLM& m) {
            std::vector<std::string> out;
            const auto exported = m.family_export();
            for (const auto& nd : exported->ir["block"]["nodes"])
                if (nd["op"] == "gelu") out.push_back(nd["approx"].get<std::string>());
            return out;
        };
        CHECK(gelu_nodes(erf_m) == std::vector<std::string>{"none"});
        CHECK(gelu_nodes(tanh_m) == std::vector<std::string>{"tanh"});
        CHECK(!erf_m.gpt2_export_blocker().empty());
        parity::FlexConfig bad = small(V, d, H, T);
        bad.act = "gelu_tanh";  // mtstudio's name for "gelu"; FlexConfig itself refuses an unknown
                                // activation
        CHECK(bad.problem().find("activation must be") != std::string::npos);

        // An embed expr before the table: x*2 on the token rows, then the sinusoidal table, equals
        // a model with no embed expr whose token table is doubled (a power of two: exact), bit for
        // bit.
        parity::FlexConfig bc = small(V, d, H, T);
        bc.pos = "sinusoidal";
        bc.act = "gelu_erf";
        parity::FlexConfig pc = bc;
        bc.embed_expr = "expr:x*2";
        bc.embed_before_pos = true;
        parity::FlexLM before(bc, 94), plain(pc, 94);
        CHECK(before.parameter_count() == plain.parameter_count());
        for (size_t i = 0; i < plain.wte->weight->data.rows(); ++i)
            for (size_t j = 0; j < plain.wte->weight->data.cols(); ++j)
                plain.wte->weight->data(i, j) *= 2.0f;
        const Var pa = before.forward(ids), pb = plain.forward(ids);
        bool same = true;
        for (size_t i = 0; i < pa->data.rows(); ++i)
            for (size_t j = 0; j < pa->data.cols(); ++j)
                same = same && pa->data(i, j) == pb->data(i, j);
        CHECK(same);
        parity::FlexConfig after_c =
            bc;  // the default placement: x*2 after the table, a different model
        after_c.embed_before_pos = false;
        parity::FlexLM after(after_c, 94);
        const Var pc_out = after.forward(ids);
        bool differs = false;
        for (size_t i = 0; i < pa->data.rows(); ++i)
            for (size_t j = 0; j < pa->data.cols(); ++j)
                differs = differs || pa->data(i, j) != pc_out->data(i, j);
        CHECK(differs);
        std::vector<std::string> eops;
        const auto before_export = before.family_export();
        for (const auto& nd : before_export->ir["embed"]["nodes"])
            eops.push_back(nd["op"].get<std::string>());
        CHECK((eops == std::vector<std::string>{"embedding", "expr", "pos_embedding"}));
        eops.clear();
        const auto after_export = after.family_export();
        for (const auto& nd : after_export->ir["embed"]["nodes"])
            eops.push_back(nd["op"].get<std::string>());
        CHECK((eops == std::vector<std::string>{"embedding", "pos_embedding", "expr"}));
        parity::FlexConfig nb = bc;
        nb.pos = "rope";
        CHECK(nb.problem().find("needs a position table") != std::string::npos);
        nb = pc;
        nb.embed_before_pos = true;
        CHECK(nb.problem().find("needs an embed expr") != std::string::npos);

        // A tied head (the Transformer's shared embedding and pre-softmax matrix): no head
        // parameters, the logits h E^T, gradients into E through both uses, exported with tied_to
        // and no output.weight tensor.
        parity::FlexConfig tie_c = bc;
        tie_c.tie_head = true;
        tie_c.place_attn = tie_c.place_ffn = "post";
        parity::FlexLM tied(tie_c, 95);
        auto untied_c = tie_c;
        untied_c.tie_head = false;
        parity::FlexLM untied(untied_c, 95);
        CHECK(untied.parameter_count() - tied.parameter_count() == V * d);
        CHECK(!tied.head);
        size_t nt = 0;
        const double wt = model_gradcheck(tied, ids_mod(8, V, 96), {"wte", "mlp.c_fc"}, 24, nt);
        CHECK(wt < 2e-3);
        const auto tfx = tied.family_export();
        CHECK(tfx.has_value() && tfx->arch == "paperkiln");
        const auto& th = tfx->ir["head"]["nodes"];
        CHECK(th.size() == 1 && th[0]["op"] == "linear" && th[0]["tied_to"] == "token_embd.weight");
        bool has_output = false;
        for (const auto& t : tfx->tensors) has_output = has_output || t.name == "output.weight";
        CHECK(!has_output);
        std::printf(
            "16. GELU forms: gelu_erf vs binary64 %.2e, op fd %.2e, %.2e from the tanh form; flex "
            "gelu_erf "
            "gradcheck %zu entries %.2e, exported gelu approx none (tanh form: approx tanh); embed "
            "expr "
            "before the sinusoidal table = the doubled token table bit for bit, exported "
            "embedding, expr, "
            "pos_embedding; tied head gradcheck %zu entries %.2e, exported with tied_to and no "
            "output.weight\n",
            worst_fwd, worst_op, gap, n, w, nt, wt);
    }

    std::printf("FLEX-OK 16/16\n");
    return 0;
}
