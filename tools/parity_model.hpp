#pragma once
// Shared 4-lane parity model for the SRD experiments (srd_parity.cpp,
// srd_needle.cpp): 2 pre-LN blocks, learned positional embeddings,
// attention flavor pluggable. Kept identical across experiments so results
// compare across runs.
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "microtorch/nn.hpp"
#include "microtorch/srd.hpp"

namespace parity {

using namespace microtorch;

enum class AttnKind { EXACT, KIMI, SRD, SWA };

class ParityLM : public nn::Module {
public:
    // swa_window/swa_sinks are read only for AttnKind::SWA (S1 baseline).
    ParityLM(AttnKind kind, size_t vocab, size_t d, size_t n_heads, size_t n_ctx, unsigned seed,
             size_t swa_window = 64, size_t swa_sinks = 1)
        : kind_(kind) {
        wte = mod<nn::Embedding>("wte", vocab, d, seed + 1);
        wpe = mod<nn::Embedding>("wpe", n_ctx, d, seed + 2);
        for (int b = 0; b < 2; ++b) {
            const unsigned s = seed + 10 * (b + 1);
            ln1.push_back(mod<nn::LayerNorm>("ln1_" + std::to_string(b), d));
            ln2.push_back(mod<nn::LayerNorm>("ln2_" + std::to_string(b), d));
            mlp.push_back(mod<nn::MLP>("mlp_" + std::to_string(b), d, 4 * d, s + 3));
            switch (kind) {
                case AttnKind::EXACT:
                    exact.push_back(
                        mod<nn::CausalSelfAttention>("attn_" + std::to_string(b), d, n_heads, s));
                    break;
                case AttnKind::KIMI:
                    kimi.push_back(
                        mod<nn::KimiLinearAttention>("attn_" + std::to_string(b), d, n_heads, s));
                    break;
                case AttnKind::SRD:
                    srd.push_back(mod<nn::SurpriseRoutedAttention>("attn_" + std::to_string(b), d,
                                                                   n_heads, s));
                    break;
                case AttnKind::SWA:
                    swa.push_back(mod<nn::SlidingWindowAttention>(
                        "attn_" + std::to_string(b), d, n_heads, swa_window, swa_sinks, s));
                    break;
            }
        }
        ln_f = mod<nn::LayerNorm>("ln_f", d);
        head = mod<nn::Linear>("head", d, vocab, false, seed + 99);
    }

    // seq_len 0 = one sequence; > 0 = stacked mini-batch. All three
    // attention kinds are block-aware (exact via the fused mask, kimi
    // via per-block prefix-sum reset, srd through both paths).
    Var forward(const std::vector<int>& ids, size_t seq_len = 0) const {
        const size_t sl = seq_len == 0 ? ids.size() : seq_len;
        std::vector<int> pos(ids.size());
        for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i % sl);
        Var h = ops::add(wte->forward(ids), wpe->forward(pos));
        for (int b = 0; b < 2; ++b) {
            Var a;
            Var n1 = ln1[b]->forward(h);
            switch (kind_) {
                case AttnKind::EXACT:
                    a = exact[b]->forward(n1, seq_len);
                    break;
                case AttnKind::KIMI:
                    a = kimi[b]->forward(n1, seq_len);
                    break;
                case AttnKind::SRD:
                    a = srd[b]->forward(n1, seq_len);
                    break;
                case AttnKind::SWA:
                    a = swa[b]->forward(n1, seq_len);
                    break;
            }
            h = ops::add(h, a);
            h = ops::add(h, mlp[b]->forward(ln2[b]->forward(h)));
        }
        return head->forward(ln_f->forward(h));
    }

    Var mean_gate() const {
        return ops::scale(ops::add(ops::mean(srd[0]->gate()), ops::mean(srd[1]->gate())), 0.5f);
    }
    void set_falsifier(bool on) {
        for (auto& s : srd) s->shuffle_predictor = on;
    }

    AttnKind kind_;
    std::shared_ptr<nn::Embedding> wte, wpe;
    std::vector<std::shared_ptr<nn::LayerNorm>> ln1, ln2;
    std::vector<std::shared_ptr<nn::MLP>> mlp;
    std::vector<std::shared_ptr<nn::CausalSelfAttention>> exact;
    std::vector<std::shared_ptr<nn::KimiLinearAttention>> kimi;
    std::vector<std::shared_ptr<nn::SurpriseRoutedAttention>> srd;
    std::vector<std::shared_ptr<nn::SlidingWindowAttention>> swa;
    std::shared_ptr<nn::LayerNorm> ln_f;
    std::shared_ptr<nn::Linear> head;
};

// The AttnRes LM (TECH_TRANSFER item 1 as a trainable preset): the same
// embedding/head shell as ParityLM, but the residual STREAM is replaced
// by nn::AttnResStack — attention over depth. Each transformer sublayer
// (attn-with-ln, mlp-with-ln) is one AttnRes layer with its own
// pseudo-query, so a 2-block model contributes 4 depth-attention
// sources plus the embedding. Block form (S=2) banks one representation
// per transformer block, K3-style.
class AttnResLM : public nn::Module {
public:
    AttnResLM(size_t vocab, size_t d, size_t n_heads, size_t n_ctx, unsigned seed,
              size_t n_blocks = 2, size_t attnres_block_size = 2) {
        wte = mod<nn::Embedding>("wte", vocab, d, seed + 1);
        wpe = mod<nn::Embedding>("wpe", n_ctx, d, seed + 2);
        std::vector<std::shared_ptr<nn::Module>> owned;
        std::vector<std::function<Var(const Var&)>> fns;
        for (size_t b = 0; b < n_blocks; ++b) {
            const unsigned s = seed + 10 * static_cast<unsigned>(b + 1);
            auto lna = std::make_shared<nn::LayerNorm>(d);
            auto attn = std::make_shared<nn::CausalSelfAttention>(d, n_heads, s);
            auto lnm = std::make_shared<nn::LayerNorm>(d);
            auto mlp = std::make_shared<nn::MLP>(d, 4 * d, s + 3);
            // Two composite sublayers per block; each is one AttnRes
            // source. seq_len flows through the mutable member so the
            // fixed closures stay batch-aware.
            struct AttnSub : nn::Module {
                AttnSub(std::shared_ptr<nn::LayerNorm> l,
                        std::shared_ptr<nn::CausalSelfAttention> a)
                    : ln(l), attn(a) {
                    adopt("ln", l);
                    adopt("attn", a);
                }
                std::shared_ptr<nn::LayerNorm> ln;
                std::shared_ptr<nn::CausalSelfAttention> attn;
            };
            struct MlpSub : nn::Module {
                MlpSub(std::shared_ptr<nn::LayerNorm> l, std::shared_ptr<nn::MLP> m)
                    : ln(l), mlp(m) {
                    adopt("ln", l);
                    adopt("mlp", m);
                }
                std::shared_ptr<nn::LayerNorm> ln;
                std::shared_ptr<nn::MLP> mlp;
            };
            auto asub = std::make_shared<AttnSub>(lna, attn);
            auto msub = std::make_shared<MlpSub>(lnm, mlp);
            owned.push_back(asub);
            owned.push_back(msub);
            fns.push_back([asub, this](const Var& x) {
                return asub->attn->forward(asub->ln->forward(x), cur_seq_len_);
            });
            fns.push_back(
                [msub](const Var& x) { return msub->mlp->forward(msub->ln->forward(x)); });
        }
        stack = mod<nn::AttnResStack>("stack", owned, fns, d, attnres_block_size);
        ln_f = mod<nn::LayerNorm>("ln_f", d);
        head = mod<nn::Linear>("head", d, vocab, false, seed + 99);
    }

    Var forward(const std::vector<int>& ids, size_t seq_len = 0) const {
        cur_seq_len_ = seq_len;
        const size_t sl = seq_len == 0 ? ids.size() : seq_len;
        std::vector<int> pos(ids.size());
        for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i % sl);
        Var h = ops::add(wte->forward(ids), wpe->forward(pos));
        return head->forward(ln_f->forward(stack->forward(h)));
    }

    std::shared_ptr<nn::Embedding> wte, wpe;
    std::shared_ptr<nn::AttnResStack> stack;
    std::shared_ptr<nn::LayerNorm> ln_f;
    std::shared_ptr<nn::Linear> head;

private:
    mutable size_t cur_seq_len_ = 0;
};

// The FLEX family: the paper-faithful decoder. Everything papers/fetch.py
// extracts becomes constructor-real here — depth, d_ff, norm flavor
// (LayerNorm|RMSNorm), FFN activation (GELU|ReLU|SwiGLU), and position
// encoding (learned|sinusoidal). Pre-norm wiring throughout (post-norm is
// not yet a knob; fetch.py does not extract it either). RoPE routes to
// the llama family instead — its block already owns that composition.
//
// Equivalence pin (tests/test_flex.cpp): with norm=layernorm, act=gelu,
// pos=learned, n_layers=2, d_ff=4d, FlexLM reproduces ParityLM(EXACT)
// bit-for-bit — same seed layout, same modules, same tape.
struct FlexConfig {
    size_t vocab = 0, d = 128, n_layers = 2, n_heads = 4, d_ff = 512, n_ctx = 128;
    std::string norm = "layernorm";  // layernorm | rmsnorm
    std::string act = "gelu";        // gelu | relu | swiglu
    std::string pos = "learned";     // learned | sinusoidal
    // Registry #0001 (Highway Networks, arXiv 1505.00387): how each
    // sublayer's output combines with its input stream.
    //   residual: y = x + f(x)                      (default; the pin)
    //   highway:  y = x + T(x) * (f(x) - x),  T = sigmoid(W_T x + b_T)
    //   plain:    y = f(x)                          (no skip at all)
    std::string residual = "residual";
    float gate_bias_init = -2.0f;  // paper's negative bias: carry-dominant start
    // Deep SWA (ROADMAP 1a): sliding-window attention at ANY depth.
    // SlidingWindowAttention is interface- and seed-identical to
    // CausalSelfAttention, so attention="swa" at n_layers=2 with default
    // flavors reproduces ParityLM(SWA) bitwise (tests/test_deep_swa.cpp).
    std::string attention = "exact";  // exact | swa
    size_t window = 64, sinks = 1;    // read only when attention == "swa"
};

class FlexBlock : public nn::Module {
public:
    FlexBlock(const FlexConfig& c, unsigned s)
        : norm_(c.norm), act_(c.act), residual_(c.residual) {
        if (residual_ == "highway") {
            // One gate per sublayer, T(x) on the raw stream. Seeds are far
            // from the existing layout so the equivalence pin's seed map
            // is untouched for default configs.
            gate_a = mod<nn::Linear>("gate_attn", c.d, c.d, true, s + 200);
            gate_m = mod<nn::Linear>("gate_mlp", c.d, c.d, true, s + 201);
            for (size_t j = 0; j < c.d; ++j) {
                gate_a->b->data(0, j) = c.gate_bias_init;
                gate_m->b->data(0, j) = c.gate_bias_init;
            }
        }
        if (norm_ == "layernorm") {
            ln1 = mod<nn::LayerNorm>("ln_1", c.d);
            ln2 = mod<nn::LayerNorm>("ln_2", c.d);
        } else {
            n1_w = reg("norm1.weight", Matrix(1, c.d, 1.0f));
            n2_w = reg("norm2.weight", Matrix(1, c.d, 1.0f));
        }
        if (c.attention == "swa")
            swa_attn = mod<nn::SlidingWindowAttention>("attn", c.d, c.n_heads,
                                                       c.window, c.sinks, s);
        else
            attn = mod<nn::CausalSelfAttention>("attn", c.d, c.n_heads, s);
        if (act_ == "swiglu") {
            gate = mod<nn::Linear>("mlp.gate_proj", c.d, c.d_ff, false, s + 3 + 17);
            up = mod<nn::Linear>("mlp.up_proj", c.d, c.d_ff, false, s + 3 + 18);
            down = mod<nn::Linear>("mlp.down_proj", c.d_ff, c.d, false, s + 3 + 19);
        } else if (act_ == "gelu") {
            mlp = mod<nn::MLP>("mlp", c.d, c.d_ff, s + 3);  // ParityLM's exact layout
        } else {  // relu: nn::MLP's Linears (same seeds), relu in between
            fc = mod<nn::Linear>("mlp.c_fc", c.d, c.d_ff, true, s + 3 + 17);
            proj = mod<nn::Linear>("mlp.c_proj", c.d_ff, c.d, true, s + 3 + 19);
        }
    }

    // Sublayer combine per FlexConfig::residual. Highway's identity uses
    // the algebraic rewrite y = x + T*(f-x) (no ones tensor needed); at
    // b_T = -2, T ~= 0.12, so blocks start carry-dominant as published.
    Var combine(const Var& x, const Var& f,
                const std::shared_ptr<nn::Linear>& g) const {
        if (residual_ == "highway")
            return ops::add(x, ops::mul(ops::sigmoid(g->forward(x)),
                                        ops::sub(f, x)));
        if (residual_ == "plain") return f;
        return ops::add(x, f);
    }

    Var forward(const Var& x, size_t seq_len = 0) const {
        auto norm1 = [&](const Var& v) { return ln1 ? ln1->forward(v) : ops::rmsnorm(v, n1_w); };
        auto norm2 = [&](const Var& v) { return ln2 ? ln2->forward(v) : ops::rmsnorm(v, n2_w); };
        Var a_out = attn ? attn->forward(norm1(x), seq_len)
                         : swa_attn->forward(norm1(x), seq_len);
        Var h = combine(x, a_out, gate_a);
        Var n = norm2(h);
        Var f;
        if (mlp) {
            f = mlp->forward(n);
        } else if (gate) {
            f = down->forward(ops::mul(ops::silu(gate->forward(n)), up->forward(n)));
        } else {
            f = proj->forward(ops::relu(fc->forward(n)));
        }
        return combine(h, f, gate_m);
    }

    std::string norm_, act_, residual_;
    std::shared_ptr<nn::LayerNorm> ln1, ln2;
    Var n1_w, n2_w;
    std::shared_ptr<nn::CausalSelfAttention> attn;
    std::shared_ptr<nn::SlidingWindowAttention> swa_attn;  // attention="swa"
    std::shared_ptr<nn::MLP> mlp;
    std::shared_ptr<nn::Linear> gate, up, down, fc, proj;
    std::shared_ptr<nn::Linear> gate_a, gate_m;  // highway only
};

class FlexLM : public nn::Module {
public:
    FlexLM(const FlexConfig& c, unsigned seed) : cfg(c) {
        wte = mod<nn::Embedding>("wte", c.vocab, c.d, seed + 1);
        if (c.pos == "learned") {
            wpe = mod<nn::Embedding>("wpe", c.n_ctx, c.d, seed + 2);
        } else {  // sinusoidal: the Vaswani fixed table, not a parameter
            sin_table = Matrix(c.n_ctx, c.d);
            for (size_t p = 0; p < c.n_ctx; ++p)
                for (size_t i = 0; i < c.d; ++i) {
                    const double angle = p / std::pow(10000.0, double(2 * (i / 2)) / double(c.d));
                    sin_table(p, i) =
                        static_cast<float>(i % 2 == 0 ? std::sin(angle) : std::cos(angle));
                }
        }
        for (size_t b = 0; b < c.n_layers; ++b)
            blocks.push_back(mod<FlexBlock>("layers." + std::to_string(b), c,
                                            seed + 10 * static_cast<unsigned>(b + 1)));
        ln_f_ln = c.norm == "layernorm" ? mod<nn::LayerNorm>("ln_f", c.d) : nullptr;
        if (!ln_f_ln) ln_f_w = reg("norm.weight", Matrix(1, c.d, 1.0f));
        head = mod<nn::Linear>("head", c.d, c.vocab, false, seed + 99);
    }

    Var forward(const std::vector<int>& ids, size_t seq_len = 0) const {
        const size_t sl = seq_len == 0 ? ids.size() : seq_len;
        std::vector<int> pos(ids.size());
        for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i % sl);
        Var h;
        if (wpe) {
            h = ops::add(wte->forward(ids), wpe->forward(pos));
        } else {
            Matrix pe(ids.size(), cfg.d);
            for (size_t r = 0; r < ids.size(); ++r)
                for (size_t j = 0; j < cfg.d; ++j) pe(r, j) = sin_table(pos[r], j);
            h = ops::add(wte->forward(ids), make_var(std::move(pe), false));
        }
        for (auto& b : blocks) h = b->forward(h, seq_len);
        Var n = ln_f_ln ? ln_f_ln->forward(h) : ops::rmsnorm(h, ln_f_w);
        return head->forward(n);
    }

    FlexConfig cfg;
    std::shared_ptr<nn::Embedding> wte, wpe;
    Matrix sin_table;
    std::vector<std::shared_ptr<FlexBlock>> blocks;
    std::shared_ptr<nn::LayerNorm> ln_f_ln;
    Var ln_f_w;
    std::shared_ptr<nn::Linear> head;
};

}  // namespace parity
