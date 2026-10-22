#pragma once
// Shared 4-lane parity model for the SRD experiments (srd_parity.cpp,
// srd_needle.cpp): 2 pre-LN blocks, learned positional embeddings,
// attention flavor pluggable. Kept identical across experiments so results
// compare across runs.
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "microtorch/expr.hpp"
#include "microtorch/gguf.hpp"
#include "microtorch/ir.hpp"
#include "microtorch/mamba.hpp"
#include "microtorch/nn.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/srd.hpp"

namespace parity {

using namespace microtorch;

enum class AttnKind { EXACT, KIMI, SRD, SWA };

// GPT-2 export helpers (gguf::export_gguf_gpt2 reads HF GPT-2 names, which
// are also nn::GPT2's): one pre-LN block's tensors as h.<b>.*, copied.
// Attn: nn::CausalSelfAttention or nn::SlidingWindowAttention (same layout).
template <class Attn>
void put_gpt2_block(std::map<std::string, Matrix>& sd, size_t b, const nn::LayerNorm& ln1, const Attn& attn,
                    const nn::LayerNorm& ln2, const nn::MLP& mlp) {
    const std::string h = "h." + std::to_string(b) + ".";
    sd[h + "ln_1.weight"] = ln1.weight->data;
    sd[h + "ln_1.bias"] = ln1.bias->data;
    sd[h + "attn.c_attn.weight"] = attn.c_attn->W->data;
    sd[h + "attn.c_attn.bias"] = attn.c_attn->b->data;
    sd[h + "attn.c_proj.weight"] = attn.c_proj->W->data;
    sd[h + "attn.c_proj.bias"] = attn.c_proj->b->data;
    sd[h + "ln_2.weight"] = ln2.weight->data;
    sd[h + "ln_2.bias"] = ln2.bias->data;
    sd[h + "mlp.c_fc.weight"] = mlp.c_fc->W->data;
    sd[h + "mlp.c_fc.bias"] = mlp.c_fc->b->data;
    sd[h + "mlp.c_proj.weight"] = mlp.c_proj->W->data;
    sd[h + "mlp.c_proj.bias"] = mlp.c_proj->b->data;
}

inline void put_gpt2_shell(std::map<std::string, Matrix>& sd, const nn::Embedding& wte, const nn::Embedding& wpe,
                           const nn::LayerNorm& ln_f, const nn::Linear& head) {
    sd["wte.weight"] = wte.weight->data;
    sd["wpe.weight"] = wpe.weight->data;
    sd["ln_f.weight"] = ln_f.weight->data;
    sd["ln_f.bias"] = ln_f.bias->data;
    sd["lm_head.weight"] = head.W->data;  // untied: its own [d, vocab] matrix
}

// A family's GGUF + IR export (microtorch/ir.hpp, Families): the gpt2
// architecture when the tensors are exactly llama.cpp's gpt2 set (gpt2_sd,
// HF GPT-2 names, for gguf::export_gguf_gpt2), else "paperkiln" (tensors,
// for gguf::export_gguf_paperkiln). Either way the IR is the model.
struct FamilyExport {
    std::string arch;  // "gpt2" | "paperkiln"
    ir::json ir;
    std::map<std::string, Matrix> gpt2_sd;
    std::vector<gguf::NamedTensor> tensors;
    size_t heads = 0, d_ff = 0;
    size_t constants = 0;  // tensors that are no parameter (a fixed table, an unread zero query)
    std::string warning;   // what an engine reading the tensors without the IR gets wrong
};

inline void put_tensor(std::vector<gguf::NamedTensor>& t, const std::string& name, const Matrix& m,
                       gguf::NamedTensor::Layout layout) {
    t.push_back({name, m, layout});
}
// weight [in, out] -> base.weight [out][in], bias -> base.bias.
inline void put_linear(std::vector<gguf::NamedTensor>& t, const std::string& base, const nn::Linear& l) {
    put_tensor(t, base + ".weight", l.W->data, gguf::NamedTensor::Layout::PROJ);
    if (l.b) put_tensor(t, base + ".bias", l.b->data, gguf::NamedTensor::Layout::VEC);
}
inline void put_layernorm(std::vector<gguf::NamedTensor>& t, const std::string& base, const nn::LayerNorm& ln) {
    put_tensor(t, base + ".weight", ln.weight->data, gguf::NamedTensor::Layout::VEC);
    put_tensor(t, base + ".bias", ln.bias->data, gguf::NamedTensor::Layout::VEC);
}

// The document around a family's nodes, laid out as nn::gpt2_ir's.
// `positions`: the model reads input "pos" (no recurrent-only model does).
inline ir::json family_doc(const std::string& hint, size_t vocab, size_t d, size_t n_layers, size_t ctx,
                           bool positions, const ir::json& embed, const ir::json& state, const ir::json& nodes,
                           const ir::json& head) {
    bool kv = false, recurrent = false;
    for (const auto& s : state) (s["kind"] == "kv" ? kv : recurrent) = true;
    ir::json doc;
    doc["ir_version"] = ir::kVersion;
    doc["family_hint"] = hint;
    doc["model"] = {{"vocab", vocab}, {"d", d}, {"n_layers", n_layers}, {"ctx", ctx}, {"dtype", "f32"}};
    doc["inputs"] = ir::json::array({{{"name", "tokens"}, {"kind", "token_ids"}}});
    if (positions)
        doc["inputs"].push_back({{"name", "pos"},
                                 {"kind", "positions"},
                                 {"note", "absolute positions; prefill 0..T-1, decode = tokens seen so far; must "
                                          "stay below model.ctx"}});
    doc["embed"] = {{"nodes", embed}};
    doc["block"] = {{"repeat", n_layers}, {"state", state}, {"nodes", nodes}};
    doc["head"] = {{"nodes", head}};
    doc["outputs"] = ir::json::array({{{"name", "logits"}, {"kind", "logits"}}});
    std::string dec = "the same graph with T = 1";
    if (kv) dec += "; attention reads and appends to its kv state";
    if (recurrent) dec += "; each recurrent op applies one update to its recurrent state";
    doc["modes"] = {{"prefill", recurrent ? "all nodes over T tokens; each recurrent op scans t = 0..T-1 in order"
                                          : "all nodes over T tokens"},
                    {"decode", dec}};
    return doc;
}

// Learned positions + the ln_f / untied head shell of the gpt2-shaped models.
inline ir::json gpt2_embed_ir(const nn::Embedding& wte) {
    return ir::json::array({wte.export_ir("tokens", "x", "token_embd.weight"),
                            ir::pos_embedding("x", "pos", "position_embd.weight", "x")});
}
inline ir::json gpt2_head_ir(const nn::LayerNorm& ln_f, const nn::Linear& head, const std::string& in = "x") {
    return ir::json::array({ln_f.export_ir(in, "h", "output_norm"), head.export_ir("h", "logits", "output.weight")});
}
// x += mlp(ln2 x), as gpt2_block_ir.
inline void gpt2_ffn_ir(ir::json& n, const nn::LayerNorm& ln2, const nn::MLP& mlp) {
    const std::string p = "blk.{i}.";
    n.push_back(ln2.export_ir("x", "h2", p + "ffn_norm"));
    for (const auto& node : mlp.export_ir("h2", "f", p + "ffn_up.weight", p + "ffn_down.weight")) n.push_back(node);
    n.push_back(ir::binary("add", "x", "f", "x"));
}

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

    // GPT-2 export (GGUF arch "gpt2" + IR): exact or sliding-window
    // attention (same tensors; the IR carries the window and sinks, and
    // llama.cpp's gpt2 computes dense attention). kimi and srd have no
    // llama.cpp gpt2 form and no IR op yet.
    std::string gpt2_export_blocker() const {
        return kind_ == AttnKind::EXACT || kind_ == AttnKind::SWA
                   ? ""
                   : "attention is not softmax attention (kimi/srd have no gpt2 export)";
    }
    bool windowed() const { return kind_ == AttnKind::SWA; }
    ir::json export_ir() const {
        if (!gpt2_export_blocker().empty()) throw std::runtime_error("ParityLM export_ir: " + gpt2_export_blocker());
        std::vector<ir::json> blocks;
        const size_t ctx = wpe->weight->data.rows();
        for (int b = 0; b < 2; ++b)
            blocks.push_back(windowed() ? nn::gpt2_block_ir(*ln1[b], *swa[b], *ln2[b], *mlp[b], ctx)
                                        : nn::gpt2_block_ir(*ln1[b], *exact[b], *ln2[b], *mlp[b], ctx));
        return nn::gpt2_ir(*wte, *wpe, blocks, *ln_f, head.get());
    }
    std::map<std::string, Matrix> gpt2_state_dict() const {
        if (!gpt2_export_blocker().empty()) throw std::runtime_error("ParityLM gpt2_state_dict: " + gpt2_export_blocker());
        std::map<std::string, Matrix> sd;
        put_gpt2_shell(sd, *wte, *wpe, *ln_f, *head);
        for (int b = 0; b < 2; ++b) {
            if (windowed())
                put_gpt2_block(sd, b, *ln1[b], *swa[b], *ln2[b], *mlp[b]);
            else
                put_gpt2_block(sd, b, *ln1[b], *exact[b], *ln2[b], *mlp[b]);
        }
        return sd;
    }

    // kimi (gpt2 tensors, linear_attention in the IR) and srd (paperkiln:
    // the router's predictor is not a gpt2 tensor); nullopt for exact/swa,
    // which export through gpt2_state_dict/export_ir above.
    std::optional<FamilyExport> family_export() const {
        if (kind_ != AttnKind::KIMI && kind_ != AttnKind::SRD) return std::nullopt;
        const size_t vocab = wte->weight->data.rows(), d = wte->weight->data.cols();
        const size_t ctx = wpe->weight->data.rows();
        const std::string p = "blk.{i}.";
        FamilyExport fx;
        fx.d_ff = 4 * d;
        ir::json n = ir::json::array(), state = ir::json::array();
        n.push_back(ln1[0]->export_ir("x", "h", p + "attn_norm"));
        if (kind_ == AttnKind::KIMI) {
            const auto& a = *kimi[0];
            fx.heads = a.H;
            n.push_back(a.c_attn->export_ir("h", "qkv", p + "attn_qkv.weight"));
            n.push_back(ir::split("qkv", {d, d, d}, {"q", "k", "v"}));
            n.push_back(ir::linear_attention("q", "k", "v", "lin", a.H, a.dk, "a"));
            n.push_back(a.c_proj->export_ir("a", "o", p + "attn_output.weight"));
            state.push_back(ir::linear_attention_state("lin", a.H, a.dk));
        } else {
            // SurpriseRoutedAttention::forward, node for node: the router on
            // the normalised input, both paths on one qkv, the per-token blend.
            const auto& a = *srd[0];
            fx.heads = a.H;
            n.push_back(a.predictor->fc1->export_ir("h", "p", p + "srd_pred_up.weight"));
            n.push_back(ir::gelu("p", "tanh", "p"));
            n.push_back(a.predictor->fc2->export_ir("p", "p", p + "srd_pred_down.weight"));
            n.push_back(ir::binary("sub", "h", "p", "p"));
            n.push_back(ir::rms_row("p", 1e-8f, "g"));  // ops::rms_row's default eps
            n.push_back(ir::affine("g", nn::SurpriseRoutedAttention::GATE_SCALE,
                                   nn::SurpriseRoutedAttention::GATE_BIAS, "g"));
            n.push_back(ir::unary("sigmoid", "g", "g"));
            n.push_back(ir::affine("g", -1.0f, 1.0f, "gi"));
            n.push_back(a.c_attn->export_ir("h", "qkv", p + "attn_qkv.weight"));
            n.push_back(ir::split("qkv", {d, d, d}, {"q", "k", "v"}));
            const float sc = 1.0f / std::sqrt(static_cast<float>(a.dk));
            n.push_back(ir::attention("q", "k", "v", "kv", a.H, a.H, a.dk, sc, /*causal=*/true, 0, 0, "a"));
            n.push_back(ir::linear_attention("q", "k", "v", "lin", a.H, a.dk, "l"));
            n.push_back(ir::binary("mul_col", "a", "g", "a"));
            n.push_back(ir::binary("mul_col", "l", "gi", "l"));
            n.push_back(ir::binary("add", "a", "l", "a"));
            n.push_back(a.c_proj->export_ir("a", "o", p + "attn_output.weight"));
            state.push_back(ir::kv_state("kv", a.H, a.H, a.dk, ctx, 0, 0));
            state.push_back(ir::linear_attention_state("lin", a.H, a.dk));
        }
        n.push_back(ir::binary("add", "x", "o", "x"));
        gpt2_ffn_ir(n, *ln2[0], *mlp[0]);
        fx.ir = family_doc(kind_ == AttnKind::KIMI ? "kimi" : "srd", vocab, d, 2, ctx, /*positions=*/true,
                           gpt2_embed_ir(*wte), state, n, gpt2_head_ir(*ln_f, *head));
        if (kind_ == AttnKind::KIMI) {
            fx.arch = "gpt2";
            put_gpt2_shell(fx.gpt2_sd, *wte, *wpe, *ln_f, *head);
            for (int b = 0; b < 2; ++b) put_gpt2_block(fx.gpt2_sd, b, *ln1[b], *kimi[b], *ln2[b], *mlp[b]);
            fx.warning =
                "kimi linear attention lives in the embedded IR only: an engine reading the gpt2 tensors "
                "without it (llama.cpp) computes softmax attention, a different model";
            return fx;
        }
        fx.arch = "paperkiln";
        using L = gguf::NamedTensor::Layout;
        put_tensor(fx.tensors, "token_embd.weight", wte->weight->data, L::ROWS);
        put_tensor(fx.tensors, "position_embd.weight", wpe->weight->data, L::ROWS);
        for (int b = 0; b < 2; ++b) {
            const std::string g = "blk." + std::to_string(b) + ".";
            const auto& a = *srd[b];
            put_layernorm(fx.tensors, g + "attn_norm", *ln1[b]);
            put_linear(fx.tensors, g + "srd_pred_up", *a.predictor->fc1);
            put_linear(fx.tensors, g + "srd_pred_down", *a.predictor->fc2);
            put_linear(fx.tensors, g + "attn_qkv", *a.c_attn);
            put_linear(fx.tensors, g + "attn_output", *a.c_proj);
            put_layernorm(fx.tensors, g + "ffn_norm", *ln2[b]);
            put_linear(fx.tensors, g + "ffn_up", *mlp[b]->c_fc);
            put_linear(fx.tensors, g + "ffn_down", *mlp[b]->c_proj);
        }
        put_layernorm(fx.tensors, "output_norm", *ln_f);
        put_linear(fx.tensors, "output", *head);
        return fx;
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
            // Kept for export only (registration is through `owned`).
            ln_a.push_back(lna);
            attns.push_back(attn);
            ln_m.push_back(lnm);
            mlps.push_back(mlp);
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

    // microtorch/ir.hpp, Families: attnres. The stack's block form with
    // two sublayers per bank entry is one transformer block per IR block:
    // block i's attention sublayer is AttnRes layer 2i (query w[2i]), its
    // FFN layer 2i + 1, the final aggregation w[2n].
    std::optional<FamilyExport> family_export() const {
        const size_t n_blocks = attns.size();
        if (stack->block_size != 2)
            throw std::runtime_error("AttnResLM export: needs the block form with two sublayers per block");
        const size_t vocab = wte->weight->data.rows(), d = wte->weight->data.cols();
        const size_t ctx = wpe->weight->data.rows();
        const float eps = ops::kRmsNormEps;  // depth_attend's parameter-free rmsnorm
        const std::string p = "blk.{i}.";
        ir::json embed = gpt2_embed_ir(*wte);
        embed.push_back(ir::list({"x"}, "bank"));
        ir::json n = ir::json::array();
        n.push_back(ir::depth_attend({"bank"}, p + "attnres_attn_q.weight", eps, "h"));
        n.push_back(ln_a[0]->export_ir("h", "h", p + "attn_norm"));
        for (const auto& node :
             attns[0]->export_ir("h", "o", p + "attn_qkv.weight", p + "attn_output.weight", "kv"))
            n.push_back(node);
        n.push_back(ir::depth_attend({"bank", "o"}, p + "attnres_ffn_q.weight", eps, "h2"));
        n.push_back(ln_m[0]->export_ir("h2", "h2", p + "ffn_norm"));
        for (const auto& node : mlps[0]->export_ir("h2", "f", p + "ffn_up.weight", p + "ffn_down.weight"))
            n.push_back(node);
        n.push_back(ir::binary("add", "o", "f", "o"));  // the block's bank entry: the sum of its two outputs
        n.push_back(ir::append("bank", "o", "bank"));
        ir::json head_nodes = ir::json::array({ir::depth_attend({"bank"}, "attnres_out_q.weight", eps, "x")});
        for (const auto& node : gpt2_head_ir(*ln_f, *head)) head_nodes.push_back(node);
        const auto& a0 = *attns[0];
        FamilyExport fx;
        fx.arch = "paperkiln";
        fx.heads = a0.H;
        fx.d_ff = 4 * d;
        fx.ir = family_doc("attnres", vocab, d, n_blocks, ctx, /*positions=*/true, embed,
                           ir::json::array({ir::kv_state("kv", a0.H, a0.H, a0.dk, ctx, 0, 0)}), n, head_nodes);
        using L = gguf::NamedTensor::Layout;
        put_tensor(fx.tensors, "token_embd.weight", wte->weight->data, L::ROWS);
        put_tensor(fx.tensors, "position_embd.weight", wpe->weight->data, L::ROWS);
        for (size_t b = 0; b < n_blocks; ++b) {
            const std::string g = "blk." + std::to_string(b) + ".";
            // Layer 0 has no query (one source); a zero row keeps every
            // block's tensor set the same, and depth_attend never reads it.
            put_tensor(fx.tensors, g + "attnres_attn_q.weight", b == 0 ? Matrix(1, d) : stack->w[2 * b]->data,
                       L::VEC);
            put_layernorm(fx.tensors, g + "attn_norm", *ln_a[b]);
            put_linear(fx.tensors, g + "attn_qkv", *attns[b]->c_attn);
            put_linear(fx.tensors, g + "attn_output", *attns[b]->c_proj);
            put_tensor(fx.tensors, g + "attnres_ffn_q.weight", stack->w[2 * b + 1]->data, L::VEC);
            put_layernorm(fx.tensors, g + "ffn_norm", *ln_m[b]);
            put_linear(fx.tensors, g + "ffn_up", *mlps[b]->c_fc);
            put_linear(fx.tensors, g + "ffn_down", *mlps[b]->c_proj);
        }
        put_tensor(fx.tensors, "attnres_out_q.weight", stack->w[2 * n_blocks]->data, L::VEC);
        put_layernorm(fx.tensors, "output_norm", *ln_f);
        put_linear(fx.tensors, "output", *head);
        fx.constants = 1;
        return fx;
    }

    std::shared_ptr<nn::Embedding> wte, wpe;
    std::vector<std::shared_ptr<nn::LayerNorm>> ln_a, ln_m;
    std::vector<std::shared_ptr<nn::CausalSelfAttention>> attns;
    std::vector<std::shared_ptr<nn::MLP>> mlps;
    std::shared_ptr<nn::AttnResStack> stack;
    std::shared_ptr<nn::LayerNorm> ln_f;
    std::shared_ptr<nn::Linear> head;

private:
    mutable size_t cur_seq_len_ = 0;
};

// The FLEX family: the paper-faithful decoder. Everything papers/fetch.py
// extracts becomes constructor-real here — depth, d_ff, norm flavor
// (LayerNorm|RMSNorm, or one defined at runtime: "expr:<e>"), FFN
// activation (GELU|ReLU|SwiGLU, or "expr:<e>", microtorch/expr.hpp), and
// position encoding (learned|sinusoidal|rope). Pre-norm wiring throughout
// (post-norm is not yet a knob). mtstudio sends position=rope with the
// llama flavours (rmsnorm, swiglu, residual, exact) to the llama family;
// rope with any other flavour is flex.
//
//   pos "rope": no position table; every head of q and k rotates after the
//     fused qkv projection, ops::apply_rope with the llama block's
//     convention (interleaved pairs (x[2j], x[2j+1]) within a head, angle
//     pos * rope_theta^(-2j/head_dim), all heads), positions restarting at
//     0 for each sequence of a stacked batch. Exported as the IR rope node
//     between split and attention, as the llama block's.
//   norm "expr:<e>": y = e(x) * g + b row by row, e an expr over the one
//     input x (the row reductions rowsum/rowmean/rowmax allowed), g and b
//     learned [d] (init 1 and 0: layernorm's affine). Exported as an expr
//     node, then a linear whose weight is diag(g) [d, d] and whose bias is
//     b (blk.{i}.attn_norm_gain / ffn_norm_gain, output_norm_gain): every
//     off-diagonal product is +-0, so the linear computes e(x) * g + b
//     exactly as the training ops (mul_row, add_bias) do.
//
// Equivalence pin (tests/test_flex.cpp): with norm=layernorm, act=gelu,
// pos=learned, n_layers=2, d_ff=4d, FlexLM reproduces ParityLM(EXACT)
// bit-for-bit — same seed layout, same modules, same tape.
struct FlexConfig {
    size_t vocab = 0, d = 128, n_layers = 2, n_heads = 4, d_ff = 512, n_ctx = 128;
    std::string norm = "layernorm";  // layernorm | rmsnorm | expr:<e> (e over x, row reductions allowed)
    std::string act = "gelu";        // gelu | relu | swiglu | expr:<e> (e over x)
    std::string pos = "learned";     // learned | sinusoidal | rope
    float rope_theta = 10000.0f;     // pos == "rope" (the llama block's default)
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

// The expr norm: e(x) * g + b (FlexConfig, norm "expr:<e>").
inline Var flex_expr_norm(const Var& x, const Expr& e, const Var& g, const Var& b) {
    return ops::add_bias(ops::mul_row(ops::expr({x}, e), g), b);
}

// diag(g) as a linear weight [d, d] (the expr norm's gain in the GGUF).
inline Matrix flex_diag(const Matrix& g) {
    Matrix m(g.cols(), g.cols(), 0.0f);
    for (size_t j = 0; j < g.cols(); ++j) m(j, j) = g(0, j);
    return m;
}

class FlexBlock : public nn::Module {
public:
    FlexBlock(const FlexConfig& c, unsigned s)
        : norm_(c.norm), act_(c.act), residual_(c.residual), rope_(c.pos == "rope"), rope_theta_(c.rope_theta) {
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
            if (norm_.rfind("expr:", 0) == 0) {  // e(x) * g + b: the gain above, a bias here
                norm_expr = std::make_shared<const Expr>(Expr::parse(norm_.substr(5), 1));
                n1_b = reg("norm1.bias", Matrix(1, c.d, 0.0f));
                n2_b = reg("norm2.bias", Matrix(1, c.d, 0.0f));
            }
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
        } else {  // relu or expr: nn::MLP's Linears (same seeds), the activation in between
            if (act_.rfind("expr:", 0) == 0)
                act_expr = std::make_shared<const Expr>(Expr::parse(act_.substr(5), 1));
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

    // The attention sublayer. Without rope, the module's own forward. With
    // rope, CausalSelfAttention/SlidingWindowAttention::forward with
    // ops::apply_rope (all heads) between the fused qkv projection and the
    // per-head attention, as LlamaBlock::forward places it.
    Var attend(const Var& n, size_t seq_len) const {
        if (!rope_) return attn ? attn->forward(n, seq_len) : swa_attn->forward(n, seq_len);
        const nn::Linear& c_attn = attn ? *attn->c_attn : *swa_attn->c_attn;
        const nn::Linear& c_proj = attn ? *attn->c_proj : *swa_attn->c_proj;
        const size_t H = attn ? attn->H : swa_attn->H, dk = attn ? attn->dk : swa_attn->dk, d = H * dk;
        const size_t rows = n->data.rows(), sl = seq_len == 0 ? rows : seq_len;
        std::vector<int> pos(rows);
        for (size_t i = 0; i < rows; ++i) pos[i] = static_cast<int>(i % sl);
        const Var qkv = ops::apply_rope(c_attn.forward(n), pos, rope_theta_, dk, /*all_heads=*/true);
        const float sc = 1.0f / std::sqrt(static_cast<float>(dk));
        std::vector<Var> heads;
        for (size_t h = 0; h < H; ++h) {
            Var q = ops::slice_cols(qkv, h * dk, (h + 1) * dk);
            Var k = ops::slice_cols(qkv, d + h * dk, d + (h + 1) * dk);
            Var v = ops::slice_cols(qkv, 2 * d + h * dk, 2 * d + (h + 1) * dk);
            heads.push_back(attn ? ops::fused_attention(q, k, v, sc, seq_len, attn->causal)
                                 : ops::swa_attention(q, k, v, sc, swa_attn->window, swa_attn->sinks, seq_len));
        }
        return c_proj.forward(ops::concat_cols(heads));
    }

    Var forward(const Var& x, size_t seq_len = 0) const {
        auto norm1 = [&](const Var& v) {
            return ln1 ? ln1->forward(v) : norm_expr ? flex_expr_norm(v, *norm_expr, n1_w, n1_b) : ops::rmsnorm(v, n1_w);
        };
        auto norm2 = [&](const Var& v) {
            return ln2 ? ln2->forward(v) : norm_expr ? flex_expr_norm(v, *norm_expr, n2_w, n2_b) : ops::rmsnorm(v, n2_w);
        };
        Var a_out = attend(norm1(x), seq_len);
        Var h = combine(x, a_out, gate_a);
        Var n = norm2(h);
        Var f;
        if (mlp) {
            f = mlp->forward(n);
        } else if (gate) {
            f = down->forward(ops::mul(ops::silu(gate->forward(n)), up->forward(n)));
        } else {
            const Var u = fc->forward(n);
            f = proj->forward(act_expr ? ops::expr({u}, *act_expr) : ops::relu(u));
        }
        return combine(h, f, gate_m);
    }

    std::string norm_, act_, residual_;
    bool rope_ = false;
    float rope_theta_ = 10000.0f;
    std::shared_ptr<nn::LayerNorm> ln1, ln2;
    Var n1_w, n2_w;
    Var n1_b, n2_b;                         // norm "expr:<e>" only
    std::shared_ptr<const Expr> norm_expr;  // norm "expr:<e>", parsed once
    std::shared_ptr<nn::CausalSelfAttention> attn;
    std::shared_ptr<nn::SlidingWindowAttention> swa_attn;  // attention="swa"
    std::shared_ptr<nn::MLP> mlp;
    std::shared_ptr<nn::Linear> gate, up, down, fc, proj;
    std::shared_ptr<nn::Linear> gate_a, gate_m;  // highway only
    std::shared_ptr<const Expr> act_expr;        // act "expr:<e>", parsed once
};

class FlexLM : public nn::Module {
public:
    FlexLM(const FlexConfig& c, unsigned seed) : cfg(c) {
        wte = mod<nn::Embedding>("wte", c.vocab, c.d, seed + 1);
        if (c.pos == "learned") {
            wpe = mod<nn::Embedding>("wpe", c.n_ctx, c.d, seed + 2);
        } else if (c.pos != "rope") {  // sinusoidal: the Vaswani fixed table, not a parameter
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
        if (c.norm.rfind("expr:", 0) == 0) {
            norm_expr = std::make_shared<const Expr>(Expr::parse(c.norm.substr(5), 1));
            ln_f_b = reg("norm.bias", Matrix(1, c.d, 0.0f));
        }
        head = mod<nn::Linear>("head", c.d, c.vocab, false, seed + 99);
    }

    Var forward(const std::vector<int>& ids, size_t seq_len = 0) const {
        const size_t sl = seq_len == 0 ? ids.size() : seq_len;
        std::vector<int> pos(ids.size());
        for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i % sl);
        Var h;
        if (wpe) {
            h = ops::add(wte->forward(ids), wpe->forward(pos));
        } else if (cfg.pos == "rope") {  // positions enter in each block's attention
            h = wte->forward(ids);
        } else {
            Matrix pe(ids.size(), cfg.d);
            for (size_t r = 0; r < ids.size(); ++r)
                for (size_t j = 0; j < cfg.d; ++j) pe(r, j) = sin_table(pos[r], j);
            h = ops::add(wte->forward(ids), make_var(std::move(pe), false));
        }
        for (auto& b : blocks) h = b->forward(h, seq_len);
        Var n = ln_f_ln       ? ln_f_ln->forward(h)
                : norm_expr ? flex_expr_norm(h, *norm_expr, ln_f_w, ln_f_b)
                            : ops::rmsnorm(h, ln_f_w);
        return head->forward(n);
    }

    // GPT-2 export: only the GPT-2 flavour (the equivalence pin's config at
    // any depth and d_ff): layernorm, gelu, learned positions, residual,
    // exact or sliding-window attention (the window and sinks travel in
    // the IR). "" when exportable, else the reason.
    std::string gpt2_export_blocker() const {
        if (cfg.norm != "layernorm") return "norm " + cfg.norm + " (gpt2 export needs layernorm)";
        if (cfg.act != "gelu") return "activation " + cfg.act + " (gpt2 export needs gelu)";
        if (cfg.pos != "learned") return "position " + cfg.pos + " (gpt2 export needs learned)";
        if (cfg.residual != "residual") return "residual " + cfg.residual + " (gpt2 export needs residual)";
        if (cfg.attention != "exact" && cfg.attention != "swa")
            return "attention " + cfg.attention + " (gpt2 export needs exact or swa)";
        return "";
    }
    bool windowed() const { return cfg.attention == "swa"; }
    ir::json export_ir() const {
        if (!gpt2_export_blocker().empty()) throw std::runtime_error("FlexLM export_ir: " + gpt2_export_blocker());
        std::vector<ir::json> irs;
        for (const auto& b : blocks)
            irs.push_back(b->swa_attn ? nn::gpt2_block_ir(*b->ln1, *b->swa_attn, *b->ln2, *b->mlp, cfg.n_ctx)
                                      : nn::gpt2_block_ir(*b->ln1, *b->attn, *b->ln2, *b->mlp, cfg.n_ctx));
        return nn::gpt2_ir(*wte, *wpe, irs, *ln_f_ln, head.get());
    }
    std::map<std::string, Matrix> gpt2_state_dict() const {
        if (!gpt2_export_blocker().empty()) throw std::runtime_error("FlexLM gpt2_state_dict: " + gpt2_export_blocker());
        return gpt2_tensors();
    }

    // Every other flavour (microtorch/ir.hpp, Families: flex, flex_highway,
    // flex_plain); nullopt for the GPT-2 flavour, which exports as gpt2.
    std::optional<FamilyExport> family_export() const {
        if (gpt2_export_blocker().empty()) return std::nullopt;
        const auto& c = cfg;
        const std::string p = "blk.{i}.";
        const auto& b0 = *blocks[0];
        const bool rms = c.norm == "rmsnorm", plain = c.residual == "plain", highway = c.residual == "highway";
        const bool rope = c.pos == "rope";
        const size_t H = b0.attn ? b0.attn->H : b0.swa_attn->H, dk = c.d / H;
        ir::json n = ir::json::array();
        // The expr norm: its expr node, then the gain and bias as a linear
        // with weight diag(g) (FlexConfig; ir.hpp, Families: flex).
        auto expr_norm = [&](ir::json& nodes, const std::string& in, const std::string& out, const std::string& name) {
            nodes.push_back(ir::expr({in}, norm_expr->text(), out));
            nodes.push_back({{"op", "linear"},
                             {"in", ir::json::array({out})},
                             {"weight", name + "_gain.weight"},
                             {"bias", name + "_gain.bias"},
                             {"_note", "the expr norm's per-feature gain g and bias b: weight diag(g) [d, d]"},
                             {"out", out}});
        };
        auto norm = [&](const std::shared_ptr<nn::LayerNorm>& ln, const std::string& in, const std::string& out,
                        const std::string& name) {
            if (norm_expr)
                expr_norm(n, in, out, name);
            else
                n.push_back(rms ? ir::rmsnorm(in, name + ".weight", ops::kRmsNormEps, out)
                                : ln->export_ir(in, out, name));
        };
        // FlexBlock::combine of the stream x with the sublayer output f.
        auto combine = [&](const std::string& f, const std::shared_ptr<nn::Linear>& gate, const std::string& name) {
            if (highway) {
                n.push_back(gate->export_ir("x", "t", p + name + ".weight"));
                n.push_back(ir::unary("sigmoid", "t", "t"));
                n.push_back(ir::binary("sub", f, "x", f));
                n.push_back(ir::binary("mul", "t", f, f));
            }
            if (!plain) n.push_back(ir::binary("add", "x", f, "x"));
        };
        // plain: the sublayer's last node writes the stream itself.
        const std::string o = plain ? "x" : "o", f = plain ? "x" : "f";
        norm(b0.ln1, "x", "h", p + "attn_norm");
        auto attn = b0.attn ? b0.attn->export_ir("h", o, p + "attn_qkv.weight", p + "attn_output.weight", "kv")
                            : b0.swa_attn->export_ir("h", o, p + "attn_qkv.weight", p + "attn_output.weight", "kv");
        if (rope) {  // FlexBlock::attend: the rope node between split and attention, as LlamaBlock's
            ir::json with_rope = ir::json::array();
            for (const auto& node : attn) {
                if (node["op"] == "attention") with_rope.push_back(ir::rope("q", "k", "pos", H, dk, c.rope_theta, true));
                with_rope.push_back(node);
            }
            attn = with_rope;
        }
        for (const auto& node : attn) n.push_back(node);
        combine(o, b0.gate_a, "highway_attn");
        norm(b0.ln2, "x", "h2", p + "ffn_norm");
        if (b0.mlp) {
            for (const auto& node : b0.mlp->export_ir("h2", f, p + "ffn_up.weight", p + "ffn_down.weight"))
                n.push_back(node);
        } else if (b0.gate) {  // swiglu: down(silu(gate h2) * up h2)
            n.push_back(b0.gate->export_ir("h2", "g", p + "ffn_gate.weight"));
            n.push_back(ir::unary("silu", "g", "g"));
            n.push_back(b0.up->export_ir("h2", "u", p + "ffn_up.weight"));
            n.push_back(ir::binary("mul", "g", "u", "g"));
            n.push_back(b0.down->export_ir("g", f, p + "ffn_down.weight"));
        } else {  // relu, or the runtime-defined activation
            n.push_back(b0.fc->export_ir("h2", "u", p + "ffn_up.weight"));
            n.push_back(b0.act_expr ? ir::expr({"u"}, b0.act_expr->text(), "u")
                                    : ir::unary("relu", "u", "u"));
            n.push_back(b0.proj->export_ir("u", f, p + "ffn_down.weight"));
        }
        combine(f, b0.gate_m, "highway_ffn");
        const size_t window = b0.swa_attn ? c.window : 0, sinks = b0.swa_attn ? c.sinks : 0;
        const ir::json state = ir::json::array({ir::kv_state("kv", H, H, dk, c.n_ctx, window, sinks)});
        ir::json head_nodes = ir::json::array();
        if (norm_expr)
            expr_norm(head_nodes, "x", "h", "output_norm");
        else
            head_nodes.push_back(rms ? ir::rmsnorm("x", "output_norm.weight", ops::kRmsNormEps, "h")
                                     : ln_f_ln->export_ir("x", "h", "output_norm"));
        head_nodes.push_back(head->export_ir("h", "logits", "output.weight"));
        FamilyExport fx;
        fx.heads = H;
        fx.d_ff = c.d_ff;
        fx.ir = family_doc(c.residual == "residual" ? "flex" : "flex_" + c.residual, c.vocab, c.d, c.n_layers,
                           c.n_ctx, /*positions=*/true,
                           rope ? ir::json::array({wte->export_ir("tokens", "x", "token_embd.weight")})
                                : gpt2_embed_ir(*wte),
                           state, n, head_nodes);
        if (c.norm == "layernorm" && c.act == "gelu" && c.pos == "learned" && plain) {
            // llama.cpp's gpt2 tensors exactly, but no gpt2 graph computes
            // a block without its skip connections.
            fx.arch = "gpt2";
            fx.gpt2_sd = gpt2_tensors();
            fx.warning =
                "residual plain lives in the embedded IR only: an engine reading the gpt2 tensors without it "
                "(llama.cpp) adds skip connections this model does not have";
            return fx;
        }
        fx.arch = "paperkiln";
        using L = gguf::NamedTensor::Layout;
        put_tensor(fx.tensors, "token_embd.weight", wte->weight->data, L::ROWS);
        // Sinusoidal positions: the fixed table, as the model adds it. RoPE: no table.
        if (!rope) put_tensor(fx.tensors, "position_embd.weight", wpe ? wpe->weight->data : sin_table, L::ROWS);
        fx.constants = wpe || rope ? 0 : 1;
        auto put_expr_norm = [&](const std::string& name, const Var& g, const Var& b) {
            put_tensor(fx.tensors, name + "_gain.weight", flex_diag(g->data), L::PROJ);
            put_tensor(fx.tensors, name + "_gain.bias", b->data, L::VEC);
        };
        for (size_t b = 0; b < blocks.size(); ++b) {
            const std::string g = "blk." + std::to_string(b) + ".";
            const auto& k = *blocks[b];
            if (norm_expr)
                put_expr_norm(g + "attn_norm", k.n1_w, k.n1_b);
            else if (rms)
                put_tensor(fx.tensors, g + "attn_norm.weight", k.n1_w->data, L::VEC);
            else
                put_layernorm(fx.tensors, g + "attn_norm", *k.ln1);
            const nn::Linear& qkv = k.attn ? *k.attn->c_attn : *k.swa_attn->c_attn;
            const nn::Linear& out = k.attn ? *k.attn->c_proj : *k.swa_attn->c_proj;
            put_linear(fx.tensors, g + "attn_qkv", qkv);
            put_linear(fx.tensors, g + "attn_output", out);
            if (highway) put_linear(fx.tensors, g + "highway_attn", *k.gate_a);
            if (norm_expr)
                put_expr_norm(g + "ffn_norm", k.n2_w, k.n2_b);
            else if (rms)
                put_tensor(fx.tensors, g + "ffn_norm.weight", k.n2_w->data, L::VEC);
            else
                put_layernorm(fx.tensors, g + "ffn_norm", *k.ln2);
            if (k.mlp) {
                put_linear(fx.tensors, g + "ffn_up", *k.mlp->c_fc);
                put_linear(fx.tensors, g + "ffn_down", *k.mlp->c_proj);
            } else if (k.gate) {
                put_linear(fx.tensors, g + "ffn_gate", *k.gate);
                put_linear(fx.tensors, g + "ffn_up", *k.up);
                put_linear(fx.tensors, g + "ffn_down", *k.down);
            } else {
                put_linear(fx.tensors, g + "ffn_up", *k.fc);
                put_linear(fx.tensors, g + "ffn_down", *k.proj);
            }
            if (highway) put_linear(fx.tensors, g + "highway_ffn", *k.gate_m);
        }
        if (norm_expr)
            put_expr_norm("output_norm", ln_f_w, ln_f_b);
        else if (rms)
            put_tensor(fx.tensors, "output_norm.weight", ln_f_w->data, L::VEC);
        else
            put_layernorm(fx.tensors, "output_norm", *ln_f_ln);
        put_linear(fx.tensors, "output", *head);
        return fx;
    }

private:
    std::map<std::string, Matrix> gpt2_tensors() const {
        std::map<std::string, Matrix> sd;
        put_gpt2_shell(sd, *wte, *wpe, *ln_f_ln, *head);
        for (size_t b = 0; b < blocks.size(); ++b) {
            const auto& k = *blocks[b];
            if (k.swa_attn)
                put_gpt2_block(sd, b, *k.ln1, *k.swa_attn, *k.ln2, *k.mlp);
            else
                put_gpt2_block(sd, b, *k.ln1, *k.attn, *k.ln2, *k.mlp);
        }
        return sd;
    }

public:

    FlexConfig cfg;
    std::shared_ptr<nn::Embedding> wte, wpe;
    Matrix sin_table;
    std::vector<std::shared_ptr<FlexBlock>> blocks;
    std::shared_ptr<nn::LayerNorm> ln_f_ln;
    Var ln_f_w;
    Var ln_f_b;                             // norm "expr:<e>" only
    std::shared_ptr<const Expr> norm_expr;  // norm "expr:<e>", parsed once
    std::shared_ptr<nn::Linear> head;
};

// The Mamba-style SSM LM (mamba::MambaModel's stack, same names and seeds)
// as an mtstudio family: token embedding, no positions, mamba::MambaBlock x
// L (pre-LN, S4 scan with ops::ssm_scan, gelu output gates, skip), final
// LayerNorm, biased head. MambaModel::forward takes token ids as a float
// column; this one takes ids, and runs a stacked mini-batch one sequence at
// a time so no scan state crosses a sequence boundary.
class MambaLM : public nn::Module {
public:
    MambaLM(size_t vocab, size_t d, size_t n_layers, size_t d_state, unsigned seed) : d_state(d_state) {
        embed = mod<nn::Embedding>("embed", vocab, d);
        norm_final = mod<nn::LayerNorm>("norm_final", d);
        lm_head = mod<nn::Linear>("lm_head", d, vocab, true, seed + 21);
        for (size_t i = 0; i < n_layers; ++i)
            blocks.push_back(mod<mamba::MambaBlock>("block_" + std::to_string(i), d, d_state,
                                                    seed + 30 + static_cast<unsigned>(i)));
    }

    Var forward(const std::vector<int>& ids, size_t seq_len = 0) const {
        const size_t sl = seq_len == 0 ? ids.size() : seq_len;
        if (sl == 0 || ids.size() % sl != 0) throw std::runtime_error("MambaLM: rows not a multiple of seq_len");
        if (sl == ids.size()) return one(ids);
        Var out;
        for (size_t b = 0; b < ids.size() / sl; ++b) {
            std::vector<int> part(ids.begin() + b * sl, ids.begin() + (b + 1) * sl), rows(sl);
            for (size_t t = 0; t < sl; ++t) rows[t] = static_cast<int>(b * sl + t);
            Var placed = ops::scatter_add_rows(one(part), rows, ids.size());
            out = out ? ops::add(out, placed) : placed;
        }
        return out;
    }

    // microtorch/ir.hpp, Families: mamba (paperkiln).
    std::optional<FamilyExport> family_export(size_t ctx) const {
        const size_t vocab = embed->weight->data.rows(), d = embed->weight->data.cols(), n = d_state;
        const std::string p = "blk.{i}.";
        const auto& b0 = *blocks[0];
        const auto& s4 = *b0.s4;
        ir::json nd = ir::json::array();
        nd.push_back(b0.norm->export_ir("x", "h", p + "ssm_norm"));
        nd.push_back(s4.proj_in->export_ir("h", "u", p + "ssm_in.weight"));
        nd.push_back(ir::ssm_scan("u", "ssm", p, n, "y"));
        nd.push_back(s4.proj_out->export_ir("y", "y", p + "ssm_out.weight"));
        nd.push_back(s4.gate_proj->export_ir("h", "g", p + "ssm_gate.weight"));
        nd.push_back(ir::gelu("g", "tanh", "g"));
        nd.push_back(ir::binary("mul", "y", "g", "y"));
        nd.push_back(b0.out_gate->export_ir("x", "g", p + "ssm_out_gate.weight"));
        nd.push_back(ir::gelu("g", "tanh", "g"));
        nd.push_back(ir::binary("mul", "y", "g", "y"));
        nd.push_back(ir::binary("add", "x", "y", "x"));
        const ir::json embed_nodes = ir::json::array({embed->export_ir("tokens", "x", "token_embd.weight")});
        const ir::json head_nodes = ir::json::array(
            {norm_final->export_ir("x", "h", "output_norm"), lm_head->export_ir("h", "logits", "output.weight")});
        FamilyExport fx;
        fx.arch = "paperkiln";
        fx.ir = family_doc("mamba", vocab, d, blocks.size(), ctx, /*positions=*/false, embed_nodes,
                           ir::json::array({ir::ssm_state("ssm", n)}), nd, head_nodes);
        using L = gguf::NamedTensor::Layout;
        put_tensor(fx.tensors, "token_embd.weight", embed->weight->data, L::ROWS);
        for (size_t b = 0; b < blocks.size(); ++b) {
            const std::string g = "blk." + std::to_string(b) + ".";
            const auto& k = *blocks[b];
            put_layernorm(fx.tensors, g + "ssm_norm", *k.norm);
            put_linear(fx.tensors, g + "ssm_in", *k.s4->proj_in);
            put_tensor(fx.tensors, g + "ssm_a.weight", k.s4->A->data, L::ROWS);  // A[i][j] row-major
            put_tensor(fx.tensors, g + "ssm_b.weight", k.s4->B->data, L::VEC);   // [n, 1]
            put_tensor(fx.tensors, g + "ssm_c.weight", k.s4->C->data, L::VEC);   // [1, n]
            put_tensor(fx.tensors, g + "ssm_d.weight", k.s4->D->data, L::VEC);   // [1, 1]
            put_linear(fx.tensors, g + "ssm_out", *k.s4->proj_out);
            put_linear(fx.tensors, g + "ssm_gate", *k.s4->gate_proj);
            put_linear(fx.tensors, g + "ssm_out_gate", *k.out_gate);
        }
        put_layernorm(fx.tensors, "output_norm", *norm_final);
        put_linear(fx.tensors, "output", *lm_head);
        return fx;
    }

    size_t d_state;
    std::shared_ptr<nn::Embedding> embed;
    std::vector<std::shared_ptr<mamba::MambaBlock>> blocks;
    std::shared_ptr<nn::LayerNorm> norm_final;
    std::shared_ptr<nn::Linear> lm_head;

private:
    Var one(const std::vector<int>& ids) const {
        Var h = embed->forward(ids);
        for (const auto& b : blocks) h = b->forward(h);
        return lm_head->forward(norm_final->forward(h));
    }
};

}  // namespace parity
