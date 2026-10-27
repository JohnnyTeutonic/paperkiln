#pragma once
// Shared 4-lane parity model for the SRD experiments (srd_parity.cpp,
// srd_needle.cpp): 2 pre-LN blocks, learned positional embeddings,
// attention flavor pluggable. Kept identical across experiments so results
// compare across runs.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "microtorch/expr.hpp"
#include "microtorch/gguf.hpp"
#include "microtorch/ir.hpp"
#include "microtorch/mamba.hpp"
#include "microtorch/nn.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/recurrence.hpp"
#include "microtorch/srd.hpp"

namespace parity {

using namespace microtorch;

// RECURRENCE: kimi's projections (same tensors, names and seeds) around an IR recurrence node
// (IR_SPEC.md 6.10) over q, k, v [T, d] -> [T, d]; its body defaults to the linear_attention one
// (ir::linear_attention_body), with which the lane computes the KIMI lane's model bit for bit.
enum class AttnKind { EXACT, KIMI, SRD, SWA, RECURRENCE };

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

// A family's GGUF + IR export (microtorch/ir.hpp, Families): the tensor set is
// llama.cpp's gpt2 set (arch "gpt2": gpt2_sd, HF GPT-2 names, written by
// gguf::export_gguf_gpt2) or paperkiln's own (arch "paperkiln": tensors, for
// gguf::export_gguf_paperkiln). Either way the IR is the model, and the file's
// general.architecture is "paperkiln" (keramos IR_SPEC 3.1, rule 38: no
// llama.cpp graph computes these families).
struct FamilyExport {
    std::string arch;  // the tensor set: "gpt2" | "paperkiln"
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
    // recurrence_body (RECURRENCE only): the node's body (ir::recurrence's), null for the default.
    ParityLM(AttnKind kind, size_t vocab, size_t d, size_t n_heads, size_t n_ctx, unsigned seed,
             size_t swa_window = 64, size_t swa_sinks = 1, const ir::json* recurrence_body = nullptr)
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
                case AttnKind::RECURRENCE:
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
        if (kind == AttnKind::RECURRENCE) {
            rec_body = recurrence_body ? *recurrence_body : ir::linear_attention_body(n_heads, d / n_heads);
            rec_node = ir::recurrence({"q", "k", "v"}, "rec", rec_body, "a");
            rec = std::make_shared<Recurrence>(Recurrence::from_json(rec_node));
            if (rec->inputs().size() != 3 || rec->output().numel != d || !rec->weights().empty())
                throw std::runtime_error("attention recurrence: the body reads q, k and v and writes a row, each of "
                                         "width d = " + std::to_string(d) + ", with no weights of its own");
            for (const auto& t : rec->inputs())
                if (t.numel != d)
                    throw std::runtime_error("attention recurrence: input " + t.name + " has width " +
                                             std::to_string(t.numel) + ", not d = " + std::to_string(d));
        }
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
                case AttnKind::RECURRENCE: {
                    // kimi's projections around the recurrence: q, k, v the thirds of qkv.
                    const size_t d = wte->weight->data.cols();
                    Var qkv = kimi[b]->c_attn->forward(n1);
                    Var y = ops::recurrence(*rec, {ops::slice_cols(qkv, 0, d), ops::slice_cols(qkv, d, 2 * d),
                                                   ops::slice_cols(qkv, 2 * d, 3 * d)}, {}, seq_len);
                    a = kimi[b]->c_proj->forward(y);
                    break;
                }
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
        if (kind_ != AttnKind::KIMI && kind_ != AttnKind::SRD && kind_ != AttnKind::RECURRENCE) return std::nullopt;
        const size_t vocab = wte->weight->data.rows(), d = wte->weight->data.cols();
        const size_t ctx = wpe->weight->data.rows();
        const std::string p = "blk.{i}.";
        FamilyExport fx;
        fx.d_ff = 4 * d;
        ir::json n = ir::json::array(), state = ir::json::array();
        n.push_back(ln1[0]->export_ir("x", "h", p + "attn_norm"));
        if (kind_ == AttnKind::KIMI || kind_ == AttnKind::RECURRENCE) {
            const auto& a = *kimi[0];
            fx.heads = a.H;
            n.push_back(a.c_attn->export_ir("h", "qkv", p + "attn_qkv.weight"));
            n.push_back(ir::split("qkv", {d, d, d}, {"q", "k", "v"}));
            if (kind_ == AttnKind::KIMI) {
                n.push_back(ir::linear_attention("q", "k", "v", "lin", a.H, a.dk, "a"));
                state.push_back(ir::linear_attention_state("lin", a.H, a.dk));
            } else {
                n.push_back(rec_node);
                state.push_back(ir::recurrence_state("rec", rec_node));
            }
            n.push_back(a.c_proj->export_ir("a", "o", p + "attn_output.weight"));
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
        fx.ir = family_doc(kind_ == AttnKind::KIMI ? "kimi" : kind_ == AttnKind::SRD ? "srd" : "recurrence", vocab, d,
                           2, ctx, /*positions=*/true, gpt2_embed_ir(*wte), state, n, gpt2_head_ir(*ln_f, *head));
        if (kind_ == AttnKind::KIMI || kind_ == AttnKind::RECURRENCE) {
            fx.arch = "gpt2";
            put_gpt2_shell(fx.gpt2_sd, *wte, *wpe, *ln_f, *head);
            for (int b = 0; b < 2; ++b) put_gpt2_block(fx.gpt2_sd, b, *ln1[b], *kimi[b], *ln2[b], *mlp[b]);
            fx.warning = std::string(kind_ == AttnKind::KIMI ? "kimi linear attention" : "the recurrence") +
                         " lives in the embedded IR only: the file's architecture is paperkiln (gpt2's tensors), "
                         "which llama.cpp refuses rather than computing softmax attention";
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
    // RECURRENCE: the body, the node (in q, k, v; state "rec"; out "a") and its compiled form.
    ir::json rec_body, rec_node;
    std::shared_ptr<Recurrence> rec;
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
//     node, then a scale node with weight g [d] and bias b [d]
//     (blk.{i}.attn_norm_gain / ffn_norm_gain, output_norm_gain), which
//     computes (e(x) * g) + b in the training ops' order (mul_row, then
//     add_bias).
//
// Equivalence pin (tests/test_flex.cpp): with norm=layernorm, act=gelu,
// pos=learned, n_layers=2, d_ff=4d, FlexLM reproduces ParityLM(EXACT)
// bit-for-bit — same seed layout, same modules, same tape.
struct FlexConfig {
    size_t vocab = 0, d = 128, n_layers = 2, n_heads = 4, d_ff = 512, n_ctx = 128;
    std::string norm = "layernorm";  // layernorm | rmsnorm | expr:<e> (e over the row, row reductions allowed)
    std::string act = "gelu";        // gelu | relu | swiglu | expr:<e>
    std::string pos = "learned";     // learned | sinusoidal | rope | t5_relative
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
    // The inputs of an expr activation or norm, in the expression's order (x, then a, b, ...;
    // LOCAL_LLM_STACK.md section 11 item 4.2). Activation: "up" (the up-projection, exactly once)
    // and "gate" (a second projection d -> d_ff, at most once: a gated unit). Norm: "row" (the
    // row, exactly once). Either slot: "scalar" (one learned number, as DyT's alpha) or
    // "channel" (a learned vector over the slot's width: d_ff for the activation, d for the
    // norm), each starting at init. Empty = the one input (up, row). Each norm has its own
    // learned inputs (a block's norms, the head's); the activation's are per block.
    // A norm's learned input may start differently by where the norm sits (DyT on LLaMA: alpha 0.8 in
    // the attention sublayer's norms, 0.2 in the FFN's and the head's): init_attn / init_ffn /
    // init_final, NaN = init.
    struct ExprInput {
        std::string kind;
        float init = 1.0f;
        float init_attn = NAN, init_ffn = NAN, init_final = NAN;
        float init_at(const std::string& where) const {
            const float v = where == "attn" ? init_attn : where == "ffn" ? init_ffn : where == "final" ? init_final : NAN;
            return std::isnan(v) ? init : v;
        }
    };
    std::vector<ExprInput> act_inputs, norm_inputs;
    // An expr on the embed section's output before the first block (DyT on LLaMA: a learned scalar
    // times the token embedding): "" none, else "expr:<e>" over embed_inputs ("row" once, learned
    // scalars and channels of width d).
    std::string embed_expr;
    std::vector<ExprInput> embed_inputs;
    // Where each sublayer's norm n sits, f the sublayer: pre x + f(n(x)) (the default); post
    // n(x + f(x)) (the original Transformer); sandwich x + n'(f(n(x))) (a second norm on the
    // output, Gemma 2); branch_post x + n(f(x)) (Primer's FFN: the norm after the
    // transformation, inside the residual branch); none x + f(x).
    std::string place_attn = "pre", place_ffn = "pre";
    // The norm before the head: -1 = present unless both sublayers are post-norm (the stream
    // then leaves the last block normalised, as in the original Transformer); 0 absent; 1 present.
    int final_norm = -1;
    bool has_final_norm() const {
        return final_norm < 0 ? !(place_attn == "post" && place_ffn == "post") : final_norm == 1;
    }
    // position "t5_relative" (Raffel et al. 2020, section 2.1): no position table; every head adds
    // a learned bias, by bucketed query-key distance, to its attention scores
    // (ops::relpos_attention), from one table [rel_buckets, heads] shared by every layer. Trains and
    // exports: attention's position_bias {weight, buckets} (keramos IR_SPEC 5.15, revision 1.3), the
    // buckets ops::t5_position_buckets' table (FlexLM::family_export).
    size_t rel_buckets = 32, rel_max_distance = 128;
    // position "shaw_relative" (Shaw, Uszkoreit and Vaswani 2018, arXiv 1803.02155, eqs. 3-4): no position
    // table; each block learns relative key vectors (and, with rel_values, value vectors) for the clipped
    // query-key distance 0 .. rel_max_distance (Shaw's clipping distance k), shared by the heads or one set
    // per head (rel_per_head; Shaw's base model), added inside attention (ops::shaw_attention). Exported as
    // attention.relative_kv (keramos IR_SPEC 5.15, revision 1.4).
    bool rel_per_head = false, rel_values = true;
    // Primer's Multi-DConv-Head Attention (So et al. 2021, arXiv 2109.08668): a causal depthwise
    // convolution of width qkv_conv over time on every column of the fused q, k, v projection, before
    // RoPE and the heads (0 = none). No bias (Primer's tf.nn.depthwise_conv2d). Exported as a
    // conv1d_causal node over 3d channels with its recurrent slot (IR_SPEC 6.6).
    size_t qkv_conv = 0;
    // What is wrong with this configuration, or "" (mtstudio and the tests refuse with it).
    std::string problem() const;
};

inline bool flex_placement_ok(const std::string& p) {
    return p == "pre" || p == "post" || p == "sandwich" || p == "branch_post" || p == "none";
}
inline bool flex_pre_like(const std::string& p) { return p == "pre" || p == "sandwich"; }

inline std::string FlexConfig::problem() const {
    if (!flex_placement_ok(place_attn) || !flex_placement_ok(place_ffn))
        return "norm_placement must be pre, post, sandwich, branch_post or none (attention: " + place_attn +
               ", ffn: " + place_ffn + ")";
    auto check_inputs = [](const std::vector<ExprInput>& ins, const std::string& own, const std::string& slot,
                           bool gate_ok) -> std::string {
        size_t n_own = 0, n_gate = 0;
        for (const auto& i : ins) {
            if (i.kind == own) ++n_own;
            else if (i.kind == "gate" && gate_ok) ++n_gate;
            else if (i.kind != "scalar" && i.kind != "channel")
                return slot + "_inputs: unknown input \"" + i.kind + "\" (" + own + (gate_ok ? ", gate" : "") +
                       ", scalar, channel)";
            if (!std::isfinite(i.init)) return slot + "_inputs: an init is not finite";
        }
        if (ins.size() > Expr::kMaxInputs) return slot + "_inputs: more than 23 inputs";
        if (n_own != 1) return slot + "_inputs must name \"" + own + "\" exactly once";
        if (n_gate > 1) return slot + "_inputs: at most one gate";
        return "";
    };
    if (!act_inputs.empty()) {
        if (act.rfind("expr:", 0) != 0) return "activation_inputs need an expr activation";
        const std::string e = check_inputs(act_inputs, "up", "activation", true);
        if (!e.empty()) return e;
    }
    if (!norm_inputs.empty()) {
        if (norm.rfind("expr:", 0) != 0) return "norm_inputs need an expr norm";
        const std::string e = check_inputs(norm_inputs, "row", "norm", false);
        if (!e.empty()) return e;
    }
    if (!embed_expr.empty() && embed_expr.rfind("expr:", 0) != 0) return "embed must be expr:<e>";
    if (!embed_inputs.empty()) {
        if (embed_expr.empty()) return "embed_inputs need an embed expr";
        const std::string e = check_inputs(embed_inputs, "row", "embed", false);
        if (!e.empty()) return e;
    }
    if (pos == "t5_relative") {
        if (attention != "exact") return "position t5_relative needs exact attention";
        if (rel_buckets < 2 || rel_max_distance <= rel_buckets / 2)
            return "position t5_relative needs rel_buckets >= 2 and rel_max_distance > rel_buckets / 2";
    }
    if (pos == "shaw_relative" && attention != "exact") return "position shaw_relative needs exact attention";
    if (qkv_conv == 1) return "qkv_conv must be 0 (none) or a width >= 2";
    return "";
}

// A learned expr input as the expression's [T, w] operand: 1 * p, p a [1, 1] scalar or a [1, w]
// vector broadcast over the rows (exact; the IR's scale node over a row of ones computes the same).
inline Var flex_param_row(const Var& p, size_t T, size_t w) {
    const Var r = p->data.cols() == 1 && w != 1 ? ops::broadcast_cols(p, w) : p;
    return ops::mul_row(make_var(Matrix(T, w, 1.0f)), r);
}

// The expr's operands in its input order: `own` (the row, the up-projection) and `gate` where the
// inputs name them, each learned input as flex_param_row. No inputs listed = {own}.
inline std::vector<Var> flex_expr_args(const std::vector<FlexConfig::ExprInput>& ins, const Var& own,
                                       const Var& gate, const std::vector<Var>& params) {
    if (ins.empty()) return {own};
    std::vector<Var> args;
    size_t k = 0;
    const size_t T = own->data.rows(), w = own->data.cols();
    for (const auto& i : ins) {
        if (i.kind == "up" || i.kind == "row") args.push_back(own);
        else if (i.kind == "gate") args.push_back(gate);
        else args.push_back(flex_param_row(params.at(k++), T, w));
    }
    return args;
}

// One norm of a flex block or head: layernorm (ln), rmsnorm (w), or the expr norm
// e(row, learned inputs) * w + b (the gain w and bias b per feature, then the expr's own learned
// inputs in its order). Unused = none of them set.
struct FlexNorm {
    std::shared_ptr<nn::LayerNorm> ln;
    Var w, b;
    std::vector<Var> params;
    bool on() const { return ln || w; }
};

// The expr norm: e(x) * g + b (FlexConfig, norm "expr:<e>").
inline Var flex_expr_norm(const Var& x, const Expr& e, const Var& g, const Var& b) {
    return ops::add_bias(ops::mul_row(ops::expr({x}, e), g), b);
}

inline Var flex_norm_forward(const FlexNorm& n, const Var& x, const Expr* e,
                             const std::vector<FlexConfig::ExprInput>& ins) {
    if (n.ln) return n.ln->forward(x);
    if (!e) return ops::rmsnorm(x, n.w);
    if (ins.empty()) return flex_expr_norm(x, *e, n.w, n.b);
    return ops::add_bias(ops::mul_row(ops::expr(flex_expr_args(ins, x, nullptr, n.params), *e), n.w), n.b);
}

// The expr's input letters (x, a, b, ...), for parameter and tensor names.
inline std::string flex_input_letter(size_t i) { return i == 0 ? "x" : std::string(1, char('a' + i - 1)); }

// T5's relative position bias for FlexBlock::attend: the shared table and the bucket of each distance.
struct FlexRelPos {
    Var table;
    const std::vector<int>* buckets = nullptr;
};

class FlexBlock : public nn::Module {
public:
    FlexBlock(const FlexConfig& c, unsigned s)
        : norm_(c.norm), act_(c.act), residual_(c.residual), place_attn_(c.place_attn), place_ffn_(c.place_ffn),
          rope_(c.pos == "rope"), rope_theta_(c.rope_theta), act_in_(c.act_inputs), norm_in_(c.norm_inputs) {
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
        // Norm slots: 0 the attention sublayer's, 1 the FFN's, 2 and 3 their sandwich output norms.
        // Registered in that order (the default pre-norm block registers exactly as before).
        const bool on[4] = {c.place_attn != "none", c.place_ffn != "none", c.place_attn == "sandwich",
                            c.place_ffn == "sandwich"};
        static const char* ln_names[4] = {"ln_1", "ln_2", "ln_1_out", "ln_2_out"};
        static const char* n_names[4] = {"norm1", "norm2", "norm1_out", "norm2_out"};
        if (norm_ == "layernorm") {
            for (int i = 0; i < 4; ++i)
                if (on[i]) ns[i].ln = mod<nn::LayerNorm>(ln_names[i], c.d);
        } else {
            for (int i = 0; i < 4; ++i)
                if (on[i]) ns[i].w = reg(std::string(n_names[i]) + ".weight", Matrix(1, c.d, 1.0f));
            if (norm_.rfind("expr:", 0) == 0) {  // e(x, ...) * g + b: the gain above, a bias here
                norm_expr = std::make_shared<const Expr>(
                    Expr::parse(norm_.substr(5), std::max<size_t>(1, norm_in_.size())));
                for (int i = 0; i < 4; ++i)
                    if (on[i]) ns[i].b = reg(std::string(n_names[i]) + ".bias", Matrix(1, c.d, 0.0f));
                for (int i = 0; i < 4; ++i)
                    for (size_t j = 0; on[i] && j < norm_in_.size(); ++j)
                        if (norm_in_[j].kind != "row")
                            ns[i].params.push_back(reg(std::string(n_names[i]) + ".p_" + flex_input_letter(j),
                                                       Matrix(1, norm_in_[j].kind == "scalar" ? 1 : c.d,
                                                              norm_in_[j].init_at(i % 2 == 0 ? "attn" : "ffn"))));
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
                act_expr = std::make_shared<const Expr>(
                    Expr::parse(act_.substr(5), std::max<size_t>(1, act_in_.size())));
            fc = mod<nn::Linear>("mlp.c_fc", c.d, c.d_ff, true, s + 3 + 17);
            proj = mod<nn::Linear>("mlp.c_proj", c.d_ff, c.d, true, s + 3 + 19);
            for (size_t j = 0; j < act_in_.size(); ++j) {
                if (act_in_[j].kind == "gate")
                    fc_gate = mod<nn::Linear>("mlp.c_gate", c.d, c.d_ff, true, s + 3 + 18);
                else if (act_in_[j].kind != "up")
                    act_params.push_back(reg("mlp.p_" + flex_input_letter(j),
                                             Matrix(1, act_in_[j].kind == "scalar" ? 1 : c.d_ff, act_in_[j].init)));
            }
        }
        // Registered after everything else, from seeds of their own: a block without them is unchanged.
        if (c.qkv_conv) {
            // MDHA's depthwise kernel [3d, K] (q's columns, then k's, then v's): torch Conv1d's default at
            // fan-in K, uniform in +-1/sqrt(K), as Mamba1Block's conv (Primer states no initialisation).
            const size_t K = c.qkv_conv;
            std::mt19937 g(s + 300);
            const float cb = 1.0f / std::sqrt(static_cast<float>(K));
            std::uniform_real_distribution<float> uc(-cb, cb);
            Matrix w(3 * c.d, K);
            for (size_t i = 0; i < w.rows(); ++i)
                for (size_t k = 0; k < K; ++k) w(i, k) = uc(g);
            qkv_conv_w = reg("attn.qkv_conv.weight", std::move(w));
        }
        if (c.pos == "shaw_relative") {
            // Shaw's relative key / value vectors [K + 1, W] (row n: distance n), W = head_dim shared by the
            // heads or d (one set per head); small normal values, seeded (the paper states no
            // initialisation).
            shaw_k_ = c.rel_max_distance;
            const size_t W = c.rel_per_head ? c.d : c.d / c.n_heads;
            auto table = [&](unsigned seed) {
                Matrix t(c.rel_max_distance + 1, W);
                std::mt19937 g(seed);
                std::normal_distribution<float> nd(0.0f, 0.02f);
                for (size_t i = 0; i < t.rows(); ++i)
                    for (size_t j = 0; j < t.cols(); ++j) t(i, j) = nd(g);
                return t;
            };
            rel_k = reg("attn.rel_k.weight", table(s + 301));
            if (c.rel_values) rel_v = reg("attn.rel_v.weight", table(s + 302));
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
    // per-head attention, as LlamaBlock::forward places it. With T5's
    // relative bias, ops::relpos_attention per head on the same projections. With Shaw's relative
    // vectors, ops::shaw_attention per head. With MDHA, the depthwise causal convolution on the fused
    // projection first (before RoPE), each sequence of seq_len rows from zero history.
    Var attend(const Var& n, size_t seq_len, const FlexRelPos* rel = nullptr) const {
        if (!rope_ && !rel && !rel_k && !qkv_conv_w)
            return attn ? attn->forward(n, seq_len) : swa_attn->forward(n, seq_len);
        const nn::Linear& c_attn = attn ? *attn->c_attn : *swa_attn->c_attn;
        const nn::Linear& c_proj = attn ? *attn->c_proj : *swa_attn->c_proj;
        const size_t H = attn ? attn->H : swa_attn->H, dk = attn ? attn->dk : swa_attn->dk, d = H * dk;
        const size_t rows = n->data.rows(), sl = seq_len == 0 ? rows : seq_len;
        Var qkv = c_attn.forward(n);
        if (qkv_conv_w) qkv = ops::conv1d_causal(qkv, qkv_conv_w, nullptr, seq_len);
        if (rope_) {
            std::vector<int> pos(rows);
            for (size_t i = 0; i < rows; ++i) pos[i] = static_cast<int>(i % sl);
            qkv = ops::apply_rope(qkv, pos, rope_theta_, dk, /*all_heads=*/true);
        }
        const float sc = 1.0f / std::sqrt(static_cast<float>(dk));
        std::vector<Var> heads;
        for (size_t h = 0; h < H; ++h) {
            Var q = ops::slice_cols(qkv, h * dk, (h + 1) * dk);
            Var k = ops::slice_cols(qkv, d + h * dk, d + (h + 1) * dk);
            Var v = ops::slice_cols(qkv, 2 * d + h * dk, 2 * d + (h + 1) * dk);
            if (rel)
                heads.push_back(ops::relpos_attention(q, k, v, rel->table, h, *rel->buckets, sc, seq_len));
            else if (rel_k)
                heads.push_back(ops::shaw_attention(q, k, v, rel_k, rel_v, rel_k->data.cols() == dk ? 0 : h * dk,
                                                    shaw_k_, sc, seq_len));
            else
                heads.push_back(attn ? ops::fused_attention(q, k, v, sc, seq_len, attn->causal)
                                     : ops::swa_attention(q, k, v, sc, swa_attn->window, swa_attn->sinks, seq_len));
        }
        return c_proj.forward(ops::concat_cols(heads));
    }

    Var norm(int slot, const Var& v) const { return flex_norm_forward(ns[slot], v, norm_expr.get(), norm_in_); }

    // The FFN's activation over its up-projection u (and the gate's projection of n).
    Var activate(const Var& n, const Var& u) const {
        if (!act_expr) return ops::relu(u);
        if (act_in_.empty()) return ops::expr({u}, *act_expr);
        const Var g = fc_gate ? fc_gate->forward(n) : nullptr;
        return ops::expr(flex_expr_args(act_in_, u, g, act_params), *act_expr);
    }

    Var forward(const Var& x, size_t seq_len = 0, const FlexRelPos* rel = nullptr) const {
        Var a_out = attend(flex_pre_like(place_attn_) ? norm(0, x) : x, seq_len, rel);
        if (place_attn_ == "sandwich") a_out = norm(2, a_out);
        if (place_attn_ == "branch_post") a_out = norm(0, a_out);
        Var h = combine(x, a_out, gate_a);
        if (place_attn_ == "post") h = norm(0, h);
        const Var n = flex_pre_like(place_ffn_) ? norm(1, h) : h;
        Var f;
        if (mlp) {
            f = mlp->forward(n);
        } else if (gate) {
            f = down->forward(ops::mul(ops::silu(gate->forward(n)), up->forward(n)));
        } else {
            f = proj->forward(activate(n, fc->forward(n)));
        }
        if (place_ffn_ == "sandwich") f = norm(3, f);
        if (place_ffn_ == "branch_post") f = norm(1, f);
        Var y = combine(h, f, gate_m);
        if (place_ffn_ == "post") y = norm(1, y);
        return y;
    }

    std::string norm_, act_, residual_, place_attn_, place_ffn_;
    bool rope_ = false;
    float rope_theta_ = 10000.0f;
    std::vector<FlexConfig::ExprInput> act_in_, norm_in_;
    FlexNorm ns[4];                         // the attention's, the FFN's, their sandwich output norms
    std::shared_ptr<const Expr> norm_expr;  // norm "expr:<e>", parsed once
    std::shared_ptr<nn::CausalSelfAttention> attn;
    std::shared_ptr<nn::SlidingWindowAttention> swa_attn;  // attention="swa"
    std::shared_ptr<nn::MLP> mlp;
    std::shared_ptr<nn::Linear> gate, up, down, fc, proj;
    std::shared_ptr<nn::Linear> fc_gate;         // activation_inputs with a "gate"
    std::vector<Var> act_params;                 // activation_inputs' learned scalars and channels
    std::shared_ptr<nn::Linear> gate_a, gate_m;  // highway only
    Var qkv_conv_w;                              // qkv_conv (MDHA): [3d, K]
    Var rel_k, rel_v;                            // shaw_relative: [K + 1, W] each (rel_v null: keys only)
    size_t shaw_k_ = 0;                          // its clipping distance K
    std::shared_ptr<const Expr> act_expr;        // act "expr:<e>", parsed once
};

class FlexLM : public nn::Module {
public:
    FlexLM(const FlexConfig& c, unsigned seed) : cfg(c) {
        const std::string bad = c.problem();
        if (!bad.empty()) throw std::runtime_error("flex: " + bad);
        wte = mod<nn::Embedding>("wte", c.vocab, c.d, seed + 1);
        if (c.pos == "learned") {
            wpe = mod<nn::Embedding>("wpe", c.n_ctx, c.d, seed + 2);
        } else if (c.pos == "sinusoidal") {  // the Vaswani fixed table, not a parameter
            sin_table = Matrix(c.n_ctx, c.d);
            for (size_t p = 0; p < c.n_ctx; ++p)
                for (size_t i = 0; i < c.d; ++i) {
                    const double angle = p / std::pow(10000.0, double(2 * (i / 2)) / double(c.d));
                    sin_table(p, i) =
                        static_cast<float>(i % 2 == 0 ? std::sin(angle) : std::cos(angle));
                }
        } else if (c.pos != "rope" && c.pos != "t5_relative" && c.pos != "shaw_relative") {
            throw std::runtime_error("flex: unknown position " + c.pos +
                                     " (learned | sinusoidal | rope | t5_relative | shaw_relative)");
        }
        for (size_t b = 0; b < c.n_layers; ++b)
            blocks.push_back(mod<FlexBlock>("layers." + std::to_string(b), c,
                                            seed + 10 * static_cast<unsigned>(b + 1)));
        if (c.has_final_norm()) {
            if (c.norm == "layernorm") {
                ln_f_ln = mod<nn::LayerNorm>("ln_f", c.d);
                fn.ln = ln_f_ln;
            } else {
                ln_f_w = reg("norm.weight", Matrix(1, c.d, 1.0f));
                fn.w = ln_f_w;
            }
        }
        if (c.norm.rfind("expr:", 0) == 0) {
            norm_expr = std::make_shared<const Expr>(Expr::parse(c.norm.substr(5), std::max<size_t>(1, c.norm_inputs.size())));
            if (c.has_final_norm()) {
                ln_f_b = reg("norm.bias", Matrix(1, c.d, 0.0f));
                fn.b = ln_f_b;
                for (size_t j = 0; j < c.norm_inputs.size(); ++j)
                    if (c.norm_inputs[j].kind != "row")
                        fn.params.push_back(reg("norm.p_" + flex_input_letter(j),
                                                Matrix(1, c.norm_inputs[j].kind == "scalar" ? 1 : c.d,
                                                       c.norm_inputs[j].init_at("final"))));
            }
        }
        head = mod<nn::Linear>("head", c.d, c.vocab, false, seed + 99);
        if (!c.embed_expr.empty()) {
            embed_expr = std::make_shared<const Expr>(
                Expr::parse(c.embed_expr.substr(5), std::max<size_t>(1, c.embed_inputs.size())));
            for (size_t j = 0; j < c.embed_inputs.size(); ++j)
                if (c.embed_inputs[j].kind != "row")
                    embed_params.push_back(reg("embed.p_" + flex_input_letter(j),
                                               Matrix(1, c.embed_inputs[j].kind == "scalar" ? 1 : c.d,
                                                      c.embed_inputs[j].init)));
        }
        if (c.pos == "t5_relative") {
            // T5 initialises its bias table like an embedding; small normal values, seeded.
            Matrix t(c.rel_buckets, c.n_heads);
            std::mt19937 g(seed + 7);
            std::normal_distribution<float> nd(0.0f, 0.02f);
            for (size_t i = 0; i < t.rows(); ++i)
                for (size_t j = 0; j < t.cols(); ++j) t(i, j) = nd(g);
            rel_bias = reg("rel_bias.weight", std::move(t));
            rel_buckets = ops::t5_position_buckets(c.rel_buckets, c.rel_max_distance);
        }
    }

    Var forward(const std::vector<int>& ids, size_t seq_len = 0) const {
        const size_t sl = seq_len == 0 ? ids.size() : seq_len;
        std::vector<int> pos(ids.size());
        for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i % sl);
        Var h;
        if (wpe) {
            h = ops::add(wte->forward(ids), wpe->forward(pos));
        } else if (cfg.pos == "rope" || cfg.pos == "t5_relative" ||
                   cfg.pos == "shaw_relative") {  // positions enter in each block's attention
            h = wte->forward(ids);
        } else {
            Matrix pe(ids.size(), cfg.d);
            for (size_t r = 0; r < ids.size(); ++r)
                for (size_t j = 0; j < cfg.d; ++j) pe(r, j) = sin_table(pos[r], j);
            h = ops::add(wte->forward(ids), make_var(std::move(pe), false));
        }
        if (embed_expr) h = ops::expr(flex_expr_args(cfg.embed_inputs, h, nullptr, embed_params), *embed_expr);
        FlexRelPos rel{rel_bias, &rel_buckets};
        for (auto& b : blocks) h = b->forward(h, seq_len, rel_bias ? &rel : nullptr);
        const Var n = fn.on() ? flex_norm_forward(fn, h, norm_expr.get(), cfg.norm_inputs) : h;
        return head->forward(n);
    }

    // GPT-2 export: only the GPT-2 flavour (the equivalence pin's config at
    // any depth and d_ff): layernorm, gelu, learned positions, residual,
    // exact or sliding-window attention (the window and sinks travel in
    // the IR), pre-norm with the final norm. "" when exportable, else the reason.
    std::string gpt2_export_blocker() const {
        if (cfg.norm != "layernorm") return "norm " + cfg.norm + " (gpt2 export needs layernorm)";
        if (cfg.act != "gelu") return "activation " + cfg.act + " (gpt2 export needs gelu)";
        if (cfg.pos != "learned") return "position " + cfg.pos + " (gpt2 export needs learned)";
        if (cfg.residual != "residual") return "residual " + cfg.residual + " (gpt2 export needs residual)";
        if (cfg.attention != "exact" && cfg.attention != "swa")
            return "attention " + cfg.attention + " (gpt2 export needs exact or swa)";
        if (!cfg.embed_expr.empty()) return "embed " + cfg.embed_expr + " (gpt2 export has no embed expr)";
        if (cfg.place_attn != "pre" || cfg.place_ffn != "pre" || !cfg.has_final_norm())
            return "norm placement " + cfg.place_attn + "/" + cfg.place_ffn + " (gpt2 export needs pre-norm)";
        if (cfg.qkv_conv) return "qkv_conv (gpt2 export has no convolution)";
        return "";
    }
    bool windowed() const { return cfg.attention == "swa"; }
    ir::json export_ir() const {
        if (!gpt2_export_blocker().empty()) throw std::runtime_error("FlexLM export_ir: " + gpt2_export_blocker());
        std::vector<ir::json> irs;
        for (const auto& b : blocks)
            irs.push_back(b->swa_attn ? nn::gpt2_block_ir(*b->ns[0].ln, *b->swa_attn, *b->ns[1].ln, *b->mlp, cfg.n_ctx)
                                      : nn::gpt2_block_ir(*b->ns[0].ln, *b->attn, *b->ns[1].ln, *b->mlp, cfg.n_ctx));
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
        // T5's bias: the attention attribute position_bias {weight, buckets} (keramos IR_SPEC 5.15, revision
        // 1.3), the one table rel_pos_bias.weight [rel_buckets, heads] read by every layer.
        const bool t5 = c.pos == "t5_relative";
        // Shaw's relative vectors: the attention attribute relative_kv {key, value, max_distance} (IR_SPEC
        // 5.15, revision 1.4), each block's own blk.{i}.attn_rel_k / attn_rel_v [K + 1, W].
        const bool shaw = c.pos == "shaw_relative";
        // MDHA: a conv1d_causal node on the fused projection before the split, over the recurrent slot
        // "qkv_conv" (its K - 1 previous rows), weight blk.{i}.attn_qkv_conv.weight [3d, K], no bias.
        const bool mdha = c.qkv_conv > 0;
        const std::string p = "blk.{i}.";
        const auto& b0 = *blocks[0];
        const bool plain = c.residual == "plain", highway = c.residual == "highway";
        const bool rope = c.pos == "rope";
        const size_t H = b0.attn ? b0.attn->H : b0.swa_attn->H, dk = c.d / H;
        ir::json n = ir::json::array();
        // A norm (slot n of a block, or the head's): layernorm, rmsnorm, or the expr norm: its learned
        // inputs as rows (a ones row from the expr "1", then a scale node per input over it), the expr
        // node, then the gain g and bias b [d] as a scale node (FlexConfig; ir.hpp, Families: flex).
        auto emit_norm = [&](ir::json& nodes, const FlexNorm& nm, const std::string& in, const std::string& out,
                             const std::string& name) {
            if (nm.ln) {
                nodes.push_back(nm.ln->export_ir(in, out, name));
                return;
            }
            if (!norm_expr) {
                nodes.push_back(ir::rmsnorm(in, name + ".weight", ops::kRmsNormEps, out));
                return;
            }
            std::vector<std::string> ins;
            if (c.norm_inputs.empty()) {
                ins = {in};
            } else {
                bool ones = false;
                for (size_t j = 0; j < c.norm_inputs.size(); ++j) {
                    if (c.norm_inputs[j].kind == "row") {
                        ins.push_back(in);
                        continue;
                    }
                    if (!ones) nodes.push_back(ir::expr({in}, "1", "one"));
                    ones = true;
                    const std::string v = "p_" + flex_input_letter(j);
                    nodes.push_back(ir::scale("one", name + "_p_" + flex_input_letter(j) + ".weight", "", v));
                    ins.push_back(v);
                }
            }
            nodes.push_back(ir::expr(ins, norm_expr->text(), out));
            nodes.push_back(ir::scale(out, name + "_gain.weight", name + "_gain.bias", out));
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
        const bool pre_a = flex_pre_like(c.place_attn), pre_f = flex_pre_like(c.place_ffn);
        // A norm on the sublayer's output (sandwich, branch_post) runs on o / f before the combine;
        // with plain residuals o and f are the stream itself, which nothing reads again.
        if (pre_a) emit_norm(n, b0.ns[0], "x", "h", p + "attn_norm");
        auto attn = b0.attn ? b0.attn->export_ir(pre_a ? "h" : "x", o, p + "attn_qkv.weight", p + "attn_output.weight", "kv")
                            : b0.swa_attn->export_ir(pre_a ? "h" : "x", o, p + "attn_qkv.weight", p + "attn_output.weight", "kv");
        if (rope || t5 || shaw || mdha) {  // FlexBlock::attend: the rope node between split and attention, as LlamaBlock's
            ir::json edited = ir::json::array();
            for (auto node : attn) {
                if (mdha && node["op"] == "split")
                    edited.push_back({{"op", "conv1d_causal"}, {"in", ir::json::array({"qkv"})}, {"state", "qkv_conv"},
                                      {"weight", p + "attn_qkv_conv.weight"}, {"bias", nullptr}, {"channels", 3 * c.d},
                                      {"kernel", c.qkv_conv}, {"out", "qkv"}});
                if (rope && node["op"] == "attention") edited.push_back(ir::rope("q", "k", "pos", H, dk, c.rope_theta, true));
                if (t5 && node["op"] == "attention")
                    node["position_bias"] = {{"weight", "rel_pos_bias.weight"}, {"buckets", rel_buckets}};
                if (shaw && node["op"] == "attention")
                    node["relative_kv"] = {{"key", p + "attn_rel_k.weight"},
                                           {"value", c.rel_values ? ir::json(p + "attn_rel_v.weight") : ir::json(nullptr)},
                                           {"max_distance", c.rel_max_distance}};
                edited.push_back(node);
            }
            attn = edited;
        }
        for (const auto& node : attn) n.push_back(node);
        if (c.place_attn == "sandwich") emit_norm(n, b0.ns[2], o, o, p + "attn_post_norm");
        if (c.place_attn == "branch_post") emit_norm(n, b0.ns[0], o, o, p + "attn_norm");
        combine(o, b0.gate_a, "highway_attn");
        if (c.place_attn == "post") emit_norm(n, b0.ns[0], "x", "x", p + "attn_norm");
        if (pre_f) emit_norm(n, b0.ns[1], "x", "h2", p + "ffn_norm");
        const std::string fin = pre_f ? "h2" : "x";
        if (b0.mlp) {
            for (const auto& node : b0.mlp->export_ir(fin, f, p + "ffn_up.weight", p + "ffn_down.weight"))
                n.push_back(node);
        } else if (b0.gate) {  // swiglu: down(silu(gate h2) * up h2)
            n.push_back(b0.gate->export_ir(fin, "g", p + "ffn_gate.weight"));
            n.push_back(ir::unary("silu", "g", "g"));
            n.push_back(b0.up->export_ir(fin, "u", p + "ffn_up.weight"));
            n.push_back(ir::binary("mul", "g", "u", "g"));
            n.push_back(b0.down->export_ir("g", f, p + "ffn_down.weight"));
        } else {  // relu, or the runtime-defined activation (its gate and learned inputs first)
            n.push_back(b0.fc->export_ir(fin, "u", p + "ffn_up.weight"));
            if (!b0.act_expr) {
                n.push_back(ir::unary("relu", "u", "u"));
            } else if (c.act_inputs.empty()) {
                n.push_back(ir::expr({"u"}, b0.act_expr->text(), "u"));
            } else {
                std::vector<std::string> ins;
                bool ones = false;
                for (size_t j = 0; j < c.act_inputs.size(); ++j) {
                    const auto& k = c.act_inputs[j].kind;
                    if (k == "up") {
                        ins.push_back("u");
                    } else if (k == "gate") {
                        n.push_back(b0.fc_gate->export_ir(fin, "g", p + "ffn_gate.weight"));
                        ins.push_back("g");
                    } else {
                        if (!ones) n.push_back(ir::expr({"u"}, "1", "one"));
                        ones = true;
                        const std::string v = "p_" + flex_input_letter(j);
                        n.push_back(ir::scale("one", p + "ffn_act_p_" + flex_input_letter(j) + ".weight", "", v));
                        ins.push_back(v);
                    }
                }
                n.push_back(ir::expr(ins, b0.act_expr->text(), "u"));
            }
            n.push_back(b0.proj->export_ir("u", f, p + "ffn_down.weight"));
        }
        if (c.place_ffn == "sandwich") emit_norm(n, b0.ns[3], f, f, p + "ffn_post_norm");
        if (c.place_ffn == "branch_post") emit_norm(n, b0.ns[1], f, f, p + "ffn_norm");
        combine(f, b0.gate_m, "highway_ffn");
        if (c.place_ffn == "post") emit_norm(n, b0.ns[1], "x", "x", p + "ffn_norm");
        const size_t window = b0.swa_attn ? c.window : 0, sinks = b0.swa_attn ? c.sinks : 0;
        ir::json state = ir::json::array({ir::kv_state("kv", H, H, dk, c.n_ctx, window, sinks)});
        if (mdha)
            state.push_back({{"name", "qkv_conv"}, {"kind", "recurrent"}, {"op", "conv1d_causal"}, {"dtype", "f32"},
                             {"tensors", ir::json::array({{{"name", "x"},
                                                           {"shape", ir::json::array({c.qkv_conv - 1, 3 * c.d})},
                                                           {"init", 0}}})}});
        ir::json head_nodes = ir::json::array();
        if (fn.on())
            emit_norm(head_nodes, fn, "x", "h", "output_norm");
        head_nodes.push_back(head->export_ir(fn.on() ? "h" : "x", "logits", "output.weight"));
        FamilyExport fx;
        fx.heads = H;
        fx.d_ff = c.d_ff;
        ir::json embed_nodes = rope || t5 || shaw ? ir::json::array({wte->export_ir("tokens", "x", "token_embd.weight")})
                                                  : gpt2_embed_ir(*wte);
        if (embed_expr) {  // the expr on the embed output: its learned inputs as scale nodes over a ones row
            std::vector<std::string> ins;
            bool ones = false;
            for (size_t j = 0; j < std::max<size_t>(1, c.embed_inputs.size()); ++j) {
                if (c.embed_inputs.empty() || c.embed_inputs[j].kind == "row") {
                    ins.push_back("x");
                    continue;
                }
                if (!ones) embed_nodes.push_back(ir::expr({"x"}, "1", "one"));
                ones = true;
                const std::string v = "p_" + flex_input_letter(j);
                embed_nodes.push_back(ir::scale("one", "embd_p_" + flex_input_letter(j) + ".weight", "", v));
                ins.push_back(v);
            }
            embed_nodes.push_back(ir::expr(ins, embed_expr->text(), "x"));
        }
        fx.ir = family_doc(c.residual == "residual" ? "flex" : "flex_" + c.residual, c.vocab, c.d, c.n_layers,
                           c.n_ctx, /*positions=*/true, embed_nodes, state, n, head_nodes);
        if (c.norm == "layernorm" && c.act == "gelu" && c.pos == "learned" && plain && c.place_attn == "pre" &&
            c.place_ffn == "pre" && c.has_final_norm() && !mdha) {
            // llama.cpp's gpt2 tensors exactly, but no gpt2 graph computes
            // a block without its skip connections.
            fx.arch = "gpt2";
            fx.gpt2_sd = gpt2_tensors();
            fx.warning =
                "residual plain lives in the embedded IR only: the file's architecture is paperkiln (gpt2's "
                "tensors), which llama.cpp refuses rather than adding skip connections this model lacks";
            return fx;
        }
        fx.arch = "paperkiln";
        using L = gguf::NamedTensor::Layout;
        put_tensor(fx.tensors, "token_embd.weight", wte->weight->data, L::ROWS);
        // Sinusoidal positions: the fixed table, as the model adds it. RoPE, T5's bias and Shaw's vectors: no table.
        if (!rope && !t5 && !shaw)
            put_tensor(fx.tensors, "position_embd.weight", wpe ? wpe->weight->data : sin_table, L::ROWS);
        fx.constants = wpe || rope || t5 || shaw ? 0 : 1;
        // A learned expr input [1, 1] or [1, w] as its [w] tensor (a scalar repeated: the IR's scale
        // over a row of ones then gives 1 * p at every entry).
        auto put_param = [&](const std::string& name, const Var& v, size_t w) {
            Matrix m(1, w);
            for (size_t j = 0; j < w; ++j) m(0, j) = v->data(0, v->data.cols() == 1 ? 0 : j);
            put_tensor(fx.tensors, name, m, L::VEC);
        };
        auto put_norm = [&](const std::string& name, const FlexNorm& nm) {
            if (nm.ln) {
                put_layernorm(fx.tensors, name, *nm.ln);
            } else if (!norm_expr) {
                put_tensor(fx.tensors, name + ".weight", nm.w->data, L::VEC);
            } else {
                put_tensor(fx.tensors, name + "_gain.weight", nm.w->data, L::VEC);
                put_tensor(fx.tensors, name + "_gain.bias", nm.b->data, L::VEC);
                for (size_t j = 0, k = 0; j < c.norm_inputs.size(); ++j)
                    if (c.norm_inputs[j].kind != "row")
                        put_param(name + "_p_" + flex_input_letter(j) + ".weight", nm.params[k++], c.d);
            }
        };
        for (size_t b = 0; b < blocks.size(); ++b) {
            const std::string g = "blk." + std::to_string(b) + ".";
            const auto& k = *blocks[b];
            if (k.ns[0].on()) put_norm(g + "attn_norm", k.ns[0]);
            const nn::Linear& qkv = k.attn ? *k.attn->c_attn : *k.swa_attn->c_attn;
            const nn::Linear& out = k.attn ? *k.attn->c_proj : *k.swa_attn->c_proj;
            put_linear(fx.tensors, g + "attn_qkv", qkv);
            if (k.qkv_conv_w) put_tensor(fx.tensors, g + "attn_qkv_conv.weight", k.qkv_conv_w->data, L::ROWS);  // [3d, K]
            if (k.rel_k) put_tensor(fx.tensors, g + "attn_rel_k.weight", k.rel_k->data, L::ROWS);  // [K + 1, W]
            if (k.rel_v) put_tensor(fx.tensors, g + "attn_rel_v.weight", k.rel_v->data, L::ROWS);
            put_linear(fx.tensors, g + "attn_output", out);
            if (k.ns[2].on()) put_norm(g + "attn_post_norm", k.ns[2]);
            if (highway) put_linear(fx.tensors, g + "highway_attn", *k.gate_a);
            if (k.ns[1].on()) put_norm(g + "ffn_norm", k.ns[1]);
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
                if (k.fc_gate) put_linear(fx.tensors, g + "ffn_gate", *k.fc_gate);
                for (size_t j = 0, q = 0; j < c.act_inputs.size(); ++j)
                    if (c.act_inputs[j].kind == "scalar" || c.act_inputs[j].kind == "channel")
                        put_param(g + "ffn_act_p_" + flex_input_letter(j) + ".weight", k.act_params[q++], c.d_ff);
            }
            if (k.ns[3].on()) put_norm(g + "ffn_post_norm", k.ns[3]);
            if (highway) put_linear(fx.tensors, g + "highway_ffn", *k.gate_m);
        }
        if (fn.on()) put_norm("output_norm", fn);
        put_linear(fx.tensors, "output", *head);
        if (t5) put_tensor(fx.tensors, "rel_pos_bias.weight", rel_bias->data, L::ROWS);
        for (size_t j = 0, k = 0; j < c.embed_inputs.size(); ++j)
            if (c.embed_inputs[j].kind != "row")
                put_param("embd_p_" + flex_input_letter(j) + ".weight", embed_params[k++], c.d);
        return fx;
    }

private:
    std::map<std::string, Matrix> gpt2_tensors() const {
        std::map<std::string, Matrix> sd;
        put_gpt2_shell(sd, *wte, *wpe, *ln_f_ln, *head);
        for (size_t b = 0; b < blocks.size(); ++b) {
            const auto& k = *blocks[b];
            if (k.swa_attn)
                put_gpt2_block(sd, b, *k.ns[0].ln, *k.swa_attn, *k.ns[1].ln, *k.mlp);
            else
                put_gpt2_block(sd, b, *k.ns[0].ln, *k.attn, *k.ns[1].ln, *k.mlp);
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
    FlexNorm fn;                            // the norm before the head (cfg.has_final_norm())
    std::shared_ptr<const Expr> norm_expr;  // norm "expr:<e>", parsed once
    std::shared_ptr<nn::Linear> head;
    std::shared_ptr<const Expr> embed_expr; // cfg.embed_expr, parsed once
    std::vector<Var> embed_params;          // its learned inputs
    Var rel_bias;                           // position t5_relative: [rel_buckets, heads], every layer's
    std::vector<int> rel_buckets;           // its bucket by distance (ops::t5_position_buckets)
};

// Mamba-1 (Gu and Dao 2023; llama.cpp's mamba graph, keramos IR_SPEC 6.6-6.7 and tools/gguf_ir.py's
// mamba builder, node for node): a pre-RMSNorm block, in_proj d -> 2 Di (x | z), the depthwise causal
// conv1d (kernel K, biased) and SiLU on x, x_proj Di -> R + 2 N (dt | B | C), dt_proj R -> Di (biased),
// the selective scan (softplus(dt), A = -exp(A_log), the D skip), times SiLU(z), out_proj Di -> d, the
// residual. Di = expand * d, R = dt_rank. LOCAL_LLM_STACK.md section 11 item 4.4.
struct Mamba1Config {
    bool on = false;
    size_t d_conv = 4, expand = 2, dt_rank = 0;  // dt_rank 0 = ceil(d / 16)
};

class Mamba1Block : public nn::Module {
public:
    // Mamba's initialisation: in/x/out projections as nn::Linear's; conv1d weight and bias uniform in
    // +-1/sqrt(K) (torch's Conv1d at fan-in K); dt_proj's weight uniform in +-R^-1/2 and its bias the
    // inverse softplus of a log-uniform dt in [1e-3, 1e-1]; A_log[c, n] = log(n + 1); D = 1.
    Mamba1Block(size_t d, size_t d_state, const Mamba1Config& c, unsigned seed)
        : d_(d), N(d_state), K(c.d_conv), Di(c.expand * d), R(c.dt_rank ? c.dt_rank : (d + 15) / 16) {
        if (K < 2 || Di == 0 || N == 0) throw std::runtime_error("mamba1: d_conv >= 2, expand >= 1, d_state >= 1");
        norm_w = reg("norm.weight", Matrix(1, d, 1.0f));
        in_proj = mod<nn::Linear>("in_proj", d, 2 * Di, false, seed + 1);
        std::mt19937 g(seed + 2);
        const float cb = 1.0f / std::sqrt(static_cast<float>(K));
        std::uniform_real_distribution<float> uc(-cb, cb);
        Matrix cw(Di, K), cbias(1, Di);
        for (size_t i = 0; i < Di; ++i)
            for (size_t k = 0; k < K; ++k) cw(i, k) = uc(g);
        for (size_t i = 0; i < Di; ++i) cbias(0, i) = uc(g);
        conv_w = reg("conv1d.weight", cw);
        conv_b = reg("conv1d.bias", cbias);
        x_proj = mod<nn::Linear>("x_proj", Di, R + 2 * N, false, seed + 3);
        dt_proj = mod<nn::Linear>("dt_proj", R, Di, true, seed + 4);
        std::mt19937 g2(seed + 5);
        const float ds = 1.0f / std::sqrt(static_cast<float>(R));
        std::uniform_real_distribution<float> ud(-ds, ds), u01(0.0f, 1.0f);
        for (size_t i = 0; i < R; ++i)
            for (size_t j = 0; j < Di; ++j) dt_proj->W->data(i, j) = ud(g2);
        for (size_t j = 0; j < Di; ++j) {
            const double dt = std::exp(u01(g2) * (std::log(0.1) - std::log(0.001)) + std::log(0.001));
            dt_proj->b->data(0, j) = static_cast<float>(dt + std::log(-std::expm1(-dt)));  // softplus^-1
        }
        Matrix al(Di, N);
        for (size_t i = 0; i < Di; ++i)
            for (size_t n = 0; n < N; ++n) al(i, n) = std::log(static_cast<float>(n + 1));
        A_log = reg("A_log", al);
        Dskip = reg("D", Matrix(1, Di, 1.0f));
        out_proj = mod<nn::Linear>("out_proj", Di, d, false, seed + 6);
    }

    // A = -exp(A_log), the matrix the scan reads and the GGUF stores (ssm_a).
    Var A() const { return ops::scale(ops::exp(A_log), -1.0f); }

    // x [rows, d]: sequences of seq_len rows (0 = one), each scanned from a zero state.
    Var forward(const Var& x, size_t seq_len) const {
        Var h = ops::rmsnorm(x, norm_w);
        Var xz = in_proj->forward(h);
        Var xs = ops::slice_cols(xz, 0, Di), z = ops::slice_cols(xz, Di, 2 * Di);
        xs = ops::silu(ops::conv1d_causal(xs, conv_w, conv_b, seq_len));
        Var xdb = x_proj->forward(xs);
        Var dt = dt_proj->forward(ops::slice_cols(xdb, 0, R));
        Var Bm = ops::slice_cols(xdb, R, R + N), Cm = ops::slice_cols(xdb, R + N, R + 2 * N);
        Var y = ops::selective_scan(xs, dt, Bm, Cm, A(), Dskip, /*dt_softplus=*/true, seq_len);
        y = ops::mul(y, ops::silu(z));
        return ops::add(x, out_proj->forward(y));
    }

    // The block's IR nodes (tools/gguf_ir.py mamba) and its two recurrent slots.
    ir::json export_nodes() const {
        const std::string p = "blk.{i}.";
        ir::json n = ir::json::array();
        n.push_back(ir::rmsnorm("x", p + "attn_norm.weight", ops::kRmsNormEps, "h"));
        n.push_back(in_proj->export_ir("h", "xz", p + "ssm_in.weight"));
        n.push_back(ir::split("xz", {Di, Di}, {"xs", "z"}));
        n.push_back({{"op", "conv1d_causal"}, {"in", ir::json::array({"xs"})}, {"state", "conv"},
                     {"weight", p + "ssm_conv1d.weight"}, {"bias", p + "ssm_conv1d.bias"}, {"channels", Di},
                     {"kernel", K}, {"out", "xs"}});
        n.push_back(ir::unary("silu", "xs", "xs"));
        n.push_back(x_proj->export_ir("xs", "xdb", p + "ssm_x.weight"));
        n.push_back(ir::split("xdb", {R, N, N}, {"dt", "bm", "cm"}));
        n.push_back(dt_proj->export_ir("dt", "dt", p + "ssm_dt.weight"));
        n.push_back({{"op", "selective_scan"}, {"in", ir::json::array({"xs", "dt", "bm", "cm"})}, {"state", "ssm"},
                     {"a", p + "ssm_a"}, {"d", p + "ssm_d"}, {"channels", Di}, {"d_state", N},
                     {"dt_softplus", true}, {"out", "y"}});
        n.push_back(ir::unary("silu", "z", "z"));
        n.push_back(ir::binary("mul", "y", "z", "y"));
        n.push_back(out_proj->export_ir("y", "o", p + "ssm_out.weight"));
        n.push_back(ir::binary("add", "x", "o", "x"));
        return n;
    }
    ir::json export_state() const {
        return ir::json::array(
            {{{"name", "conv"}, {"kind", "recurrent"}, {"op", "conv1d_causal"}, {"dtype", "f32"},
              {"tensors", ir::json::array({{{"name", "x"}, {"shape", ir::json::array({K - 1, Di})}, {"init", 0}}})}},
             {{"name", "ssm"}, {"kind", "recurrent"}, {"op", "selective_scan"}, {"dtype", "f32"},
              {"tensors", ir::json::array({{{"name", "s"}, {"shape", ir::json::array({Di, N})}, {"init", 0}}})}}});
    }

    size_t d_, N, K, Di, R;
    Var norm_w, conv_w, conv_b, A_log, Dskip;
    std::shared_ptr<nn::Linear> in_proj, x_proj, dt_proj, out_proj;
};

// The Mamba-style SSM LM (mamba::MambaModel's stack, same names and seeds)
// as an mtstudio family: token embedding, no positions, mamba::MambaBlock x
// L (pre-LN, S4 scan with ops::ssm_scan, gelu output gates, skip), final
// LayerNorm, biased head. MambaModel::forward takes token ids as a float
// column; this one takes ids, and runs a stacked mini-batch one sequence at
// a time so no scan state crosses a sequence boundary.
//
// With m1.on (arch.custom.ssm "mamba1"): Mamba1Block x L instead (above), a
// final RMSNorm and an unbiased head; a stacked mini-batch runs as one graph,
// each sequence's conv and scan starting from a zero state.
class MambaLM : public nn::Module {
public:
    MambaLM(size_t vocab, size_t d, size_t n_layers, size_t d_state, unsigned seed, const Mamba1Config& m1 = {})
        : d_state(d_state), m1cfg(m1) {
        if (m1.on) {
            embed = mod<nn::Embedding>("embed", vocab, d);
            for (size_t i = 0; i < n_layers; ++i)
                m1blocks.push_back(mod<Mamba1Block>("layers." + std::to_string(i), d, d_state, m1,
                                                    seed + 100 * static_cast<unsigned>(i + 1)));
            norm_f = reg("norm_f.weight", Matrix(1, d, 1.0f));
            lm_head = mod<nn::Linear>("lm_head", d, vocab, false, seed + 21);
            return;
        }
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
        if (m1cfg.on) {
            Var h = embed->forward(ids);
            for (const auto& b : m1blocks) h = b->forward(h, sl);
            return lm_head->forward(ops::rmsnorm(h, norm_f));
        }
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
        if (m1cfg.on) return mamba1_export(ctx);
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

    // Mamba-1 (keramos tools/gguf_ir.py's mamba graph): llama.cpp's mamba tensor names, A stored as
    // -exp(A_log) computed by the op the forward runs (so the file holds the forward's floats).
    FamilyExport mamba1_export(size_t ctx) const {
        const size_t vocab = embed->weight->data.rows(), d = embed->weight->data.cols();
        const auto& b0 = *m1blocks.at(0);
        FamilyExport fx;
        fx.arch = "paperkiln";
        const ir::json embed_nodes = ir::json::array({embed->export_ir("tokens", "x", "token_embd.weight")});
        const ir::json head_nodes =
            ir::json::array({ir::rmsnorm("x", "output_norm.weight", ops::kRmsNormEps, "h"),
                             lm_head->export_ir("h", "logits", "output.weight")});
        fx.ir = family_doc("mamba1", vocab, d, m1blocks.size(), ctx, /*positions=*/false, embed_nodes,
                           b0.export_state(), b0.export_nodes(), head_nodes);
        using L = gguf::NamedTensor::Layout;
        put_tensor(fx.tensors, "token_embd.weight", embed->weight->data, L::ROWS);
        NoGrad ng;
        for (size_t b = 0; b < m1blocks.size(); ++b) {
            const std::string g = "blk." + std::to_string(b) + ".";
            const auto& k = *m1blocks[b];
            put_tensor(fx.tensors, g + "attn_norm.weight", k.norm_w->data, L::VEC);
            put_linear(fx.tensors, g + "ssm_in", *k.in_proj);
            put_tensor(fx.tensors, g + "ssm_conv1d.weight", k.conv_w->data, L::ROWS);  // [Di, K]
            put_tensor(fx.tensors, g + "ssm_conv1d.bias", k.conv_b->data, L::VEC);
            put_linear(fx.tensors, g + "ssm_x", *k.x_proj);
            put_linear(fx.tensors, g + "ssm_dt", *k.dt_proj);
            put_tensor(fx.tensors, g + "ssm_a", k.A()->data, L::ROWS);  // [Di, N]
            put_tensor(fx.tensors, g + "ssm_d", k.Dskip->data, L::VEC);
            put_linear(fx.tensors, g + "ssm_out", *k.out_proj);
        }
        put_tensor(fx.tensors, "output_norm.weight", norm_f->data, L::VEC);
        put_linear(fx.tensors, "output", *lm_head);
        return fx;
    }

    size_t d_state;
    Mamba1Config m1cfg;
    std::vector<std::shared_ptr<Mamba1Block>> m1blocks;
    Var norm_f;
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
