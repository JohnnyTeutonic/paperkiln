// Attention Residuals on the llama block (include/microtorch/llama_attnres.hpp).
#include "microtorch/llama_attnres.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "microtorch/depth_attend.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/ir.hpp"

namespace microtorch {
namespace nn {

AttnResQueries::AttnResQueries(size_t d, size_t applications) {
    attn_q.assign(applications, Var());
    ffn_q.assign(applications, Var());
    // Zero-initialised: no random draw, so every other tensor of the model
    // starts as in the plain llama of the same seed.
    for (size_t a = 0; a < applications; ++a) {
        if (a > 0) attn_q[a] = reg("attn_q." + std::to_string(a), Matrix(1, d));
        ffn_q[a] = reg("ffn_q." + std::to_string(a), Matrix(1, d));
    }
    out_q = reg("out_q", Matrix(1, d));
}

void attnres_check(const LlamaConfig& cfg) {
    if (!cfg.attn_res.on) return;
    if (cfg.attn_res.scale != "sources" && cfg.attn_res.scale != "none")
        throw std::runtime_error("llama attn_res: scale must be sources or none, got " + cfg.attn_res.scale);
    if (cfg.recursion.routed())
        throw std::runtime_error(
            "llama attn_res with recursion schedule mor is not implemented (the bank's per-row presence, "
            "IR_SPEC_DRAFT section 4, is exploratory)");
    if (cfg.recursion.on() && cfg.recursion.inject != "none")
        throw std::runtime_error("llama attn_res: recursion inject add has no residual stream to add to");
    if (cfg.moe.experts > 0) throw std::runtime_error("llama attn_res with MoE is not wired");
    const size_t sources = 2 * layer_schedule(cfg).size() + 1;
    if (sources > device::devops::kDepthAttendMaxSources)
        throw std::runtime_error("llama attn_res: " + std::to_string(sources) + " sources exceed the device kernel's " +
                                 std::to_string(device::devops::kDepthAttendMaxSources));
}

namespace {

// LlamaBlock::forward's attention sublayer without the residual add: the
// o_proj output for input x (rows of B stacked sequences of seq_len).
Var attn_sub(const LlamaBlock& b, const Var& x, const std::vector<int>& pos, size_t seq_len) {
    Var h = ops::rmsnorm(x, b.ln1_w);
    Var q = b.q_proj->forward(h);
    Var k = b.k_proj->forward(h);
    Var v = b.v_proj->forward(h);
    Var qkv_in = ops::concat_cols({q, k, v});
    for (const Var* t : {&q, &k, &v}) device::release((*t)->data);
    Var qkv = ops::apply_rope(qkv_in, pos, b.rope_theta_, b.dk, b.rope_all_heads_);
    device::release(qkv_in->data);
    const float sc = 1.0f / std::sqrt(static_cast<float>(b.dk));
    Var heads_cat = ops::attention_heads(qkv, b.H, sc, seq_len, /*causal=*/true);
    // Not released: the output is a depth-attention source (read again by
    // later sublayers' forward and by their backward).
    return b.o_proj->forward(heads_cat);
}

Var ffn_sub(const LlamaBlock& b, const Var& x) {
    Var h2 = ops::rmsnorm(x, b.ln2_w);
    return b.down_proj->forward(ops::swiglu(b.gate_proj->forward(h2), b.up_proj->forward(h2)));
}

// LlamaBlock::forward_cached_rows's attention (causal, every cached row) without the residual add.
Var attn_sub_cached(const LlamaBlock& b, const Var& x, const std::vector<int>& pos, std::vector<float>& kc,
                    std::vector<float>& vc) {
    const size_t n = x->data.rows(), d = x->data.cols();
    if (n > 0 && static_cast<size_t>(pos.back()) + 1 > b.n_ctx)
        throw std::runtime_error("llama: kv cache past the context length");
    const size_t base = kc.size() / d;
    Var h = ops::rmsnorm(x, b.ln1_w);
    Var qkv_in = ops::concat_cols({b.q_proj->forward(h), b.k_proj->forward(h), b.v_proj->forward(h)});
    Var qkv = ops::apply_rope(qkv_in, pos, b.rope_theta_, b.dk, b.rope_all_heads_);
    device::materialize(qkv->data);
    const Matrix& m = qkv->data;
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < d; ++c) {
            kc.push_back(m(i, d + c));
            vc.push_back(m(i, 2 * d + c));
        }
    const float sc = 1.0f / std::sqrt(static_cast<float>(b.dk));
    const size_t dk = b.dk;
    Matrix att(n, d);
    std::vector<float> a;
    for (size_t i = 0; i < n; ++i) {
        const size_t hi = base + i + 1;
        a.resize(hi);
        for (size_t hd = 0; hd < b.H; ++hd) {
            const size_t o = hd * dk;
            float mx = -1e30f;
            for (size_t j = 0; j < hi; ++j) {
                float s = 0.0f;
                for (size_t c = 0; c < dk; ++c) s += m(i, o + c) * kc[j * d + o + c];
                a[j] = s * sc;
                mx = std::max(mx, a[j]);
            }
            float z = 0.0f;
            for (size_t j = 0; j < hi; ++j) z += (a[j] = std::exp(a[j] - mx));
            for (size_t j = 0; j < hi; ++j) a[j] /= z;
            for (size_t c = 0; c < dk; ++c) {
                float acc = 0.0f;
                for (size_t j = 0; j < hi; ++j) acc += a[j] * vc[j * d + o + c];
                att(i, o + c) = acc;
            }
        }
    }
    return b.o_proj->forward(make_var(std::move(att)));
}

float mix_scale(const LlamaConfig& cfg, size_t sources) {
    return cfg.attn_res.scale == "sources" && sources > 1 ? static_cast<float>(sources) : 1.0f;
}

Var head(const Llama& m, const Var& x) {
    Var h = ops::rmsnorm(x, m.norm_w);
    if (m.lm_head) return m.lm_head->forward(h);
    return ops::matmul(h, ops::transpose(m.embed_tokens->weight));
}

}  // namespace

Var attnres_forward(const Llama& m, const std::vector<int>& ids, size_t seq_len, MoEAux*) {
    const size_t sl = seq_len == 0 ? ids.size() : seq_len;
    if (ids.size() % sl != 0) throw std::runtime_error("ids not a multiple of seq_len");
    std::vector<int> pos(ids.size());
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i % sl);
    const AttnResQueries& q = *m.attn_res;
    const bool ckpt = m.checkpoint_blocks && grad_enabled();
    std::vector<Var> bank{m.embed_tokens->forward(ids)};
    auto mix = [&](const Var& w) {
        return ops::depth_attend(bank, w, ops::kRmsNormEps, mix_scale(m.cfg, bank.size()));
    };
    for (size_t a = 0; a < m.schedule.size(); ++a) {
        const auto& b = m.blocks[m.schedule[a]];
        Var h = mix(q.attn_q[a]);
        bank.push_back(ckpt ? checkpoint([b, pos, seq_len](const Var& v) { return attn_sub(*b, v, pos, seq_len); }, h)
                            : attn_sub(*b, h, pos, seq_len));
        Var h2 = mix(q.ffn_q[a]);
        bank.push_back(ckpt ? checkpoint([b](const Var& v) { return ffn_sub(*b, v); }, h2) : ffn_sub(*b, h2));
    }
    return head(m, mix(q.out_q));
}

Var attnres_forward_cached(const Llama& m, const std::vector<int>& ids, LlamaKV& kv) {
    if (ids.empty()) throw std::runtime_error("llama: forward_cached needs at least one new token");
    const size_t A = m.schedule.size();
    if (kv.k.empty()) {
        kv.k.assign(A, {});
        kv.v.assign(A, {});
        kv.d = m.cfg.d;
        kv.len = 0;
    }
    if (kv.k.size() != A) throw std::runtime_error("llama: kv cache is for another depth");
    std::vector<int> pos(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) pos[i] = static_cast<int>(kv.len + i);
    const AttnResQueries& q = *m.attn_res;
    // Depth attention is per row: the new rows' bank is all decode needs.
    std::vector<Var> bank{m.embed_tokens->forward(ids)};
    auto mix = [&](const Var& w) {
        return ops::depth_attend(bank, w, ops::kRmsNormEps, mix_scale(m.cfg, bank.size()));
    };
    for (size_t a = 0; a < A; ++a) {
        const auto& b = m.blocks[m.schedule[a]];
        bank.push_back(attn_sub_cached(*b, mix(q.attn_q[a]), pos, kv.k[a], kv.v[a]));
        bank.push_back(ffn_sub(*b, mix(q.ffn_q[a])));
    }
    kv.len += ids.size();
    return head(m, mix(q.out_q));
}

std::vector<std::pair<std::string, std::string>> attnres_gguf_names(const Llama& m) {
    const bool shared = m.schedule.size() != m.blocks.size();
    const std::string pre = shared ? "rec_app." : "blk.";
    std::vector<std::pair<std::string, std::string>> out;
    for (size_t a = 0; a < m.schedule.size(); ++a) {
        const std::string g = pre + std::to_string(a) + ".";
        if (a > 0) out.emplace_back("attn_res.attn_q." + std::to_string(a), g + "attnres_attn_q.weight");
        out.emplace_back("attn_res.ffn_q." + std::to_string(a), g + "attnres_ffn_q.weight");
    }
    out.emplace_back("attn_res.out_q", "attnres_out_q.weight");
    return out;
}

std::string attnres_unused_query_name(const Llama& m) {
    return (m.schedule.size() != m.blocks.size() ? "rec_app." : "blk.") + std::string("0.attnres_attn_q.weight");
}

ir::json attnres_export_ir(const Llama& m) {
    const auto& cfg = m.cfg;
    const LlamaBlock& b0 = *m.blocks.at(0);
    const size_t A = m.schedule.size();
    const bool shared = A != m.blocks.size();
    const bool scaled = cfg.attn_res.scale == "sources";
    const float eps = ops::kRmsNormEps;
    const float sc = 1.0f / std::sqrt(static_cast<float>(b0.dk));
    const std::string p = "blk.{i}.", qp = shared ? "rec_app.{i}." : "blk.{i}.";
    // A depth_attend node over the bank into x, then (scale "sources") x * S.
    auto mix_nodes = [&](ir::json& n, const std::string& weight, const std::vector<float>& per_layer) {
        ir::json da = ir::depth_attend({"bank"}, weight, eps, "x");
        if (shared) {
            // The query is the application's own (IR_SPEC_DRAFT section 3).
            ir::json t = ir::json::object();
            for (const auto& [k, v] : da.items()) {
                if (k == "out") t["tensor_index"] = "application";
                t[k] = v;
            }
            da = t;
        }
        n.push_back(da);
        if (scaled)
            n.push_back({{"op", "affine"}, {"in", ir::json::array({"x"})}, {"scale", {{"per_layer", per_layer}}},
                         {"shift", 0.0}, {"out", "x"}});
    };
    std::vector<float> s_attn(A), s_ffn(A);
    for (size_t a = 0; a < A; ++a) {
        s_attn[a] = static_cast<float>(2 * a + 1);
        s_ffn[a] = static_cast<float>(2 * a + 2);
    }
    ir::json n = ir::json::array();
    mix_nodes(n, qp + "attnres_attn_q.weight", s_attn);
    n.push_back(ir::rmsnorm("x", p + "attn_norm.weight", eps, "h"));
    n.push_back(b0.q_proj->export_ir("h", "q", p + "attn_q.weight"));
    n.push_back(b0.k_proj->export_ir("h", "k", p + "attn_k.weight"));
    n.push_back(b0.v_proj->export_ir("h", "v", p + "attn_v.weight"));
    n.push_back(ir::rope("q", "k", "pos", b0.H, b0.dk, b0.rope_theta_, b0.rope_all_heads_));
    n.push_back(ir::attention("q", "k", "v", "kv", b0.H, b0.H, b0.dk, sc, /*causal=*/true, 0, 0, "a"));
    n.push_back(b0.o_proj->export_ir("a", "o", p + "attn_output.weight"));
    n.push_back(ir::append("bank", "o", "bank"));
    mix_nodes(n, qp + "attnres_ffn_q.weight", s_ffn);
    n.push_back(ir::rmsnorm("x", p + "ffn_norm.weight", eps, "h2"));
    n.push_back(b0.gate_proj->export_ir("h2", "g", p + "ffn_gate.weight"));
    n.push_back(b0.up_proj->export_ir("h2", "u", p + "ffn_up.weight"));
    n.push_back(ir::unary("silu", "g", "g"));
    n.push_back(ir::binary("mul", "g", "u", "gu"));
    n.push_back(b0.down_proj->export_ir("gu", "f", p + "ffn_down.weight"));
    n.push_back(ir::append("bank", "f", "bank"));

    ir::json head_linear;
    if (m.lm_head) {
        head_linear = m.lm_head->export_ir("h", "logits", "output.weight");
    } else {
        head_linear = {{"op", "linear"},       {"in", ir::json::array({"h"})},
                       {"weight", "output.weight"}, {"bias", nullptr},
                       {"tied_to", "token_embd.weight"}, {"out", "logits"}};
    }
    ir::json head_nodes = ir::json::array({ir::depth_attend({"bank"}, "attnres_out_q.weight", eps, "x")});
    if (scaled) head_nodes.push_back(ir::affine("x", static_cast<float>(2 * A + 1), 0.0f, "x"));
    head_nodes.push_back(ir::rmsnorm("x", "output_norm.weight", eps, "h"));
    head_nodes.push_back(head_linear);

    ir::json doc;
    doc["ir_version"] = ir::kVersion;
    doc["family_hint"] = "llama_attnres";
    doc["model"] = {{"vocab", cfg.vocab}, {"d", cfg.d}, {"n_layers", A}, {"ctx", cfg.n_ctx}, {"dtype", "f32"}};
    doc["inputs"] = ir::json::array({{{"name", "tokens"}, {"kind", "token_ids"}},
                                     {{"name", "pos"},
                                      {"kind", "positions"},
                                      {"note", "absolute positions; prefill 0..T-1, decode = cache length"}}});
    ir::json embed = ir::json::array({m.embed_tokens->export_ir("tokens", "x", "token_embd.weight")});
    embed.push_back(ir::list({"x"}, "bank"));
    doc["embed"] = {{"nodes", embed}};
    doc["block"] = {{"repeat", A}};
    if (shared) {
        const auto& r = cfg.recursion;
        doc["block"]["weight_index"] = {{"per_layer", m.schedule}};
        doc["block"]["_recursion"] = {{"prelude", r.prelude}, {"block", r.block}, {"repeats", r.repeats},
                                      {"coda", r.coda},       {"schedule", r.schedule},
                                      {"inject", r.inject},   {"stored_layers", m.blocks.size()}};
    }
    doc["block"]["_attn_res"] = {{"form", "full"}, {"scale", cfg.attn_res.scale}, {"sources", 2 * A + 1},
                                 {"queries", shared ? "application" : "layer"}};
    doc["block"]["state"] = ir::json::array({ir::kv_state("kv", b0.H, b0.H, b0.dk, b0.n_ctx, 0, 0)});
    doc["block"]["nodes"] = n;
    doc["head"] = {{"nodes", head_nodes}};
    doc["outputs"] = ir::json::array({{{"name", "logits"}, {"kind", "logits"}}});
    doc["modes"] = {{"prefill", "all nodes over T tokens"},
                    {"decode", "the same graph with T = 1, attention reads and appends to the kv state; the bank "
                               "holds the current token's sources only (depth attention is per row)"}};
    return doc;
}

}  // namespace nn
}  // namespace microtorch
