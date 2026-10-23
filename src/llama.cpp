#include "microtorch/llama.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "microtorch/device_cache.hpp"
#include "microtorch/ir.hpp"

namespace microtorch {
namespace nn {

LlamaBlock::LlamaBlock(const LlamaConfig& cfg, unsigned seed)
    : H(cfg.n_heads),
      dk(cfg.d / cfg.n_heads),
      n_ctx(cfg.n_ctx),
      rope_theta_(cfg.rope_theta),
      rms_eps_(cfg.rms_eps),
      rope_all_heads_(cfg.rope_all_heads) {
    if (cfg.d % cfg.n_heads != 0) throw std::runtime_error("llama: d must divide by n_heads");
    // HF names throughout; Linear registers its matrix as "weight", so the
    // collected dotted paths match LlamaForCausalLM exactly.
    ln1_w = reg("input_layernorm.weight", Matrix(1, cfg.d, 1.0f));
    ln2_w = reg("post_attention_layernorm.weight", Matrix(1, cfg.d, 1.0f));
    q_proj = mod<Linear>("self_attn.q_proj", cfg.d, cfg.d, false, seed + 1);
    k_proj = mod<Linear>("self_attn.k_proj", cfg.d, cfg.d, false, seed + 2);
    v_proj = mod<Linear>("self_attn.v_proj", cfg.d, cfg.d, false, seed + 3);
    o_proj = mod<Linear>("self_attn.o_proj", cfg.d, cfg.d, false, seed + 4);
    if (cfg.moe.experts > 0) {
        // Registered as "mlp": layers.N.mlp.router.weight,
        // layers.N.mlp.experts.{e}.{gate,up,down}_proj.weight. Expert 0 takes
        // the dense FFN's seeds (seed + 5, 6, 7).
        moe = mod<MoEFFN>("mlp", cfg.d, cfg.d_ff, cfg.moe, seed + 5);
    } else {
        gate_proj = mod<Linear>("mlp.gate_proj", cfg.d, cfg.d_ff, false, seed + 5);
        up_proj = mod<Linear>("mlp.up_proj", cfg.d, cfg.d_ff, false, seed + 6);
        down_proj = mod<Linear>("mlp.down_proj", cfg.d_ff, cfg.d, false, seed + 7);
    }
}

Var LlamaBlock::forward(const Var& x, const std::vector<int>& pos, size_t seq_len, MoEAux* aux) const {
    // ---- attention sublayer (pre-RMSNorm) ----
    Var h = ops::rmsnorm(x, ln1_w);
    Var q = q_proj->forward(h);
    Var k = k_proj->forward(h);
    Var v = v_proj->forward(h);
    // ops::apply_rope expects the fused [T, 3d] qkv layout; build it, rotate
    // each head's q/k columns (head 0 only under the legacy rope_heads =
    // "first"), attend per head. concat is a tape op, so gradients flow.
    //
    // Memory: the tape keeps every node until the step ends, but the
    // projections and their concat are read by no backward (concat and
    // rope backwards touch gradients only) and by no later forward op once
    // copied onward, so their device copies are released as soon as the
    // next op has run (device::release; no-op unless deferring); bit for
    // bit the same step.
    Var qkv_in = ops::concat_cols({q, k, v});
    for (const Var* t : {&q, &k, &v}) device::release((*t)->data);
    Var qkv = ops::apply_rope(qkv_in, pos, rope_theta_, dk, rope_all_heads_);
    device::release(qkv_in->data);

    // Fused multi-head attention: causal within each sequence,
    // block-isolated across a stacked batch, no mask materialization
    // (ops::attention_heads = concat of ops::fused_attention per head;
    // FD-checked). Its output, the head concat, feeds o_proj.
    const float sc = 1.0f / std::sqrt(static_cast<float>(dk));
    Var heads_cat = ops::attention_heads(qkv, H, sc, seq_len, /*causal=*/true);
    Var attn = o_proj->forward(heads_cat);
    Var x1 = ops::add(x, attn);
    device::release(attn->data);  // add's backward reads no data

    // ---- SwiGLU FFN sublayer (dense, or a top-k mixture of SwiGLU experts) ----
    Var h2 = ops::rmsnorm(x1, ln2_w);
    Var ffn = moe ? moe->forward(h2, aux)
                  : down_proj->forward(ops::swiglu(gate_proj->forward(h2), up_proj->forward(h2)));
    Var out = ops::add(x1, ffn);
    if (!moe) device::release(ffn->data);
    return out;
}

Var LlamaBlock::forward_cached(const Var& x, size_t pos0, std::vector<float>& kc, std::vector<float>& vc) const {
    const size_t n = x->data.rows(), d = x->data.cols();
    if (pos0 + n > n_ctx) throw std::runtime_error("llama: kv cache past the context length");
    Var h = ops::rmsnorm(x, ln1_w);
    Var qkv_in = ops::concat_cols({q_proj->forward(h), k_proj->forward(h), v_proj->forward(h)});
    std::vector<int> pos(n);
    for (size_t i = 0; i < n; ++i) pos[i] = static_cast<int>(pos0 + i);
    Var qkv = ops::apply_rope(qkv_in, pos, rope_theta_, dk, rope_all_heads_);
    device::materialize(qkv->data);
    const Matrix& m = qkv->data;  // [n, 3d]: q | k | v
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < d; ++c) {
            kc.push_back(m(i, d + c));
            vc.push_back(m(i, 2 * d + c));
        }
    // fused_attention's host reference, row by row over the cache: scaled
    // scores, max-subtracted exp, normalised, then the weighted values.
    const float sc = 1.0f / std::sqrt(static_cast<float>(dk));
    Matrix att(n, d);
    std::vector<float> a;
    for (size_t i = 0; i < n; ++i) {
        const size_t hi = pos0 + i + 1;  // causal: keys 0..own position
        a.resize(hi);
        for (size_t hd = 0; hd < H; ++hd) {
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
    Var x1 = ops::add(x, o_proj->forward(make_var(std::move(att))));
    Var h2 = ops::rmsnorm(x1, ln2_w);
    Var ffn = moe ? moe->forward(h2, nullptr)
                  : down_proj->forward(ops::swiglu(gate_proj->forward(h2), up_proj->forward(h2)));
    return ops::add(x1, ffn);
}

ir::json LlamaBlock::export_ir() const {
    // forward() above, node for node. The scale is the float it computes.
    const float sc = 1.0f / std::sqrt(static_cast<float>(dk));
    const std::string p = "blk.{i}.";
    ir::json n = ir::json::array();
    n.push_back(ir::rmsnorm("x", p + "attn_norm.weight", ops::kRmsNormEps, "h"));
    n.push_back(q_proj->export_ir("h", "q", p + "attn_q.weight"));
    n.push_back(k_proj->export_ir("h", "k", p + "attn_k.weight"));
    n.push_back(v_proj->export_ir("h", "v", p + "attn_v.weight"));
    n.push_back(ir::rope("q", "k", "pos", H, dk, rope_theta_, rope_all_heads_));
    n.push_back(ir::attention("q", "k", "v", "kv", H, H, dk, sc, /*causal=*/true, 0, 0, "a"));
    n.push_back(o_proj->export_ir("a", "o", p + "attn_output.weight"));
    n.push_back(ir::binary("add", "x", "o", "x"));
    n.push_back(ir::rmsnorm("x", p + "ffn_norm.weight", ops::kRmsNormEps, "h2"));
    if (moe) {
        n.push_back(moe->export_ir("h2", "f", p));
    } else {
        n.push_back(gate_proj->export_ir("h2", "g", p + "ffn_gate.weight"));
        n.push_back(up_proj->export_ir("h2", "u", p + "ffn_up.weight"));
        n.push_back(ir::unary("silu", "g", "g"));
        n.push_back(ir::binary("mul", "g", "u", "gu"));
        n.push_back(down_proj->export_ir("gu", "f", p + "ffn_down.weight"));
    }
    n.push_back(ir::binary("add", "x", "f", "x"));
    return {{"state", ir::json::array({ir::kv_state("kv", H, H, dk, n_ctx, 0, 0)})}, {"nodes", n}};
}

std::vector<size_t> layer_schedule(const LlamaConfig& cfg) {
    std::vector<size_t> s;
    const auto& r = cfg.recursion;
    if (!r.on()) {
        for (size_t i = 0; i < cfg.n_layers; ++i) s.push_back(i);
        return s;
    }
    if (r.schedule != "fixed")
        throw std::runtime_error("llama recursion: schedule \"" + r.schedule +
                                 "\" is not implemented (only \"fixed\": every token takes every application)");
    if (r.repeats < 1) throw std::runtime_error("llama recursion: repeats must be >= 1");
    if (cfg.n_layers != r.prelude + r.block + r.coda)
        throw std::runtime_error("llama recursion: n_layers " + std::to_string(cfg.n_layers) +
                                 " is not prelude + block + coda = " +
                                 std::to_string(r.prelude + r.block + r.coda) + " (n_layers counts stored layers)");
    for (size_t i = 0; i < r.prelude; ++i) s.push_back(i);
    for (size_t k = 0; k < r.repeats; ++k)
        for (size_t j = 0; j < r.block; ++j) s.push_back(r.prelude + j);
    for (size_t i = 0; i < r.coda; ++i) s.push_back(r.prelude + r.block + i);
    return s;
}

std::vector<size_t> injection_points(const LlamaConfig& cfg) {
    const auto& r = cfg.recursion;
    std::vector<size_t> at;
    if (!r.on() || r.inject == "none") return at;
    if (r.inject != "add")
        throw std::runtime_error("llama recursion: inject must be none or add, got " + r.inject);
    for (size_t k = 1; k < r.repeats; ++k) at.push_back(r.prelude + k * r.block);
    return at;
}

Llama::Llama(const LlamaConfig& cfg_, unsigned seed) : cfg(cfg_) {
    // Stored layers are built exactly as in a plain model of n_layers
    // (seed + 100 (i + 1) for layer i), so recursion with repeats 1 and no
    // prelude or coda is the plain model bit for bit.
    schedule = layer_schedule(cfg);
    inject_before.assign(schedule.size(), 0);
    for (const size_t a : injection_points(cfg)) inject_before[a] = 1;
    embed_tokens = mod<Embedding>("embed_tokens", cfg.vocab, cfg.d, seed + 11);
    for (size_t i = 0; i < cfg.n_layers; ++i)
        blocks.push_back(
            mod<LlamaBlock>("layers." + std::to_string(i), cfg, seed + 100 * (unsigned)(i + 1)));
    norm_w = reg("norm.weight", Matrix(1, cfg.d, 1.0f));
    if (!cfg.tie_embeddings) lm_head = mod<Linear>("lm_head", cfg.d, cfg.vocab, false, seed + 12);
}

Var Llama::forward(const std::vector<int>& ids, size_t seq_len, MoEAux* aux) const {
    // RoPE positions restart every seq_len (0 = one sequence).
    const size_t sl = seq_len == 0 ? ids.size() : seq_len;
    if (ids.size() % sl != 0) throw std::runtime_error("ids not a multiple of seq_len");
    std::vector<int> pos(ids.size());
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i % sl);
    // A checkpointed segment's forward records no tape, so auxiliary losses
    // built inside it could not be differentiated: refuse rather than drop them.
    if (aux && checkpoint_blocks && grad_enabled())
        throw std::runtime_error("llama: MoE auxiliary losses are not supported with checkpoint_blocks");
    Var h = embed_tokens->forward(ids);
    const Var e = h;  // the token embedding, re-added under inject "add"
    // Each application of a shared layer is its own stretch of tape over the
    // same parameter Vars, so backward accumulates the layer's gradient once
    // per application (Variable::accumulate), in reverse application order.
    for (size_t l = 0; l < schedule.size(); ++l) {
        const auto& b = blocks[schedule[l]];
        if (inject_before[l]) h = ops::add(h, e);
        h = checkpoint_blocks && grad_enabled()
                ? checkpoint(
                      [b, pos, seq_len](const Var& in) { return b->forward(in, pos, seq_len); }, h)
                : b->forward(h, pos, seq_len, aux);
    }
    h = ops::rmsnorm(h, norm_w);
    // Tied head: logits = h @ E^T (gradients flow into the embedding both
    // through the gather and through the head matmul).
    if (lm_head) return lm_head->forward(h);
    return ops::matmul(h, ops::transpose(embed_tokens->weight));
}

Var Llama::forward_cached(const std::vector<int>& ids, LlamaKV& kv) const {
    if (ids.empty()) throw std::runtime_error("llama: forward_cached needs at least one new token");
    if (kv.k.empty()) {
        kv.k.assign(schedule.size(), {});
        kv.v.assign(schedule.size(), {});
        kv.d = cfg.d;
        kv.len = 0;
    }
    if (kv.k.size() != schedule.size()) throw std::runtime_error("llama: kv cache is for another depth");
    Var h = embed_tokens->forward(ids);
    const Var e = h;
    // One cache per application (a shared layer sees different inputs on
    // each pass).
    for (size_t l = 0; l < schedule.size(); ++l) {
        if (inject_before[l]) h = ops::add(h, e);
        h = blocks[schedule[l]]->forward_cached(h, kv.len, kv.k[l], kv.v[l]);
    }
    kv.len += ids.size();
    h = ops::rmsnorm(h, norm_w);
    if (lm_head) return lm_head->forward(h);
    return ops::matmul(h, ops::transpose(embed_tokens->weight));
}

ir::json Llama::export_ir() const {
    // One block template for all layers. Every LlamaBlock is built from the
    // same config, but check: a block that differed would need a per-layer
    // override, which this exporter does not write.
    const ir::json blk = blocks.at(0)->export_ir();
    for (size_t i = 1; i < blocks.size(); ++i)
        if (blocks[i]->export_ir() != blk)
            throw std::runtime_error("llama export_ir: block " + std::to_string(i) + " differs from block 0");

    // Tied head: the exporter writes output.weight = E when it is given one
    // (mtstudio does); tied_to tells an engine where to read otherwise.
    ir::json head_linear;
    if (lm_head) {
        head_linear = lm_head->export_ir("h", "logits", "output.weight");
    } else {
        head_linear = {{"op", "linear"},       {"in", ir::json::array({"h"})},
                       {"weight", "output.weight"}, {"bias", nullptr},
                       {"tied_to", "token_embd.weight"}, {"out", "logits"}};
    }
    // A recursed model (cfg.recursion) is a different network from the
    // llama of its stored layers: its own hint, so no engine takes the llama
    // fast path for it, and n_layers / block.repeat count applications.
    // Recursion with one repeat and no sharing is the plain llama.
    const bool shared = schedule.size() != blocks.size();
    ir::json doc;
    doc["ir_version"] = ir::kVersion;
    doc["family_hint"] = shared ? "llama_recursive" : "llama";
    doc["model"] = {{"vocab", cfg.vocab},
                    {"d", cfg.d},
                    {"n_layers", schedule.size()},
                    {"ctx", cfg.n_ctx},
                    {"dtype", "f32"}};
    doc["inputs"] = ir::json::array({{{"name", "tokens"}, {"kind", "token_ids"}},
                                     {{"name", "pos"},
                                      {"kind", "positions"},
                                      {"note", "absolute positions; prefill 0..T-1, decode = cache length"}}});
    const bool inject = std::any_of(inject_before.begin(), inject_before.end(), [](char c) { return c != 0; });
    doc["embed"] = {{"nodes", ir::json::array({embed_tokens->export_ir("tokens", "x", "token_embd.weight")})}};
    // inject "add": a second gather keeps the embedding as e (EXACT, the same
    // rows), and each block starts with x = x + c_i e, c_i = 1 before the
    // first application of repeats 1.. and 0 elsewhere (affine with a
    // per-layer scale; x + 0 e is x up to the sign of zero).
    if (inject) doc["embed"]["nodes"].push_back(embed_tokens->export_ir("tokens", "e", "token_embd.weight"));
    doc["block"] = {{"repeat", schedule.size()}};
    if (shared) {
        // Application i reads its tensors from stored layer schedule[i]
        // (ir.hpp, "weight_index"); its state slot is its own.
        const auto& r = cfg.recursion;
        doc["block"]["weight_index"] = {{"per_layer", schedule}};
        doc["block"]["_recursion"] = {{"prelude", r.prelude}, {"block", r.block}, {"repeats", r.repeats},
                                      {"coda", r.coda},       {"schedule", r.schedule},
                                      {"inject", r.inject},   {"stored_layers", blocks.size()}};
    }
    doc["block"]["state"] = blk["state"];
    if (inject) {
        std::vector<float> c(inject_before.size(), 0.0f);
        for (size_t l = 0; l < c.size(); ++l) c[l] = inject_before[l] ? 1.0f : 0.0f;
        ir::json nodes = ir::json::array();
        nodes.push_back({{"op", "affine"}, {"in", ir::json::array({"e"})}, {"scale", {{"per_layer", c}}},
                         {"shift", 0.0}, {"out", "inj"}});
        nodes.push_back(ir::binary("add", "x", "inj", "x"));
        for (const auto& n : blk["nodes"]) nodes.push_back(n);
        doc["block"]["nodes"] = nodes;
    } else {
        doc["block"]["nodes"] = blk["nodes"];
    }
    const ir::json final_norm = ir::rmsnorm("x", "output_norm.weight", ops::kRmsNormEps, "h");
    doc["head"] = {{"nodes", ir::json::array({final_norm, head_linear})}};
    doc["outputs"] = ir::json::array({{{"name", "logits"}, {"kind", "logits"}}});
    doc["modes"] = {{"prefill", "all nodes over T tokens"},
                    {"decode", "the same graph with T = 1, attention reads and appends to the kv state"}};
    return doc;
}

}  // namespace nn
}  // namespace microtorch
