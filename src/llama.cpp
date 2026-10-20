#include "microtorch/llama.hpp"

#include <cmath>
#include <stdexcept>

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

Llama::Llama(const LlamaConfig& cfg_, unsigned seed) : cfg(cfg_) {
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
    for (const auto& b : blocks) {
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
    ir::json doc;
    doc["ir_version"] = ir::kVersion;
    doc["family_hint"] = "llama";
    doc["model"] = {{"vocab", cfg.vocab},
                    {"d", cfg.d},
                    {"n_layers", cfg.n_layers},
                    {"ctx", cfg.n_ctx},
                    {"dtype", "f32"}};
    doc["inputs"] = ir::json::array({{{"name", "tokens"}, {"kind", "token_ids"}},
                                     {{"name", "pos"},
                                      {"kind", "positions"},
                                      {"note", "absolute positions; prefill 0..T-1, decode = cache length"}}});
    doc["embed"] = {{"nodes", ir::json::array({embed_tokens->export_ir("tokens", "x", "token_embd.weight")})}};
    doc["block"] = {{"repeat", blocks.size()}, {"state", blk["state"]}, {"nodes", blk["nodes"]}};
    const ir::json final_norm = ir::rmsnorm("x", "output_norm.weight", ops::kRmsNormEps, "h");
    doc["head"] = {{"nodes", ir::json::array({final_norm, head_linear})}};
    doc["outputs"] = ir::json::array({{{"name", "logits"}, {"kind", "logits"}}});
    doc["modes"] = {{"prefill", "all nodes over T tokens"},
                    {"decode", "the same graph with T = 1, attention reads and appends to the kv state"}};
    return doc;
}

}  // namespace nn
}  // namespace microtorch
