#include "microtorch/llama.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "microtorch/device_cache.hpp"
#include "microtorch/ir.hpp"
#include "microtorch/llama_attnres.hpp"

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

Var LlamaBlock::forward_rows(const Var& x, const std::vector<int>& pos,
                             const std::vector<std::pair<size_t, size_t>>& segs) const {
    // forward() with the attention run per range (one sequence's active
    // tokens each), RoPE at the rows' own absolute positions.
    if (moe) throw std::runtime_error("llama: forward_rows (schedule mor) has no MoE path");
    const size_t n = x->data.rows();
    if (pos.size() != n) throw std::runtime_error("llama: forward_rows needs one position per row");
    size_t covered = 0;
    for (const auto& [s0, len] : segs) {
        if (s0 != covered || len == 0) throw std::runtime_error("llama: forward_rows ranges must tile the rows");
        covered += len;
    }
    if (covered != n) throw std::runtime_error("llama: forward_rows ranges must tile the rows");
    Var h = ops::rmsnorm(x, ln1_w);
    Var q = q_proj->forward(h);
    Var k = k_proj->forward(h);
    Var v = v_proj->forward(h);
    Var qkv_in = ops::concat_cols({q, k, v});
    for (const Var* t : {&q, &k, &v}) device::release((*t)->data);
    Var qkv = ops::apply_rope(qkv_in, pos, rope_theta_, dk, rope_all_heads_);
    device::release(qkv_in->data);
    const float sc = 1.0f / std::sqrt(static_cast<float>(dk));
    Var heads_cat;
    if (segs.size() == 1) {
        heads_cat = ops::attention_heads(qkv, H, sc, 0, /*causal=*/true);
    } else {
        // Each range attends on its own (gather its rows, causal attention,
        // then one scatter: every row is written once, a copy).
        std::vector<Var> parts;
        std::vector<std::vector<int>> where;
        for (const auto& [s0, len] : segs) {
            std::vector<int> idx(len);
            for (size_t i = 0; i < len; ++i) idx[i] = static_cast<int>(s0 + i);
            parts.push_back(ops::attention_heads(ops::gather_rows(qkv, idx), H, sc, 0, /*causal=*/true));
            where.push_back(std::move(idx));
        }
        heads_cat = ops::scatter_add_rows(parts, where, n);
    }
    Var attn = o_proj->forward(heads_cat);
    Var x1 = ops::add(x, attn);
    device::release(attn->data);
    Var h2 = ops::rmsnorm(x1, ln2_w);
    Var ffn = down_proj->forward(ops::swiglu(gate_proj->forward(h2), up_proj->forward(h2)));
    Var out = ops::add(x1, ffn);
    device::release(ffn->data);
    return out;
}

Var LlamaBlock::forward_cached(const Var& x, size_t pos0, std::vector<float>& kc, std::vector<float>& vc) const {
    const size_t n = x->data.rows();
    if (kc.size() != pos0 * x->data.cols()) throw std::runtime_error("llama: kv cache length is not pos0");
    std::vector<int> pos(n);
    for (size_t i = 0; i < n; ++i) pos[i] = static_cast<int>(pos0 + i);
    return forward_cached_rows(x, pos, kc, vc);
}

Var LlamaBlock::forward_cached_rows(const Var& x, const std::vector<int>& pos, std::vector<float>& kc,
                                    std::vector<float>& vc) const {
    const size_t n = x->data.rows(), d = x->data.cols();
    if (pos.size() != n) throw std::runtime_error("llama: forward_cached_rows needs one position per row");
    if (n > 0 && static_cast<size_t>(pos.back()) + 1 > n_ctx)
        throw std::runtime_error("llama: kv cache past the context length");
    const size_t base = kc.size() / d;  // rows cached before these
    Var h = ops::rmsnorm(x, ln1_w);
    Var qkv_in = ops::concat_cols({q_proj->forward(h), k_proj->forward(h), v_proj->forward(h)});
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
        const size_t hi = base + i + 1;  // causal: every cached row, then rows 0..i of x
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
    if (r.schedule != "fixed" && r.schedule != "mor")
        throw std::runtime_error("llama recursion: schedule \"" + r.schedule +
                                 "\" is not implemented (\"fixed\": every token takes every application; \"mor\": "
                                 "per-token depth, causal token choice)");
    if (r.repeats < 1) throw std::runtime_error("llama recursion: repeats must be >= 1");
    if (r.schedule == "mor") {
        const auto& m = r.mor;
        if (m.route != "token_choice")
            throw std::runtime_error("llama recursion mor: route must be token_choice (causal), got " + m.route);
        if (m.gate != "score" && m.gate != "none")
            throw std::runtime_error("llama recursion mor: gate must be score or none, got " + m.gate);
        if (m.kv_rows != "active")
            throw std::runtime_error("llama recursion mor: kv_rows must be active (kv_from is reserved), got " +
                                     m.kv_rows);
        if (r.inject != "none") throw std::runtime_error("llama recursion mor: inject must be none");
        if (m.force_depth > r.repeats) throw std::runtime_error("llama recursion mor: force_depth > repeats");
        if (cfg.moe.experts > 0) throw std::runtime_error("llama recursion mor: MoE is not wired");
    }
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
    // Its own seed, so every other tensor starts as in the fixed schedule.
    if (cfg.recursion.routed())
        rec_router = mod<Linear>("recursion.router", cfg.d, cfg.recursion.repeats, false, seed + 13);
    // Attention Residuals (llama_attnres.hpp): zero-initialised queries, no
    // random draw, so every other tensor starts as in the plain model.
    if (cfg.attn_res.on) {
        attnres_check(cfg);
        attn_res = mod<AttnResQueries>("attn_res", cfg.d, schedule.size());
    }
}

Var Llama::route(const Var& h, std::vector<int>& depth, std::vector<float>& margin, MoEAux* aux) const {
    const size_t N = h->data.rows(), R = cfg.recursion.repeats;
    Var logits = rec_router->forward(h);  // [N, R]
    // Host-read: the depths set every later step's row sets (as MoE's top-k).
    device::materialize(logits->data);
    depth.assign(N, 1);
    margin.assign(N, std::numeric_limits<float>::infinity());
    const size_t force = cfg.recursion.mor.force_depth;
    for (size_t t = 0; t < N; ++t) {
        size_t best = 0;
        for (size_t j = 1; j < R; ++j)
            if (logits->data(t, j) > logits->data(t, best)) best = j;  // ties: the lower index
        float second = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < R; ++j)
            if (j != best) second = std::max(second, logits->data(t, j));
        if (R > 1) margin[t] = logits->data(t, best) - second;
        depth[t] = static_cast<int>(force ? force : best + 1);
    }
    Var probs = ops::softmax_row(logits);
    std::vector<int> rows(N), cols(N);
    for (size_t t = 0; t < N; ++t) {
        rows[t] = static_cast<int>(t);
        cols[t] = depth[t] - 1;
    }
    Var g = ops::gather_elems(probs, rows, cols, 1);  // [N, 1], the chosen depth's probability
    if (aux) {
        // balance = R sum_i f_i P_i = mean_t (P coef)[t], coef_i = R f_i (a
        // constant), as MoEFFN's load-balance loss; z as its router z-loss.
        std::vector<double> frac(R, 0.0);
        for (int dd : depth) frac[static_cast<size_t>(dd - 1)] += 1.0;
        Matrix coef(R, 1);
        for (size_t i = 0; i < R; ++i) {
            frac[i] /= static_cast<double>(N);
            coef(i, 0) = static_cast<float>(static_cast<double>(R) * frac[i]);
        }
        Var lb = ops::mean(ops::matmul(probs, make_var(std::move(coef))));
        Var lse = ops::logsumexp_row(logits);
        Var z = ops::mean(ops::mul(lse, lse));
        aux->lb = aux->lb ? ops::add(aux->lb, lb) : lb;
        aux->z = aux->z ? ops::add(aux->z, z) : z;
        ++aux->layers;
        aux->frac.push_back(std::move(frac));
    }
    return g;
}

Var Llama::forward(const std::vector<int>& ids, size_t seq_len, MoEAux* aux) const {
    if (attn_res) return attnres_forward(*this, ids, seq_len, aux);
    // RoPE positions restart every seq_len (0 = one sequence).
    const size_t sl = seq_len == 0 ? ids.size() : seq_len;
    if (ids.size() % sl != 0) throw std::runtime_error("ids not a multiple of seq_len");
    std::vector<int> pos(ids.size());
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i % sl);
    // A checkpointed segment's forward records no tape, so auxiliary losses
    // built inside it could not be differentiated: refuse rather than drop them.
    if (aux && cfg.moe.experts > 0 && checkpoint_blocks && grad_enabled())
        throw std::runtime_error("llama: MoE auxiliary losses are not supported with checkpoint_blocks");
    Var h = embed_tokens->forward(ids);
    const Var e = h;  // the token embedding, re-added under inject "add"
    // Each application of a shared layer is its own stretch of tape over the
    // same parameter Vars, so backward accumulates the layer's gradient once
    // per application (Variable::accumulate), in reverse application order.
    const bool ckpt = checkpoint_blocks && grad_enabled();
    auto apply = [&](size_t l, const Var& in) {
        const auto& b = blocks[schedule[l]];
        return ckpt ? checkpoint([b, pos, seq_len](const Var& v) { return b->forward(v, pos, seq_len); }, in)
                    : b->forward(in, pos, seq_len, aux);
    };
    if (!cfg.recursion.routed()) {
        for (size_t l = 0; l < schedule.size(); ++l) {
            if (inject_before[l]) h = ops::add(h, e);
            h = apply(l, h);
        }
    } else {
        // schedule "mor" (llama.hpp): prelude, routing, the recursion steps
        // each over its active rows, coda.
        const auto& r = cfg.recursion;
        const bool scored = r.mor.gate == "score";
        const size_t N = h->data.rows();
        for (size_t l = 0; l < r.prelude; ++l) h = apply(l, h);
        std::vector<int> depth;
        std::vector<float> margin;
        Var g = route(h, depth, margin, aux);
        for (size_t k = 0; k < r.repeats; ++k) {
            std::vector<int> idx;
            for (size_t t = 0; t < N; ++t)
                if (static_cast<size_t>(depth[t]) > k) idx.push_back(static_cast<int>(t));
            if (idx.empty()) break;  // depths are nested: nobody goes deeper
            const size_t a0 = r.prelude + k * r.block;
            if (idx.size() == N) {
                // Every token: the fixed schedule's applications, unchanged.
                Var y = h;
                for (size_t j = 0; j < r.block; ++j) y = apply(a0 + j, y);
                h = scored ? ops::add(h, ops::mul_col(ops::sub(y, h), g)) : y;
                continue;
            }
            // The active rows, each sequence's in position order: one range
            // per sequence, attention within it at absolute positions.
            std::vector<int> pa(idx.size());
            std::vector<std::pair<size_t, size_t>> segs;
            size_t prev = std::numeric_limits<size_t>::max();
            for (size_t i = 0; i < idx.size(); ++i) {
                pa[i] = pos[static_cast<size_t>(idx[i])];
                const size_t seq = static_cast<size_t>(idx[i]) / sl;
                if (seq != prev) segs.emplace_back(i, 0);
                ++segs.back().second;
                prev = seq;
            }
            Var xa = ops::gather_rows(h, idx);
            Var y = xa;
            for (size_t j = 0; j < r.block; ++j) {
                const auto& b = blocks[schedule[a0 + j]];
                y = ckpt ? checkpoint([b, pa, segs](const Var& v) { return b->forward_rows(v, pa, segs); }, y)
                         : b->forward_rows(y, pa, segs);
            }
            Var dlt = ops::sub(y, xa);
            if (scored) dlt = ops::mul_col(dlt, ops::gather_rows(g, idx));
            h = ops::add(h, ops::scatter_add_rows(dlt, idx, N));
        }
        for (size_t l = r.prelude + r.repeats * r.block; l < schedule.size(); ++l) h = apply(l, h);
        last_depth = std::move(depth);
        last_margin = std::move(margin);
    }
    h = ops::rmsnorm(h, norm_w);
    // Tied head: logits = h @ E^T (gradients flow into the embedding both
    // through the gather and through the head matmul).
    if (lm_head) return lm_head->forward(h);
    return ops::matmul(h, ops::transpose(embed_tokens->weight));
}

Var Llama::forward_cached(const std::vector<int>& ids, LlamaKV& kv) const {
    if (attn_res) return attnres_forward_cached(*this, ids, kv);
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
    if (!cfg.recursion.routed()) {
        // One cache per application (a shared layer sees different inputs on
        // each pass).
        for (size_t l = 0; l < schedule.size(); ++l) {
            if (inject_before[l]) h = ops::add(h, e);
            h = blocks[schedule[l]]->forward_cached(h, kv.len, kv.k[l], kv.v[l]);
        }
    } else {
        // schedule "mor": forward()'s steps over the new rows; each
        // application's cache holds exactly the rows of the tokens active at
        // it (kv_rows "active"), with their positions in kv.pos.
        const auto& r = cfg.recursion;
        const bool scored = r.mor.gate == "score";
        const size_t n = ids.size();
        if (kv.pos.size() != schedule.size()) {
            if (kv.len != 0) throw std::runtime_error("llama: a mor kv cache needs its row positions");
            kv.pos.assign(schedule.size(), {});
        }
        std::vector<int> pall(n);
        for (size_t i = 0; i < n; ++i) pall[i] = static_cast<int>(kv.len + i);
        auto cached = [&](size_t l, const Var& x, const std::vector<int>& p) {
            Var y = blocks[schedule[l]]->forward_cached_rows(x, p, kv.k[l], kv.v[l]);
            kv.pos[l].insert(kv.pos[l].end(), p.begin(), p.end());
            return y;
        };
        for (size_t l = 0; l < r.prelude; ++l) h = cached(l, h, pall);
        std::vector<int> depth;
        std::vector<float> margin;
        Var g = route(h, depth, margin, nullptr);
        for (size_t k = 0; k < r.repeats; ++k) {
            std::vector<int> idx, pa;
            for (size_t t = 0; t < n; ++t)
                if (static_cast<size_t>(depth[t]) > k) {
                    idx.push_back(static_cast<int>(t));
                    pa.push_back(pall[t]);
                }
            if (idx.empty()) break;
            const size_t a0 = r.prelude + k * r.block;
            if (idx.size() == n) {
                Var y = h;
                for (size_t j = 0; j < r.block; ++j) y = cached(a0 + j, y, pall);
                h = scored ? ops::add(h, ops::mul_col(ops::sub(y, h), g)) : y;
                continue;
            }
            Var xa = ops::gather_rows(h, idx);
            Var y = xa;
            for (size_t j = 0; j < r.block; ++j) y = cached(a0 + j, y, pa);
            Var dlt = ops::sub(y, xa);
            if (scored) dlt = ops::mul_col(dlt, ops::gather_rows(g, idx));
            h = ops::add(h, ops::scatter_add_rows(dlt, idx, n));
        }
        for (size_t l = r.prelude + r.repeats * r.block; l < schedule.size(); ++l) h = cached(l, h, pall);
        last_depth = std::move(depth);
        last_margin = std::move(margin);
    }
    kv.len += ids.size();
    h = ops::rmsnorm(h, norm_w);
    if (lm_head) return lm_head->forward(h);
    return ops::matmul(h, ops::transpose(embed_tokens->weight));
}

ir::json Llama::export_ir() const {
    if (attn_res) return attnres_export_ir(*this);
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
    const bool routed = cfg.recursion.routed();
    if (routed && cfg.recursion.mor.force_depth)
        throw std::runtime_error("llama export_ir: force_depth is a test setting with no IR form");
    const bool shared = schedule.size() != blocks.size() || routed;
    ir::json doc;
    doc["ir_version"] = ir::kVersion;
    // A routed model is not a fixed recursion: its own hint, so an engine
    // with a llama_recursive path does not run it without the routing.
    doc["family_hint"] = routed ? "llama_mor" : shared ? "llama_recursive" : "llama";
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
    if (routed) {
        // ir.hpp depth_route: the router at step 0's first application, the
        // step of every application (-1 outside the recursion), and the kv
        // slot's rows restricted to the tokens active at each application.
        const auto& r = cfg.recursion;
        std::vector<int> step(schedule.size(), -1);
        for (size_t k = 0; k < r.repeats; ++k)
            for (size_t j = 0; j < r.block; ++j) step[r.prelude + k * r.block + j] = static_cast<int>(k);
        for (auto& st : doc["block"]["state"]) st["kv_rows"] = r.mor.kv_rows;
        doc["block"]["_recursion"]["mor"] = {{"route", r.mor.route}, {"gate", r.mor.gate},
                                             {"kv_rows", r.mor.kv_rows}};
        ir::json route = {{"op", "depth_route"},
                          {"in", ir::json::array({"x"})},
                          {"weight", "rec.0.router.weight"},
                          {"route", r.mor.route},
                          {"n_rec", r.repeats},
                          {"gate", r.mor.gate},
                          {"step", {{"per_layer", step}}},
                          {"out", "rec_depth"}};
        ir::json nodes = ir::json::array({route});
        for (const auto& n : blk["nodes"]) nodes.push_back(n);
        doc["block"]["nodes"] = nodes;
    } else if (inject) {
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
