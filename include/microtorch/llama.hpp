#pragma once
// Llama-family model (M1 completion, STUDIO_PLAN section 4): RMSNorm,
// RoPE, SwiGLU, no biases. Parameter names follow the HF LlamaForCausalLM
// convention EXACTLY ("layers.N.self_attn.q_proj.weight", ...), so:
//   - load_state_dict() ingests HF checkpoints (Qwen et al.) untranslated,
//   - gguf::export_gguf_llama(state_dict, cfg, weights_in_out=true) exports
//     a tinyllama.cpp-servable GGUF with zero name mapping.
// That closes the studio loop: train/fine-tune here -> chat there.
#include "microtorch/moe.hpp"
#include "microtorch/nn.hpp"

namespace microtorch {
namespace nn {

class AttnResQueries;  // microtorch/llama_attnres.hpp

struct LlamaConfig {
    size_t vocab = 4096, d = 128, n_layers = 2, n_heads = 4;
    size_t d_ff = 512;  // SwiGLU intermediate
    size_t n_ctx = 256;
    float rms_eps = 1e-6f;
    float rope_theta = 10000.0f;
    bool tie_embeddings = true;  // lm_head shares embed_tokens
    // RoPE on every head (spec arch.rope_heads = "all"). false reproduces
    // the legacy rotation of head 0 only ("first"), which every llama run
    // recorded without a rope_heads field was trained with.
    bool rope_all_heads = true;
    // Mixture-of-experts FFN (moe.hpp). experts == 0 (the default) is the
    // dense SwiGLU block, constructed exactly as before.
    MoEConfig moe;
    // Attention Residuals on the sublayer stream (spec arch.custom.attn_res;
    // microtorch/llama_attnres.hpp; family llama_attnres). Off by default:
    // the plain residual stream, untouched. scale "sources" (default) or
    // "none" (Kimi's convex mixture; see llama_attnres.hpp for why the
    // default rescales).
    struct AttnRes {
        bool on = false;
        std::string scale = "sources";
    } attn_res;
    // Tied-block recursion (spec arch.custom.recursion). Off (block == 0,
    // the default): every stored layer is applied once, in order, as before.
    // On: the model stores n_layers = prelude + block + coda distinct layers
    // (layers.0 .. layers.{n_layers-1}) and applies
    //   prelude layers 0 .. P-1 once each,
    //   then the block's layers P .. P+B-1 in order, `repeats` times over,
    //   then coda layers P+B .. P+B+C-1 once each,
    // so the effective depth is P + B * repeats + C with the parameters of
    // P + B + C layers. Each application has its own KV cache (its inputs
    // differ), and backward runs through every application: a shared
    // layer's gradient is the sum over its applications. "schedule" names
    // how applications are chosen: "fixed" (every token takes every
    // application) or "mor" (per-token depth, below), both over the same
    // application -> stored-layer map, layer_schedule().
    // inject "add" (an ablation; "none" is the default): the token
    // embedding e (the embedding's output, before the prelude) is added to
    // the stream before each repeat after the first, x = x + e, with no
    // parameters (as looped GPT-BERT and HRM; z_{t+1} = F(x + z_t)).
    //
    // schedule "mor" (Mixture-of-Recursions, causal token choice;
    // tiny-chat amendment 4, ir.hpp depth_route): recursion step k is the
    // block's k-th repeat, applications prelude + k*block .. + block - 1. A
    // router (linear d -> repeats, no bias; state key recursion.router.weight,
    // GGUF rec.0.router.weight) reads each token's own hidden state at the
    // input of step 0 (after the prelude): p = softmax(logits), depth n =
    // 1 + argmax over the logits (ties to the lower index), gate g = p[n-1].
    // The token takes steps 0 .. n-1, each as x <- x + g (S(x) - x) (gate
    // "score"; "none": x <- S(x), for tests), and skips the rest unchanged.
    // At every application of step k only the tokens with n > k are rows:
    // each sequence's active tokens attend causally among themselves at
    // their absolute positions (kv_rows "active"); an exited token writes no
    // key/value row there or later. A token's depth depends on its own
    // prefix only, so prefill equals decode. Auxiliary losses (through
    // MoEAux when one is passed): balance = N_r sum_i f_i P_i (f_i the
    // fraction of tokens choosing depth i, P_i the mean of p_i; 1 when
    // balanced) and z = mean_t logsumexp(logits_t)^2; their coefficients are
    // the caller's (mtstudio: mor.balance, mor.z_loss).
    struct MoR {
        std::string route = "token_choice";
        std::string gate = "score";      // "score" | "none"
        std::string kv_rows = "active";  // the only policy implemented
        // Tests only: every token takes this depth (1 .. repeats); 0 = routed.
        size_t force_depth = 0;
    };
    struct Recursion {
        size_t prelude = 0, block = 0, repeats = 1, coda = 0;
        std::string schedule = "fixed";
        std::string inject = "none";
        MoR mor;
        bool on() const { return block > 0; }
        bool routed() const { return on() && schedule == "mor"; }
    } recursion;
};

// The applications before which inject "add" adds the embedding: the first
// application of repeats 1 .. repeats-1 (empty when injection or recursion is off).
std::vector<size_t> injection_points(const LlamaConfig& cfg);

// Application a (0 <= a < effective depth) -> the stored layer it runs: the
// identity 0 .. n_layers-1 when recursion is off. Throws when recursion is
// on and n_layers != prelude + block + coda, repeats < 1, or the schedule
// is neither "fixed" nor "mor" (under "mor" the map lists every
// application a token can take; each token runs a prefix of the repeats).
std::vector<size_t> layer_schedule(const LlamaConfig& cfg);

// Keys and values of every layer application for the tokens decoded so far
// (inference only; one entry per application, so a recursed model keeps
// one cache per pass through a shared layer): k[l], v[l] are [len, d] after RoPE, rows at absolute positions
// 0..len-1. truncate() drops a suffix, so a chat turn that re-sends its
// history reuses the shared prefix.
// Under schedule "mor" an application keeps only the rows of the tokens
// active there: pos[l] lists their absolute positions (ascending), and
// truncate() keeps the rows before position n. Empty pos: row j is
// position j, as before.
struct LlamaKV {
    std::vector<std::vector<float>> k, v;  // per layer, row-major [len, d]
    std::vector<std::vector<int>> pos;     // per layer (schedule "mor" only)
    size_t len = 0, d = 0;
    void truncate(size_t n) {
        if (n >= len) return;
        for (size_t l = 0; l < k.size(); ++l) {
            size_t rows = n;
            if (l < pos.size()) {
                auto& p = pos[l];
                rows = 0;
                while (rows < p.size() && p[rows] < static_cast<int>(n)) ++rows;
                p.resize(rows);
            }
            k[l].resize(rows * d);
            v[l].resize(rows * d);
        }
        len = n;
    }
};

class LlamaBlock : public Module {
public:
    LlamaBlock(const LlamaConfig& cfg, unsigned seed);
    // seq_len 0 = single sequence; otherwise x rows are B stacked
    // sequences of seq_len, attention-isolated by a block-diagonal mask.
    // aux: MoE blocks append their routing record (ignored when dense).
    Var forward(const Var& x, const std::vector<int>& positions, size_t seq_len = 0,
                MoEAux* aux = nullptr) const;
    // Inference with a cache: x holds the rows of positions pos0.. (pos0 =
    // the cache length); their keys and values are appended to (k, v) and
    // each row attends to every cached row up to its own position. The
    // same arithmetic as forward() except the attention's summation order,
    // so logits agree to rounding, not bit for bit. Call under NoGrad.
    Var forward_cached(const Var& x, size_t pos0, std::vector<float>& k, std::vector<float>& v) const;
    // The same with explicit absolute positions (ascending, after every
    // cached row's): row i attends to the cached rows and to rows 0..i of
    // x. forward_cached(x, pos0, ..) is this with pos = pos0, pos0 + 1, ..
    // over a cache holding pos0 rows.
    Var forward_cached_rows(const Var& x, const std::vector<int>& pos, std::vector<float>& k,
                            std::vector<float>& v) const;
    // Training forward over a subset of rows (schedule "mor"): x's rows in
    // `segs` ranges (start, length), each range one sequence's tokens in
    // position order at absolute positions `pos`; attention is causal within
    // each range and never across ranges.
    Var forward_rows(const Var& x, const std::vector<int>& pos,
                     const std::vector<std::pair<size_t, size_t>>& segs) const;
    // The IR block (microtorch/ir.hpp): {"state": [kv], "nodes": [...]},
    // GGUF names templated on "{i}". Reads x and pos, writes x.
    ir::json export_ir() const;

    Var ln1_w, ln2_w;  // input/post_attention layernorm gammas
    std::shared_ptr<Linear> q_proj, k_proj, v_proj, o_proj;
    std::shared_ptr<Linear> gate_proj, up_proj, down_proj;  // dense FFN (null under MoE)
    std::shared_ptr<MoEFFN> moe;                             // MoE FFN (null when dense)
    size_t H, dk, n_ctx;
    float rope_theta_, rms_eps_;
    bool rope_all_heads_;
};

class Llama : public Module {
public:
    explicit Llama(const LlamaConfig& cfg, unsigned seed = 0);
    // ids: one sequence, or B sequences of seq_len concatenated (RoPE
    // positions restart per sequence; attention is block-isolated).
    // aux (MoE models): collects every block's routing record and auxiliary
    // losses for this forward. Not supported under checkpoint_blocks.
    Var forward(const std::vector<int>& ids, size_t seq_len = 0, MoEAux* aux = nullptr) const;
    // Logits for `ids`, the tokens after the kv.len already cached (one
    // sequence, positions kv.len..), extending kv. kv.len + ids.size() must
    // not exceed cfg.n_ctx. Call under NoGrad.
    Var forward_cached(const std::vector<int>& ids, LlamaKV& kv) const;
    // The whole model as a paperkiln IR document (microtorch/ir.hpp) over
    // the tensor names export_gguf_llama writes: embed -> block x n_layers
    // -> head. Everything but "tokenizer", which the vocabulary owns and
    // the caller adds.
    ir::json export_ir() const;

    LlamaConfig cfg;
    // Activation checkpointing per block application (see autograd.hpp).
    bool checkpoint_blocks = false;
    std::shared_ptr<Embedding> embed_tokens;
    // The stored layers (cfg.n_layers of them, registered once each, so
    // parameters() lists every tensor once and an optimiser updates it once
    // per step) and the order they are applied in (layer_schedule(cfg)).
    std::vector<std::shared_ptr<LlamaBlock>> blocks;
    std::vector<size_t> schedule;
    std::vector<char> inject_before;  // per application: add the embedding first
    Var norm_w;                       // final RMSNorm gamma
    std::shared_ptr<Linear> lm_head;  // null when tied
    // schedule "mor": the router ([d, repeats], state key
    // recursion.router.weight; null otherwise) and, after every forward or
    // forward_cached, each row's depth (1 .. repeats) and its routing margin
    // (largest logit minus the runner-up; repeats 1: +inf) for the rows that
    // call processed: the depth trace.
    std::shared_ptr<Linear> rec_router;
    mutable std::vector<int> last_depth;
    mutable std::vector<float> last_margin;
    // cfg.attn_res.on: the pseudo-queries (registered as "attn_res"; null otherwise).
    std::shared_ptr<AttnResQueries> attn_res;

private:
    // Routing for the rows of h (schedule "mor"): depths, margins, and the
    // gate column [rows, 1] (on the tape); aux collects the losses.
    Var route(const Var& h, std::vector<int>& depth, std::vector<float>& margin, MoEAux* aux) const;
};

}  // namespace nn
}  // namespace microtorch
