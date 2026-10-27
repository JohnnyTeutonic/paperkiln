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
    // Third-party attention options (keramos IR_SPEC thirdparty profile;
    // LOCAL_LLM_STACK.md section 11 item 4.4: Qwen/Gemma-style blocks). Each
    // default is the plain llama block, constructed and computed exactly as
    // before.
    //   n_kv_heads (0 = n_heads): grouped-query attention, k_proj and v_proj
    //     [d, n_kv_heads * head_dim]; query head h reads kv head h / (n_heads
    //     / n_kv_heads) (ops::repeat_heads, then the fused attention).
    //   rope_pairing "interleaved" (adjacent pairs) | "neox" (x[j], x[j +
    //     rope_dims / 2]); rope_dims (0 = head_dim): partial rotary, only each
    //     head's first rope_dims dims rotate (ops::apply_rope).
    //   qk_norm: a per-head RMSNorm on q and on k before RoPE (Qwen3, Gemma
    //     3; ops::rmsnorm_group with group head_dim), weights
    //     self_attn.q_norm.weight and self_attn.k_norm.weight [1, head_dim],
    //     GGUF blk.{i}.attn_q_norm.weight / attn_k_norm.weight.
    //   qk_temp (0 = off; needs qk_norm): a learned temperature per head,
    //     initialised to qk_temp, multiplying each head's normed q (QK-norm
    //     with a learned scale: Henry et al. 2020; Dehghani et al. 2023).
    //     Weight self_attn.q_temp.weight [1, n_heads]; the export writes it
    //     repeated over each head's columns as blk.{i}.attn_q_temp.weight [d]
    //     and an IR "scale" node on q after its norm.
    // Not wired with attn_res (llama_attnres runs its own attention).
    size_t n_kv_heads = 0;
    std::string rope_pairing = "interleaved";
    size_t rope_dims = 0;
    bool qk_norm = false;
    float qk_temp = 0.0f;
    size_t kv_heads() const { return n_kv_heads ? n_kv_heads : n_heads; }
    // Any of the options above away from its default.
    bool thirdparty() const {
        return kv_heads() != n_heads || rope_pairing != "interleaved" || (rope_dims && rope_dims != d / n_heads) ||
               qk_norm;
    }
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
    // Per-turn state tokens (spec arch.state_tokens; ir.hpp embedding
    // state_weight, attention mask "turn_state", kv slot evict "turn_state";
    // experiments/tiny_chat_long_2026-10/IR_SPEC_DRAFT.md section 1). state_k
    // == 0 (the default): none, every path exactly as before. state_k = k:
    // an id v with vocab <= v < vocab + k reads row v - vocab of a learned
    // state_embd [k, d] (state key state_embd.weight, the embedding's
    // initialiser) instead of the token embedding; the tied head covers ids
    // 0 .. vocab - 1 only, so state rows are never logits. state_mask
    // "turn_state": query p sees key j <= p iff j is a state position or in
    // p's segment (ops::turn_state_codes / attention_heads_turn_state);
    // "causal": plain causal attention over the same ids (the S-full
    // control). Ids without a state id run the plain model's ops, bit for
    // bit. Refused with schedule "mor" (the mask over each step's active
    // rows is not implemented).
    size_t state_k = 0;
    std::string state_mask = "turn_state";
    bool turn_state() const { return state_k > 0 && state_mask == "turn_state"; }
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

// Per-turn state tokens: the template contract on ids (IR_SPEC_DRAFT.md
// sections 1.2 and 1.5; tools/state_tokens.py is the same in Python).
// vocab and k as LlamaConfig::vocab / state_k; eot, user, assistant the ids
// of <|endoftext|>, <|user|>, <|assistant|>.
struct StateTemplate {
    size_t vocab = 0, k = 0;
    int eot = -1, user = -1, assistant = -1;
};
struct StateInsertCount {
    size_t blocks = 0;  // state blocks inserted
    size_t closed = 0;  // of them, after a reply that had no <|endoftext|> (one was added)
};
// The serialiser and the engine's template: an exchange is open from a
// <|user|> on, has a reply once an <|assistant|> follows it, and closes at
// the next <|endoftext|>, which then gets the k state ids vocab .. vocab +
// k - 1 after it. A <|user|> that arrives while a reply is open (the reply
// stopped early: max_new_tokens, a stop string, a cut) first gets
// <|endoftext|> and the k ids, so the next turn always starts a new
// segment. Every <|endoftext|> ends the exchange state (documents without
// exchanges get nothing). Throws on an id >= vocab (the input is text ids).
std::vector<int> insert_state_ids(const std::vector<int>& ids, const StateTemplate& t,
                                  StateInsertCount* count = nullptr);
// Raw ids (IR_SPEC_DRAFT 1.2): every id >= vocab must belong to a complete
// block vocab, .., vocab + k - 1, in that order, directly after an
// <|endoftext|> that closed an exchange. "" when the ids satisfy that, else
// the first violation.
std::string state_ids_error(const std::vector<int>& ids, const StateTemplate& t);

// Keys and values of every layer application for the tokens decoded so far
// (inference only; one entry per application, so a recursed model keeps
// one cache per pass through a shared layer): k[l], v[l] are [len, d] after RoPE, rows at absolute positions
// 0..len-1. truncate() drops a suffix, so a chat turn that re-sends its
// history reuses the shared prefix.
// Under schedule "mor" an application keeps only the rows of the tokens
// active there: pos[l] lists their absolute positions (ascending), and
// truncate() keeps the rows before position n. Empty pos: row j is
// position j, as before.
//
// State-token models (LlamaConfig::state_k > 0) also record the ids seen
// (positions 0 .. len-1), each kept row's position and turn_state code
// (ops::turn_state_codes; every layer keeps the same rows) and the code
// tracker (seg, run) at len. evict (ir.hpp kv slot "evict": "turn_state";
// mask turn_state only): once the last state position of block n is
// written, every text row of segments <= n is dropped, so after n complete
// turns each layer holds n k state rows plus the current turn's rows.
// Positions stay absolute (decode pos = tokens seen, not rows kept). The
// rows a later query sees are the same either way and are summed in the
// same order, so evicting changes no logit (bit for bit;
// tests/test_state_tokens.cpp).
struct LlamaKV {
    std::vector<std::vector<float>> k, v;  // per layer, row-major [rows, d]
    std::vector<std::vector<int>> pos;     // per layer (schedule "mor" only)
    size_t len = 0, d = 0;
    std::vector<int> ids, row_pos, row_code;  // state-token models only
    bool evict = false;
    int seg = 0, run = 0, evicted = 0;  // tracker at len; segments whose text rows are gone
    // Rows each layer holds (len unless rows were evicted).
    size_t rows() const { return d && !k.empty() ? k[0].size() / d : 0; }
    // Drop every row at position >= n and return the length kept: n, or
    // less when n falls inside segment m < evicted (a rewind into a turn
    // whose text rows were evicted): then every row of segments >= m goes
    // (the state rows of earlier segments stay), the cache ends at segment
    // m's first position, and the caller re-prefills from there. n >= len
    // keeps everything and returns len. Under schedule "mor" each layer
    // keeps its rows before position n.
    size_t truncate(size_t n);
    // State models: the code tracker after ids[0 .. n) (forward_cached and
    // truncate keep seg/run in step with it).
    void retrack(size_t n, size_t vocab, size_t k_state);
    size_t state_vocab = 0, state_k = 0;  // set by forward_cached on a state model
};

class LlamaBlock : public Module {
public:
    LlamaBlock(const LlamaConfig& cfg, unsigned seed);
    // seq_len 0 = single sequence; otherwise x rows are B stacked
    // sequences of seq_len, attention-isolated by a block-diagonal mask.
    // aux: MoE blocks append their routing record (ignored when dense).
    // ts_code (state tokens, mask turn_state): one ops::turn_state_codes
    // entry per row; null = causal.
    Var forward(const Var& x, const std::vector<int>& positions, size_t seq_len = 0,
                MoEAux* aux = nullptr, const std::vector<int>* ts_code = nullptr) const;
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
    // cached_code / new_code (state tokens, mask turn_state): the cached
    // rows' and x's rows' turn_state codes; a row then attends to the
    // visible ones only (ops::turn_state_visible), in ascending position.
    // Null: every row up to its own position (causal).
    Var forward_cached_rows(const Var& x, const std::vector<int>& pos, std::vector<float>& k,
                            std::vector<float>& v, const std::vector<int>* cached_code = nullptr,
                            const std::vector<int>* new_code = nullptr) const;
    // Training forward over a subset of rows (schedule "mor"): x's rows in
    // `segs` ranges (start, length), each range one sequence's tokens in
    // position order at absolute positions `pos`; attention is causal within
    // each range and never across ranges.
    Var forward_rows(const Var& x, const std::vector<int>& pos,
                     const std::vector<std::pair<size_t, size_t>>& segs) const;
    // The IR block (microtorch/ir.hpp): {"state": [kv], "nodes": [...]},
    // GGUF names templated on "{i}". Reads x and pos, writes x.
    ir::json export_ir() const;

    bool turn_state = false;  // attention mask turn_state (LlamaConfig::turn_state())
    Var ln1_w, ln2_w;  // input/post_attention layernorm gammas
    std::shared_ptr<Linear> q_proj, k_proj, v_proj, o_proj;
    std::shared_ptr<Linear> gate_proj, up_proj, down_proj;  // dense FFN (null under MoE)
    std::shared_ptr<MoEFFN> moe;                             // MoE FFN (null when dense)
    Var q_norm_w, k_norm_w;  // qk_norm's [1, dk] weights (null otherwise)
    Var q_temp_w;            // qk_temp's [1, H] per-head temperatures (null otherwise)
    size_t H, dk, n_ctx;
    size_t KV;          // kv heads (LlamaConfig::kv_heads())
    size_t rope_dims_;  // rotated dims per head (dk unless partial)
    float rope_theta_, rms_eps_;
    bool rope_all_heads_;
    bool rope_neox_ = false;

    // The attention's fused [T, 3d] qkv after RoPE (k and v repeated to H
    // heads under grouped-query attention): q/k/v projections of h, the
    // per-head q/k norms, the kv repeat, concat, RoPE. Without the
    // thirdparty options it is the original sequence of ops, bit for bit.
    Var attn_qkv(const Var& h, const std::vector<int>& pos) const;
};

class Llama : public Module {
public:
    explicit Llama(const LlamaConfig& cfg, unsigned seed = 0);
    // ids: one sequence, or B sequences of seq_len concatenated (RoPE
    // positions restart per sequence; attention is block-isolated).
    // aux (MoE models): collects every block's routing record and auxiliary
    // losses for this forward. Not supported under checkpoint_blocks.
    Var forward(const std::vector<int>& ids, size_t seq_len = 0, MoEAux* aux = nullptr) const;
    // forward() up to its head: the final-normed hidden states [rows, d]
    // (the same ops, bit for bit), for the fused LM head + cross-entropy
    // (ops::linear_cross_entropy with head_weight() and head_layout();
    // mtstudio's train.fused_head). forward() is this, then the head.
    Var forward_hidden(const std::vector<int>& ids, size_t seq_len = 0, MoEAux* aux = nullptr) const;
    // The head's weight in ops::linear_cross_entropy's terms: lm_head's W
    // [d, vocab] (HeadLayout::InOut) or, tied, the token embedding [vocab, d]
    // (HeadLayout::VocabMajor).
    const Var& head_weight() const { return lm_head ? lm_head->W : embed_tokens->weight; }
    ops::HeadLayout head_layout() const { return lm_head ? ops::HeadLayout::InOut : ops::HeadLayout::VocabMajor; }
    // Logits for `ids`, the tokens after the kv.len already cached (one
    // sequence, positions kv.len..), extending kv. kv.len + ids.size() must
    // not exceed cfg.n_ctx. Call under NoGrad. State models: ids may hold
    // state ids; under mask turn_state each row attends to the visible
    // cached rows and kv.evict drops finished turns' text rows after the
    // call (LlamaKV).
    Var forward_cached(const std::vector<int>& ids, LlamaKV& kv) const;
    // The input rows: the token embedding, and under state tokens the state
    // embedding's rows for state ids (one copy per row, so bit for bit the
    // gathers). Ids >= vocab + state_k are refused.
    Var embed(const std::vector<int>& ids) const;
    // The whole model as a paperkiln IR document (microtorch/ir.hpp) over
    // the tensor names export_gguf_llama writes: embed -> block x n_layers
    // -> head. Everything but "tokenizer", which the vocabulary owns and
    // the caller adds.
    ir::json export_ir() const;

    LlamaConfig cfg;
    // Activation checkpointing per block application (see autograd.hpp).
    bool checkpoint_blocks = false;
    std::shared_ptr<Embedding> embed_tokens;
    std::shared_ptr<Embedding> state_embd;  // [state_k, d]; null without state tokens
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
