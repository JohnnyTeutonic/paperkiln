// mtstudio — the run-spec driver (STUDIO_PLAN.md M1).
//
//   mtstudio run spec.json          execute the spec
//   mtstudio plan spec.json         print the resolved plan and exit
//   mtstudio export spec.json       re-export trained weights (GGUF F32 / Q8_0)
//
// One JSON describes the lifecycle; this driver executes it stage by
// stage and emits a JSONL event stream (stdout + <out>/events.jsonl) that
// the M2 UI will consume. v0 scope: arch presets + custom dims, corpus +
// GGUF vocab, train with early stopping + checkpoint/resume, safetensors
// export, GGUF export (F32 or Q8_0) for llama-family models, serve-command print.
// (arXiv arch population and the finetune stage are the documented next
// increments; papers/fetch.py already emits the config this schema takes.)
#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// cpp-httplib (third_party/httplib, MIT): the portable HTTP layer behind
// `serve` and `chat`. Included before anything that might pull in
// <windows.h> so its NOMINMAX/winsock ordering wins.
#include <nlohmann/json.hpp>
#include <regex>
#include "httplib.h"

#include "evaluation_probes.hpp"
#include "microtorch/bpe_tokenizer.hpp"
#include "microtorch/chat_template.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/gguf.hpp"
#include "microtorch/ir.hpp"
#include "microtorch/llama.hpp"
#include "microtorch/llama3_tokenizer.hpp"
#include "microtorch/llama_attnres.hpp"
#include "microtorch/posttrain/grpo.hpp"
#include "microtorch/posttrain/verifier.hpp"
#include "microtorch/safetensors.hpp"
#include "microtorch/word_tokenizer.hpp"
#include "parity_model.hpp"

using namespace microtorch;
using nlohmann::json;

namespace {

struct Spec {
    std::string name = "run";
    // arch
    std::string family = "gpt2";      // gpt2 | llama | flex | mamba
    std::string attention = "exact";  // exact | kimi | srd | recurrence (gpt2 family)
    // attention "recurrence": arch.custom.recurrence, the node's body (IR_SPEC.md 6.10: index, inputs
    // q, k, v, let, update, readout); null = the linear_attention body (ir::linear_attention_body).
    json recurrence;
    size_t d = 128, layers = 2, heads = 4, T = 128;
    // Paper-faithful flavor knobs (the flex family; empty = family default).
    std::string norm, activation, position;
    std::string residual;           // "" = family default | residual | highway | plain
    // flex only (parity::FlexConfig; LOCAL_LLM_STACK 11 item 4.2): arch.custom.activation_inputs and
    // norm_inputs (an expr slot's inputs in order: "up" / "gate" / "row" / "scalar" / "channel", or
    // {"kind": ..., "init": v}); norm_placement ("pre" | "post" | "sandwich" | "branch_post" | "none",
    // or {"attn": .., "ffn": ..}); final_norm (true / false; unset = parity::FlexConfig's rule);
    // position "t5_relative" with rel_buckets and rel_max_distance; position "shaw_relative" with
    // rel_max_distance (Shaw's clipping distance), rel_per_head and rel_values; qkv_conv (Primer's MDHA width).
    std::vector<parity::FlexConfig::ExprInput> act_inputs, norm_inputs, embed_inputs;
    std::string embed_expr;  // arch.custom.embed "expr:<e>" (parity::FlexConfig embed_expr)
    // arch.custom.embed_before_positions: the embed expr on the token embedding, before the
    // position table is added (parity::FlexConfig embed_before_pos; the Transformer's
    // sqrt(d_model)).
    bool embed_before_pos = false;
    // arch.custom.tie_head: the head reads the token embedding (parity::FlexConfig tie_head).
    bool tie_head = false;
    // arch.custom.logits "expr:<e>" (parity::FlexConfig logits_expr); the flex family's q/k norm
    // (parity::FlexConfig qk_norm "per_head" | "full"), set from arch.custom.qk_norm when the family is flex.
    std::string logits_expr, flex_qk_norm;
    std::string place_attn = "pre", place_ffn = "pre";
    int final_norm = -1;
    size_t rel_buckets = 32, rel_max_distance = 128;
    bool rel_per_head = false, rel_values = true;
    size_t qkv_conv = 0;
    bool flex_only_keys = false;  // any of the above given
    float gate_bias_init = -2.0f;   // highway only (registry #0001)
    size_t d_ff = 0;                // 0 = family default (4d gpt2/flex, 3d llama)
    size_t window = 64, sinks = 1;  // swa lane only (S1 baseline)
    size_t d_state = 64;            // mamba family: the S4 scan's state width
    // mamba family: arch.custom.ssm "s4" (the S4-style scan, mamba::MambaBlock) or "mamba1" (Mamba-1's
    // selective scan, parity::Mamba1Block: d_conv, expand, dt_rank; d_state its N).
    std::string ssm = "s4";
    size_t d_conv = 4, expand = 2, dt_rank = 0;
    parity::Mamba1Config mamba1() const {
        parity::Mamba1Config m;
        m.on = ssm == "mamba1";
        m.d_conv = d_conv;
        m.expand = expand;
        m.dt_rank = dt_rank;
        return m;
    }
    // llama family, the thirdparty attention options (nn::LlamaConfig; keramos
    // IR_SPEC thirdparty profile): arch.custom.kv_heads (0 = heads; grouped-
    // query attention), rope_pairing "interleaved" | "neox", rope_dims (0 =
    // head_dim; partial rotary), qk_norm (per-head q/k RMSNorm), rope_theta,
    // qk_temp (0 = off; with qk_norm: a learned per-head temperature on the
    // normed q, initialised to this value; nn::LlamaConfig::qk_temp).
    // Each left out = the plain llama block, untouched.
    size_t kv_heads = 0, rope_dims = 0;
    std::string rope_pairing = "interleaved";
    bool qk_norm = false;
    float qk_temp = 0.0f;
    float rope_theta = 10000.0f;
    // llama family: BitNet b1.58 BitLinear projections (arch.custom.bitlinear
    // {"weights": "ternary", "act_bits": 8, "in_norm": true}; nn::LlamaConfig::
    // bitlinear; LOCAL_LLM_WORK R3). false = no key = float projections.
    bool bitlinear = false;
    int bl_act_bits = 8;
    bool bl_in_norm = true;
    // llama family: arch.custom.tie_embeddings (default true: the head is the
    // token embedding). false: a separate lm_head.weight, as a dequantised
    // llama.cpp file with its own output.weight needs.
    bool tie_embeddings = true;
    bool thirdparty() const {
        return (kv_heads && kv_heads != heads) || rope_pairing != "interleaved" ||
               (rope_dims && rope_dims != d / heads) || qk_norm;
    }
    // llama family: RoPE on "all" heads or, legacy, the "first" head only.
    // Empty = not stated: a new run gets "all"; resuming or loading a run
    // takes what the run recorded, and a run that recorded nothing is
    // "first" (see resolve_rope_heads).
    std::string rope_heads;
    // llama family: mixture-of-experts FFN (arch.moe; include/microtorch/moe.hpp).
    // moe_experts == 0 = no arch.moe key = the dense model, untouched.
    size_t moe_experts = 0, moe_top_k = 1;
    std::string moe_gate;  // "renorm" (Mixtral) | "raw" (Switch); resolved in parse_spec
    float moe_aux = 0.01f, moe_z = 0.001f;  // load-balance and router z-loss coefficients
    // llama family: Attention Residuals on the sublayer stream (arch.custom.attn_res;
    // nn::LlamaConfig::AttnRes, microtorch/llama_attnres.hpp). false = no key = the plain stream.
    bool attn_res = false;
    std::string attn_res_scale = "sources";
    // llama family: tied-block recursion (arch.custom.recursion; nn::LlamaConfig::Recursion).
    // rec_block == 0 = no recursion key = the plain model, untouched. With it,
    // `layers` counts STORED layers (prelude + block + coda).
    size_t rec_prelude = 0, rec_block = 0, rec_repeats = 1, rec_coda = 0;
    std::string rec_schedule = "fixed";
    std::string rec_inject = "none";  // none | add (nn::LlamaConfig::Recursion::inject)
    // schedule "mor" (arch.custom.recursion.mor; tiny-chat amendment 4):
    // causal token choice, the router's balancing and z-loss coefficients.
    std::string rec_mor_route = "token_choice", rec_mor_gate = "score", rec_mor_kv = "active";
    float rec_mor_balance = 0.1f, rec_mor_z = 0.001f;
    // llama family: per-turn state tokens (arch.state_tokens {"k": 4, "mask":
    // "turn_state" | "causal"}; nn::LlamaConfig state_k / state_mask;
    // experiments/tiny_chat_long_2026-10/IR_SPEC_DRAFT.md section 1). state_k
    // == 0 = no key = the plain model, untouched.
    size_t state_k = 0;
    std::string state_mask = "turn_state";
    // data
    std::string corpus, vocab_gguf;
    size_t vocab_cap = 4096;
    size_t max_tokens = 400000;  // corpus read cap (0 = whole file)
    // data.anneal {"corpus", "from_fraction"}: training windows come from this
    // second corpus (read whole) once step > from_fraction * steps. Windows
    // are drawn at random positions, so a tail inside data.corpus would just be
    // mixed in. Validation stays on data.corpus's held-out 5%.
    std::string anneal_corpus;
    double anneal_from = 0.0;
    // post-training (docs/decisions/0001-posttrain-charter.md, Phase 3):
    // objective "lm" streams data.corpus; "sft" trains on data.sft_examples
    // ({prompt, completion} JSONL) with the prompt masked out of the loss,
    // or a whole multi-turn dialogue as {"segments": [{"text", "loss"}...]}
    // with every loss segment (each assistant turn) trained at once;
    // "rejection_sft" reads rejection-sampler records and keeps accepted ones.
    std::string objective = "lm";
    std::string sft_examples;
    // Conversation rows ({"messages"}, LOCAL_LLM_STACK.md section 3 point 3b): train.sft_loss
    // {"turns": "all" | "final" | "each"}; sft_turns_set when the spec states it (legacy rendered
    // rows then refuse, their loss being fixed by their rendering).
    std::string sft_turns = microtorch::chat::kDefaultSftTurns;
    bool sft_turns_set = false;
    // chat.template (section 3 point 3c): a registry id or a record; null = the vocabulary's
    // default (paperkiln/1 for BPE, paperkiln/legacy-word for word-level).
    json chat_template;
    bool allow_human_labels = false;
    std::string init_from;  // safetensors to start from (fresh runs only)
    // Sparse upcycling (fresh MoE runs only): a DENSE llama safetensors whose
    // FFN is copied into every expert; every other tensor is copied as is.
    std::string upcycle_from;
    // objective "grpo" (Phase 4): train.grpo.* and data.grpo_prompts /
    // data.grpo_eval ({prompt, answer, uri} JSONL). The reference model for
    // the KL term is train.init_from, frozen.
    std::string grpo_prompts, grpo_eval;
    int grpo_group = 8, grpo_prompts_per_step = 4, grpo_max_new = 8, grpo_eval_samples = 4;
    float grpo_beta = 0.05f, grpo_temp = 1.0f;
    bool grpo_std_norm = true;
    std::string grpo_reward = "exact_match";   // exact_match | format | synthetic_judge
    std::string grpo_advantage = "normal";     // normal | shuffled (the falsifier)
    json grpo_judge = json::object();           // SyntheticJudgeParams overrides
    // A deliberately buggy reward key (PT4's falsifier channel): exact match
    // against references shifted by this amount. 0 = the true references.
    int grpo_reward_shift = 0;
    // objective "provenance_po" (Phase 5): data.pref_pairs, train.po.*
    std::string pref_pairs;
    float po_beta = 0.1f;
    // uniform | provenance | shuffled (the falsifier). Default uniform: PT5
    // found the provenance table inert (WEIGHTING_INERT, 18 Oct 2026), which
    // retires it as a default; it stays available by name.
    std::string po_weight_mode = "uniform";
    json po_weights = json::object();            // evidence_kind -> weight overrides
    // RPO-style NLL on chosen completions (Pang et al. 2024), weighted by the
    // same pair weights: counters DPO's likelihood displacement. 0 = plain DPO.
    float po_nll = 0.0f;
    // train
    int steps = 500;
    float lr = 3e-3f, clip = 1.0f, lambda_gate = 0.05f;
    int eval_every = 50, ckpt_every = 100;
    // SFT evaluation: the first eval_examples held-out examples (0 = all;
    // the held-out tail of a mixed file is already a random sample), eval_batch
    // of them per forward (1 = the original one-by-one pass; the masked
    // mean is the same up to float summation order).
    size_t eval_examples = 0;
    int eval_batch = 1;
    // LM evaluation: held-out windows of T per eval (fixed seed, so every
    // eval reads the same windows). 8 = the original cadence; comparisons
    // between runs want more (8 x 512 tokens has an SE of a few hundredths).
    int eval_windows = 8;
    int batch = 1;                    // sequences per FORWARD (stacked rows, one graph)
    int accum = 1;                    // batches accumulated per optimizer step
    bool ckpt_act = false;            // activation checkpointing per block
    unsigned seed = 7;                // model init + data-order seed (Atlas multi-seed)
    std::string optimizer = "adamw";  // adamw | muon (hybrid: hidden matrices
                                      // to Muon, embeddings/vectors to AdamW)
    float muon_lr = 0.02f;
    // Mixed precision and the attention kernel (24 Oct 2026; opt-in, the
    // defaults are the FP32 tiled step bit for bit): train.precision "fp32" |
    // "bf16" (BF16 GEMM inputs, FP32 accumulate, FP32 masters; device_cache.hpp
    // Precision) and train.attention_kernel "tiled" | "flash" (ops::AttnKernel).
    std::string precision = "fp32";
    std::string attention_kernel = "tiled";
    // The fused LM head + cross-entropy (26 Oct 2026; opt-in, off by default,
    // and off is the unfused step bit for bit): train.fused_head true runs the
    // training step's head and loss as ops::linear_cross_entropy over
    // Llama::forward_hidden, never holding the [tokens, vocab] logits;
    // train.fused_head_chunk sets its chunk rows (0 = the op's default). The
    // llama family's lm and sft objectives; evaluation keeps the unfused head.
    bool fused_head = false;
    size_t fused_head_chunk = 0;
    int gradmap_every = 5;   // per-layer grad-norm event cadence
    size_t es_patience = 0;  // early stopping (0 = off)
    float es_min_delta = 0.0f;
    // Learning-rate schedule (train.lr_schedule), a pure function of the
    // step so a resumed run follows it exactly: linear warmup over
    // warmup_steps, then constant ("none") or cosine decay to
    // min_lr_frac * lr at train.steps. Absent = constant lr, as before.
    size_t lr_warmup = 0;
    std::string lr_decay = "none";
    float lr_min_frac = 0.1f;
    // export/serve
    bool exp_safetensors = true, exp_gguf = false;
    // Quantised GGUFs (formats gguf_q8_0 / gguf_q8_0_out / gguf_q8_0_all /
    // gguf_q4_0 / gguf_q4_k_m / gguf_q4_k_s): the LlamaExportConfig::quant
    // values, each written to <name>_<quant>.gguf.
    std::vector<std::string> exp_gguf_quants;
    // export.draft_table (opt-in): ember's corpus draft table for this model (EMBER_DRAFT=table, ember
    // bace837), <name>.draft_table.bin beside the GGUF, written by tools/draft_table.py from the chat text
    // the model was trained on: export.draft_table.sources (chat-format .txt or JSONL rows), else
    // data.sft_examples; at most max_bytes of each source (0: all). Unset or false: no table, the export
    // unchanged.
    bool draft_table = false;
    std::vector<std::string> draft_sources;
    size_t draft_max_bytes = 0;
    bool serve = false;
    std::string out_dir = "mtstudio_out";
    json evaluation_probes;
    json specification;
};

int evaluation_probes_stage(const Spec&, const std::function<Var(const std::vector<int>&)>&,
                            const std::vector<std::string>&, const std::map<std::string, int>&,
                            const std::string&, nn::Module&, const std::string&, int = -1);
void preflight_probe_inputs(const Spec&, const std::vector<std::string>&,
                            const std::map<std::string, int>&);

// Known presets; "custom" reads arch.custom.* instead.
const std::map<std::string, std::array<size_t, 4>> PRESETS = {
    // name -> {d, layers, heads, T}
    {"gpt2-nano", {128, 2, 4, 128}},  {"llama-tiny", {128, 2, 4, 128}},
    {"gpt2-small", {256, 4, 8, 256}}, {"kimi-tiny", {128, 2, 4, 128}},
    {"srd-tiny", {128, 2, 4, 128}},   {"attnres-tiny", {128, 2, 4, 128}},
    {"swa-tiny", {128, 2, 4, 128}},   {"mamba-tiny", {128, 2, 4, 128}},
};

// export.formats "gguf_q8_0" | "gguf_q8_0_out" | "gguf_q8_0_all" | "gguf_q4_0" |
// "gguf_q4_k_m" | "gguf_q4_k_s" -> the LlamaExportConfig::quant value. Any
// other gguf_* format is refused: a mistyped format must not quietly export
// nothing.
std::string gguf_quant_of(const std::string& fmt) {
    const std::string q = fmt.substr(5);
    if (q != "q8_0" && q != "q8_0_out" && q != "q8_0_all" && q != "q4_0" && q != "q4_k_m" && q != "q4_k_s")
        throw std::runtime_error("unknown export format " + fmt +
                                 " (gguf, gguf_q8_0, gguf_q8_0_out, gguf_q8_0_all, gguf_q4_0, gguf_q4_k_m, "
                                 "gguf_q4_k_s, safetensors)");
    return q;
}

Spec parse_spec(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open spec " + path);
    json j = json::parse(f, nullptr, true, /*ignore_comments=*/true);
    Spec s;
    s.specification = j;
    s.name = j.value("name", s.name);
    if (j.contains("evaluation_probes")) {
        studio_probes::validate_spec(j);
        s.evaluation_probes = j["evaluation_probes"];
    }

    const json arch = j.value("arch", json::object());
    if (arch.contains("preset")) {
        const std::string p = arch["preset"];
        auto it = PRESETS.find(p);
        if (it == PRESETS.end()) throw std::runtime_error("unknown preset " + p);
        s.d = it->second[0];
        s.layers = it->second[1];
        s.heads = it->second[2];
        s.T = it->second[3];
        if (p.rfind("kimi", 0) == 0) s.attention = "kimi";
        if (p.rfind("srd", 0) == 0) s.attention = "srd";
        if (p.rfind("swa", 0) == 0) s.attention = "swa";
        if (p.rfind("attnres", 0) == 0) s.attention = "attnres";
        if (p.rfind("llama", 0) == 0) s.family = "llama";
        if (p.rfind("mamba", 0) == 0) s.family = "mamba";
    }
    if (arch.contains("rope_heads")) {
        s.rope_heads = arch["rope_heads"].get<std::string>();
        if (s.rope_heads != "all" && s.rope_heads != "first")
            throw std::runtime_error("arch.rope_heads must be \"all\" or \"first\", got \"" +
                                     s.rope_heads + "\"");
    }
    if (arch.contains("custom")) {
        const json c = arch["custom"];
        s.d = c.value("d", s.d);
        s.layers = c.value("layers", s.layers);
        s.heads = c.value("heads", s.heads);
        s.attention = c.value("attention", s.attention);
        if (c.contains("recurrence")) {
            if (s.attention != "recurrence")
                throw std::runtime_error("arch.custom.recurrence needs attention \"recurrence\"");
            s.recurrence = c["recurrence"];
        }
        s.norm = c.value("norm", s.norm);
        s.activation = c.value("activation", s.activation);
        s.position = c.value("position", s.position);
        s.residual = c.value("residual", s.residual);
        auto expr_inputs = [&](const char* key, std::vector<parity::FlexConfig::ExprInput>& out) {
            if (!c.contains(key)) return;
            if (!c[key].is_array() || c[key].empty())
                throw std::runtime_error(std::string("arch.custom.") + key + " must be a non-empty array");
            for (const auto& e : c[key]) {
                parity::FlexConfig::ExprInput in;
                if (e.is_string()) {
                    in.kind = e.get<std::string>();
                } else if (e.is_object() && e.contains("kind")) {
                    in.kind = e["kind"].get<std::string>();
                    in.init = e.value("init", in.init);
                    in.init_attn = e.value("init_attn", in.init_attn);
                    in.init_ffn = e.value("init_ffn", in.init_ffn);
                    in.init_final = e.value("init_final", in.init_final);
                    for (const auto& [k, v] : e.items())
                        if (k != "kind" && k != "init" && k != "init_attn" && k != "init_ffn" && k != "init_final" &&
                            k.rfind("_", 0) != 0)
                            throw std::runtime_error(std::string("arch.custom.") + key + ": unknown key " + k +
                                                     " (kind, init, init_attn, init_ffn, init_final)");
                } else {
                    throw std::runtime_error(std::string("arch.custom.") + key +
                                             " entries are a kind or {\"kind\": .., \"init\": ..}");
                }
                out.push_back(in);
            }
            s.flex_only_keys = true;
        };
        expr_inputs("activation_inputs", s.act_inputs);
        expr_inputs("norm_inputs", s.norm_inputs);
        expr_inputs("embed_inputs", s.embed_inputs);
        if (c.contains("embed")) {
            s.embed_expr = c["embed"].get<std::string>();
            s.flex_only_keys = true;
        }
        if (c.contains("embed_before_positions")) {
            s.embed_before_pos = c["embed_before_positions"].get<bool>();
            s.flex_only_keys = true;
        }
        if (c.contains("tie_head")) {
            s.tie_head = c["tie_head"].get<bool>();
            s.flex_only_keys = true;
        }
        if (c.contains("logits")) {
            s.logits_expr = c["logits"].get<std::string>();
            s.flex_only_keys = true;
        }
        if (c.contains("norm_placement")) {
            const json& p = c["norm_placement"];
            if (p.is_string()) {
                s.place_attn = s.place_ffn = p.get<std::string>();
            } else if (p.is_object()) {
                for (const auto& [k, v] : p.items())
                    if (k != "attn" && k != "ffn" && k.rfind("_", 0) != 0)
                        throw std::runtime_error("arch.custom.norm_placement: unknown key " + k + " (attn, ffn)");
                s.place_attn = p.value("attn", s.place_attn);
                s.place_ffn = p.value("ffn", s.place_ffn);
            } else {
                throw std::runtime_error("arch.custom.norm_placement must be a placement or {\"attn\": .., \"ffn\": ..}");
            }
            if (!parity::flex_placement_ok(s.place_attn) || !parity::flex_placement_ok(s.place_ffn))
                throw std::runtime_error("arch.custom.norm_placement must be pre, post, sandwich, branch_post or none");
            s.flex_only_keys = s.flex_only_keys || s.place_attn != "pre" || s.place_ffn != "pre";
        }
        if (c.contains("final_norm")) {
            s.final_norm = c["final_norm"].get<bool>() ? 1 : 0;
            s.flex_only_keys = true;
        }
        if (c.contains("rel_buckets") || c.contains("rel_max_distance")) {
            s.rel_buckets = c.value("rel_buckets", s.rel_buckets);
            s.rel_max_distance = c.value("rel_max_distance", s.rel_max_distance);
            s.flex_only_keys = true;
        }
        if (c.contains("rel_per_head") || c.contains("rel_values")) {
            s.rel_per_head = c.value("rel_per_head", s.rel_per_head);
            s.rel_values = c.value("rel_values", s.rel_values);
            s.flex_only_keys = true;
        }
        if (c.contains("qkv_conv")) {
            s.qkv_conv = c["qkv_conv"].get<size_t>();
            s.flex_only_keys = true;
        }
        s.gate_bias_init = c.value("gate_bias_init", s.gate_bias_init);
        s.d_ff = c.value("d_ff", s.d_ff);
        s.window = c.value("window", s.window);
        s.sinks = c.value("sinks", s.sinks);
        s.d_state = c.value("d_state", s.d_state);
        s.ssm = c.value("ssm", s.ssm);
        s.d_conv = c.value("d_conv", s.d_conv);
        s.expand = c.value("expand", s.expand);
        s.dt_rank = c.value("dt_rank", s.dt_rank);
        if (s.ssm != "s4" && s.ssm != "mamba1") throw std::runtime_error("arch.custom.ssm must be s4 or mamba1, got " + s.ssm);
        if ((c.contains("ssm") || c.contains("d_conv") || c.contains("expand") || c.contains("dt_rank")) &&
            s.family != "mamba")
            throw std::runtime_error("arch.custom.ssm, d_conv, expand and dt_rank need the mamba preset");
        if (s.ssm != "mamba1" && (c.contains("d_conv") || c.contains("expand") || c.contains("dt_rank")))
            throw std::runtime_error("arch.custom.d_conv, expand and dt_rank need ssm mamba1");
        if (s.ssm == "mamba1" && (s.d_conv < 2 || s.expand < 1))
            throw std::runtime_error("arch.custom.d_conv must be >= 2 and expand >= 1");
        s.kv_heads = c.value("kv_heads", s.kv_heads);
        s.rope_pairing = c.value("rope_pairing", s.rope_pairing);
        s.rope_dims = c.value("rope_dims", s.rope_dims);
        if (c.contains("qk_norm")) {
            // true or "per_head": per head (the llama family's, or flex's); "full": over the whole q and k
            // rows, which only the flex family builds; false: none.
            const json& q = c["qk_norm"];
            const std::string m = q.is_boolean() ? (q.get<bool>() ? "per_head" : "") : q.get<std::string>();
            if (!m.empty() && m != "per_head" && m != "full")
                throw std::runtime_error("arch.custom.qk_norm must be true, false, per_head or full, got " + m);
            s.qk_norm = m == "per_head";
            if (m == "full") {
                s.flex_qk_norm = m;
                s.flex_only_keys = true;
            }
        }
        s.qk_temp = c.value("qk_temp", s.qk_temp);
        if (s.qk_temp != 0.0f && (!s.qk_norm || !(s.qk_temp > 0.0f) || !std::isfinite(s.qk_temp)))
            throw std::runtime_error(
                "arch.custom.qk_temp must be positive and needs qk_norm (finite values only)");
        s.rope_theta = c.value("rope_theta", s.rope_theta);
        s.tie_embeddings = c.value("tie_embeddings", s.tie_embeddings);
        if (c.contains("bitlinear")) {
            const json b = c["bitlinear"];
            if (!b.is_object()) throw std::runtime_error("arch.custom.bitlinear must be an object");
            for (const auto& [k, v] : b.items())
                if (k != "weights" && k != "act_bits" && k != "in_norm" && k.rfind("_", 0) != 0)
                    throw std::runtime_error("arch.custom.bitlinear: unknown key " + k +
                                             " (weights, act_bits, in_norm)");
            const std::string w = b.value("weights", std::string("ternary"));
            if (w != "ternary")
                throw std::runtime_error("arch.custom.bitlinear.weights must be \"ternary\", got " +
                                         w);
            s.bitlinear = true;
            s.bl_act_bits = b.value("act_bits", s.bl_act_bits);
            s.bl_in_norm = b.value("in_norm", s.bl_in_norm);
            if (s.bl_act_bits != 0 && (s.bl_act_bits < 2 || s.bl_act_bits > 8))
                throw std::runtime_error("arch.custom.bitlinear.act_bits must be 0 or in 2..8");
        }
        if (s.rope_pairing != "interleaved" && s.rope_pairing != "neox")
            throw std::runtime_error("arch.custom.rope_pairing must be interleaved or neox, got " + s.rope_pairing);
        if (!(s.rope_theta > 0.0f)) throw std::runtime_error("arch.custom.rope_theta must be positive");
        if (c.contains("recursion")) {
            // {"block": B, "repeats": R, "prelude": P, "coda": C, "schedule": "fixed"}:
            // B stored layers applied R times between P and C unshared ones.
            const json r = c["recursion"];
            if (!r.is_object()) throw std::runtime_error("arch.custom.recursion must be an object");
            for (const auto& [k, v] : r.items())
                if (k != "block" && k != "repeats" && k != "prelude" && k != "coda" && k != "schedule" &&
                    k != "inject" && k != "mor" && k.rfind("_", 0) != 0)
                    throw std::runtime_error("arch.custom.recursion: unknown key " + k +
                                             " (block, repeats, prelude, coda, schedule, inject, mor)");
            s.rec_block = r.value("block", size_t(0));
            s.rec_repeats = r.value("repeats", size_t(1));
            s.rec_prelude = r.value("prelude", size_t(0));
            s.rec_coda = r.value("coda", size_t(0));
            s.rec_schedule = r.value("schedule", s.rec_schedule);
            s.rec_inject = r.value("inject", s.rec_inject);
            if (s.rec_inject != "none" && s.rec_inject != "add")
                throw std::runtime_error("arch.custom.recursion.inject must be none or add, got " + s.rec_inject);
            if (s.rec_block < 1) throw std::runtime_error("arch.custom.recursion.block must be >= 1");
            if (s.rec_repeats < 1) throw std::runtime_error("arch.custom.recursion.repeats must be >= 1");
            if (s.rec_schedule != "fixed" && s.rec_schedule != "mor")
                throw std::runtime_error("arch.custom.recursion.schedule \"" + s.rec_schedule +
                                         "\" is not implemented (\"fixed\" or \"mor\")");
            if (r.contains("mor") && s.rec_schedule != "mor")
                throw std::runtime_error("arch.custom.recursion.mor needs schedule \"mor\"");
            if (s.rec_schedule == "mor") {
                // {"route": "token_choice", "gate": "score", "balance": 0.1,
                //  "z_loss": 0.001, "kv_rows": "active"} (nn::LlamaConfig::MoR).
                const json m = r.value("mor", json::object());
                if (!m.is_object()) throw std::runtime_error("arch.custom.recursion.mor must be an object");
                for (const auto& [k, v] : m.items())
                    if (k != "route" && k != "gate" && k != "balance" && k != "z_loss" && k != "kv_rows" &&
                        k.rfind("_", 0) != 0)
                        throw std::runtime_error("arch.custom.recursion.mor: unknown key " + k +
                                                 " (route, gate, balance, z_loss, kv_rows)");
                s.rec_mor_route = m.value("route", s.rec_mor_route);
                s.rec_mor_gate = m.value("gate", s.rec_mor_gate);
                s.rec_mor_kv = m.value("kv_rows", s.rec_mor_kv);
                s.rec_mor_balance = m.value("balance", s.rec_mor_balance);
                s.rec_mor_z = m.value("z_loss", s.rec_mor_z);
                if (s.rec_mor_route != "token_choice")
                    throw std::runtime_error("arch.custom.recursion.mor.route must be token_choice (causal; expert "
                                             "choice ranks tokens across the sequence and cannot decode), got " +
                                             s.rec_mor_route);
                if (s.rec_mor_gate != "score" && s.rec_mor_gate != "none")
                    throw std::runtime_error("arch.custom.recursion.mor.gate must be score or none, got " +
                                             s.rec_mor_gate);
                if (s.rec_mor_kv != "active")
                    throw std::runtime_error("arch.custom.recursion.mor.kv_rows must be active (kv_from is reserved), "
                                             "got " + s.rec_mor_kv);
                if (s.rec_mor_balance < 0.0f || s.rec_mor_z < 0.0f)
                    throw std::runtime_error("arch.custom.recursion.mor.balance and z_loss must be >= 0");
                if (s.rec_inject != "none")
                    throw std::runtime_error("arch.custom.recursion: inject add is not wired with schedule mor");
            }
            const size_t stored = s.rec_prelude + s.rec_block + s.rec_coda;
            if (c.contains("layers") && s.layers != stored)
                throw std::runtime_error("arch.custom.layers " + std::to_string(s.layers) +
                                         " counts stored layers and must equal recursion prelude + block + coda = " +
                                         std::to_string(stored));
            s.layers = stored;
        }
        if (c.contains("attn_res")) {
            // {"form": "full", "scale": "sources" | "none"}: every sublayer reads attention over
            // the embedding and all earlier sublayer outputs (zero-initialised pseudo-queries).
            const json a = c["attn_res"];
            if (!a.is_object()) throw std::runtime_error("arch.custom.attn_res must be an object");
            for (const auto& [k, v] : a.items())
                if (k != "form" && k != "scale" && k.rfind("_", 0) != 0)
                    throw std::runtime_error("arch.custom.attn_res: unknown key " + k + " (form, scale)");
            const std::string form = a.value("form", std::string("full"));
            if (form != "full")
                throw std::runtime_error("arch.custom.attn_res.form \"" + form +
                                         "\" is not implemented (\"full\": every sublayer output is a source)");
            s.attn_res = true;
            s.attn_res_scale = a.value("scale", s.attn_res_scale);
            if (s.attn_res_scale != "sources" && s.attn_res_scale != "none")
                throw std::runtime_error("arch.custom.attn_res.scale must be sources or none, got " +
                                         s.attn_res_scale);
        }
    }
    // Family resolution for the flavor knobs. RoPE with the llama block's
    // flavours (rmsnorm, swiglu, residual stream, exact attention; each
    // stated or left out) is the llama family; RoPE with any other flavour,
    // and every other flavor combination, is the flex family — the
    // paper-faithful decoder (rope on every head, as the llama block).
    if (s.family == "mamba" && (!s.norm.empty() || !s.activation.empty() || !s.position.empty() ||
                                !s.residual.empty() || s.attention != "exact" || s.flex_only_keys))
        throw std::runtime_error("the mamba preset takes no flavor knobs or attention kind (d, layers, d_state)");
    if (s.family == "mamba") {
    } else if (s.position == "rope") {
        const bool llama_block = (s.norm.empty() || s.norm == "rmsnorm") &&
                                 (s.activation.empty() || s.activation == "swiglu") &&
                                 (s.residual.empty() || s.residual == "residual") && s.attention == "exact" &&
                                 !s.flex_only_keys;
        if (llama_block) {
            s.family = "llama";
        } else {
            if (s.attention != "exact" && s.attention != "swa")
                throw std::runtime_error("position=rope with attention " + s.attention +
                                         ": the flex family takes exact or swa attention");
            if (s.rope_heads == "first")
                throw std::runtime_error("arch.rope_heads=first is the legacy llama layout; the flex "
                                         "family rotates every head");
            s.family = "flex";
        }
    } else if (!s.norm.empty() || !s.activation.empty() || !s.position.empty() ||
               !s.residual.empty() || s.flex_only_keys) {
        if (s.attention != "exact" && s.attention != "swa")
            throw std::runtime_error(
                "flavor knobs (norm/activation/position/residual) require "
                "exact or swa attention; kimi/srd/attnres are their own presets");
        s.family = "flex";
    }
    // The 2-block ParityLM cannot honor depth; flex CAN, and with
    // layernorm/gelu/learned it is the same block bit-for-bit (the
    // tests/test_flex.cpp equivalence pin). Promote rather than
    // silently truncate. kimi/srd stay 2-block and must say so.
    if (s.family == "gpt2" && s.layers != 2) {
        // exact AND swa promote to flex (deep SWA, ROADMAP 1a); kimi/srd
        // remain 2-block parity models.
        if (s.attention == "exact" || s.attention == "swa")
            s.family = "flex";
        else if (s.attention == "kimi" || s.attention == "srd" || s.attention == "recurrence")
            throw std::runtime_error(
                "kimi/srd/recurrence are 2-block parity models (layers must be 2)");
    }

    if (s.flex_only_keys && s.family != "flex")
        throw std::runtime_error(
            "arch.custom.activation_inputs, norm_inputs, embed, embed_inputs, "
            "embed_before_positions, tie_head, logits, norm_placement, "
            "final_norm, qk_norm full and rel_buckets / rel_max_distance / rel_per_head / "
            "rel_values / qkv_conv need the flex family");
    if (s.family == "flex" && s.qk_norm) {  // a per-head q/k norm on the flex family: its own, not llama's
        s.flex_qk_norm = "per_head";
        s.qk_norm = false;
    }
    if (s.attn_res && s.family != "llama")
        throw std::runtime_error("arch.custom.attn_res needs the llama family (rmsnorm, swiglu, rope, exact attention); "
                                 "the gpt2 attnres preset is its own family");
    if (s.rec_block > 0 && s.family != "llama")
        throw std::runtime_error("arch.custom.recursion needs the llama family (rmsnorm, swiglu, rope, exact attention)");
    if ((s.bitlinear || !s.tie_embeddings) && s.family != "llama")
        throw std::runtime_error("arch.custom.bitlinear and tie_embeddings need the llama family");
    if (s.bitlinear && (s.attn_res || arch.contains("moe")))
        throw std::runtime_error(
            "arch.custom.bitlinear is wired for dense llama blocks only (no moe, no attn_res)");
    // A stated RoPE base also on flex with position rope (parity::FlexConfig rope_theta; the IR rope node's theta).
    if ((s.thirdparty() || (s.rope_theta != 10000.0f && !(s.family == "flex" && s.position == "rope"))) &&
        s.family != "llama")
        throw std::runtime_error("arch.custom.kv_heads, rope_pairing, rope_dims and qk_norm need the llama family "
                                 "(rmsnorm, swiglu, rope, exact attention), rope_theta the llama family or flex with "
                                 "position rope");
    if (s.thirdparty()) {
        const size_t kv = s.kv_heads ? s.kv_heads : s.heads, hd = s.heads ? s.d / s.heads : 0;
        if (kv == 0 || s.heads % kv != 0)
            throw std::runtime_error("arch.custom.kv_heads " + std::to_string(kv) + " must divide heads " +
                                     std::to_string(s.heads));
        if (s.rope_dims && (s.rope_dims % 2 != 0 || s.rope_dims > hd))
            throw std::runtime_error("arch.custom.rope_dims must be even and at most the head width " +
                                     std::to_string(hd));
        if (s.attn_res)
            throw std::runtime_error("arch.custom.kv_heads, rope_pairing, rope_dims and qk_norm are not wired with "
                                     "attn_res (llama_attnres runs its own attention)");
        if (s.rope_heads == "first") throw std::runtime_error("arch.custom.kv_heads etc. need arch.rope_heads all");
    }
    if (arch.contains("moe")) {
        // {"experts": E, "top_k": k, "gate": "renorm"|"raw", "aux_loss": a, "z_loss": z}
        const json m = arch["moe"];
        if (!m.is_object()) throw std::runtime_error("arch.moe must be an object");
        if (s.family != "llama") throw std::runtime_error("arch.moe needs the llama family");
        s.moe_experts = m.value("experts", size_t(0));
        s.moe_top_k = m.value("top_k", size_t(1));
        // Default gate: renormalised over the top k when k >= 2 (Mixtral; the
        // upcycling convention), the raw probability when k = 1 (Switch) --
        // with k = 1 a renormalised gate is the constant 1 and the router
        // would learn from the auxiliary losses only.
        s.moe_gate = m.value("gate", std::string(s.moe_top_k >= 2 ? "renorm" : "raw"));
        s.moe_aux = m.value("aux_loss", s.moe_aux);
        s.moe_z = m.value("z_loss", s.moe_z);
        if (s.moe_experts < 1) throw std::runtime_error("arch.moe.experts must be >= 1");
        if (s.moe_top_k < 1 || s.moe_top_k > s.moe_experts)
            throw std::runtime_error("arch.moe.top_k must be in [1, experts]");
        if (s.moe_gate != "renorm" && s.moe_gate != "raw")
            throw std::runtime_error("arch.moe.gate must be \"renorm\" or \"raw\"");
        if (s.moe_aux < 0.0f || s.moe_z < 0.0f)
            throw std::runtime_error("arch.moe.aux_loss and z_loss must be >= 0");
    }

    if (arch.contains("state_tokens")) {
        const json st = arch["state_tokens"];
        if (!st.is_object()) throw std::runtime_error("arch.state_tokens must be an object");
        for (const auto& [k, v] : st.items())
            if (k != "k" && k != "mask" && k.rfind("_", 0) != 0)
                throw std::runtime_error("arch.state_tokens: unknown key " + k + " (k, mask)");
        s.state_k = st.value("k", size_t(4));
        s.state_mask = st.value("mask", s.state_mask);
        if (s.state_k < 1) throw std::runtime_error("arch.state_tokens.k must be >= 1");
        if (s.state_mask != "turn_state" && s.state_mask != "causal")
            throw std::runtime_error("arch.state_tokens.mask must be turn_state or causal, got " + s.state_mask);
        if (s.family != "llama") throw std::runtime_error("arch.state_tokens needs the llama family");
    }

    const json data = j.value("data", json::object());
    s.corpus = data.value("corpus", "");
    s.vocab_gguf = data.value("vocab", "");
    s.vocab_cap = data.value("vocab_cap", s.vocab_cap);
    s.max_tokens = data.value("max_tokens", s.max_tokens);
    s.T = data.value("T", s.T);
    s.sft_examples = data.value("sft_examples", "");
    s.grpo_prompts = data.value("grpo_prompts", "");
    s.grpo_eval = data.value("grpo_eval", "");
    s.pref_pairs = data.value("pref_pairs", "");
    if (data.contains("anneal")) {
        const json an = data["anneal"];
        s.anneal_corpus = an.value("corpus", "");
        s.anneal_from = an.value("from_fraction", 0.0);
        if (s.anneal_corpus.empty() || !(s.anneal_from > 0.0 && s.anneal_from < 1.0))
            throw std::runtime_error("data.anneal needs a corpus and from_fraction in (0, 1)");
    }

    const json tr = j.value("train", json::object());
    s.steps = tr.value("steps", s.steps);
    s.lr = tr.value("lr", s.lr);
    s.clip = tr.value("clip", s.clip);
    s.eval_every = tr.value("eval_every", s.eval_every);
    s.eval_examples = tr.value("eval_examples", s.eval_examples);
    s.eval_batch = tr.value("eval_batch", s.eval_batch);
    if (s.eval_batch < 1) throw std::runtime_error("train.eval_batch must be >= 1");
    s.eval_windows = tr.value("eval_windows", s.eval_windows);
    if (s.eval_windows < 1) throw std::runtime_error("train.eval_windows must be >= 1");
    s.ckpt_every = tr.value("checkpoint_every", s.ckpt_every);
    s.batch = tr.value("batch", s.batch);
    s.accum = tr.value("accum", s.accum);
    s.ckpt_act = tr.value("checkpoint_activations", s.ckpt_act);
    s.seed = tr.value("seed", s.seed);
    if (tr.contains("sft_loss")) {
        const json& sl = tr["sft_loss"];
        if (!sl.is_object()) throw std::runtime_error("train.sft_loss must be an object {\"turns\": ...}");
        for (auto it = sl.begin(); it != sl.end(); ++it)
            if (it.key() != "turns" && it.key().rfind("_", 0) != 0)
                throw std::runtime_error("train.sft_loss: unknown key " + it.key() + " (turns)");
        s.sft_turns = sl.value("turns", s.sft_turns);
        s.sft_turns_set = true;
        if (s.sft_turns != "all" && s.sft_turns != "final" && s.sft_turns != "each")
            throw std::runtime_error("train.sft_loss.turns must be all, final or each (got " + s.sft_turns + ")");
    }
    if (j.contains("chat")) {
        const json& ch = j["chat"];
        for (auto it = ch.begin(); it != ch.end(); ++it)
            if (it.key() != "template" && it.key().rfind("_", 0) != 0)
                throw std::runtime_error("chat: unknown key " + it.key() + " (template)");
        if (ch.contains("template")) {
            s.chat_template = ch["template"];
            // Checked now: an unknown id or field is refused before any work.
            if (s.chat_template.is_string()) microtorch::chat::get(s.chat_template.get<std::string>());
            else microtorch::chat::Template::from_json(s.chat_template);
        }
    }
    if (tr.contains("lr_schedule")) {
        // {"warmup_steps": N, "decay": "none" | "cosine", "min_lr_frac": f}
        const json ls = tr["lr_schedule"];
        if (!ls.is_object()) throw std::runtime_error("train.lr_schedule must be an object");
        s.lr_warmup = ls.value("warmup_steps", s.lr_warmup);
        s.lr_decay = ls.value("decay", s.lr_decay);
        s.lr_min_frac = ls.value("min_lr_frac", s.lr_min_frac);
        if (s.lr_decay != "none" && s.lr_decay != "cosine")
            throw std::runtime_error("train.lr_schedule.decay must be \"none\" or \"cosine\"");
        if (s.lr_min_frac < 0.0f || s.lr_min_frac > 1.0f)
            throw std::runtime_error("train.lr_schedule.min_lr_frac must be in [0, 1]");
    }
    s.objective = tr.value("objective", s.objective);
    s.allow_human_labels = tr.value("allow_human_labels", s.allow_human_labels);
    s.init_from = tr.value("init_from", "");
    s.upcycle_from = tr.value("upcycle_from", "");
    if (!s.upcycle_from.empty()) {
        if (s.moe_experts == 0) throw std::runtime_error("train.upcycle_from needs arch.moe (the MoE to upcycle into)");
        if (!s.init_from.empty()) throw std::runtime_error("train.upcycle_from and train.init_from are exclusive");
    }
    if (s.objective != "lm" && s.objective != "sft" && s.objective != "rejection_sft" && s.objective != "grpo" &&
        s.objective != "provenance_po")
        throw std::runtime_error("train.objective must be lm, sft, rejection_sft, grpo or provenance_po (got " +
                                 s.objective + ")");
    if (s.objective == "provenance_po") {
        const json po = tr.value("po", json::object());
        s.po_beta = po.value("beta", s.po_beta);
        s.po_weight_mode = po.value("weight_mode", s.po_weight_mode);
        s.po_weights = po.value("weights", json::object());
        s.po_nll = po.value("nll", s.po_nll);
        if (s.po_nll < 0.0f) throw std::runtime_error("train.po.nll must be >= 0");
        if (s.init_from.empty() || s.pref_pairs.empty())
            throw std::runtime_error("objective provenance_po needs train.init_from (the reference) and data.pref_pairs");
        if (s.po_weight_mode != "provenance" && s.po_weight_mode != "uniform" && s.po_weight_mode != "shuffled")
            throw std::runtime_error("train.po.weight_mode must be provenance, uniform or shuffled");
    }
    if ((s.objective == "sft" || s.objective == "rejection_sft") && s.sft_examples.empty())
        throw std::runtime_error("train.objective " + s.objective + " needs data.sft_examples");
    if (s.objective == "grpo") {
        const json g = tr.value("grpo", json::object());
        s.grpo_group = g.value("group_size", s.grpo_group);
        s.grpo_prompts_per_step = g.value("prompts_per_step", s.grpo_prompts_per_step);
        s.grpo_max_new = g.value("max_new", s.grpo_max_new);
        s.grpo_eval_samples = g.value("eval_samples", s.grpo_eval_samples);
        s.grpo_beta = g.value("beta", s.grpo_beta);
        s.grpo_temp = g.value("temperature", s.grpo_temp);
        s.grpo_std_norm = g.value("std_norm", s.grpo_std_norm);
        s.grpo_reward = g.value("reward", s.grpo_reward);
        s.grpo_advantage = g.value("advantage", s.grpo_advantage);
        s.grpo_judge = g.value("judge", json::object());
        s.grpo_reward_shift = g.value("reward_shift", 0);
        if (s.grpo_reward_shift != 0 && s.grpo_reward != "exact_match")
            throw std::runtime_error("train.grpo.reward_shift applies to the exact_match reward only");
        if (s.init_from.empty() || s.grpo_prompts.empty())
            throw std::runtime_error("objective grpo needs train.init_from (the reference) and data.grpo_prompts");
        if (s.grpo_group < 2) throw std::runtime_error("train.grpo.group_size must be >= 2");
        if (!(s.grpo_temp > 0.0f)) throw std::runtime_error("train.grpo.temperature must be > 0");
    }
    s.optimizer = tr.value("optimizer", s.optimizer);
    s.muon_lr = tr.value("muon_lr", s.muon_lr);
    s.precision = tr.value("precision", s.precision);
    s.attention_kernel = tr.value("attention_kernel", s.attention_kernel);
    if (s.precision != "fp32" && s.precision != "bf16")
        throw std::runtime_error("train.precision must be fp32 or bf16 (got " + s.precision + ")");
    if (s.attention_kernel != "tiled" && s.attention_kernel != "flash")
        throw std::runtime_error("train.attention_kernel must be tiled or flash (got " + s.attention_kernel + ")");
    // Refuse what would be ignored: only the llama family's attention runs
    // through ops::attention_heads, and only its projections are checked
    // under bf16 (tests/test_mixed_precision.cpp).
    if ((s.precision != "fp32" || s.attention_kernel != "tiled") && s.family != "llama")
        throw std::runtime_error("train.precision bf16 and train.attention_kernel flash are wired for the llama family only");
    if (s.attention_kernel == "flash" && (s.heads == 0 || s.d % s.heads != 0 ||
                                          !device::devops::flash_attention_supported(s.d / s.heads)))
        throw std::runtime_error("train.attention_kernel flash supports head widths up to 64 (d / heads)");
    s.fused_head = tr.value("fused_head", s.fused_head);
    s.fused_head_chunk = tr.value("fused_head_chunk", s.fused_head_chunk);
    if (tr.contains("fused_head_chunk") && !s.fused_head)
        throw std::runtime_error("train.fused_head_chunk needs train.fused_head true");
    if (s.fused_head && s.family != "llama")
        throw std::runtime_error("train.fused_head is wired for the llama family only");
    if (s.fused_head && (s.objective == "grpo" || s.objective == "provenance_po"))
        throw std::runtime_error("train.fused_head covers the lm and sft objectives (grpo and provenance_po read the logits)");
    s.gradmap_every = tr.value("gradmap_every", s.gradmap_every);
    if (tr.contains("early_stopping")) {
        s.es_patience = tr["early_stopping"].value("patience", size_t(0));
        s.es_min_delta = tr["early_stopping"].value("min_delta", 0.0f);
    }

    const json ex = j.value("export", json::object());
    for (const auto& fmt : ex.value("formats", std::vector<std::string>{"safetensors"})) {
        if (fmt == "gguf") s.exp_gguf = true;
        if (fmt == "safetensors") s.exp_safetensors = true;
        if (fmt.rfind("gguf_", 0) == 0) s.exp_gguf_quants.push_back(gguf_quant_of(fmt));
    }
    if (ex.contains("draft_table")) {
        const json& dt = ex["draft_table"];
        if (dt.is_boolean()) {
            s.draft_table = dt.get<bool>();
        } else if (dt.is_object()) {
            s.draft_table = true;
            for (auto it = dt.begin(); it != dt.end(); ++it) {
                if (it.key() == "sources" && it.value().is_array()) {
                    s.draft_sources = it.value().get<std::vector<std::string>>();
                } else if (it.key() == "max_bytes" && it.value().is_number_unsigned()) {
                    s.draft_max_bytes = it.value().get<size_t>();
                } else {
                    throw std::runtime_error("export.draft_table: " + it.key() +
                                             " is not a key it takes (sources: a list of paths; max_bytes: a "
                                             "non-negative integer)");
                }
            }
        } else {
            throw std::runtime_error("export.draft_table must be true, false or {\"sources\": [...], \"max_bytes\": N}");
        }
    }
    s.serve =j.value("serve", json::object()).value("on_finish", false);
    s.out_dir = j.value("out_dir", s.out_dir);
    if (s.moe_experts > 0) {
        // The auxiliary losses ride the lm/sft loss; the RL objectives and
        // activation checkpointing do not carry them yet (refuse, never drop).
        if (s.objective == "grpo" || s.objective == "provenance_po")
            throw std::runtime_error("arch.moe: objective " + s.objective + " is not wired for MoE yet (lm, sft, rejection_sft)");
        if (s.ckpt_act)
            throw std::runtime_error("arch.moe: train.checkpoint_activations is not supported with MoE yet");
        if (s.rec_block > 0) throw std::runtime_error("arch.moe with arch.custom.recursion is not wired yet");
    }
    if (s.state_k > 0) {
        // Refuse what is not wired rather than train or serve it wrongly.
        if (s.moe_experts > 0) throw std::runtime_error("arch.state_tokens with arch.moe is not wired yet");
        if (s.attn_res) throw std::runtime_error("arch.state_tokens with arch.custom.attn_res is not wired yet");
        if (s.rec_block > 0 && s.rec_schedule == "mor")
            throw std::runtime_error("arch.state_tokens with recursion schedule mor is not wired (the turn_state mask "
                                     "over each step's active rows); schedule fixed is");
        if (s.objective == "grpo" || s.objective == "provenance_po")
            throw std::runtime_error("arch.state_tokens: objective " + s.objective +
                                     " is not wired (its rows get no state ids); lm, sft and rejection_sft are");
        for (const auto& q : s.exp_gguf_quants)
            if (q == "q4_0" || q == "q4_k_m" || q == "q4_k_s")
                throw std::runtime_error("export gguf_" + q + ": llama.cpp's mix has no state embedding "
                                         "(gguf and gguf_q8_0* only)");
    }
    if (s.rec_block > 0)
        for (const auto& q : s.exp_gguf_quants)
            if (q == "q4_0" || q == "q4_k_m" || q == "q4_k_s")
                // Those mixes are defined as llama.cpp's quantiser's output, and
                // a recursed model is a "paperkiln" file llama.cpp does not read.
                throw std::runtime_error("export gguf_" + q + ": llama.cpp's mix has no reference for a recursed "
                                         "model (gguf and gguf_q8_0* only)");
    if (s.attn_res) {
        // nn::attnres_check refuses the same at construction; plan says so first.
        if (s.moe_experts > 0) throw std::runtime_error("arch.custom.attn_res with arch.moe is not wired");
        if (s.rec_block > 0 && s.rec_schedule == "mor")
            throw std::runtime_error("arch.custom.attn_res with recursion schedule mor is not implemented (the bank's "
                                     "per-row presence, IR_SPEC_DRAFT section 4, is exploratory); fixed recursion "
                                     "is supported, one pair of queries per application");
        if (s.rec_block > 0 && s.rec_inject != "none")
            throw std::runtime_error("arch.custom.attn_res: recursion inject add has no residual stream to add to");
        const size_t apps = s.rec_block > 0 ? s.rec_prelude + s.rec_block * s.rec_repeats + s.rec_coda : s.layers;
        if (2 * apps + 1 > 128)
            throw std::runtime_error("arch.custom.attn_res: " + std::to_string(2 * apps + 1) +
                                     " depth sources exceed the depth_attend kernel's 128");
        for (const auto& q : s.exp_gguf_quants)
            if (q == "q4_0" || q == "q4_k_m" || q == "q4_k_s")
                throw std::runtime_error("export gguf_" + q + ": llama.cpp's mix has no reference for an "
                                         "attention-residual model (gguf and gguf_q8_0* only)");
    }
    if (s.draft_table) {
        if (!s.exp_gguf && s.exp_gguf_quants.empty())
            throw std::runtime_error("export.draft_table needs a GGUF export (export.formats gguf or gguf_*): the "
                                     "table is written beside it, from its vocabulary");
        if (s.draft_sources.empty() && s.sft_examples.empty())
            throw std::runtime_error("export.draft_table needs the chat text the model was trained on: "
                                     "data.sft_examples, or export.draft_table.sources");
    }
    if (!s.evaluation_probes.is_null() && !s.exp_safetensors)
        throw std::runtime_error("evaluation_probes requires final safetensors export");
    return s;
}

// The spec's attention residuals onto a llama config (no-op without arch.custom.attn_res).
void apply_attn_res(nn::LlamaConfig& lc, const Spec& s) {
    lc.attn_res.on = s.attn_res;
    lc.attn_res.scale = s.attn_res_scale;
    // Called at every llama construction site, so the head tying and the
    // BitLinear option (arch.custom.tie_embeddings / bitlinear) ride along.
    lc.tie_embeddings = s.tie_embeddings;
    lc.bitlinear.on = s.bitlinear;
    lc.bitlinear.act_bits = s.bl_act_bits;
    lc.bitlinear.in_norm = s.bl_in_norm;
}

// The spec's thirdparty attention options and RoPE base onto a llama config
// (no-op at the defaults).
void apply_thirdparty(nn::LlamaConfig& lc, const Spec& s) {
    lc.n_kv_heads = s.kv_heads;
    lc.rope_pairing = s.rope_pairing;
    lc.rope_dims = s.rope_dims;
    lc.qk_norm = s.qk_norm;
    lc.qk_temp = s.qk_temp;
    lc.rope_theta = s.rope_theta;
}

// The spec's MoE settings onto a llama config (no-op for a dense spec).
void apply_moe(nn::LlamaConfig& lc, const Spec& s) {
    lc.moe.experts = s.moe_experts;
    lc.moe.top_k = s.moe_top_k;
    lc.moe.renormalize = s.moe_gate != "raw";
}

// The spec's recursion onto a llama config (no-op without arch.custom.recursion).
void apply_recursion(nn::LlamaConfig& lc, const Spec& s) {
    lc.recursion.block = s.rec_block;
    lc.recursion.repeats = s.rec_repeats;
    lc.recursion.prelude = s.rec_prelude;
    lc.recursion.coda = s.rec_coda;
    lc.recursion.schedule = s.rec_schedule;
    lc.recursion.inject = s.rec_inject;
    lc.recursion.mor.route = s.rec_mor_route;
    lc.recursion.mor.gate = s.rec_mor_gate;
    lc.recursion.mor.kv_rows = s.rec_mor_kv;
}

// The spec's state tokens onto a llama config (no-op without arch.state_tokens).
void apply_state(nn::LlamaConfig& lc, const Spec& s) {
    lc.state_k = s.state_k;
    lc.state_mask = s.state_mask;
}

// The chat template's ids the state-token serialiser needs (nn::insert_state_ids).
nn::StateTemplate state_template(const Spec& s, const std::map<std::string, int>& vocab) {
    nn::StateTemplate t;
    t.vocab = vocab.size();
    t.k = s.state_k;
    auto id = [&](const char* name) {
        const auto it = vocab.find(name);
        if (it == vocab.end())
            throw std::runtime_error(std::string("arch.state_tokens needs ") + name +
                                     " in the vocabulary (the chat template's specials)");
        return it->second;
    };
    t.eot = id("<|endoftext|>");
    t.user = id("<|user|>");
    t.assistant = id("<|assistant|>");
    return t;
}

// nn::insert_state_ids carrying a per-id flag along (a loss flag, or "is a
// continuation token"): inserted ids get 0. The insertions are an
// <|endoftext|> before a <|user|> and the state ids, so the greedy match
// below pairs every original id with its copy.
std::vector<int> insert_state_flagged(const std::vector<int>& ids, const std::vector<uint8_t>& flag,
                                      const nn::StateTemplate& t, std::vector<uint8_t>& out_flag,
                                      nn::StateInsertCount* count = nullptr) {
    std::vector<int> out = nn::insert_state_ids(ids, t, count);
    out_flag.assign(out.size(), 0);
    size_t j = 0;
    for (size_t i = 0; i < out.size(); ++i)
        if (j < ids.size() && out[i] == ids[j]) out_flag[i] = flag[j++];
    if (j != ids.size()) throw std::runtime_error("insert_state_flagged: ids lost in the insertion");
    return out;
}

// Language-model rows under state tokens: no loss where the target is a
// state id or the input is a state position (its target is the template's
// next <|user|>), IR_SPEC_DRAFT section 1.5. Masked targets become 0 (in range).
void state_loss_mask(const std::vector<int>& x, std::vector<int>& y, std::vector<uint8_t>& mask, size_t V) {
    mask.assign(y.size(), 1);
    for (size_t i = 0; i < y.size(); ++i)
        if (static_cast<size_t>(x[i]) >= V || static_cast<size_t>(y[i]) >= V) {
            mask[i] = 0;
            if (static_cast<size_t>(y[i]) >= V) y[i] = 0;
        }
}

// schedule "mor": per-token depth (nn::LlamaConfig::MoR).
bool routed(const Spec& s) { return s.rec_block > 0 && s.rec_schedule == "mor"; }

// SFT evaluation inside a step window: exactly when the training steps defer
// (so the forward is one this run already runs under deferral), unless
// MICROTORCH_DEVICE_EVAL=0. Always false on a CPU build.
bool sft_eval_on_device() {
    const char* v = std::getenv("MICROTORCH_DEVICE_EVAL");
    if (v && std::strcmp(v, "0") == 0) return false;
    return device::get() == device::Device::CUDA && device::device_ops_enabled() &&
           device::step_residency_enabled() && device::defer_downloads_enabled();
}

// Layer applications per forward: prelude + block * repeats + coda, or layers.
size_t effective_layers(const Spec& s) {
    return s.rec_block > 0 ? s.rec_prelude + s.rec_block * s.rec_repeats + s.rec_coda : s.layers;
}

// ---- events: JSONL to stdout + file (the M2 UI's feed) ----
struct Events {
    std::ofstream file;
    explicit Events(const std::string& path) : file(path, std::ios::app) {}
    void emit(const json& j) {
        const std::string line = j.dump();
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
        file << line << "\n";
        file.flush();
    }
};

// On resume, drop event lines past the checkpoint step. events.jsonl is
// append-only and a killed run has usually logged steps beyond its last
// checkpoint; without this the resumed run appends a second copy of those
// steps and every downstream reader sees duplicates (seen on the 4 Sep
// 2026 resume probe: 209 step lines by step 140). Lines without a step
// field (start, resume history) are kept.
void truncate_events_after(const std::string& path, int step) {
    std::ifstream in(path);
    if (!in.good()) return;
    std::vector<std::string> keep;
    std::string line;
    while (std::getline(in, line)) {
        const json e = json::parse(line, nullptr, false);
        if (!e.is_discarded() && e.contains("step") && e["step"].is_number() &&
            e["step"].get<int>() > step)
            continue;
        keep.push_back(line);
    }
    in.close();
    std::ofstream out(path, std::ios::trunc);
    for (const auto& l : keep) out << l << "\n";
}

// What a run directory records about RoPE head coverage: the last
// "model" event's rope_heads, where a model event without the field is a
// run from before the field existed ("first"); failing any model event,
// out_dir/spec.json's arch.rope_heads; "" when nothing is recorded.
std::string recorded_rope_heads(const std::string& out_dir) {
    std::string rec;
    bool saw_model = false;
    std::ifstream in(out_dir + "/events.jsonl");
    std::string line;
    while (std::getline(in, line)) {
        const json e = json::parse(line, nullptr, false);
        if (e.is_discarded() || !e.is_object() || e.value("event", "") != "model") continue;
        saw_model = true;
        rec = e.contains("rope_heads") && e["rope_heads"].is_string()
                  ? e["rope_heads"].get<std::string>()
                  : "first";
    }
    if (saw_model) return rec;
    std::ifstream sf(out_dir + "/spec.json");
    if (!sf) return "";
    const json j = json::parse(sf, nullptr, false, /*ignore_comments=*/true);
    if (j.is_discarded() || !j.is_object() || !j.contains("arch") || !j["arch"].is_object())
        return "";
    const json& a = j["arch"];
    return a.contains("rope_heads") && a["rope_heads"].is_string()
               ? a["rope_heads"].get<std::string>()
               : "";
}

// The RoPE head coverage a run is built with. `existing` = the out_dir
// holds a trained or resumable model (resume, sample, chat, serve):
// then the run's own record decides, and a run that recorded nothing is
// legacy ("first"). A fresh run takes the spec's value, else "all". A
// spec that states a value contradicting the record is refused rather
// than silently loading weights into the other architecture.
std::string resolve_rope_heads(const Spec& s, bool existing) {
    if (!existing) return s.rope_heads.empty() ? "all" : s.rope_heads;
    std::string rec = recorded_rope_heads(s.out_dir);
    if (rec.empty()) rec = s.rope_heads.empty() ? "first" : s.rope_heads;
    if (!s.rope_heads.empty() && s.rope_heads != rec)
        throw std::runtime_error("spec says arch.rope_heads=\"" + s.rope_heads + "\" but " +
                                 s.out_dir + " was trained with \"" + rec +
                                 "\" (runs that recorded no rope_heads are \"first\")");
    return rec;
}

// GGUF vocab reader + word tokenizer (the srd_parity path).
std::vector<std::string> read_gguf_vocab(const std::string& path);
// A BPE vocabulary's text tokenizer: paperkiln's own (tokenizer.ggml.model
// "bpe", tools/bpe.py) or the llama.cpp form (model "gpt2", pre "llama-bpe",
// microtorch/llama3_tokenizer.hpp), with what an export must carry.
struct TextBpe {
    std::string model, pre;
    std::vector<int32_t> token_type;
    std::function<std::vector<int>(const std::string&)> encode;
    std::function<std::string(const std::vector<int>&)> decode;
};
// Set by read_gguf_vocab for a BPE vocabulary. Null for word-level
// vocabularies, whose code paths are unchanged.
std::shared_ptr<TextBpe> g_bpe;
std::string g_tokenizer_model;
std::string ids_to_text(const std::vector<int>& ids, const std::vector<std::string>& tokens);
std::vector<int> tokenize(const std::string& text, const std::map<std::string, int>& vocab,
                          size_t max_tokens);
std::vector<int> read_corpus(const std::string& path, const std::map<std::string, int>& vocab, size_t n_vocab,
                             size_t max_tokens);

// The chat template a model renders conversations with (LOCAL_LLM_STACK.md section 3 point 3;
// microtorch/chat_template.hpp): the spec's chat.template, else paperkiln/1 for a BPE vocabulary
// and paperkiln/legacy-word for a word-level one (read_gguf_vocab has run). Per-turn state tokens
// are defined over paperkiln/1's markers only (ir.hpp).
chat::Template chat_template_of(const Spec& s) {
    chat::Template t = s.chat_template.is_string() ? chat::get(s.chat_template.get<std::string>())
                       : s.chat_template.is_object() ? chat::Template::from_json(s.chat_template)
                       : g_bpe ? chat::v1()
                               : chat::legacy_word();
    if (s.state_k > 0 && t.id() != chat::kV1)
        throw std::runtime_error("arch.state_tokens is defined over the paperkiln/1 template only (chat.template is " +
                                 t.id() + ")");
    // A template with tool turns (paperkiln/2) needs each control string as one control token of the
    // vocabulary (tools/chat_v2_vocab.py vocab): the llama.cpp form, whose tokeniser parses them out.
    if (t.has_tool && g_bpe) {
        if (g_bpe->model != "gpt2")
            throw std::runtime_error("chat template " + t.id() + " needs a llama.cpp-form vocabulary with its control "
                                     "tokens (tools/chat_v2_vocab.py vocab)");
        for (const auto& r : t.reserved) {
            const auto ids = g_bpe->encode(r);
            if (ids.size() != 1 || ids[0] < 0 || static_cast<size_t>(ids[0]) >= g_bpe->token_type.size() ||
                g_bpe->token_type[static_cast<size_t>(ids[0])] != 3)
                throw std::runtime_error("chat template " + t.id() + " needs '" + r +
                                         "' as a control token of the vocabulary (tools/chat_v2_vocab.py vocab)");
        }
    }
    return t;
}

// Whether exports declare the template (paperkiln.chat_template, tokenizer.chat_template and a
// Modelfile): a spec that names one, or a vocabulary whose ids 0-3 are paperkiln's specials.
// Word-level vocabularies write no keys.
bool declares_chat_template(const Spec& s, const std::vector<std::string>& tokens) {
    if (!s.chat_template.is_null()) return true;
    if (!g_bpe || tokens.size() < 4) return false;
    for (size_t i = 0; i < 4; ++i)
        if (tokens[i] != chat::kV1Specials[i]) return false;
    return true;
}

parity::AttnKind attn_kind(const std::string& s) {
    if (s == "kimi") return parity::AttnKind::KIMI;
    if (s == "srd") return parity::AttnKind::SRD;
    if (s == "swa") return parity::AttnKind::SWA;
    if (s == "recurrence") return parity::AttnKind::RECURRENCE;
    return parity::AttnKind::EXACT;
}

// The gpt2-family parity model; attention "recurrence" passes arch.custom.recurrence (the node's
// body, IR_SPEC.md 6.10) as the IR's ordered json, or nothing for the default body.
std::shared_ptr<parity::ParityLM> make_parity(const Spec& s, size_t vocab) {
    const ir::json body = s.recurrence.is_null() ? ir::json() : ir::json::parse(s.recurrence.dump());
    return std::make_shared<parity::ParityLM>(attn_kind(s.attention), vocab, s.d, s.heads, s.T, s.seed, s.window,
                                              s.sinks, s.recurrence.is_null() ? nullptr : &body);
}

// Per-module L2 gradient norms, grouped by the first dotted-path segment
// ("wte", "attn_0", "mlp_1", ...). This is the data the M2 node-graph
// glows with: fading nodes = vanishing gradients, flashing = exploding.
json grad_map(const nn::Module& m) {
    std::map<std::string, double> sq;
    for (const auto& [name, p] : m.named_parameters()) {
        if (p->grad.rows() == 0) continue;
        // Group at the first segment — except structural containers
        // ("layers.N", "h.N", "blocks.N"), which keep their index so the
        // node graph gets per-block resolution instead of one blob.
        auto cut = name.find('.');
        std::string group = cut == std::string::npos ? name : name.substr(0, cut);
        if ((group == "layers" || group == "h" || group == "blocks") && cut != std::string::npos) {
            const auto cut2 = name.find('.', cut + 1);
            group = cut2 == std::string::npos ? name : name.substr(0, cut2);
        }
        double acc = 0;
        for (size_t i = 0; i < p->grad.rows(); ++i)
            for (size_t j = 0; j < p->grad.cols(); ++j)
                acc += static_cast<double>(p->grad(i, j)) * p->grad(i, j);
        sq[group] += acc;
    }
    json out = json::object();
    for (const auto& [k, v] : sq) out[k] = std::sqrt(v);
    return out;
}

// The frozen reference model for grpo's KL term: the spec's architecture
// with the weights of train.init_from. Defined after LoadedLM.
std::function<Var(const std::vector<int>&, size_t)> make_reference(const Spec& s);

// One pure-temperature completion (charter 0001 §4): the whole vocabulary,
// no top-k, no nucleus, no banned tokens, so the draw is from the policy
// itself. `forward` maps a context (at most T ids) to logits. Shared by
// `sample-set` and the grpo objective, so the samples a run is trained on
// and the samples it is evaluated on come from the same code.
struct Sampled {
    std::vector<int> ids;
    double logp = 0.0, logp_sample = 0.0;  // at temperature 1 and at the sampling temperature
    std::string stop = "length";
};

template <class F>
Sampled sample_pure(F&& forward, const std::vector<int>& prompt, size_t T, size_t V, float temp, int max_new, int eos,
                    std::mt19937& gen) {
    Sampled out;
    std::vector<int> ids = prompt;
    for (int t = 0; t < max_new; ++t) {
        std::vector<int> ctx = ids;
        if (ctx.size() > T) ctx.assign(ids.end() - T, ids.end());
        Var logits = forward(ctx);
        const size_t last = logits->data.rows() - 1;
        double m1 = -1e300, mt = -1e300;
        for (size_t j = 0; j < V; ++j) {
            const double x = logits->data(last, j);
            m1 = std::max(m1, x);
            mt = std::max(mt, x / temp);
        }
        double z1 = 0.0, zt = 0.0;
        std::vector<double> p(V);
        for (size_t j = 0; j < V; ++j) {
            const double x = logits->data(last, j);
            z1 += std::exp(x - m1);
            zt += (p[j] = std::exp(x / temp - mt));
        }
        std::uniform_real_distribution<double> u(0.0, zt);
        double r = u(gen);
        size_t pick = V - 1;
        for (size_t j = 0; j < V; ++j)
            if ((r -= p[j]) <= 0) {
                pick = j;
                break;
            }
        const double x = logits->data(last, pick);
        out.logp += x - m1 - std::log(z1);
        out.logp_sample += x / temp - mt - std::log(zt);
        if (static_cast<int>(pick) == eos) {
            out.stop = "eos";
            break;
        }
        ids.push_back(static_cast<int>(pick));
        out.ids.push_back(static_cast<int>(pick));
    }
    return out;
}

// Supervised fine-tuning data (Phase 3). Each example becomes one row block
// of length T: x = seq[0..T-1], y = seq[1..T], where seq = prompt tokens +
// completion tokens + end-of-text, padded. mask[i] = 1 exactly when target
// y[i] is a completion token (or the closing end-of-text), so prompt and
// padding contribute nothing to the loss or its gradient. The tail 5% of
// examples (at least one) is held out for validation.
struct SftSet {
    std::vector<std::vector<int>> x, y;
    std::vector<std::vector<uint8_t>> m;
    // State tokens (arch.state_tokens): blocks inserted (and of them after a
    // reply without <|endoftext|>), whole exchanges cut from the end of a
    // dialogue row and rows dropped because the state ids pushed them past
    // T + 1.
    size_t state_blocks = 0, state_closed = 0, state_turns_cut = 0, state_rows_dropped = 0;
    // Conversation rows: read, rows written, dropped (first exchange, or under "each" the
    // exchange's prefix, past T + 1), exchanges cut from the end, rows with no trained reply.
    size_t conv_rows = 0, conv_out = 0, conv_over_T = 0, conv_exchanges_cut = 0, conv_untrained = 0;
    size_t val_start = 0;
    size_t target_tokens = 0;   // kept targets in the training split
    size_t completion_unk = 0;  // <unk> among kept targets, all splits
    json provenance = json::object();
};

SftSet load_sft(const Spec& s, const std::map<std::string, int>& vocab) {
    std::ifstream f(s.sft_examples);
    if (!f) throw std::runtime_error("cannot open data.sft_examples " + s.sft_examples);
    const int eos = wordtok::eos_id(vocab);
    const int pad = eos >= 0 ? eos : 0;
    const bool rej = s.objective == "rejection_sft";
    SftSet d;
    const nn::StateTemplate stt = s.state_k > 0 ? state_template(s, vocab) : nn::StateTemplate{};
    const chat::Template tmpl = chat_template_of(s);
    // State tokens on one row (IR_SPEC_DRAFT 1.5): the k ids after every
    // <|endoftext|> that closes an exchange, flags (loss, continuation) carried
    // with 0 on inserted ids. A block that would end the row is left out: no
    // position after it reads it, so it changes no loss or gradient. Rows past
    // T + 1: cut at the last whole exchange that fits (cut_ok), else dropped.
    // Returns false when the row is dropped.
    auto with_state = [&](std::vector<int>& seq, std::vector<uint8_t>& fl, bool cut_ok) {
        nn::StateInsertCount c;
        std::vector<uint8_t> f2;
        std::vector<int> out = insert_state_flagged(seq, fl, stt, f2, &c);
        const int S0 = static_cast<int>(stt.vocab);
        if (out.size() > s.T + 1) {
            size_t end = 0, ends_after = 0;
            for (size_t i = 0; i < out.size(); ++i)
                if (out[i] == stt.eot && i + 1 <= s.T + 1 && (i + 1 == out.size() || out[i + 1] == S0)) end = i + 1;
            if (!cut_ok || end == 0) {
                ++d.state_rows_dropped;
                return false;
            }
            for (size_t i = end; i < out.size(); ++i)
                if (out[i] == stt.eot && (i + 1 == out.size() || out[i + 1] == S0)) ++ends_after;
            d.state_turns_cut += ends_after;
            out.resize(end);
            f2.resize(end);
        }
        if (out.size() >= stt.k && out.back() == S0 + static_cast<int>(stt.k) - 1) {
            out.resize(out.size() - stt.k);
            f2.resize(f2.size() - stt.k);
        }
        size_t blocks = 0;
        for (int t : out) blocks += t == S0;
        d.state_blocks += blocks;
        d.state_closed += c.closed;
        seq = std::move(out);
        fl = std::move(f2);
        return true;
    };
    std::map<std::string, size_t> kinds;
    size_t n = 0, skipped = 0;
    std::string line;
    while (std::getline(f, line)) {
        ++n;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        const json r = json::parse(line, nullptr, false);
        if (!r.is_discarded() && !rej && r.contains("messages")) {
            // A conversation (LOCAL_LLM_STACK.md section 3 point 3a) rendered by the model's chat
            // template, with train.sft_loss (3b) choosing the trained replies: a reply's content
            // and its close (paperkiln/1: the <|endoftext|> after it) carry loss; markers, user
            // and system text, padding, state ids and state positions never do. Pieces are
            // tokenised one by one (special tokens are pre-tokeniser boundaries, so this equals
            // the whole rendering: tests/test_chat_template.py). A row keeps the longest prefix
            // of whole exchanges that fits T + 1 ("each": one row per reply, dropped when its
            // prefix does not fit); a reply is never cut. Under "all" these rows equal the
            // segments rows chat_data.py --all-turns writes, and under "each" its
            // {prompt, completion} rows (G2, tests/test_chat_template_sft.py).
            ++d.conv_rows;
            std::vector<chat::Message> msgs;
            std::vector<chat::Piece> pieces;
            try {
                msgs = chat::messages_from_json(r["messages"]);
                // v2 tool definitions, read as ordered JSON so that they render in their own member order
                std::vector<chat::json> tools;
                if (r.contains("tools") && !r["tools"].is_null()) {
                    const chat::json ro = chat::json::parse(line);  // kept alive for the loop
                    if (!ro["tools"].is_array()) throw chat::Error("tools must be a list of tool definitions");
                    for (const auto& t : ro["tools"]) tools.push_back(t);
                }
                pieces = chat::render_pieces(msgs, tmpl, /*generation_prompt=*/false, -1, tools);
            } catch (const chat::Error& e) {
                throw std::runtime_error(s.sft_examples + ":" + std::to_string(n) + ": " + e.what());
            }
            // Exchanges (chat::exchanges): a user message and everything before the next one (v1: one
            // reply; v2: replies, tool calls and tool results). Per piece: its ids, its message index
            // (-1: bos / default system) and whether it is reply text (content, a tool call, the
            // close). Pieces come in message order, so "the pieces of messages before `end`" is a
            // prefix of the piece list.
            const auto exr = chat::exchanges(msgs);
            const size_t E = exr.size();
            std::vector<std::vector<int>> p_ids;
            std::vector<int> p_msg;
            std::vector<uint8_t> p_reply;
            for (const auto& p : pieces) {
                p_ids.push_back(tokenize(p.text, vocab, 1u << 30));
                p_msg.push_back(p.index);
                p_reply.push_back(p.role == "assistant" && (p.kind == "content" || p.kind == "close" ||
                                                            p.kind == "call_open" || p.kind == "call"));
            }
            auto tokens_before = [&](size_t end) {  // tokens of every piece of messages before `end`
                size_t c = 0;
                for (size_t j = 0; j < pieces.size() && p_msg[j] < static_cast<int>(end); ++j) c += p_ids[j].size();
                return c;
            };
            auto trainable = [&](size_t i) { return msgs[i].role == "assistant" && msgs[i].loss != 0; };
            std::vector<std::pair<size_t, std::vector<int>>> rows;  // (messages kept, trained messages)
            if (s.sft_turns == "each") {
                for (size_t i = 0; i < msgs.size(); ++i) {
                    if (!trainable(i)) continue;
                    if (tokens_before(i + 1) > s.T + 1) {
                        ++d.conv_over_T;
                        continue;
                    }
                    rows.push_back({i + 1, {static_cast<int>(i)}});
                }
            } else {
                size_t k = 0;
                while (k < E && tokens_before(exr[k].second) <= s.T + 1) ++k;
                if (k == 0) {
                    ++d.conv_over_T;
                    continue;
                }
                d.conv_exchanges_cut += E - k;
                const size_t end = exr[k - 1].second;
                std::vector<int> tr;
                for (size_t i = 0; i < end; ++i)
                    if (trainable(i)) tr.push_back(static_cast<int>(i));
                if (s.sft_turns == "final")
                    tr = tr.empty() || tr.back() != static_cast<int>(end) - 1 ? std::vector<int>{} : std::vector<int>{tr.back()};
                rows.push_back({end, tr});
            }
            for (const auto& [end, tr] : rows) {
                std::vector<int> seq;
                std::vector<uint8_t> lt;
                for (size_t j = 0; j < pieces.size() && p_msg[j] < static_cast<int>(end); ++j) {
                    seq.insert(seq.end(), p_ids[j].begin(), p_ids[j].end());
                    const uint8_t on = p_reply[j] && std::find(tr.begin(), tr.end(), p_msg[j]) != tr.end();
                    lt.insert(lt.end(), p_ids[j].size(), on);
                }
                if (std::find(lt.begin(), lt.end(), uint8_t(1)) == lt.end()) {
                    ++d.conv_untrained;
                    continue;
                }
                if (s.state_k > 0 && !with_state(seq, lt, /*cut_ok=*/s.sft_turns != "each")) continue;
                if (seq.size() > s.T + 1)
                    throw std::runtime_error("example at line " + std::to_string(n) + " has " +
                                             std::to_string(seq.size()) + " tokens; data.T + 1 = " +
                                             std::to_string(s.T + 1));
                std::vector<int> x(s.T, pad), y(s.T, pad);
                std::vector<uint8_t> m(s.T, 0);
                size_t trained = 0;
                for (size_t i = 0; i + 1 < seq.size(); ++i) {
                    x[i] = seq[i];
                    y[i] = seq[i + 1];
                    m[i] = lt[i + 1];
                    trained += m[i];
                    if (m[i] && y[i] == 0) ++d.completion_unk;
                }
                if (!trained) {
                    ++d.state_rows_dropped;  // every trained reply was cut by the state blocks
                    continue;
                }
                d.x.push_back(std::move(x));
                d.y.push_back(std::move(y));
                d.m.push_back(std::move(m));
                ++d.conv_out;
            }
            continue;
        }
        if (!r.is_discarded() && s.sft_turns_set)
            throw std::runtime_error(s.sft_examples + ":" + std::to_string(n) +
                                     ": train.sft_loss applies to conversation rows ({\"messages\"}); this row is "
                                     "a rendered one, whose loss its rendering fixes");
        if (!r.is_discarded() && !rej && r.contains("segments")) {
            // A whole dialogue in one row: the segments' tokens in order, end-of-text
            // appended after every loss segment (as after a completion), and a target
            // trained exactly when it falls in a loss segment or on its end-of-text.
            const json& segs = r["segments"];
            if (!segs.is_array() || segs.empty())
                throw std::runtime_error(s.sft_examples + ":" + std::to_string(n) + ": segments must be a non-empty array");
            std::vector<int> seq;
            std::vector<uint8_t> lt;  // lt[j] = 1: token j is a trained target
            for (const auto& sg : segs) {
                if (!sg.is_object() || !sg.contains("text") || !sg.contains("loss") || !sg["text"].is_string() ||
                    !sg["loss"].is_boolean())
                    throw std::runtime_error(s.sft_examples + ":" + std::to_string(n) +
                                             ": each segment needs text (string) and loss (bool)");
                auto t = tokenize(sg["text"].get<std::string>(), vocab, 100000);
                const bool loss = sg["loss"].get<bool>();
                if (t.empty()) throw std::runtime_error("example " + std::to_string(n) + " has an empty segment");
                if (loss && eos >= 0) t.push_back(eos);
                seq.insert(seq.end(), t.begin(), t.end());
                lt.insert(lt.end(), t.size(), loss ? 1 : 0);
            }
            // Token 0 is never a target: a loss segment there would silently lose its first token.
            if (lt[0]) throw std::runtime_error("example " + std::to_string(n) + ": the first segment cannot carry loss");
            if (s.state_k > 0 && !with_state(seq, lt, /*cut_ok=*/true)) continue;
            if (seq.size() > s.T + 1)
                throw std::runtime_error("example at line " + std::to_string(n) + " has " + std::to_string(seq.size()) +
                                         " tokens; data.T + 1 = " + std::to_string(s.T + 1));
            std::vector<int> x(s.T, pad), y(s.T, pad);
            std::vector<uint8_t> m(s.T, 0);
            size_t trained = 0;
            for (size_t i = 0; i + 1 < seq.size(); ++i) {
                x[i] = seq[i];
                y[i] = seq[i + 1];
                m[i] = lt[i + 1];
                trained += m[i];
                if (m[i] && y[i] == 0) ++d.completion_unk;
            }
            if (!trained && s.state_k > 0) {
                ++d.state_rows_dropped;  // every trained turn was cut
                continue;
            }
            if (!trained) throw std::runtime_error("example " + std::to_string(n) + " has no loss segment");
            d.x.push_back(std::move(x));
            d.y.push_back(std::move(y));
            d.m.push_back(std::move(m));
            continue;
        }
        if (r.is_discarded() || !r.contains("prompt") || !r.contains("completion"))
            throw std::runtime_error(s.sft_examples + ":" + std::to_string(n) +
                                     ": need prompt and completion (or segments)");
        if (rej) {
            // A rejection-sampler record must carry its receipt (charter §6).
            for (const char* k : {"reward", "rationale", "evidence_kind", "evidence_uri", "accepted"})
                if (!r.contains(k))
                    throw std::runtime_error(s.sft_examples + ":" + std::to_string(n) + ": record has no " + k +
                                             " (rejection_sft trains only on receipted records)");
            if (!r["accepted"].get<bool>()) {
                ++skipped;
                continue;
            }
            const std::string kind = r["evidence_kind"].get<std::string>();
            if (kind == "human" && !s.allow_human_labels)
                throw std::runtime_error("human-labelled record at line " + std::to_string(n) +
                                         ": set train.allow_human_labels to train on audit evidence");
            if (r["rationale"].get<std::string>().empty() || r["evidence_uri"].get<std::string>().empty())
                throw std::runtime_error(s.sft_examples + ":" + std::to_string(n) + ": empty receipt");
            ++kinds[kind];
        }
        const auto p = tokenize(r["prompt"].get<std::string>(), vocab, 100000);
        auto c = tokenize(r["completion"].get<std::string>(), vocab, 100000);
        if (eos >= 0) c.push_back(eos);
        std::vector<int> seq = p;
        seq.insert(seq.end(), c.begin(), c.end());
        if (p.empty() || c.empty()) throw std::runtime_error("example " + std::to_string(n) + " is empty");
        // comp[j] = 1: token j is a completion token (a trained target).
        std::vector<uint8_t> comp(seq.size(), 0);
        for (size_t j = p.size(); j < seq.size(); ++j) comp[j] = 1;
        if (s.state_k > 0 && !with_state(seq, comp, /*cut_ok=*/false)) continue;
        if (seq.size() > s.T + 1)
            throw std::runtime_error("example at line " + std::to_string(n) + " has " + std::to_string(seq.size()) +
                                     " tokens; data.T + 1 = " + std::to_string(s.T + 1));
        std::vector<int> x(s.T, pad), y(s.T, pad);
        std::vector<uint8_t> m(s.T, 0);
        for (size_t i = 0; i + 1 < seq.size(); ++i) {
            x[i] = seq[i];
            y[i] = seq[i + 1];
            m[i] = comp[i + 1];
            if (m[i] && y[i] == 0) ++d.completion_unk;
        }
        d.x.push_back(std::move(x));
        d.y.push_back(std::move(y));
        d.m.push_back(std::move(m));
    }
    if (d.x.size() < 2) throw std::runtime_error("data.sft_examples needs at least two usable examples");
    d.val_start = d.x.size() - std::max<size_t>(1, d.x.size() / 20);
    for (size_t e = 0; e < d.val_start; ++e)
        for (uint8_t v : d.m[e]) d.target_tokens += v;
    if (rej) {
        d.provenance["evidence_kinds"] = kinds;
        d.provenance["not_accepted_skipped"] = skipped;
    }
    return d;
}

// Row-wise log-softmax in double, returned as float (grpo's reference
// log-probabilities and its KL/entropy monitors).
Matrix log_softmax_rows(const Matrix& L) {
    Matrix out(L.rows(), L.cols());
    for (size_t i = 0; i < L.rows(); ++i) {
        double mx = -1e300, z = 0.0;
        for (size_t j = 0; j < L.cols(); ++j) mx = std::max(mx, static_cast<double>(L(i, j)));
        for (size_t j = 0; j < L.cols(); ++j) z += std::exp(L(i, j) - mx);
        const double lz = mx + std::log(z);
        for (size_t j = 0; j < L.cols(); ++j) out(i, j) = static_cast<float>(L(i, j) - lz);
    }
    return out;
}

// GRPO prompts (Phase 4): {prompt, answer, uri} JSONL for training
// (data.grpo_prompts) and, optionally, held-out evaluation (data.grpo_eval).
// The reward verifier is train.grpo.reward; exact match and the format
// verifier always run beside it as monitors, so every step can report
// hack_frac (format passes, exact match fails) whatever the reward is.
struct GrpoData {
    std::vector<std::string> prompts, eval_prompts;
    std::vector<std::vector<int>> prompt_ids, eval_ids;
    std::shared_ptr<posttrain::Verifier> reward, exact, format;
};

GrpoData load_grpo(const Spec& s, const std::map<std::string, int>& vocab) {
    GrpoData g;
    posttrain::ReferenceSet refs;
    auto read = [&](const std::string& path, std::vector<std::string>& texts, std::vector<std::vector<int>>& ids) {
        for (auto& [prompt, ref] : posttrain::load_references(path)) {
            if (!refs.emplace(prompt, ref).second)
                throw std::runtime_error("grpo: prompt appears in both training and evaluation sets");
            auto t = tokenize(prompt, vocab, 100000);
            if (t.empty()) throw std::runtime_error("grpo: a prompt has no in-vocabulary token");
            if (t.size() + static_cast<size_t>(s.grpo_max_new) + 1 > s.T + 1)
                throw std::runtime_error("grpo: prompt + max_new exceeds data.T");
            texts.push_back(prompt);
            ids.push_back(std::move(t));
        }
    };
    read(s.grpo_prompts, g.prompts, g.prompt_ids);
    if (!s.grpo_eval.empty()) read(s.grpo_eval, g.eval_prompts, g.eval_ids);
    g.exact = std::make_shared<posttrain::ExactMatchVerifier>(refs);
    g.format = std::make_shared<posttrain::FormatVerifier>();
    if (s.grpo_reward == "exact_match" && s.grpo_reward_shift != 0) {
        // Every receipt says the key is shifted: the evidence URI carries it.
        posttrain::ReferenceSet shifted;
        for (const auto& [prompt, ref] : refs)
            shifted.emplace(prompt, posttrain::Reference{ref.answer + s.grpo_reward_shift,
                                                         ref.uri + "#SHIFTED" + std::to_string(s.grpo_reward_shift)});
        g.reward = std::make_shared<posttrain::ExactMatchVerifier>(shifted);
    } else if (s.grpo_reward == "exact_match") {
        g.reward = g.exact;
    } else if (s.grpo_reward == "format") {
        g.reward = g.format;
    } else if (s.grpo_reward == "synthetic_judge") {
        posttrain::SyntheticJudgeParams p;
        const json& j = s.grpo_judge;
        p.accept_wrong_with_cue = j.value("cue", p.accept_wrong_with_cue);
        p.accept_wrong_near = j.value("near", p.accept_wrong_near);
        p.near_k = j.value("near_k", p.near_k);
        p.accept_wrong_other = j.value("other", p.accept_wrong_other);
        p.reject_correct = j.value("reject_correct", p.reject_correct);
        p.seed = j.value("seed", p.seed);
        g.reward = std::make_shared<posttrain::SyntheticJudgeVerifier>(refs, p);
    } else {
        throw std::runtime_error("train.grpo.reward must be exact_match, format or synthetic_judge");
    }
    return g;
}

// Preference pairs (Phase 5). Lines: {"prompt", "chosen", "rejected",
// "chosen_evidence": {evidence_kind, evidence_uri, audited, ...},
// "rejected_evidence": {...}}. A pair's weight is the smaller of its two
// receipts' weights under the charter §4 table (frozen defaults below,
// overridable by train.po.weights): exact_match, unit_test, human 1.0;
// format 0.5; judge 0.3; a receipt missing its kind or URI 0. An audit that
// overturned a judge verdict ("audit_reward" present and different from the
// receipt's reward) makes that receipt 0.
struct PrefData {
    std::vector<std::vector<int>> prompt, chosen, rejected;
    std::vector<float> w;
    std::vector<std::string> kind;  // the weaker receipt's kind, for the histogram
    size_t val_start = 0;
    size_t chosen_open = 0, rejected_open = 0;  // replies with no end-of-text (chosen_stop / rejected_stop)
};

float receipt_weight(const json& e, const json& table) {
    if (!e.is_object() || !e.contains("evidence_kind") || !e.contains("evidence_uri")) return 0.0f;
    const std::string kind = e["evidence_kind"].get<std::string>();
    if (kind.empty() || e["evidence_uri"].get<std::string>().empty()) return 0.0f;
    if (kind == "judge" && e.contains("audit_reward") && e.contains("reward") &&
        e["audit_reward"].get<double>() != e["reward"].get<double>())
        return 0.0f;
    static const std::map<std::string, float> defaults = {
        {"exact_match", 1.0f}, {"unit_test", 1.0f}, {"human", 1.0f}, {"format", 0.5f}, {"judge", 0.3f}};
    if (table.contains(kind)) return table[kind].get<float>();
    const auto it = defaults.find(kind);
    return it == defaults.end() ? 0.0f : it->second;
}

PrefData load_pref(const Spec& s, const std::map<std::string, int>& vocab) {
    std::ifstream f(s.pref_pairs);
    if (!f) throw std::runtime_error("cannot open data.pref_pairs " + s.pref_pairs);
    const int eos = wordtok::eos_id(vocab);
    PrefData d;
    std::string line;
    size_t n = 0;
    while (std::getline(f, line)) {
        ++n;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        const json r = json::parse(line, nullptr, false);
        if (r.is_discarded() || !r.contains("prompt") || !r.contains("chosen") || !r.contains("rejected"))
            throw std::runtime_error(s.pref_pairs + ":" + std::to_string(n) + ": need prompt, chosen, rejected");
        const json ce = r.value("chosen_evidence", json()), re = r.value("rejected_evidence", json());
        float w = 1.0f;
        std::string kind = "uniform";
        if (s.po_weight_mode != "uniform") {
            const float wc = receipt_weight(ce, s.po_weights), wr = receipt_weight(re, s.po_weights);
            w = std::min(wc, wr);
            const json& weaker = wc <= wr ? ce : re;
            kind = weaker.is_object() ? weaker.value("evidence_kind", std::string("unattributed")) : "unattributed";
        }
        auto p = tokenize(r["prompt"].get<std::string>(), vocab, 100000);
        auto c = tokenize(r["chosen"].get<std::string>(), vocab, 100000);
        auto j = tokenize(r["rejected"].get<std::string>(), vocab, 100000);
        // A reply ends on end-of-text unless its pair says how sampling stopped
        // ("chosen_stop" / "rejected_stop", tools/posttrain/chat_judge.py pairs): only "eos"
        // ends on it; a reply the token budget cut ("length") or that stopped on a
        // generated role marker ("turn") is trained as the prefix it is, never given an
        // end the model did not write.
        const bool c_end = r.value("chosen_stop", std::string("eos")) == "eos";
        const bool j_end = r.value("rejected_stop", std::string("eos")) == "eos";
        d.chosen_open += !c_end;
        d.rejected_open += !j_end;
        if (eos >= 0) {
            if (c_end) c.push_back(eos);
            if (j_end) j.push_back(eos);
        }
        if (p.empty() || c.empty() || j.empty()) throw std::runtime_error("pair " + std::to_string(n) + " is empty");
        if (p.size() + std::max(c.size(), j.size()) > s.T + 1)
            throw std::runtime_error("pair at line " + std::to_string(n) + " exceeds data.T + 1 tokens");
        d.prompt.push_back(std::move(p));
        d.chosen.push_back(std::move(c));
        d.rejected.push_back(std::move(j));
        d.w.push_back(w);
        d.kind.push_back(kind);
    }
    if (d.w.size() < 2) throw std::runtime_error("data.pref_pairs needs at least two pairs");
    d.val_start = d.w.size() - std::max<size_t>(1, d.w.size() / 20);
    if (s.po_weight_mode == "shuffled") {
        // The falsifier: the same weights (so the same sum and ESS) attached
        // to the wrong pairs, within the training split.
        std::mt19937 g(static_cast<unsigned>(s.seed) * 7919u + 17u);
        std::shuffle(d.w.begin(), d.w.begin() + d.val_start, g);
    }
    return d;
}

// Rows for a set of pairs: each pair contributes its chosen then its
// rejected sequence (prompt + completion), each padded to L. Completion
// target rows carry the pair index and sign; prompt and padding rows -1.
struct PrefBatch {
    std::vector<int> x, y, row_pair;
    std::vector<int8_t> sign;
    size_t L = 0, seqs = 0;
};

PrefBatch pref_rows(const PrefData& d, const std::vector<size_t>& pairs, int pad) {
    PrefBatch b;
    for (size_t i : pairs)
        b.L = std::max({b.L, d.prompt[i].size() + d.chosen[i].size() - 1, d.prompt[i].size() + d.rejected[i].size() - 1});
    for (size_t k = 0; k < pairs.size(); ++k) {
        const size_t i = pairs[k];
        for (int side = 0; side < 2; ++side) {
            std::vector<int> q = d.prompt[i];
            const auto& comp = side == 0 ? d.chosen[i] : d.rejected[i];
            q.insert(q.end(), comp.begin(), comp.end());
            for (size_t t = 0; t < b.L; ++t) {
                const bool real = t + 1 < q.size();
                const bool tgt = real && t + 1 >= d.prompt[i].size();
                b.x.push_back(real ? q[t] : pad);
                b.y.push_back(real ? q[t + 1] : pad);
                b.row_pair.push_back(tgt ? static_cast<int>(k) : -1);
                b.sign.push_back(side == 0 ? 1 : -1);
            }
            ++b.seqs;
        }
    }
    return b;
}

// sum_rows sign * log p(target) per pair, from a logits matrix.
std::vector<double> pair_margins(const Matrix& logits, const PrefBatch& b, size_t P) {
    const Matrix lp = log_softmax_rows(logits);
    std::vector<double> m(P, 0.0);
    for (size_t r = 0; r < b.y.size(); ++r)
        if (b.row_pair[r] >= 0) m[b.row_pair[r]] += b.sign[r] * static_cast<double>(lp(r, b.y[r]));
    return m;
}

// The IR's tokenizer section (microtorch/ir.hpp): the GGUF carries the
// vocabulary itself; this names the specials an engine needs. BPE: the
// tokenizer's fixed specials by role. Word-level: out-of-vocabulary words
// map to id 0, and the end-of-text word when the vocabulary has one. No
// BOS is ever prepended.
ir::json ir_tokenizer(const std::vector<std::string>& tokens, const std::map<std::string, int>& vocab) {
    ir::json sp = ir::json::object();
    if (g_bpe) {
        static const char* roles[] = {"unk", "eot", "user", "assistant"};  // bpe::specials() order
        for (size_t k = 0; k < bpe::specials().size() && k < 4; ++k) {
            const auto it = std::find(tokens.begin(), tokens.end(), bpe::specials()[k]);
            if (it != tokens.end()) sp[roles[k]] = it - tokens.begin();
        }
    } else {
        sp["unk"] = 0;
        if (const int e = wordtok::eos_id(vocab); e >= 0) sp["eot"] = e;
    }
    const char* type = !g_bpe ? "word" : g_bpe->model == "gpt2" ? "llama3_bpe" : "bpe";
    return {{"type", type}, {"source", "gguf"}, {"specials", sp}, {"bos", nullptr}};
}

// The vocabulary onto an export config (llama or gpt2): its tokens, a BPE
// vocabulary's merges, and the end-of-text id when it has one.
template <class Cfg>
void set_export_tokenizer(Cfg& gc, const Spec& s, const std::vector<std::string>& tokens,
                          const std::map<std::string, int>& vocab) {
    gc.vocab_size = (uint32_t)tokens.size();
    gc.tokens = tokens;
    // A BPE vocabulary travels with the model: without its merges an
    // engine reading the export would split text into bytes only.
    if (g_bpe) {
        gc.tokenizer_model = g_bpe->model;
        gc.tokenizer_pre = g_bpe->pre;
        gc.merges = bpe::read_gguf_tokenizer(s.vocab_gguf).merges;
        // Token types travel only with the llama.cpp form (they mark the
        // specials control tokens); old "bpe" exports stay byte-identical.
        if (g_bpe->model == "gpt2") gc.token_type = g_bpe->token_type;
    }
    // End-of-text in the vocabulary -> the GGUF's eos id, so an
    // external engine stops where this one does (else keep the
    // exporter's default).
    if (const int e = wordtok::eos_id(vocab); e >= 0) gc.eos_token_id = (uint32_t)e;
    // The chat template as data (LOCAL_LLM_STACK.md section 3 point 3c): the record and, for a
    // paperkiln record, the Jinja llama.cpp, LM Studio and Ollama's import read.
    if (declares_chat_template(s, tokens)) {
        const chat::Template t = chat_template_of(s);
        gc.chat_template = t.to_json().dump();
        if (t.name == "paperkiln") gc.chat_template_jinja = chat::jinja(t);
    }
}

// <name>.Modelfile beside an F32 GGUF that declares a paperkiln template (Ollama's TEMPLATE and
// stop parameters; `ollama create NAME -f <name>.Modelfile` from that folder).
void write_chat_modelfile(const Spec& s, const std::vector<std::string>& tokens, const std::string& gguf_path) {
    if (!declares_chat_template(s, tokens)) return;
    const chat::Template t = chat_template_of(s);
    if (t.name != "paperkiln") return;
    const std::string base = std::filesystem::path(gguf_path).filename().string();
    const std::string mpath = gguf_path.substr(0, gguf_path.size() - 5) + ".Modelfile";
    std::ofstream f(mpath, std::ios::binary);
    f << chat::modelfile(base, t);
    if (!f) throw std::runtime_error("export: could not write " + mpath);
}

// The IR document for a model's own export_ir(), with the vocabulary's
// tokenizer section after "model" as in spec/examples/*.ir.json, written beside the
// GGUFs as <name>.ir.json; returns the compact dump the GGUFs embed.
std::string write_ir_doc(const Spec& s, const ir::json& model_ir, const std::vector<std::string>& tokens,
                         const std::map<std::string, int>& vocab, const std::string& ir_path) {
    ir::json doc = {{"_comment", "paperkiln IR v" + std::to_string(ir::kVersion) + " for " + s.name +
                                     ", exported by mtstudio. Schema and semantics: microtorch/spec/IR_SPEC.md "
                                     "(schema spec/ir.schema.json)."}};
    for (const auto& [k, v] : model_ir.items()) {
        doc[k] = v;
        if (k == "model") doc["tokenizer"] = ir_tokenizer(tokens, vocab);
    }
    std::ofstream f(ir_path);
    f << doc.dump(2) << "\n";
    if (!f) throw std::runtime_error("export: could not write " + ir_path);
    return doc.dump();
}

// GGUF exports of a trained model into out_dir: <name>.gguf (F32) when
// s.exp_gguf, and <name>_<quant>.gguf for each of s.exp_gguf_quants (the
// Q8_0 and 4-bit variants). Every variant embeds the model's IR (paperkiln.ir),
// also written once beside them as <name>.ir.json. Shared by `run` and
// `export`. The llama family exports through `llama`; the gpt2 family
// (`gpt`, exact or sliding-window attention) and the GPT-2 flavour of flex
// (`flex`) export F32 in llama.cpp's gpt2 architecture; everything else is
// reported as skipped and gets no IR. Sliding-window models keep the gpt2
// tensors (so the file loads anywhere) and put the window and sinks in the
// IR only: an engine reading the GGUF alone computes dense attention.
void export_ggufs(const Spec& s, nn::Module& model_ref, const nn::Llama* llama, const parity::ParityLM* gpt,
                  const parity::FlexLM* flex, const parity::AttnResLM* attnres, const parity::MambaLM* mamba,
                  const std::vector<std::string>& tokens,
                  const std::map<std::string, int>& vocab, const std::string& rope_heads,
                  const std::string& out_dir, const std::function<void(const json&)>& emit) {
    // kimi, srd, attnres and the non-GPT-2 flex flavours (FamilyExport).
    const std::optional<parity::FamilyExport> fx = gpt       ? gpt->family_export()
                                                   : flex    ? flex->family_export()
                                                   : attnres ? attnres->family_export()
                                                   : mamba   ? mamba->family_export(s.T)
                                                             : std::nullopt;
    if (llama && s.bitlinear) {
        // No engine or IR op computes BitLinear's in-norm and int8 activations
        // (LOCAL_LLM_WORK R3): the latent weights in a GGUF would be served wrong.
        emit({{"event", "export_skipped"},
              {"format", "gguf"},
              {"reason",
               "arch.custom.bitlinear: no GGUF/IR form for BitLinear layers; the safetensors carry "
               "the "
               "latent weights and mtstudio serves them"}});
        std::fprintf(stderr, "export: no IR written (bitlinear model)\n");
    } else if (llama && s.moe_experts > 0 && s.moe_gate == "raw") {
        // llama.cpp's llama-architecture MoE renormalises its top-k gates;
        // a raw-gate (Switch) model would be served with different outputs.
        emit({{"event", "export_skipped"},
              {"format", "gguf"},
              {"reason", "moe gate=raw: llama.cpp renormalises top-k gates, so it would not reproduce this model"}});
        std::fprintf(stderr, "export: no IR written (no GGUF for a gate=raw MoE model)\n");
    } else if (llama) {
        auto sd2 = model_ref.state_dict();
        // Tied head: inject lm_head = E^T in microtorch [in, out]
        // layout; the exporter transposes it back into llama
        // [vocab, hidden] byte order under weights_in_out.
        if (!sd2.count("lm_head.weight")) {
            const Matrix& E = llama->embed_tokens->weight->data;
            Matrix ET(E.cols(), E.rows());
            for (size_t i = 0; i < E.rows(); ++i)
                for (size_t j = 0; j < E.cols(); ++j) ET(j, i) = E(i, j);
            sd2.emplace("lm_head.weight", std::move(ET));
        }
        gguf::LlamaExportConfig gc;
        gc.name = s.name;
        gc.embedding_length = (uint32_t)s.d;
        // Stored layers: a recursed model writes its shared layers once.
        gc.block_count = (uint32_t)s.layers;
        // A recursed model applies them more than once, which llama.cpp's
        // llama graph cannot express: architecture "paperkiln" (its keys
        // paperkiln.*, ir.hpp), so llama.cpp refuses the file rather than
        // running the stored layers once each, and only the IR (block
        // weight_index) reproduces it.
        if (effective_layers(s) != s.layers || routed(s)) gc.architecture = "paperkiln";
        // State tokens: the extended embedding and the template are not
        // llama.cpp's (ir.hpp embedding state_weight; paperkiln.state_tokens.k).
        if (s.state_k > 0) {
            gc.architecture = "paperkiln";
            gc.state_tokens_k = static_cast<uint32_t>(s.state_k);
        }
        // schedule "mor": the router, written in full (ir.hpp depth_route).
        if (llama->rec_router) gc.extra_proj.emplace_back("recursion.router.weight", "rec.0.router.weight");
        if (llama->attn_res) {
            // llama_attnres (llama_attnres.hpp): llama.cpp's graph has no depth
            // attention, so the file is "paperkiln"; the pseudo-queries are [d]
            // vectors, application 0's attention query (never read) as zeros.
            gc.architecture = "paperkiln";
            sd2.emplace("attn_res.attn_q.0", Matrix(1, s.d));
            gc.extra_vec.emplace_back("attn_res.attn_q.0", nn::attnres_unused_query_name(*llama));
            for (const auto& kv : nn::attnres_gguf_names(*llama)) gc.extra_vec.push_back(kv);
        }
        gc.head_count = (uint32_t)s.heads;
        gc.head_count_kv = (uint32_t)llama->cfg.kv_heads();
        gc.rope_freq_base = s.rope_theta;
        if (s.rope_dims) gc.rope_dimension_count = (uint32_t)s.rope_dims;
        // NEOX pairs, partial rotary and per-head q/k norms are not llama.cpp's
        // llama graph (interleaved RoPE over the whole head, no q/k norm): the
        // file is "paperkiln" and only the IR reproduces it. Grouped-query
        // attention alone is llama's (head_count_kv).
        if (s.rope_pairing != "interleaved" || (s.rope_dims && s.rope_dims != s.d / s.heads) || s.qk_norm)
            gc.architecture = "paperkiln";
        // qk_temp: each stored layer's [1, H] temperatures repeated over their
        // heads' columns, the [d] row the IR's scale node reads
        // (blk.{i}.attn_q_temp.weight; LlamaBlock::export_ir).
        if (s.qk_temp != 0.0f) {
            const size_t dk = s.d / s.heads;
            for (size_t l = 0; l < llama->blocks.size(); ++l) {
                const Matrix& t = llama->blocks[l]->q_temp_w->data;
                Matrix row(1, s.d);
                for (size_t c = 0; c < s.d; ++c) row(0, c) = t(0, c / dk);
                const std::string key = "layers." + std::to_string(l) + ".self_attn.q_temp.row";
                sd2.emplace(key, std::move(row));
                gc.extra_vec.emplace_back(key, "blk." + std::to_string(l) + ".attn_q_temp.weight");
            }
        }
        // The model's real FFN width. Exports before 19 Oct 2026 wrote 3d
        // whatever arch.custom.d_ff said; llama.cpp checks ffn tensor
        // shapes against this value.
        gc.feed_forward_length = (uint32_t)llama->cfg.d_ff;
        if (s.moe_experts > 0) {
            gc.expert_count = (uint32_t)s.moe_experts;
            gc.expert_used_count = (uint32_t)s.moe_top_k;
        }
        gc.context_length = (uint32_t)s.T;
        // The epsilon ops::rmsnorm trains with (src/ops.cpp). Exports
        // before 19 Oct 2026 wrote 1e-6, so engines normalised slightly
        // differently from training (greedy parity held at 40/40).
        gc.rms_eps = 1e-5f;
        if (s.precision != "fp32") gc.train_precision = s.precision;  // paperkiln.train.precision
        gc.weights_in_out = true;  // microtorch Linear is [in, out]
        set_export_tokenizer(gc, s, tokens, vocab);
        // The IR, from the modules themselves (Llama::export_ir).
        const std::string ir_path = out_dir + "/" + s.name + ".ir.json";
        gc.ir = write_ir_doc(s, llama->export_ir(), tokens, vocab, ir_path);
        std::vector<std::string> quants;
        if (s.exp_gguf) quants.push_back("f32");
        quants.insert(quants.end(), s.exp_gguf_quants.begin(), s.exp_gguf_quants.end());
        for (const auto& q : quants) {
            gc.quant = q;
            const bool f32 = q == "f32";
            const std::string gpath = out_dir + "/" + s.name + (f32 ? "" : "_" + q) + ".gguf";
            gguf::export_gguf_llama(gpath, sd2, gc);
            if (f32) write_chat_modelfile(s, tokens, gpath);
            json xe = {{"event", "export"},
                       {"format", f32 ? "gguf" : "gguf_" + q},
                       {"path", gpath},
                       {"rope_heads", rope_heads},
                       {"ir", ir_path}};
            // GGUF llama (rope.dimension_count = d/H) means RoPE on every
            // head; engines reading it reproduce this model only under "all".
            if (rope_heads == "first" && s.heads > 1)
                xe["warning"] =
                    "rope_heads=first: GGUF engines rotate every head, so they will not "
                    "reproduce this model's outputs";
            emit(xe);
        }
    } else if ((gpt || flex) &&
               (gpt ? gpt->gpt2_export_blocker() : flex->gpt2_export_blocker()).empty()) {
        // llama.cpp's gpt2 architecture: LayerNorm with bias, fused qkv,
        // tanh GELU (llama.cpp's gpt2 uses the same approximation), learned
        // positions, untied head. Quantised variants are not offered yet.
        for (const auto& q : s.exp_gguf_quants)
            emit({{"event", "export_skipped"},
                  {"format", "gguf_" + q},
                  {"reason", "the gpt2 export is F32 only"}});
        if (!s.exp_gguf) return;
        gguf::Gpt2ExportConfig gc;
        gc.name = s.name;
        gc.context_length = (uint32_t)s.T;  // = the learned position table's rows
        gc.embedding_length = (uint32_t)s.d;
        gc.block_count = (uint32_t)s.layers;
        gc.head_count = (uint32_t)s.heads;
        gc.feed_forward_length = (uint32_t)(gpt ? 4 * s.d : flex->cfg.d_ff);
        // One epsilon in the file: every LayerNorm's must be the same.
        const float eps = gpt ? gpt->ln_f->eps : flex->ln_f_ln->eps;
        gc.layer_norm_eps = eps;
        const ir::json model_ir = gpt ? gpt->export_ir() : flex->export_ir();
        for (const auto& n : model_ir["block"]["nodes"])
            if (n["op"] == "layernorm" && n["eps"] != ir::f32(eps))
                throw std::runtime_error("export: the gpt2 GGUF holds one layer-norm epsilon, but the blocks differ");
        set_export_tokenizer(gc, s, tokens, vocab);
        const std::string ir_path = out_dir + "/" + s.name + ".ir.json";
        gc.ir = write_ir_doc(s, model_ir, tokens, vocab, ir_path);
        const std::string gpath = out_dir + "/" + s.name + ".gguf";
        // A window and sinks live in the IR only: llama.cpp's gpt2 graph has none, so the file says
        // "paperkiln", which llama.cpp refuses, instead of a gpt2 it would compute densely (keramos rule 38).
        const bool windowed = gpt ? gpt->windowed() : flex->windowed();
        if (windowed) gc.architecture = "paperkiln";
        std::vector<std::string> unmapped;
        const auto sd = gpt ? gpt->gpt2_state_dict() : flex->gpt2_state_dict();
        gguf::export_gguf_gpt2(gpath, sd, gc, &unmapped);
        if (!unmapped.empty()) throw std::runtime_error("export: gpt2 tensor not exported: " + unmapped[0]);
        write_chat_modelfile(s, tokens, gpath);
        // Every trained parameter must be in the file (gpt2_state_dict is a
        // copy under other names, so count them).
        if (sd.size() != model_ref.named_parameters().size())
            throw std::runtime_error("export: the gpt2 state dict has " + std::to_string(sd.size()) +
                                     " tensors, the model " + std::to_string(model_ref.named_parameters().size()));
        json xe = {{"event", "export"}, {"format", "gguf"}, {"path", gpath}, {"arch", gc.architecture}, {"ir", ir_path}};
        if (windowed) {
            // Absent for dense gpt2 exports: their event is unchanged.
            xe["attention"] = "swa";
            xe["window"] = s.window;
            xe["sinks"] = s.sinks;
            xe["warning"] =
                "sliding-window attention lives in the embedded IR only: the file's architecture is paperkiln "
                "(gpt2's tensors), which llama.cpp refuses rather than computing dense attention";
        }
        emit(xe);
    } else if (fx) {
        // F32, the IR from the family's own export (microtorch/ir.hpp,
        // Families); gpt2 when the tensors are exactly llama.cpp's gpt2 set.
        for (const auto& q : s.exp_gguf_quants)
            emit({{"event", "export_skipped"}, {"format", "gguf_" + q}, {"reason", "the " + fx->arch + " export is F32 only"}});
        if (!s.exp_gguf) return;
        const std::string ir_path = out_dir + "/" + s.name + ".ir.json";
        const std::string gpath = out_dir + "/" + s.name + ".gguf";
        const std::string ir_str = write_ir_doc(s, fx->ir, tokens, vocab, ir_path);
        const size_t n_params = model_ref.named_parameters().size();
        if (fx->arch == "gpt2") {
            gguf::Gpt2ExportConfig gc;
            // gpt2's tensor set, but only the IR states the model (kimi, recurrence, plain flex): architecture
            // paperkiln, so llama.cpp refuses the file instead of computing softmax gpt2 (keramos rule 38).
            gc.architecture = "paperkiln";
            gc.name = s.name;
            gc.context_length = (uint32_t)s.T;
            gc.embedding_length = (uint32_t)s.d;
            gc.block_count = (uint32_t)s.layers;
            gc.head_count = (uint32_t)fx->heads;
            gc.feed_forward_length = (uint32_t)fx->d_ff;
            for (const auto& n : fx->ir["block"]["nodes"])
                if (n["op"] == "layernorm" && n["eps"] != ir::f32(1e-5f))
                    throw std::runtime_error("export: the gpt2 GGUF holds layer-norm epsilon 1e-5 only");
            set_export_tokenizer(gc, s, tokens, vocab);
            gc.ir = ir_str;
            std::vector<std::string> unmapped;
            gguf::export_gguf_gpt2(gpath, fx->gpt2_sd, gc, &unmapped);
            if (!unmapped.empty()) throw std::runtime_error("export: gpt2 tensor not exported: " + unmapped[0]);
            if (fx->gpt2_sd.size() != n_params)
                throw std::runtime_error("export: the gpt2 state dict has " + std::to_string(fx->gpt2_sd.size()) +
                                         " tensors, the model " + std::to_string(n_params));
        } else {
            gguf::PaperkilnExportConfig gc;
            gc.name = s.name;
            gc.context_length = (uint32_t)s.T;
            gc.embedding_length = (uint32_t)s.d;
            gc.block_count = (uint32_t)fx->ir["model"]["n_layers"].get<size_t>();
            gc.head_count = (uint32_t)fx->heads;
            gc.feed_forward_length = (uint32_t)fx->d_ff;
            set_export_tokenizer(gc, s, tokens, vocab);
            gc.ir = ir_str;
            // Every trained parameter in the file, once (the extra tensors
            // are the fixed tables and unread rows FamilyExport counts).
            if (fx->tensors.size() != n_params + fx->constants)
                throw std::runtime_error("export: " + std::to_string(fx->tensors.size()) + " tensors for " +
                                         std::to_string(n_params) + " parameters and " +
                                         std::to_string(fx->constants) + " constants");
            gguf::export_gguf_paperkiln(gpath, fx->tensors, gc);
        }
        write_chat_modelfile(s, tokens, gpath);
        json xe = {{"event", "export"},           {"format", "gguf"}, {"path", gpath}, {"arch", "paperkiln"},
                   {"family", fx->ir["family_hint"]}, {"ir", ir_path}};
        if (!fx->warning.empty()) xe["warning"] = fx->warning;
        emit(xe);
    } else {
        const std::string why = gpt     ? gpt->gpt2_export_blocker()
                                : flex  ? flex->gpt2_export_blocker()
                                        : "family " + s.family + " (" + s.attention + ") has no GGUF export";
        emit({{"event", "export_skipped"}, {"format", "gguf"}, {"reason", why}});
        // No IR for a model without an export; none beats a wrong one.
        std::fprintf(stderr, "export: no IR written (%s)\n", why.c_str());
    }
}

// A runtime-defined activation ("expr:<e>"), checked before any training:
// (1) its autograd backward (the primitives' own backwards, tape-composed)
// against central finite differences, loss = sum(expr(x) .* R) on [4, 8]
// random inputs kept 0.2 away from 0 (abs, relu, max and where kink there),
// h = 1e-2 as tests/test_expr.cpp, failing above 2e-3 relative; (2) value
// and gradient finite on x in [-20, 20] step 0.25. A failure throws: a
// wrong or non-finite gradient would train silently. The event is the
// receipt (worst error, the probe).
json expr_gradcheck(const std::string& text, const std::string& slot = "activation",
                    const std::vector<parity::FlexConfig::ExprInput>& ins = {}) {
    // Several inputs (a gate, learned scalars or channels; parity::FlexConfig): every input is
    // checked. The slot's own rows and the gate are random as x is; a learned input is its init
    // plus noise (finite differences around where training starts), and its probe value is its init.
    const size_t n_in = std::max<size_t>(1, ins.size());
    auto learned = [&](size_t k) { return k < ins.size() && (ins[k].kind == "scalar" || ins[k].kind == "channel"); };
    const Expr e = Expr::parse(text, n_in);
    const size_t T = 4, w = 8;
    std::mt19937 gen(20261023);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<Var> xs;
    Matrix R(T, w);
    for (size_t k = 0; k < n_in; ++k) {
        Matrix X(T, w);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < w; ++j) {
                const float v = nd(gen);
                X(i, j) = learned(k) ? ins[k].init + 0.3f * v : (v < 0 ? -0.2f : 0.2f) + v;
            }
        xs.push_back(make_var(X, true));
    }
    for (size_t i = 0; i < T; ++i)
        for (size_t j = 0; j < w; ++j) R(i, j) = nd(gen);
    const Var r = make_var(R);
    auto loss_of = [&]() {
        return ops::scale(ops::mean(ops::mul(ops::expr(xs, e), r)), static_cast<float>(T * w));
    };
    backward(loss_of());
    const float h = 1e-2f;
    double worst = 0;
    for (const Var& x : xs)
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < w; ++j) {
                const float keep = x->data(i, j);
                NoGrad ng;
                x->data(i, j) = keep + h;
                const double up = loss_of()->data(0, 0);
                x->data(i, j) = keep - h;
                const double dn = loss_of()->data(0, 0);
                x->data(i, j) = keep;
                const double fd = (up - dn) / (2.0 * h), a = x->grad(i, j);
                const double rel = std::abs(a - fd) / (1.0 + std::max(std::abs(a), std::abs(fd)));
                worst = std::isfinite(rel) ? std::max(worst, rel) : INFINITY;
            }
    const size_t n = 161;  // -20 .. 20 step 0.25
    Matrix P(1, n);
    for (size_t j = 0; j < n; ++j) P(0, j) = -20.0f + 0.25f * static_cast<float>(j);
    std::vector<Var> ps;
    for (size_t k = 0; k < n_in; ++k) ps.push_back(make_var(learned(k) ? Matrix(1, n, ins[k].init) : P, true));
    const Var y = ops::expr(ps, e);
    backward(ops::scale(ops::mean(y), static_cast<float>(n)));
    std::vector<float> bad;
    for (size_t j = 0; j < n; ++j) {
        bool ok = std::isfinite(y->data(0, j));
        for (const Var& q : ps) ok = ok && std::isfinite(q->grad(0, j));
        if (!ok) bad.push_back(P(0, j));
    }
    json kinds = json::array();
    for (size_t k = 0; k < n_in; ++k)
        kinds.push_back(k < ins.size() ? json{{"kind", ins[k].kind}, {"init", ins[k].init}} : json("x"));
    json ev = {{"event", "expr_gradcheck"}, {"expr", text}, {"inputs", kinds}, {"fd_points", T * w * n_in},
               {"fd_h", h}, {"fd_worst_rel", worst}, {"fd_tol", 2e-3},
               {"probe", n_in > 1 ? "rows in [-20, 20] step 0.25, learned inputs at their init"
                                  : "x in [-20, 20] step 0.25"},
               {"probe_nonfinite", bad.size()}, {"pass", worst <= 2e-3 && bad.empty()}};
    if (!(worst <= 2e-3))
        throw std::runtime_error("expr " + slot + " \"" + text + "\": autograd gradient disagrees with finite "
                                 "differences (worst relative error " + std::to_string(worst) + " > 2e-3)");
    if (!bad.empty())
        throw std::runtime_error("expr " + slot + " \"" + text + "\": value or gradient not finite at x = " +
                                 std::to_string(bad[0]) + " (" + std::to_string(bad.size()) +
                                 " of 161 probe points in [-20, 20])");
    return ev;
}

// The flex family's configuration from a spec (training, export --from and chat build the same model).
parity::FlexConfig flex_config(const Spec& s, size_t vocab) {
    parity::FlexConfig fc;
    fc.vocab = vocab;
    fc.d = s.d;
    fc.n_layers = s.layers;
    fc.n_heads = s.heads;
    fc.d_ff = s.d_ff ? s.d_ff : 4 * s.d;
    fc.n_ctx = s.T;
    if (!s.norm.empty()) fc.norm = s.norm;
    // "gelu_tanh" names the tanh form explicitly (the arXiv chain writes it); flex's "gelu" is that
    // form.
    if (!s.activation.empty()) fc.act = s.activation == "gelu_tanh" ? "gelu" : s.activation;
    if (!s.position.empty()) fc.pos = s.position;
    if (!s.residual.empty()) fc.residual = s.residual;
    fc.gate_bias_init = s.gate_bias_init;
    fc.attention = s.attention;
    fc.window = s.window;
    fc.sinks = s.sinks;
    fc.act_inputs = s.act_inputs;
    fc.norm_inputs = s.norm_inputs;
    fc.embed_expr = s.embed_expr;
    fc.embed_inputs = s.embed_inputs;
    fc.embed_before_pos = s.embed_before_pos;
    fc.tie_head = s.tie_head;
    fc.place_attn = s.place_attn;
    fc.place_ffn = s.place_ffn;
    fc.final_norm = s.final_norm;
    fc.rel_buckets = s.rel_buckets;
    fc.rel_max_distance = s.rel_max_distance;
    fc.rel_per_head = s.rel_per_head;
    fc.rel_values = s.rel_values;
    fc.qkv_conv = s.qkv_conv;
    fc.qk_norm = s.flex_qk_norm;
    fc.rope_theta = s.rope_theta;
    fc.logits_expr = s.logits_expr;
    const std::string bad = fc.problem();
    if (!bad.empty()) throw std::runtime_error("arch.custom: " + bad);
    return fc;
}

// export.draft_table (defined with the child-process helpers below).
std::string draft_table_tool();
void write_draft_table(const Spec& s, const std::string& out_dir, const std::function<void(const json&)>& emit);

int run(const Spec& s, bool plan_only, const std::string& spec_path = "") {
    std::unique_ptr<studio_probes::ArtifactGuard> probe_guard;
    if (!plan_only && !s.evaluation_probes.is_null())
        probe_guard =
            std::make_unique<studio_probes::ArtifactGuard>(s.out_dir + "/evaluation_probes.json");
    std::printf("== mtstudio: %s ==\n", s.name.c_str());
    std::printf("arch: %s d=%zu layers=%zu heads=%zu | T=%zu vocab_cap=%zu\n", s.attention.c_str(),
                s.d, s.layers, s.heads, s.T, s.vocab_cap);
    std::printf(
        "train: %d steps batch=%d accum=%d lr=%g clip=%g eval_every=%d "
        "ckpt_every=%d early_stop(patience=%zu, min_delta=%g)\n",
        s.steps, s.batch, s.accum, s.lr, s.clip, s.eval_every, s.ckpt_every, s.es_patience,
        s.es_min_delta);
    std::string quants;
    for (const auto& q : s.exp_gguf_quants) quants += " gguf_" + q;
    std::printf("export: %s%s%s | serve: %s | out: %s\n", s.exp_safetensors ? "safetensors " : "",
                s.exp_gguf ? "gguf" : "", quants.c_str(), s.serve ? "yes" : "no", s.out_dir.c_str());
    if (s.draft_table) {
        std::printf("draft_table: %s from %s\n", (s.out_dir + "/" + s.name + ".draft_table.bin").c_str(),
                    s.draft_sources.empty() ? "data.sft_examples" : "export.draft_table.sources");
        // Before training, not after it: a missing tool must not cost a run.
        if (!plan_only && !std::filesystem::exists(draft_table_tool()))
            throw std::runtime_error("export.draft_table: " + draft_table_tool() +
                                     " not found (run from the repository root, or set MTSTUDIO_DRAFT_TABLE)");
    }
    if (s.family == "llama")
        std::printf("rope_heads: %s\n", s.rope_heads.empty()
                                            ? "unstated (all for a new run; a resumed run keeps "
                                              "its recorded value, first if none)"
                                            : s.rope_heads.c_str());
    if (s.rec_block > 0)
        std::printf("recursion: prelude %zu + block %zu x %zu + coda %zu = %zu layer applications over %zu stored "
                    "layers (schedule %s, inject %s)\n",
                    s.rec_prelude, s.rec_block, s.rec_repeats, s.rec_coda, effective_layers(s), s.layers,
                    s.rec_schedule.c_str(), s.rec_inject.c_str());
    if (s.attn_res)
        std::printf("attn_res: full (sublayer level), scale %s, %zu depth sources, %zu zero-initialised "
                    "pseudo-queries of %zu\n",
                    s.attn_res_scale.c_str(), 2 * effective_layers(s) + 1, 2 * effective_layers(s), s.d);
    if (routed(s))
        std::printf("mor: causal token choice over depths 1..%zu (router %zu x %zu at step 0's input), gate %s, "
                    "balance %g, z_loss %g, kv_rows %s\n",
                    s.rec_repeats, s.rec_repeats, s.d, s.rec_mor_gate.c_str(), s.rec_mor_balance, s.rec_mor_z,
                    s.rec_mor_kv.c_str());
    if (s.state_k > 0)
        std::printf("state_tokens: k %zu after every finished reply, mask %s, %zu parameters\n", s.state_k,
                    s.state_mask.c_str(), s.state_k * s.d);
    if (s.moe_experts > 0)
        std::printf("moe: %zu experts, top_k=%zu, gate=%s, aux_loss=%g, z_loss=%g%s%s\n", s.moe_experts,
                    s.moe_top_k, s.moe_gate.c_str(), s.moe_aux, s.moe_z,
                    s.upcycle_from.empty() ? "" : ", upcycle_from=", s.upcycle_from.c_str());
    if (plan_only) return 0;
    if (s.vocab_gguf.empty() || (s.objective == "lm" && s.corpus.empty()))
        throw std::runtime_error("spec needs data.vocab, and data.corpus for objective lm");
    // Only the kimi/srd parity lanes are depth-fixed; flex and llama take
    // any depth, and attnres wires s.layers into its stack.
    if (s.family == "gpt2" && s.attention != "attnres" && s.layers != 2)
        throw std::runtime_error(
            "kimi/srd parity lanes: layers must be 2 "
            "(exact/swa at depth ride the flex family)");

    std::filesystem::create_directories(s.out_dir);
    // Resuming = a checkpoint step > 0 in state.txt. A resumed run keeps the
    // RoPE coverage it was trained with (resolve_rope_heads).
    int ckpt_step = 0;
    {
        std::ifstream st(s.out_dir + "/state.txt");
        std::string line1;
        if (std::getline(st, line1)) ckpt_step = std::atoi(line1.c_str());
    }
    const std::string rope_heads = resolve_rope_heads(s, ckpt_step > 0);
    if (!spec_path.empty()) {
        // Keep the spec beside its outputs: `mtstudio chat <out_dir>`
        // (and anything else handed only a run directory) rebuilds the
        // model from it. Read fully before writing, so a spec that already
        // lives at out_dir/spec.json is rewritten unchanged. A llama spec
        // gets its resolved arch.rope_heads written in.
        std::ifstream sf(spec_path, std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(sf)), std::istreambuf_iterator<char>());
        sf.close();
        if (s.family == "llama" && !text.empty()) {
            json j = json::parse(text, nullptr, false, /*ignore_comments=*/true);
            if (!j.is_discarded() && j.is_object()) {
                if (!j.contains("arch") || !j["arch"].is_object()) j["arch"] = json::object();
                if (j["arch"].value("rope_heads", "") != rope_heads) {
                    j["arch"]["rope_heads"] = rope_heads;
                    text = j.dump(2) + "\n";
                }
                // An MoE spec records its resolved gate convention, so a
                // later default change can never reinterpret a trained run.
                if (s.moe_experts > 0 && j["arch"].contains("moe") && j["arch"]["moe"].is_object() &&
                    j["arch"]["moe"].value("gate", "") != s.moe_gate) {
                    j["arch"]["moe"]["gate"] = s.moe_gate;
                    text = j.dump(2) + "\n";
                }
            }
        }
        if (!text.empty()) std::ofstream(s.out_dir + "/spec.json", std::ios::binary) << text;
    }
    // Resuming? Trim the event log to the checkpoint before appending.
    if (ckpt_step > 0) truncate_events_after(s.out_dir + "/events.jsonl", ckpt_step);
    Events ev(s.out_dir + "/events.jsonl");
    ev.emit({{"event", "start"}, {"name", s.name}, {"steps", s.steps}});

    // Data.
    auto tokens = read_gguf_vocab(s.vocab_gguf);
    if (g_bpe && s.vocab_cap > 0 && s.vocab_cap < tokens.size())
        throw std::runtime_error("data.vocab_cap " + std::to_string(s.vocab_cap) + " is below the BPE vocabulary's " +
                                 std::to_string(tokens.size()) + " tokens: merged ids would fall outside the model");
    if (s.vocab_cap > 0 && s.vocab_cap < tokens.size()) tokens.resize(s.vocab_cap);
    std::map<std::string, int> vocab;
    for (size_t i = 0; i < tokens.size(); ++i) vocab.emplace(tokens[i], static_cast<int>(i));
    if (!s.evaluation_probes.is_null()) try {
            preflight_probe_inputs(s, tokens, vocab);
        } catch (const std::exception& error) {
            studio_probes::publish(
                s.out_dir + "/evaluation_probes.json",
                {{"schema_version", 1},
                 {"status", "ERROR"},
                 {"configuration", s.evaluation_probes},
                 {"checkpoint", nullptr},
                 {"cases", json::array()},
                 {"summary", {{"total", 0}, {"passed", 0}, {"failed", 0}, {"errors", 0}}},
                 {"global_errors", 1},
                 {"error", studio_probes::safe_error(error)}});
            throw;
        }
    const bool sft = s.objective == "sft" || s.objective == "rejection_sft";
    const bool grpo = s.objective == "grpo";
    const bool po = s.objective == "provenance_po";
    std::vector<int> ids, anneal_ids;
    size_t val_start = 0;
    if (!s.anneal_corpus.empty() && s.objective != "lm")
        throw std::runtime_error("data.anneal is for objective lm only");
    SftSet sd;
    GrpoData gd;
    PrefData pd;
    std::vector<size_t> po_train;  // training pairs with weight > 0
    if (po) {
        pd = load_pref(s, vocab);
        std::map<std::string, double> wsum;
        std::map<std::string, size_t> wcount;
        double sw = 0.0, sw2 = 0.0;
        for (size_t i = 0; i < pd.val_start; ++i) {
            if (pd.w[i] > 0.0f) po_train.push_back(i);
            sw += pd.w[i];
            sw2 += static_cast<double>(pd.w[i]) * pd.w[i];
            wsum[pd.kind[i]] += pd.w[i];
            ++wcount[pd.kind[i]];
        }
        if (po_train.empty()) throw std::runtime_error("provenance_po: every training pair has weight 0");
        ev.emit({{"event", "data"},
                 {"objective", s.objective},
                 {"pairs", pd.w.size()},
                 {"val_pairs", pd.w.size() - pd.val_start},
                 {"pairs_with_weight", po_train.size()},
                 {"weight_sum", sw},
                 {"ess_kish", sw2 > 0 ? sw * sw / sw2 : 0.0},
                 {"weight_mode", s.po_weight_mode},
                 {"weight_sum_by_kind", wsum},
                 {"pairs_by_kind", wcount},
                 {"chosen_open", pd.chosen_open},
                 {"rejected_open", pd.rejected_open},
                 {"beta", s.po_beta},
                 {"nll", s.po_nll},
                 {"vocab", tokens.size()}});
    } else if (grpo) {
        gd = load_grpo(s, vocab);
        ev.emit({{"event", "data"},
                 {"objective", s.objective},
                 {"prompts", gd.prompts.size()},
                 {"eval_prompts", gd.eval_prompts.size()},
                 {"vocab", tokens.size()},
                 {"reward", gd.reward->name()},
                 {"group_size", s.grpo_group},
                 {"prompts_per_step", s.grpo_prompts_per_step},
                 {"beta", s.grpo_beta},
                 {"temperature", s.grpo_temp},
                 {"std_norm", s.grpo_std_norm},
                 {"advantage", s.grpo_advantage},
                 {"reward_shift", s.grpo_reward_shift}});
    } else if (!sft) {
        ids = read_corpus(s.corpus, vocab, tokens.size(), s.max_tokens);
        if (s.state_k > 0) {
            // State tokens (IR_SPEC_DRAFT 1.5): inserted into the stream before
            // windowing; the held-out tail is cut from the stream with them.
            const nn::StateTemplate stt = state_template(s, vocab);
            const size_t before = ids.size();
            nn::StateInsertCount c;
            ids = nn::insert_state_ids(ids, stt, &c);
            ev.emit({{"event", "state_tokens"}, {"k", s.state_k}, {"mask", s.state_mask}, {"corpus", "data.corpus"},
                     {"tokens_before", before}, {"tokens_after", ids.size()}, {"blocks", c.blocks},
                     {"closed_early", c.closed}});
        }
        // Hold out the tail 5% for validation (early stopping's signal).
        val_start = ids.size() - ids.size() / 20;
        // Both parts must hold a window of T + 1 targets: the window draws below are
        // unsigned, and a shorter part wrapped round and read past the corpus.
        if (val_start < s.T + 2 || ids.size() - val_start < s.T + 2)
            throw std::runtime_error("data.corpus has " + std::to_string(ids.size()) + " tokens: its training part and "
                                     "held-out 5% each need at least data.T + 2 = " + std::to_string(s.T + 2));
        ev.emit({{"event", "data"},
                 {"tokens", ids.size()},
                 {"vocab", tokens.size()},
                 {"val_tokens", ids.size() - val_start}});
        if (!s.anneal_corpus.empty()) {
            anneal_ids = read_corpus(s.anneal_corpus, vocab, tokens.size(), 0);
            if (s.state_k > 0) {
                const size_t before = anneal_ids.size();
                nn::StateInsertCount c;
                anneal_ids = nn::insert_state_ids(anneal_ids, state_template(s, vocab), &c);
                ev.emit({{"event", "state_tokens"}, {"k", s.state_k}, {"mask", s.state_mask},
                         {"corpus", "data.anneal"}, {"tokens_before", before}, {"tokens_after", anneal_ids.size()},
                         {"blocks", c.blocks}, {"closed_early", c.closed}});
            }
            if (anneal_ids.size() < static_cast<size_t>(s.T) + 2)
                throw std::runtime_error("data.anneal corpus is shorter than one training window");
            ev.emit({{"event", "anneal_data"},
                     {"tokens", anneal_ids.size()},
                     {"from_step", static_cast<int>(s.anneal_from * s.steps)}});
        }
    } else {
        sd = load_sft(s, vocab);
        ev.emit({{"event", "data"},
                 {"objective", s.objective},
                 {"examples", sd.x.size()},
                 {"val_examples", sd.x.size() - sd.val_start},
                 {"tokens", sd.target_tokens},
                 {"vocab", tokens.size()},
                 {"completion_unk", sd.completion_unk},
                 {"provenance", sd.provenance}});
        if (s.state_k > 0)
            ev.emit({{"event", "state_tokens"}, {"k", s.state_k}, {"mask", s.state_mask},
                     {"corpus", "data.sft_examples"}, {"blocks", sd.state_blocks}, {"closed_early", sd.state_closed},
                     {"turns_cut_over_T", sd.state_turns_cut}, {"rows_dropped_over_T", sd.state_rows_dropped}});
        if (sd.conv_rows > 0)
            ev.emit({{"event", "sft_conversations"}, {"chat_template", chat_template_of(s).id()},
                     {"sft_loss", {{"turns", s.sft_turns}}}, {"conversations", sd.conv_rows}, {"rows", sd.conv_out},
                     {"dropped_over_T", sd.conv_over_T}, {"exchanges_cut_over_T", sd.conv_exchanges_cut},
                     {"untrained", sd.conv_untrained}});
    }

    // Model + optimizer (+ resume). Two families behind one seam: the
    // gpt2 parity model (exact/kimi/srd attention) or nn::Llama (RMSNorm/
    // RoPE/SwiGLU, HF names -> GGUF-exportable).
    std::shared_ptr<parity::ParityLM> gpt;
    std::shared_ptr<nn::Llama> llama;
    std::shared_ptr<parity::AttnResLM> attnres;
    std::shared_ptr<parity::FlexLM> flex;
    std::shared_ptr<parity::MambaLM> mamba;
    if (s.family == "mamba") {
        mamba = std::make_shared<parity::MambaLM>(tokens.size(), s.d, s.layers, s.d_state, s.seed, s.mamba1());
        if (s.ckpt_act)
            throw std::runtime_error("train.checkpoint_activations requires the llama family for now");
    } else if (s.family == "flex") {
        // The paper-faithful decoder: every extracted flavor is
        // constructor-real (norm/activation/position/d_ff/depth).
        parity::FlexConfig fc = flex_config(s, tokens.size());
        if (fc.norm.rfind("expr:", 0) == 0) {
            // A runtime-defined norm (e(x) * g + b, parity_model.hpp
            // FlexConfig): its expr gradient-checked before training, as an
            // expr activation's is.
            json gc = expr_gradcheck(fc.norm.substr(5), "norm", fc.norm_inputs);
            gc["slot"] = "norm";
            ev.emit(gc);
            std::printf("expr norm %s: gradient vs finite differences worst rel %.2e, finite on [-20, 20]\n",
                        fc.norm.substr(5).c_str(), gc["fd_worst_rel"].get<double>());
        } else if (fc.norm != "layernorm" && fc.norm != "rmsnorm")
            throw std::runtime_error("unknown norm " + fc.norm);
        if (!fc.logits_expr.empty()) {
            json gc = expr_gradcheck(fc.logits_expr.substr(5), "logits");
            gc["slot"] = "logits";
            ev.emit(gc);
            std::printf("expr logits %s: gradient vs finite differences worst rel %.2e\n", fc.logits_expr.substr(5).c_str(),
                        gc["fd_worst_rel"].get<double>());
        }
        if (!fc.embed_expr.empty()) {
            json gc = expr_gradcheck(fc.embed_expr.substr(5), "embed", fc.embed_inputs);
            gc["slot"] = "embed";
            ev.emit(gc);
            std::printf("expr embed %s: gradient vs finite differences worst rel %.2e\n", fc.embed_expr.substr(5).c_str(),
                        gc["fd_worst_rel"].get<double>());
        }
        if (fc.act.rfind("expr:", 0) == 0) {
            // A runtime-defined activation: parse (throws naming the column),
            // then its gradient against finite differences before training.
            json gc = expr_gradcheck(fc.act.substr(5), "activation", fc.act_inputs);
            gc["slot"] = "activation";
            ev.emit(gc);
            std::printf("expr activation %s: gradient vs finite differences worst rel %.2e, finite on "
                        "[-20, 20]\n", fc.act.substr(5).c_str(), gc["fd_worst_rel"].get<double>());
        } else if (fc.act != "gelu" && fc.act != "gelu_erf" && fc.act != "relu" &&
                   fc.act != "swiglu")
            throw std::runtime_error("unknown activation " + fc.act);
        if (fc.pos != "learned" && fc.pos != "sinusoidal" && fc.pos != "rope" && fc.pos != "t5_relative" &&
            fc.pos != "shaw_relative")
            throw std::runtime_error("unknown position " + fc.pos +
                                     " (learned | sinusoidal | rope | t5_relative | shaw_relative)");
        if (fc.residual != "residual" && fc.residual != "highway" && fc.residual != "plain")
            throw std::runtime_error("unknown residual " + fc.residual +
                                     " (residual | highway | plain)");
        if (fc.attention != "exact" && fc.attention != "swa")
            throw std::runtime_error("flex attention must be exact or swa, got " + fc.attention);
        flex = std::make_shared<parity::FlexLM>(fc, s.seed);
        if (s.ckpt_act)
            throw std::runtime_error(
                "train.checkpoint_activations requires the llama family for now");
    } else if (s.attention == "attnres") {
        // TECH_TRANSFER item 1 as a preset: the residual stream replaced
        // by attention over depth (nn::AttnResStack, K3 block form).
        attnres =
            std::make_shared<parity::AttnResLM>(tokens.size(), s.d, s.heads, s.T, s.seed, s.layers);
        if (s.ckpt_act) {
            throw std::runtime_error(
                "train.checkpoint_activations requires the llama family for now");
        }
    } else if (s.family == "llama") {
        nn::LlamaConfig lc;
        lc.vocab = tokens.size();
        lc.d = s.d;
        lc.n_layers = s.layers;
        lc.n_heads = s.heads;
        lc.d_ff = s.d_ff ? s.d_ff : 3 * s.d;
        lc.n_ctx = s.T;
        lc.rope_all_heads = rope_heads == "all";
        apply_moe(lc, s);
        apply_attn_res(lc, s);
        apply_recursion(lc, s);
        apply_state(lc, s);
        apply_thirdparty(lc, s);
        llama = std::make_shared<nn::Llama>(lc, s.seed);
        llama->checkpoint_blocks = s.ckpt_act;
    } else {
        gpt = make_parity(s, tokens.size());
        if (s.ckpt_act) {
            throw std::runtime_error(
                "train.checkpoint_activations requires the llama family for now");
        }
    }
    // Atlas stage-0 structural echo: the run's identity as a data point.
    nn::Module& model_pick =
        mamba     ? static_cast<nn::Module&>(*mamba)
        : flex    ? static_cast<nn::Module&>(*flex)
        : attnres ? static_cast<nn::Module&>(*attnres)
                  : (llama ? static_cast<nn::Module&>(*llama) : static_cast<nn::Module&>(*gpt));
    const size_t n_params = model_pick.parameter_count();
    // Resolved flavor labels — what the constructed model ACTUALLY is,
    // family defaults filled in (the Atlas structural echo must never
    // under-describe the data point).
    const std::string r_norm = flex ? flex->cfg.norm : (llama ? "rmsnorm" : "layernorm");
    const std::string r_act = flex ? flex->cfg.act : (llama ? "swiglu" : "gelu");
    const std::string r_pos = flex ? flex->cfg.pos : (llama ? "rope" : mamba ? "none" : "learned");
    const size_t r_dff = flex ? flex->cfg.d_ff : (llama ? llama->cfg.d_ff : 4 * s.d);
    json model_ev = {{"event", "model"},
                     {"family", s.family},
                     {"attention", s.attention},
                     {"d", s.d},
                     {"layers", s.layers},
                     {"heads", s.heads},
                     {"T", s.T},
                     {"norm", r_norm},
                     {"activation", r_act},
                     {"position", r_pos},
                     {"residual", flex ? flex->cfg.residual : "residual"},
                     {"gate_bias_init", flex ? flex->cfg.gate_bias_init : -2.0f},
                     {"norm_placement", flex ? json{{"attn", flex->cfg.place_attn}, {"ffn", flex->cfg.place_ffn}}
                                             : json("pre")},
                     {"final_norm", flex ? flex->cfg.has_final_norm() : true},
                     {"window", s.window},
                     {"sinks", s.sinks},
                     {"d_ff", r_dff},
                     {"vocab", tokens.size()},
                     {"batch", s.batch},
                     {"accum", s.accum},
                     {"lr", s.lr},
                     {"seed", s.seed},
                     {"checkpoint_activations", s.ckpt_act},
                     {"params", n_params}};
    if (flex && (!flex->cfg.act_inputs.empty() || !flex->cfg.norm_inputs.empty())) {
        auto kinds = [](const std::vector<parity::FlexConfig::ExprInput>& v) {
            json a = json::array();
            for (const auto& i : v) a.push_back({{"kind", i.kind}, {"init", i.init}});
            return a;
        };
        model_ev["activation_inputs"] = kinds(flex->cfg.act_inputs);
        model_ev["norm_inputs"] = kinds(flex->cfg.norm_inputs);
    }
    if (flex && !flex->cfg.embed_expr.empty()) {
        json a = json::array();
        for (const auto& i : flex->cfg.embed_inputs) a.push_back({{"kind", i.kind}, {"init", i.init}});
        model_ev["embed"] = {{"expr", flex->cfg.embed_expr}, {"inputs", a}};
        if (flex->cfg.embed_before_pos) model_ev["embed"]["before_positions"] = true;
    }
    if (flex && flex->cfg.pos == "t5_relative")
        model_ev["t5_relative"] = {{"buckets", flex->cfg.rel_buckets}, {"max_distance", flex->cfg.rel_max_distance}};
    if (flex && flex->cfg.pos == "shaw_relative")
        model_ev["shaw_relative"] = {{"max_distance", flex->cfg.rel_max_distance},
                                     {"per_head", flex->cfg.rel_per_head}, {"values", flex->cfg.rel_values}};
    if (flex && flex->cfg.qkv_conv) model_ev["qkv_conv"] = flex->cfg.qkv_conv;
    if (flex && !flex->cfg.qk_norm.empty()) model_ev["qk_norm"] = flex->cfg.qk_norm;
    if (flex && !flex->cfg.logits_expr.empty()) model_ev["logits"] = flex->cfg.logits_expr;
    if (flex && flex->cfg.tie_head) model_ev["tie_head"] = true;
    // The GELU's form, where the activation is one (keramos IR_SPEC 5.8's approx): flex's "gelu" is
    // the tanh form.
    if (flex && (r_act == "gelu" || r_act == "gelu_erf"))
        model_ev["gelu_approx"] = r_act == "gelu" ? "tanh" : "none";
    if (s.lr_warmup > 0 || s.lr_decay != "none")
        model_ev["lr_schedule"] = {
            {"warmup_steps", s.lr_warmup}, {"decay", s.lr_decay}, {"min_lr_frac", s.lr_min_frac}};
    if (llama) model_ev["rope_heads"] = rope_heads;
    // Absent at the defaults: the plain llama's model event is unchanged.
    if (llama && s.thirdparty())
        model_ev["thirdparty"] = {{"kv_heads", llama->cfg.kv_heads()}, {"rope_pairing", s.rope_pairing},
                                  {"rope_dims", s.rope_dims ? s.rope_dims : s.d / s.heads}, {"qk_norm", s.qk_norm}};
    // Absent without qk_temp: every other model event is unchanged.
    if (llama && s.qk_temp != 0.0f) model_ev["thirdparty"]["qk_temp"] = s.qk_temp;
    if ((llama || flex) && s.rope_theta != 10000.0f) model_ev["rope_theta"] = s.rope_theta;
    // Absent for the S4-style mamba: its model event is unchanged.
    if (mamba && s.ssm == "mamba1")
        model_ev["ssm"] = {{"kind", "mamba1"}, {"d_state", s.d_state}, {"d_conv", s.d_conv}, {"expand", s.expand},
                           {"dt_rank", mamba->m1blocks.at(0)->R}};
    // Absent without state tokens: the model event is unchanged.
    if (s.state_k > 0)
        model_ev["state_tokens"] = {{"k", s.state_k}, {"mask", s.state_mask}, {"params", s.state_k * s.d}};
    // Absent without recursion: the model event is unchanged.
    if (s.rec_block > 0)
        model_ev["recursion"] = {{"block", s.rec_block},     {"repeats", s.rec_repeats},
                                 {"prelude", s.rec_prelude}, {"coda", s.rec_coda},
                                 {"schedule", s.rec_schedule}, {"inject", s.rec_inject},
                                 {"effective_layers", effective_layers(s)}};
    if (routed(s))
        model_ev["recursion"]["mor"] = {{"route", s.rec_mor_route}, {"gate", s.rec_mor_gate},
                                        {"balance", s.rec_mor_balance}, {"z_loss", s.rec_mor_z},
                                        {"kv_rows", s.rec_mor_kv}, {"router_params", s.d * s.rec_repeats}};
    // Absent without BitLinear: the model event is unchanged.
    if (s.bitlinear)
        model_ev["bitlinear"] = {
            {"weights", "ternary"}, {"act_bits", s.bl_act_bits}, {"in_norm", s.bl_in_norm}};
    if (!s.tie_embeddings) model_ev["tie_embeddings"] = false;
    // Absent without attention residuals: the model event is unchanged.
    if (s.attn_res)
        model_ev["attn_res"] = {{"form", "full"}, {"scale", s.attn_res_scale},
                                {"sources", 2 * effective_layers(s) + 1}, {"query_params", 2 * effective_layers(s) * s.d}};
    // Absent for dense models: their model event is unchanged.
    if (s.moe_experts > 0)
        model_ev["moe"] = {{"experts", s.moe_experts},
                           {"top_k", s.moe_top_k},
                           {"gate", s.moe_gate},
                           {"aux_loss", s.moe_aux},
                           {"z_loss", s.moe_z}};
    if (sft || grpo || po) model_ev["objective"] = s.objective;
    // Process-wide for this run (eval included); absent from the model event
    // at the defaults, so FP32 tiled runs' events are unchanged.
    device::set_matmul_precision(s.precision == "bf16" ? device::Precision::BF16 : device::Precision::FP32);
    ops::set_attention_kernel(s.attention_kernel == "flash" ? ops::AttnKernel::Flash : ops::AttnKernel::Tiled);
    if (s.precision != "fp32") model_ev["precision"] = s.precision;
    if (s.attention_kernel != "tiled") model_ev["attention_kernel"] = s.attention_kernel;
    if (s.fused_head) model_ev["fused_head"] = true;  // absent when off: events unchanged
    if (s.fused_head_chunk) model_ev["fused_head_chunk"] = s.fused_head_chunk;
    if (ops::blocked_attention_enabled()) model_ev["attention_impl"] = "blocked";  // absent = stacked  // absent for "lm": events unchanged
    ev.emit(model_ev);
    nn::Module& model_ref = model_pick;
    auto fwd = [&](const std::vector<int>& ids, size_t seq_len = 0) {
        if (mamba) return mamba->forward(ids, seq_len);
        if (flex) return flex->forward(ids, seq_len);
        if (attnres) return attnres->forward(ids, seq_len);
        return llama ? llama->forward(ids, seq_len) : gpt->forward(ids, seq_len);
    };
    // batch > 1 is supported for every family and attention kind: exact
    // via the block-diagonal fused mask, kimi via per-block prefix-sum
    // reset, srd through both of its paths.
    model_ref.train();
    // Optimizer. "muon" is the deployment-faithful hybrid (TECH_TRANSFER
    // item 3): per-head Muon on qkv projections (columns are head-major in
    // the [in, out] layout; fused c_attn carries 3*H head blocks), full-
    // matrix Muon on the remaining hidden matrices, AdamW for embeddings,
    // vectors and the head — Muon is never applied outside its remit.
    struct Optim {
        std::vector<nn::AdamW> adamw;
        std::vector<nn::Muon> muon;
        void zero_grad() {
            for (auto& o : adamw) o.zero_grad();
            for (auto& o : muon) o.zero_grad();
        }
        void step() {
            for (auto& o : adamw) o.step();
            for (auto& o : muon) o.step();
        }
        // Scale every optimiser's rate from its base (the spec's lr / muon_lr).
        void set_lr_scale(float f, float adamw_lr, float muon_lr) {
            for (auto& o : adamw) o.lr = adamw_lr * f;
            for (auto& o : muon) o.lr = muon_lr * f;
        }
    } opt;
    // train.lr_schedule as a multiplier of the base rate at `step` (1-based).
    auto lr_scale = [&s](int step) -> float {
        if (s.lr_warmup > 0 && static_cast<size_t>(step) <= s.lr_warmup)
            return static_cast<float>(step) / static_cast<float>(s.lr_warmup);
        if (s.lr_decay != "cosine" || s.steps <= s.lr_warmup) return 1.0f;
        const double p = static_cast<double>(step - static_cast<int>(s.lr_warmup)) /
                         static_cast<double>(s.steps - s.lr_warmup);
        const double cosine = 0.5 * (1.0 + std::cos(3.14159265358979323846 * std::min(1.0, p)));
        return static_cast<float>(s.lr_min_frac + (1.0 - s.lr_min_frac) * cosine);
    };
    const bool scheduled = s.lr_warmup > 0 || s.lr_decay != "none";
    if (s.optimizer == "muon") {
        std::vector<Var> qkv, hidden, rest;
        for (const auto& [name, p] : model_ref.named_parameters()) {
            const bool matrix = p->data.rows() > 1 && p->data.cols() > 1;
            const bool excluded =
                name.find("embed") != std::string::npos || name.find("wte") != std::string::npos ||
                name.find("wpe") != std::string::npos || name.find("head") != std::string::npos ||
                name.find("norm") != std::string::npos || name.find("ln") != std::string::npos;
            const bool is_qkv = name.find("c_attn") != std::string::npos ||
                                name.find("q_proj") != std::string::npos ||
                                name.find("k_proj") != std::string::npos ||
                                name.find("v_proj") != std::string::npos;
            if (matrix && !excluded && is_qkv)
                qkv.push_back(p);
            else if (matrix && !excluded)
                hidden.push_back(p);
            else
                rest.push_back(p);
        }
        if (s.kv_heads && s.kv_heads != s.heads)
            throw std::runtime_error("optimizer muon: its per-head qkv split assumes one kv head per query head; "
                                     "grouped-query attention (arch.custom.kv_heads) is not wired with it");
        const size_t nh = s.family == "llama" ? s.heads : 3 * s.heads;
        if (!qkv.empty()) opt.muon.emplace_back(qkv, s.muon_lr, 0.95f, true, 5, nh);
        if (!hidden.empty()) opt.muon.emplace_back(hidden, s.muon_lr);
        if (!rest.empty()) opt.adamw.emplace_back(rest, s.lr);
        std::printf(
            "optimizer: muon hybrid — %zu qkv (per-head n=%zu), %zu hidden, "
            "%zu adamw\n",
            qkv.size(), nh, hidden.size(), rest.size());
    } else {
        opt.adamw.emplace_back(model_ref.parameters(), s.lr);
    }
    // Checkpoint = three files. model.safetensors (weights),
    // optim.safetensors (AdamW m/v and Muon momentum, every optimizer
    // instance), and state.txt: line 1 the step (the pre-existing
    // contract; anything that only reads that line still works), line 2 a
    // JSON record with the AdamW timesteps and the early-stopping state.
    // A checkpoint missing optim.safetensors resumes with a COLD optimizer
    // and says so in the resume event — that is the pre-3-Sep behaviour,
    // kept for old checkpoints, never silent.
    const std::string ckpt = s.out_dir + "/model.safetensors";
    const std::string optim_path = s.out_dir + "/optim.safetensors";
    const std::string state_path = s.out_dir + "/state.txt";
    int start_step = 0;
    float best_val = 1e30f;
    size_t evals_flat = 0;
    {
        std::ifstream st(state_path);
        std::string line1, line2;
        if (std::getline(st, line1) && (start_step = std::atoi(line1.c_str())) > 0) {
            model_ref.load_state_dict(load_safetensors(ckpt));
            std::string optim_status = "cold";
            std::ifstream of(optim_path);
            if (of.good()) {
                of.close();
                const auto osd = load_safetensors(optim_path);
                for (size_t i = 0; i < opt.adamw.size(); ++i)
                    opt.adamw[i].load_state_dict(osd, "adamw." + std::to_string(i));
                for (size_t i = 0; i < opt.muon.size(); ++i)
                    opt.muon[i].load_state_dict(osd, "muon." + std::to_string(i));
                optim_status = "restored";
            }
            if (std::getline(st, line2) && !line2.empty()) {
                const json rec = json::parse(line2, nullptr, false);
                if (!rec.is_discarded()) {
                    best_val = rec.value("best_val", best_val);
                    evals_flat = rec.value("evals_flat", evals_flat);
                    if (rec.contains("adamw_t"))
                        for (size_t i = 0; i < opt.adamw.size() && i < rec["adamw_t"].size(); ++i)
                            opt.adamw[i].set_t(rec["adamw_t"][i].get<long>());
                }
            }
            ev.emit({{"event", "resume"},
                     {"step", start_step},
                     {"optimizer", optim_status},
                     {"best_val", best_val},
                     {"evals_flat", evals_flat}});
        } else
            start_step = 0;
    }
    if (start_step == 0 && !s.init_from.empty()) {
        // Post-training starts from a trained base; a resumed run already
        // carries those weights in its checkpoint.
        model_ref.load_state_dict(load_safetensors(s.init_from), /*strict=*/true);
        ev.emit({{"event", "init"}, {"from", s.init_from}});
    }
    if (start_step == 0 && !s.upcycle_from.empty()) {
        // Sparse upcycling: the dense FFN into every expert, everything else
        // copied; the routers keep their fresh (small random) init. Strict
        // otherwise: every dense tensor must land and only routers may be missing.
        std::vector<std::string> routers;
        const auto up = nn::upcycle_state_dict(load_safetensors(s.upcycle_from), s.moe_experts, &routers);
        model_ref.load_state_dict(up, /*strict=*/true, routers);
        ev.emit({{"event", "upcycle"},
                 {"from", s.upcycle_from},
                 {"experts", s.moe_experts},
                 {"routers_fresh", routers.size()}});
    }

    // Data order follows the spec seed so multi-seed sweeps vary both init
    // and batch composition (offset keeps seed=7 runs distinct from the
    // old fixed-123 stream only in the documented way).
    std::mt19937 rng(123 + 1000003u * s.seed);
    const auto t_train0 = std::chrono::steady_clock::now();
    int last_step = start_step;
    // Resume determinism: each step consumed accum*batch draws.
    for (int i = 0; i < start_step * s.accum * s.batch; ++i) rng();
    auto save = [&](int step) {
        save_safetensors(ckpt, model_ref.state_dict());
        std::map<std::string, Matrix> osd;
        std::vector<long> ts;
        for (size_t i = 0; i < opt.adamw.size(); ++i) {
            auto part = opt.adamw[i].state_dict("adamw." + std::to_string(i));
            osd.insert(part.begin(), part.end());
            ts.push_back(opt.adamw[i].t());
        }
        for (size_t i = 0; i < opt.muon.size(); ++i) {
            auto part = opt.muon[i].state_dict("muon." + std::to_string(i));
            osd.insert(part.begin(), part.end());
        }
        save_safetensors(optim_path, osd);
        std::ofstream st(state_path);
        st << step << "\n";
        st << json({{"best_val", best_val}, {"evals_flat", evals_flat}, {"adamw_t", ts}}).dump()
           << "\n";
    };

    // grpo: the frozen reference (train.init_from) for the KL term.
    std::function<Var(const std::vector<int>&, size_t)> ref_fwd;
    if (grpo || po) ref_fwd = make_reference(s);
    const int eos_tok = wordtok::eos_id(vocab);
    // provenance_po: the reference margin of every pair, once (it is constant).
    std::vector<double> po_ref(pd.w.size(), 0.0);
    if (po) {
        NoGrad ng;
        for (size_t i = 0; i < pd.w.size(); ++i) {
            const PrefBatch b = pref_rows(pd, {i}, eos_tok >= 0 ? eos_tok : 0);
            po_ref[i] = pair_margins(ref_fwd(b.x, b.L)->data, b, 1)[0];
        }
    }

    // MICROTORCH_STEP_TIMING=1: per-phase wall time on each step event
    // ("t_ms"). Each mark waits for the device first, so the phases add up
    // to the step; the waits cost a little throughput, so it is a profiling
    // switch, never on for a timed run. It also logs the device pool's
    // live MiB at each mark ("live_mb"), the step's high-water mark
    // ("peak_mb") and the pool's cache ("cached_mb").
    const bool step_timing = [] {
        const char* v = std::getenv("MICROTORCH_STEP_TIMING");
        return v && std::strcmp(v, "1") == 0;
    }();
    json t_ms, live_mb;
    auto t_last = std::chrono::steady_clock::now();
    auto mib = [](size_t b) { return static_cast<double>(b) / (1 << 20); };
    auto mark = [&](const char* phase) {
        if (!step_timing) return;
        device::synchronize();
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - t_last).count();
        t_ms[phase] = (t_ms.contains(phase) ? t_ms[phase].get<double>() : 0.0) + ms;
        live_mb[phase] = mib(device::pool_live_bytes());
        t_last = now;
    };

    // Train.
    bool stopped_early = false;
    const bool is_srd = attn_kind(s.attention) == parity::AttnKind::SRD;
    for (int step = start_step + 1; step <= s.steps; ++step) {
        last_step = step;
        mark("other");  // eval, checkpoint and logging since the last step
        t_ms = json::object();
        live_mb = json::object();
        if (step_timing) device::pool_reset_peak();
        // The anneal corpus has no held-out part: all of it is training text.
        const bool annealing = !anneal_ids.empty() && step > static_cast<int>(s.anneal_from * s.steps);
        const std::vector<int>& src = annealing ? anneal_ids : ids;
        const size_t lim = (sft || grpo || po) ? 0 : annealing ? anneal_ids.size() - s.T - 1 : val_start - s.T - 1;
        // Mini-batching + accumulation: each of s.accum micro-steps stacks
        // s.batch sequences into ONE forward ([batch*T, d] rows; positions
        // and the attention mask restart per sequence — receipts in
        // tests/test_batching.cpp), backward pre-scaled by 1/accum so the
        // summed gradient is the mean over all batch*accum sequences.
        // Phase B2 step window (docs/CUDA_PHASE_B2.md): device operand
        // caches are trusted only inside it. Closed BEFORE clip/opt
        // mutate host data, so eval and the optimizer can never read a
        // stale device copy. No-op unless MICROTORCH_STEP_RESIDENCY=1
        // on a CUDA build.
        device::step_begin();
        opt.zero_grad();
        mark("zero_grad");
        float task_mean = 0, gate_mean = 0;
        // MoE monitors, averaged over the accum micro-steps (unscaled
        // per-layer means; the loss adds coefficient * these).
        double moe_lb = 0, moe_z = 0;
        std::vector<std::vector<double>> moe_frac;
        double mor_lb = 0, mor_z = 0;
        std::vector<double> mor_frac;  // schedule "mor": the step's depth fractions
        json gmet;
        if (grpo) {
            // Phase 4 step: P prompts x G pure-temperature samples from the
            // current policy, scored in process, group-relative advantages,
            // one gradient step on ops::policy_gradient_kl. Per-token weights
            // carry the constant length normaliser N = P*G*max_new.
            const size_t V = tokens.size();
            const int P = s.grpo_prompts_per_step, G = s.grpo_group;
            struct GS {
                size_t prompt;
                std::vector<int> comp;
                bool eos;
                float reward, exact, fmt;
                bool abstained;
            };
            std::vector<GS> batch;
            {
                NoGrad ng;
                for (int p = 0; p < P; ++p) {
                    const size_t pi = rng() % gd.prompts.size();
                    for (int g = 0; g < G; ++g) {
                        std::seed_seq ss{static_cast<unsigned>(s.seed), static_cast<unsigned>(step),
                                         static_cast<unsigned>(p), static_cast<unsigned>(g)};
                        std::mt19937 gen(ss);
                        const Sampled sm = sample_pure([&](const std::vector<int>& c) { return fwd(c, 0); },
                                                       gd.prompt_ids[pi], s.T, V, s.grpo_temp, s.grpo_max_new, eos_tok,
                                                       gen);
                        const std::string text = ids_to_text(sm.ids, tokens);
                        const auto rr = gd.reward->verify(gd.prompts[pi], text);
                        batch.push_back({pi, sm.ids, sm.stop == "eos", rr.abstained() ? 0.0f : rr.reward,
                                         gd.exact->verify(gd.prompts[pi], text).reward,
                                         gd.format->verify(gd.prompts[pi], text).reward, rr.abstained()});
                    }
                }
            }
            std::vector<float> rewards;
            for (const auto& b : batch) rewards.push_back(b.reward);
            posttrain::GroupStats gst;
            const auto A = posttrain::group_advantages(
                rewards, static_cast<size_t>(G), s.grpo_std_norm, &gst, 1e-6f,
                s.grpo_advantage == "shuffled" ? posttrain::AdvantageMode::Shuffled : posttrain::AdvantageMode::Normal,
                static_cast<unsigned>(s.seed) * 1000003u + static_cast<unsigned>(step));
            std::vector<std::vector<int>> seqs;
            size_t Lmax = 0;
            for (const auto& b : batch) {
                std::vector<int> q = gd.prompt_ids[b.prompt];
                q.insert(q.end(), b.comp.begin(), b.comp.end());
                if (b.eos && eos_tok >= 0) q.push_back(eos_tok);
                Lmax = std::max(Lmax, q.size() - 1);
                seqs.push_back(std::move(q));
            }
            const float N = static_cast<float>(P * G * s.grpo_max_new);
            const int pad = eos_tok >= 0 ? eos_tok : 0;
            std::vector<int> x, y;
            std::vector<float> w, kk;
            for (size_t i = 0; i < batch.size(); ++i) {
                const auto& q = seqs[i];
                const size_t pl = gd.prompt_ids[batch[i].prompt].size();
                for (size_t t = 0; t < Lmax; ++t) {
                    const bool real = t + 1 < q.size();
                    const bool comp = real && t + 1 >= pl;
                    x.push_back(real ? q[t] : pad);
                    y.push_back(real ? q[t + 1] : pad);
                    w.push_back(comp ? A[i] / N : 0.0f);
                    kk.push_back(comp ? s.grpo_beta / N : 0.0f);
                }
            }
            const size_t seq_len = batch.size() > 1 ? Lmax : 0;
            Var logits = fwd(x, seq_len);
            Matrix ref_lp;
            {
                NoGrad ng;
                ref_lp = log_softmax_rows(ref_fwd(x, seq_len)->data);
            }
            Var loss = ops::policy_gradient_kl(logits, y, w, kk, ref_lp);
            backward(loss);
            task_mean = loss->data(0, 0);
            // Monitors over completion tokens: exact KL to the reference
            // and policy entropy (in nats), from this step's own logits.
            const Matrix lp = log_softmax_rows(logits->data);
            double kl = 0.0, ent = 0.0;
            size_t nt = 0;
            for (size_t r = 0; r < kk.size(); ++r) {
                if (kk[r] == 0.0f && w[r] == 0.0f) continue;
                for (size_t j = 0; j < lp.cols(); ++j) {
                    const double l = lp(r, j), pr = std::exp(l);
                    kl += pr * (l - ref_lp(r, j));
                    ent -= pr * l;
                }
                ++nt;
            }
            double rw = 0, ex = 0, hk = 0, rh = 0;
            size_t abst = 0;
            for (const auto& b : batch) {
                rw += b.reward;
                ex += b.exact > 0.5f ? 1.0 : 0.0;
                hk += (b.fmt > 0.5f && b.exact < 0.5f) ? 1.0 : 0.0;
                rh += (b.reward > 0.5f && b.exact < 0.5f) ? 1.0 : 0.0;
                abst += b.abstained;
            }
            const double nb = static_cast<double>(batch.size());
            gmet = {{"reward_mean", rw / nb}, {"truth", ex / nb}, {"hack_frac", hk / nb},
                    {"reward_hack_frac", rh / nb},
                    {"kl", nt ? kl / nt : 0.0}, {"entropy", nt ? ent / nt : 0.0},
                    {"flat_groups", gst.flat_groups}, {"groups", gst.groups}, {"abstained", abst}};
        }
        if (po) {
            // Phase 5 step: s.batch pairs drawn from the weighted training
            // pairs, one dpo_weighted step against the frozen reference.
            std::vector<size_t> pick;
            for (int b = 0; b < s.batch; ++b) pick.push_back(po_train[rng() % po_train.size()]);
            const PrefBatch b = pref_rows(pd, pick, eos_tok >= 0 ? eos_tok : 0);
            std::vector<double> refm;
            std::vector<float> w;
            double sw = 0.0, sw2 = 0.0;
            for (size_t i : pick) {
                refm.push_back(po_ref[i]);
                w.push_back(pd.w[i]);
                sw += pd.w[i];
                sw2 += static_cast<double>(pd.w[i]) * pd.w[i];
            }
            Var logits = fwd(b.x, b.L);
            Var loss = ops::dpo_weighted(logits, b.y, b.row_pair, b.sign, refm, w, s.po_beta);
            double nll_v = 0.0;
            if (s.po_nll > 0.0f) {
                // Weighted mean NLL of the chosen completions:
                //   nll * sum_p w_p mean_t(-log p(chosen_t)) / sum_p w_p,
                // as ops::policy_gradient_kl with k = 0 (already gradchecked).
                std::vector<size_t> nchosen(pick.size(), 0);
                for (size_t r = 0; r < b.y.size(); ++r)
                    if (b.row_pair[r] >= 0 && b.sign[r] == 1) ++nchosen[b.row_pair[r]];
                std::vector<float> wn(b.y.size(), 0.0f), kz(b.y.size(), 0.0f);
                for (size_t r = 0; r < b.y.size(); ++r)
                    if (b.row_pair[r] >= 0 && b.sign[r] == 1) {
                        const int p = b.row_pair[r];
                        wn[r] = static_cast<float>(s.po_nll * w[p] / (sw * nchosen[p]));
                    }
                Var nll = ops::policy_gradient_kl(logits, b.y, wn, kz, Matrix(b.y.size(), logits->data.cols()));
                nll_v = nll->data(0, 0) / s.po_nll;
                loss = ops::add(loss, nll);
            }
            backward(loss);
            task_mean = loss->data(0, 0);
            const auto pm = pair_margins(logits->data, b, pick.size());
            double mm = 0.0, acc = 0.0;
            for (size_t k = 0; k < pick.size(); ++k) {
                mm += pm[k] - refm[k];
                acc += pm[k] - refm[k] > 0 ? 1.0 : 0.0;
            }
            gmet = {{"weight_sum", sw}, {"ess_kish", sw2 > 0 ? sw * sw / sw2 : 0.0},
                    {"margin", mm / pick.size()}, {"pref_acc", acc / pick.size()}, {"chosen_nll", nll_v}};
        }
        for (int k = 0; !grpo && !po && k < s.accum; ++k) {
            std::vector<int> x, y;
            std::vector<uint8_t> mask;
            x.reserve(s.batch * s.T);
            y.reserve(s.batch * s.T);
            for (int b = 0; b < s.batch; ++b) {
                if (sft) {
                    const size_t e = rng() % sd.val_start;
                    x.insert(x.end(), sd.x[e].begin(), sd.x[e].end());
                    y.insert(y.end(), sd.y[e].begin(), sd.y[e].end());
                    mask.insert(mask.end(), sd.m[e].begin(), sd.m[e].end());
                    continue;
                }
                const size_t at = rng() % lim;
                x.insert(x.end(), src.begin() + at, src.begin() + at + s.T);
                y.insert(y.end(), src.begin() + at + 1, src.begin() + at + s.T + 1);
            }
            mark("data");
            nn::MoEAux maux;
            // Declared in the order the unfused step always declared them, so
            // they also die in that order.
            Var logits, task;
            if (!s.fused_head) {
                logits = s.moe_experts > 0 || routed(s) ? llama->forward(x, s.batch > 1 ? s.T : 0, &maux)
                                                        : fwd(x, s.batch > 1 ? s.T : 0);
                mark("forward");
                // State tokens: no loss at state targets or state positions.
                if (!sft && s.state_k > 0) state_loss_mask(x, y, mask, tokens.size());
                task = sft || s.state_k > 0 ? ops::cross_entropy_masked(logits, y, mask) : ops::cross_entropy(logits, y);
            } else {
                // train.fused_head: the hidden states, then the head and the loss
                // in one op (the same mask; the upstream gradient is the 1 / accum
                // the backward below sends).
                const Var hid = llama->forward_hidden(x, s.batch > 1 ? s.T : 0,
                                                      s.moe_experts > 0 || routed(s) ? &maux : nullptr);
                mark("forward");
                if (!sft && s.state_k > 0) state_loss_mask(x, y, mask, tokens.size());
                device::ProfileTag head_tag("gemm_head");
                task = ops::linear_cross_entropy(hid, llama->head_weight(), llama->head_layout(), nullptr, y,
                                                 sft || s.state_k > 0 ? mask : std::vector<uint8_t>{},
                                                 1.0f / static_cast<float>(s.accum), s.fused_head_chunk);
            }
            Var loss = task;
            if (is_srd) loss = ops::add(task, ops::scale(gpt->mean_gate(), s.lambda_gate));
            if (s.moe_experts > 0) {
                // Mean over MoE layers, so the coefficients do not scale with depth.
                const float L = static_cast<float>(maux.layers);
                loss = ops::add(loss, ops::add(ops::scale(maux.lb, s.moe_aux / L), ops::scale(maux.z, s.moe_z / L)));
                moe_lb += maux.lb->data(0, 0) / L / s.accum;
                moe_z += maux.z->data(0, 0) / L / s.accum;
                if (moe_frac.empty()) moe_frac.assign(maux.frac.size(), std::vector<double>(s.moe_experts, 0.0));
                for (size_t l = 0; l < maux.frac.size(); ++l)
                    for (size_t e = 0; e < s.moe_experts; ++e) moe_frac[l][e] += maux.frac[l][e] / s.accum;
            }
            if (routed(s)) {
                // The router's balancing loss and z-loss (amendment 4: 0.1 and 1e-3).
                loss = ops::add(loss, ops::add(ops::scale(maux.lb, s.rec_mor_balance),
                                               ops::scale(maux.z, s.rec_mor_z)));
                mor_lb += maux.lb->data(0, 0) / s.accum;
                mor_z += maux.z->data(0, 0) / s.accum;
                if (mor_frac.empty()) mor_frac.assign(s.rec_repeats, 0.0);
                for (size_t i = 0; i < s.rec_repeats; ++i) mor_frac[i] += maux.frac.at(0)[i] / s.accum;
            }
            mark("loss");
            backward(ops::scale(loss, 1.0f / static_cast<float>(s.accum)));
            mark("backward");
            task_mean += task->data(0, 0) / static_cast<float>(s.accum);
            if (is_srd) gate_mean += gpt->mean_gate()->data(0, 0) / static_cast<float>(s.accum);
        }
        // Per-module grad norms BEFORE clipping: this is the true signal
        // the glow UI wants (clipping would mask explosions).
        const bool want_gm = s.gradmap_every > 0 && step % s.gradmap_every == 0;
        // AdamW alone with deferred downloads: clip and the update run
        // inside the window on the device-fresh grads, and the window then
        // downloads the updated parameters, not every gradient (the grads
        // used to cross the bus twice and the clip loop ran on the host).
        // grad_map and Muon read grads on the host: they close it first.
        const bool opt_in_window = device::deferring() && opt.muon.empty() && !want_gm;
        if (!opt_in_window) device::step_end();
        mark("step_end");
        json gm;
        if (want_gm) gm = grad_map(model_ref);
        const float total_norm = ops::clip_grad_norm(model_ref.parameters(), s.clip);
        mark("clip");
        const float lr_f = scheduled ? lr_scale(step) : 1.0f;
        if (scheduled) opt.set_lr_scale(lr_f, s.lr, s.muon_lr);
        opt.step();
        mark("optimizer");
        if (opt_in_window) {
            // The grads are spent (the next zero_grad rewrites them): drop
            // their device copies instead of downloading them, so host grad
            // storage is NOT current after an in-window step. Nothing reads
            // it between steps on this path (grad_map closes the window first).
            const auto params = model_ref.parameters();
            for (const auto& p : params) device::discard(p->grad);
            // Between two plain training steps nothing reads the parameters
            // on the host, so the updated values stay on the device and the
            // next window adopts them (no download now, no upload then).
            // Eval, checkpoint and the last step read them: download first.
            // MICROTORCH_KEEP_PARAMS=0 downloads every step. The llama
            // family (dense and MoE) only: its forward reads every
            // parameter through the device seams, verified bit for bit.
            // The gpt2-family forward still has host reads of raw parameter
            // storage (a kept run took a wrong step 2, 20 Oct 2026).
            const bool keep_ok = [&llama] {
                const char* v = std::getenv("MICROTORCH_KEEP_PARAMS");
                return llama != nullptr && !(v && std::strcmp(v, "0") == 0);
            }();
            const bool host_next = step % s.eval_every == 0 || step % s.ckpt_every == 0 || step == s.steps;
            if (keep_ok && !host_next) {
                std::vector<const Matrix*> keep;
                for (const auto& p : params) keep.push_back(&p->data);
                device::step_end_keeping(keep);
            } else {
                device::step_end();
            }
            mark("step_end");
        }

        json e = {
            {"event", "step"}, {"step", step}, {"loss", task_mean}, {"grad_norm", total_norm}};
        if (scheduled) e["lr"] = s.lr * lr_f;
        if (step_timing) {
            e["t_ms"] = t_ms;
            e["live_mb"] = live_mb;
            e["peak_mb"] = mib(device::pool_peak_bytes());
            e["cached_mb"] = mib(device::pool_cached_bytes());
        }
        if (device::gpu_profile_on()) {
            json g = json::object();
            for (const auto& [op, ms] : device::gpu_profile_take()) g[op] = ms;
            e["gpu_ms"] = g;
            json gn = json::object();
            for (const auto& [op, n] : device::gpu_profile_counts()) gn[op] = n;
            e["gpu_n"] = gn;  // timed spans per op (launches, one or a few kernels each)
        }
        if (is_srd) e["gate"] = gate_mean;
        if (!moe_frac.empty()) {
            // Expert collapse is visible here: per-layer token fractions,
            // their normalised entropy (1 = balanced) and the smallest share.
            double ent = 0, fmin = 1;
            json fr = json::array();
            for (const auto& f : moe_frac) {
                ent += nn::routing_entropy(f) / moe_frac.size();
                json row = json::array();
                for (double v : f) {
                    row.push_back(std::round(v * 1e4) / 1e4);
                    fmin = std::min(fmin, v);
                }
                fr.push_back(row);
            }
            e["aux_lb"] = moe_lb;
            e["aux_z"] = moe_z;
            e["moe_entropy"] = ent;
            e["moe_min_frac"] = fmin;
            e["moe_frac"] = fr;
        }
        if (!mor_frac.empty()) {
            // Depth fractions (depth 1 .. repeats), the mean depth in recursion
            // steps, their normalised entropy, and the router's two losses.
            json fr = json::array();
            double mean = 0;
            for (size_t i = 0; i < mor_frac.size(); ++i) {
                fr.push_back(std::round(mor_frac[i] * 1e6) / 1e6);
                mean += static_cast<double>(i + 1) * mor_frac[i];
            }
            e["mor_frac"] = fr;
            e["mor_depth"] = mean;
            e["mor_entropy"] = nn::routing_entropy(mor_frac);
            e["mor_balance"] = mor_lb;
            e["mor_z"] = mor_z;
        }
        if (!gm.is_null()) e["grads"] = gm;
        for (auto it = gmet.begin(); it != gmet.end(); ++it) e[it.key()] = it.value();
        ev.emit(e);

        if (grpo && step % s.eval_every == 0 && !gd.eval_prompts.empty()) {
            // Held-out monitor with fixed sampling seeds, so successive
            // evaluations differ only by the policy.
            NoGrad ng;
            model_ref.eval();
            double tr = 0, rw = 0, hk = 0, rh = 0;
            size_t n = 0;
            for (size_t ei = 0; ei < gd.eval_prompts.size(); ++ei)
                for (int j = 0; j < s.grpo_eval_samples; ++j) {
                    std::seed_seq ss{999u, static_cast<unsigned>(ei), static_cast<unsigned>(j)};
                    std::mt19937 gen(ss);
                    const Sampled sm = sample_pure([&](const std::vector<int>& c) { return fwd(c, 0); }, gd.eval_ids[ei],
                                                   s.T, tokens.size(), s.grpo_temp, s.grpo_max_new, eos_tok, gen);
                    const std::string text = ids_to_text(sm.ids, tokens);
                    const float ex = gd.exact->verify(gd.eval_prompts[ei], text).reward;
                    const float fm = gd.format->verify(gd.eval_prompts[ei], text).reward;
                    const auto rr = gd.reward->verify(gd.eval_prompts[ei], text);
                    tr += ex > 0.5f;
                    rw += rr.abstained() ? 0.0f : rr.reward;
                    hk += fm > 0.5f && ex < 0.5f;
                    rh += !rr.abstained() && rr.reward > 0.5f && ex < 0.5f;
                    ++n;
                }
            model_ref.train();
            ev.emit({{"event", "grpo_eval"}, {"step", step}, {"truth_acc", tr / n}, {"reward_mean", rw / n},
                     {"hack_frac", hk / n}, {"reward_hack_frac", rh / n}, {"samples", n}});
        } else if (po && step % s.eval_every == 0) {
            // Held-out pairs, unweighted: loss and preference accuracy.
            NoGrad ng;
            model_ref.eval();
            double loss = 0.0, acc = 0.0;
            const size_t nv = pd.w.size() - pd.val_start;
            for (size_t i = pd.val_start; i < pd.w.size(); ++i) {
                const PrefBatch b = pref_rows(pd, {i}, eos_tok >= 0 ? eos_tok : 0);
                const double m = pair_margins(fwd(b.x, b.L)->data, b, 1)[0] - po_ref[i];
                const double x = -s.po_beta * m;
                loss += x > 0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
                acc += m > 0 ? 1.0 : 0.0;
            }
            model_ref.train();
            ev.emit({{"event", "po_eval"}, {"step", step}, {"val_loss", loss / nv}, {"pref_acc", acc / nv},
                     {"pairs", nv}});
        } else if (!grpo && !po && step % s.eval_every == 0) {
            NoGrad ng;
            model_ref.eval();
            double vl = 0;
            // schedule "mor": the evaluation windows' depth counts.
            std::vector<double> ev_depth;
            auto note_depth = [&]() {
                if (!routed(s) || !llama) return;
                if (ev_depth.empty()) ev_depth.assign(s.rec_repeats, 0.0);
                for (int d : llama->last_depth) ev_depth[static_cast<size_t>(d - 1)] += 1.0;
            };
            bool dev_eval = false;
            const auto t_eval = std::chrono::steady_clock::now();
            if (sft) {
                // Every held-out example, the masked mean over all their
                // completion targets (each example weighted by its targets).
                // On the device (25 Oct 2026): when the run's own steps defer
                // (CUDA, device op set, step residency, deferred downloads),
                // the evaluation forwards run inside one step window too, so
                // the parameters upload once per evaluation and activations,
                // logits and the softmax stay on the device; only each loss
                // scalar comes home. Outside a window every GEMM uploaded its
                // weight and transposed it on the host, once per example: 10-20
                // minutes per eval_every on an L4. The same kernels compute the
                // loss either way (fp32 tolerance, test_cuda_ops leg 9's bound);
                // MICROTORCH_DEVICE_EVAL=0 keeps the windowless path. Nothing
                // changes on a CPU build or with deferral off.
                dev_eval = sft_eval_on_device();
                if (dev_eval) device::step_begin();
                double sum = 0.0;
                size_t kept = 0;
                const size_t end = s.eval_examples ? std::min(sd.x.size(), sd.val_start + s.eval_examples) : sd.x.size();
                for (size_t e0 = sd.val_start; e0 < end; e0 += static_cast<size_t>(s.eval_batch)) {
                    const size_t e1 = std::min(end, e0 + static_cast<size_t>(s.eval_batch));
                    std::vector<int> x, y;
                    std::vector<uint8_t> m;
                    size_t k = 0;
                    for (size_t e = e0; e < e1; ++e) {
                        x.insert(x.end(), sd.x[e].begin(), sd.x[e].end());
                        y.insert(y.end(), sd.y[e].begin(), sd.y[e].end());
                        m.insert(m.end(), sd.m[e].begin(), sd.m[e].end());
                        for (uint8_t v : sd.m[e]) k += v;
                    }
                    if (k == 0) continue;
                    // One example: the original call. Several: stacked as in training.
                    Var lg = e1 - e0 == 1 ? fwd(x) : fwd(x, s.T);
                    note_depth();
                    sum += static_cast<double>(ops::cross_entropy_masked(lg, y, m)->data(0, 0)) * k;
                    kept += k;
                }
                if (dev_eval) device::step_end();
                vl = sum / static_cast<double>(kept);
            } else {
                const int NV = s.eval_windows;
                std::mt19937 vrng(999);
                for (int k = 0; k < NV; ++k) {
                    const size_t va = val_start + vrng() % (ids.size() - val_start - s.T - 1);
                    std::vector<int> vx(ids.begin() + va, ids.begin() + va + s.T);
                    std::vector<int> vy(ids.begin() + va + 1, ids.begin() + va + s.T + 1);
                    if (s.state_k > 0) {
                        std::vector<uint8_t> vm;
                        state_loss_mask(vx, vy, vm, tokens.size());
                        vl += ops::cross_entropy_masked(fwd(vx), vy, vm)->data(0, 0);
                    } else {
                        vl += ops::cross_entropy(fwd(vx), vy)->data(0, 0);
                    }
                    note_depth();
                }
                vl /= NV;
            }
            model_ref.train();
            json eve = {{"event", "eval"}, {"step", step}, {"val_loss", vl}};
            if (sft && device::get() == device::Device::CUDA) {
                // Device runs only (CPU events are unchanged): which path ran
                // and its wall time, for the device-evaluation receipts.
                eve["eval_device"] = dev_eval;
                eve["eval_s"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_eval).count();
            }
            if (!ev_depth.empty()) {
                double tot = 0, mean = 0;
                for (double c : ev_depth) tot += c;
                json fr = json::array();
                std::vector<double> f(ev_depth.size());
                for (size_t i = 0; i < ev_depth.size(); ++i) {
                    f[i] = ev_depth[i] / tot;
                    fr.push_back(std::round(f[i] * 1e6) / 1e6);
                    mean += static_cast<double>(i + 1) * f[i];
                }
                eve["mor_frac"] = fr;
                eve["mor_depth"] = mean;
                eve["mor_entropy"] = nn::routing_entropy(f);
            }
            ev.emit(eve);
            if (s.es_patience > 0) {
                if (vl < best_val - s.es_min_delta) {
                    best_val = static_cast<float>(vl);
                    evals_flat = 0;
                } else if (++evals_flat >= s.es_patience) {
                    ev.emit({{"event", "early_stop"}, {"step", step}, {"best_val", best_val}});
                    stopped_early = true;
                }
            } else if (vl < best_val)
                best_val = static_cast<float>(vl);
        }
        if (step % s.ckpt_every == 0 || stopped_early || step == s.steps) {
            save(step);
            if (stopped_early) break;
        }
    }

    // Export.
    if (s.exp_safetensors) {
        const std::string spath = s.out_dir + "/" + s.name + ".safetensors";
        save_safetensors(spath, model_ref.state_dict());
        ev.emit({{"event", "export"}, {"format", "safetensors"}, {"path", spath}});
    }
    if (s.exp_gguf || !s.exp_gguf_quants.empty())
        export_ggufs(s, model_ref, llama.get(), gpt.get(), flex.get(), attnres.get(), mamba.get(), tokens, vocab, rope_heads, s.out_dir,
                     [&](const json& j) { ev.emit(j); });
    const double wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_train0).count();
    ev.emit({{"event", "done"},
             {"best_val", best_val},
             {"early_stopped", stopped_early},
             {"final_step", last_step},
             {"wall_seconds", wall_s}});
    // Atlas stage-0 result row: one durable JSON per run, joining the
    // structural echo with the outcome. atlas_extract.py enriches it with
    // behavioural features computed from events.jsonl.
    {
        json result = {{"name", s.name},
                       {"family", s.family},
                       {"attention", s.attention},
                       {"d", s.d},
                       {"layers", s.layers},
                       {"heads", s.heads},
                       {"T", s.T},
                       {"batch", s.batch},
                       {"accum", s.accum},
                       {"lr", s.lr},
                       {"seed", s.seed},
                       {"checkpoint_activations", s.ckpt_act},
                       {"params", n_params},
                       {"steps_requested", s.steps},
                       {"final_step", last_step},
                       {"best_val", best_val},
                       {"early_stopped", stopped_early},
                       {"wall_seconds", wall_s},
                       {"tokens_per_second", wall_s > 0 ? (last_step - start_step) *
                                                              static_cast<double>(s.batch) *
                                                              s.accum * s.T / wall_s
                                                        : 0.0}};
        if (s.precision != "fp32") result["precision"] = s.precision;
        if (s.attention_kernel != "tiled") result["attention_kernel"] = s.attention_kernel;
        if (s.fused_head) result["fused_head"] = true;
        std::ofstream rf(s.out_dir + "/result.json");
        rf << result.dump(2) << "\n";
    }
    if (!s.evaluation_probes.is_null()) {
        model_ref.eval();
        if (evaluation_probes_stage(
                s, [&](const std::vector<int>& ids) { return fwd(ids); }, tokens, vocab,
                s.out_dir + "/evaluation_probes.json", model_ref, s.vocab_gguf, last_step))
            return 1;
    }
    // After the run's record: a table that fails leaves the weights, the GGUFs and result.json in place
    // (`mtstudio export --from` writes it again), and the run still exits non-zero. Reported on stdout and in
    // the table's receipt (<table>.json), not in events.jsonl, whose last event stays "done".
    if (s.draft_table)
        write_draft_table(s, s.out_dir, [](const json& j) { std::printf("%s\n", j.dump().c_str()); });

    if (s.serve) {
        if (llama && s.exp_gguf) {
            std::printf(
                "serve: tinyllama %s/%s.gguf %s/%s.gguf 4 prompt "
                "\"once upon a time\" --max-tokens 40 -ngl 0 "
                "--top-k 1 --raw-prompt\n",
                s.out_dir.c_str(), s.name.c_str(), s.out_dir.c_str(), s.name.c_str());
        } else {
            std::printf(
                "serve: exported to %s/%s.safetensors (gguf serving "
                "needs family=llama + gguf export)\n",
                s.out_dir.c_str(), s.name.c_str());
        }
    }
    return 0;
}

}  // namespace

// ---- GGUF vocab + tokenizer (shared logic with srd_parity) ----
namespace {
template <typename T>
T rd(const std::vector<uint8_t>& b, size_t& p) {
    T v;
    std::memcpy(&v, b.data() + p, sizeof(T));
    p += sizeof(T);
    return v;
}
std::string rd_str(const std::vector<uint8_t>& b, size_t& p) {
    const uint64_t n = rd<uint64_t>(b, p);
    std::string s(reinterpret_cast<const char*>(b.data() + p), n);
    p += n;
    return s;
}
// A GGUF's declared context, "<general.architecture>.context_length" (-1 when absent), read from the
// metadata alone (streamed: the tensors are never read).
long gguf_context_length(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    auto rd_u32 = [&]() { uint32_t v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v; };
    auto rd_u64 = [&]() { uint64_t v = 0; f.read(reinterpret_cast<char*>(&v), 8); return v; };
    auto rd_s = [&]() { std::string s(rd_u64(), '\0'); f.read(s.data(), static_cast<std::streamsize>(s.size())); return s; };
    static const size_t kScalar[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (rd_u32() != 0x46554747u) throw std::runtime_error(path + ": not GGUF");
    rd_u32();
    rd_u64();
    const uint64_t n_meta = rd_u64();
    std::string arch;
    std::map<std::string, long> ctx;
    for (uint64_t i = 0; i < n_meta && f; ++i) {
        const std::string key = rd_s();
        const uint32_t t = rd_u32();
        if (t == 8) {
            const std::string v = rd_s();
            if (key == "general.architecture") arch = v;
        } else if (t == 9) {
            const uint32_t et = rd_u32();
            const uint64_t n = rd_u64();
            for (uint64_t k = 0; k < n && f; ++k) {
                if (et == 8) rd_s();
                else if (et <= 12 && kScalar[et]) f.seekg(static_cast<std::streamoff>(kScalar[et]), std::ios::cur);
                else throw std::runtime_error(path + ": bad GGUF array type");
            }
        } else if (t <= 12 && kScalar[t]) {
            uint64_t v = 0;
            f.read(reinterpret_cast<char*>(&v), static_cast<std::streamsize>(kScalar[t]));
            const std::string suf = ".context_length";
            if (key.size() > suf.size() && key.compare(key.size() - suf.size(), suf.size(), suf) == 0)
                ctx[key.substr(0, key.size() - suf.size())] = static_cast<long>(v);
        } else {
            throw std::runtime_error(path + ": bad GGUF metadata type");
        }
    }
    const auto it = ctx.find(arch);
    return it == ctx.end() ? -1 : it->second;
}
std::vector<std::string> read_gguf_vocab(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> b(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(b.size()));
    size_t p = 0;
    if (rd<uint32_t>(b, p) != 0x46554747u) throw std::runtime_error("not GGUF");
    rd<uint32_t>(b, p);
    rd<uint64_t>(b, p);
    const uint64_t n_meta = rd<uint64_t>(b, p);
    std::vector<std::string> tokens;
    std::string model;
    // Byte sizes of the GGUF scalar types (0-7, 10-12; 8 string, 9 array): every
    // one is skipped, so a llama.cpp-form vocabulary's token_type (int32 array)
    // and add_bos_token (bool) read like any other key.
    static const size_t kScalar[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    auto skip_scalar = [&](uint32_t t) {
        if (t > 12 || t == 8 || t == 9) throw std::runtime_error("bad meta");
        if (p + kScalar[t] > b.size()) throw std::runtime_error("truncated GGUF");
        p += kScalar[t];
    };
    for (uint64_t i = 0; i < n_meta; ++i) {
        const std::string key = rd_str(b, p);
        const uint32_t vt = rd<uint32_t>(b, p);
        if (vt == 8) {
            std::string v = rd_str(b, p);
            if (key == "tokenizer.ggml.model") model = std::move(v);
        } else if (vt == 9) {
            const uint32_t et = rd<uint32_t>(b, p);
            const uint64_t n = rd<uint64_t>(b, p);
            for (uint64_t k = 0; k < n; ++k) {
                if (et == 8) {
                    std::string t = rd_str(b, p);
                    if (key == "tokenizer.ggml.tokens") tokens.push_back(std::move(t));
                } else {
                    skip_scalar(et);
                }
            }
        } else {
            skip_scalar(vt);
        }
    }
    g_bpe.reset();
    g_tokenizer_model = model;
    const bpe::GgufTokenizer g =
        model == "bpe" || model == "gpt2" ? bpe::read_gguf_tokenizer(path) : bpe::GgufTokenizer{};
    if (model == "bpe") {
        auto t = std::make_shared<bpe::Tokenizer>(bpe::load_gguf(path));
        g_bpe = std::make_shared<TextBpe>(TextBpe{g.model, g.pre, g.token_type,
                                                  [t](const std::string& s) { return t->encode(s); },
                                                  [t](const std::vector<int>& v) { return t->decode(v); }});
    } else if (model == "gpt2" && g.pre == "llama-bpe") {
        auto t = std::make_shared<llama3::Tokenizer>(llama3::load_gguf(path));
        g_bpe = std::make_shared<TextBpe>(TextBpe{g.model, g.pre, g.token_type,
                                                  [t](const std::string& s) { return t->encode(s); },
                                                  [t](const std::vector<int>& v) { return t->decode(v); }});
    }
    return tokens;
}
// The tokenizer itself lives in microtorch/word_tokenizer.hpp (shared
// with its tests); this keeps the srd_parity-era call shape.
std::vector<int> tokenize(const std::string& text, const std::map<std::string, int>& vocab,
                          size_t max_tokens) {
    if (g_bpe) {
        std::vector<int> ids = g_bpe->encode(text);
        if (ids.size() > max_tokens) ids.resize(max_tokens);
        return ids;
    }
    return wordtok::tokenize(text, vocab, max_tokens);
}
// A training corpus: raw text, tokenised here, or a pre-tokenised ".ids"
// file (tools/corpus_ids.py), read as is so a resume does not re-tokenise
// gigabytes. .ids layout, little-endian: "PKIDS001", uint32 id width (2 or
// 4 bytes), uint32 vocabulary size, uint64 count, then the ids. The
// vocabulary size must equal the model's. max_tokens 0 = all.
std::vector<int> read_corpus(const std::string& path, const std::map<std::string, int>& vocab, size_t n_vocab,
                             size_t max_tokens) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open corpus " + path);
    if (path.size() < 4 || path.compare(path.size() - 4, 4, ".ids") != 0) {
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        return tokenize(text, vocab, max_tokens ? max_tokens : text.size() + 1);
    }
    char magic[8];
    uint32_t width = 0, nv = 0;
    uint64_t count = 0;
    f.read(magic, 8);
    f.read(reinterpret_cast<char*>(&width), 4);
    f.read(reinterpret_cast<char*>(&nv), 4);
    f.read(reinterpret_cast<char*>(&count), 8);
    if (!f || std::string(magic, 8) != "PKIDS001" || (width != 2 && width != 4))
        throw std::runtime_error(path + ": not a PKIDS001 id file");
    if (nv != n_vocab)
        throw std::runtime_error(path + ": ids are for a " + std::to_string(nv) + "-token vocabulary, the model has " +
                                 std::to_string(n_vocab));
    const uint64_t n = max_tokens ? std::min<uint64_t>(count, max_tokens) : count;
    std::vector<int> ids(n);
    if (width == 4) {
        f.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(n * 4));
    } else {
        std::vector<uint16_t> v(n);
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 2));
        std::copy(v.begin(), v.end(), ids.begin());
    }
    if (!f) throw std::runtime_error(path + ": truncated id file");
    for (int id : ids)
        if (id < 0 || static_cast<size_t>(id) >= n_vocab) throw std::runtime_error(path + ": id out of range");
    return ids;
}
// Ids back to text: BPE concatenates bytes; word-level joins with spaces
// (the format every word-level receipt was written in).
std::string ids_to_text(const std::vector<int>& ids, const std::vector<std::string>& tokens) {
    if (g_bpe) return g_bpe->decode(ids);
    std::string text;
    for (int id : ids) text += (text.empty() ? "" : " ") + tokens[id];
    return text;
}
}  // namespace

// ---- M2 live mode + chat: HTTP over cpp-httplib (POSIX and Windows).
// serve:
//   GET /              -> the studio UI (index.html)
//   GET /events.jsonl  -> the run dir's current event stream
//   The UI polls /events.jsonl every 2s when served over http, turning
//   the dashboard into a live training monitor.
// chat: a trained model behind a chat page (chat_cmd below).
#ifndef _WIN32
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

// Generated by CMake from studio/chat.html (kChatHtml, kChatHtmlSize):
// the chat page travels inside the binary, so `mtstudio chat` works from
// any directory, the build dir included.
#include "mtstudio_chat_html.hpp"

namespace {
std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// A spec's model with its exported safetensors loaded, ready to generate.
// One construction switch for `sample` and `chat` (mirrors run(); the
// spec is the single source of architecture truth for every path).
struct LoadedLM {
    Spec s;
    std::vector<std::string> tokens;
    std::string vocab_path;
    std::map<std::string, int> vocab;
    int eos = -1;  // end-of-text id, -1 when the vocabulary has none
    std::shared_ptr<parity::ParityLM> gpt;
    std::shared_ptr<nn::Llama> llama;
    std::shared_ptr<parity::AttnResLM> attnres;
    std::shared_ptr<parity::FlexLM> flex;
    std::shared_ptr<parity::MambaLM> mamba;
    std::string rope_heads;  // llama family: what the run was trained with

    explicit LoadedLM(const Spec& spec) : s(spec) {
        // A run directory moved away from its vocab file still carries the
        // vocabulary inside its exported GGUF (the same capped list).
        std::string vpath = s.vocab_gguf;
        const std::string exported = s.out_dir + "/" + s.name + ".gguf";
        if (!std::ifstream(vpath).good() && std::ifstream(exported).good()) vpath = exported;
        vocab_path = vpath;
        tokens = read_gguf_vocab(vpath);
        if (s.vocab_cap > 0 && s.vocab_cap < tokens.size()) tokens.resize(s.vocab_cap);
        for (size_t i = 0; i < tokens.size(); ++i) vocab.emplace(tokens[i], static_cast<int>(i));
        eos = wordtok::eos_id(vocab);

        if (s.family == "mamba") {
            mamba = std::make_shared<parity::MambaLM>(tokens.size(), s.d, s.layers, s.d_state, s.seed, s.mamba1());
        } else if (s.family == "flex") {
            const parity::FlexConfig fc = flex_config(s, tokens.size());
            flex = std::make_shared<parity::FlexLM>(fc, s.seed);
        } else if (s.attention == "attnres") {
            attnres = std::make_shared<parity::AttnResLM>(tokens.size(), s.d, s.heads, s.T, s.seed,
                                                          s.layers);
        } else if (s.family == "llama") {
            nn::LlamaConfig lc;
            lc.vocab = tokens.size();
            lc.d = s.d;
            lc.n_layers = s.layers;
            lc.n_heads = s.heads;
            lc.d_ff = s.d_ff ? s.d_ff : 3 * s.d;
            lc.n_ctx = s.T;
            // Bare weights (no events.jsonl, so no run record) with no
            // arch.rope_heads in the spec: the coverage is unknown, and the
            // legacy default "first" silently served an all-heads model with
            // the wrong RoPE (20 Oct 2026). chat_eval.py refuses the same case.
            if (s.rope_heads.empty() && !std::filesystem::exists(s.out_dir + "/events.jsonl"))
                throw std::runtime_error(s.out_dir + " has bare weights and no run record: set arch.rope_heads "
                                         "(\"all\" for models trained since 18 Oct 2026, \"first\" for older ones)");
            rope_heads = resolve_rope_heads(s, /*existing=*/true);
            lc.rope_all_heads = rope_heads == "all";
            apply_moe(lc, s);
            apply_attn_res(lc, s);
            apply_recursion(lc, s);
            apply_state(lc, s);
            apply_thirdparty(lc, s);
            llama = std::make_shared<nn::Llama>(lc, s.seed);
        } else {
            gpt = make_parity(s, tokens.size());
        }
        const std::string ckpt = s.out_dir + "/" + s.name + ".safetensors";
        model().load_state_dict(load_safetensors(ckpt), /*strict=*/true);
        model().eval();
        if (s.state_k > 0) st = state_template(s, vocab);
    }
    // State tokens (arch.state_tokens): the template's ids; st.k == 0 otherwise.
    nn::StateTemplate st;
    nn::Module& model() {
        if (mamba) return *mamba;
        return flex ? static_cast<nn::Module&>(*flex)
               : attnres
                   ? static_cast<nn::Module&>(*attnres)
                   : (llama ? static_cast<nn::Module&>(*llama) : static_cast<nn::Module&>(*gpt));
    }
    Var forward(const std::vector<int>& ids, size_t seq_len = 0) {
        if (mamba) return mamba->forward(ids, seq_len);
        if (flex) return flex->forward(ids, seq_len);
        if (attnres) return attnres->forward(ids, seq_len);
        return llama ? llama->forward(ids, seq_len) : gpt->forward(ids, seq_len);
    }
    // One sampled next token over the last T ids: temperature, top-k, then
    // nucleus (top_p < 1 keeps the smallest head of the top-k holding that
    // share of its mass). top_p >= 1 is the original quick-look sampler,
    // draw for draw. Call under NoGrad.
    // Repetition penalty (LOCAL_LLM_STACK.md section 3 point 4): rep_p > 1
    // and rep_n > 0 scale every distinct id among the last rep_n of `ids`,
    // except `no_penalty` (the stop set), as l > 0 ? l / p : l * p, before
    // the special-token ban and the temperature. Defaults leave it off.
    // KV cache (llama only; chat --kv-cache): off by default, because the
    // greedy references other engines are gated against come from the
    // full recompute, and cached attention sums in another order (equal to
    // rounding, tests/test_kv_cache.cpp). The cache holds kv_ids; a call
    // reuses its longest common prefix with the new context.
    bool kv_on = false;
    nn::LlamaKV kv;
    std::vector<int> kv_ids;
    std::vector<float> last_logits(const std::vector<int>& ctx) {
        Var logits;
        if (kv_on && llama) {
            size_t p = 0;
            while (p < kv_ids.size() && p < ctx.size() && kv_ids[p] == ctx[p]) ++p;
            if (p == ctx.size()) --p;  // the last token's logits are not cached
            // Less than p after a rewind into a turn whose rows were evicted
            // (nn::LlamaKV::truncate): re-prefill from there.
            p = kv.truncate(p);
            logits = llama->forward_cached(std::vector<int>(ctx.begin() + static_cast<long>(p), ctx.end()), kv);
            kv_ids = ctx;
        } else {
            logits = forward(ctx);
        }
        const size_t last = logits->data.rows() - 1, V = logits->data.cols();
        std::vector<float> l(V);
        for (size_t j = 0; j < V; ++j) l[j] = logits->data(last, j);
        return l;
    }
    int next(const std::vector<int>& ids, float temp, int topk, float top_p, std::mt19937& gen,
             float rep_p = 1.0f, size_t rep_n = 0, const std::vector<int>& no_penalty = {}) {
        std::vector<int> ctx = ids;
        if (ctx.size() > s.T) ctx.assign(ids.end() - s.T, ids.end());
        std::vector<float> l = last_logits(ctx);
        const size_t V = l.size();
        if (rep_p != 1.0f && rep_n > 0) {
            std::vector<char> seen(V, 0);
            for (int x : no_penalty)
                if (x >= 0 && static_cast<size_t>(x) < V) seen[x] = 1;
            const size_t from = ids.size() > rep_n ? ids.size() - rep_n : 0;
            for (size_t i = from; i < ids.size(); ++i) {
                const int t = ids[i];
                if (t < 0 || static_cast<size_t>(t) >= V || seen[t]) continue;
                seen[t] = 1;
                l[t] = l[t] > 0.0f ? l[t] / rep_p : l[t] * rep_p;
            }
        }
        std::vector<std::pair<float, int>> scored(V);
        for (size_t j = 0; j < V; ++j) scored[j] = {l[j] / std::max(temp, 1e-4f), static_cast<int>(j)};
        // Ban special tokens from generation (standard sampler hygiene;
        // a small-vocab model otherwise floods the output with <unk>).
        for (const char* sp : {"<unk>", "<s>", "</s>", "<pad>"}) {
            auto it = vocab.find(sp);
            if (it != vocab.end()) scored[it->second].first = -1e30f;
        }
        size_t k = std::min<size_t>(std::max(topk, 1), V);
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        double mx = scored[0].first, z = 0;
        std::vector<double> p(k);
        for (size_t j = 0; j < k; ++j) z += (p[j] = std::exp(scored[j].first - mx));
        if (top_p > 0.0f && top_p < 1.0f) {
            double acc = 0;
            for (size_t j = 0; j < k; ++j)
                if ((acc += p[j]) >= top_p * z) {
                    k = j + 1;
                    break;
                }
            z = acc;
        }
        std::uniform_real_distribution<double> u(0.0, z);
        double r = u(gen);
        int pick = scored[k - 1].second;
        for (size_t j = 0; j < k; ++j)
            if ((r -= p[j]) <= 0) {
                pick = scored[j].second;
                break;
            }
        return pick;
    }
};

// The spec re-pointed at a .safetensors file (out_dir = its directory, name =
// its stem), which is where LoadedLM looks for weights.
Spec at_safetensors(const Spec& s, const std::string& path, const std::string& what) {
    const std::string suffix = ".safetensors";
    if (path.size() <= suffix.size() || path.compare(path.size() - suffix.size(), suffix.size(), suffix) != 0)
        throw std::runtime_error(what + " must name a .safetensors file");
    Spec r = s;
    const auto slash = path.find_last_of('/');
    r.out_dir = slash == std::string::npos ? "." : path.substr(0, slash);
    r.name = path.substr(slash == std::string::npos ? 0 : slash + 1);
    r.name.resize(r.name.size() - suffix.size());
    return r;
}

std::function<Var(const std::vector<int>&, size_t)> make_reference(const Spec& s) {
    auto lm = std::make_shared<LoadedLM>(at_safetensors(s, s.init_from, "train.init_from"));
    return [lm](const std::vector<int>& ids, size_t seq_len) { return lm->forward(ids, seq_len); };
}

// Re-export a trained model without training: rebuild the spec's model, load
// `from` (default: the run's <out_dir>/<name>.safetensors) and write the
// comma-separated GGUF `formats` (default: the spec's export.formats) into
// `out_dir` (default: the spec's), named after the spec.
int export_cmd(Spec s, const std::string& from, const std::string& formats, const std::string& out_dir) {
    if (!formats.empty()) {
        s.exp_gguf = false;
        s.exp_gguf_quants.clear();
        std::stringstream ss(formats);
        for (std::string fmt; std::getline(ss, fmt, ',');) {
            if (fmt == "gguf")
                s.exp_gguf = true;
            else if (fmt.rfind("gguf_", 0) == 0)
                s.exp_gguf_quants.push_back(gguf_quant_of(fmt));
            else
                throw std::runtime_error("export: unknown format " + fmt +
                                         " (gguf, gguf_q8_0, gguf_q8_0_out, gguf_q8_0_all, gguf_q4_0, gguf_q4_k_m, gguf_q4_k_s)");
        }
    }
    if (!s.exp_gguf && s.exp_gguf_quants.empty())
        throw std::runtime_error("export: no GGUF format requested (--formats or export.formats)");
    if (s.draft_table && !std::filesystem::exists(draft_table_tool()))
        throw std::runtime_error("export.draft_table: " + draft_table_tool() +
                                 " not found (run from the repository root, or set MTSTUDIO_DRAFT_TABLE)");
    LoadedLM lm(from.empty() ? s : at_safetensors(s, from, "--from"));
    const std::string dest = out_dir.empty() ? s.out_dir : out_dir;
    std::filesystem::create_directories(dest);
    export_ggufs(s, lm.model(), lm.llama.get(), lm.gpt.get(), lm.flex.get(), lm.attnres.get(), lm.mamba.get(),
                 lm.tokens, lm.vocab, lm.rope_heads, dest,
                 [](const json& j) { std::printf("%s\n", j.dump().c_str()); });
    if (s.draft_table)
        write_draft_table(s, dest, [](const json& j) { std::printf("%s\n", j.dump().c_str()); });
    return 0;
}

// The quick-look sampler (ECOSYSTEM.md feature 2): rebuild the spec's
// model, load its exported safetensors, and generate word-level text
// with temperature + top-k, stopping early at end-of-text. ember.cpp
// remains the real server; this closes the train→poke loop without
// leaving the studio.
void preflight_probe_inputs(const Spec& s, const std::vector<std::string>& tokens,
                            const std::map<std::string, int>& vocab) {
    if (!g_bpe && g_tokenizer_model != "word" && g_tokenizer_model != "wordlevel")
        throw std::runtime_error("unsupported probe tokenizer metadata");
    for (const auto& item : s.evaluation_probes["cases"]) {
        auto ids = tokenize(item["prompt"].get<std::string>(), vocab, 8193);
        if (ids.empty() || ids.size() > s.T)
            throw std::runtime_error("probe prompt empty/oversized; no training started");
        for (int id : ids)
            if (id < 0 || static_cast<size_t>(id) >= tokens.size() ||
                (id == 0 && tokens[0] == "<unk>"))
                throw std::runtime_error("probe prompt OOV/unencodable; no training started");
    }
}
void verify_probe_checkpoint(const std::string& path, nn::Module& model) {
    std::ifstream input(path, std::ios::binary);
    uint64_t length = 0;
    input.read(reinterpret_cast<char*>(&length), 8);
    if (!input || length > 16 * 1024 * 1024)
        throw std::runtime_error("invalid checkpoint header for probe identity");
    std::string header(length, '\0');
    input.read(header.data(), length);
    if (!input) throw std::runtime_error("truncated checkpoint header");
    json h = json::parse(header);
    auto params = model.named_parameters();
    size_t count = 0;
    for (auto it = h.begin(); it != h.end(); ++it)
        if (it.key() != "__metadata__") ++count;
    if (count != params.size())
        throw std::runtime_error("checkpoint parameter inventory differs from evaluated model");
    std::array<char, 65536> buffer{};
    for (const auto& entry : params) {
        const auto& name = entry.first;
        const auto& value = entry.second;
        device::materialize(value->data);
        const auto& meta = h.at(name);
        if (meta.at("dtype") != "F32" ||
            meta.at("shape") != json::array({value->data.rows(), value->data.cols()}))
            throw std::runtime_error("checkpoint parameter geometry differs");
        size_t bytes = value->data.rows() * value->data.cols() * sizeof(float);
        uint64_t start = meta.at("data_offsets").at(0), end = meta.at("data_offsets").at(1);
        if (end < start || end - start != bytes)
            throw std::runtime_error("checkpoint parameter bytes differ");
        input.seekg(8 + length + start);
        const char* memory = reinterpret_cast<const char*>(value->data.data());
        for (size_t offset = 0; offset < bytes; offset += buffer.size()) {
            size_t n = std::min(buffer.size(), bytes - offset);
            input.read(buffer.data(), n);
            if (!input || std::memcmp(buffer.data(), memory + offset, n))
                throw std::runtime_error(
                    "checkpoint bytes do not equal actual in-memory evaluated weights");
        }
    }
}

int evaluation_probes_stage(const Spec& s,
                            const std::function<Var(const std::vector<int>&)>& forward,
                            const std::vector<std::string>& tokens,
                            const std::map<std::string, int>& vocab, const std::string& output,
                            nn::Module& model, const std::string& vocab_path, int final_step) {
    json report = {{"schema_version", 1},
                   {"status", "ERROR"},
                   {"configuration", s.evaluation_probes},
                   {"checkpoint", nullptr},
                   {"generation",
                    {{"mode", "greedy"},
                     {"tie_break", "lowest_id"},
                     {"special_token_bans", false},
                     {"context_policy", "reject_long_prompt_then_sliding_window"}}},
                   {"cases", json::array()},
                   {"summary", {{"total", 0}, {"passed", 0}, {"failed", 0}, {"errors", 0}}},
                   {"limitations",
                    {"Literal configured probes, not a benchmark or "
                     "semantic/instruction-compliance assessment.",
                     "not_contains on empty output asserts only a literal absence.",
                     "Greedy full-vocabulary logits, no chat template, sampling, penalties or "
                     "special-token bans."}}};
    struct RestorePrecision {
        device::Precision previous = device::matmul_precision();
        ops::AttnKernel previous_attention = ops::attention_kernel();
        ~RestorePrecision() {
            device::set_matmul_precision(previous);
            ops::set_attention_kernel(previous_attention);
        }
    } restore_precision;
    device::set_matmul_precision(s.precision == "bf16" ? device::Precision::BF16
                                                       : device::Precision::FP32);
    ops::set_attention_kernel(s.attention_kernel == "flash" ? ops::AttnKernel::Flash
                                                            : ops::AttnKernel::Tiled);
    report["generation"]["requested_precision"] = s.precision;
    report["generation"]["requested_attention_kernel"] = s.attention_kernel;
    report["generation"]["runtime_attention_kernel_setting"] =
        ops::attention_kernel() == ops::AttnKernel::Flash ? "flash" : "tiled";
    report["generation"]["execution_device"] =
        device::get() == device::Device::CUDA ? "cuda" : "cpu";
    report["generation"]["runtime_matmul_precision_setting"] =
        device::matmul_precision() == device::Precision::BF16 ? "bf16" : "fp32";
    report["generation"]["eos_policy"] =
        "stop only on vocabulary literal<|endoftext|>; absent means token cap";
    report["generation"]["eos_id"] = wordtok::eos_id(vocab);
    report["generation"]["token_count_policy"] =
        "generated_ids exclude stopping EOS; generation_steps counts every forward/selection";
    int errors = 0, global_errors = 0, passed = 0, failed = 0;
    try {
        const std::string weights = s.out_dir + "/" + s.name + ".safetensors";
        verify_probe_checkpoint(weights, model);
        report["specification"] = s.specification;
        report["specification_sha256"] = ir::sha256_hex(s.specification.dump());
        report["resolved_architecture"] = {
            {"family", s.family},      {"attention", s.attention}, {"d", s.d},
            {"layers", s.layers},      {"heads", s.heads},         {"T", s.T},
            {"precision", s.precision}};
        report["final_step"] = final_step < 0 ? json(nullptr) : json(final_step);
        report["checkpoint"] = {
            {"path", weights},
            {"bytes", std::filesystem::file_size(weights)},
            {"sha256", studio_probes::file_sha256(weights)},
            {"selection", final_step < 0 ? "explicit_loaded_weights_final_step_unknown"
                                         : "final_or_early_stop_current_weights_not_best"}};
        report["configuration_sha256"] = ir::sha256_hex(s.evaluation_probes.dump());
        if (!g_bpe && g_tokenizer_model != "word" && g_tokenizer_model != "wordlevel")
            throw std::runtime_error("unsupported tokenizer metadata for probes: " +
                                     g_tokenizer_model);
        report["tokenizer"] = {{"model", g_tokenizer_model},
                               {"pre", g_bpe ? g_bpe->pre : ""},
                               {"vocab_cap", tokens.size()},
                               {"source", vocab_path},
                               {"sha256", studio_probes::file_sha256(vocab_path)},
                               {"decode", g_bpe ? "native_bpe" : "word_detokenize"}};
        std::ofstream journal(output + ".cases.jsonl", std::ios::binary);
        if (!journal) throw std::runtime_error("cannot write probe journal");
        NoGrad no_grad;
        for (const auto& item : s.evaluation_probes["cases"]) {
            json row = item;
            if (item["kind"] == "needle")
                row["expected_first_byte_offset"] =
                    item["prompt"].get<std::string>().find(item["expected"].get<std::string>());
            row["status"] = "ERROR";
            row["generated_text"] = "";
            row["generated_ids"] = json::array();
            row["prompt_ids"] = json::array();
            row["prompt_tokens"] = 0;
            row["generated_tokens"] = 0;
            row["generation_steps"] = 0;
            row["context_evicted_tokens"] = 0;
            row["stop_reason"] = "error";
            try {
                auto ids = tokenize(item["prompt"].get<std::string>(), vocab, 8193);
                row["prompt_ids"] = ids;
                row["prompt_tokens"] = ids.size();
                if (ids.empty() || ids.size() > s.T)
                    throw std::runtime_error(
                        "empty or context-oversized prompt; no prompt truncation performed");
                for (int id : ids)
                    if (id < 0 || static_cast<size_t>(id) >= tokens.size() ||
                        (id == 0 && tokens[0] == "<unk>"))
                        throw std::runtime_error("unencodable/OOV or capped-vocabulary prompt");
                std::vector<int> generated;
                size_t decoded_upper_bound = 0;
                const int eos = wordtok::eos_id(vocab);
                int evicted = 0;
                row["stop_reason"] = "token_limit";
                for (int i = 0; i < s.evaluation_probes["max_new_tokens"].get<int>(); ++i) {
                    row["generation_steps"] = i + 1;
                    std::vector<int> ctx = ids;
                    if (ctx.size() > s.T) {
                        evicted = std::max(evicted, static_cast<int>(ctx.size() - s.T));
                        ctx.erase(ctx.begin(), ctx.end() - s.T);
                    }
                    row["context_evicted_tokens"] = evicted;
                    auto logits = forward(ctx);
                    device::materialize(logits->data);
                    if (logits->data.rows() != ctx.size() || logits->data.cols() != tokens.size())
                        throw std::runtime_error("probe logits shape mismatch");
                    int pick = -1;
                    float maximum = -std::numeric_limits<float>::infinity();
                    for (size_t j = 0; j < tokens.size(); ++j) {
                        float x = logits->data(ctx.size() - 1, j);
                        if (!std::isfinite(x)) throw std::runtime_error("nonfinite probe logits");
                        if (pick < 0 || x > maximum) {
                            maximum = x;
                            pick = static_cast<int>(j);
                        }
                    }
                    if (pick < 0) throw std::runtime_error("no selectable probe logits");
                    if (pick == eos) {
                        row["stop_reason"] = "eos";
                        break;
                    }
                    if (tokens[pick].size() + 1 > 65536 - decoded_upper_bound)
                        throw std::runtime_error(
                            "probe decoded output byte budget exceeded before decoding");
                    decoded_upper_bound += tokens[pick].size() + 1;
                    generated.push_back(pick);
                    ids.push_back(pick);
                    row["generated_ids"] = generated;
                    row["generated_tokens"] = generated.size();
                }
                const std::string text =
                    g_bpe ? g_bpe->decode(generated) : wordtok::detokenize(generated, tokens);
                if (text.size() > 65536)
                    throw std::runtime_error("probe decoded output exceeds65536byte limit");
                if (!studio_probes::valid_utf8(text)) {
                    row["generated_bytes_hex"] = studio_probes::hex_bytes(text);
                    throw std::runtime_error(
                        "invalid or incomplete UTF8 generated bytes; literal contract not scored");
                }
                row["generated_text"] = text;
                bool match = studio_probes::matches(row["generated_text"].get<std::string>(),
                                                    item["contract"], item["expected"]);
                row["status"] = match ? "PASS" : "FAIL";
                if (match)
                    ++passed;
                else
                    ++failed;
            } catch (const std::exception& e) {
                row["error"] = studio_probes::safe_error(e);
                row["stop_reason"] = "error";
                ++errors;
            }
            report["cases"].push_back(row);
            journal << row.dump() << "\n";
            journal.flush();
            if (!journal) throw std::runtime_error("probe case journal write failed");
        }
        verify_probe_checkpoint(weights, model);
        if (studio_probes::file_sha256(weights) != report["checkpoint"]["sha256"])
            throw std::runtime_error("checkpoint changed during probes");
        report["summary"] = {{"total", report["cases"].size()},
                             {"passed", passed},
                             {"failed", failed},
                             {"errors", errors}};
        report["status"] = errors ? "COMPLETE_WITH_ERRORS" : "COMPLETE";
    } catch (const std::exception& e) {
        report["error"] = studio_probes::safe_error(e);
        report["status"] = "ERROR";
        ++global_errors;
    }
    report["summary"] = {{"total", report["cases"].size()},
                         {"passed", passed},
                         {"failed", failed},
                         {"errors", errors}};
    report["global_errors"] = global_errors;
    studio_probes::publish(output, report);
    return errors || global_errors ? 1 : 0;
}

int sample_cmd(const Spec& s, const std::string& prompt, int n_new, float temp, int topk,
               unsigned sseed, const std::string& out_file) {
    LoadedLM lm(s);
    auto ids = tokenize(prompt, lm.vocab, 100000);
    if (ids.empty()) throw std::runtime_error("no prompt word is in the model's vocabulary");
    std::mt19937 gen(sseed);
    std::string text = prompt;
    NoGrad ng;
    for (int t = 0; t < n_new; ++t) {
        const int pick = lm.next(ids, temp, topk, 1.0f, gen);
        if (pick == lm.eos) break;
        ids.push_back(pick);
        text += g_bpe ? g_bpe->decode({pick}) : " " + lm.tokens[pick];
    }
    std::printf("%s\n", text.c_str());
    if (!out_file.empty()) {
        std::ofstream f(out_file, std::ios::trunc);
        f << text << "\n";
    }
    return 0;
}

// The post-training sampler (docs/decisions/0001-posttrain-charter.md §4,
// Phase 2). Unlike LoadedLM::next it is PURE temperature sampling over the
// whole vocabulary: no top-k, no nucleus, no banned tokens. Truncation or
// bans would make the sampled distribution differ from the policy, so the
// samples would be off-policy for GRPO without anything saying so. Every
// record carries the sampled ids and their log-probability under the policy
// at temperature 1 (logp) and at the sampling temperature (logp_sample).
// Each (prompt, sample) gets its own generator seeded by
// seed_seq{seed, prompt index, sample index}, so one record replays alone.
//
// Input: JSONL lines with "prompt" (and optionally "uri"). Output: one JSONL
// record per completion. tools/posttrain/rejection_sample.py scores and
// filters these records.
int sample_set_cmd(const Spec& s, const std::string& prompts_path, const std::string& out_path, int n,
                   float temp, unsigned seed, int max_new) {
    if (!(temp > 0.0f)) throw std::runtime_error("sample-set: --temp must be > 0 (pure sampling)");
    LoadedLM lm(s);
    std::ifstream in(prompts_path);
    if (!in) throw std::runtime_error("cannot open " + prompts_path);
    std::ofstream out(out_path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write " + out_path);
    const size_t V = lm.tokens.size();
    NoGrad ng;
    std::string line;
    size_t pi = 0, written = 0;
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        const json pr = json::parse(line);
        // {"prompt": text} or {"messages": [...]} (a prompt conversation, rendered by the template).
        const std::string prompt = pr.contains("messages")
                                       ? chat::render(chat::messages_from_json(pr["messages"]), chat_template_of(s), true)
                                       : pr.at("prompt").get<std::string>();
        std::vector<int> pids = tokenize(prompt, lm.vocab, 100000);
        if (lm.st.k > 0) pids = nn::insert_state_ids(pids, lm.st);
        if (pids.empty()) throw std::runtime_error("prompt " + std::to_string(pi) + " has no in-vocabulary token");
        size_t unk = 0;
        for (int id : pids) unk += id == 0;
        for (int k = 0; k < n; ++k) {
            std::seed_seq ss{seed, static_cast<unsigned>(pi), static_cast<unsigned>(k)};
            std::mt19937 gen(ss);
            const Sampled sm = sample_pure([&](const std::vector<int>& c) { return lm.forward(c); }, pids, s.T, V,
                                           temp, max_new, lm.eos, gen);
            const std::vector<int>& comp = sm.ids;
            const double logp = sm.logp, logp_s = sm.logp_sample;
            const std::string& stop = sm.stop;
            const std::string text = ids_to_text(comp, lm.tokens);
            json rec = {{"prompt", prompt},
                        {"completion", text},
                        {"completion_ids", comp},
                        {"prompt_ids", pids},
                        {"prompt_unk", unk},
                        {"sample_index", k},
                        {"prompt_index", pi},
                        {"seed", seed},
                        {"temperature", temp},
                        {"logp", logp},
                        {"logp_sample", logp_s},
                        {"stop", stop},
                        {"sampler", "pure-temperature-v1"},
                        {"model", s.name},
                        {"checkpoint", s.out_dir + "/" + s.name + ".safetensors"}};
            if (pr.contains("uri")) rec["prompt_uri"] = pr["uri"];
            out << rec.dump() << "\n";
            ++written;
        }
        ++pi;
    }
    std::fprintf(stderr, "sample-set: %zu prompts x %d = %zu completions -> %s\n", pi, n, written,
                 out_path.c_str());
    return 0;
}

// Accepts "1706.03762", "2302.13971v1", "cs/9901002", or any arxiv.org
// URL containing one of those (abs/, pdf/, e-print/). Returns the bare
// id, or "" if nothing that looks like an arXiv id is present — the
// gate before the id is ever placed on an exec argv.
std::string sanitize_arxiv(std::string s) {
    for (const char* p : {"abs/", "pdf/", "e-print/"}) {
        const auto k = s.rfind(p);
        if (k != std::string::npos) s = s.substr(k + std::strlen(p));
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(0, 1);
    if (s.size() > 4 && s.substr(s.size() - 4) == ".pdf") s.resize(s.size() - 4);
    static const std::regex id_re(
        R"(^(\d{4}\.\d{4,5}(v\d+)?|[a-z][a-z-]{1,12}(\.[A-Z]{2})?/\d{7}(v\d+)?)$)");
    return s.size() <= 32 && std::regex_match(s, id_re) ? s : "";
}

// ---- child processes (the in-page Train and fetch buttons) ----
// A started program with stdout+stderr appended to a log, reaped without
// blocking. fork/exec on POSIX, CreateProcess on Windows; the server is
// multi-threaded, so the POSIX child only calls async-signal-safe
// functions between fork and exec.
struct Child {
#ifdef _WIN32
    HANDLE h = nullptr;
    bool running() const { return h != nullptr; }
#else
    pid_t pid = -1;
    bool running() const { return pid > 0; }
#endif
};

#ifdef _WIN32
// One argument quoted for CommandLineToArgvW (backslashes before a quote
// doubled, the quote escaped).
std::string win_quote(const std::string& a) {
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
    std::string out = "\"";
    size_t bs = 0;
    for (char c : a) {
        if (c == '\\') {
            ++bs;
            continue;
        }
        if (c == '"') {
            out.append(bs * 2 + 1, '\\');
        } else {
            out.append(bs, '\\');
        }
        out += c;
        bs = 0;
    }
    out.append(bs * 2, '\\');
    return out + "\"";
}
#endif

bool spawn(Child& c, const std::vector<std::string>& argv, const std::string& log) {
#ifdef _WIN32
    std::string cmd;
    for (const auto& a : argv) cmd += (cmd.empty() ? "" : " ") + win_quote(a);
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE lf = CreateFileA(log.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                            OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (lf == INVALID_HANDLE_VALUE) return false;
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = lf;
    si.hStdError = lf;
    PROCESS_INFORMATION pi{};
    std::vector<char> line(cmd.begin(), cmd.end());
    line.push_back('\0');
    const BOOL ok = CreateProcessA(nullptr, line.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                                   nullptr, nullptr, &si, &pi);
    CloseHandle(lf);
    if (!ok) return false;
    CloseHandle(pi.hThread);
    c.h = pi.hProcess;
    return true;
#else
    std::vector<char*> av;
    for (const auto& a : argv) av.push_back(const_cast<char*>(a.c_str()));
    av.push_back(nullptr);
    const int lf = ::open(log.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (lf < 0) return false;
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::dup2(lf, 1);  // dup2 clears O_CLOEXEC on the copies
        ::dup2(lf, 2);
        ::execvp(av[0], av.data());
        _exit(127);
    }
    ::close(lf);
    if (pid < 0) return false;
    c.pid = pid;
    return true;
#endif
}

// True once a running child has exited (it is then cleared);
// *exited_ok says whether it exited with status 0.
bool reap(Child& c, bool* exited_ok) {
#ifdef _WIN32
    if (!c.h || WaitForSingleObject(c.h, 0) != WAIT_OBJECT_0) return false;
    DWORD code = 1;
    GetExitCodeProcess(c.h, &code);
    CloseHandle(c.h);
    c.h = nullptr;
    if (exited_ok) *exited_ok = code == 0;
#else
    int st = 0;
    if (c.pid <= 0 || ::waitpid(c.pid, &st, WNOHANG) != c.pid) return false;
    c.pid = -1;
    if (exited_ok) *exited_ok = WIFEXITED(st) && WEXITSTATUS(st) == 0;
#endif
    return true;
}

// Cancel only a process this server owns; no process-name matching.
bool cancel_child(Child& child) {
#ifdef _WIN32
    if (!child.h) return true;
    if (!TerminateProcess(child.h, 1) && WaitForSingleObject(child.h, 0) != WAIT_OBJECT_0)
        return false;
    if (WaitForSingleObject(child.h, 1000) != WAIT_OBJECT_0) return false;
    CloseHandle(child.h);
    child.h = nullptr;
#else
    if (child.pid <= 0) return true;
    if (::kill(child.pid, SIGKILL) != 0 && errno != ESRCH) return false;
    int status = 0;
    while (::waitpid(child.pid, &status, 0) < 0) {
        if (errno != EINTR && errno != ECHILD) return false;
        if (errno == ECHILD) break;
    }
    child.pid = -1;
#endif
    return true;
}

// ---- export.draft_table: ember's corpus draft table beside the GGUF (tools/draft_table.py) ----
std::string draft_table_tool() {
    const char* t = std::getenv("MTSTUDIO_DRAFT_TABLE");
    return t ? t : "tools/draft_table.py";
}

// <out_dir>/<name>.draft_table.bin and its receipt (.json), from the vocabulary of the GGUF just written (the
// F32 file, else the first quantised one) and the chat text the model was trained on; the tool's output goes
// to <name>.draft_table.log. Blocks until it is done; a failure throws (a run never ends without the table
// it was asked for and without saying why).
void write_draft_table(const Spec& s, const std::string& out_dir, const std::function<void(const json&)>& emit) {
    std::string gguf;
    if (s.exp_gguf) gguf = out_dir + "/" + s.name + ".gguf";
    for (size_t i = 0; gguf.empty() && i < s.exp_gguf_quants.size(); ++i)
        gguf = out_dir + "/" + s.name + "_" + s.exp_gguf_quants[i] + ".gguf";
    if (gguf.empty() || !std::filesystem::exists(gguf))
        throw std::runtime_error("export.draft_table: no GGUF was written to " + out_dir +
                                 " (the table is built from the model's vocabulary)");
    const char* py = std::getenv("MTSTUDIO_PYTHON");
#ifdef _WIN32
    const std::string python = py ? py : "python";
#else
    const std::string python = py ? py : "python3";
#endif
    const std::vector<std::string> sources = s.draft_sources.empty() ? std::vector<std::string>{s.sft_examples}
                                                                    : s.draft_sources;
    const std::string out = out_dir + "/" + s.name + ".draft_table.bin";
    const std::string log = out_dir + "/" + s.name + ".draft_table.log";
    std::vector<std::string> argv = {python, draft_table_tool(), "build", "--gguf", gguf, "--out", out};
    if (s.draft_max_bytes) {
        argv.push_back("--max-bytes");
        argv.push_back(std::to_string(s.draft_max_bytes));
    }
    argv.insert(argv.end(), sources.begin(), sources.end());
    std::remove(log.c_str());
    std::remove(out.c_str());
    std::printf("draft_table: building %s (log %s)\n", out.c_str(), log.c_str());
    std::fflush(stdout);
    Child c;
    if (!spawn(c, argv, log)) throw std::runtime_error("export.draft_table: could not start " + python);
    bool ok = false;
    while (!reap(c, &ok)) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (!ok || !std::filesystem::exists(out))
        throw std::runtime_error("export.draft_table: " + draft_table_tool() + " failed; see " + log);
    emit({{"event", "export"}, {"format", "draft_table"}, {"path", out}, {"gguf", gguf}, {"sources", sources},
          {"receipt", out + ".json"}});
}

// This binary's own path, for re-launching it as "mtstudio run <spec>".
std::string self_exe(const char* argv0) {
#ifdef _WIN32
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) return std::string(buf, n);
#else
    std::error_code ec;
    const auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) return p.string();
#endif
    return argv0;
}

void text_reply(httplib::Response& res, int status, const std::string& body,
                const char* ctype = "text/plain") {
    res.status = status;
    res.set_content(body, ctype);
}

// Bare status bodies ("404") for anything a handler left empty, as the
// hand-rolled server answered.
void plain_errors(httplib::Server& svr) {
    svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (!res.body.empty()) return httplib::Server::HandlerResponse::Unhandled;
        res.set_content(std::to_string(res.status), "text/plain");
        return httplib::Server::HandlerResponse::Handled;
    });
}

int serve_ui(const std::string& out_dir, int port, const std::string& ui_path,
             const std::string& spec_path, const std::string& self) {
    const std::string ui = slurp(ui_path);
    if (ui.empty()) throw std::runtime_error("cannot read UI at " + ui_path + " (set MTSTUDIO_UI)");
    httplib::Server svr;
    svr.set_default_headers({{"Cache-Control", "no-store"}});
    svr.set_payload_max_length(1 << 20);
    plain_errors(svr);

    // The in-page Train button: POST /train launches this binary as
    // "mtstudio run <spec>" with output logged into out_dir. The body, when
    // present, is the spec the page built (form edits or an extracted
    // paper): it is checked with parse_spec, its out_dir is set to the
    // served out_dir so the live view follows the run, and it is written to
    // out_dir/train_spec.json. An empty body trains the armed spec. One run
    // at a time; state is reaped non-blockingly per request. POST /fetch
    // launches papers/fetch.py the same way (the drag-an-arXiv-id flow).
    // Handlers run on httplib's worker threads: `mu` guards this state.
    std::mutex mu, sample_mu;
    Child run_child, fetch_child, extract_child;
    std::string extract_state = "idle", extract_path, extract_id;
    struct ExtractCleanup {
        Child& child;
        ~ExtractCleanup() { cancel_child(child); }
    } extract_cleanup{extract_child};
    bool run_finished = false, fetch_failed = false;
    std::string trained_spec = spec_path;  // what /sample reads: the last spec trained
    svr.set_pre_routing_handler([&](const httplib::Request&, httplib::Response&) {
        std::lock_guard<std::mutex> lk(mu);
        if (reap(run_child, nullptr)) run_finished = true;
        bool ok = true;
        if (reap(fetch_child, &ok)) fetch_failed = !ok;
        if (reap(extract_child, &ok))
            extract_state = ok && !slurp(extract_path).empty() ? "done" : "failed";
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.Post("/train", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(mu);
        if (run_child.running()) {
            text_reply(res, 200, "training…");
            return;
        }
        std::string to_run = spec_path;
        if (!req.body.empty()) {
            // The page's spec: validate, pin its out_dir, write it beside the run.
            const std::string path = out_dir + "/train_spec.json";
            try {
                json j = json::parse(req.body, nullptr, true, /*ignore_comments=*/true);
                if (!j.is_object()) throw std::runtime_error("spec must be a JSON object");
                studio_probes::validate_spec(j);
                j["out_dir"] = out_dir;
                std::ofstream(path, std::ios::trunc) << j.dump(2) << "\n";
                parse_spec(path);
            } catch (const std::exception& e) {
                text_reply(res, 400, std::string("spec rejected: ") + e.what());
                return;
            }
            to_run = path;
            run_finished = false;
        }
        if (to_run.empty()) {
            text_reply(res, 409, "no spec armed (serve <dir> <port> <spec>)");
        } else if (run_finished) {
            text_reply(res, 200, "run complete");
        } else {
            const bool ok = spawn(run_child, {self, "run", to_run}, out_dir + "/run.log");
            if (ok) trained_spec = to_run;
            text_reply(res, 200, ok ? "training…" : "fork failed");
        }
    });
    svr.Post("/fetch", [&](const httplib::Request& req, httplib::Response& res) {
        // Body = an arXiv id or URL. Launches the paper fetcher, which
        // writes arch.json + paper.html into out_dir; the page polls
        // /fetchstatus, then reads both over this same server.
        const std::string id = sanitize_arxiv(req.body);
        std::lock_guard<std::mutex> lk(mu);
        if (id.empty()) {
            text_reply(res, 400, "not an arXiv id (want 1706.03762-style, or an arxiv.org URL)");
        } else if (fetch_child.running()) {
            text_reply(res, 200, "fetching…");
        } else {
            std::remove((out_dir + "/arch.json").c_str());
            std::remove((out_dir + "/paper.html").c_str());
            fetch_failed = false;
            const char* fp = std::getenv("MTSTUDIO_FETCH");
            const char* py = std::getenv("MTSTUDIO_PYTHON");
#ifdef _WIN32
            const std::string python = py ? py : "python";
#else
            const std::string python = py ? py : "python3";
#endif
            const bool ok = spawn(fetch_child,
                                  {python, fp ? fp : "papers/fetch.py", id, "--json",
                                   out_dir + "/arch.json", "--emit-html", out_dir + "/paper.html"},
                                  out_dir + "/fetch.log");
            text_reply(res, 200, ok ? "fetching " + id + "…" : "fork failed");
        }
    });
    // A localhost listener alone does not prevent a foreign web page from
    // sending a simple POST. Paid calls require this exact local origin.
    auto extraction_request_allowed = [&](const httplib::Request& req) {
        const auto local = "127.0.0.1:" + std::to_string(port);
        const auto named = "localhost:" + std::to_string(port);
        const auto host = req.get_header_value("Host");
        if (host != local && host != named) return false;
        if (!req.has_header("Origin")) return true;  // explicit local CLI requests
        const auto origin = req.get_header_value("Origin");
        return origin == "http://" + local || origin == "http://" + named;
    };
    svr.Post("/extract", [&](const httplib::Request& req, httplib::Response& res) {
        if (!extraction_request_allowed(req)) {
            text_reply(res, 403, "extraction requests must come from this local Studio origin");
            return;
        }
        std::lock_guard<std::mutex> lk(mu);
        if (extract_child.running()) {
            text_reply(res, 409, "an extraction is already running");
            return;
        }
        const char* key = std::getenv("ANTHROPIC_API_KEY");
        if (!key || !*key) {
            text_reply(res, 409,
                       "ANTHROPIC_API_KEY is not configured in the Studio server environment");
            return;
        }
        try {
            const auto request = json::parse(req.body);
            const std::string id = sanitize_arxiv(request.at("arxiv_id").get<std::string>());
            const double budget = request.at("budget_usd").get<double>();
            if (id.empty() || !std::isfinite(budget) || budget <= 0 || budget > 5)
                throw std::runtime_error(
                    "provide an arXiv id and a spend cap greater than zero, at most $5");
            extract_id =
                "studio-" +
                std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
            extract_path = out_dir + "/" + extract_id + ".extraction.json";
            const char* py = std::getenv("MTSTUDIO_PYTHON");
#ifdef _WIN32
            const std::string python = py ? py : "python";
#else
            const std::string python = py ? py : "python3";
#endif
            const char* worker = std::getenv("MTSTUDIO_EXTRACT");
#ifdef _WIN32
            const auto parent_pid = GetCurrentProcessId();
#else
            const auto parent_pid = ::getpid();
#endif
            const bool ok = spawn(extract_child,
                                  {python, worker ? worker : "tools/studio_extract.py", id, "--out",
                                   extract_path, "--budget-usd", json(budget).dump(), "--tag",
                                   extract_id, "--parent-pid", std::to_string(parent_pid)},
                                  out_dir + "/" + extract_id + ".log");
            extract_state = ok ? "extracting" : "failed";
            text_reply(res, ok ? 202 : 500,
                       ok ? "extracting" : "could not start the extraction worker");
        } catch (const std::exception&) {
            text_reply(res, 400,
                       "provide JSON with arxiv_id and a finite positive budget_usd, at most $5");
        }
    });
    svr.Post("/extractcancel", [&](const httplib::Request& req, httplib::Response& res) {
        if (!extraction_request_allowed(req)) {
            text_reply(res, 403, "cancellation must come from this local Studio origin");
            return;
        }
        std::lock_guard<std::mutex> lk(mu);
        if (extract_child.running()) {
            if (!cancel_child(extract_child)) {
                text_reply(res, 500, "could not cancel the extraction worker");
                return;
            }
            extract_state = "cancelled";
        }
        text_reply(res, 200, extract_state);
    });
    svr.Get("/extractstatus", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(mu);
        const char* key = std::getenv("ANTHROPIC_API_KEY");
        json status = {{"state", extract_state}, {"configured", key && *key}, {"job", extract_id}};
        if (extract_state == "failed") {
            try {
                status["error"] = json::parse(slurp(extract_path + ".error.json")).at("error");
            } catch (...) {
                status["error"] = "extraction worker failed; no result was applied";
            }
        }
        text_reply(res, 200, status.dump(), "application/json");
    });
    svr.Get("/extractresult", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(mu);
        if (extract_state != "done") {
            text_reply(res, 409, "no completed extraction for the current job");
            return;
        }
        text_reply(res, 200, slurp(extract_path), "application/json");
    });
    svr.Post("/sample", [&](const httplib::Request& req, httplib::Response& res) {
        // In-page quick-look generation: body = the prompt. Runs the
        // `mtstudio sample` path in-process (a tiny model on CPU answers
        // in seconds), one request at a time; ember.cpp and `mtstudio
        // chat` are the real servers.
        std::string spec_now;
        {
            std::lock_guard<std::mutex> lk(mu);
            spec_now = trained_spec;
        }
        if (spec_now.empty()) {
            text_reply(res, 409, "no spec armed (serve <dir> <port> <spec>)");
            return;
        }
        std::lock_guard<std::mutex> lk(sample_mu);
        const std::string sf = out_dir + "/sample.txt";
        std::remove(sf.c_str());
        std::string log;
        try {
            sample_cmd(parse_spec(spec_now), req.body.empty() ? "once upon a time" : req.body, 40,
                       0.8f, 40, 1234, sf);
            log = "ok\n";
        } catch (const std::exception& e) {
            log = std::string("mtstudio: ") + e.what() + "\n";
        }
        std::ofstream(out_dir + "/sample.log", std::ios::trunc) << log;
        const std::string body = slurp(sf);
        if (!body.empty()) {
            text_reply(res, 200, body, "text/plain; charset=utf-8");
        } else {
            text_reply(res, 500,
                       "sampling failed — train (and export safetensors) first; "
                       "details in sample.log");
        }
    });
    svr.Get("/evaluation-probes", [&](const httplib::Request&, httplib::Response& res) {
        const auto content = slurp(out_dir + "/evaluation_probes.json");
        if (content.empty())
            text_reply(res, 404, "no evaluation probe receipt");
        else
            text_reply(res, 200, content, "application/json; charset=utf-8");
    });
    svr.Get("/fetchstatus", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(mu);
        const char* s = fetch_child.running()                    ? "fetching"
                        : fetch_failed                           ? "failed (see fetch.log)"
                        : !slurp(out_dir + "/arch.json").empty() ? "done"
                                                                 : "idle";
        text_reply(res, 200, s);
    });
    svr.Get("/spec", [&](const httplib::Request&, httplib::Response& res) {
        const std::string body = spec_path.empty() ? "" : slurp(spec_path);
        if (body.empty()) {
            text_reply(res, 404, "404");
        } else {
            text_reply(res, 200, body, "application/json");
        }
    });
    svr.Get("/events.jsonl", [&](const httplib::Request&, httplib::Response& res) {
        text_reply(res, 200, slurp(out_dir + "/events.jsonl"), "application/jsonl");
    });
    svr.Get(R"(/|/index\.html)", [&](const httplib::Request&, httplib::Response& res) {
        text_reply(res, 200, ui, "text/html; charset=utf-8");
    });
    const auto cut = ui_path.find_last_of("/\\");
    const std::string ui_dir = cut == std::string::npos ? "." : ui_path.substr(0, cut);
    svr.Get("/findings", [&](const httplib::Request&, httplib::Response& res) {
        // The findings registry, served from the repo beside the UI
        // (studio/../atlas/findings.jsonl) — the viewer renders it as
        // cards with status badges.
        const std::string body = slurp(ui_dir + "/../atlas/findings.jsonl");
        if (!body.empty()) {
            text_reply(res, 200, body, "application/jsonl");
        } else {
            text_reply(res, 404, "404");
        }
    });
    svr.Get(R"(/atlas(\.html)?)", [&](const httplib::Request&, httplib::Response& res) {
        // The designed-experiment viewer, served from beside the UI; in
        // served mode it auto-loads the out_dir's atlas_rows.jsonl.
        const std::string body = slurp(ui_dir + "/atlas.html");
        if (!body.empty()) {
            text_reply(res, 200, body, "text/html; charset=utf-8");
        } else {
            text_reply(res, 404, "404");
        }
    });
    svr.Get(R"(/([^/]+))", [&](const httplib::Request& req, httplib::Response& res) {
        // Serve sibling files from out_dir (the diff-to-paper page, the
        // fetched arch.json and atlas_rows.jsonl ride the same localhost
        // as the dashboard, so no file:// URL gymnastics from
        // Windows/WSL). Name only — no slashes or dots-paths — and a fixed
        // extension whitelist.
        const std::string name = req.matches[1];
        const char* ctype = nullptr;
        auto ends = [&](const char* suf) {
            const size_t n = std::strlen(suf);
            return name.size() > n && name.compare(name.size() - n, n, suf) == 0;
        };
        if (ends(".html")) ctype = "text/html; charset=utf-8";
        if (ends(".json")) ctype = "application/json";
        if (ends(".jsonl")) ctype = "application/jsonl";
        if (ends(".log")) ctype = "text/plain; charset=utf-8";
        // Trained-artifact downloads (the page's export links).
        if (ends(".safetensors") || ends(".gguf")) ctype = "application/octet-stream";
        const bool safe =
            ctype && name.find('\\') == std::string::npos && name.find("..") == std::string::npos;
        const std::string body = safe ? slurp(out_dir + "/" + name) : "";
        if (!body.empty()) {
            text_reply(res, 200, body, ctype);
        } else {
            text_reply(res, 404, "404");
        }
    });

    if (!svr.bind_to_port("127.0.0.1", port))
        throw std::runtime_error("bind failed (port in use?)");
    std::printf(
        "mtstudio serve: http://localhost:%d/  (events from %s, "
        "Ctrl-C to stop)\n",
        port, out_dir.c_str());
    std::fflush(stdout);
    svr.listen_after_bind();
    return 0;
}

// ---- mtstudio chat: a trained model behind a chat page ----
//   GET  /        the chat page (studio/chat.html, embedded at build time;
//                 MTSTUDIO_CHAT_UI=<path> serves that file instead, re-read
//                 per request, for editing the page without rebuilding)
//   GET  /card    <out_dir>/card.json (tools/model_card.py), else 404
//   GET  /health  liveness and what is loaded
//   POST /chat    {user_input, max_new_tokens?, temperature?, top_k?,
//                 top_p?, history?, seed?} -> {reply, stop_reason, tokens}
//                 stop_reason: "eos" | "turn" | "length"
// The request shape is tinyllama.cpp server.cpp's, so either engine can
// answer the studio's chat panel. The prompt is the dialogue template of
// the word-level chat corpora, "user: <text> assistant:", in the model's
// own tokenisation (history turns, oldest first, prefix it in the same
// form); generation stops at end-of-text, when the model opens another
// turn ("user :", or a second "assistant :"; the marker is dropped), or
// at max_new_tokens. The model loads once; a mutex
// serialises generation across httplib's worker threads.

// `<out_dir>` (its spec.json, recorded by `mtstudio run`) or a spec file.
Spec resolve_chat_spec(const std::string& arg) {
    namespace fs = std::filesystem;
    if (fs::is_directory(arg)) {
        const std::string sp = (fs::path(arg) / "spec.json").string();
        if (!fs::exists(sp))
            throw std::runtime_error(arg +
                                     " has no spec.json (runs from before it was recorded: "
                                     "pass the spec file instead)");
        Spec s = parse_spec(sp);
        s.out_dir = arg;  // the directory named wins: run dirs get moved
        return s;
    }
    return parse_spec(arg);
}

// The request's history as whole (user, assistant) turns, oldest first:
// [{role, content}] lists pair each assistant message with the user message
// before it, as chat_prompt_bpe reads them.
// Role-form history pairs strictly (section 3 point 3a, as ember's /chat): after an optional
// leading system entry (the caller reads it), user and assistant alternate, starting with user and
// ending with assistant (the new message is user_input); anything else is refused by name
// (chat::Error, HTTP 400). Pair-form entries {user, assistant} are whole turns.
json history_turns(const json& req) {
    json turns = json::array();
    if (!req.contains("history") || !req["history"].is_array()) return turns;
    bool have_user = false;
    std::string pending_user;
    const json& h = req["history"];
    for (size_t i = 0; i < h.size(); ++i) {
        const json& turn = h[i];
        if (!turn.is_object()) continue;
        if (turn.contains("role")) {
            const std::string role = turn.value("role", "");
            if (role == "system" && i == 0) continue;
            if (role == "user") {
                if (have_user)
                    throw chat::Error("history has two user messages in a row: user and assistant messages "
                                      "must alternate (section 3 point 3a)");
                pending_user = turn.value("content", "");
                have_user = true;
            } else if (role == "assistant") {
                if (!have_user)
                    throw chat::Error("history has an assistant message without a user message before it: user "
                                      "and assistant messages must alternate, starting with user (section 3 point 3a)");
                turns.push_back({{"user", pending_user}, {"assistant", turn.value("content", "")}});
                pending_user.clear();
                have_user = false;
            } else {
                throw chat::Error("history role '" + role + "' is not allowed here (user, assistant; system only first)");
            }
            continue;
        }
        if (have_user)
            throw chat::Error("history has a user message without its assistant reply before a {user, assistant} pair");
        turns.push_back({{"user", turn.value("user", "")}, {"assistant", turn.value("assistant", "")}});
    }
    if (have_user)
        throw chat::Error("history ends with a user message: it must end with an assistant reply (the new message "
                          "goes in user_input)");
    return turns;
}

// The served prompt (LOCAL_LLM_STACK.md section 3, points 3-5): the request's history as whole
// (user, assistant) turns (history_turns), the optional system message and the new message,
// rendered by the model's chat template (microtorch/chat_template.hpp). paperkiln/1 gives
// <|user|>U<|assistant|>A<|endoftext|> per past turn, then <|user|>U<|assistant|>, as the string
// builder it replaced (tests/test_chat_template.cpp); paperkiln/legacy-word gives the word-level
// "user: U assistant: A " form. The system message (fold under paperkiln/1) is never dropped.
std::string chat_prompt(const json& req, const chat::Template& tmpl, const std::string* system, int thinking = -1) {
    std::vector<std::pair<std::string, std::string>> turns;
    for (const auto& t : history_turns(req))
        turns.push_back({t["user"].get<std::string>(), t["assistant"].get<std::string>()});
    const std::string user = req["user_input"].get<std::string>();
    return chat::render(chat::from_pairs(turns, &user, system), tmpl, /*generation_prompt=*/true, thinking);
}

int chat_cmd(const std::string& target, const std::string& host, int port, bool kv_cache, bool evict) {
    const Spec s = resolve_chat_spec(target);
    LoadedLM lm(s);
    if (kv_cache && !lm.llama) throw std::runtime_error("chat --kv-cache: only the llama family has a KV cache");
    // --evict (ir.hpp kv evict "turn_state"): finished turns keep only their
    // state rows in the cache; the replies are those of --kv-cache alone.
    if (evict && (!kv_cache || s.state_k == 0 || s.state_mask != "turn_state"))
        throw std::runtime_error("chat --evict needs --kv-cache and a state-token model with the turn_state mask");
    lm.kv_on = kv_cache;
    // The served context is the spec's T, and the run's exported GGUF declares the T it was exported at
    // (every exporter writes s.T): the two must agree, or another engine serving the file would drop
    // turns and stop at a different length (the v2 gate, 27 Oct 2026: a GGUF exported at 256, served at 512).
    if (const std::string g = s.out_dir + "/" + s.name + ".gguf"; std::filesystem::exists(g)) {
        const long c = gguf_context_length(g);
        if (c >= 0 && c != static_cast<long>(s.T))
            throw std::runtime_error("chat: " + g + " declares context " + std::to_string(c) + " but the spec's data.T is " +
                                     std::to_string(s.T) + ": re-export the GGUF at the served T (mtstudio export)");
    }
    lm.kv.evict = evict;
    // The model's own token sequences for a new turn ("user :",
    // "assistant :"), unusable if a piece is out of vocabulary (it would
    // be <unk>, which the sampler never emits anyway).
    // The template's turn stops: a reply ends where the model would open a new turn (paperkiln/1:
    // <|user|>, <|assistant|>; legacy-word: "user:", "assistant:", skipped when out of vocabulary).
    const chat::Template tmpl = chat_template_of(s);
    std::vector<std::vector<int>> turn_markers;
    for (const auto& m : tmpl.stop_turn) {
        auto seq = tokenize(m, lm.vocab, g_bpe ? 4 : 16);
        if (g_bpe || std::find(seq.begin(), seq.end(), 0) == seq.end()) turn_markers.push_back(seq);
    }
    // The repetition penalty's exempt stop set: end-of-text and the
    // single-token BPE turn markers (the ids generation stops on).
    std::vector<int> stop_ids;
    if (lm.eos >= 0) stop_ids.push_back(lm.eos);
    for (const auto& m : turn_markers)
        if (m.size() == 1) stop_ids.push_back(m[0]);
    std::mutex gen_mu;
    std::mt19937 seeder{std::random_device{}()};

    httplib::Server svr;
    svr.set_default_headers({{"Cache-Control", "no-store"}});
    svr.set_payload_max_length(1 << 20);
    plain_errors(svr);
    auto cors = [](httplib::Response& res) {
        // Open, like tinyllama_server: the studio page (another port)
        // posts here.
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    };

    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        const char* override_path = std::getenv("MTSTUDIO_CHAT_UI");
        const std::string page =
            override_path ? slurp(override_path)
                          : std::string(reinterpret_cast<const char*>(kChatHtml), kChatHtmlSize);
        if (page.empty())
            text_reply(res, 404, "chat page not found at MTSTUDIO_CHAT_UI");
        else
            text_reply(res, 200, page, "text/html; charset=utf-8");
    });
    svr.Get("/card", [&](const httplib::Request&, httplib::Response& res) {
        const std::string body = slurp(s.out_dir + "/card.json");
        if (body.empty())
            text_reply(res, 404, "no card.json in " + s.out_dir);
        else
            text_reply(res, 200, body, "application/json");
    });
    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(json({{"status", "ok"},
                              {"model", s.name},
                              {"family", s.family},
                              {"attention", s.attention},
                              {"rope_heads", lm.rope_heads.empty() ? json() : json(lm.rope_heads)},
                              {"params", lm.model().parameter_count()},
                              {"vocab", lm.tokens.size()},
                              {"context", s.T},
                              {"state_tokens", s.state_k},
                              {"chat_template", {{"name", tmpl.name}, {"version", json::parse(tmpl.version.dump())}}},
                              {"eos", lm.eos >= 0}})
                            .dump(),
                        "application/json");
    });
    svr.Options("/chat", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        res.status = 204;
    });
    svr.Post("/chat", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto fail = [&](int status, const std::string& msg) {
            res.status = status;
            res.set_content(json({{"error", msg}}).dump(), "application/json");
        };
        const json r = json::parse(req.body, nullptr, false);
        // v2 (a template with tool turns): user_input may be left out when the history ends on a tool
        // message (the reply follows the tool results).
        if (r.is_discarded() || !r.is_object() ||
            (r.contains("user_input") && !r["user_input"].is_null() ? !r["user_input"].is_string() : !tmpl.has_tool)) {
            fail(400, "body must be JSON with a string 'user_input'");
            return;
        }
        try {
            // Defaults are tinyllama_server's.
            const int max_new = std::min(std::max(r.value("max_new_tokens", 60), 1), 1024);
            const float temp = r.value("temperature", 0.1f);
            const int top_k = r.value("top_k", 40);
            const float top_p = r.value("top_p", 0.9f);
            // Repetition penalty (section 3 point 4): p in [1, 2], n in [0, ctx];
            // n defaults to 64, or the context when that is smaller.
            const float rep_p = r.value("repeat_penalty", 1.0f);
            const long rep_n = r.value("repeat_last_n", std::min(64L, static_cast<long>(s.T)));
            if (!(rep_p >= 1.0f && rep_p <= 2.0f) || rep_n < 0 || rep_n > static_cast<long>(s.T)) {
                fail(400, "repeat_penalty must be in [1, 2] and repeat_last_n in [0, " + std::to_string(s.T) + "]");
                return;
            }
            // The context contract (LOCAL_LLM_STACK.md section 3, shared with
            // ember's server): drop the OLDEST whole turn while prompt +
            // max_new_tokens exceeds the context; reject a prompt that still
            // fills it (the newest message is never truncated); and never
            // attend over more than T positions, so a reply that reaches the
            // limit ends with P + reply = T + 1 and stop_reason "length".
            const size_t T = s.T;
            // The system message (section 3 point 4): "system", or a leading {"role": "system"}
            // history entry, not both and nowhere else; empty means none. It is pinned: only
            // whole turns are dropped below.
            std::string system;
            bool has_system = false;
            if (r.contains("system")) {
                if (!r["system"].is_string()) {
                    fail(400, "system must be a string");
                    return;
                }
                system = r["system"].get<std::string>();
                has_system = true;
            }
            if (r.contains("history") && r["history"].is_array())
                for (size_t i = 0; i < r["history"].size(); ++i) {
                    const json& h = r["history"][i];
                    if (!h.is_object() || h.value("role", "") != "system") continue;
                    if (i != 0) {
                        fail(400, "a system message is allowed only as the first history entry");
                        return;
                    }
                    if (has_system) {
                        fail(400, "give the system message once: the 'system' field or a leading history entry");
                        return;
                    }
                    system = h.value("content", "");
                    has_system = true;
                }
            has_system = has_system && !system.empty();
            // "thinking": false selects a template's no-think generation prompt (ignored without one).
            const int thinking = r.contains("thinking") && r["thinking"].is_boolean() ? (r["thinking"].get<bool>() ? 1 : 0) : -1;
            std::vector<int> ids;
            size_t turns_used = 0;
            if (!tmpl.has_tool) {
                const json turns = history_turns(r);
                json q = r;
                size_t first = 0;
                while (true) {
                    q["history"] = json(turns.begin() + static_cast<std::ptrdiff_t>(first), turns.end());
                    const std::string prompt = chat_prompt(q, tmpl, has_system ? &system : nullptr, thinking);
                    ids = tokenize(prompt, lm.vocab, prompt.size() + 1);
                    // State tokens: the template's blocks after every finished
                    // reply (the history's replies all end with <|endoftext|>).
                    if (lm.st.k > 0) ids = nn::insert_state_ids(ids, lm.st);
                    if (first == turns.size() || ids.size() + static_cast<size_t>(max_new) <= T) break;
                    ++first;
                }
                turns_used = turns.size() - first;
            } else {
                // v2 (paperkiln/2): the request as messages. History entries in role form pass through
                // as they are (user, assistant with optional tool_calls, tool with optional
                // tool_call_id; the template checks the structure), pair-form entries are a user and an
                // assistant message; then user_input when given. Tool definitions ("tools") and the
                // history are read as ordered JSON (definitions render in their own member order).
                const chat::json ro = chat::json::parse(req.body);
                std::vector<chat::json> tools;
                if (ro.contains("tools") && !ro["tools"].is_null()) {
                    if (!ro["tools"].is_array()) throw chat::Error("tools must be a list of tool definitions");
                    for (const auto& t : ro["tools"]) tools.push_back(t);
                }
                chat::json hist = chat::json::array();
                if (has_system) hist.push_back({{"role", "system"}, {"content", system}});
                if (ro.contains("history") && ro["history"].is_array())
                    for (size_t i = 0; i < ro["history"].size(); ++i) {
                        const auto& h = ro["history"][i];
                        if (!h.is_object()) continue;
                        if (h.contains("role")) {
                            if (h["role"] == "system") continue;  // taken above (first entry only)
                            hist.push_back(h);
                        } else {
                            hist.push_back({{"role", "user"}, {"content", h.value("user", std::string())}});
                            hist.push_back({{"role", "assistant"}, {"content", h.value("assistant", std::string())}});
                        }
                    }
                const bool has_input = ro.contains("user_input") && ro["user_input"].is_string();
                const std::string last = hist.empty() ? std::string() : hist.back().value("role", std::string());
                if (has_input && last == "user")
                    throw chat::Error("history ends with a user message: it must end with an assistant reply or a tool "
                                      "result (the new message goes in user_input)");
                if (!has_input && last != "tool")
                    throw chat::Error("user_input may be left out only when the history ends on a tool message");
                if (has_input) hist.push_back({{"role", "user"}, {"content", ro["user_input"].get<std::string>()}});
                const std::vector<chat::Message> msgs = chat::messages_from_json(hist);
                // The context rule (section 3 point 5, v2): drop the OLDEST whole exchange (a user message
                // and everything before the next one) while prompt + max_new_tokens exceeds the context;
                // the system turn and its tool definitions are pinned, the newest exchange is never cut.
                const auto exr = chat::exchanges(msgs);
                const size_t head = exr.empty() ? msgs.size() : exr[0].first;
                size_t first = 0;
                while (true) {
                    std::vector<chat::Message> kept(msgs.begin(), msgs.begin() + static_cast<std::ptrdiff_t>(head));
                    if (!exr.empty())
                        kept.insert(kept.end(), msgs.begin() + static_cast<std::ptrdiff_t>(exr[first].first), msgs.end());
                    const std::string prompt = chat::render(kept, tmpl, /*generation_prompt=*/true, thinking, tools);
                    ids = tokenize(prompt, lm.vocab, prompt.size() + 1);
                    if (exr.empty() || first + 1 >= exr.size() || ids.size() + static_cast<size_t>(max_new) <= T) break;
                    ++first;
                }
                turns_used = exr.empty() ? 0 : exr.size() - 1 - first;
            }
            if (ids.size() >= T) {
                fail(400, "message is " + std::to_string(ids.size()) + " tokens; the model's context is " +
                              std::to_string(T));
                return;
            }
            const std::vector<int> prompt_ids = ids;
            std::lock_guard<std::mutex> lk(gen_mu);
            std::mt19937 gen(r.contains("seed") ? r["seed"].get<unsigned>() : seeder());
            NoGrad ng;
            std::vector<int> out;
            std::string stop = "length";
            int stop_id = -1;
            for (int t = 0; t < max_new; ++t) {
                if (ids.size() > T) break;  // the context limit: "length"
                const int pick = lm.next(ids, temp, top_k, top_p, gen, rep_p, static_cast<size_t>(rep_n), stop_ids);
                if (pick == lm.eos) {
                    stop = "eos";
                    stop_id = pick;
                    break;
                }
                ids.push_back(pick);
                out.push_back(pick);
                for (const auto& m : turn_markers)
                    if (wordtok::ends_with(out, m)) {
                        out.resize(out.size() - m.size());
                        stop = "turn";
                        if (m.size() == 1) stop_id = m[0];
                        break;
                    }
                if (stop == "turn") break;
            }
            auto decode = [&](const std::vector<int>& v) {
                return g_bpe ? g_bpe->decode(v) : wordtok::detokenize(v, lm.tokens);
            };
            auto trim = [](const std::string& x) {
                const auto b = x.find_first_not_of(" \t\r\n");
                const auto e = x.find_last_not_of(" \t\r\n");
                return b == std::string::npos ? std::string() : x.substr(b, e - b + 1);
            };
            std::string reply = decode(out);
            json calls = json::array();
            if (tmpl.has_tool) {
                // v2: the reply split on the <|tool_call|> id (chat::parse_reply); a part that does not
                // parse leaves the whole reply as content. Control tokens and spares decode as their text.
                const auto cid = tokenize(tmpl.tool_call_open, lm.vocab, 4);
                const chat::ParsedReply pr = chat::parse_reply(out, cid.size() == 1 ? cid[0] : -2, decode);
                if (pr.parsed) {
                    reply = pr.content;
                    for (const auto& c : pr.tool_calls)
                        calls.push_back({{"id", c.id},
                                         {"type", "function"},
                                         {"function", {{"name", c.name}, {"arguments", c.arguments}}}});
                }
            }
            reply = trim(reply);
            json body = {{"reply", reply},
                         {"stop_reason", stop},
                         {"tokens", out.size()},
                         {"history_turns_used", turns_used}};
            if (!calls.empty()) body["tool_calls"] = calls;
            // return_ids: the prompt ids, the reply ids (without the stop) and the stop id (-1 at "length",
            // ember's convention, api.h), as ember's /chat reports them (the parity and prompt checks read them).
            if (r.value("return_ids", false)) {
                body["prompt_ids"] = prompt_ids;
                body["ids"] = out;
                body["stop_id"] = stop_id;
            }
            // The cache's rows per layer after the reply (state models: n k
            // state rows plus the current turn under --evict).
            if (lm.kv_on) body["kv_rows"] = lm.kv.rows();
            res.set_content(body.dump(), "application/json");
        } catch (const chat::Error& e) {
            fail(400, e.what());  // the template refuses the conversation (section 3 point 3)
        } catch (const json::exception& e) {
            fail(400, std::string("bad request field: ") + e.what());
        } catch (const std::exception& e) {
            fail(500, std::string("generation failed: ") + e.what());
        }
    });

    if (!svr.bind_to_port(host, port))
        throw std::runtime_error("cannot bind " + host + ":" + std::to_string(port) +
                                 " (port in use?)");
    const bool dialogue = lm.vocab.count("user") && lm.vocab.count("assistant");
    std::printf("mtstudio chat: http://%s:%d/  (%s, %zu params, vocab %zu, end-of-text %s)\n",
                host.c_str(), port, s.name.c_str(), lm.model().parameter_count(), lm.tokens.size(),
                lm.eos >= 0 ? "on" : "absent: replies run to max_new_tokens");
    if (!dialogue)
        std::printf(
            "mtstudio chat: this vocabulary has no 'user'/'assistant' words, so the "
            "model was not trained on dialogue; replies will read as continuations\n");
    if (host != "127.0.0.1" && host != "localhost" && host != "::1")
        std::printf(
            "mtstudio chat: listening beyond this machine (%s); anyone who can reach it "
            "can use the model\n",
            host.c_str());
    std::printf("Ctrl-C to stop\n");
    std::fflush(stdout);
    svr.listen_after_bind();
    return 0;
}

// `mtstudio score`: the log-probability of given continuations under the
// model (tools/carry5_eval.py's forced-choice scorer). Each input line holds
// a templated "prompt" and a "continuation" (or "prompt_ids" /
// "continuation_ids"); both are tokenised on their own with the model's
// tokeniser, as `chat` tokenises a prompt, and concatenated. One full
// forward pass (no KV cache, as `chat` without --kv-cache) over all ids but
// the last gives, for every continuation token, its log-probability under
// the softmax at temperature 1 (double precision) and the argmax with the
// specials `chat` bans from generation (<unk>, <s>, </s>, <pad>) left out,
// so a continuation equal to `chat`'s greedy reply plus its stop token has
// argmax == continuation at every position (the parity gate in
// carry5_eval.py). A sequence longer than T + 1 ids is reported over_ctx
// and not scored: `chat` never attends past T positions.
// State-token models (arch.state_tokens): a text prompt gets the template's
// state blocks (nn::insert_state_ids over prompt + continuation, so a
// continuation spanning a finished reply gets its block too, as context
// only: state ids are never scored); prompt_ids may hold state ids, each in
// a complete block after a reply's <|endoftext|> (nn::state_ids_error), and
// continuation_ids are text ids. --kv-cache scores through the KV cache,
// prefilled in chunks that end where a state block completes (and of at
// most --kv-step ids; 1 = token by token); --evict drops finished turns'
// text rows as it goes (turn_state only), with the same logits bit for bit
// (llama.hpp LlamaKV). Records then carry kv_rows (rows per layer held after
// the context) and state_blocks (complete blocks in the context).
// --logits FILE (25 Oct 2026, the CUDA-vs-CPU gate): every scored position's
// whole logits row (V float32, little-endian), appended in record order, so a
// record's rows are its continuation positions in order; each record then
// carries logits_rows. Nothing else changes.
//
// --batch-tokens N (opt-in, 28 Oct 2026; LOCAL_LLM_WORK E5.2): consecutive requests, in input order, share
// one forward of at most N context ids (a longer request runs alone). Each sequence keeps its own positions
// and attends within itself (nn::Llama::forward_packed_hidden: the attention per sequence as the unbatched
// forward runs it), the LM head runs on the scored rows only, and the batch runs inside one device step
// window (with MICROTORCH_STEP_RESIDENCY=1 the weights and activations stay on the device; --no-window
// leaves it out). The records are the unbatched ones: same fields, values computed by the same loop, same
// order. The log-probs equal the unbatched ones bit for bit when every GEMM gives a row the summation order
// it gets alone (the device's tiled GEMM; on the host, shapes for which ops::host_gemm_rows_free holds),
// and to float rounding otherwise (cuBLAS chooses its algorithm by shape; the host GEMM's recursion splits
// K by M): IR_SPEC 4.5's "same kernel" condition. Requests the packed forward has no path for (families
// other than llama, MoE, Attention Residuals, a routed recursion, turn_state prompts holding a state id)
// are scored one by one exactly as without the option. Not with --kv-cache or --logits. Without the
// option nothing changes.
int score_cmd(const std::string& target, const std::string& in_path, const std::string& out_path, bool kv_cache,
              bool evict, size_t kv_step, const std::string& logits_path = "", size_t batch_tokens = 0,
              bool batch_window = true) {
    const Spec s = resolve_chat_spec(target);
    if (batch_tokens > 0 && (kv_cache || !logits_path.empty()))
        throw std::runtime_error("score --batch-tokens: not with --kv-cache or --logits");
    LoadedLM lm(s);
    if (kv_cache && !lm.llama) throw std::runtime_error("score --kv-cache: only the llama family has a KV cache");
    if (evict && (!kv_cache || s.state_k == 0 || s.state_mask != "turn_state"))
        throw std::runtime_error("score --evict needs --kv-cache and a state-token model with the turn_state mask");
    std::ifstream in(in_path);
    if (!in) throw std::runtime_error("cannot open " + in_path);
    std::ofstream out(out_path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write " + out_path);
    std::ofstream lout;
    if (!logits_path.empty()) {
        lout.open(logits_path, std::ios::binary | std::ios::trunc);
        if (!lout) throw std::runtime_error("cannot write " + logits_path);
    }
    const size_t V = lm.tokens.size();
    std::vector<char> banned(V, 0);
    for (const char* sp : {"<unk>", "<s>", "</s>", "<pad>"}) {
        auto it = lm.vocab.find(sp);
        if (it != lm.vocab.end()) banned[static_cast<size_t>(it->second)] = 1;
    }
    const chat::Template tmpl = chat_template_of(s);
    auto ids_of = [&](const json& r, const char* ids_key, const char* text_key) {
        if (r.contains(ids_key)) return r[ids_key].get<std::vector<int>>();
        // {"messages": [...]}: a prompt conversation, rendered with the generation prompt.
        if (std::string(text_key) == "prompt" && r.contains("messages"))
            return tokenize(chat::render(chat::messages_from_json(r["messages"]), tmpl, true), lm.vocab, 1u << 30);
        return tokenize(r.at(text_key).get<std::string>(), lm.vocab, 1u << 30);
    };
    if (batch_tokens > 0) {
        // The unbatched loop below, split in two: every request is read and checked first (the same rules
        // and messages), then scored in packed batches.
        struct Req {
            json rec;
            std::vector<int> ctx, c;
            std::vector<size_t> at;  // at[j]: the position of continuation token j in prompt + continuation
            bool over = false;
        };
        std::vector<Req> reqs;
        std::string line;
        size_t n = 0, over = 0;
        while (std::getline(in, line)) {
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            const json r = json::parse(line);
            const std::vector<int> p = ids_of(r, "prompt_ids", "prompt");
            const std::vector<int> c = ids_of(r, "continuation_ids", "continuation");
            if (p.empty() || c.empty())
                throw std::runtime_error("score: line " + std::to_string(n) + " has an empty prompt or continuation");
            const size_t PV = V + (lm.st.k > 0 ? lm.st.k : 0);
            for (int t : p)
                if (t < 0 || static_cast<size_t>(t) >= PV) throw std::runtime_error("score: prompt id out of range");
            for (int t : c)
                if (t < 0 || static_cast<size_t>(t) >= V) throw std::runtime_error("score: continuation id out of range");
            Req q;
            q.rec = {{"i", n}, {"prompt_tokens", p.size()}, {"continuation_ids", c}};
            if (r.contains("key")) q.rec["key"] = r["key"];
            std::vector<int> ids = p;
            ids.insert(ids.end(), c.begin(), c.end());
            q.at.resize(c.size());
            for (size_t j = 0; j < c.size(); ++j) q.at[j] = p.size() + j;
            if (lm.st.k > 0) {
                if (r.contains("prompt_ids")) {
                    const std::string err = nn::state_ids_error(ids, lm.st);
                    if (!err.empty()) throw std::runtime_error("score: line " + std::to_string(n) + ": " + err);
                } else {
                    std::vector<uint8_t> fl(ids.size(), 0), f2;
                    for (size_t j = p.size(); j < ids.size(); ++j) fl[j] = 1;
                    ids = insert_state_flagged(ids, fl, lm.st, f2);
                    q.at.clear();
                    for (size_t k = 0; k < ids.size(); ++k)
                        if (f2[k]) q.at.push_back(k);
                    q.rec["prompt_tokens"] = q.at[0];
                }
            }
            q.c = c;
            if (q.at.back() + 1 > s.T + 1) {
                q.rec["over_ctx"] = true;
                q.over = true;
                ++over;
            } else {
                q.ctx.assign(ids.begin(), ids.begin() + static_cast<long>(q.at.back()));
            }
            reqs.push_back(std::move(q));
            ++n;
        }
        NoGrad ng;
        // The unbatched loop's per-request record from a logits matrix: row_of(j) is the row holding the
        // logits before continuation token j.
        auto finish = [&](Req& q, const Matrix& L, const std::function<size_t(size_t)>& row_of) {
            if (L.cols() != V) throw std::runtime_error("score: logits width differs from the vocabulary");
            if (lm.st.k > 0) {
                int seg = 0, run = 0;
                for (int t : q.ctx) ops::turn_state_step(t, V, lm.st.k, seg, run);
                q.rec["state_blocks"] = static_cast<size_t>(seg);
            }
            std::vector<double> lp;
            std::vector<int> am;
            double sum = 0.0;
            for (size_t j = 0; j < q.c.size(); ++j) {
                const size_t row = row_of(j);
                double mx = -1e300;
                int best = -1;
                double best_v = -1e300;
                for (size_t k = 0; k < V; ++k) {
                    const double x = L(row, k);
                    mx = std::max(mx, x);
                    if (!banned[k] && (best < 0 || x > best_v)) {
                        best = static_cast<int>(k);
                        best_v = x;
                    }
                }
                double z = 0.0;
                for (size_t k = 0; k < V; ++k) z += std::exp(static_cast<double>(L(row, k)) - mx);
                const double l = static_cast<double>(L(row, static_cast<size_t>(q.c[j]))) - mx - std::log(z);
                lp.push_back(l);
                am.push_back(best);
                sum += l;
            }
            q.rec["over_ctx"] = false;
            q.rec["logprobs"] = lp;
            q.rec["argmax"] = am;
            q.rec["sum"] = sum;
        };
        size_t batches = 0, packed = 0, alone = 0;
        for (size_t i = 0; i < reqs.size();) {
            if (reqs[i].over) {
                ++i;
                continue;
            }
            if (!lm.llama || !lm.llama->packable(reqs[i].ctx)) {
                // The unbatched path for this request.
                Req& q = reqs[i];
                const Var logits = lm.forward(q.ctx);
                if (lm.llama && lm.llama->rec_router) {
                    q.rec["depth"] = lm.llama->last_depth;
                    float mm = std::numeric_limits<float>::infinity();
                    for (float m : lm.llama->last_margin) mm = std::min(mm, m);
                    if (std::isfinite(mm)) q.rec["depth_margin_min"] = mm;
                }
                if (logits->data.rows() != q.ctx.size() || logits->data.cols() != V)
                    throw std::runtime_error("score: logits shape differs from the context");
                finish(q, logits->data, [&](size_t j) { return q.at[j] - 1; });
                ++alone;
                ++i;
                continue;
            }
            // A batch: consecutive packable requests (over_ctx ones are skipped) while the ids fit.
            std::vector<size_t> members;
            size_t tok = 0, k = i;
            for (; k < reqs.size(); ++k) {
                if (reqs[k].over) continue;
                if (!lm.llama->packable(reqs[k].ctx)) break;
                if (!members.empty() && tok + reqs[k].ctx.size() > batch_tokens) break;
                members.push_back(k);
                tok += reqs[k].ctx.size();
            }
            std::vector<int> ids;
            ids.reserve(tok);
            std::vector<std::pair<size_t, size_t>> segs;
            std::vector<int> rows;
            std::vector<size_t> first_row;  // per member: its first row in `rows`
            for (size_t m : members) {
                const Req& q = reqs[m];
                segs.emplace_back(ids.size(), q.ctx.size());
                first_row.push_back(rows.size());
                for (size_t j = 0; j < q.c.size(); ++j) rows.push_back(static_cast<int>(ids.size() + q.at[j] - 1));
                ids.insert(ids.end(), q.ctx.begin(), q.ctx.end());
            }
            if (batch_window) device::step_begin();
            Var logits = lm.llama->head(lm.llama->forward_packed_hidden(ids, segs, rows));
            if (batch_window) device::step_end();
            if (logits->data.rows() != rows.size()) throw std::runtime_error("score: packed logits rows differ");
            for (size_t b = 0; b < members.size(); ++b) {
                const size_t r0 = first_row[b];
                finish(reqs[members[b]], logits->data, [r0](size_t j) { return r0 + j; });
            }
            ++batches;
            packed += members.size();
            i = k;
        }
        for (const Req& q : reqs) out << q.rec.dump() << "\n";
        std::fprintf(stderr,
                     "score: %zu continuations (%zu over the context of %zu; %zu in %zu packed batches of at most %zu "
                     "ids, %zu alone) -> %s\n",
                     n, over, static_cast<size_t>(s.T), packed, batches, batch_tokens, alone, out_path.c_str());
        return 0;
    }
    NoGrad ng;
    std::string line;
    size_t n = 0, over = 0;
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        const json r = json::parse(line);
        const std::vector<int> p = ids_of(r, "prompt_ids", "prompt");
        const std::vector<int> c = ids_of(r, "continuation_ids", "continuation");
        if (p.empty() || c.empty())
            throw std::runtime_error("score: line " + std::to_string(n) + " has an empty prompt or continuation");
        const size_t PV = V + (lm.st.k > 0 ? lm.st.k : 0);  // prompt ids may be state ids
        for (int t : p)
            if (t < 0 || static_cast<size_t>(t) >= PV) throw std::runtime_error("score: prompt id out of range");
        for (int t : c)
            if (t < 0 || static_cast<size_t>(t) >= V) throw std::runtime_error("score: continuation id out of range");
        json rec = {{"i", n}, {"prompt_tokens", p.size()}, {"continuation_ids", c}};
        if (r.contains("key")) rec["key"] = r["key"];
        std::vector<int> ids = p;
        ids.insert(ids.end(), c.begin(), c.end());
        // at[j]: the position of continuation token j in ids.
        std::vector<size_t> at(c.size());
        for (size_t j = 0; j < c.size(); ++j) at[j] = p.size() + j;
        if (lm.st.k > 0) {
            if (r.contains("prompt_ids")) {
                const std::string err = nn::state_ids_error(ids, lm.st);
                if (!err.empty()) throw std::runtime_error("score: line " + std::to_string(n) + ": " + err);
            } else {
                std::vector<uint8_t> fl(ids.size(), 0), f2;
                for (size_t j = p.size(); j < ids.size(); ++j) fl[j] = 1;
                ids = insert_state_flagged(ids, fl, lm.st, f2);
                at.clear();
                for (size_t q = 0; q < ids.size(); ++q)
                    if (f2[q]) at.push_back(q);
                rec["prompt_tokens"] = at[0];
            }
        }
        if (at.back() + 1 > s.T + 1) {
            rec["over_ctx"] = true;
            out << rec.dump() << "\n";
            ++n;
            ++over;
            continue;
        }
        const std::vector<int> ctx(ids.begin(), ids.begin() + static_cast<long>(at.back()));
        Var logits;
        if (kv_cache) {
            // Chunks that end where a state block completes (eviction acts
            // between them) and of at most kv_step ids.
            nn::LlamaKV kv;
            kv.evict = evict;
            Matrix all(ctx.size(), V);
            size_t q0 = 0;
            int seg = 0, run = 0;
            for (size_t q = 0; q < ctx.size(); ++q) {
                if (lm.st.k > 0) ops::turn_state_step(ctx[q], V, lm.st.k, seg, run);
                const bool block_end = lm.st.k > 0 && run == 0 && static_cast<size_t>(ctx[q]) == V + lm.st.k - 1;
                if (q + 1 == ctx.size() || block_end || (kv_step > 0 && q + 1 - q0 == kv_step)) {
                    const Matrix l =
                        lm.llama->forward_cached(std::vector<int>(ctx.begin() + static_cast<long>(q0),
                                                                  ctx.begin() + static_cast<long>(q + 1)), kv)->data;
                    for (size_t i = 0; i < l.rows(); ++i)
                        for (size_t k = 0; k < V; ++k) all(q0 + i, k) = l(i, k);
                    q0 = q + 1;
                }
            }
            rec["kv_rows"] = kv.rows();
            logits = make_var(std::move(all));
        } else {
            logits = lm.forward(ctx);
        }
        if (lm.st.k > 0) {
            size_t blocks = 0;
            int seg = 0, run = 0;
            for (int t : ctx) ops::turn_state_step(t, V, lm.st.k, seg, run);
            blocks = static_cast<size_t>(seg);
            rec["state_blocks"] = blocks;
        }
        if (lm.llama && lm.llama->rec_router) {
            // schedule "mor": the depth trace (ir.hpp constraint 5): each
            // context position's depth, 1 .. repeats, and the smallest
            // routing margin (largest router logit minus the runner-up) over
            // them, so a consumer can tell a near-tie from a disagreement.
            rec["depth"] = lm.llama->last_depth;
            float mm = std::numeric_limits<float>::infinity();
            for (float m : lm.llama->last_margin) mm = std::min(mm, m);
            if (std::isfinite(mm)) rec["depth_margin_min"] = mm;
        }
        if (logits->data.rows() != ctx.size() || logits->data.cols() != V)
            throw std::runtime_error("score: logits shape differs from the context");
        std::vector<double> lp;
        std::vector<int> am;
        double sum = 0.0;
        for (size_t j = 0; j < c.size(); ++j) {
            const size_t row = at[j] - 1;
            double mx = -1e300;
            int best = -1;
            double best_v = -1e300;
            for (size_t k = 0; k < V; ++k) {
                const double x = logits->data(row, k);
                mx = std::max(mx, x);
                if (!banned[k] && (best < 0 || x > best_v)) {
                    best = static_cast<int>(k);
                    best_v = x;
                }
            }
            double z = 0.0;
            for (size_t k = 0; k < V; ++k) z += std::exp(static_cast<double>(logits->data(row, k)) - mx);
            const double l = static_cast<double>(logits->data(row, static_cast<size_t>(c[j]))) - mx - std::log(z);
            lp.push_back(l);
            am.push_back(best);
            sum += l;
            if (lout.is_open()) {
                std::vector<float> rowv(V);
                for (size_t k = 0; k < V; ++k) rowv[k] = logits->data(row, k);
                lout.write(reinterpret_cast<const char*>(rowv.data()), static_cast<std::streamsize>(V * sizeof(float)));
            }
        }
        if (lout.is_open()) rec["logits_rows"] = c.size();
        rec["over_ctx"] = false;
        rec["logprobs"] = lp;
        rec["argmax"] = am;
        rec["sum"] = sum;
        out << rec.dump() << "\n";
        ++n;
    }
    std::fprintf(stderr, "score: %zu continuations (%zu over the context of %zu) -> %s\n", n, over,
                 static_cast<size_t>(s.T), out_path.c_str());
    return 0;
}
}  // namespace

// The SFT rows a spec trains on (load_sft), one JSON line per row {x, y, m} (positions after the
// last target trimmed), then a summary on stdout: the G2 gate (tests/test_chat_template_sft.py)
// compares conversation rows with the rendered rows they replace.
int sft_dump_cmd(const Spec& s, const std::string& out_path) {
    auto tokens = read_gguf_vocab(s.vocab_gguf);
    if (g_bpe && s.vocab_cap > 0 && s.vocab_cap < tokens.size())
        throw std::runtime_error("data.vocab_cap " + std::to_string(s.vocab_cap) + " is below the BPE vocabulary's " +
                                 std::to_string(tokens.size()) + " tokens: merged ids would fall outside the model");
    if (s.vocab_cap > 0 && s.vocab_cap < tokens.size()) tokens.resize(s.vocab_cap);
    std::map<std::string, int> vocab;
    for (size_t i = 0; i < tokens.size(); ++i) vocab.emplace(tokens[i], static_cast<int>(i));
    const SftSet d = load_sft(s, vocab);
    std::ofstream out(out_path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write " + out_path);
    for (size_t e = 0; e < d.x.size(); ++e) {
        size_t n = d.x[e].size();
        while (n > 0 && d.m[e][n - 1] == 0) --n;
        out << json({{"x", std::vector<int>(d.x[e].begin(), d.x[e].begin() + n)},
                     {"y", std::vector<int>(d.y[e].begin(), d.y[e].begin() + n)},
                     {"m", std::vector<int>(d.m[e].begin(), d.m[e].begin() + n)}})
                   .dump()
            << "\n";
    }
    std::printf("%s\n", json({{"rows", d.x.size()}, {"val_start", d.val_start}, {"target_tokens", d.target_tokens},
                              {"conversations", d.conv_rows}, {"conversation_rows", d.conv_out},
                              {"dropped_over_T", d.conv_over_T}, {"exchanges_cut_over_T", d.conv_exchanges_cut},
                              {"untrained", d.conv_untrained}, {"state_blocks", d.state_blocks},
                              {"state_rows_dropped", d.state_rows_dropped}, {"state_turns_cut", d.state_turns_cut}})
                            .dump()
                            .c_str());
    return 0;
}

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "";
    try {
        // Honour MICROTORCH_DEVICE / MICROTORCH_STEP_RESIDENCY like the
        // test suites do (no-ops on CPU builds). Rung C's CUDA runs go
        // through here (docs/CUDA_PHASE_B2.md).
        device::set_from_env();
        if ((cmd == "run" || cmd == "plan") && argc >= 3)
            return run(parse_spec(argv[2]), cmd == "plan", argv[2]);
        if (cmd == "probes" && argc >= 3) {
            Spec spec = parse_spec(argv[2]);
            if (spec.evaluation_probes.is_null())
                throw std::runtime_error("spec has no evaluation_probes");
            std::string output = spec.out_dir + "/evaluation_probes.json";
            if (argc == 5 && std::string(argv[3]) == "--out")
                output = argv[4];
            else if (argc != 3)
                throw std::runtime_error("probes spec.json [--out NEW.json]");
            studio_probes::ArtifactGuard guard(output);
            try {
                LoadedLM lm(spec);
                return evaluation_probes_stage(
                    spec, [&](const std::vector<int>& ids) { return lm.forward(ids); }, lm.tokens,
                    lm.vocab, output, lm.model(), lm.vocab_path, -1);
            } catch (const std::exception& error) {
                studio_probes::publish(output,
                                       studio_probes::error_report(spec.evaluation_probes, error));
                return 1;
            }
        }
        if (cmd == "sample" && argc >= 3) {
            std::string prompt = "once upon a time", out_file;
            int n_new = 40, topk = 40;
            float temp = 0.8f;
            unsigned sseed = 1234;
            for (int i = 3; i + 1 < argc; i += 2) {
                const std::string k = argv[i], v = argv[i + 1];
                if (k == "--prompt") prompt = v;
                if (k == "--tokens") n_new = std::atoi(v.c_str());
                if (k == "--temp") temp = static_cast<float>(std::atof(v.c_str()));
                if (k == "--topk") topk = std::atoi(v.c_str());
                if (k == "--seed") sseed = static_cast<unsigned>(std::atoi(v.c_str()));
                if (k == "--out") out_file = v;
            }
            return sample_cmd(parse_spec(argv[2]), prompt, n_new, temp, topk, sseed, out_file);
        }
        if (cmd == "sample-set" && argc >= 3) {
            std::string prompts, out;
            int n = 8, max_new = 16;
            float temp = 1.0f;
            unsigned sseed = 17;
            for (int i = 3; i + 1 < argc; i += 2) {
                const std::string k = argv[i], v = argv[i + 1];
                if (k == "--prompts") prompts = v;
                else if (k == "--out") out = v;
                else if (k == "--n") n = std::atoi(v.c_str());
                else if (k == "--max-new") max_new = std::atoi(v.c_str());
                else if (k == "--temp") temp = static_cast<float>(std::atof(v.c_str()));
                else if (k == "--seed") sseed = static_cast<unsigned>(std::atoll(v.c_str()));
                else throw std::runtime_error("sample-set: unknown option " + k);
            }
            if (prompts.empty() || out.empty()) throw std::runtime_error("sample-set needs --prompts and --out");
            return sample_set_cmd(parse_spec(argv[2]), prompts, out, n, temp, sseed, max_new);
        }
        if (cmd == "serve" && argc >= 3) {
            const int port = argc > 3 ? std::atoi(argv[3]) : 8123;
            const char* ui = std::getenv("MTSTUDIO_UI");
            // Optional 4th arg: a spec the browser can launch via the
            // in-page Train button (POST /train).
            const std::string spec = argc > 4 ? argv[4] : "";
            return serve_ui(argv[2], port, ui ? ui : "studio/index.html", spec, self_exe(argv[0]));
        }
        if (cmd == "export" && argc >= 3) {
            std::string from, formats, out_dir;
            for (int i = 3; i + 1 < argc; i += 2) {
                const std::string k = argv[i], v = argv[i + 1];
                if (k == "--from") from = v;
                else if (k == "--formats") formats = v;
                else if (k == "--out-dir") out_dir = v;
                else throw std::runtime_error("export: unknown option " + k);
            }
            return export_cmd(parse_spec(argv[2]), from, formats, out_dir);
        }
        if (cmd == "chat" && argc >= 3) {
            std::string host = "127.0.0.1";
            int port = 8080;
            bool kv_cache = false, evict = false;
            for (int i = 3; i < argc; ++i) {
                const std::string k = argv[i];
                if (k == "--kv-cache") {
                    kv_cache = true;
                } else if (k == "--evict") {
                    evict = true;
                } else if (i + 1 < argc && (k == "--port" || k == "--host")) {
                    const std::string v = argv[++i];
                    if (k == "--port") port = std::atoi(v.c_str());
                    if (k == "--host") host = v;
                }
            }
            return chat_cmd(argv[2], host, port, kv_cache, evict);
        }
        if (cmd == "sft-dump" && argc >= 3) {
            std::string out_path;
            for (int i = 3; i + 1 < argc; i += 2)
                if (std::string(argv[i]) == "--out") out_path = argv[i + 1];
            if (out_path.empty()) throw std::runtime_error("sft-dump needs --out");
            return sft_dump_cmd(parse_spec(argv[2]), out_path);
        }
        if (cmd == "score" && argc >= 3) {
            std::string in_path, out_path, logits_path;
            bool kv_cache = false, evict = false, window = true;
            size_t kv_step = 0, batch_tokens = 0;
            for (int i = 3; i < argc; ++i) {
                const std::string k = argv[i];
                if (k == "--kv-cache") kv_cache = true;
                else if (k == "--evict") evict = true;
                else if (k == "--no-window") window = false;
                else if (i + 1 < argc && k == "--batch-tokens") {
                    const long long v = std::atoll(argv[++i]);
                    if (v <= 0) throw std::runtime_error("score --batch-tokens needs a positive number of ids");
                    batch_tokens = static_cast<size_t>(v);
                }
                else if (i + 1 < argc && k == "--in") in_path = argv[++i];
                else if (i + 1 < argc && k == "--out") out_path = argv[++i];
                else if (i + 1 < argc && k == "--logits") logits_path = argv[++i];
                else if (i + 1 < argc && k == "--kv-step") kv_step = static_cast<size_t>(std::atoll(argv[++i]));
                else throw std::runtime_error("score: unknown option " + k);
            }
            if (in_path.empty() || out_path.empty()) throw std::runtime_error("score needs --in and --out");
            if (!window && batch_tokens == 0) throw std::runtime_error("score --no-window needs --batch-tokens");
            return score_cmd(argv[2], in_path, out_path, kv_cache, evict, kv_step, logits_path, batch_tokens, window);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mtstudio: %s\n", e.what());
        return 1;
    }
    std::fprintf(
        stderr,
        "usage: mtstudio run|plan spec.json\n"
        "       mtstudio sample spec.json [--prompt P] [--tokens N] [--temp T] "
        "[--topk K] [--seed S] [--out FILE]\n"
        "       mtstudio sample-set spec.json --prompts P.jsonl --out R.jsonl [--n 8] "
        "[--temp 1.0] [--seed 17] [--max-new 16]\n"
        "       mtstudio probes spec.json [--out NEW.json]\n"
        "       mtstudio export spec.json [--from model.safetensors] "
        "[--formats gguf,gguf_q8_0,gguf_q8_0_out,gguf_q8_0_all,gguf_q4_0,gguf_q4_k_m,gguf_q4_k_s]\n"
        "         [--out-dir DIR]\n"
        "       mtstudio serve <out_dir> [port] [spec]   (MTSTUDIO_UI "
        "overrides the index.html path)\n"
        "       mtstudio chat <out_dir|spec.json> [--port 8080] [--host 127.0.0.1] [--kv-cache "
        "[--evict]]\n"
        "       mtstudio score <out_dir|spec.json> --in REQ.jsonl --out RES.jsonl [--kv-cache "
        "[--evict] "
        "[--kv-step N]] [--logits ROWS.f32] [--batch-tokens N [--no-window]]\n"
        "         (per-token log-probs of {prompt | messages, continuation})\n"
        "       mtstudio sft-dump spec.json --out ROWS.jsonl   (the SFT rows the spec trains "
        "on)\n");
    return 2;
}
