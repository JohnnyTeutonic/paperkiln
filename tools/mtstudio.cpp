// mtstudio — the run-spec driver (STUDIO_PLAN.md M1).
//
//   mtstudio run spec.json          execute the spec
//   mtstudio plan spec.json         print the resolved plan and exit
//
// One JSON describes the lifecycle; this driver executes it stage by
// stage and emits a JSONL event stream (stdout + <out>/events.jsonl) that
// the M2 UI will consume. v0 scope: arch presets + custom dims, corpus +
// GGUF vocab, train with early stopping + checkpoint/resume, safetensors
// export, GGUF export for llama-family models, serve-command print.
// (arXiv arch population and the finetune stage are the documented next
// increments; papers/fetch.py already emits the config this schema takes.)
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <vector>

// cpp-httplib (third_party/httplib, MIT): the portable HTTP layer behind
// `serve` and `chat`. Included before anything that might pull in
// <windows.h> so its NOMINMAX/winsock ordering wins.
#include <nlohmann/json.hpp>
#include <regex>
#include "httplib.h"

#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/gguf.hpp"
#include "microtorch/llama.hpp"
#include "microtorch/safetensors.hpp"
#include "microtorch/word_tokenizer.hpp"
#include "parity_model.hpp"

using namespace microtorch;
using nlohmann::json;

namespace {

struct Spec {
    std::string name = "run";
    // arch
    std::string family = "gpt2";      // gpt2 | llama | flex
    std::string attention = "exact";  // exact | kimi | srd (gpt2 family)
    size_t d = 128, layers = 2, heads = 4, T = 128;
    // Paper-faithful flavor knobs (the flex family; empty = family default).
    std::string norm, activation, position;
    std::string residual;           // "" = family default | residual | highway | plain
    float gate_bias_init = -2.0f;   // highway only (registry #0001)
    size_t d_ff = 0;                // 0 = family default (4d gpt2/flex, 3d llama)
    size_t window = 64, sinks = 1;  // swa lane only (S1 baseline)
    // llama family: RoPE on "all" heads or, legacy, the "first" head only.
    // Empty = not stated: a new run gets "all"; resuming or loading a run
    // takes what the run recorded, and a run that recorded nothing is
    // "first" (see resolve_rope_heads).
    std::string rope_heads;
    // data
    std::string corpus, vocab_gguf;
    size_t vocab_cap = 4096;
    size_t max_tokens = 400000;  // corpus read cap (0 = whole file)
    // train
    int steps = 500;
    float lr = 3e-3f, clip = 1.0f, lambda_gate = 0.05f;
    int eval_every = 50, ckpt_every = 100;
    int batch = 1;                    // sequences per FORWARD (stacked rows, one graph)
    int accum = 1;                    // batches accumulated per optimizer step
    bool ckpt_act = false;            // activation checkpointing per block
    unsigned seed = 7;                // model init + data-order seed (Atlas multi-seed)
    std::string optimizer = "adamw";  // adamw | muon (hybrid: hidden matrices
                                      // to Muon, embeddings/vectors to AdamW)
    float muon_lr = 0.02f;
    int gradmap_every = 5;   // per-layer grad-norm event cadence
    size_t es_patience = 0;  // early stopping (0 = off)
    float es_min_delta = 0.0f;
    // export/serve
    bool exp_safetensors = true, exp_gguf = false;
    bool serve = false;
    std::string out_dir = "mtstudio_out";
};

// Known presets; "custom" reads arch.custom.* instead.
const std::map<std::string, std::array<size_t, 4>> PRESETS = {
    // name -> {d, layers, heads, T}
    {"gpt2-nano", {128, 2, 4, 128}},  {"llama-tiny", {128, 2, 4, 128}},
    {"gpt2-small", {256, 4, 8, 256}}, {"kimi-tiny", {128, 2, 4, 128}},
    {"srd-tiny", {128, 2, 4, 128}},   {"attnres-tiny", {128, 2, 4, 128}},
    {"swa-tiny", {128, 2, 4, 128}},
};

Spec parse_spec(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open spec " + path);
    json j = json::parse(f, nullptr, true, /*ignore_comments=*/true);
    Spec s;
    s.name = j.value("name", s.name);

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
        s.norm = c.value("norm", s.norm);
        s.activation = c.value("activation", s.activation);
        s.position = c.value("position", s.position);
        s.residual = c.value("residual", s.residual);
        s.gate_bias_init = c.value("gate_bias_init", s.gate_bias_init);
        s.d_ff = c.value("d_ff", s.d_ff);
        s.window = c.value("window", s.window);
        s.sinks = c.value("sinks", s.sinks);
    }
    // Family resolution for the flavor knobs. RoPE lives inside the llama
    // block (rmsnorm/swiglu come with it); every other flavor combination
    // is the flex family — the paper-faithful decoder.
    if (s.position == "rope") {
        if ((!s.norm.empty() && s.norm != "rmsnorm") ||
            (!s.activation.empty() && s.activation != "swiglu") ||
            (!s.residual.empty() && s.residual != "residual"))
            throw std::runtime_error(
                "position=rope currently implies the llama block "
                "(rmsnorm + swiglu, plain residual stream); drop the "
                "conflicting norm/activation/residual or pick "
                "position=learned|sinusoidal for the flex family");
        s.family = "llama";
    } else if (!s.norm.empty() || !s.activation.empty() || !s.position.empty() ||
               !s.residual.empty()) {
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
        else if (s.attention == "kimi" || s.attention == "srd")
            throw std::runtime_error(
                "kimi/srd presets are 2-block parity models (layers must be 2)");
    }

    const json data = j.value("data", json::object());
    s.corpus = data.value("corpus", "");
    s.vocab_gguf = data.value("vocab", "");
    s.vocab_cap = data.value("vocab_cap", s.vocab_cap);
    s.max_tokens = data.value("max_tokens", s.max_tokens);
    s.T = data.value("T", s.T);

    const json tr = j.value("train", json::object());
    s.steps = tr.value("steps", s.steps);
    s.lr = tr.value("lr", s.lr);
    s.clip = tr.value("clip", s.clip);
    s.eval_every = tr.value("eval_every", s.eval_every);
    s.ckpt_every = tr.value("checkpoint_every", s.ckpt_every);
    s.batch = tr.value("batch", s.batch);
    s.accum = tr.value("accum", s.accum);
    s.ckpt_act = tr.value("checkpoint_activations", s.ckpt_act);
    s.seed = tr.value("seed", s.seed);
    s.optimizer = tr.value("optimizer", s.optimizer);
    s.muon_lr = tr.value("muon_lr", s.muon_lr);
    s.gradmap_every = tr.value("gradmap_every", s.gradmap_every);
    if (tr.contains("early_stopping")) {
        s.es_patience = tr["early_stopping"].value("patience", size_t(0));
        s.es_min_delta = tr["early_stopping"].value("min_delta", 0.0f);
    }

    const json ex = j.value("export", json::object());
    for (const auto& fmt : ex.value("formats", std::vector<std::string>{"safetensors"})) {
        if (fmt == "gguf") s.exp_gguf = true;
        if (fmt == "safetensors") s.exp_safetensors = true;
    }
    s.serve = j.value("serve", json::object()).value("on_finish", false);
    s.out_dir = j.value("out_dir", s.out_dir);
    return s;
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
std::vector<int> tokenize(const std::string& text, const std::map<std::string, int>& vocab,
                          size_t max_tokens);

parity::AttnKind attn_kind(const std::string& s) {
    if (s == "kimi") return parity::AttnKind::KIMI;
    if (s == "srd") return parity::AttnKind::SRD;
    if (s == "swa") return parity::AttnKind::SWA;
    return parity::AttnKind::EXACT;
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

int run(const Spec& s, bool plan_only, const std::string& spec_path = "") {
    std::printf("== mtstudio: %s ==\n", s.name.c_str());
    std::printf("arch: %s d=%zu layers=%zu heads=%zu | T=%zu vocab_cap=%zu\n", s.attention.c_str(),
                s.d, s.layers, s.heads, s.T, s.vocab_cap);
    std::printf(
        "train: %d steps batch=%d accum=%d lr=%g clip=%g eval_every=%d "
        "ckpt_every=%d early_stop(patience=%zu, min_delta=%g)\n",
        s.steps, s.batch, s.accum, s.lr, s.clip, s.eval_every, s.ckpt_every, s.es_patience,
        s.es_min_delta);
    std::printf("export: %s%s | serve: %s | out: %s\n", s.exp_safetensors ? "safetensors " : "",
                s.exp_gguf ? "gguf" : "", s.serve ? "yes" : "no", s.out_dir.c_str());
    if (s.family == "llama")
        std::printf("rope_heads: %s\n", s.rope_heads.empty()
                                            ? "unstated (all for a new run; a resumed run keeps "
                                              "its recorded value, first if none)"
                                            : s.rope_heads.c_str());
    if (plan_only) return 0;
    if (s.corpus.empty() || s.vocab_gguf.empty())
        throw std::runtime_error("spec needs data.corpus and data.vocab");
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
    if (s.vocab_cap > 0 && s.vocab_cap < tokens.size()) tokens.resize(s.vocab_cap);
    std::map<std::string, int> vocab;
    for (size_t i = 0; i < tokens.size(); ++i) vocab.emplace(tokens[i], static_cast<int>(i));
    std::ifstream cf(s.corpus);
    if (!cf) throw std::runtime_error("cannot open corpus " + s.corpus);
    std::string text((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
    auto ids = tokenize(text, vocab, s.max_tokens ? s.max_tokens : text.size() + 1);
    // Hold out the tail 5% for validation (early stopping's signal).
    const size_t val_start = ids.size() - ids.size() / 20;
    ev.emit({{"event", "data"},
             {"tokens", ids.size()},
             {"vocab", tokens.size()},
             {"val_tokens", ids.size() - val_start}});

    // Model + optimizer (+ resume). Two families behind one seam: the
    // gpt2 parity model (exact/kimi/srd attention) or nn::Llama (RMSNorm/
    // RoPE/SwiGLU, HF names -> GGUF-exportable).
    std::shared_ptr<parity::ParityLM> gpt;
    std::shared_ptr<nn::Llama> llama;
    std::shared_ptr<parity::AttnResLM> attnres;
    std::shared_ptr<parity::FlexLM> flex;
    if (s.family == "flex") {
        // The paper-faithful decoder: every extracted flavor is
        // constructor-real (norm/activation/position/d_ff/depth).
        parity::FlexConfig fc;
        fc.vocab = tokens.size();
        fc.d = s.d;
        fc.n_layers = s.layers;
        fc.n_heads = s.heads;
        fc.d_ff = s.d_ff ? s.d_ff : 4 * s.d;
        fc.n_ctx = s.T;
        if (!s.norm.empty()) fc.norm = s.norm;
        if (!s.activation.empty()) fc.act = s.activation;
        if (!s.position.empty()) fc.pos = s.position;
        if (!s.residual.empty()) fc.residual = s.residual;
        fc.gate_bias_init = s.gate_bias_init;
        fc.attention = s.attention;
        fc.window = s.window;
        fc.sinks = s.sinks;
        if (fc.norm != "layernorm" && fc.norm != "rmsnorm")
            throw std::runtime_error("unknown norm " + fc.norm);
        if (fc.act != "gelu" && fc.act != "relu" && fc.act != "swiglu")
            throw std::runtime_error("unknown activation " + fc.act);
        if (fc.pos != "learned" && fc.pos != "sinusoidal")
            throw std::runtime_error("unknown position " + fc.pos + " (rope = llama family)");
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
        llama = std::make_shared<nn::Llama>(lc, s.seed);
        llama->checkpoint_blocks = s.ckpt_act;
    } else {
        gpt = std::make_shared<parity::ParityLM>(attn_kind(s.attention), tokens.size(), s.d,
                                                 s.heads, s.T, s.seed, s.window, s.sinks);
        if (s.ckpt_act) {
            throw std::runtime_error(
                "train.checkpoint_activations requires the llama family for now");
        }
    }
    // Atlas stage-0 structural echo: the run's identity as a data point.
    nn::Module& model_pick =
        flex      ? static_cast<nn::Module&>(*flex)
        : attnres ? static_cast<nn::Module&>(*attnres)
                  : (llama ? static_cast<nn::Module&>(*llama) : static_cast<nn::Module&>(*gpt));
    const size_t n_params = model_pick.parameter_count();
    // Resolved flavor labels — what the constructed model ACTUALLY is,
    // family defaults filled in (the Atlas structural echo must never
    // under-describe the data point).
    const std::string r_norm = flex ? flex->cfg.norm : (llama ? "rmsnorm" : "layernorm");
    const std::string r_act = flex ? flex->cfg.act : (llama ? "swiglu" : "gelu");
    const std::string r_pos = flex ? flex->cfg.pos : (llama ? "rope" : "learned");
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
    if (llama) model_ev["rope_heads"] = rope_heads;
    ev.emit(model_ev);
    nn::Module& model_ref = model_pick;
    auto fwd = [&](const std::vector<int>& ids, size_t seq_len = 0) {
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
    } opt;
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

    // Train.
    bool stopped_early = false;
    const bool is_srd = attn_kind(s.attention) == parity::AttnKind::SRD;
    for (int step = start_step + 1; step <= s.steps; ++step) {
        last_step = step;
        const size_t lim = val_start - s.T - 1;
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
        float task_mean = 0, gate_mean = 0;
        for (int k = 0; k < s.accum; ++k) {
            std::vector<int> x, y;
            x.reserve(s.batch * s.T);
            y.reserve(s.batch * s.T);
            for (int b = 0; b < s.batch; ++b) {
                const size_t at = rng() % lim;
                x.insert(x.end(), ids.begin() + at, ids.begin() + at + s.T);
                y.insert(y.end(), ids.begin() + at + 1, ids.begin() + at + s.T + 1);
            }
            Var logits = fwd(x, s.batch > 1 ? s.T : 0);
            Var task = ops::cross_entropy(logits, y);
            Var loss = task;
            if (is_srd) loss = ops::add(task, ops::scale(gpt->mean_gate(), s.lambda_gate));
            backward(ops::scale(loss, 1.0f / static_cast<float>(s.accum)));
            task_mean += task->data(0, 0) / static_cast<float>(s.accum);
            if (is_srd) gate_mean += gpt->mean_gate()->data(0, 0) / static_cast<float>(s.accum);
        }
        device::step_end();
        // Per-module grad norms BEFORE clipping: this is the true signal
        // the glow UI wants (clipping would mask explosions).
        json gm;
        if (s.gradmap_every > 0 && step % s.gradmap_every == 0) gm = grad_map(model_ref);
        const float total_norm = ops::clip_grad_norm(model_ref.parameters(), s.clip);
        opt.step();

        json e = {
            {"event", "step"}, {"step", step}, {"loss", task_mean}, {"grad_norm", total_norm}};
        if (is_srd) e["gate"] = gate_mean;
        if (!gm.is_null()) e["grads"] = gm;
        ev.emit(e);

        if (step % s.eval_every == 0) {
            NoGrad ng;
            model_ref.eval();
            double vl = 0;
            const int NV = 8;
            std::mt19937 vrng(999);
            for (int k = 0; k < NV; ++k) {
                const size_t va = val_start + vrng() % (ids.size() - val_start - s.T - 1);
                std::vector<int> vx(ids.begin() + va, ids.begin() + va + s.T);
                std::vector<int> vy(ids.begin() + va + 1, ids.begin() + va + s.T + 1);
                vl += ops::cross_entropy(fwd(vx), vy)->data(0, 0);
            }
            vl /= NV;
            model_ref.train();
            ev.emit({{"event", "eval"}, {"step", step}, {"val_loss", vl}});
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
    if (s.exp_gguf) {
        if (llama) {
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
            gc.block_count = (uint32_t)s.layers;
            gc.head_count = (uint32_t)s.heads;
            gc.feed_forward_length = (uint32_t)(3 * s.d);
            gc.vocab_size = (uint32_t)tokens.size();
            gc.context_length = (uint32_t)s.T;
            gc.rms_eps = 1e-6f;
            gc.weights_in_out = true;  // microtorch Linear is [in, out]
            gc.tokens = tokens;
            // End-of-text in the vocabulary -> the GGUF's eos id, so an
            // external engine stops where this one does (else keep the
            // exporter's default).
            if (const int e = wordtok::eos_id(vocab); e >= 0) gc.eos_token_id = (uint32_t)e;
            const std::string gpath = s.out_dir + "/" + s.name + ".gguf";
            gguf::export_gguf_llama(gpath, sd2, gc);
            json xe = {{"event", "export"},
                       {"format", "gguf"},
                       {"path", gpath},
                       {"rope_heads", rope_heads}};
            // GGUF llama (rope.dimension_count = d/H) means RoPE on every
            // head; engines reading it reproduce this model only under "all".
            if (rope_heads == "first" && s.heads > 1)
                xe["warning"] =
                    "rope_heads=first: GGUF engines rotate every head, so they will not "
                    "reproduce this model's outputs";
            ev.emit(xe);
        } else {
            ev.emit({{"event", "export_skipped"},
                     {"format", "gguf"},
                     {"reason", "gpt2-family blocks are not llama-shaped"}});
        }
    }
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
        std::ofstream rf(s.out_dir + "/result.json");
        rf << result.dump(2) << "\n";
    }

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
    for (uint64_t i = 0; i < n_meta; ++i) {
        const std::string key = rd_str(b, p);
        const uint32_t vt = rd<uint32_t>(b, p);
        switch (vt) {
            case 4:
                rd<uint32_t>(b, p);
                break;
            case 5:
                rd<int32_t>(b, p);
                break;
            case 6:
                rd<float>(b, p);
                break;
            case 8:
                rd_str(b, p);
                break;
            case 9: {
                const uint32_t et = rd<uint32_t>(b, p);
                const uint64_t n = rd<uint64_t>(b, p);
                for (uint64_t k = 0; k < n; ++k) {
                    if (et == 8) {
                        std::string t = rd_str(b, p);
                        if (key == "tokenizer.ggml.tokens") tokens.push_back(std::move(t));
                    } else if (et == 6)
                        rd<float>(b, p);
                    else
                        throw std::runtime_error("bad array");
                }
                break;
            }
            default:
                throw std::runtime_error("bad meta");
        }
    }
    return tokens;
}
// The tokenizer itself lives in microtorch/word_tokenizer.hpp (shared
// with its tests); this keeps the srd_parity-era call shape.
std::vector<int> tokenize(const std::string& text, const std::map<std::string, int>& vocab,
                          size_t max_tokens) {
    return wordtok::tokenize(text, vocab, max_tokens);
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
    std::map<std::string, int> vocab;
    int eos = -1;  // end-of-text id, -1 when the vocabulary has none
    std::shared_ptr<parity::ParityLM> gpt;
    std::shared_ptr<nn::Llama> llama;
    std::shared_ptr<parity::AttnResLM> attnres;
    std::shared_ptr<parity::FlexLM> flex;
    std::string rope_heads;  // llama family: what the run was trained with

    explicit LoadedLM(const Spec& spec) : s(spec) {
        // A run directory moved away from its vocab file still carries the
        // vocabulary inside its exported GGUF (the same capped list).
        std::string vpath = s.vocab_gguf;
        const std::string exported = s.out_dir + "/" + s.name + ".gguf";
        if (!std::ifstream(vpath).good() && std::ifstream(exported).good()) vpath = exported;
        tokens = read_gguf_vocab(vpath);
        if (s.vocab_cap > 0 && s.vocab_cap < tokens.size()) tokens.resize(s.vocab_cap);
        for (size_t i = 0; i < tokens.size(); ++i) vocab.emplace(tokens[i], static_cast<int>(i));
        eos = wordtok::eos_id(vocab);

        if (s.family == "flex") {
            parity::FlexConfig fc;
            fc.vocab = tokens.size();
            fc.d = s.d;
            fc.n_layers = s.layers;
            fc.n_heads = s.heads;
            fc.d_ff = s.d_ff ? s.d_ff : 4 * s.d;
            fc.n_ctx = s.T;
            if (!s.norm.empty()) fc.norm = s.norm;
            if (!s.activation.empty()) fc.act = s.activation;
            if (!s.position.empty()) fc.pos = s.position;
            if (!s.residual.empty()) fc.residual = s.residual;
            fc.gate_bias_init = s.gate_bias_init;
            fc.attention = s.attention;
            fc.window = s.window;
            fc.sinks = s.sinks;
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
            rope_heads = resolve_rope_heads(s, /*existing=*/true);
            lc.rope_all_heads = rope_heads == "all";
            llama = std::make_shared<nn::Llama>(lc, s.seed);
        } else {
            gpt = std::make_shared<parity::ParityLM>(attn_kind(s.attention), tokens.size(), s.d,
                                                     s.heads, s.T, s.seed, s.window, s.sinks);
        }
        const std::string ckpt = s.out_dir + "/" + s.name + ".safetensors";
        model().load_state_dict(load_safetensors(ckpt), /*strict=*/true);
        model().eval();
    }
    nn::Module& model() {
        return flex ? static_cast<nn::Module&>(*flex)
               : attnres
                   ? static_cast<nn::Module&>(*attnres)
                   : (llama ? static_cast<nn::Module&>(*llama) : static_cast<nn::Module&>(*gpt));
    }
    Var forward(const std::vector<int>& ids) {
        if (flex) return flex->forward(ids);
        if (attnres) return attnres->forward(ids);
        return llama ? llama->forward(ids) : gpt->forward(ids);
    }
    // One sampled next token over the last T ids: temperature, top-k, then
    // nucleus (top_p < 1 keeps the smallest head of the top-k holding that
    // share of its mass). top_p >= 1 is the original quick-look sampler,
    // draw for draw. Call under NoGrad.
    int next(const std::vector<int>& ids, float temp, int topk, float top_p, std::mt19937& gen) {
        std::vector<int> ctx = ids;
        if (ctx.size() > s.T) ctx.assign(ids.end() - s.T, ids.end());
        Var logits = forward(ctx);
        const size_t last = logits->data.rows() - 1, V = logits->data.cols();
        std::vector<std::pair<float, int>> scored(V);
        for (size_t j = 0; j < V; ++j)
            scored[j] = {logits->data(last, j) / std::max(temp, 1e-4f), static_cast<int>(j)};
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

// The quick-look sampler (ECOSYSTEM.md feature 2): rebuild the spec's
// model, load its exported safetensors, and generate word-level text
// with temperature + top-k, stopping early at end-of-text. ember.cpp
// remains the real server; this closes the train→poke loop without
// leaving the studio.
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
        text += " " + lm.tokens[pick];
    }
    std::printf("%s\n", text.c_str());
    if (!out_file.empty()) {
        std::ofstream f(out_file, std::ios::trunc);
        f << text << "\n";
    }
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
    // "mtstudio run <spec>" with output logged into out_dir. One run at
    // a time; state is reaped non-blockingly per request. POST /fetch
    // launches papers/fetch.py the same way (the drag-an-arXiv-id flow).
    // Handlers run on httplib's worker threads: `mu` guards this state.
    std::mutex mu, sample_mu;
    Child run_child, fetch_child;
    bool run_finished = false, fetch_failed = false;
    svr.set_pre_routing_handler([&](const httplib::Request&, httplib::Response&) {
        std::lock_guard<std::mutex> lk(mu);
        if (reap(run_child, nullptr)) run_finished = true;
        bool ok = true;
        if (reap(fetch_child, &ok)) fetch_failed = !ok;
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.Post("/train", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(mu);
        if (spec_path.empty()) {
            text_reply(res, 409, "no spec armed (serve <dir> <port> <spec>)");
        } else if (run_child.running()) {
            text_reply(res, 200, "training…");
        } else if (run_finished) {
            text_reply(res, 200, "run complete");
        } else {
            const bool ok = spawn(run_child, {self, "run", spec_path}, out_dir + "/run.log");
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
    svr.Post("/sample", [&](const httplib::Request& req, httplib::Response& res) {
        // In-page quick-look generation: body = the prompt. Runs the
        // `mtstudio sample` path in-process (a tiny model on CPU answers
        // in seconds), one request at a time; ember.cpp and `mtstudio
        // chat` are the real servers.
        if (spec_path.empty()) {
            text_reply(res, 409, "no spec armed (serve <dir> <port> <spec>)");
            return;
        }
        std::lock_guard<std::mutex> lk(sample_mu);
        const std::string sf = out_dir + "/sample.txt";
        std::remove(sf.c_str());
        std::string log;
        try {
            sample_cmd(parse_spec(spec_path), req.body.empty() ? "once upon a time" : req.body, 40,
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

std::string chat_prompt(const json& req) {
    std::string p;
    if (req.contains("history") && req["history"].is_array()) {
        // [{role: user|assistant, content}] or [{user, assistant}] turns.
        for (const auto& turn : req["history"]) {
            if (!turn.is_object()) continue;
            if (turn.contains("role")) {
                const std::string role = turn.value("role", "");
                if (role == "user" || role == "assistant")
                    p += role + ": " + turn.value("content", "") + " ";
                continue;
            }
            if (turn.contains("user")) p += "user: " + turn.value("user", "") + " ";
            if (turn.contains("assistant")) p += "assistant: " + turn.value("assistant", "") + " ";
        }
    }
    return p + "user: " + req["user_input"].get<std::string>() + " assistant:";
}

int chat_cmd(const std::string& target, const std::string& host, int port) {
    const Spec s = resolve_chat_spec(target);
    LoadedLM lm(s);
    // The model's own token sequences for a new turn ("user :",
    // "assistant :"), unusable if a piece is out of vocabulary (it would
    // be <unk>, which the sampler never emits anyway).
    std::vector<std::vector<int>> turn_markers;
    for (const char* m : {"user:", "assistant:"}) {
        auto seq = tokenize(m, lm.vocab, 16);
        if (std::find(seq.begin(), seq.end(), 0) == seq.end()) turn_markers.push_back(seq);
    }
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
        if (r.is_discarded() || !r.is_object() || !r.contains("user_input") ||
            !r["user_input"].is_string()) {
            fail(400, "body must be JSON with a string 'user_input'");
            return;
        }
        try {
            // Defaults are tinyllama_server's.
            const int max_new = std::min(std::max(r.value("max_new_tokens", 60), 1), 1024);
            const float temp = r.value("temperature", 0.1f);
            const int top_k = r.value("top_k", 40);
            const float top_p = r.value("top_p", 0.9f);
            const std::string prompt = chat_prompt(r);
            std::lock_guard<std::mutex> lk(gen_mu);
            std::mt19937 gen(r.contains("seed") ? r["seed"].get<unsigned>() : seeder());
            NoGrad ng;
            std::vector<int> ids = tokenize(prompt, lm.vocab, prompt.size() + 1), out;
            std::string stop = "length";
            for (int t = 0; t < max_new; ++t) {
                const int pick = lm.next(ids, temp, top_k, top_p, gen);
                if (pick == lm.eos) {
                    stop = "eos";
                    break;
                }
                ids.push_back(pick);
                out.push_back(pick);
                for (const auto& m : turn_markers)
                    if (wordtok::ends_with(out, m)) {
                        out.resize(out.size() - m.size());
                        stop = "turn";
                        break;
                    }
                if (stop == "turn") break;
            }
            std::string reply = wordtok::detokenize(out, lm.tokens);
            const auto b = reply.find_first_not_of(" \t\r\n");
            const auto e = reply.find_last_not_of(" \t\r\n");
            reply = b == std::string::npos ? "" : reply.substr(b, e - b + 1);
            res.set_content(
                json({{"reply", reply}, {"stop_reason", stop}, {"tokens", out.size()}}).dump(),
                "application/json");
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
}  // namespace

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "";
    try {
        // Honour MICROTORCH_DEVICE / MICROTORCH_STEP_RESIDENCY like the
        // test suites do (no-ops on CPU builds). Rung C's CUDA runs go
        // through here (docs/CUDA_PHASE_B2.md).
        device::set_from_env();
        if ((cmd == "run" || cmd == "plan") && argc >= 3)
            return run(parse_spec(argv[2]), cmd == "plan", argv[2]);
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
        if (cmd == "serve" && argc >= 3) {
            const int port = argc > 3 ? std::atoi(argv[3]) : 8123;
            const char* ui = std::getenv("MTSTUDIO_UI");
            // Optional 4th arg: a spec the browser can launch via the
            // in-page Train button (POST /train).
            const std::string spec = argc > 4 ? argv[4] : "";
            return serve_ui(argv[2], port, ui ? ui : "studio/index.html", spec, self_exe(argv[0]));
        }
        if (cmd == "chat" && argc >= 3) {
            std::string host = "127.0.0.1";
            int port = 8080;
            for (int i = 3; i + 1 < argc; i += 2) {
                const std::string k = argv[i], v = argv[i + 1];
                if (k == "--port") port = std::atoi(v.c_str());
                if (k == "--host") host = v;
            }
            return chat_cmd(argv[2], host, port);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mtstudio: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr,
                 "usage: mtstudio run|plan spec.json\n"
                 "       mtstudio sample spec.json [--prompt P] [--tokens N] [--temp T] "
                 "[--topk K] [--seed S] [--out FILE]\n"
                 "       mtstudio serve <out_dir> [port] [spec]   (MTSTUDIO_UI "
                 "overrides the index.html path)\n"
                 "       mtstudio chat <out_dir|spec.json> [--port 8080] [--host 127.0.0.1]\n");
    return 2;
}
