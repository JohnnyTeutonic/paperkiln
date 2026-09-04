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
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <regex>

#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/gguf.hpp"
#include "microtorch/llama.hpp"
#include "microtorch/safetensors.hpp"
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
    std::string residual;             // "" = family default | residual | highway | plain
    float gate_bias_init = -2.0f;     // highway only (registry #0001)
    size_t d_ff = 0;                // 0 = family default (4d gpt2/flex, 3d llama)
    size_t window = 64, sinks = 1;  // swa lane only (S1 baseline)
    // data
    std::string corpus, vocab_gguf;
    size_t vocab_cap = 4096;
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

int run(const Spec& s, bool plan_only) {
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
    if (plan_only) return 0;
    if (s.corpus.empty() || s.vocab_gguf.empty())
        throw std::runtime_error("spec needs data.corpus and data.vocab");
    // Only the kimi/srd parity lanes are depth-fixed; flex and llama take
    // any depth, and attnres wires s.layers into its stack.
    if (s.family == "gpt2" && s.attention != "attnres" && s.layers != 2)
        throw std::runtime_error("kimi/srd parity lanes: layers must be 2 "
                                 "(exact/swa at depth ride the flex family)");

    std::system(("mkdir -p " + s.out_dir).c_str());
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
    auto ids = tokenize(text, vocab, 400000);
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
            throw std::runtime_error("flex attention must be exact or swa, got " +
                                     fc.attention);
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
    ev.emit({{"event", "model"},
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
             {"params", n_params}});
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
            const std::string gpath = s.out_dir + "/" + s.name + ".gguf";
            gguf::export_gguf_llama(gpath, sd2, gc);
            ev.emit({{"event", "export"}, {"format", "gguf"}, {"path", gpath}});
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
std::vector<int> tokenize(const std::string& text, const std::map<std::string, int>& vocab,
                          size_t max_tokens) {
    std::vector<int> ids;
    std::string cur;
    auto flush = [&]() {
        if (cur.empty()) return;
        auto it = vocab.find(cur);
        ids.push_back(it == vocab.end() ? 0 : it->second);
        cur.clear();
    };
    for (char ch : text) {
        if (ids.size() >= max_tokens) break;
        const unsigned char c = static_cast<unsigned char>(ch);
        if (std::isalpha(c) || c == '\'' || std::isdigit(c)) {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else {
            flush();
            if (!std::isspace(c)) {
                std::string pch(1, static_cast<char>(c));
                auto it = vocab.find(pch);
                ids.push_back(it == vocab.end() ? 0 : it->second);
            }
        }
    }
    flush();
    return ids;
}
}  // namespace

// ---- M2 live mode: minimal HTTP server (POSIX; runs under WSL/Linux).
// GET /              -> the studio UI (index.html)
// GET /events.jsonl  -> the run dir's current event stream
// The UI polls /events.jsonl every 2s when served over http, turning the
// dashboard into a live training monitor.
#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// The quick-look sampler (ECOSYSTEM.md feature 2): rebuild the spec's
// model, load its exported safetensors, and generate word-level text
// with temperature + top-k. ember.cpp remains the real server; this
// closes the train→poke loop without leaving the studio.
int sample_cmd(const Spec& s, const std::string& prompt, int n_new, float temp, int topk,
               unsigned sseed, const std::string& out_file) {
    auto tokens = read_gguf_vocab(s.vocab_gguf);
    if (s.vocab_cap > 0 && s.vocab_cap < tokens.size()) tokens.resize(s.vocab_cap);
    std::map<std::string, int> vocab;
    for (size_t i = 0; i < tokens.size(); ++i) vocab.emplace(tokens[i], static_cast<int>(i));

    // Same construction switch as run() — the spec is the single source
    // of architecture truth for both paths.
    std::shared_ptr<parity::ParityLM> gpt;
    std::shared_ptr<nn::Llama> llama;
    std::shared_ptr<parity::AttnResLM> attnres;
    std::shared_ptr<parity::FlexLM> flex;
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
        attnres =
            std::make_shared<parity::AttnResLM>(tokens.size(), s.d, s.heads, s.T, s.seed, s.layers);
    } else if (s.family == "llama") {
        nn::LlamaConfig lc;
        lc.vocab = tokens.size();
        lc.d = s.d;
        lc.n_layers = s.layers;
        lc.n_heads = s.heads;
        lc.d_ff = s.d_ff ? s.d_ff : 3 * s.d;
        lc.n_ctx = s.T;
        llama = std::make_shared<nn::Llama>(lc, s.seed);
    } else {
        gpt = std::make_shared<parity::ParityLM>(attn_kind(s.attention), tokens.size(), s.d,
                                                 s.heads, s.T, s.seed, s.window, s.sinks);
    }
    nn::Module& model_ref =
        flex      ? static_cast<nn::Module&>(*flex)
        : attnres ? static_cast<nn::Module&>(*attnres)
                  : (llama ? static_cast<nn::Module&>(*llama) : static_cast<nn::Module&>(*gpt));
    const std::string ckpt = s.out_dir + "/" + s.name + ".safetensors";
    model_ref.load_state_dict(load_safetensors(ckpt), /*strict=*/true);
    model_ref.eval();
    auto fwd = [&](const std::vector<int>& ids) {
        if (flex) return flex->forward(ids);
        if (attnres) return attnres->forward(ids);
        return llama ? llama->forward(ids) : gpt->forward(ids);
    };

    auto ids = tokenize(prompt, vocab, 100000);
    if (ids.empty()) throw std::runtime_error("no prompt word is in the model's vocabulary");
    std::mt19937 gen(sseed);
    std::string text = prompt;
    NoGrad ng;
    for (int t = 0; t < n_new; ++t) {
        std::vector<int> ctx = ids;
        if (ctx.size() > s.T) ctx.assign(ids.end() - s.T, ids.end());
        Var logits = fwd(ctx);
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
        const size_t k = std::min<size_t>(std::max(topk, 1), V);
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        double mx = scored[0].first, z = 0;
        std::vector<double> p(k);
        for (size_t j = 0; j < k; ++j) z += (p[j] = std::exp(scored[j].first - mx));
        std::uniform_real_distribution<double> u(0.0, z);
        double r = u(gen);
        int pick = scored[k - 1].second;
        for (size_t j = 0; j < k; ++j)
            if ((r -= p[j]) <= 0) {
                pick = scored[j].second;
                break;
            }
        ids.push_back(pick);
        text += " " + tokens[pick];
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

int serve_ui(const std::string& out_dir, int port, const std::string& ui_path,
             const std::string& spec_path = "") {
#ifdef _WIN32
    std::fprintf(stderr,
                 "mtstudio serve: POSIX-only for now (run under "
                 "WSL); Windows needs a winsock port.\n");
    (void)out_dir;
    (void)port;
    (void)ui_path;
    return 1;
#else
    const std::string ui = slurp(ui_path);
    if (ui.empty()) throw std::runtime_error("cannot read UI at " + ui_path + " (set MTSTUDIO_UI)");
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket failed");
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
        throw std::runtime_error("bind failed (port in use?)");
    if (::listen(fd, 8) < 0) throw std::runtime_error("listen failed");
    std::printf(
        "mtstudio serve: http://localhost:%d/  (events from %s, "
        "Ctrl-C to stop)\n",
        port, out_dir.c_str());

    auto respond = [](int c, const char* status, const char* ctype, const std::string& body) {
        char head[256];
        const int n = std::snprintf(head, sizeof(head),
                                    "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                                    "Cache-Control: no-store\r\nConnection: close\r\n\r\n",
                                    status, ctype, body.size());
        (void)!::write(c, head, n);
        (void)!::write(c, body.data(), body.size());
    };

    // The in-page Train button: POST /train forks this binary as
    // "mtstudio run <spec>" with output logged into out_dir. One run at
    // a time; state is reaped non-blockingly per request. POST /fetch
    // forks papers/fetch.py the same way (the drag-an-arXiv-id flow).
    pid_t run_pid = -1, fetch_pid = -1;
    bool run_finished = false, fetch_failed = false;
    for (;;) {
        const int c = ::accept(fd, nullptr, nullptr);
        if (c < 0) continue;
        if (run_pid > 0) {
            int st = 0;
            if (::waitpid(run_pid, &st, WNOHANG) == run_pid) {
                run_pid = -1;
                run_finished = true;
            }
        }
        if (fetch_pid > 0) {
            int st = 0;
            if (::waitpid(fetch_pid, &st, WNOHANG) == fetch_pid) {
                fetch_pid = -1;
                fetch_failed = !WIFEXITED(st) || WEXITSTATUS(st) != 0;
            }
        }
        // Read the whole request: loop until the header terminator and
        // any Content-Length worth of body have arrived (browser POST
        // headers alone can exceed a single small read).
        std::string line;
        {
            char buf[4096];
            size_t hdr_end = std::string::npos, want = 0;
            ssize_t r;
            while ((r = ::read(c, buf, sizeof(buf))) > 0) {
                line.append(buf, static_cast<size_t>(r));
                if (hdr_end == std::string::npos) {
                    hdr_end = line.find("\r\n\r\n");
                    if (hdr_end != std::string::npos) {
                        auto cl = line.find("Content-Length:");
                        if (cl == std::string::npos) cl = line.find("content-length:");
                        if (cl != std::string::npos)
                            want = std::strtoul(line.c_str() + cl + 15, nullptr, 10);
                    }
                }
                if (hdr_end != std::string::npos && line.size() >= hdr_end + 4 + want) break;
                if (line.size() > 65536) break;
            }
        }
        if (line.rfind("POST /train", 0) == 0) {
            if (spec_path.empty()) {
                respond(c, "409 Conflict", "text/plain",
                        "no spec armed (serve <dir> <port> <spec>)");
            } else if (run_pid > 0) {
                respond(c, "200 OK", "text/plain", "training…");
            } else if (run_finished) {
                respond(c, "200 OK", "text/plain", "run complete");
            } else {
                const pid_t pid = ::fork();
                if (pid == 0) {
                    const std::string log = out_dir + "/run.log";
                    (void)!::freopen(log.c_str(), "a", stdout);
                    (void)!::freopen(log.c_str(), "a", stderr);
                    ::execl("/proc/self/exe", "mtstudio", "run", spec_path.c_str(),
                            static_cast<char*>(nullptr));
                    _exit(127);
                }
                run_pid = pid;
                respond(c, "200 OK", "text/plain", pid > 0 ? "training…" : "fork failed");
            }
        } else if (line.rfind("POST /fetch", 0) == 0) {
            // Body = an arXiv id or URL. Forks the paper fetcher, which
            // writes arch.json + paper.html into out_dir; the page polls
            // /fetchstatus, then reads both over this same server.
            const auto he = line.find("\r\n\r\n");
            const std::string id =
                sanitize_arxiv(he == std::string::npos ? "" : line.substr(he + 4));
            if (id.empty()) {
                respond(c, "400 Bad Request", "text/plain",
                        "not an arXiv id (want 1706.03762-style, or an arxiv.org URL)");
            } else if (fetch_pid > 0) {
                respond(c, "200 OK", "text/plain", "fetching…");
            } else {
                ::unlink((out_dir + "/arch.json").c_str());
                ::unlink((out_dir + "/paper.html").c_str());
                fetch_failed = false;
                const char* fp = std::getenv("MTSTUDIO_FETCH");
                const std::string fetcher = fp ? fp : "papers/fetch.py";
                const std::string aj = out_dir + "/arch.json", ph = out_dir + "/paper.html";
                const pid_t pid = ::fork();
                if (pid == 0) {
                    const std::string log = out_dir + "/fetch.log";
                    (void)!::freopen(log.c_str(), "a", stdout);
                    (void)!::freopen(log.c_str(), "a", stderr);
                    ::execlp("python3", "python3", fetcher.c_str(), id.c_str(), "--json",
                             aj.c_str(), "--emit-html", ph.c_str(), static_cast<char*>(nullptr));
                    _exit(127);
                }
                fetch_pid = pid;
                respond(c, "200 OK", "text/plain",
                        pid > 0 ? "fetching " + id + "…" : "fork failed");
            }
        } else if (line.rfind("POST /sample", 0) == 0) {
            // In-page quick-look generation: body = the prompt. Forks
            // "mtstudio sample <spec>" and waits (a tiny model on CPU
            // answers in seconds); ember.cpp stays the real server.
            const auto he = line.find("\r\n\r\n");
            const std::string prompt = he == std::string::npos ? "" : line.substr(he + 4);
            if (spec_path.empty()) {
                respond(c, "409 Conflict", "text/plain",
                        "no spec armed (serve <dir> <port> <spec>)");
            } else {
                const std::string sf = out_dir + "/sample.txt";
                ::unlink(sf.c_str());
                const pid_t pid = ::fork();
                if (pid == 0) {
                    const std::string log = out_dir + "/sample.log";
                    (void)!::freopen(log.c_str(), "w", stdout);
                    (void)!::freopen(log.c_str(), "w", stderr);
                    ::execl("/proc/self/exe", "mtstudio", "sample", spec_path.c_str(), "--prompt",
                            prompt.empty() ? "once upon a time" : prompt.c_str(), "--out",
                            sf.c_str(), static_cast<char*>(nullptr));
                    _exit(127);
                }
                int st = 0;
                ::waitpid(pid, &st, 0);
                const std::string body = slurp(sf);
                if (!body.empty()) {
                    respond(c, "200 OK", "text/plain; charset=utf-8", body);
                } else {
                    respond(c, "500 Internal Server Error", "text/plain",
                            "sampling failed — train (and export safetensors) first; "
                            "details in sample.log");
                }
            }
        } else if (line.rfind("GET /fetchstatus", 0) == 0) {
            const char* s = fetch_pid > 0                            ? "fetching"
                            : fetch_failed                           ? "failed (see fetch.log)"
                            : !slurp(out_dir + "/arch.json").empty() ? "done"
                                                                     : "idle";
            respond(c, "200 OK", "text/plain", s);
        } else if (line.rfind("GET /spec", 0) == 0) {
            const std::string body = spec_path.empty() ? "" : slurp(spec_path);
            if (body.empty()) {
                respond(c, "404 Not Found", "text/plain", "404");
            } else {
                respond(c, "200 OK", "application/json", body);
            }
        } else if (line.rfind("GET /events.jsonl", 0) == 0) {
            respond(c, "200 OK", "application/jsonl", slurp(out_dir + "/events.jsonl"));
        } else if (line.rfind("GET / ", 0) == 0 || line.rfind("GET /index.html", 0) == 0) {
            respond(c, "200 OK", "text/html; charset=utf-8", ui);
        } else if (line.rfind("GET /findings", 0) == 0) {
            // The findings registry, served from the repo beside the UI
            // (studio/../atlas/findings.jsonl) — the viewer renders it
            // as cards with status badges.
            const auto cut = ui_path.find_last_of("/\\");
            const std::string dir = cut == std::string::npos ? "." : ui_path.substr(0, cut);
            const std::string body = slurp(dir + "/../atlas/findings.jsonl");
            if (!body.empty()) {
                respond(c, "200 OK", "application/jsonl", body);
            } else {
                respond(c, "404 Not Found", "text/plain", "404");
            }
        } else if (line.rfind("GET /atlas", 0) == 0) {
            // The designed-experiment viewer, served from beside the UI;
            // in served mode it auto-loads the out_dir's atlas_rows.jsonl.
            const auto cut = ui_path.find_last_of("/\\");
            const std::string dir = cut == std::string::npos ? "." : ui_path.substr(0, cut);
            const std::string body = slurp(dir + "/atlas.html");
            if (!body.empty()) {
                respond(c, "200 OK", "text/html; charset=utf-8", body);
            } else {
                respond(c, "404 Not Found", "text/plain", "404");
            }
        } else if (line.rfind("GET /", 0) == 0) {
            // Serve sibling files from out_dir (the diff-to-paper page
            // and the fetched arch.json ride the same localhost as the
            // dashboard, so no file:// URL gymnastics from Windows/WSL).
            // Name only — no slashes or dots-paths — and a fixed
            // extension whitelist.
            const size_t sp = line.find(' ', 4);
            std::string name = sp == std::string::npos ? "" : line.substr(5, sp - 5);
            const char* ctype = nullptr;
            auto ends = [&](const char* suf) {
                const size_t n = std::strlen(suf);
                return name.size() > n && name.compare(name.size() - n, n, suf) == 0;
            };
            if (ends(".html")) ctype = "text/html; charset=utf-8";
            if (ends(".json")) ctype = "application/json";
            if (ends(".log")) ctype = "text/plain; charset=utf-8";
            // Trained-artifact downloads (the page's export links).
            if (ends(".safetensors") || ends(".gguf")) ctype = "application/octet-stream";
            const bool safe = ctype && name.find('/') == std::string::npos &&
                              name.find("..") == std::string::npos;
            const std::string body = safe ? slurp(out_dir + "/" + name) : "";
            if (!body.empty()) {
                respond(c, "200 OK", ctype, body);
            } else {
                respond(c, "404 Not Found", "text/plain", "404");
            }
        } else {
            respond(c, "404 Not Found", "text/plain", "404");
        }
        ::close(c);
    }
#endif
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
            return run(parse_spec(argv[2]), cmd == "plan");
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
            return serve_ui(argv[2], port, ui ? ui : "studio/index.html", spec);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mtstudio: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr,
                 "usage: mtstudio run|plan spec.json\n"
                 "       mtstudio serve <out_dir> [port]   (MTSTUDIO_UI "
                 "overrides the index.html path)\n");
    return 2;
}
