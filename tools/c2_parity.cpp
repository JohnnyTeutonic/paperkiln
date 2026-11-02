// C2 (docs/ECOSYSTEM.md): paperkiln's leg of the cross-engine golden pin. Loads an HF
// LlamaForCausalLM directory (config.json + model.safetensors, as coalfire's --export-hf writes it)
// into microtorch::Llama, runs one forward over the ids in IDS (whitespace-separated) and writes
// every row's logits as float32 [T, vocab] to OUT.
//
//   c2_parity HF_DIR IDS OUT
//
// HF stores Linear weights [out, in] and paperkiln [in, out]: every *_proj.weight and
// lm_head.weight is transposed; norm gammas [d] load as [1, d]; the "model." prefix is dropped
// (paperkiln's names are the paths below it). HF's rotate-half RoPE is paperkiln's rope_pairing
// "neox". The load is strict: every tensor must land on a parameter and every parameter must be
// loaded. Prints one JSON line.
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "microtorch/autograd.hpp"
#include "microtorch/llama.hpp"
#include "microtorch/safetensors.hpp"

using json = nlohmann::json;
using namespace microtorch;
using nn::Llama;
using nn::LlamaConfig;

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

Matrix transposed(const Matrix& m) {
    Matrix t(m.cols(), m.rows());
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j) t(j, i) = m(i, j);
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::fprintf(stderr, "usage: c2_parity HF_DIR IDS OUT\n");
            return 2;
        }
        const std::string dir = argv[1];
        json c = json::parse(std::ifstream(dir + "/config.json"));
        if (c.value("model_type", "") != "llama")
            throw std::runtime_error("config.json: model_type is not llama");
        LlamaConfig cfg;
        cfg.vocab = c.at("vocab_size");
        cfg.d = c.at("hidden_size");
        cfg.n_layers = c.at("num_hidden_layers");
        cfg.n_heads = c.at("num_attention_heads");
        cfg.n_kv_heads = c.value("num_key_value_heads", cfg.n_heads);
        if (cfg.n_kv_heads == cfg.n_heads) cfg.n_kv_heads = 0;
        cfg.d_ff = c.at("intermediate_size");
        cfg.n_ctx = c.at("max_position_embeddings");
        cfg.rms_eps = c.at("rms_norm_eps");
        cfg.rope_theta = c.value("rope_theta", 10000.0);
        cfg.tie_embeddings = c.value("tie_word_embeddings", false);
        cfg.rope_all_heads = true;
        cfg.rope_pairing = "neox";
        if (c.value("hidden_act", "silu") != "silu")
            throw std::runtime_error("config.json: hidden_act is not silu");

        std::map<std::string, Matrix> sd;
        for (auto& [hf_name, m] : load_safetensors(dir + "/model.safetensors")) {
            // paperkiln's state dict is LlamaForCausalLM's paths below "model."
            // (tests/test_llama.cpp)
            const std::string name = hf_name.rfind("model.", 0) == 0 ? hf_name.substr(6) : hf_name;
            if (ends_with(name, "_proj.weight") || name == "lm_head.weight")
                sd[name] = transposed(m);
            else if (ends_with(name, "norm.weight") && m.cols() == 1)
                sd[name] = transposed(m);
            else
                sd[name] = m;
        }
        Llama model(cfg, 0);
        model.load_state_dict(sd, /*strict=*/true);

        std::vector<int> ids;
        std::ifstream in(argv[2]);
        for (int t; in >> t;) ids.push_back(t);
        if (ids.empty() || ids.size() > cfg.n_ctx)
            throw std::runtime_error("ids: empty or longer than n_ctx");

        NoGrad ng;
        Var logits = model.forward(ids);
        const Matrix& L = logits->data;
        if (L.rows() != ids.size() || L.cols() != cfg.vocab)
            throw std::runtime_error("logits shape");
        std::vector<float> rows;
        rows.reserve(L.rows() * L.cols());
        for (size_t i = 0; i < L.rows(); ++i)
            for (size_t j = 0; j < L.cols(); ++j) rows.push_back(L(i, j));
        std::FILE* out = std::fopen(argv[3], "wb");
        if (!out) throw std::runtime_error("cannot open logits output");
        const size_t written = std::fwrite(rows.data(), sizeof(float), rows.size(), out);
        std::fclose(out);
        if (written != rows.size()) throw std::runtime_error("cannot write logits");
        std::printf(
            "{\"engine\": \"paperkiln\", \"tensors\": %zu, \"rows\": %zu, \"vocab\": %zu, \"d\": "
            "%zu, "
            "\"layers\": %zu, \"heads\": %zu, \"rope_pairing\": \"neox\", \"tied\": %s}\n",
            sd.size(), L.rows(), L.cols(), cfg.d, cfg.n_layers, cfg.n_heads,
            cfg.tie_embeddings ? "true" : "false");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "c2_parity: %s\n", e.what());
        return 1;
    }
}
