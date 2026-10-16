// Timing-model exporter for ember.cpp's TTFT work (ember.cpp docs/TTFT.md).
//
//   make_timing_gguf <vocab.gguf> <out.gguf> [d layers heads kv_heads ff ctx seed]
//
// Writes a Llama-family GGUF through export_gguf_llama with SEEDED RANDOM
// weights at a chosen shape, borrowing the word-level vocabulary of an
// existing paperkiln export. Defaults are the TinyLlama-1.1B shape
// (d=2048, 22 layers, 32 heads / 4 KV heads, ff=5632) with a 4096-token
// context, so prompts of 128..4096 tokens fit and prefill is compute-bound.
//
// What this model is for: wall-clock and kernel timing only. The weights
// are N(0, 0.02) noise (norm gains 1.0), so the generated text is
// meaningless; nothing about output quality may be read from it. Prefill
// cost depends on shapes, not on weight values, which is why a random
// model is a fair timing subject. The vocabulary is the source file's
// (4096 words for story3k), so the LM head is smaller than TinyLlama's
// 32000 -- it is computed for one token per prefill and barely registers.
//
// Generate it where it is used (the Colab VM): at F32 the default shape is
// ~4.4 GB, far too large to push over the home uplink.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "microtorch/gguf.hpp"

using namespace microtorch;

namespace {

// Minimal GGUF v3 metadata reader: only the tokenizer block is needed.
// Same spec-based parsing as tools/gguf_roundtrip.cpp, stopped before the
// tensor table.
struct Vocab {
    std::vector<std::string> tokens;
    std::string model = "word";
    std::map<std::string, uint32_t> ids;  // bos/eos/unk/pad token ids
};

template <typename T>
T rd(std::ifstream& f) {
    T v;
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!f) throw std::runtime_error("truncated GGUF header");
    return v;
}

std::string rd_str(std::ifstream& f) {
    const uint64_t n = rd<uint64_t>(f);
    std::string s(n, '\0');
    f.read(&s[0], static_cast<std::streamsize>(n));
    return s;
}

Vocab read_vocab(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    if (rd<uint32_t>(f) != 0x46554747u) throw std::runtime_error("not a GGUF file: " + path);
    if (rd<uint32_t>(f) != 3) throw std::runtime_error("unsupported GGUF version");
    rd<uint64_t>(f);  // tensor count
    const uint64_t n_meta = rd<uint64_t>(f);

    Vocab v;
    for (uint64_t i = 0; i < n_meta; ++i) {
        const std::string key = rd_str(f);
        const uint32_t vt = rd<uint32_t>(f);
        if (vt == 4 || vt == 5) {
            const uint32_t x = rd<uint32_t>(f);
            if (key.rfind("tokenizer.ggml.", 0) == 0) v.ids[key] = x;
        } else if (vt == 6) {
            rd<float>(f);
        } else if (vt == 8) {
            std::string s = rd_str(f);
            if (key == "tokenizer.ggml.model") v.model = s;
        } else if (vt == 9) {
            const uint32_t et = rd<uint32_t>(f);
            const uint64_t n = rd<uint64_t>(f);
            for (uint64_t k = 0; k < n; ++k) {
                if (et == 8) {
                    std::string tok = rd_str(f);
                    if (key == "tokenizer.ggml.tokens") v.tokens.push_back(std::move(tok));
                } else if (et == 6) {
                    rd<float>(f);
                } else {
                    throw std::runtime_error("unexpected array element type in " + key);
                }
            }
        } else {
            throw std::runtime_error("unexpected metadata type " + std::to_string(vt) + " for " + key);
        }
    }
    if (v.tokens.empty()) throw std::runtime_error(path + " has no tokenizer.ggml.tokens");
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 10) {
        std::fprintf(stderr,
                     "usage: make_timing_gguf <vocab.gguf> <out.gguf> "
                     "[d layers heads kv_heads ff ctx seed]\n");
        return 2;
    }
    uint32_t d = 2048, layers = 22, heads = 32, kv_heads = 4, ff = 5632, ctx = 4096, seed = 42;
    if (argc == 10) {
        d = std::stoul(argv[3]);
        layers = std::stoul(argv[4]);
        heads = std::stoul(argv[5]);
        kv_heads = std::stoul(argv[6]);
        ff = std::stoul(argv[7]);
        ctx = std::stoul(argv[8]);
        seed = std::stoul(argv[9]);
    }
    if (d % heads != 0 || heads % kv_heads != 0)
        throw std::runtime_error("need d % heads == 0 and heads % kv_heads == 0");

    const Vocab vocab = read_vocab(argv[1]);
    auto id_or = [&](const std::string& k, uint32_t def) {
        auto it = vocab.ids.find("tokenizer.ggml." + k);
        return it == vocab.ids.end() ? def : it->second;
    };

    gguf::LlamaExportConfig cfg;
    cfg.name = "paperkiln-ttft-timing";
    cfg.context_length = ctx;
    cfg.embedding_length = d;
    cfg.block_count = layers;
    cfg.feed_forward_length = ff;
    cfg.head_count = heads;
    cfg.head_count_kv = kv_heads;
    cfg.vocab_size = static_cast<uint32_t>(vocab.tokens.size());
    cfg.tokens = vocab.tokens;
    cfg.tokenizer_model = vocab.model;
    cfg.bos_token_id = id_or("bos_token_id", 1);
    cfg.eos_token_id = id_or("eos_token_id", 1);
    cfg.unk_token_id = id_or("unknown_token_id", 0);
    cfg.pad_token_id = id_or("padding_token_id", 0);

    // One generator, filled in a fixed key order: the same seed and shape
    // give a byte-identical file.
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.02f);
    auto random = [&](size_t rows, size_t cols) {
        Matrix m(rows, cols);
        float* p = &m(0, 0);
        for (size_t i = 0; i < rows * cols; ++i) p[i] = noise(rng);
        return m;
    };
    auto ones = [](size_t n) { return Matrix(1, n, 1.0f); };

    const size_t kv_dim = static_cast<size_t>(kv_heads) * (d / heads);
    // HF [out, in] layout. The LM head is written explicitly, as TinyLlama
    // (untied) and mtstudio exports both do: ember.cpp finds no LM head when
    // output.weight is absent, rather than falling back to the embeddings.
    std::map<std::string, Matrix> sd;
    sd.emplace("embed_tokens.weight", random(cfg.vocab_size, d));
    sd.emplace("lm_head.weight", random(cfg.vocab_size, d));
    sd.emplace("norm.weight", ones(d));
    for (uint32_t l = 0; l < layers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        sd.emplace(p + "self_attn.q_proj.weight", random(d, d));
        sd.emplace(p + "self_attn.k_proj.weight", random(kv_dim, d));
        sd.emplace(p + "self_attn.v_proj.weight", random(kv_dim, d));
        sd.emplace(p + "self_attn.o_proj.weight", random(d, d));
        sd.emplace(p + "mlp.gate_proj.weight", random(ff, d));
        sd.emplace(p + "mlp.up_proj.weight", random(ff, d));
        sd.emplace(p + "mlp.down_proj.weight", random(d, ff));
        sd.emplace(p + "input_layernorm.weight", ones(d));
        sd.emplace(p + "post_attention_layernorm.weight", ones(d));
    }

    std::vector<std::string> unmapped;
    gguf::export_gguf_llama(argv[2], sd, cfg, &unmapped);
    if (!unmapped.empty()) {
        std::fprintf(stderr, "unmapped tensors (exporter map drifted): %zu\n", unmapped.size());
        return 1;
    }
    std::printf("wrote %s: d=%u layers=%u heads=%u kv_heads=%u ff=%u ctx=%u vocab=%u seed=%u\n",
                argv[2], d, layers, heads, kv_heads, ff, ctx, cfg.vocab_size, seed);
    return 0;
}
