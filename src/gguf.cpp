#include "microtorch/gguf.hpp"

#include <memory>
#include <stdexcept>

#include "gguf_export.hpp"  // transformer_cpp's verified writer

namespace microtorch {
namespace gguf {

namespace {

using gguf_export::GGMLType;
using gguf_export::GGUFWriter;
using gguf_export::TensorExportInfo;

// Look up `key` allowing an optional "model." prefix on the dict side.
const Matrix* find(const std::map<std::string, Matrix>& sd, const std::string& key) {
    auto it = sd.find(key);
    if (it != sd.end()) return &it->second;
    it = sd.find("model." + key);
    if (it != sd.end()) return &it->second;
    return nullptr;
}

}  // namespace

void export_gguf_llama(const std::string& path, const std::map<std::string, Matrix>& state_dict,
                       const LlamaExportConfig& cfg, std::vector<std::string>* unmapped) {
    if (cfg.embedding_length == 0 || cfg.block_count == 0 || cfg.feed_forward_length == 0 ||
        cfg.head_count == 0 || cfg.vocab_size == 0) {
        throw std::runtime_error(
            "export_gguf_llama: embedding_length, block_count, feed_forward_length, "
            "head_count and vocab_size must all be set");
    }
    const uint32_t kv_heads = cfg.head_count_kv ? cfg.head_count_kv : cfg.head_count;
    const uint32_t head_dim = cfg.embedding_length / cfg.head_count;

    GGUFWriter writer(path);

    // ---- metadata: the exact llama.* block tinyllama.cpp reads ----
    writer.write_metadata_string("general.architecture", cfg.architecture);
    writer.write_metadata_string("general.name", cfg.name);
    writer.write_metadata_uint32("general.alignment", gguf_export::GGUF_DEFAULT_ALIGNMENT);
    const std::string& a = cfg.architecture;
    writer.write_metadata_uint32(a + ".context_length", cfg.context_length);
    writer.write_metadata_uint32(a + ".embedding_length", cfg.embedding_length);
    writer.write_metadata_uint32(a + ".block_count", cfg.block_count);
    writer.write_metadata_uint32(a + ".feed_forward_length", cfg.feed_forward_length);
    writer.write_metadata_uint32(a + ".attention.head_count", cfg.head_count);
    writer.write_metadata_uint32(a + ".attention.head_count_kv", kv_heads);
    writer.write_metadata_float32(a + ".attention.layer_norm_rms_epsilon", cfg.rms_eps);
    writer.write_metadata_uint32(a + ".vocab_size", cfg.vocab_size);
    writer.write_metadata_float32(a + ".rope.freq_base", cfg.rope_freq_base);
    writer.write_metadata_uint32(a + ".rope.dimension_count", head_dim);

    if (!cfg.tokens.empty()) {
        writer.write_metadata_string("tokenizer.ggml.model", cfg.tokenizer_model);
        writer.write_metadata_string_array("tokenizer.ggml.tokens", cfg.tokens);
        if (!cfg.merges.empty()) writer.write_metadata_string_array("tokenizer.ggml.merges", cfg.merges);
        writer.write_metadata_float_array("tokenizer.ggml.scores",
                                          std::vector<float>(cfg.tokens.size(), 0.0f));
        writer.write_metadata_uint32("tokenizer.ggml.bos_token_id", cfg.bos_token_id);
        writer.write_metadata_uint32("tokenizer.ggml.eos_token_id", cfg.eos_token_id);
        writer.write_metadata_uint32("tokenizer.ggml.unknown_token_id", cfg.unk_token_id);
        writer.write_metadata_uint32("tokenizer.ggml.padding_token_id", cfg.pad_token_id);
    }

    // ---- tensors ----
    // Keep-alive for transposed copies: GGUFWriter buffers raw pointers
    // until finalize(), so every temporary must outlive it (the same
    // `owned` idiom as transformer_cpp's exporter).
    std::vector<std::unique_ptr<std::vector<float>>> owned;
    std::vector<std::string> mapped_keys;  // dict keys we consumed

    // 2-D projection in llama layout [out][in] row-major, ne = {in, out}.
    auto add_proj = [&](const Matrix& w, const std::string& gguf_name) {
        TensorExportInfo info;
        info.type = GGMLType::F32;
        info.name = gguf_name;
        if (cfg.weights_in_out) {
            // microtorch layout [in][out] -> transpose to [out][in]
            const size_t R = w.rows(), C = w.cols();  // R=in, C=out
            auto buf = std::make_unique<std::vector<float>>(R * C);
            for (size_t i = 0; i < R; ++i)
                for (size_t j = 0; j < C; ++j) (*buf)[j * R + i] = w(i, j);
            info.shape = {static_cast<uint64_t>(R), static_cast<uint64_t>(C)};  // ne={in,out}
            info.data = buf->data();
            info.num_elements = R * C;
            owned.push_back(std::move(buf));
        } else {
            // HF layout rows=[out], cols=[in] is already llama's byte order.
            info.shape = {static_cast<uint64_t>(w.cols()),
                          static_cast<uint64_t>(w.rows())};  // ne={in,out}
            info.data = &w(0, 0);
            info.num_elements = w.rows() * w.cols();
        }
        writer.add_tensor(info);
    };

    // RMSNorm gamma stored [1, d] -> 1-D tensor {d}.
    auto add_norm = [&](const Matrix& g, const std::string& gguf_name) {
        TensorExportInfo info;
        info.name = gguf_name;
        info.shape = {static_cast<uint64_t>(g.rows() * g.cols())};
        info.type = GGMLType::F32;
        info.data = &g(0, 0);
        info.num_elements = g.rows() * g.cols();
        writer.add_tensor(info);
    };

    auto require = [&](const std::string& key) -> const Matrix& {
        const Matrix* m = find(state_dict, key);
        if (!m) throw std::runtime_error("export_gguf_llama: missing tensor " + key);
        mapped_keys.push_back(key);
        return *m;
    };

    // Embeddings: [vocab, hidden] row-major == llama layout, ne={hidden,vocab}.
    {
        const Matrix& emb = require("embed_tokens.weight");
        TensorExportInfo info;
        info.name = "token_embd.weight";
        info.shape = {static_cast<uint64_t>(emb.cols()), static_cast<uint64_t>(emb.rows())};
        info.type = GGMLType::F32;
        info.data = &emb(0, 0);
        info.num_elements = emb.rows() * emb.cols();
        writer.add_tensor(info);
    }

    for (uint32_t l = 0; l < cfg.block_count; ++l) {
        const std::string hf = "layers." + std::to_string(l) + ".";
        const std::string gg = "blk." + std::to_string(l) + ".";
        add_proj(require(hf + "self_attn.q_proj.weight"), gg + "attn_q.weight");
        add_proj(require(hf + "self_attn.k_proj.weight"), gg + "attn_k.weight");
        add_proj(require(hf + "self_attn.v_proj.weight"), gg + "attn_v.weight");
        add_proj(require(hf + "self_attn.o_proj.weight"), gg + "attn_output.weight");
        add_norm(require(hf + "input_layernorm.weight"), gg + "attn_norm.weight");
        add_proj(require(hf + "mlp.gate_proj.weight"), gg + "ffn_gate.weight");
        add_proj(require(hf + "mlp.up_proj.weight"), gg + "ffn_up.weight");
        add_proj(require(hf + "mlp.down_proj.weight"), gg + "ffn_down.weight");
        add_norm(require(hf + "post_attention_layernorm.weight"), gg + "ffn_norm.weight");
    }

    add_norm(require("norm.weight"), "output_norm.weight");

    // lm_head is optional: tied-embedding models reuse token_embd.
    if (const Matrix* head = find(state_dict, "lm_head.weight")) {
        mapped_keys.push_back("lm_head.weight");
        add_proj(*head, "output.weight");
    }

    // Report anything in the dict we did not export (biases, rotary
    // caches...) so callers can fail loudly instead of silently shipping a
    // model missing weights it needs.
    if (unmapped) {
        for (const auto& [key, m] : state_dict) {
            std::string bare = key.rfind("model.", 0) == 0 ? key.substr(6) : key;
            bool used = false;
            for (const auto& mk : mapped_keys) used = used || mk == bare;
            if (!used) unmapped->push_back(key);
        }
    }

    if (!writer.finalize()) {
        throw std::runtime_error("export_gguf_llama: finalize failed for " + path);
    }
}

}  // namespace gguf
}  // namespace microtorch
