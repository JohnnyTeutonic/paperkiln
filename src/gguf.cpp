#include "microtorch/gguf.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
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

// float32 -> IEEE fp16 bits, round to nearest even (numpy's astype(float16),
// which the reference quantiser uses for the block scale), with subnormals,
// overflow to inf and NaN handled.
uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) return static_cast<uint16_t>(sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0u));
    if (ax >= 0x477ff000u) return static_cast<uint16_t>(sign | 0x7c00u);  // >= 65520 rounds to inf
    if (ax < 0x38800000u) {  // below 2^-14: fp16 subnormal or zero
        if (ax <= 0x33000000u) return static_cast<uint16_t>(sign);  // <= 2^-25 ties to even 0
        const uint32_t e = ax >> 23, m = (ax & 0x7fffffu) | 0x800000u;
        const uint32_t shift = 126 - e;  // value in units of 2^-24 is m >> shift
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;  // may carry into the smallest normal
        return static_cast<uint16_t>(sign | h);
    }
    const uint32_t m = ax & 0x7fffffu;
    uint32_t h = (((ax >> 23) - 112) << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;  // a carry steps the exponent, as it should
    return static_cast<uint16_t>(sign | h);
}

}  // namespace

void quantize_q8_0(const float* x, size_t n, uint8_t* out) {
    if (n % 32 != 0) throw std::runtime_error("quantize_q8_0: n must be a multiple of 32");
    for (size_t b = 0; b < n / 32; ++b, x += 32, out += 34) {
        // gguf-py quants.Q8_0.quantize_blocks, operation for operation in
        // float32: d = amax / 127, id = 1 / d (0 when d is 0), q = roundf(x * id)
        // where its np_roundf rounds half away from zero, as std::round does.
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            if (!std::isfinite(x[i])) throw std::runtime_error("quantize_q8_0: non-finite input");
            amax = std::max(amax, std::fabs(x[i]));
        }
        const float d = amax / 127.0f;
        const float id = d == 0.0f ? 0.0f : 1.0f / d;
        const uint16_t dh = fp32_to_fp16(d);
        std::memcpy(out, &dh, 2);  // little-endian, as GGUF stores it
        for (int i = 0; i < 32; ++i) {
            const auto q = static_cast<int8_t>(std::round(x[i] * id));
            std::memcpy(out + 2 + i, &q, 1);
        }
    }
}

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
    const bool q8 = cfg.quant.rfind("q8_0", 0) == 0;
    if (cfg.quant != "f32" && cfg.quant != "q8_0" && cfg.quant != "q8_0_out" && cfg.quant != "q8_0_all")
        throw std::runtime_error("export_gguf_llama: quant must be f32, q8_0, q8_0_out or q8_0_all, got " +
                                 cfg.quant);
    // GGUFWriter sizes a tensor as num_elements * ggml_type_size(type), and
    // its size for Q8_0 is 1 (a byte count), so a Q8_0 tensor is handed over
    // with num_elements = its byte count. Refuse a writer that changed that.
    if (q8 && gguf_export::ggml_type_size(GGMLType::Q8_0) != 1)
        throw std::runtime_error("export_gguf_llama: GGUFWriter's Q8_0 size convention changed");

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
    if (cfg.expert_count > 0) {
        if (cfg.expert_used_count == 0 || cfg.expert_used_count > cfg.expert_count)
            throw std::runtime_error("export_gguf_llama: expert_used_count must be in [1, expert_count]");
        writer.write_metadata_uint32(a + ".expert_count", cfg.expert_count);
        writer.write_metadata_uint32(a + ".expert_used_count", cfg.expert_used_count);
    }

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

    // Every tensor goes through here: under a Q8_0 quant the selected 2-D
    // tensors are quantised into a kept-alive byte buffer (see
    // LlamaExportConfig::quant for the selection, the reference's rule).
    std::vector<std::unique_ptr<std::vector<uint8_t>>> owned_q8;
    auto add = [&](TensorExportInfo info) {
        const bool pick = q8 && info.shape.size() == 2 && info.type == GGMLType::F32 &&
                          (info.name.rfind("blk.", 0) == 0 ||
                           (cfg.quant != "q8_0" && info.name == "output.weight") ||
                           (cfg.quant == "q8_0_all" && info.name == "token_embd.weight"));
        if (pick) {
            const uint64_t ne0 = info.shape.back();  // the writer reverses: last entry is ne0
            if (ne0 % 32 != 0)
                throw std::runtime_error("export_gguf_llama: " + info.name + " has ne0 = " + std::to_string(ne0) +
                                         ", not a multiple of 32, so it cannot be stored as Q8_0");
            auto buf = std::make_unique<std::vector<uint8_t>>(info.num_elements / 32 * 34);
            quantize_q8_0(info.data, info.num_elements, buf->data());
            info.type = GGMLType::Q8_0;
            info.data = reinterpret_cast<const float*>(buf->data());  // written as raw bytes
            info.num_elements = buf->size();
            owned_q8.push_back(std::move(buf));
        }
        writer.add_tensor(info);
    };

    // NOTE: GGUFWriter takes `shape` in row-major (numpy) order and writes it
    // REVERSED, so ne0 (the contiguous axis) is the LAST entry passed. An
    // [out][in] buffer is passed as {out, in} and lands as ne = {in, out}, the
    // order llama.cpp checks. (Exports before 19 Oct 2026 passed {in, out} and
    // recorded the dims reversed; the bytes were right and ember, which reads
    // by element count, loaded them unchanged.)
    //
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
            info.shape = {static_cast<uint64_t>(C), static_cast<uint64_t>(R)};  // {out,in} -> ne={in,out}
            info.data = buf->data();
            info.num_elements = R * C;
            owned.push_back(std::move(buf));
        } else {
            // HF layout rows=[out], cols=[in] is already llama's byte order.
            info.shape = {static_cast<uint64_t>(w.rows()),
                          static_cast<uint64_t>(w.cols())};  // {out,in} -> ne={in,out}
            info.data = &w(0, 0);
            info.num_elements = w.rows() * w.cols();
        }
        add(info);
    };

    // RMSNorm gamma stored [1, d] -> 1-D tensor {d}.
    auto add_norm = [&](const Matrix& g, const std::string& gguf_name) {
        TensorExportInfo info;
        info.name = gguf_name;
        info.shape = {static_cast<uint64_t>(g.rows() * g.cols())};
        info.type = GGMLType::F32;
        info.data = &g(0, 0);
        info.num_elements = g.rows() * g.cols();
        add(info);
    };

    auto require = [&](const std::string& key) -> const Matrix& {
        const Matrix* m = find(state_dict, key);
        if (!m) throw std::runtime_error("export_gguf_llama: missing tensor " + key);
        mapped_keys.push_back(key);
        return *m;
    };

    // MoE tensors in llama.cpp's dimension order, by the same convention:
    // passing {E, out, in} for a buffer laid out [E][out][in] gives the file
    // ne = {in, out, E}, which is what llama.cpp's create_tensor checks.
    //
    // Stack E projections as [E][out][in] row-major (transposed per matrix
    // under weights_in_out). Returns the buffer; in/out via the references.
    auto pack = [&](const std::vector<const Matrix*>& ws, const std::string& gguf_name, size_t& in, size_t& out) {
        in = cfg.weights_in_out ? ws[0]->rows() : ws[0]->cols();
        out = cfg.weights_in_out ? ws[0]->cols() : ws[0]->rows();
        auto buf = std::make_unique<std::vector<float>>(ws.size() * in * out);
        for (size_t e = 0; e < ws.size(); ++e) {
            const Matrix& w = *ws[e];
            if (w.rows() != ws[0]->rows() || w.cols() != ws[0]->cols())
                throw std::runtime_error("export_gguf_llama: expert shapes differ in " + gguf_name);
            float* dst = buf->data() + e * in * out;
            for (size_t o = 0; o < out; ++o)
                for (size_t i = 0; i < in; ++i) dst[o * in + i] = cfg.weights_in_out ? w(i, o) : w(o, i);
        }
        return buf;
    };
    auto add_packed = [&](std::unique_ptr<std::vector<float>> buf, std::vector<uint64_t> shape,
                          const std::string& gguf_name) {
        TensorExportInfo info;
        info.type = GGMLType::F32;
        info.name = gguf_name;
        info.shape = std::move(shape);
        info.data = buf->data();
        info.num_elements = buf->size();
        owned.push_back(std::move(buf));
        add(info);
    };
    // Router: microtorch [d, E] (or HF [E, d]) -> [E][d] bytes, ne = {d, E}.
    auto add_router = [&](const std::string& hf_prefix, const std::string& gguf_name) {
        size_t in = 0, out = 0;
        auto buf = pack({&require(hf_prefix + "mlp.router.weight")}, gguf_name, in, out);
        if (out != cfg.expert_count) throw std::runtime_error("export_gguf_llama: router width != expert_count");
        add_packed(std::move(buf), {static_cast<uint64_t>(out), static_cast<uint64_t>(in)}, gguf_name);
    };
    // Experts: E projections -> one 3-D tensor, ne = {in, out, E}; expert e's
    // [out][in] block at element offset e * in * out (llama.cpp *_exps).
    auto add_experts = [&](const std::string& hf_prefix, const std::string& proj, const std::string& gguf_name) {
        std::vector<const Matrix*> ws;
        for (uint32_t e = 0; e < cfg.expert_count; ++e)
            ws.push_back(&require(hf_prefix + "mlp.experts." + std::to_string(e) + "." + proj));
        size_t in = 0, out = 0;
        auto buf = pack(ws, gguf_name, in, out);
        add_packed(std::move(buf),
                   {static_cast<uint64_t>(ws.size()), static_cast<uint64_t>(out), static_cast<uint64_t>(in)},
                   gguf_name);
    };

    // Embeddings: [vocab, hidden] row-major == llama layout, ne={hidden,vocab}.
    {
        const Matrix& emb = require("embed_tokens.weight");
        TensorExportInfo info;
        info.name = "token_embd.weight";
        info.shape = {static_cast<uint64_t>(emb.rows()), static_cast<uint64_t>(emb.cols())};  // {vocab,hidden} -> ne={hidden,vocab}
        info.type = GGMLType::F32;
        info.data = &emb(0, 0);
        info.num_elements = emb.rows() * emb.cols();
        add(info);
    }

    for (uint32_t l = 0; l < cfg.block_count; ++l) {
        const std::string hf = "layers." + std::to_string(l) + ".";
        const std::string gg = "blk." + std::to_string(l) + ".";
        add_proj(require(hf + "self_attn.q_proj.weight"), gg + "attn_q.weight");
        add_proj(require(hf + "self_attn.k_proj.weight"), gg + "attn_k.weight");
        add_proj(require(hf + "self_attn.v_proj.weight"), gg + "attn_v.weight");
        add_proj(require(hf + "self_attn.o_proj.weight"), gg + "attn_output.weight");
        add_norm(require(hf + "input_layernorm.weight"), gg + "attn_norm.weight");
        if (cfg.expert_count > 0) {
            add_router(hf, gg + "ffn_gate_inp.weight");
            add_experts(hf, "gate_proj.weight", gg + "ffn_gate_exps.weight");
            add_experts(hf, "up_proj.weight", gg + "ffn_up_exps.weight");
            add_experts(hf, "down_proj.weight", gg + "ffn_down_exps.weight");
        } else {
            add_proj(require(hf + "mlp.gate_proj.weight"), gg + "ffn_gate.weight");
            add_proj(require(hf + "mlp.up_proj.weight"), gg + "ffn_up.weight");
            add_proj(require(hf + "mlp.down_proj.weight"), gg + "ffn_down.weight");
        }
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
