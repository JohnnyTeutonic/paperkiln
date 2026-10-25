// 4-bit GGUF export test (LlamaExportConfig::quant q4_0 / q4_k_m / q4_k_s).
//
// 1. quantize_q4_0 / quantize_q5_0 / quantize_q5_1 on known blocks: zeros,
//    a negative and a positive maximum, two values of equal magnitude (the
//    first sets the sign), values only in the upper half, a min/max block,
//    a constant block (Q5_1's d = 0) and fp16-subnormal scales. Expected
//    bytes were produced by gguf-py 0.19 (quants.quantize), which follows
//    ggml-quants.c's quantize_row_*_ref; the exporter's bytes were in turn
//    checked file for file against llama.cpp's own quantiser (llama-cpp-python
//    0.3.36), see the commit message.
// 2. Tiny llama exports, parsed back with an independent spec reader:
//    llama.cpp's tensor order, the per-tensor type table llama.cpp chooses
//    for 8 layers at d = 64 (every k-quant falls back), the two keys it
//    appends, 32-byte offsets and file end, and every stored tensor equal to
//    the matching quantize_* of its source. Also with tied embeddings
//    (token_embd takes output.weight's type).
// 3. Refusals: MoE and wrong block lengths.
// 4. Real k-quant blocks: an 8-layer export at d = 256, d_ff = 512 (no
//    fallback anywhere), llama.cpp's types (Q4_K, Q5_K, Q6_K) and every tensor
//    equal to quantize_q4_k / q5_k / q6_k of its source.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "microtorch/gguf.hpp"

#include "check.hpp"  // Release-proof CHECK (assert vanishes under -DNDEBUG)

using namespace microtorch;

namespace {

using QuantFn = void (*)(const float*, size_t, uint8_t*);

std::vector<float> block(std::initializer_list<float> head) {
    std::vector<float> x(32, 0.0f);
    size_t i = 0;
    for (float v : head) x[i++] = v;
    return x;
}

std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) {
        s += d[p[i] >> 4];
        s += d[p[i] & 15];
    }
    return s;
}

void expect_hex(QuantFn fn, size_t bytes, const std::vector<float>& x, const std::string& want, const char* what) {
    std::vector<uint8_t> out(bytes);
    fn(x.data(), 32, out.data());
    const std::string got = hex(out.data(), bytes);
    if (got != want) std::fprintf(stderr, "%s: got %s\n%*s want %s\n", what, got.c_str(), (int)std::strlen(what), "", want.c_str());
    CHECK(got == want);
}

// ---------- minimal spec-based GGUF reader ----------
struct ParsedTensor {
    std::string name;
    std::vector<uint64_t> dims;  // ne0 first, as stored
    uint32_t type = 0;
    uint64_t offset = 0;  // relative to the data section
};

struct ParsedGGUF {
    std::vector<std::string> keys;            // in file order
    std::map<std::string, uint32_t> u32;      // UINT32 values
    std::vector<ParsedTensor> tensors;
    uint64_t data_start = 0;
    std::vector<uint8_t> bytes;
};

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

void skip_value(const std::vector<uint8_t>& b, size_t& p, uint32_t vt) {
    switch (vt) {
        case 4: case 6: p += 4; break;  // UINT32, FLOAT32
        case 8: rd_str(b, p); break;    // STRING
        case 9: {                       // ARRAY
            const uint32_t et = rd<uint32_t>(b, p);
            const uint64_t n = rd<uint64_t>(b, p);
            for (uint64_t k = 0; k < n; ++k) skip_value(b, p, et);
            break;
        }
        default: CHECK(false && "unexpected metadata value type");
    }
}

ParsedGGUF parse_gguf(const std::string& path) {
    ParsedGGUF g;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    CHECK(f && "cannot open exported gguf");
    g.bytes.resize(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(g.bytes.data()), static_cast<std::streamsize>(g.bytes.size()));
    size_t p = 0;
    CHECK(rd<uint32_t>(g.bytes, p) == 0x46554747u);  // "GGUF"
    CHECK(rd<uint32_t>(g.bytes, p) == 3u);
    const uint64_t n_tensors = rd<uint64_t>(g.bytes, p);
    const uint64_t n_meta = rd<uint64_t>(g.bytes, p);
    for (uint64_t i = 0; i < n_meta; ++i) {
        const std::string key = rd_str(g.bytes, p);
        const uint32_t vt = rd<uint32_t>(g.bytes, p);
        g.keys.push_back(key);
        if (vt == 4) {
            g.u32[key] = rd<uint32_t>(g.bytes, p);
        } else {
            skip_value(g.bytes, p, vt);
        }
    }
    for (uint64_t i = 0; i < n_tensors; ++i) {
        ParsedTensor t;
        t.name = rd_str(g.bytes, p);
        const uint32_t nd = rd<uint32_t>(g.bytes, p);
        for (uint32_t d = 0; d < nd; ++d) t.dims.push_back(rd<uint64_t>(g.bytes, p));
        t.type = rd<uint32_t>(g.bytes, p);
        t.offset = rd<uint64_t>(g.bytes, p);
        g.tensors.push_back(std::move(t));
    }
    g.data_start = (p + 31) & ~uint64_t(31);
    return g;
}

// Deterministic sign-mixed values with per-row scale, distinct per tensor.
Matrix filled(size_t r, size_t c, float base) {
    Matrix m(r, c);
    uint32_t s = 2463534242u ^ static_cast<uint32_t>(base * 1000.0f);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            m(i, j) = (static_cast<float>(s % 20001u) / 10000.0f - 1.0f) * (0.02f + 0.01f * static_cast<float>(i % 5));
        }
    return m;
}

constexpr uint32_t F32 = 0, Q4_0 = 2, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8;

size_t block_bytes(uint32_t t) { return t == Q4_0 ? 18 : t == Q5_0 ? 22 : t == Q5_1 ? 24 : 34; }
QuantFn quantizer(uint32_t t) {
    return t == Q4_0 ? gguf::quantize_q4_0 : t == Q5_0 ? gguf::quantize_q5_0 : t == Q5_1 ? gguf::quantize_q5_1
                                                                                        : gguf::quantize_q8_0;
}

}  // namespace

int main() {
    printf("=== known blocks (gguf-py 0.19 bytes) ===\n");
    const std::vector<std::pair<const char*, std::vector<float>>> V = {
        {"zeros", block({})},
        {"negmax", block({-8.0f, 1.0f, 2.0f, 3.0f, 7.5f, -0.5f, 0.49999997f, -7.0f})},
        {"posmax", block({8.0f, -8.0f, 0.5f, -0.5f, 1.25f, 3.0f})},
        {"tie_first", block({3.0f, -3.0f, 1.0f})},
        {"upper", block({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -16.0f, 15.0f, 4.0f, -4.0f, 0.25f})},
        {"minmax", block({1.0f, 2.0f, 3.0f, 32.0f, -0.1f})},
        {"const", std::vector<float>(32, 0.7f)},
        {"tiny", block({1e-6f, -3e-7f, 5e-7f})},  // fp16-subnormal scales
    };
    // negmax under Q4_0 by hand: d = -8 / -8 = 1 (0x3c00); q = trunc(x + 8.5)
    // capped at 15: -8 -> 0, 1 -> 9, 7.5 -> 15, -0.5 -> 8, 0.49999997 -> 9
    // (8.99999997 rounds to 9.0f), -7 -> 1; value j + 16 = 0 -> 8 (high nibble).
    const std::map<std::string, std::vector<std::string>> want = {
        {"zeros", {"008088888888888888888888888888888888", "0080ffffffff00000000000000000000000000000000",
                   "000000000000000000000000000000000000000000000000"}},
        {"negmax", {"003c80898a8b8f8889818888888888888888", "00385effffff000204060f0f01020000000000000000",
                    "003800c85effffff000204060f0f01020000000000000000"}},
        {"posmax", {"00bc808f8889878588888888888888888888", "00b8caffffff000f0f010e0a00000000000000000000",
                    "213800c8f5ffffff0f00000f020500000000000000000000"}},
        {"tie_first", {"00b6808f8588888888888888888888888888", "00b2faffffff000f0b00000000000000000000000000",
                       "323200c2fdffffff0f000500000000000000000000000000"}},
        {"upper", {"004008f8a868888888888888888888888888", "003cfffff6ff00f040c0000000000000000000000000",
                   "003c00ccfffff6ff00f040c0000000000000000000000000"}},
        {"minmax", {"00c488888780888888888888888888888888", "00c0f1ffffff000f0f00000000000000000000000000",
                    "243c66ae080000000102030f000000000000000000000000"}},
        {"const", {"9aad00000000000000000000000000000000", "9aa90000000000000000000000000000000000000000",
                   "00009a390000000000000000000000000000000000000000"}},
        {"tiny", {"0280808a8488888888888888888888888888", "0180faffffff00050800000000000000000000000000",
                  "01000580050000007f707377777777777777777777777777"}},
    };
    for (const auto& [name, x] : V) {
        const auto& w = want.at(name);
        expect_hex(gguf::quantize_q4_0, 18, x, w[0], (std::string("Q4_0 ") + name).c_str());
        expect_hex(gguf::quantize_q5_0, 22, x, w[1], (std::string("Q5_0 ") + name).c_str());
        expect_hex(gguf::quantize_q5_1, 24, x, w[2], (std::string("Q5_1 ") + name).c_str());
    }
    {
        // blocks are independent: the second block of a pair equals it alone
        std::vector<float> two(64, 0.0f);
        const auto& nm = V[1].second;
        std::copy(nm.begin(), nm.end(), two.begin() + 32);
        for (const auto& [fn, bb] : std::vector<std::pair<QuantFn, size_t>>{
                 {gguf::quantize_q4_0, 18}, {gguf::quantize_q5_0, 22}, {gguf::quantize_q5_1, 24}}) {
            std::vector<uint8_t> pair(2 * bb), one(bb);
            fn(two.data(), 64, pair.data());
            fn(nm.data(), 32, one.data());
            CHECK(std::memcmp(pair.data() + bb, one.data(), bb) == 0);
        }
    }
    for (QuantFn fn : {gguf::quantize_q4_0, gguf::quantize_q5_0, gguf::quantize_q5_1}) {
        bool threw = false;
        try {
            std::vector<float> x(33, 1.0f);
            std::vector<uint8_t> out(96);
            fn(x.data(), x.size(), out.data());
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        try {
            std::vector<float> x(32, 1.0f);
            x[3] = INFINITY;
            std::vector<uint8_t> out(24);
            fn(x.data(), x.size(), out.data());
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
    }
    printf("  Q4_0 / Q5_0 / Q5_1: 8 blocks each match; block independence; n %% 32 and inf refused\n");

    printf("=== tiny 4-bit exports ===\n");
    const uint32_t L = 8, D = 64, FF = 96, V_ = 7, H = 2;
    std::map<std::string, Matrix> sd;
    float base = 1.0f;
    sd.emplace("model.embed_tokens.weight", filled(V_, D, base++));
    for (uint32_t l = 0; l < L; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        sd.emplace(p + "self_attn.q_proj.weight", filled(D, D, base++));
        sd.emplace(p + "self_attn.k_proj.weight", filled(D, D, base++));
        sd.emplace(p + "self_attn.v_proj.weight", filled(D, D, base++));
        sd.emplace(p + "self_attn.o_proj.weight", filled(D, D, base++));
        sd.emplace(p + "input_layernorm.weight", filled(1, D, base++));
        sd.emplace(p + "mlp.gate_proj.weight", filled(FF, D, base++));
        sd.emplace(p + "mlp.up_proj.weight", filled(FF, D, base++));
        sd.emplace(p + "mlp.down_proj.weight", filled(D, FF, base++));
        sd.emplace(p + "post_attention_layernorm.weight", filled(1, D, base++));
    }
    sd.emplace("model.norm.weight", filled(1, D, base++));
    sd.emplace("lm_head.weight", filled(V_, D, base++));
    std::map<std::string, const Matrix*> src = {{"token_embd.weight", &sd.at("model.embed_tokens.weight")},
                                                {"output.weight", &sd.at("lm_head.weight")},
                                                {"output_norm.weight", &sd.at("model.norm.weight")}};
    for (uint32_t l = 0; l < L; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".", g = "blk." + std::to_string(l) + ".";
        for (const auto& [hf, gg] : std::map<std::string, std::string>{{"self_attn.q_proj", "attn_q"},
                                                                       {"self_attn.k_proj", "attn_k"},
                                                                       {"self_attn.v_proj", "attn_v"},
                                                                       {"self_attn.o_proj", "attn_output"},
                                                                       {"input_layernorm", "attn_norm"},
                                                                       {"mlp.gate_proj", "ffn_gate"},
                                                                       {"mlp.up_proj", "ffn_up"},
                                                                       {"mlp.down_proj", "ffn_down"},
                                                                       {"post_attention_layernorm", "ffn_norm"}})
            src[g + gg + ".weight"] = &sd.at(p + hf + ".weight");
    }

    // llama.cpp's file order: non-layer tensors by name, then each layer's
    // tensors by name.
    std::vector<std::string> order = {"output.weight", "output_norm.weight", "token_embd.weight"};
    for (uint32_t l = 0; l < L; ++l)
        for (const char* n : {"attn_k", "attn_norm", "attn_output", "attn_q", "attn_v", "ffn_down", "ffn_gate",
                              "ffn_norm", "ffn_up"})
            order.push_back("blk." + std::to_string(l) + "." + n + ".weight");

    // llama.cpp's type for each tensor at L = 8, d = 64 (written out by hand
    // from llama_tensor_get_type and its fallback; ne0 = 64 or 96 is never a
    // multiple of 256, so Q4_K -> Q5_0, Q5_K -> Q5_1, Q6_K -> Q8_0).
    // use_more_bits(i, 8) holds for i = 0 (first eighth), 3 and 6 (every
    // third after it) and 7 (last eighth).
    auto want_type = [&](const std::string& quant, const std::string& name, bool tied) -> uint32_t {
        if (name.find("norm") != std::string::npos) return F32;
        if (name == "output.weight" || (tied && name == "token_embd.weight")) return Q8_0;
        const uint32_t base_t = quant == "q4_0" ? Q4_0 : Q5_0;
        if (name == "token_embd.weight") return base_t;
        const int l = std::atoi(name.c_str() + 4);
        const bool more = l == 0 || l == 3 || l == 6 || l == 7;
        if (name.find("attn_v") != std::string::npos) {
            if (quant == "q4_k_m" && more) return Q8_0;
            if (quant == "q4_k_s" && l < 4) return Q5_1;
        }
        if (name.find("ffn_down") != std::string::npos) {
            if (quant == "q4_k_m" && more) return Q8_0;
            if (quant == "q4_k_s" && l < 1) return Q5_1;
        }
        return base_t;
    };

    gguf::LlamaExportConfig cfg;
    cfg.embedding_length = D;
    cfg.block_count = L;
    cfg.feed_forward_length = FF;
    cfg.head_count = H;
    cfg.vocab_size = V_;
    cfg.tokens = {"<unk>", "|", "a", "b", "c", "d", "e"};

    const char* tmpdir = std::getenv("TMPDIR");
#ifdef _WIN32
    if (!tmpdir) tmpdir = std::getenv("TEMP");
#endif
    const std::string tmp = tmpdir ? std::string(tmpdir) : std::string("/tmp");

    for (const bool tied : {false, true}) {
        std::map<std::string, Matrix> sdt(sd);
        if (tied) sdt.erase("lm_head.weight");
        for (const std::string quant : {"q4_0", "q4_k_m", "q4_k_s"}) {
            cfg.quant = quant;
            const std::string path = tmp + "/gguf_q4_test_" + quant + (tied ? "_tied" : "") + ".gguf";
            gguf::export_gguf_llama(path, sdt, cfg);
            const ParsedGGUF g = parse_gguf(path);
            // the two keys llama.cpp appends, last and in its order
            CHECK(g.keys.size() >= 2 && g.keys[g.keys.size() - 2] == "general.quantization_version" &&
                  g.keys.back() == "general.file_type");
            CHECK(g.u32.at("general.quantization_version") == 2u);
            CHECK(g.u32.at("general.file_type") == (quant == "q4_0" ? 2u : quant == "q4_k_s" ? 14u : 15u));
            CHECK((g.data_start % 32) == 0 && (g.bytes.size() % 32) == 0);
            std::vector<std::string> names;
            for (const auto& t : g.tensors) names.push_back(t.name);
            std::vector<std::string> want_order = order;
            if (tied) want_order.erase(want_order.begin());
            CHECK(names == want_order);
            std::map<uint32_t, int> count;
            for (size_t k = 0; k < g.tensors.size(); ++k) {
                const ParsedTensor& t = g.tensors[k];
                const Matrix& m = *src.at(t.name);
                const size_t n = m.rows() * m.cols();
                const uint32_t wt = want_type(quant, t.name, tied);
                if (t.type != wt) std::fprintf(stderr, "%s %s: type %u, want %u\n", quant.c_str(), t.name.c_str(), t.type, wt);
                CHECK(t.type == wt);
                ++count[t.type];
                if (m.rows() > 1) CHECK(t.dims == std::vector<uint64_t>({m.cols(), m.rows()}));  // ne = {in, out}
                CHECK((t.offset % 32) == 0);
                const size_t bytes = wt == F32 ? n * sizeof(float) : n / 32 * block_bytes(wt);
                const uint64_t end = (t.offset + bytes + 31) & ~uint64_t(31);
                CHECK(k + 1 < g.tensors.size() ? g.tensors[k + 1].offset == end
                                               : g.data_start + end == g.bytes.size());
                const uint8_t* at = g.bytes.data() + g.data_start + t.offset;
                if (wt == F32) {
                    CHECK(std::memcmp(at, &m(0, 0), bytes) == 0);
                } else {
                    std::vector<uint8_t> q(bytes);
                    quantizer(wt)(&m(0, 0), n, q.data());
                    CHECK(std::memcmp(at, q.data(), bytes) == 0);
                }
                for (size_t z = bytes; z < end - t.offset; ++z) CHECK(at[z] == 0);  // zero padding
            }
            std::remove(path.c_str());
            printf("  %-6s%s F32 %d, Q4_0 %d, Q5_0 %d, Q5_1 %d, Q8_0 %d: order/types/keys/offsets/bytes ok\n",
                   quant.c_str(), tied ? " tied" : "     ", count[F32], count[Q4_0], count[Q5_0], count[Q5_1],
                   count[Q8_0]);
        }
    }

    printf("=== refusals ===\n");
    auto refusal = [&](const std::map<std::string, Matrix>& m, const gguf::LlamaExportConfig& c) {
        std::string msg;
        try {
            gguf::export_gguf_llama(tmp + "/gguf_q4_test_bad.gguf", m, c);
        } catch (const std::runtime_error& e) {
            msg = e.what();
        }
        std::remove((tmp + "/gguf_q4_test_bad.gguf").c_str());
        return msg;
    };
    {
        gguf::LlamaExportConfig c = cfg;
        c.quant = "q4_0";
        c.expert_count = 4;
        c.expert_used_count = 2;
        CHECK(refusal(sd, c).find("MoE") != std::string::npos);
        c = cfg;
        c.quant = "q4_1";
        CHECK(refusal(sd, c).find("quant must be") != std::string::npos);
    }
    {
        // ne0 = 40 for ffn_down: refused by name, with its type
        std::map<std::string, Matrix> bad(sd);
        for (uint32_t l = 0; l < L; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".mlp.";
            bad[p + "gate_proj.weight"] = filled(40, D, 7.0f);
            bad[p + "up_proj.weight"] = filled(40, D, 8.0f);
            bad[p + "down_proj.weight"] = filled(D, 40, 9.0f);
        }
        gguf::LlamaExportConfig c = cfg;
        c.quant = "q4_k_m";
        c.feed_forward_length = 40;
        const std::string msg = refusal(bad, c);
        CHECK(msg.find("blk.0.ffn_down.weight") != std::string::npos && msg.find("Q8_0") != std::string::npos);
    }
    printf("  MoE, unknown quant and ne0 %% 32 != 0 refused\n");

    printf("=== real k-quant blocks (d = 256, d_ff = 512, 8 layers) ===\n");
    {
        // Every ne0 a multiple of 256, so nothing falls back: llama.cpp's types
        // stand as llama_tensor_get_type chooses them (output.weight Q6_K under
        // every ftype, q4_0's included), and each stored tensor must equal the
        // k-quant block function of its source (whose bytes against llama.cpp
        // b6000 are checked below and on pk_base_v1's tensors, CHANGELOG).
        const uint32_t W = 256, WF = 512, LW = 8;
        std::map<std::string, Matrix> wide;
        std::map<std::string, const Matrix*> wsrc;
        float b2 = 100.0f;
        wide.emplace("embed_tokens.weight", filled(V_, W, b2++));
        for (uint32_t l = 0; l < LW; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            for (const char* n : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"})
                wide.emplace(p + n + ".weight", filled(W, W, b2++));
            wide.emplace(p + "mlp.gate_proj.weight", filled(WF, W, b2++));
            wide.emplace(p + "mlp.up_proj.weight", filled(WF, W, b2++));
            wide.emplace(p + "mlp.down_proj.weight", filled(W, WF, b2++));
            wide.emplace(p + "input_layernorm.weight", filled(1, W, b2++));
            wide.emplace(p + "post_attention_layernorm.weight", filled(1, W, b2++));
        }
        wide.emplace("norm.weight", filled(1, W, b2++));
        wide.emplace("lm_head.weight", filled(V_, W, b2++));
        wsrc = {{"token_embd.weight", &wide.at("embed_tokens.weight")}, {"output.weight", &wide.at("lm_head.weight")},
                {"output_norm.weight", &wide.at("norm.weight")}};
        for (uint32_t l = 0; l < LW; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".", g = "blk." + std::to_string(l) + ".";
            for (const auto& [hf, gg] : std::map<std::string, std::string>{{"self_attn.q_proj", "attn_q"},
                                                                           {"self_attn.k_proj", "attn_k"},
                                                                           {"self_attn.v_proj", "attn_v"},
                                                                           {"self_attn.o_proj", "attn_output"},
                                                                           {"input_layernorm", "attn_norm"},
                                                                           {"mlp.gate_proj", "ffn_gate"},
                                                                           {"mlp.up_proj", "ffn_up"},
                                                                           {"mlp.down_proj", "ffn_down"},
                                                                           {"post_attention_layernorm", "ffn_norm"}})
                wsrc[g + gg + ".weight"] = &wide.at(p + hf + ".weight");
        }
        constexpr uint32_t Q4_K = 12, Q5_K = 13, Q6_K = 14;
        auto kt = [&](const std::string& quant, const std::string& name) -> uint32_t {
            if (name.find("norm") != std::string::npos) return F32;
            if (name == "output.weight") return Q6_K;
            const uint32_t base_t = quant == "q4_0" ? Q4_0 : Q4_K;
            if (name == "token_embd.weight") return base_t;
            const int l = std::atoi(name.c_str() + 4);
            const bool more = l == 0 || l == 3 || l == 6 || l == 7;  // use_more_bits(l, 8)
            if (name.find("attn_v") != std::string::npos) {
                if (quant == "q4_k_m" && more) return Q6_K;
                if (quant == "q4_k_s" && l < 4) return Q5_K;
            }
            if (name.find("ffn_down") != std::string::npos) {
                if (quant == "q4_k_m" && more) return Q6_K;
                if (quant == "q4_k_s" && l < 1) return Q5_K;
            }
            return base_t;
        };
        auto bytes_of = [](uint32_t t, size_t n) -> size_t {
            return t == F32 ? n * 4 : t == Q4_K ? n / 256 * 144 : t == Q5_K ? n / 256 * 176 : t == Q6_K ? n / 256 * 210
                                                                                                  : n / 32 * block_bytes(t);
        };
        auto quant_of = [](uint32_t t) -> QuantFn {
            return t == Q4_K ? gguf::quantize_q4_k : t == Q5_K ? gguf::quantize_q5_k : t == Q6_K ? gguf::quantize_q6_k
                                                                                               : quantizer(t);
        };
        gguf::LlamaExportConfig c = cfg;
        c.embedding_length = W;
        c.feed_forward_length = WF;
        c.block_count = LW;
        c.head_count = 4;
        for (const std::string quant : {"q4_0", "q4_k_m", "q4_k_s"}) {
            c.quant = quant;
            const std::string path = tmp + "/gguf_q4_test_k_" + quant + ".gguf";
            gguf::export_gguf_llama(path, wide, c);
            const ParsedGGUF g = parse_gguf(path);
            CHECK((g.data_start % 32) == 0 && (g.bytes.size() % 32) == 0);
            std::map<uint32_t, int> count;
            for (size_t k = 0; k < g.tensors.size(); ++k) {
                const ParsedTensor& t = g.tensors[k];
                const Matrix& m = *wsrc.at(t.name);
                const size_t n = m.rows() * m.cols();
                const uint32_t wt = kt(quant, t.name);
                if (t.type != wt) std::fprintf(stderr, "%s %s: type %u, want %u\n", quant.c_str(), t.name.c_str(), t.type, wt);
                CHECK(t.type == wt);
                ++count[t.type];
                const size_t bytes = bytes_of(wt, n);
                const uint64_t end = (t.offset + bytes + 31) & ~uint64_t(31);
                CHECK(k + 1 < g.tensors.size() ? g.tensors[k + 1].offset == end
                                               : g.data_start + end == g.bytes.size());
                const uint8_t* at = g.bytes.data() + g.data_start + t.offset;
                if (wt == F32) {
                    CHECK(std::memcmp(at, &m(0, 0), bytes) == 0);
                } else {
                    std::vector<uint8_t> q(bytes);
                    quant_of(wt)(&m(0, 0), n, q.data());
                    CHECK(std::memcmp(at, q.data(), bytes) == 0);
                }
                for (size_t z = bytes; z < end - t.offset; ++z) CHECK(at[z] == 0);
            }
            std::remove(path.c_str());
            printf("  %-6s F32 %d, Q4_0 %d, Q4_K %d, Q5_K %d, Q6_K %d: types/offsets/bytes ok\n", quant.c_str(),
                   count[F32], count[Q4_0], count[Q4_K], count[Q5_K], count[Q6_K]);
        }
        // llama.cpp b6000's own quantize_row_q4_K_ref / q5_K_ref / q6_K_ref (ggml-base, linked by a
        // harness, 25 Oct 2026) on nine 512-value patterns: FNV-1a 64 of its bytes. The patterns use
        // exact float operations only (an int over 2^23, then one product), so every compiler
        // reproduces the inputs: zeros, a constant, small noise, all negative, all positive, two
        // spikes, 1e-6 noise (fp16-subnormal scales), seven magnitudes 1e-3..1e3, one large value
        // per 32.
        const uint64_t want_k[9][3] = {
            {0x32a0e76575209103ULL, 0x412c9745d2f7d403ULL, 0x3aacd66e1b043cb3ULL},
            {0x78581d3d146ac967ULL, 0x1c30f887c624bf93ULL, 0x261842a96e237997ULL},
            {0xf14c6db94f6d2bbfULL, 0x21ea962042c3d524ULL, 0xa5e8c386b9900b49ULL},
            {0x32fcab1a69b209fcULL, 0x6801e6ce1fa16113ULL, 0x352835165525cee3ULL},
            {0x11e77dceb534954bULL, 0x8200545ff5c8d801ULL, 0xcef17161dca14c73ULL},
            {0x4679f33e3f405766ULL, 0x7b496dd07df6a59cULL, 0x4890532fb22838faULL},
            {0x772d36e7961d19d6ULL, 0xd63f0cbb286afce7ULL, 0x75c4af55b11bc2eeULL},
            {0x83f72aebd59d7444ULL, 0xf829e25fa8f0e1d8ULL, 0x1dc0113a08cc0339ULL},
            {0x6f5bef5589e16202ULL, 0xd2afa311efa37f3eULL, 0x93cb0d493bf71bb4ULL}};
        auto fnv = [](const uint8_t* b, size_t n) {
            uint64_t h = 1469598103934665603ULL;
            for (size_t i = 0; i < n; ++i) {
                h ^= b[i];
                h *= 1099511628211ULL;
            }
            return h;
        };
        const float scale7[7] = {0.001f, 0.01f, 0.1f, 1.0f, 10.0f, 100.0f, 1000.0f};
        for (int p = 0; p < 9; ++p) {
            std::vector<float> x(512);
            uint32_t s = 12345u + 7919u * static_cast<uint32_t>(p);
            for (int i = 0; i < 512; ++i) {
                s = s * 1664525u + 1013904223u;
                const float r = static_cast<float>(static_cast<int32_t>(s >> 8) - 8388608) / 8388608.0f;
                switch (p) {
                    case 0: x[i] = 0.0f; break;
                    case 1: x[i] = 0.7f; break;
                    case 2: x[i] = r * 0.02f; break;
                    case 3: x[i] = (r < 0 ? r : -r) * 0.05f; break;
                    case 4: x[i] = (r > 0 ? r : -r) * 0.05f; break;
                    case 5: x[i] = i == 37 ? 5.0f : i == 300 ? -3.0f : 0.0f; break;
                    case 6: x[i] = r * 1e-6f; break;
                    case 7: x[i] = r * scale7[i % 7]; break;
                    default: x[i] = (i % 32 == 0) ? r * 4.0f : r * 0.01f; break;
                }
            }
            const QuantFn fns[3] = {gguf::quantize_q4_k, gguf::quantize_q5_k, gguf::quantize_q6_k};
            const size_t bb[3] = {144, 176, 210};
            for (int t = 0; t < 3; ++t) {
                std::vector<uint8_t> q(2 * bb[t]);
                fns[t](x.data(), x.size(), q.data());
                const uint64_t h = fnv(q.data(), q.size());
                if (h != want_k[p][t])
                    std::fprintf(stderr, "pattern %d type %d: fnv %016llx, llama.cpp %016llx\n", p, t,
                                 static_cast<unsigned long long>(h), static_cast<unsigned long long>(want_k[p][t]));
                CHECK(h == want_k[p][t]);
            }
        }
        printf("  Q4_K / Q5_K / Q6_K: 9 patterns x 2 super-blocks equal llama.cpp b6000's reference rows\n");
        for (QuantFn fn : {gguf::quantize_q4_k, gguf::quantize_q5_k, gguf::quantize_q6_k}) {
            bool threw = false;
            try {
                std::vector<float> x(288, 1.0f);
                std::vector<uint8_t> out(512);
                fn(x.data(), x.size(), out.data());
            } catch (const std::runtime_error&) {
                threw = true;
            }
            CHECK(threw);  // n % 256 != 0
        }
    }

    printf("\n[PASS] GGUF 4-bit export\n");
    return 0;
}
