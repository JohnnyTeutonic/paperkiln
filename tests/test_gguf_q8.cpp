// Q8_0 GGUF export test.
//
// 1. quantize_q8_0 on known blocks: zeros, an exact scale of 1 (max 127),
//    half-way products (rounded away from zero, as gguf-py's np_roundf and
//    ggml's roundf do, not to even), a negative maximum, a scale that fp16
//    must round, and a scale that is an fp16 subnormal. Expected bytes were
//    produced by gguf-py 0.19 (quants.quantize(..., Q8_0)), the quantiser
//    behind the ember session's quantize_gguf.py.
// 2. A tiny llama export under each quant setting, parsed back with an
//    independent spec reader: tensor types, unchanged ggml dims, byte sizes
//    (n / 32 * 34), 32-byte-aligned offsets, and the stored blocks equal to
//    quantize_q8_0 of the source.
// 3. A selected tensor whose ne0 is not a multiple of 32 is refused by name.
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

// One block of 32 floats: the listed leading values, zeros after.
std::vector<float> block(std::initializer_list<float> head) {
    std::vector<float> x(32, 0.0f);
    size_t i = 0;
    for (float v : head) x[i++] = v;
    return x;
}

// Quantise one block; check the fp16 scale bits and the leading int8 values
// (every later value must be 0).
void expect_block(const std::vector<float>& x, uint16_t scale, std::initializer_list<int> q) {
    uint8_t out[34];
    gguf::quantize_q8_0(x.data(), 32, out);
    uint16_t d;
    std::memcpy(&d, out, 2);
    if (d != scale) std::fprintf(stderr, "scale 0x%04x, want 0x%04x\n", d, scale);
    CHECK(d == scale);
    std::vector<int> want(q);
    want.resize(32, 0);
    for (int i = 0; i < 32; ++i) {
        const int got = static_cast<int8_t>(out[2 + i]);
        if (got != want[i]) std::fprintf(stderr, "q[%d] = %d, want %d\n", i, got, want[i]);
        CHECK(got == want[i]);
    }
}

// ---------- minimal spec-based GGUF reader (tensor infos only) ----------
struct ParsedTensor {
    std::string name;
    std::vector<uint64_t> dims;  // ne0 first, as stored
    uint32_t type = 0;
    uint64_t offset = 0;  // relative to the data section
};

struct ParsedGGUF {
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
        rd_str(g.bytes, p);
        skip_value(g.bytes, p, rd<uint32_t>(g.bytes, p));
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

// Smooth, sign-mixed fixture values, distinct per tensor.
Matrix filled(size_t r, size_t c, float base) {
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            m(i, j) = base * 0.01f + 0.003f * static_cast<float>(i) - 0.002f * static_cast<float>(j);
    return m;
}

}  // namespace

int main() {
    printf("=== quantize_q8_0 known blocks ===\n");
    expect_block(block({}), 0x0000, {});                      // zeros: d = 0, id = 0
    expect_block(block({-0.0f}), 0x0000, {});                 // negative zero too
    expect_block(block({127.0f, 0.5f, 1.5f, 2.5f, -0.5f, -2.5f, 0.49999997f}), 0x3c00,  // d = 1
                 {127, 1, 2, 3, -1, -3, 0});                  // halves away from zero
    expect_block(block({-63.5f, 10.25f, -0.25f, 63.5f}), 0x3800,  // negative max, d = 0.5
                 {-127, 21, -1, 127});
    expect_block(block({1.0f, 0.5f, -0.25f, 0.003937f}), 0x2008,  // d = 1/127, fp16-rounded
                 {127, 64, -32, 0});
    expect_block(block({127.0f * 0x1p-20f, 2.5f * 0x1p-20f, -3.0f * 0x1p-20f}), 0x0010,  // subnormal d
                 {127, 3, -3});
    {
        std::vector<float> two(64, 0.0f);  // blocks are independent: 34 bytes each
        two[0] = 127.0f;
        two[32] = -63.5f;
        uint8_t out[68];
        gguf::quantize_q8_0(two.data(), 64, out);
        CHECK(out[0] == 0x00 && out[1] == 0x3c && static_cast<int8_t>(out[2]) == 127);
        CHECK(out[34] == 0x00 && out[35] == 0x38 && static_cast<int8_t>(out[36]) == -127);
    }
    bool threw = false;
    try {
        std::vector<float> x(33, 1.0f);
        std::vector<uint8_t> out(68);
        gguf::quantize_q8_0(x.data(), x.size(), out.data());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    printf("  zeros, d = 1, half-way, negative max, rounded and subnormal fp16 scales ok\n");

    printf("=== tiny Q8_0 exports ===\n");
    const uint32_t L = 2, D = 32, FF = 64, V = 5, H = 2;
    std::map<std::string, Matrix> sd;
    sd.emplace("model.embed_tokens.weight", filled(V, D, 1.0f));
    for (uint32_t l = 0; l < L; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        sd.emplace(p + "self_attn.q_proj.weight", filled(D, D, 2.0f + l));
        sd.emplace(p + "self_attn.k_proj.weight", filled(D, D, 3.0f + l));
        sd.emplace(p + "self_attn.v_proj.weight", filled(D, D, 4.0f + l));
        sd.emplace(p + "self_attn.o_proj.weight", filled(D, D, 5.0f + l));
        sd.emplace(p + "input_layernorm.weight", filled(1, D, 6.0f + l));
        sd.emplace(p + "mlp.gate_proj.weight", filled(FF, D, 7.0f + l));
        sd.emplace(p + "mlp.up_proj.weight", filled(FF, D, 8.0f + l));
        sd.emplace(p + "mlp.down_proj.weight", filled(D, FF, 9.0f + l));
        sd.emplace(p + "post_attention_layernorm.weight", filled(1, D, 10.0f + l));
    }
    sd.emplace("model.norm.weight", filled(1, D, 11.0f));
    sd.emplace("lm_head.weight", filled(V, D, 12.0f));
    // GGUF name -> source matrix ([out][in] row-major: HF layout, no transpose).
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

    gguf::LlamaExportConfig cfg;
    cfg.embedding_length = D;
    cfg.block_count = L;
    cfg.feed_forward_length = FF;
    cfg.head_count = H;
    cfg.vocab_size = V;
    cfg.tokens = {"<unk>", "|", "a", "b", "c"};

    const char* tmpdir = std::getenv("TMPDIR");
#ifdef _WIN32
    if (!tmpdir) tmpdir = std::getenv("TEMP");
#endif
    const std::string base = tmpdir ? std::string(tmpdir) : std::string("/tmp");

    for (const std::string quant : {"q8_0", "q8_0_out", "q8_0_all"}) {
        cfg.quant = quant;
        const std::string path = base + "/gguf_q8_test_" + quant + ".gguf";
        gguf::export_gguf_llama(path, sd, cfg);
        const ParsedGGUF g = parse_gguf(path);
        CHECK(g.tensors.size() == 9 * L + 3);
        CHECK((g.data_start % 32) == 0);
        size_t n_q8 = 0;
        for (size_t k = 0; k < g.tensors.size(); ++k) {
            const ParsedTensor& t = g.tensors[k];
            const Matrix& m = *src.at(t.name);
            const size_t n = m.rows() * m.cols();
            const bool want_q8 = t.dims.size() == 2 &&
                                 (t.name.rfind("blk.", 0) == 0 || (quant != "q8_0" && t.name == "output.weight") ||
                                  (quant == "q8_0_all" && t.name == "token_embd.weight"));
            CHECK(t.type == (want_q8 ? 8u : 0u));
            if (m.rows() > 1) CHECK(t.dims == std::vector<uint64_t>({m.cols(), m.rows()}));  // ne = {in, out}
            CHECK((t.offset % 32) == 0);
            const size_t bytes = want_q8 ? n / 32 * 34 : n * sizeof(float);
            if (k + 1 < g.tensors.size()) {  // the next offset is this tensor's end, aligned up
                CHECK(g.tensors[k + 1].offset == ((t.offset + bytes + 31) & ~uint64_t(31)));
            }
            const uint8_t* at = g.bytes.data() + g.data_start + t.offset;
            if (want_q8) {
                std::vector<uint8_t> q(bytes);
                gguf::quantize_q8_0(&m(0, 0), n, q.data());
                CHECK(std::memcmp(at, q.data(), bytes) == 0);
                ++n_q8;
            } else {
                CHECK(std::memcmp(at, &m(0, 0), bytes) == 0);
            }
            CHECK(g.data_start + t.offset + bytes <= g.bytes.size());
        }
        CHECK(n_q8 == 7 * L + (quant != "q8_0") + (quant == "q8_0_all"));
        std::remove(path.c_str());
        printf("  %-8s %zu tensors Q8_0, dims/sizes/offsets/bytes ok\n", quant.c_str(), n_q8);
    }

    // ne0 = FF = 40 for ffn_down: refused, naming the tensor.
    {
        std::map<std::string, Matrix> bad(sd);
        for (uint32_t l = 0; l < L; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".mlp.";
            bad[p + "gate_proj.weight"] = filled(40, D, 7.0f);
            bad[p + "up_proj.weight"] = filled(40, D, 8.0f);
            bad[p + "down_proj.weight"] = filled(D, 40, 9.0f);
        }
        gguf::LlamaExportConfig c2 = cfg;
        c2.quant = "q8_0";
        c2.feed_forward_length = 40;
        std::string msg;
        try {
            gguf::export_gguf_llama(base + "/gguf_q8_test_bad.gguf", bad, c2);
        } catch (const std::runtime_error& e) {
            msg = e.what();
        }
        CHECK(msg.find("blk.0.ffn_down.weight") != std::string::npos);
        c2.quant = "q4_1";
        msg.clear();
        try {
            gguf::export_gguf_llama(base + "/gguf_q8_test_bad.gguf", sd, c2);
        } catch (const std::runtime_error& e) {
            msg = e.what();
        }
        CHECK(msg.find("quant must be") != std::string::npos);
        std::remove((base + "/gguf_q8_test_bad.gguf").c_str());
        printf("  ne0 %% 32 != 0 refused by name; unknown quant refused\n");
    }

    printf("\n[PASS] GGUF Q8_0 export\n");
    return 0;
}
