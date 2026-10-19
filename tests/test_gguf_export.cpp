// GGUF exporter round-trip test.
//
// Exports a tiny Llama-shaped state_dict, then re-parses the file with an
// INDEPENDENT minimal GGUF v3 reader (written from the spec, not from the
// writer) and verifies:
//   1. magic/version/counts,
//   2. metadata survives (spot-checked keys),
//   3. the data section starts at align_up(header_end, 32) and every
//      tensor's bytes match the source matrices exactly.
// Check 3 is the regression gate for the 2026-07-13 alignment bug: if the
// start-of-section padding were missing again, every byte comparison
// would fail with a constant shift.
//
// Dimensions are chosen so tensor sizes are NOT multiples of 32 bytes
// (d=6, ff=10, vocab=7), forcing both start-of-section and inter-tensor
// padding to be exercised.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "microtorch/gguf.hpp"

#include "check.hpp"  // Release-proof CHECK (assert vanishes under -DNDEBUG)

using namespace microtorch;

namespace {

// ---------- minimal spec-based GGUF reader ----------
struct ParsedTensor {
    std::string name;
    std::vector<uint64_t> dims;  // as stored in file
    uint32_t type;
    uint64_t offset;  // relative to data-section start
};

struct ParsedGGUF {
    std::map<std::string, uint32_t> u32s;
    std::map<std::string, float> f32s;
    std::map<std::string, std::string> strings;
    std::vector<std::string> token_array;
    std::vector<ParsedTensor> tensors;
    uint64_t data_start = 0;     // absolute file offset
    std::vector<uint8_t> bytes;  // whole file
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
        switch (vt) {
            case 4:
                g.u32s[key] = rd<uint32_t>(g.bytes, p);
                break;  // UINT32
            case 6:
                g.f32s[key] = rd<float>(g.bytes, p);
                break;  // FLOAT32
            case 8:
                g.strings[key] = rd_str(g.bytes, p);
                break;  // STRING
            case 9: {   // ARRAY
                const uint32_t et = rd<uint32_t>(g.bytes, p);
                const uint64_t n = rd<uint64_t>(g.bytes, p);
                for (uint64_t k = 0; k < n; ++k) {
                    if (et == 8) {
                        std::string tok = rd_str(g.bytes, p);
                        if (key == "tokenizer.ggml.tokens") g.token_array.push_back(tok);
                    } else if (et == 6) {
                        rd<float>(g.bytes, p);
                    } else {
                        CHECK(false && "unexpected array element type");
                    }
                }
                break;
            }
            default:
                CHECK(false && "unexpected metadata value type");
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

    // Spec: data section begins at the next multiple of general.alignment.
    const uint64_t align = 32;
    g.data_start = (p + align - 1) & ~(align - 1);
    return g;
}

// ---------- fixture ----------
Matrix filled(size_t r, size_t c, float base) {
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            m(i, j) = base + 0.25f * static_cast<float>(i) - 0.125f * static_cast<float>(j);
    return m;
}

void expect_bytes(const ParsedGGUF& g, const std::string& name, const Matrix& src) {
    for (const auto& t : g.tensors) {
        if (t.name != name) continue;
        const size_t n = src.rows() * src.cols();
        uint64_t count = 1;
        for (auto d : t.dims) count *= d;
        CHECK(count == n);
        // src is [out][in] row-major, so ggml order is ne = {in, out}: the
        // order llama.cpp checks (ne0 is the contiguous axis).
        if (t.dims.size() == 2)
            CHECK(t.dims == std::vector<uint64_t>({src.cols(), src.rows()}));
        CHECK(t.type == 0);           // F32
        CHECK((t.offset % 32) == 0);  // every offset aligned
        const uint8_t* file_bytes = g.bytes.data() + g.data_start + t.offset;
        CHECK(std::memcmp(file_bytes, &src(0, 0), n * sizeof(float)) == 0);
        return;
    }
    CHECK(false && "tensor not found in file");
}

}  // namespace

int main() {
    // d=6 (head_dim 3), ff=10, vocab=7: norm tensors are 24 bytes and the
    // embedding 168 bytes, so nothing lands on a 32-byte boundary by luck.
    const uint32_t L = 2, D = 6, FF = 10, V = 7, H = 2;

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
    // A tensor the exporter must NOT silently ship:
    sd.emplace("model.layers.0.self_attn.q_proj.bias", filled(1, D, 13.0f));

    gguf::LlamaExportConfig cfg;
    cfg.embedding_length = D;
    cfg.block_count = L;
    cfg.feed_forward_length = FF;
    cfg.head_count = H;
    cfg.vocab_size = V;
    cfg.tokens = {"<unk>", "|", "hello", "world", "foo", "bar", "baz"};

    // Write outside the source tree: a checkout under OneDrive/Dropbox has
    // the sync client racing us for the freshly written file handle.
    const char* tmpdir = std::getenv("TMPDIR");
#ifdef _WIN32
    if (!tmpdir) tmpdir = std::getenv("TEMP");
#endif
    const std::string base = tmpdir ? std::string(tmpdir) : std::string("/tmp");
    const std::string path = base + "/gguf_export_test.gguf";
    std::vector<std::string> unmapped;
    gguf::export_gguf_llama(path, sd, cfg, &unmapped);

    printf("=== parse back ===\n");
    ParsedGGUF g = parse_gguf(path);

    // Metadata survived.
    CHECK(g.strings.at("general.architecture") == "llama");
    CHECK(g.u32s.at("llama.embedding_length") == D);
    CHECK(g.u32s.at("llama.block_count") == L);
    CHECK(g.u32s.at("llama.attention.head_count_kv") == H);  // MHA default
    CHECK(g.u32s.at("llama.rope.dimension_count") == D / H);
    CHECK(g.u32s.at("llama.vocab_size") == V);
    CHECK(g.token_array.size() == 7 && g.token_array[2] == "hello");
    printf("  metadata ok (%zu u32 keys, %zu tokens)\n", g.u32s.size(), g.token_array.size());

    // 9 per layer * 2 + embed + final norm + lm_head = 21 tensors.
    CHECK(g.tensors.size() == 21);
    CHECK((g.data_start % 32) == 0);
    printf("  21 tensors, data section aligned at %llu\n",
           static_cast<unsigned long long>(g.data_start));

    // Byte-exact round trip for every mapped tensor (HF layout: no
    // transpose, so file bytes == source bytes).
    expect_bytes(g, "token_embd.weight", sd.at("model.embed_tokens.weight"));
    expect_bytes(g, "blk.0.attn_q.weight", sd.at("model.layers.0.self_attn.q_proj.weight"));
    expect_bytes(g, "blk.1.ffn_down.weight", sd.at("model.layers.1.mlp.down_proj.weight"));
    expect_bytes(g, "blk.1.ffn_norm.weight",
                 sd.at("model.layers.1.post_attention_layernorm.weight"));
    expect_bytes(g, "output_norm.weight", sd.at("model.norm.weight"));
    expect_bytes(g, "output.weight", sd.at("lm_head.weight"));
    printf("  tensor bytes identical through align-padded data section\n");

    // The bias must be reported, not silently dropped.
    CHECK(unmapped.size() == 1 && unmapped[0] == "model.layers.0.self_attn.q_proj.bias");
    printf("  unmapped tensor reported: %s\n", unmapped[0].c_str());

    // Transpose path: microtorch-layout [in, out] weights come out as
    // llama [out][in] bytes.
    {
        std::map<std::string, Matrix> sd2(sd);
        sd2.erase("model.layers.0.self_attn.q_proj.bias");
        gguf::LlamaExportConfig cfg2 = cfg;
        cfg2.weights_in_out = true;
        const std::string path2 = base + "/gguf_export_test_T.gguf";
        gguf::export_gguf_llama(path2, sd2, cfg2);
        ParsedGGUF g2 = parse_gguf(path2);

        const Matrix& q = sd.at("model.layers.0.self_attn.q_proj.weight");
        Matrix qT(q.cols(), q.rows());
        for (size_t i = 0; i < q.rows(); ++i)
            for (size_t j = 0; j < q.cols(); ++j) qT(j, i) = q(i, j);
        expect_bytes(g2, "blk.0.attn_q.weight", qT);
        std::remove(path2.c_str());
        printf("  weights_in_out=true transposes correctly\n");
    }

    // Mixture of experts (nn::MoEFFN names, microtorch [in, out] layout):
    // router -> ffn_gate_inp {d, E}; experts stacked into 3-D *_exps
    // tensors {in, out, E}, expert e's [out][in] block at e * in * out.
    {
        const uint32_t E = 3, K = 2;
        std::map<std::string, Matrix> sd3;
        for (const auto& [k, m] : sd) {
            if (k.find("mlp.") != std::string::npos || k.find("q_proj.bias") != std::string::npos) continue;
            sd3.emplace(k, m);
        }
        for (uint32_t l = 0; l < L; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".mlp.";
            sd3.emplace(p + "router.weight", filled(D, E, 20.0f + l));  // [in=d, out=E]
            for (uint32_t e = 0; e < E; ++e) {
                const std::string q = p + "experts." + std::to_string(e) + ".";
                sd3.emplace(q + "gate_proj.weight", filled(D, FF, 30.0f + 10 * l + e));
                sd3.emplace(q + "up_proj.weight", filled(D, FF, 60.0f + 10 * l + e));
                sd3.emplace(q + "down_proj.weight", filled(FF, D, 90.0f + 10 * l + e));
            }
        }
        // lm_head in [in, out] layout for this export.
        sd3["lm_head.weight"] = filled(D, V, 12.0f);
        gguf::LlamaExportConfig cfg3 = cfg;
        cfg3.weights_in_out = true;
        cfg3.expert_count = E;
        cfg3.expert_used_count = K;
        const std::string path3 = base + "/gguf_export_test_moe.gguf";
        std::vector<std::string> un3;
        gguf::export_gguf_llama(path3, sd3, cfg3, &un3);
        ParsedGGUF g3 = parse_gguf(path3);
        CHECK(un3.empty());
        CHECK(g3.u32s.at("llama.expert_count") == E && g3.u32s.at("llama.expert_used_count") == K);
        // per layer: 4 attn + 2 norms + router + 3 exps = 10; + embed, norm, head.
        CHECK(g3.tensors.size() == 10 * L + 3);
        auto transpose = [](const Matrix& w) {
            Matrix t(w.cols(), w.rows());
            for (size_t i = 0; i < w.rows(); ++i)
                for (size_t j = 0; j < w.cols(); ++j) t(j, i) = w(i, j);
            return t;
        };
        expect_bytes(g3, "blk.1.ffn_gate_inp.weight", transpose(sd3.at("model.layers.1.mlp.router.weight")));
        for (const auto& t : g3.tensors) {
            if (t.name.find("_exps") != std::string::npos || t.name.find("gate_inp") != std::string::npos) {
                printf("  %-26s ne = {", t.name.c_str());
                for (auto d : t.dims) printf(" %llu", static_cast<unsigned long long>(d));
                printf(" }\n");
            }
            if (t.name == "blk.1.ffn_gate_inp.weight") CHECK(t.dims == std::vector<uint64_t>({D, E}));
            if (t.name == "blk.0.ffn_up_exps.weight") CHECK(t.dims == std::vector<uint64_t>({D, FF, E}));
            if (t.name == "blk.0.ffn_down_exps.weight") CHECK(t.dims == std::vector<uint64_t>({FF, D, E}));
            if (t.name != "blk.1.ffn_down_exps.weight") continue;
            CHECK(t.dims == std::vector<uint64_t>({FF, D, E}));
            for (uint32_t e = 0; e < E; ++e) {
                const Matrix want = transpose(sd3.at("model.layers.1.mlp.experts." + std::to_string(e) + ".down_proj.weight"));
                const uint8_t* at = g3.bytes.data() + g3.data_start + t.offset + size_t(e) * D * FF * sizeof(float);
                CHECK(std::memcmp(at, &want(0, 0), D * FF * sizeof(float)) == 0);
            }
        }
        std::remove(path3.c_str());
        printf("  MoE export: expert_count/used metadata, router {d,E}, 3-D *_exps tensors byte-exact per expert\n");
    }

    std::remove(path.c_str());
    printf("\n[PASS] GGUF export round trip\n");
    return 0;
}
