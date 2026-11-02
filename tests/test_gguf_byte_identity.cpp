// GGUF byte identity: every export_gguf_* variant, hashed.
//
// Each case below writes one GGUF from a fixed, platform-independent fixture
// (every weight an exact dyadic float from an integer hash) and compares the
// file's SHA-256 with the value recorded here. The hashes were recorded from
// the exporters as they stood before the byte writer moved to coalfire.cpp's
// shared header (third_party/transformer_cpp/include/gguf_writer.hpp, ROADMAP
// C3, 3 Nov 2026), so a pass means every file is the bytes it was before.
// A mismatch is a change in what paperkiln writes: fix the writer, never the
// table, unless the change to the file format is the intended one.
//
// The cases cover the three entry points (llama, gpt2, paperkiln) with F32,
// every quant variant (q8_0, q8_0_out, q8_0_all, q4_0, q4_k_m, q4_k_s, the
// 4-bit ones both through their Q5_0 / Q5_1 / Q8_0 fallbacks and in real
// k-quant blocks), word and BPE vocabularies, chat templates, the IR (with
// and without a digest), MoE, state tokens, extra tensors, q/k norms, GQA and
// tied heads.
//
//   test_gguf_byte_identity          check every case
//   test_gguf_byte_identity --print  print "name sha256" for every case
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "microtorch/gguf.hpp"
#include "microtorch/ir.hpp"

#include "check.hpp"

using namespace microtorch;

namespace {

// An exact float in [-0.5, 0.5): a 24-bit integer hash over 2^24.
float val(uint32_t seed, size_t i) {
    uint32_t h = seed * 0x9E3779B1u ^ static_cast<uint32_t>(i) * 0x85EBCA77u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return static_cast<float>(static_cast<int32_t>(h >> 8) - (1 << 23)) /
           static_cast<float>(1 << 24);
}

uint32_t g_seed = 1;
Matrix filled(size_t r, size_t c) {
    Matrix m(r, c);
    const uint32_t s = g_seed++;
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = val(s, i * c + j);
    return m;
}

std::string tmp_dir() {
    const char* t = std::getenv("TMPDIR");
#ifdef _WIN32
    if (!t) t = std::getenv("TEMP");
#endif
    return t ? std::string(t) : std::string("/tmp");
}

std::string file_sha256(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    CHECK(f && "cannot open exported gguf");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ir::sha256_hex(ss.str());
}

std::vector<std::string> word_tokens(uint32_t v) {
    std::vector<std::string> t = {"<unk>", "|"};
    for (uint32_t i = 2; i < v; ++i) t.push_back("w" + std::to_string(i));
    return t;
}

// A llama.cpp-form BPE vocabulary: byte-ish tokens, merges, token types.
template <class Cfg>
void bpe_tokens(Cfg& c, uint32_t v) {
    c.tokens.clear();
    c.token_type.clear();
    for (uint32_t i = 0; i < v; ++i) {
        c.tokens.push_back(i < 3 ? std::string(i == 0   ? "<|endoftext|>"
                                               : i == 1 ? "<|user|>"
                                                        : "<|assistant|>")
                                 : std::string(1, static_cast<char>('a' + i % 26)) +
                                       std::to_string(i));
        c.token_type.push_back(i < 3 ? 3 : 1);
    }
    c.merges = {"a3 b4", "c5 d6", "a3b4 c5d6"};
    c.tokenizer_model = "gpt2";
    c.tokenizer_pre = "llama-bpe";
    c.bos_token_id = 0;
    c.eos_token_id = 0;
    c.unk_token_id = 0;
    c.pad_token_id = 2;
}

template <class Cfg>
void chat(Cfg& c) {
    c.chat_template = R"({"name":"fixture","user":"<|user|>","assistant":"<|assistant|>"})";
    c.chat_template_jinja = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}";
}

// An IR document with a digest, and one whose negative zero has none.
const char* kIR =
    R"({"ir_version":1,"family":"llama","_note":"fixture","graph":[{"op":"embed","out":"x"}]})";
const char* kIRNoDigest = R"({"ir_version":1,"family":"llama","eps":-0.0})";

// ---- llama fixtures ----
struct LlamaDims {
    uint32_t L, D, FF, V, H, KV;
};

std::map<std::string, Matrix> llama_sd(const LlamaDims& d, bool in_out, bool head,
                                       bool qk_norm = false) {
    const uint32_t hd = d.D / d.H, kvw = d.KV * hd;
    auto proj = [&](uint32_t out, uint32_t in) {
        return in_out ? filled(in, out) : filled(out, in);
    };
    std::map<std::string, Matrix> sd;
    sd.emplace("model.embed_tokens.weight", filled(d.V, d.D));
    for (uint32_t l = 0; l < d.L; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        sd.emplace(p + "self_attn.q_proj.weight", proj(d.D, d.D));
        Matrix k = proj(kvw, d.D);
        if (l == 0)  // a block of zeros: the quantisers' d == 0 path
            for (size_t j = 0; j < 32 && j < k.cols(); ++j) k(0, j) = 0.0f;
        sd.emplace(p + "self_attn.k_proj.weight", std::move(k));
        sd.emplace(p + "self_attn.v_proj.weight", proj(kvw, d.D));
        sd.emplace(p + "self_attn.o_proj.weight", proj(d.D, d.D));
        if (qk_norm) {
            sd.emplace(p + "self_attn.q_norm.weight", filled(1, hd));
            sd.emplace(p + "self_attn.k_norm.weight", filled(1, hd));
        }
        sd.emplace(p + "input_layernorm.weight", filled(1, d.D));
        sd.emplace(p + "mlp.gate_proj.weight", proj(d.FF, d.D));
        sd.emplace(p + "mlp.up_proj.weight", proj(d.FF, d.D));
        sd.emplace(p + "mlp.down_proj.weight", proj(d.D, d.FF));
        sd.emplace(p + "post_attention_layernorm.weight", filled(1, d.D));
    }
    sd.emplace("model.norm.weight", filled(1, d.D));
    if (head) sd.emplace("lm_head.weight", proj(d.V, d.D));
    return sd;
}

gguf::LlamaExportConfig llama_cfg(const LlamaDims& d) {
    gguf::LlamaExportConfig c;
    c.embedding_length = d.D;
    c.block_count = d.L;
    c.feed_forward_length = d.FF;
    c.head_count = d.H;
    c.head_count_kv = d.KV == d.H ? 0 : d.KV;
    c.vocab_size = d.V;
    c.tokens = word_tokens(d.V);
    return c;
}

// ---- gpt2 fixture ----
std::map<std::string, Matrix> gpt2_sd(uint32_t L, uint32_t D, uint32_t FF, uint32_t V, uint32_t ctx,
                                      bool head) {
    std::map<std::string, Matrix> sd;
    sd.emplace("transformer.wte.weight", filled(V, D));
    sd.emplace("wpe.weight", filled(ctx, D));
    for (uint32_t l = 0; l < L; ++l) {
        const std::string p = (l == 0 ? "transformer.h." : "h.") + std::to_string(l) + ".";
        sd.emplace(p + "ln_1.weight", filled(1, D));
        sd.emplace(p + "ln_1.bias", filled(1, D));
        sd.emplace(p + "attn.c_attn.weight", filled(D, 3 * D));
        sd.emplace(p + "attn.c_attn.bias", filled(1, 3 * D));
        sd.emplace(p + "attn.c_proj.weight", filled(D, D));
        sd.emplace(p + "attn.c_proj.bias", filled(1, D));
        sd.emplace(p + "ln_2.weight", filled(1, D));
        sd.emplace(p + "ln_2.bias", filled(1, D));
        sd.emplace(p + "mlp.c_fc.weight", filled(D, FF));
        sd.emplace(p + "mlp.c_fc.bias", filled(1, FF));
        sd.emplace(p + "mlp.c_proj.weight", filled(FF, D));
        sd.emplace(p + "mlp.c_proj.bias", filled(1, D));
    }
    sd.emplace("ln_f.weight", filled(1, D));
    sd.emplace("ln_f.bias", filled(1, D));
    if (head) sd.emplace("lm_head.weight", filled(D, V));
    return sd;
}

gguf::Gpt2ExportConfig gpt2_cfg(uint32_t L, uint32_t D, uint32_t FF, uint32_t V, uint32_t ctx,
                                uint32_t H) {
    gguf::Gpt2ExportConfig c;
    c.context_length = ctx;
    c.embedding_length = D;
    c.block_count = L;
    c.feed_forward_length = FF;
    c.head_count = H;
    c.vocab_size = V;
    c.tokens = word_tokens(V);
    return c;
}

// The recorded SHA-256 of every case (see the header comment).
const std::map<std::string, std::string> kWant = {
    {"llama_f32_word", "d6e5a5d0e43200e9ead817f5fc53e4cc036ab93eaa318e509afc954992167ec6"},
    {"llama_f32_notok", "692f55d3aaffe66e664c43b79ad93163cfad05b73ba52b8e8ec4db37c6f0adb4"},
    {"llama_f32_in_out_options",
     "0ef7ebbeb17e727f9a767752623641048cdced314781996b667f377f9f492ff0"},
    {"llama_f32_gqa_tied_qknorm",
     "14a905282ffe8a5aa785027ea054d5dec8476ed3c326c83cbe49ea42ba39721a"},
    {"llama_f32_bpe_chat_ir", "6cfc437767f7af9494610566c268acb14d42aa629a9cb212e74a6064bee34811"},
    {"llama_f32_ir_nodigest", "6b82475ba0d1a723989677c82fffa21d58e81ed3c38bb0097e8bfbc0bae4200c"},
    {"llama_q8_0_word", "b9a52bee5c5f120397324c85d270f1ce9b2e8267c8c858963c65aaf7fec0ccae"},
    {"llama_q8_0_bpe_chat_ir", "c2f2965d96b0ad2e39ccf5525b481c857dd96206c4e99feb5ecc13637a38a70d"},
    {"llama_q8_0_out_word", "63598e4935fb47e50fa98f91e1439da1b2ad7fcbb8eb9cc4f1cf134031c5e362"},
    {"llama_q8_0_out_bpe_chat_ir",
     "d477176c186d7e44e7ab2534cb3ffaa97651b3fed0f0957a18d2ad6c134c3ac8"},
    {"llama_q8_0_all_word", "570e70cf7b6c86dc95fdadcd94f035b8d534d31286747e8f8cc292440d0118d4"},
    {"llama_q8_0_all_bpe_chat_ir",
     "ae34ae224aebe79fcb35c0580cad1f27d8f0bd99acde3b1c53b6ca687afb9b34"},
    {"llama_q8_0_all_gqa_tied", "da7437ade2f8d4aa9d48aa82c3e789ef7799b9cef4cba334fbfe5490bf1715ef"},
    {"llama_q4_0_fallback_word",
     "8c0b76a011e4a18a51d602a29854ed1b267626ce4b1221fe50800e599c131717"},
    {"llama_q4_0_fallback_tied_bpe_chat_ir",
     "9b201920f0f53648215efa07363c3fa09194bf1beb5359157c666f2dbfed8694"},
    {"llama_q4_0_kquant_ir", "f174a0cffdf31cc1b892aa42cf38325d53bcc6e95a712010c8e73e7f2081d1d1"},
    {"llama_q4_0_kquant_tied", "ac6b91e2883bbd73b39673ae489e80d2015120a05cd5a3c69aa9b529db75a345"},
    {"llama_q4_k_m_fallback_word",
     "051c6d4063798d8c8b66ee76723fbf2b5aad24bddbc534c074f877b1ebd61003"},
    {"llama_q4_k_m_fallback_tied_bpe_chat_ir",
     "12c221a9c0db866e5e6e5f39d7a93a18ddb5289b7de092085b1002e126db361a"},
    {"llama_q4_k_m_kquant_ir", "3b263061f81660c72e281114bc2ab29c4da803a440dfe66f25f4df655b8464e3"},
    {"llama_q4_k_m_kquant_tied",
     "6ff0d3883f2875d3074caba4a07dea8c61cbca32c707c70ddfbba39dd445d270"},
    {"llama_q4_k_s_fallback_word",
     "b6a689fd7d6e535d004b5e4781ec678a2c1a167151696851716fb314e0863872"},
    {"llama_q4_k_s_fallback_tied_bpe_chat_ir",
     "9d48c6ab8ad599709c52d2ce98582d827f8988280473b7c5d1a004c996e81375"},
    {"llama_q4_k_s_kquant_ir", "2d5f96a418002a777d9ca35b6928019dcc93df97049f0d83ab395cf03a1948ef"},
    {"llama_q4_k_s_kquant_tied",
     "bd90944659c274e3078ce8dba361c6c1ca3bbb987a50e92752415497a94aab59"},
    {"llama_moe_f32", "b0f978fec4f7f1883d641f5bf23872e4d84c572c2cfa9c8ae6ccde64a23f99ae"},
    {"llama_moe_q8_0_out", "58cfe9f0a1a62c5a604e173227ca89007c34c80197d991b45fcb661efcfc9e1f"},
    {"llama_state_extra_f32", "4668b58d64e0837d529762646879ebfe3dfff15abfd9aad1f27147b4d0b23003"},
    {"llama_state_extra_q8_0_all",
     "c5e21063a3b5c79c22ae6d66c589f36432b133420d8047a5acfcb02f2ebb5e4b"},
    {"gpt2_f32_word", "cff4ef89f0a889e7079f480505b715fc36cc703081e37095662a6d42444e1cff"},
    {"gpt2_f32_tied_bpe_chat_ir",
     "f15a8606bddbfdbaa97a44616c2f301938245e9788639f186e111397a54329ca"},
    {"gpt2_arch_paperkiln_ir", "48f194e3bd8c5fcebd79b850b0444642508dbd95da5bed6b9cd5b901a9bc5a8f"},
    {"paperkiln_word_ir", "5bfb8bf530b5df1b020af0e72d0dd7ef03118248567a3a1c7350902ce9fc2ae8"},
    {"paperkiln_heads_bpe_chat_ir",
     "738e0d38f3cba7631e47c71835593d0d303c49263bfd602ecf2e571d3ca5b3a8"},
    {"paperkiln_notok_ir_nodigest",
     "0f4e840a87a4c610565f936bb0695a74845ae86f6976fd05f836ef77ee07d126"},
};

}  // namespace

int main(int argc, char** argv) {
    const bool print = argc > 1 && std::string(argv[1]) == "--print";
    const std::string tmp = tmp_dir();
    std::vector<std::pair<std::string, std::function<void(const std::string&)>>> cases;

    // ---------------- llama ----------------
    const LlamaDims S{2, 64, 96, 40, 4, 4};    // ne0 64 / 96: the 4-bit variants' fallbacks
    const LlamaDims G{2, 64, 96, 40, 4, 2};    // GQA
    const LlamaDims K{8, 256, 512, 64, 4, 4};  // ne0 256 / 512: real k-quant blocks

    g_seed = 100;
    const auto sd_s = llama_sd(S, false, true);
    g_seed = 200;
    const auto sd_s_io = llama_sd(S, true, true);
    g_seed = 300;
    const auto sd_g_tied = llama_sd(G, true, false, true);
    g_seed = 400;
    const auto sd_k = llama_sd(K, false, true);
    g_seed = 500;
    const auto sd_k_tied = llama_sd(K, false, false);

    cases.push_back({"llama_f32_word", [&](const std::string& p) {
                         gguf::export_gguf_llama(p, sd_s, llama_cfg(S));
                     }});
    cases.push_back({"llama_f32_notok", [&](const std::string& p) {
                         auto c = llama_cfg(S);
                         c.tokens.clear();
                         gguf::export_gguf_llama(p, sd_s, c);
                     }});
    cases.push_back({"llama_f32_in_out_options", [&](const std::string& p) {
                         auto c = llama_cfg(S);
                         c.weights_in_out = true;
                         c.name = "fixture-in-out";
                         c.context_length = 128;
                         c.rms_eps = 1e-5f;
                         c.rope_freq_base = 500000.0f;
                         c.rope_dimension_count = 8;
                         c.train_precision = "bf16";
                         c.bos_token_id = 1;
                         c.eos_token_id = 1;
                         c.unk_token_id = 0;
                         c.pad_token_id = 0;
                         gguf::export_gguf_llama(p, sd_s_io, c);
                     }});
    cases.push_back({"llama_f32_gqa_tied_qknorm", [&](const std::string& p) {
                         auto c = llama_cfg(G);
                         c.weights_in_out = true;
                         gguf::export_gguf_llama(p, sd_g_tied, c);
                     }});
    cases.push_back({"llama_f32_bpe_chat_ir", [&](const std::string& p) {
                         auto c = llama_cfg(S);
                         bpe_tokens(c, S.V);
                         chat(c);
                         c.ir = kIR;
                         gguf::export_gguf_llama(p, sd_s, c);
                     }});
    cases.push_back({"llama_f32_ir_nodigest", [&](const std::string& p) {
                         auto c = llama_cfg(S);
                         c.ir = kIRNoDigest;
                         gguf::export_gguf_llama(p, sd_s, c);
                     }});
    for (const std::string q : {"q8_0", "q8_0_out", "q8_0_all"}) {
        cases.push_back({"llama_" + q + "_word", [&, q](const std::string& p) {
                             auto c = llama_cfg(S);
                             c.quant = q;
                             c.train_precision = "bf16";
                             gguf::export_gguf_llama(p, sd_s, c);
                         }});
        cases.push_back({"llama_" + q + "_bpe_chat_ir", [&, q](const std::string& p) {
                             auto c = llama_cfg(S);
                             bpe_tokens(c, S.V);
                             chat(c);
                             c.ir = kIR;
                             c.quant = q;
                             gguf::export_gguf_llama(p, sd_s, c);
                         }});
    }
    cases.push_back({"llama_q8_0_all_gqa_tied", [&](const std::string& p) {
                         auto c = llama_cfg(G);
                         c.weights_in_out = true;
                         c.quant = "q8_0_all";
                         gguf::export_gguf_llama(p, sd_g_tied, c);
                     }});
    for (const std::string q : {"q4_0", "q4_k_m", "q4_k_s"}) {
        cases.push_back({"llama_" + q + "_fallback_word", [&, q](const std::string& p) {
                             auto c = llama_cfg(S);
                             c.quant = q;
                             c.train_precision = "bf16";  // never written into the 4-bit variants
                             gguf::export_gguf_llama(p, sd_s, c);
                         }});
        cases.push_back({"llama_" + q + "_fallback_tied_bpe_chat_ir", [&, q](const std::string& p) {
                             auto c = llama_cfg(G);
                             c.weights_in_out = true;
                             bpe_tokens(c, G.V);
                             chat(c);
                             c.ir = kIR;
                             c.quant = q;
                             gguf::export_gguf_llama(p, sd_g_tied, c);
                         }});
        cases.push_back({"llama_" + q + "_kquant_ir", [&, q](const std::string& p) {
                             auto c = llama_cfg(K);
                             c.ir = kIR;
                             c.quant = q;
                             gguf::export_gguf_llama(p, sd_k, c);
                         }});
        cases.push_back({"llama_" + q + "_kquant_tied", [&, q](const std::string& p) {
                             auto c = llama_cfg(K);
                             c.quant = q;
                             gguf::export_gguf_llama(p, sd_k_tied, c);
                         }});
    }

    // MoE (microtorch layout) in F32 and Q8_0 (the router is a 2-D blk.* pick).
    g_seed = 600;
    std::map<std::string, Matrix> sd_moe;
    {
        const uint32_t E = 3;
        for (const auto& [k, m] : llama_sd(S, true, true))
            if (k.find("mlp.") == std::string::npos) sd_moe.emplace(k, m);
        for (uint32_t l = 0; l < S.L; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".mlp.";
            sd_moe.emplace(p + "router.weight", filled(S.D, E));
            for (uint32_t e = 0; e < E; ++e) {
                const std::string q = p + "experts." + std::to_string(e) + ".";
                sd_moe.emplace(q + "gate_proj.weight", filled(S.D, S.FF));
                sd_moe.emplace(q + "up_proj.weight", filled(S.D, S.FF));
                sd_moe.emplace(q + "down_proj.weight", filled(S.FF, S.D));
            }
        }
    }
    for (const std::string q : {"f32", "q8_0_out"}) {
        cases.push_back({"llama_moe_" + q, [&, q](const std::string& p) {
                             auto c = llama_cfg(S);
                             c.weights_in_out = true;
                             c.expert_count = 3;
                             c.expert_used_count = 2;
                             c.quant = q;
                             gguf::export_gguf_llama(p, sd_moe, c);
                         }});
    }

    // State tokens, extra projections and vectors (F32 in every variant).
    g_seed = 700;
    auto sd_extra = llama_sd(S, true, true);
    sd_extra.emplace("state_embd.weight", filled(3, S.D));
    sd_extra.emplace("recursion.router.weight", filled(S.D, 2));
    for (uint32_t l = 0; l < S.L; ++l) {
        sd_extra.emplace("attnres." + std::to_string(l) + ".attn_q", filled(1, S.D));
        sd_extra.emplace("attnres." + std::to_string(l) + ".ffn_q", filled(1, S.D));
    }
    for (const std::string q : {"f32", "q8_0_all"}) {
        cases.push_back({"llama_state_extra_" + q, [&, q](const std::string& p) {
                             auto c = llama_cfg(S);
                             c.weights_in_out = true;
                             c.state_tokens_k = 3;
                             c.extra_proj = {{"recursion.router.weight", "rec.0.router.weight"}};
                             for (uint32_t l = 0; l < S.L; ++l) {
                                 const std::string i = std::to_string(l);
                                 c.extra_vec.push_back(
                                     {"attnres." + i + ".attn_q", "blk." + i + ".attnres_attn_q"});
                                 c.extra_vec.push_back(
                                     {"attnres." + i + ".ffn_q", "blk." + i + ".attnres_ffn_q"});
                             }
                             c.quant = q;
                             c.ir = kIR;
                             gguf::export_gguf_llama(p, sd_extra, c);
                         }});
    }

    // ---------------- gpt2 ----------------
    g_seed = 800;
    const auto sd_gpt2 = gpt2_sd(2, 32, 128, 40, 16, true);
    g_seed = 900;
    const auto sd_gpt2_tied = gpt2_sd(2, 32, 128, 40, 16, false);
    cases.push_back({"gpt2_f32_word", [&](const std::string& p) {
                         gguf::export_gguf_gpt2(p, sd_gpt2, gpt2_cfg(2, 32, 128, 40, 16, 2));
                     }});
    cases.push_back({"gpt2_f32_tied_bpe_chat_ir", [&](const std::string& p) {
                         auto c = gpt2_cfg(2, 32, 128, 40, 16, 2);
                         bpe_tokens(c, 40);
                         chat(c);
                         c.ir = kIR;
                         c.name = "fixture-gpt2";
                         c.layer_norm_eps = 1e-6f;
                         c.train_attention_precision = "bf16";
                         gguf::export_gguf_gpt2(p, sd_gpt2_tied, c);
                     }});
    cases.push_back({"gpt2_arch_paperkiln_ir", [&](const std::string& p) {
                         auto c = gpt2_cfg(2, 32, 128, 40, 16, 2);
                         c.architecture = "paperkiln";
                         c.ir = kIR;
                         gguf::export_gguf_gpt2(p, sd_gpt2, c);
                     }});

    // ---------------- paperkiln ----------------
    g_seed = 1000;
    std::vector<gguf::NamedTensor> pk;
    pk.push_back({"token_embd.weight", filled(40, 24), gguf::NamedTensor::Layout::ROWS});
    pk.push_back({"blk.0.ssm_a", filled(3, 5), gguf::NamedTensor::Layout::ROWS});
    pk.push_back({"blk.0.proj.weight", filled(24, 40), gguf::NamedTensor::Layout::PROJ});
    pk.push_back({"blk.0.norm.weight", filled(1, 24), gguf::NamedTensor::Layout::VEC});
    pk.push_back({"blk.0.column", filled(7, 1), gguf::NamedTensor::Layout::VEC});
    pk.push_back({"output.weight", filled(24, 40), gguf::NamedTensor::Layout::PROJ});
    cases.push_back({"paperkiln_word_ir", [&](const std::string& p) {
                         gguf::PaperkilnExportConfig c;
                         c.context_length = 32;
                         c.embedding_length = 24;
                         c.block_count = 1;
                         c.vocab_size = 40;
                         c.tokens = word_tokens(40);
                         c.ir = kIR;
                         gguf::export_gguf_paperkiln(p, pk, c);
                     }});
    cases.push_back({"paperkiln_heads_bpe_chat_ir", [&](const std::string& p) {
                         gguf::PaperkilnExportConfig c;
                         c.name = "fixture-paperkiln";
                         c.context_length = 32;
                         c.embedding_length = 24;
                         c.block_count = 1;
                         c.vocab_size = 40;
                         c.head_count = 3;
                         c.feed_forward_length = 48;
                         bpe_tokens(c, 40);
                         chat(c);
                         c.ir = kIR;
                         gguf::export_gguf_paperkiln(p, pk, c);
                     }});
    cases.push_back({"paperkiln_notok_ir_nodigest", [&](const std::string& p) {
                         gguf::PaperkilnExportConfig c;
                         c.context_length = 32;
                         c.embedding_length = 24;
                         c.block_count = 1;
                         c.vocab_size = 40;
                         c.ir = kIRNoDigest;
                         gguf::export_gguf_paperkiln(p, pk, c);
                     }});

    int failed = 0;
    for (const auto& [name, write] : cases) {
        const std::string path = tmp + "/gguf_byte_identity_" + name + ".gguf";
        write(path);
        const std::string sha = file_sha256(path);
        std::remove(path.c_str());
        if (print) {
            std::printf("%s %s\n", name.c_str(), sha.c_str());
            continue;
        }
        auto it = kWant.find(name);
        if (it == kWant.end()) {
            std::printf("  %-44s %s  (no recorded hash)\n", name.c_str(), sha.c_str());
            ++failed;
        } else if (it->second != sha) {
            std::printf("  %-44s %s  MISMATCH, recorded %s\n", name.c_str(), sha.c_str(),
                        it->second.c_str());
            ++failed;
        } else {
            std::printf("  %-44s ok\n", name.c_str());
        }
    }
    if (print) return 0;
    CHECK(kWant.size() == cases.size());
    if (failed) {
        std::printf("\n[FAIL] %d of %zu GGUF exports differ from their recorded bytes\n", failed,
                    cases.size());
        return 1;
    }
    std::printf("\n[PASS] %zu GGUF exports byte-identical to their recorded hashes\n",
                cases.size());
    return 0;
}
