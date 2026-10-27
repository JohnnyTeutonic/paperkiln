// Same prompt, four configurations: flash attention off/on x KV cache f32/f16. Logits of every position.
// build: g++ -O2 -std=c++17 fa_f32.cpp -I llama.cpp/include -I llama.cpp/ggml/include
//        -L llama.cpp/build/bin -lllama -lggml -lggml-base -Wl,-rpath,llama.cpp/build/bin -o fa_f32
// run:   ./fa_f32 model-f32.gguf
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static std::vector<float> run(llama_model * model, bool fa, ggml_type kv, const std::vector<llama_token> & ids) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = cp.n_threads_batch = 2;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = kv;
    llama_context * ctx = llama_init_from_model(model, cp);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch b = llama_batch_init((int) ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); i++) {
        b.token[i] = ids[i]; b.pos[i] = (llama_pos) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = (int) ids.size();
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); }
    std::vector<float> out;
    for (size_t i = 0; i < ids.size(); i++) {
        const float * l = llama_get_logits_ith(ctx, (int) i);
        out.insert(out.end(), l, l + n_vocab);
    }
    llama_batch_free(b);
    llama_free(ctx);
    return out;
}

static double rel(const std::vector<float> & a, const std::vector<float> & ref) {
    double m = 1.0, d = 0.0;
    for (float r : ref) m = std::fmax(m, std::fabs(r));
    for (size_t i = 0; i < a.size(); i++) d = std::fmax(d, std::fabs((double) a[i] - ref[i]));
    return d / m;
}

int main(int argc, char ** argv) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> ids;
    for (int i = 0; i < 128; i++) ids.push_back((llama_token) ((i * 7919 + 13) % n_vocab));

    auto ref    = run(model, false, GGML_TYPE_F32, ids);
    auto off16  = run(model, false, GGML_TYPE_F16, ids);
    auto on32   = run(model, true,  GGML_TYPE_F32, ids);
    auto on16   = run(model, true,  GGML_TYPE_F16, ids);
    printf("max |logit - ref| / max|ref|, ref = flash attention off, f32 KV cache (%zu positions)\n", ids.size());
    printf("  fa off, f16 cache: %.3g\n", rel(off16, ref));
    printf("  fa on,  f32 cache: %.3g\n", rel(on32, ref));
    printf("  fa on,  f16 cache: %.3g\n", rel(on16, ref));
    printf("fa on: f32 cache logits bit-identical to f16 cache logits: %s\n",
           memcmp(on32.data(), on16.data(), on32.size() * sizeof(float)) == 0 ? "yes" : "no");
    llama_model_free(model);
    return 0;
}
