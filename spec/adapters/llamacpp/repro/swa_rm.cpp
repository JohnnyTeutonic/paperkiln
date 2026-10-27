// Rewind an iSWA model with llama_memory_seq_rm, then decode the same tokens again; compare with a fresh context.
// build: g++ -O2 -std=c++17 swa_rm.cpp -I llama.cpp/include -I llama.cpp/ggml/include
//        -L llama.cpp/build/bin -lllama -lggml -lggml-base -Wl,-rpath,llama.cpp/build/bin -o swa_rm
// run:   ./swa_rm model.gguf N_PREFILL CUT SWA_FULL
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static llama_context * make_ctx(llama_model * model, bool swa_full) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 4096; cp.n_ubatch = 512; cp.n_threads = cp.n_threads_batch = 2;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.swa_full = swa_full;
    return llama_init_from_model(model, cp);
}

// decode ids[p0, p1) at positions p0..p1-1; return the logits of every position
static std::vector<float> decode(llama_context * ctx, const std::vector<llama_token> & ids, int p0, int p1, int n_vocab) {
    llama_batch b = llama_batch_init(p1 - p0, 0, 1);
    for (int p = p0; p < p1; p++) {
        int i = p - p0;
        b.token[i] = ids[p]; b.pos[i] = p; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = p1 - p0;
    int rc = llama_decode(ctx, b);
    if (rc != 0) fprintf(stderr, "llama_decode returned %d\n", rc);
    std::vector<float> out;
    for (int i = 0; i < p1 - p0; i++) {
        const float * l = llama_get_logits_ith(ctx, i);
        out.insert(out.end(), l, l + n_vocab);
    }
    llama_batch_free(b);
    return out;
}

int main(int argc, char ** argv) {
    const int n_prefill = atoi(argv[2]), cut = atoi(argv[3]), swa_full = atoi(argv[4]), k = 4;
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> ids;
    for (int i = 0; i < n_prefill; i++) ids.push_back((llama_token) (1 + (i * 7919 + 13) % (n_vocab - 1)));

    // A: prefill, cut back to `cut`, decode ids[cut, cut + k) again
    llama_context * a = make_ctx(model, swa_full);
    llama_memory_t mem = llama_get_memory(a);
    decode(a, ids, 0, n_prefill, n_vocab);
    printf("after prefill of %d: seq_pos_min %d, seq_pos_max %d\n", n_prefill,
           llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0));
    bool ok = llama_memory_seq_rm(mem, 0, cut, -1);
    printf("llama_memory_seq_rm(mem, 0, %d, -1) returned %s; seq_pos_min %d, seq_pos_max %d\n", cut, ok ? "true" : "false",
           llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0));
    auto got = decode(a, ids, cut, cut + k, n_vocab);

    // B: fresh context, decode ids[0, cut + k)
    llama_context * b = make_ctx(model, swa_full);
    auto all = decode(b, ids, 0, cut + k, n_vocab);
    std::vector<float> want(all.end() - (size_t) k * n_vocab, all.end());

    double d = 0.0, mx = 0.0; int argmax_diff = 0;
    for (int i = 0; i < k; i++) {
        int ga = 0, wa = 0;
        for (int v = 0; v < n_vocab; v++) {
            d = std::fmax(d, std::fabs((double) got[i * n_vocab + v] - want[i * n_vocab + v]));
            mx = std::fmax(mx, std::fabs((double) want[i * n_vocab + v]));
            if (got[i * n_vocab + v] > got[i * n_vocab + ga]) ga = v;
            if (want[i * n_vocab + v] > want[i * n_vocab + wa]) wa = v;
        }
        argmax_diff += ga != wa;
    }
    printf("swa_full %d, prefill %d, cut to %d, decode %d: vs a fresh context, max |logit difference| %.3g "
           "(max |logit| %.3g), argmax differs at %d of %d\n", swa_full, n_prefill, cut, k, d, mx, argmax_diff, k);
    llama_free(a); llama_free(b); llama_model_free(model);
    return 0;
}
