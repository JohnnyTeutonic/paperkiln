// Logits of every position for the token ids in IDS (whitespace-separated), f32 KV cache, flash attention off.
// build: g++ -O2 -std=c++17 dump_logits.cpp -I llama.cpp/include -I llama.cpp/ggml/include
//        -L llama.cpp/build/bin -lllama -lggml -lggml-base -Wl,-rpath,llama.cpp/build/bin -o dump_logits
// run:   ./dump_logits model-f32.gguf ids.txt logits.f32      (writes n_tokens x n_vocab float32)
#include "llama.h"
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    std::vector<llama_token> ids;
    FILE * f = fopen(argv[2], "r");
    for (int t; fscanf(f, "%d", &t) == 1;) ids.push_back(t);
    fclose(f);
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = cp.n_threads_batch = 2;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch b = llama_batch_init((int) ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); i++) {
        b.token[i] = ids[i]; b.pos[i] = (llama_pos) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = (int) ids.size();
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    FILE * o = fopen(argv[3], "wb");
    for (size_t i = 0; i < ids.size(); i++) fwrite(llama_get_logits_ith(ctx, (int) i), sizeof(float), n_vocab, o);
    fclose(o);
    printf("wrote %zu x %d logits\n", ids.size(), n_vocab);
    llama_batch_free(b); llama_free(ctx); llama_model_free(model);
    return 0;
}
