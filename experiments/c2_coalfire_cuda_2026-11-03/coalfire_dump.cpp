// C2 CUDA leg: docs/receipts/c2_coalfire_parity_20261103/coalfire_dump.cpp, unchanged except
// that it first reports the build's CUDA state as one JSON line, {"use_cuda": ..., "devices":
// ..., "device": ..., "cc": ...}, a CUDA build exits 3 without a visible device instead of
// running, and the config's two uninitialised flags are set to false. Loads a coalfire format-3 checkpoint of the --tiny llama-family model, runs
// forward_batch (the path training optimises through; golden_batch_test.cpp explains why not
// forward()) over the ids in IDS and writes every row's logits as float32 [T, vocab] to OUT. The
// config is rebuilt exactly as coalfire's src/golden_batch_test.cpp does for --tiny: the
// format-3 checkpoint does not record the architecture.
//
//   coalfire_dump CKPT VOCAB_FILE_0 VOCAB_FILE_1 SEQ_LEN IDS OUT
//
// build: with exactly the compile flags coalfire's own build gave src/transformer.cpp (vm_c2.sh
// takes them from compile_commands.json), linked against that build's libtransformer_core.a. In
// a CUDA build the headers change under USE_CUDA, so flags that differ from the library's would
// compile a different class layout.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "architecture.hpp"
#include "model_saver.hpp"
#include "safetensors_export.hpp"
#include "tiktoken_tokenizer.hpp"
#include "transformer.hpp"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8) {
        std::cerr << "usage: coalfire_dump CKPT VOCAB_FILE_0 VOCAB_FILE_1 SEQ_LEN IDS OUT "
                     "[EXPORT_HF_DIR]\n";
        return 2;
    }
#ifdef USE_CUDA
    {
        int n = 0;
        const cudaError_t e = cudaGetDeviceCount(&n);
        cudaDeviceProp p{};
        if (e == cudaSuccess && n > 0) cudaGetDeviceProperties(&p, 0);
        std::printf(
            "{\"use_cuda\": true, \"devices\": %d, \"device\": \"%s\", \"cc\": \"%d.%d\", "
            "\"error\": \"%s\"}\n",
            e == cudaSuccess ? n : 0, (e == cudaSuccess && n > 0) ? p.name : "",
            p.major, p.minor, e == cudaSuccess ? "" : cudaGetErrorName(e));
        std::fflush(stdout);
        if (e != cudaSuccess || n == 0) {
            std::cerr << "coalfire_dump: CUDA build without a visible device\n";
            return 3;
        }
    }
#else
    std::printf("{\"use_cuda\": false}\n");
#endif
    auto tokenizer = std::make_shared<TiktokenTokenizer>();
    tokenizer->build_vocabulary_from_plain_text(argv[2]);  // the order train.cpp uses
    tokenizer->build_vocabulary_from_plain_text(argv[3]);

    TransformerConfig c;  // golden_batch_test.cpp make_config, --tiny
    c.vocab_size = std::min(tokenizer->vocab_size(), static_cast<size_t>(5000));
    c.hidden_size = 128;
    c.num_heads = 4;
    c.num_layers = 2;
    c.intermediate_size = 512;
    c.max_seq_length = std::stoul(argv[4]);
    c.head_dim = c.hidden_size / c.num_heads;
    // TransformerConfig gives these two no initialiser, and constructing the layers reads use_fp16
    // (MultiHeadAttention stores it); set them so nothing reads an indeterminate value. This forward
    // selects no FP16 path either way.
    c.use_fp16 = false;
    c.use_gradient_checkpointing = false;
    arch::ArchitectureSpec spec = arch::ArchitectureSpec::from_family("llama");
    spec.apply(c);
    transformer_runtime::llama_no_bias = !c.use_biases;

    Transformer model(c, tokenizer);
    auto bias_stats =
        [&](const char* when) {  // min/max of every FFN bias entry, per layer and tensor
            for (size_t l = 0; l < model.getLayers().size(); ++l) {
                FeedForward* ff = model.getLayers()[l]->getFeedForward();
                if (!ff) continue;
                auto& p = ff->parameters();
                const char* names[] = {"gate", "up", "down"};
                int k = 0;
                for (FloatVector* b : {&p.gate_proj_bias, &p.up_proj_bias, &p.down_proj_bias}) {
                    float lo = 1e30f, hi = -1e30f;
                    for (size_t i = 0; i < b->size(); ++i)
                        lo = std::min(lo, (*b)[i]), hi = std::max(hi, (*b)[i]);
                    std::printf(
                        "{\"bias\": \"%s\", \"layer\": %zu, \"tensor\": \"%s\", \"n\": %zu, "
                        "\"min\": %g, \"max\": %g}\n",
                        when, l, names[k++], b->size(), lo, hi);
                }
            }
        };
    if (std::getenv("C2_BIAS_STATS")) bias_stats("after construction");
    ModelSaver saver;
    if (!saver.loadCheckpoint(model, argv[1])) {
        std::cerr << "coalfire_dump: cannot load " << argv[1] << "\n";
        return 1;
    }
    model.set_tokenizer(tokenizer);
    model.set_training(false);
    if (std::getenv("C2_BIAS_STATS")) bias_stats("after checkpoint load");
    // The CPU FeedForward adds gate/up/down biases unconditionally; the exports write none. Report
    // their size, and with C2_ZERO_FFN_BIAS=1 zero them (diagnostic: the model the exports
    // describe).
    {
        const bool zero = std::getenv("C2_ZERO_FFN_BIAS") != nullptr;
        float worst = 0.0f;
        for (auto& layer : model.getLayers()) {
            FeedForward* ff = layer->getFeedForward();
            if (!ff) continue;
            auto& p = ff->parameters();
            for (FloatVector* b : {&p.gate_proj_bias, &p.up_proj_bias, &p.down_proj_bias})
                for (size_t i = 0; i < b->size(); ++i) {
                    worst = std::max(worst, std::abs((*b)[i]));
                    if (zero) (*b)[i] = 0.0f;
                }
        }
        std::printf("{\"ffn_bias_max_abs\": %g, \"zeroed\": %s}\n", worst, zero ? "true" : "false");
    }
    if (argc == 8) {  // the HF export of the model as loaded from the checkpoint, to compare with
                      // the training export
        safetensors_export::SafetensorsExportConfig st;
        st.model_name = "transformer_cpp_lm";
        if (!safetensors_export::export_to_hf(model, *tokenizer, argv[7], st)) {
            std::cerr << "coalfire_dump: export_to_hf failed\n";
            return 1;
        }
    }

    std::vector<int> ids;
    std::ifstream in(argv[5]);
    for (int t; in >> t;) ids.push_back(t);
    if (ids.empty()) {
        std::cerr << "coalfire_dump: no ids\n";
        return 1;
    }
    TransformerOutput out = model.forward_batch({ids}, ids.size());
    const Matrix& L = out.logits;
    if (L.rows() < ids.size()) {
        std::cerr << "coalfire_dump: logits rows " << L.rows() << " for " << ids.size() << " ids\n";
        return 1;
    }
    std::ofstream f(argv[6], std::ios::binary);
    for (size_t i = 0; i < ids.size(); ++i)
        for (size_t v = 0; v < L.cols(); ++v) {
            const float x = L(i, v);
            f.write(reinterpret_cast<const char*>(&x), sizeof x);
        }
    std::printf(
        "{\"engine\": \"coalfire\", \"rows\": %zu, \"vocab\": %zu, \"use_rope\": %d, "
        "\"use_rms_norm\": %d, "
        "\"use_biases\": %d, \"eps\": %g}\n",
        ids.size(), static_cast<size_t>(L.cols()), c.use_rope, c.use_rms_norm, c.use_biases,
        c.layer_norm_epsilon);
    return 0;
}
