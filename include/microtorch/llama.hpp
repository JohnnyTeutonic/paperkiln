#pragma once
// Llama-family model (M1 completion, STUDIO_PLAN section 4): RMSNorm,
// RoPE, SwiGLU, no biases. Parameter names follow the HF LlamaForCausalLM
// convention EXACTLY ("layers.N.self_attn.q_proj.weight", ...), so:
//   - load_state_dict() ingests HF checkpoints (Qwen et al.) untranslated,
//   - gguf::export_gguf_llama(state_dict, cfg, weights_in_out=true) exports
//     a tinyllama.cpp-servable GGUF with zero name mapping.
// That closes the studio loop: train/fine-tune here -> chat there.
#include "microtorch/nn.hpp"

namespace microtorch {
namespace nn {

struct LlamaConfig {
    size_t vocab = 4096, d = 128, n_layers = 2, n_heads = 4;
    size_t d_ff = 512;  // SwiGLU intermediate
    size_t n_ctx = 256;
    float rms_eps = 1e-6f;
    float rope_theta = 10000.0f;
    bool tie_embeddings = true;  // lm_head shares embed_tokens
    // RoPE on every head (spec arch.rope_heads = "all"). false reproduces
    // the legacy rotation of head 0 only ("first"), which every llama run
    // recorded without a rope_heads field was trained with.
    bool rope_all_heads = true;
};

class LlamaBlock : public Module {
public:
    LlamaBlock(const LlamaConfig& cfg, unsigned seed);
    // seq_len 0 = single sequence; otherwise x rows are B stacked
    // sequences of seq_len, attention-isolated by a block-diagonal mask.
    Var forward(const Var& x, const std::vector<int>& positions, size_t seq_len = 0) const;

    Var ln1_w, ln2_w;  // input/post_attention layernorm gammas
    std::shared_ptr<Linear> q_proj, k_proj, v_proj, o_proj;
    std::shared_ptr<Linear> gate_proj, up_proj, down_proj;
    size_t H, dk;
    float rope_theta_, rms_eps_;
    bool rope_all_heads_;
};

class Llama : public Module {
public:
    explicit Llama(const LlamaConfig& cfg, unsigned seed = 0);
    // ids: one sequence, or B sequences of seq_len concatenated (RoPE
    // positions restart per sequence; attention is block-isolated).
    Var forward(const std::vector<int>& ids, size_t seq_len = 0) const;

    LlamaConfig cfg;
    // Activation checkpointing per block (see autograd.hpp).
    bool checkpoint_blocks = false;
    std::shared_ptr<Embedding> embed_tokens;
    std::vector<std::shared_ptr<LlamaBlock>> blocks;
    Var norm_w;                       // final RMSNorm gamma
    std::shared_ptr<Linear> lm_head;  // null when tied
};

}  // namespace nn
}  // namespace microtorch
