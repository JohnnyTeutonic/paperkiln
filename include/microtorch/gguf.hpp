#pragma once
// GGUF export: the bridge that completes the pipeline
//   HF checkpoint -> load_safetensors -> (fine-tune on the tape) -> export_gguf
//   -> tinyllama.cpp serves it.
//
// The byte-level writer is transformer_cpp's GGUFWriter -- the one that
// already survived the 2026-07-13 alignment postmortem (EXPORT_NOTES.md:
// data section starts at align_up(header_end, 32); offsets are relative to
// that aligned start) and is argmax-parity verified against tinyllama.cpp.
// This header only maps microtorch state_dicts onto that writer.
#include <map>
#include <string>
#include <vector>

#include "microtorch/primitives.hpp"

namespace microtorch {
namespace gguf {

// Hyperparameters + tokenizer for a Llama-family export. Mirrors the
// metadata block tinyllama.cpp reads (llama.* keys).
struct LlamaExportConfig {
    std::string name = "microtorch-model";
    std::string architecture = "llama";
    uint32_t context_length = 2048;
    uint32_t embedding_length = 0;     // d_model (required)
    uint32_t block_count = 0;          // n_layers (required)
    uint32_t feed_forward_length = 0;  // d_ff (required)
    uint32_t head_count = 0;           // n_heads (required)
    uint32_t head_count_kv = 0;        // 0 -> defaults to head_count (MHA)
    uint32_t vocab_size = 0;           // required
    float rms_eps = 1e-6f;
    float rope_freq_base = 10000.0f;
    // Mixture of experts (llama.cpp's Mixtral layout; nn::MoEFFN). 0 = dense.
    // When set, each layer's FFN is read from
    //   layers.N.mlp.router.weight                    -> blk.N.ffn_gate_inp.weight  ne={d, E}
    //   layers.N.mlp.experts.{e}.{gate,up}_proj.weight -> blk.N.ffn_{gate,up}_exps.weight ne={d, ff, E}
    //   layers.N.mlp.experts.{e}.down_proj.weight      -> blk.N.ffn_down_exps.weight ne={ff, d, E}
    // with metadata <arch>.expert_count / expert_used_count. llama.cpp's
    // llama-architecture MoE renormalises the top-k softmax gates, so only
    // a renormalised-gate model is served faithfully.
    uint32_t expert_count = 0, expert_used_count = 0;

    // Weight layout of the 2-D projection tensors in the state_dict:
    //  false (default): HF torch.nn.Linear rows=[out], cols=[in] -- what
    //    load_safetensors returns for Llama/Qwen checkpoints. Written as-is
    //    (that IS the llama.cpp layout).
    //  true: microtorch nn::Linear rows=[in], cols=[out] (y = xW). Each
    //    projection is transposed on export, exactly like transformer_cpp's
    //    verified add_transposed path.
    bool weights_in_out = false;

    // Optional word/BPE token list; empty -> no tokenizer block is written
    // (tinyllama.cpp can still load tensors; generation needs a vocab).
    std::vector<std::string> tokens;
    std::string tokenizer_model = "word";
    // BPE merges ("a b" token-id pairs in rank order, tools/bpe.py); written
    // as tokenizer.ggml.merges when non-empty, so a BPE export carries its
    // own tokenizer.
    std::vector<std::string> merges;
    uint32_t bos_token_id = 1, eos_token_id = 1;
    uint32_t unk_token_id = 0, pad_token_id = 0;
};

// Export a state_dict with HF-Llama naming to GGUF:
//   model.embed_tokens.weight            -> token_embd.weight
//   model.layers.N.self_attn.{q,k,v,o}_proj.weight -> blk.N.attn_{q,k,v,output}.weight
//   model.layers.N.input_layernorm.weight          -> blk.N.attn_norm.weight
//   model.layers.N.mlp.{gate,up,down}_proj.weight  -> blk.N.ffn_{gate,up,down}.weight
//   model.layers.N.post_attention_layernorm.weight -> blk.N.ffn_norm.weight
//   model.norm.weight                    -> output_norm.weight
//   lm_head.weight                       -> output.weight (omitted if absent:
//                                           tied embeddings)
// A leading "model." prefix is optional on every key. Tensors that do not
// match the map (e.g. Qwen's attention biases, rotary caches) are collected
// into `unmapped` if given, and skipped -- fail loudly by checking it.
// Throws std::runtime_error on missing required tensors or config fields.
void export_gguf_llama(const std::string& path, const std::map<std::string, Matrix>& state_dict,
                       const LlamaExportConfig& cfg, std::vector<std::string>* unmapped = nullptr);

}  // namespace gguf
}  // namespace microtorch
