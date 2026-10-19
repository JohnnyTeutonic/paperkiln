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
#include <cstddef>
#include <cstdint>
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

    // Weight storage. "f32" (default) writes every tensor as F32. The Q8_0
    // variants (llama.cpp's block format, see quantize_q8_0) quantise the 2-D
    // tensors below and leave everything else F32, matching the ember
    // session's quantize_gguf.py byte for byte:
    //   "q8_0"      every 2-D blk.* tensor (attention, dense FFN, MoE router)
    //   "q8_0_out"  ... plus output.weight (the LM head, as llama.cpp does)
    //   "q8_0_all"  ... plus token_embd.weight
    // 3-D MoE expert stacks and the 1-D norms always stay F32. A selected
    // tensor whose ne0 is not a multiple of 32 throws, naming the tensor.
    // No metadata changes (no general.file_type), as in the reference.
    //
    // The 4-bit variants write the file llama.cpp's own quantiser
    // (llama_model_quantize, no importance matrix) makes from this
    // exporter's F32 file, byte for byte, for the llama.cpp ftype of the
    // same name. (One exception, llama.cpp's: it copies metadata strings as
    // C strings, so a token "\0" in the source vocabulary comes out empty
    // there; this exporter keeps it.)
    //   "q4_0"    MOSTLY_Q4_0
    //   "q4_k_m"  MOSTLY_Q4_K_M
    //   "q4_k_s"  MOSTLY_Q4_K_S
    // As llama.cpp does, they write the tensors in its order (non-layer
    // tensors, then layer by layer, by name within each), append
    // general.quantization_version = 2 and general.file_type, pad the last
    // tensor to 32 bytes, keep the 1-D norms F32 and choose each 2-D
    // tensor's type with its llama_tensor_get_type rules: output.weight (or
    // token_embd.weight when there is no output.weight) Q6_K; under q4_k_m,
    // Q6_K for attn_v / ffn_down in the layers use_more_bits picks; under
    // q4_k_s, Q5_K for attn_v in the first four layers and ffn_down in the
    // first eighth; the ftype's type for the rest. A k-quant (Q4_K, Q5_K,
    // Q6_K) needs ne0 to be a multiple of 256; where it is not, llama.cpp
    // falls back to Q5_0, Q5_1 and Q8_0 respectively, and so does this
    // exporter. The k-quant blocks themselves are not implemented: a tensor
    // that llama.cpp would store as one throws, naming it. (On a d = 384,
    // d_ff = 1152 model q4_k_m and q4_k_s therefore contain no Q4_K at all,
    // as with llama.cpp.) Dense models only: MoE throws.
    std::string quant = "f32";

    // The model's paperkiln IR (microtorch/ir.hpp; nn::Llama::export_ir), a
    // JSON document. When non-empty it is written compact as the string key
    // paperkiln.ir, with its ir_version as the uint32 paperkiln.ir_version,
    // after every other metadata key: only the header grows, so tensor infos
    // and data are the bytes an export without it writes. Empty (the
    // default) writes neither key.
    std::string ir;
};

// Q8_0 blocks of `n` floats (n a multiple of 32) into n / 32 * 34 bytes:
// per 32 values an fp16 scale d = max|x| / 127, then 32 int8
// q = round(x * (1 / d)), half away from zero. Float32 arithmetic and a
// round-to-nearest-even fp16 conversion, so the bytes equal gguf-py's
// quants.Q8_0 (itself bit-exact with ggml-quants.c). Throws on n % 32 != 0
// or a non-finite input.
void quantize_q8_0(const float* x, size_t n, uint8_t* out);

// ggml-quants.c's reference blocks (quantize_row_*_ref, what llama.cpp's
// quantiser runs without an importance matrix), bit for bit; n a multiple
// of 32, finite inputs, else they throw. Per 32 values:
//   Q4_0 (n / 32 * 18 bytes): fp16 d = m / -8, m the first value of largest
//     magnitude; 16 bytes of nibbles q = min(15, trunc(x / d + 8.5)), value
//     j in the low nibble of byte j and value 16 + j in its high nibble.
//   Q5_0 (22 bytes): fp16 d = m / -16; 4 bytes qh holding bit 4 of each
//     q = min(31, trunc(x / d + 16.5)) (bit j for value j); 16 bytes of
//     low nibbles laid out as in Q4_0.
//   Q5_1 (24 bytes): fp16 d = (max - min) / 31, fp16 min, then qh and the
//     nibbles of q = min(31, trunc((x - min) / d + 0.5)) as in Q5_0.
// Here x / d is x * (1 / d), and 1 / d is taken as 0 when d is 0.
void quantize_q4_0(const float* x, size_t n, uint8_t* out);
void quantize_q5_0(const float* x, size_t n, uint8_t* out);
void quantize_q5_1(const float* x, size_t n, uint8_t* out);

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

// Hyperparameters + tokenizer for a GPT-2-family export in llama.cpp's
// "gpt2" architecture (pre-LN blocks, LayerNorm with bias, fused qkv with
// bias, tanh-approximation GELU MLP with biases, learned absolute
// positions). Tokenizer and IR fields as in LlamaExportConfig.
struct Gpt2ExportConfig {
    std::string name = "microtorch-model";
    // Learned positions: position_embd has exactly this many rows, and
    // llama.cpp checks it against gpt2.context_length.
    uint32_t context_length = 0;
    uint32_t embedding_length = 0;     // d_model (required)
    uint32_t block_count = 0;          // n_layers (required)
    uint32_t feed_forward_length = 0;  // MLP hidden width (required)
    uint32_t head_count = 0;           // n_heads (required)
    uint32_t vocab_size = 0;           // required
    float layer_norm_eps = 1e-5f;      // nn::LayerNorm's default
    std::vector<std::string> tokens;
    std::string tokenizer_model = "word";
    std::vector<std::string> merges;
    uint32_t bos_token_id = 1, eos_token_id = 1;
    uint32_t unk_token_id = 0, pad_token_id = 0;
    std::string ir;  // written as paperkiln.ir / paperkiln.ir_version when non-empty
};

// Export a GPT-2 state_dict (HF GPT-2 names; every 2-D projection in the
// HF Conv1D / microtorch nn::Linear layout rows=[in], cols=[out], so each is
// transposed to llama.cpp's [out][in]) to an F32 GGUF:
//   wte.weight [vocab, d]                -> token_embd.weight     ne={d, vocab}
//   wpe.weight [ctx, d]                  -> position_embd.weight  ne={d, ctx}
//   h.N.ln_1.{weight,bias}               -> blk.N.attn_norm.{weight,bias}
//   h.N.attn.c_attn.weight [d, 3d], bias -> blk.N.attn_qkv.weight ne={d, 3d}, .bias {3d}
//                                           (columns q | k | v, head h at h*dk within each)
//   h.N.attn.c_proj.{weight,bias}        -> blk.N.attn_output.{weight,bias}
//   h.N.ln_2.{weight,bias}               -> blk.N.ffn_norm.{weight,bias}
//   h.N.mlp.c_fc.{weight,bias}           -> blk.N.ffn_up.{weight,bias}    ne={d, ff}, {ff}
//   h.N.mlp.c_proj.{weight,bias}         -> blk.N.ffn_down.{weight,bias}  ne={ff, d}, {d}
//   ln_f.{weight,bias}                   -> output_norm.{weight,bias}
//   lm_head.weight [d, vocab]            -> output.weight ne={d, vocab} (omitted if
//                                           absent: llama.cpp then ties to token_embd)
// with general.architecture = "gpt2" and gpt2.{context_length,
// embedding_length, block_count, feed_forward_length, attention.head_count,
// attention.layer_norm_epsilon, vocab_size}. A leading "transformer." is
// optional on every key. Every shape is checked against the config; keys
// outside the map go to `unmapped` if given. F32 only.
void export_gguf_gpt2(const std::string& path, const std::map<std::string, Matrix>& state_dict,
                      const Gpt2ExportConfig& cfg, std::vector<std::string>* unmapped = nullptr);

// A family no llama.cpp graph computes (microtorch/ir.hpp, Families): the
// embedded IR is the model, the tensors are what it names.
struct PaperkilnExportConfig {
    std::string name = "microtorch-model";
    // Written as paperkiln.<key>; 0 = not applicable, key omitted (except
    // the four required ones).
    uint32_t context_length = 0, embedding_length = 0, block_count = 0, vocab_size = 0;  // required
    uint32_t head_count = 0, feed_forward_length = 0;
    std::vector<std::string> tokens;
    std::string tokenizer_model = "word";
    std::vector<std::string> merges;
    uint32_t bos_token_id = 1, eos_token_id = 1;
    uint32_t unk_token_id = 0, pad_token_id = 0;
    std::string ir;  // required: without it nothing can run the file
};

// One tensor, in microtorch's layout, with its GGUF name:
//   ROWS  [rows, cols] written as is (embedding tables, A of ssm_scan): ne={cols, rows}
//   PROJ  an nn::Linear weight [in, out], transposed to [out][in]: ne={in, out}
//   VEC   a [1, n] row (norms, biases, pseudo-queries; also a [n, 1] column): ne={n}
struct NamedTensor {
    enum class Layout { ROWS, PROJ, VEC };
    std::string name;
    Matrix m;
    Layout layout;
};

// general.architecture = "paperkiln", F32 tensors in the order given.
// Throws on a duplicate name or a missing required field.
void export_gguf_paperkiln(const std::string& path, const std::vector<NamedTensor>& tensors,
                           const PaperkilnExportConfig& cfg);

}  // namespace gguf
}  // namespace microtorch
