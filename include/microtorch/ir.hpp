#pragma once
// paperkiln IR export (LOCAL_LLM_STACK.md section 4): the network as a
// versioned JSON graph that travels inside the GGUF, so an engine can
// interpret any paperkiln model rather than recognise a fixed architecture.
//
// Exported per module, never traced from the tape: each module emits its
// own node spec (Embedding/Linear/LayerNorm/MLP/CausalSelfAttention/
// SlidingWindowAttention/MoEFFN/LlamaBlock/Llama::export_ir,
// nn::gpt2_block_ir); the free functions below
// cover the ops that have no module of their own (rmsnorm, rope, attention,
// split, pos_embedding and the elementwise ops a block wires inline).
//
// Shape of a document (ir/chat_sftfinal.ir.json is the worked example):
//   ir_version, family_hint, model {vocab, d, n_layers, ctx, dtype},
//   tokenizer, inputs, embed {nodes}, block {repeat, state, nodes}, head
//   {nodes}, outputs, modes. Tensors flow as [T, width] row matrices; a
//   node reads the values named in "in" and writes "out" (a later write to
//   the same name replaces it). "weight" names a GGUF tensor, [out, in]
//   row-major; "{i}" in a block is the block index. dtype is the compute
//   dtype; each tensor's storage type is its GGUF type. Keys starting with
//   "_" are prose for humans and carry no semantics.
//
// Ops (v0):
//   embedding   rows of weight [vocab, d] gathered by token id. EXACT.
//   pos_embedding  learned absolute positions: in [x, pos], out = x +
//               weight[pos] row by row, weight [ctx, d] (position_embd).
//               pos must be < ctx: an engine stops or re-prefills a sliding
//               window before then (mtstudio re-runs the last ctx tokens
//               from position 0). One float add per element. EXACT.
//   linear      x W^T (+ bias), weight [out, in], bias [out] (the bias is
//               added once to the finished dot product); "tied_to" names
//               the tensor to read instead when "weight" is absent. ROUNDING.
//   split       in [x], sizes [s0, s1, ...] summing to x's width, out
//               [o0, o1, ...]: o_k = the s_k consecutive columns of x after
//               the first s0 + ... + s_{k-1} (a fused qkv projection read as
//               q, k, v). Copies. EXACT.
//   rmsnorm     x / sqrt(mean(x^2) + eps) * weight. ROUNDING.
//   layernorm   (x - mean(x)) / sqrt(var(x) + eps) * weight + bias over each
//               row, var the biased (divide by width) mean of squared
//               deviations; weight and bias [d]. ROUNDING.
//   gelu        approx "tanh": 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
//               (GPT-2's, and what paperkiln trains with); "none": the exact
//               0.5 x (1 + erf(x / sqrt(2))). An engine must not substitute
//               one for the other. ROUNDING.
//   rope        rotates q and k in place; rope_heads "all" (every head) or
//               "first" (head 0 only, legacy runs); pairing "interleaved" =
//               adjacent pairs (x[2j], x[2j+1]) within a head, angle
//               pos * theta^(-2j/head_dim). EXACT.
//   attention   softmax(q k^T * scale) v per head over the kv state slot it
//               names; causal, window (0 = unbounded) and sinks. ROUNDING.
//               Sliding window with sinks (ops::swa_attention; window >= 1):
//               with p the query's position and j a key's (both counted from
//               the first token of the slot, 0-based), j is visible iff
//                 j <= p  and  (j < sinks  or  j >= p - window + 1),
//               i.e. the first `sinks` positions plus the last `window`
//               positions up to and including p. Sinks are positions, not
//               parameters (no tensor). The softmax runs over that union,
//               each key once (the sink range is cut at the window's start
//               when they meet); masked keys contribute exactly 0, and the
//               row max is taken over visible keys only. paperkiln sums
//               sinks first, then the window, each in ascending j. window
//               >= p + 1 with sinks 0 is plain causal attention.
//               Decode: the slot must keep the k/v rows of positions
//               0..sinks-1 and of the last `window` positions (p - window +
//               1 .. p, the new token included); anything older is never
//               read again, so sinks + window rows per kv head suffice.
//               Positions stay absolute: an engine that re-prefills past
//               model.ctx (as mtstudio does: the last ctx tokens from
//               position 0) also restarts the slot, and the sinks become the
//               first tokens of the re-prefilled text, not the original ones.
//   add, mul    elementwise. EXACT.  silu  x * sigmoid(x). ROUNDING.
//   moe_ffn     a mixture of SwiGLU experts in llama.cpp's layout:
//                 router  [E, d] tensor (ffn_gate_inp), logits = x R^T
//                 experts {gate, up, down}: 3-D stacks (ffn_*_exps), expert
//                         e's [out, in] matrix at slice e
//                 n_experts E, top_k k: top-k of the E logits per token,
//                         ties to the lower expert index
//                 gate_norm "renorm": gates = softmax over the k selected
//                         logits (the E-way softmax renormalised over them);
//                         "raw": the E-way softmax's selected entries
//                 activation "silu": expert_e(x) = down_e(silu(gate_e x) * up_e x)
//                 out = sum over the selected e of gate_e * expert_e(x),
//                 accumulated in ascending expert index. ROUNDING.
// Every op has the same prefill (T tokens) and decode (T = 1) semantics;
// attention additionally reads and appends to its kv state in decode.
// A kv state declares the same window and sinks as the attention node that
// reads it (0 and 0 = keep every row up to max_len).
//
// GGUF: the compact dump is the string key kGGUFKey and the version the
// uint32 key kGGUFVersionKey, written after every other metadata key.
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace microtorch {
namespace ir {

// Insertion-ordered, so a node reads op, in, attributes, out.
using json = nlohmann::ordered_json;

constexpr uint32_t kVersion = 0;
constexpr const char* kGGUFKey = "paperkiln.ir";
constexpr const char* kGGUFVersionKey = "paperkiln.ir_version";

// A float32 attribute as the shortest decimal that reads back as the same
// float (1e-5f -> 1e-05, not 9.999999747378752e-06).
double f32(float v);

// Nodes for the ops that no module owns.
json rmsnorm(const std::string& in, const std::string& weight, float eps, const std::string& out);
json rope(const std::string& q, const std::string& k, const std::string& pos, size_t heads, size_t head_dim,
          float theta, bool all_heads);
json attention(const std::string& q, const std::string& k, const std::string& v, const std::string& state,
               size_t heads, size_t kv_heads, size_t head_dim, float scale, bool causal, size_t window,
               size_t sinks, const std::string& out);
json kv_state(const std::string& name, size_t heads, size_t kv_heads, size_t head_dim, size_t max_len,
              size_t window, size_t sinks);
json binary(const std::string& op, const std::string& a, const std::string& b, const std::string& out);
json unary(const std::string& op, const std::string& in, const std::string& out);
json layernorm(const std::string& in, const std::string& weight, const std::string& bias, float eps,
               const std::string& out);
// approx: "tanh" | "none" (anything else throws).
json gelu(const std::string& in, const std::string& approx, const std::string& out);
json split(const std::string& in, const std::vector<size_t>& sizes, const std::vector<std::string>& outs);
json pos_embedding(const std::string& x, const std::string& pos, const std::string& weight, const std::string& out);

}  // namespace ir
}  // namespace microtorch
