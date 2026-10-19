#pragma once
// paperkiln IR export (LOCAL_LLM_STACK.md section 4): the network as a
// versioned JSON graph that travels inside the GGUF, so an engine can
// interpret any paperkiln model rather than recognise a fixed architecture.
//
// Exported per module, never traced from the tape: each module emits its
// own node spec (Embedding/Linear/MoEFFN/LlamaBlock/Llama::export_ir); the
// free functions below cover the ops that have no module of their own
// (rmsnorm, rope, attention and the elementwise ops a block wires inline).
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
//   linear      x W^T (+ bias), weight [out, in]; "tied_to" names the tensor
//               to read instead when "weight" is absent. ROUNDING.
//   rmsnorm     x / sqrt(mean(x^2) + eps) * weight. ROUNDING.
//   rope        rotates q and k in place; rope_heads "all" (every head) or
//               "first" (head 0 only, legacy runs); pairing "interleaved" =
//               adjacent pairs (x[2j], x[2j+1]) within a head, angle
//               pos * theta^(-2j/head_dim). EXACT.
//   attention   softmax(q k^T * scale) v per head over the kv state slot it
//               names; causal, window (0 = unbounded) and sinks. ROUNDING.
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
//
// GGUF: the compact dump is the string key kGGUFKey and the version the
// uint32 key kGGUFVersionKey, written after every other metadata key.
#include <cstdint>
#include <string>

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

}  // namespace ir
}  // namespace microtorch
