#pragma once
// paperkiln IR export: the network as a versioned JSON graph that travels
// inside the GGUF, so an engine can interpret any paperkiln model rather than
// recognise a fixed architecture.
//
// The IR's semantics are specified in spec/IR_SPEC.md (IR version 1; schema
// spec/ir.schema.json, validator spec/tools/ir_validate.py, reference interpreter
// spec/tools/ir_numpy.py, conformance suite spec/conformance/). This header
// documents the C++ API that writes documents; where it and the specification
// differ, the specification holds.
//
// Exported per module, never traced from the tape: each module emits its own
// node spec (Embedding/Linear/LayerNorm/MLP/CausalSelfAttention/
// SlidingWindowAttention/MoEFFN/LlamaBlock/Llama::export_ir,
// nn::gpt2_block_ir; the other families through family_export() in
// tools/parity_model.hpp); the free functions below cover the ops that have no
// module of their own (rmsnorm, rope, attention, split, pos_embedding and the
// elementwise ops a block wires inline). Each returns one node (or one state
// slot) of IR_SPEC.md sections 5 to 7, attributes in the order op, in,
// attributes, out.
//
// kVersion is the ir_version exporters write: 1, the version IR_SPEC.md
// specifies (section 2.3; a version 0 document is a version 1 document, and
// ember reads both from ce41ffb on). GGUF: the compact dump is the string key kGGUFKey and the version
// the uint32 key kGGUFVersionKey, written after every other metadata key
// (IR_SPEC.md section 3.1).
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace microtorch {
namespace ir {

// Insertion-ordered, so a node reads op, in, attributes, out.
using json = nlohmann::ordered_json;

constexpr uint32_t kVersion = 1;
constexpr const char* kGGUFKey = "paperkiln.ir";
constexpr const char* kGGUFVersionKey = "paperkiln.ir_version";
// The document's digest (IR_SPEC.md section 3.10), 64 lowercase hex digits,
// written after kGGUFVersionKey.
constexpr const char* kGGUFDigestKey = "paperkiln.ir_sha256";

// A float32 attribute as the shortest decimal that reads back as the same
// float (1e-5f -> 1e-05, not 9.999999747378752e-06).
double f32(float v);

// The IR digest (IR_SPEC.md section 3.10): the RFC 8785 serialisation of the
// document with every "_" key removed (canonical), and the sha256 of its UTF-8
// bytes as 64 lowercase hex digits (digest). A value with no digest (a
// negative zero, an integer beyond 2^53, a non-finite number, malformed
// UTF-8) throws DigestError. spec/tools/ir_validate.py's canonical() and digest()
// are the reference; spec/conformance/digest/ holds the vectors.
struct DigestError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
std::string canonical(const json& doc, bool keep_prose = false);  // keep_prose: the serialiser alone (tests)
std::string digest(const json& doc);
std::string sha256_hex(const std::string& bytes);
// A JSON text: parsed (a duplicate member name or a malformed text throws
// DigestError), then canonical / digest.
json parse_ijson(const std::string& text);
std::string digest_text(const std::string& text);

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
// (x * weight[j]) + bias[j] per feature; bias "" = none (written as null).
json scale(const std::string& in, const std::string& weight, const std::string& bias, const std::string& out);
// (x * scale) + shift.
json affine(const std::string& in, float scale, float shift, const std::string& out);
json rms_row(const std::string& in, float eps, const std::string& out);
// Causal elu+1 linear attention over the recurrent slot `state`.
json linear_attention(const std::string& q, const std::string& k, const std::string& v, const std::string& state,
                      size_t heads, size_t head_dim, const std::string& out);
// The recurrent slot linear_attention reads: num and den, [heads, head_dim].
json linear_attention_state(const std::string& name, size_t heads, size_t head_dim);
// ins[0] a list, ins[1..] tensors; weight = the pseudo-query [d].
json depth_attend(const std::vector<std::string>& ins, const std::string& weight, float eps, const std::string& out);
json list(const std::vector<std::string>& ins, const std::string& out);
json append(const std::string& list, const std::string& x, const std::string& out);
// prefix + "ssm_a.weight" etc. hold A [n, n], b, c [n], d [1].
json ssm_scan(const std::string& in, const std::string& state, const std::string& prefix, size_t n,
              const std::string& out);
json ssm_state(const std::string& name, size_t n);
// A runtime-defined op: text in the grammar of microtorch/expr.hpp.
json expr(const std::vector<std::string>& ins, const std::string& text, const std::string& out);
// A recurrence (IR_SPEC.md 6.10, microtorch/recurrence.hpp): `body` holds index, inputs and
// update, readout, and optionally weights, init and let; in, state and out are added. Its slot:
// one tensor per update assignment, with that assignment's shape, init "body" when the node's
// init assigns it, else 0 (throws std::invalid_argument when the node does not check).
json recurrence(const std::vector<std::string>& ins, const std::string& state, const json& body,
                const std::string& out);
json recurrence_state(const std::string& name, const json& node);
// Section 6.4's elu+1 linear attention as a recurrence body (IR_SPEC.md 6.10.6), den_floor 1e-8,
// over inputs q, k, v [heads, head_dim]: the text spec/tools/ir_recurrence.py's rewrite writes.
json linear_attention_body(size_t heads, size_t head_dim);

}  // namespace ir
}  // namespace microtorch
