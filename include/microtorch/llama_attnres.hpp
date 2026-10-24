#pragma once
// Attention Residuals on the llama block (nn::LlamaConfig::AttnRes; spec
// arch.custom.attn_res; family_hint "llama_attnres";
// experiments/tiny_chat_long_2026-10/IR_SPEC_DRAFT.md section 2, cell X).
//
// The residual stream becomes a list of sources ("bank"): the token
// embedding, then every attention and every FFN sublayer output, in order
// (2 A + 1 sources after A block applications). Each sublayer's input is
// attention over depth (ops::depth_attend) with its own pseudo-query
// against the RMSNormed sources, and the head reads one more:
//   bank = [e]
//   application a (stored layer m_a = schedule[a]):
//     h  = mix(bank; attn_q[a]);  bank += attn(rmsnorm(h; attn_norm))
//     h2 = mix(bank; ffn_q[a]);   bank += ffn(rmsnorm(h2; ffn_norm))
//   logits = head(rmsnorm(mix(bank; out_q); output_norm))
// mix(bank; q) = depth_attend(bank; q) * S under scale "sources" (S =
// bank.size(); the IR's depth_attend then affine(S, 0)), depth_attend alone
// under "none" (Kimi's convex mixture, as nn::AttnResStack). Queries start
// at ZERO: uniform depth weights, so under "sources" every sublayer input
// is S * (sum of the sources) / S = the plain model's residual stream x, and
// the model starts as the plain llama of the same seed to rounding
// (tests/test_llama_attnres.cpp; "none" divides every sublayer input by S,
// which the RMSNorms undo only up to their epsilon: rmsnorm(x / S) =
// x / sqrt(mean(x^2) + S^2 eps), a large change at init when the stream's
// mean square is near S^2 eps).
//
// attn_q[0] would attend over one source (the embedding) and is never read,
// so it is not a parameter (exported as zeros). Under fixed recursion every
// application has its own two queries (the IR's "tensor_index":
// "application"; IR_SPEC_DRAFT section 3); schedule "mor", inject "add",
// MoE and state tokens are refused.
#include <memory>
#include <vector>

#include "microtorch/llama.hpp"

namespace microtorch {
namespace nn {

class AttnResQueries : public Module {
public:
    AttnResQueries(size_t d, size_t applications);
    std::vector<Var> attn_q, ffn_q;  // per application; attn_q[0] is null
    Var out_q;
};

// Throws for the combinations the attention-residual stream does not
// support (schedule mor, inject add, MoE).
void attnres_check(const LlamaConfig& cfg);
Var attnres_forward(const Llama& m, const std::vector<int>& ids, size_t seq_len, MoEAux* aux);
Var attnres_forward_cached(const Llama& m, const std::vector<int>& ids, LlamaKV& kv);
ir::json attnres_export_ir(const Llama& m);
// The GGUF names of the queries, state-dict key -> GGUF name: blk.{a}.attnres_attn_q.weight (zeros for
// a = 0, written by the caller), blk.{a}.attnres_ffn_q.weight and attnres_out_q.weight; under recursion
// rec_app.{a}.* (one per application).
std::vector<std::pair<std::string, std::string>> attnres_gguf_names(const Llama& m);
// The GGUF name of application 0's attention query, never read (written as zeros).
std::string attnres_unused_query_name(const Llama& m);

}  // namespace nn
}  // namespace microtorch
