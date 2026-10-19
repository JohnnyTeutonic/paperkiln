#pragma once
// Mixture-of-experts SwiGLU feed-forward layer (19 Oct 2026): the first
// step towards sparse-upcycling a trained dense llama-family model.
//
//   logits = x R                       router, Linear d -> E, no bias
//   S_t    = top-k of logits[t, :]     ties go to the lower expert index
//   g_t    = gates over S_t            (convention below)
//   y_t    = sum_{e in S_t} g_t,e * expert_e(x_t)
//   expert_e(x) = down_e( silu(gate_e(x)) * up_e(x) )   (the dense FFN's shapes)
//
// Gate convention (MoEConfig::renormalize):
//   true  (Mixtral): g_t = softmax over the k selected LOGITS, which equals
//         the full softmax renormalised over S_t. Gates sum to 1, so
//         identical experts reproduce the dense FFN for ANY router -- the
//         property sparse upcycling starts from. With k = 1 the single gate
//         is the constant 1 and the task loss gives the router no gradient
//         (only the auxiliary losses train it), so use k >= 2 with this.
//   false (Switch):  g_t,e = softmax(logits[t, :])_e, the raw probability.
//         With k = 1 this is Switch Transformer routing; the output is
//         scaled by p, so an upcycled model does NOT start at the dense
//         function under this convention.
// llama.cpp's llama-architecture MoE (Mixtral) renormalises, so only
// renormalize = true exports faithfully to GGUF.
//
// Composition: built on the tape from ops::gather_rows / scatter_add_rows /
// gather_elems plus the existing Linear/silu/mul/mul_col/softmax_row (see
// ops.hpp for why this is not one fused op). Tokens are dispatched per
// expert, so compute scales with k, not E.
//
// Auxiliary losses (per MoE layer, returned through MoEAux, not added here):
//   load balance  lb = E * sum_e f_e * P_e      (Switch Transformer eq. 4)
//     f_e = (#(token, slot) pairs routed to e) / (T k)   -- a constant
//     P_e = mean_t softmax(logits)[t, e]                 -- differentiable
//     lb = 1 under perfectly uniform routing. (HF's Mixtral loss divides
//     by T rather than T k, so it equals k times this.)
//   router z-loss z = mean_t logsumexp(logits[t, :])^2  (ST-MoE, Zoph 2022)
#include <map>
#include <string>
#include <vector>

#include "microtorch/nn.hpp"

namespace microtorch {
namespace nn {

struct MoEConfig {
    size_t experts = 0;  // 0 = dense FFN (MoE off)
    size_t top_k = 1;
    bool renormalize = true;  // see the gate convention above
};

class SwiGLUExpert : public Module {
public:
    // Seeds base, base+1, base+2 for gate/up/down: the dense LlamaBlock's
    // own init order, so expert 0 of an E-expert layer built from the same
    // base seed starts equal to the dense FFN.
    SwiGLUExpert(size_t d, size_t d_ff, unsigned seed_base);
    Var forward(const Var& x) const;
    std::shared_ptr<Linear> gate_proj, up_proj, down_proj;
};

// Per-forward MoE record, filled by every MoE layer the forward passes
// through when a pointer is supplied (training only; eval passes none).
struct MoEAux {
    Var lb, z;          // SUMS over the MoE layers seen (null before the first)
    size_t layers = 0;  // number of MoE layers that contributed
    // Per layer: fraction of (token, slot) assignments per expert (sums to 1).
    std::vector<std::vector<double>> frac;
};

// Normalised entropy of a routing-fraction vector, H(f) / log E in [0, 1]:
// 1 = perfectly balanced, 0 = every token on one expert (collapse).
double routing_entropy(const std::vector<double>& frac);

class MoEFFN : public Module {
public:
    // Names: "router.weight" [d, E] and "experts.{e}.{gate,up,down}_proj.weight",
    // so inside a llama block (registered as "mlp") the safetensors keys are
    // layers.N.mlp.router.weight and layers.N.mlp.experts.{e}.gate_proj.weight.
    MoEFFN(size_t d, size_t d_ff, const MoEConfig& cfg, unsigned seed);
    Var forward(const Var& x, MoEAux* aux = nullptr) const;

    MoEConfig cfg;
    std::shared_ptr<Linear> router;
    std::vector<std::shared_ptr<SwiGLUExpert>> experts;
};

// Sparse upcycling (Komatsuzaki et al. 2022): map a DENSE llama state_dict
// onto an E-expert MoE llama's names. Every layers.N.mlp.{gate,up,down}_proj.weight
// is copied into each layers.N.mlp.experts.{e}.*; every other tensor passes
// through unchanged. Router weights are not produced: their names are
// appended to *router_keys (pass them as load_state_dict's missing_ok, so
// the routers keep their fresh init). Throws if the dict holds no dense MLP.
std::map<std::string, Matrix> upcycle_state_dict(const std::map<std::string, Matrix>& dense,
                                                 size_t n_experts,
                                                 std::vector<std::string>* router_keys);

}  // namespace nn
}  // namespace microtorch
