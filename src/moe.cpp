#include "microtorch/moe.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "microtorch/device.hpp"
#include "microtorch/ir.hpp"
#include "microtorch/ops.hpp"

namespace microtorch {
namespace nn {

SwiGLUExpert::SwiGLUExpert(size_t d, size_t d_ff, unsigned s) {
    gate_proj = mod<Linear>("gate_proj", d, d_ff, false, s);
    up_proj = mod<Linear>("up_proj", d, d_ff, false, s + 1);
    down_proj = mod<Linear>("down_proj", d_ff, d, false, s + 2);
}

Var SwiGLUExpert::forward(const Var& x) const {
    // The dense LlamaBlock expression, verbatim.
    return down_proj->forward(ops::swiglu(gate_proj->forward(x), up_proj->forward(x)));
}

double routing_entropy(const std::vector<double>& frac) {
    if (frac.size() < 2) return 1.0;
    double h = 0.0;
    for (double f : frac)
        if (f > 0.0) h -= f * std::log(f);
    return h / std::log(static_cast<double>(frac.size()));
}

MoEFFN::MoEFFN(size_t d, size_t d_ff, const MoEConfig& c, unsigned seed) : cfg(c) {
    if (cfg.experts == 0) throw std::runtime_error("MoEFFN: experts must be >= 1");
    if (cfg.top_k == 0 || cfg.top_k > cfg.experts)
        throw std::runtime_error("MoEFFN: top_k must be in [1, experts]");
    // Router init: the Linear default (N(0, 0.02^2)), NOT zeros. Zero logits
    // tie every expert, the tie rule then sends every token to experts
    // 0..k-1, and the others receive no tokens (hence no gradient) until the
    // load-balance loss has moved the router. A small random router spreads
    // tokens from step 0; for an upcycled model with renormalised gates the
    // output is independent of the router at init, so the random draw costs
    // nothing in function (tests/test_moe.cpp, upcycling check).
    router = mod<Linear>("router", d, cfg.experts, false, seed + 3);
    // Expert e's seeds: base + 10007 e. Expert 0 therefore uses the dense
    // FFN's seeds (the block passes its dense base, seed + 5).
    for (size_t e = 0; e < cfg.experts; ++e)
        experts.push_back(mod<SwiGLUExpert>("experts." + std::to_string(e), d, d_ff,
                                            seed + 10007u * static_cast<unsigned>(e)));
}

ir::json MoEFFN::export_ir(const std::string& in, const std::string& out, const std::string& prefix) const {
    // The forward below, as the IR states it: gates from the k selected
    // logits (renorm) or the full softmax (raw); experts summed by index.
    const bool renorm = cfg.renormalize;
    return {{"op", "moe_ffn"},
            {"in", ir::json::array({in})},
            {"router", prefix + "ffn_gate_inp.weight"},
            {"experts",
             {{"gate", prefix + "ffn_gate_exps.weight"},
              {"up", prefix + "ffn_up_exps.weight"},
              {"down", prefix + "ffn_down_exps.weight"}}},
            {"n_experts", cfg.experts},
            {"top_k", cfg.top_k},
            {"gate_norm", renorm ? "renorm" : "raw"},
            {"activation", "silu"},
            {"routing", renorm ? "softmax over E logits, top-k (ties to the lower index), renormalise "
                                 "the k probabilities, sum of gate * SwiGLU_e(x)"
                               : "softmax over E logits, top-k (ties to the lower index), "
                                 "sum of p * SwiGLU_e(x)"},
            {"_note",
             "paperkiln computes the renormalised gates as a softmax over the k selected logits (the same "
             "quantity) and accumulates the expert outputs in ascending expert index"},
            {"out", out}};
}

Var MoEFFN::forward(const Var& x, MoEAux* aux) const {
    const size_t T = x->data.rows(), E = cfg.experts, k = cfg.top_k;
    Var logits = router->forward(x);  // [T, E]
    // Host-read: top-k selection. The routing has to reach the host anyway
    // (each expert's token list sets its matrix shapes), and [T, E] logits
    // are a few hundred KB; everything after the selection stays on device.
    device::materialize(logits->data);

    // Top-k per token: descending logit, ties to the lower index. Selection
    // by logits equals selection by softmax probabilities (monotone).
    std::vector<int> rows(T * k), cols(T * k);
    std::vector<int> order(E);
    for (size_t t = 0; t < T; ++t) {
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) {
            const float la = logits->data(t, a), lb = logits->data(t, b);
            return la > lb || (la == lb && a < b);
        });
        for (size_t j = 0; j < k; ++j) {
            rows[t * k + j] = static_cast<int>(t);
            cols[t * k + j] = order[j];
        }
    }

    // Gates [T, k]. Renormalised: softmax over the selected logits, which
    // is softmax(logits) restricted to S_t and renormalised, with no
    // division op needed. Raw: the full softmax's selected entries.
    Var probs;  // full softmax, built once if either path needs it
    Var gates;
    if (cfg.renormalize) {
        gates = ops::softmax_row(ops::gather_elems(logits, rows, cols, k));
    } else {
        probs = ops::softmax_row(logits);
        gates = ops::gather_elems(probs, rows, cols, k);
    }

    // Dispatch: per expert, its tokens (ascending) and their slot in the gate row.
    std::vector<std::vector<int>> tok(E), slot(E);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < k; ++j) {
            const int e = cols[t * k + j];
            tok[e].push_back(static_cast<int>(t));
            slot[e].push_back(static_cast<int>(j));
        }
    // Combine: one scatter-add of every expert's gated rows, ascending
    // expert index (the per-token sum order a chain of per-expert scatters
    // and adds gives, without E dense [T, d] intermediates).
    std::vector<Var> contribs;
    std::vector<std::vector<int>> contrib_tok;
    for (size_t e = 0; e < E; ++e) {
        if (tok[e].empty()) continue;  // no tokens: no compute, no gradient
        Var xe = ops::gather_rows(x, tok[e]);
        Var ye = experts[e]->forward(xe);
        Var ge = ops::gather_elems(gates, tok[e], slot[e], 1);  // [n_e, 1]
        contribs.push_back(ops::mul_col(ye, ge));
        contrib_tok.push_back(tok[e]);
    }
    Var out = ops::scatter_add_rows(contribs, contrib_tok, T);

    if (aux) {
        std::vector<double> frac(E, 0.0);
        for (int e : cols) frac[e] += 1.0;
        Matrix coef(E, 1);  // E * f_e, a constant column
        for (size_t e = 0; e < E; ++e) {
            frac[e] /= static_cast<double>(T * k);
            coef(e, 0) = static_cast<float>(static_cast<double>(E) * frac[e]);
        }
        if (!probs) probs = ops::softmax_row(logits);
        // lb = E sum_e f_e mean_t P[t,e] = mean_t (P coef)[t]
        Var lb = ops::mean(ops::matmul(probs, make_var(std::move(coef))));
        Var lse = ops::logsumexp_row(logits);
        Var z = ops::mean(ops::mul(lse, lse));
        aux->lb = aux->lb ? ops::add(aux->lb, lb) : lb;
        aux->z = aux->z ? ops::add(aux->z, z) : z;
        ++aux->layers;
        aux->frac.push_back(std::move(frac));
    }
    return out;
}

std::map<std::string, Matrix> upcycle_state_dict(const std::map<std::string, Matrix>& dense,
                                                 size_t n_experts,
                                                 std::vector<std::string>* router_keys) {
    if (n_experts == 0) throw std::runtime_error("upcycle_state_dict: n_experts must be >= 1");
    static const char* kProj[] = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"};
    std::map<std::string, Matrix> out;
    size_t mapped = 0;
    for (const auto& [key, m] : dense) {
        bool is_mlp = false;
        for (const char* p : kProj) {
            const std::string tail = std::string("mlp.") + p;
            if (key.size() >= tail.size() && key.compare(key.size() - tail.size(), tail.size(), tail) == 0) {
                const std::string prefix = key.substr(0, key.size() - tail.size());  // "layers.N."
                for (size_t e = 0; e < n_experts; ++e)
                    out.emplace(prefix + "mlp.experts." + std::to_string(e) + "." + p, m);
                if (std::string(p) == "gate_proj.weight" && router_keys)
                    router_keys->push_back(prefix + "mlp.router.weight");
                is_mlp = true;
                ++mapped;
            }
        }
        if (!is_mlp) out.emplace(key, m);
    }
    if (mapped == 0) throw std::runtime_error("upcycle_state_dict: no dense mlp.{gate,up,down}_proj weights found");
    return out;
}

}  // namespace nn
}  // namespace microtorch
