#pragma once
// Phase 1b: Module/Parameter/state_dict and the layer zoo, composed
// entirely from ops::* so no layer owns any new calculus -- a layer's
// backward is correct because the tape's ops are gradchecked, which is the
// whole point of doing 1a first.
//
// Naming follows the torch convention (dotted paths, "weight"/"bias") so
// that 1c's safetensors loader maps HF checkpoints onto modules by name
// with no translation table beyond an optional prefix strip.
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "microtorch/ops.hpp"

namespace microtorch {
namespace nn {

class Module {
public:
    virtual ~Module() = default;

    // Dotted-path collection over the registration tree.
    std::vector<std::pair<std::string, Var>> named_parameters() const;
    std::vector<Var> parameters() const;
    // Total scalar parameter count — the first Atlas structural feature
    // (atlas/ARCHITECTURE_ATLAS.md stage 0).
    size_t parameter_count() const;
    std::map<std::string, Matrix> state_dict() const;
    // strict: every entry must land on a parameter and every parameter must
    // be hit -- loading a frontier checkpoint should fail loudly, not
    // half-load (missing_ok lists names allowed to stay untouched).
    void load_state_dict(const std::map<std::string, Matrix>& sd, bool strict = true,
                         const std::vector<std::string>& missing_ok = {});

    void train() { set_training(true); }
    void eval() { set_training(false); }
    bool training() const { return training_; }

protected:
    Var reg(const std::string& name, Matrix init);  // register param
    template <typename M, typename... A>
    std::shared_ptr<M> mod(const std::string& name, A&&... a) {  // register child
        auto m = std::make_shared<M>(std::forward<A>(a)...);
        children_.emplace_back(name, m);
        return m;
    }
    // Register an EXISTING module as a child (AttnResStack wraps layers
    // constructed by the caller; mod<> only covers in-place construction).
    void adopt(const std::string& name, std::shared_ptr<Module> child) {
        children_.emplace_back(name, std::move(child));
    }

private:
    void collect(const std::string& prefix, std::vector<std::pair<std::string, Var>>& out) const;
    void set_training(bool t);
    std::vector<std::pair<std::string, Var>> params_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> children_;
    bool training_ = true;
};

class Linear : public Module {
public:
    // W stored [in, out], y = x W (+ b). This is ALSO HF-GPT-2's Conv1D
    // storage order, so its checkpoints load without transposition.
    Linear(size_t in, size_t out, bool bias = true, unsigned seed = 0);
    Var forward(const Var& x) const;
    Var W, b;  // b empty when bias=false
};

class LayerNorm : public Module {
public:
    explicit LayerNorm(size_t d, float eps = 1e-5f);
    Var forward(const Var& x) const;
    Var weight, bias;
    float eps;
};

class Embedding : public Module {
public:
    Embedding(size_t vocab, size_t d, unsigned seed = 0);
    Var forward(const std::vector<int>& ids) const;
    Var weight;
};

// Pre-LN self-attention (GPT-2 layout): fused qkv projection, heads split
// by slice_cols, output projection. Everything differentiable is a
// composition of checked ops. `causal` (default true) adds the additive
// -1e9 mask; DiT passes false, since patches attend bidirectionally.
class CausalSelfAttention : public Module {
public:
    CausalSelfAttention(size_t d, size_t n_heads, unsigned seed = 0, bool causal = true);
    // x: [T, d] for one sequence, or [B*T, d] for a stacked mini-batch
    // with seq_len = T (sequences are isolated by a block-diagonal mask;
    // seq_len 0 = single sequence spanning all rows).
    Var forward(const Var& x, size_t seq_len = 0) const;
    std::shared_ptr<Linear> c_attn, c_proj;
    size_t H, dk;
    bool causal;
};

// Sparse-phase S1 baseline: sliding-window attention with sinks.
// Interface-identical to CausalSelfAttention (same qkv/proj layout, same
// seeds), differing only in the attention op — swap one line to compare.
class SlidingWindowAttention : public Module {
public:
    SlidingWindowAttention(size_t d, size_t n_heads, size_t window, size_t sinks = 1,
                           unsigned seed = 0);
    Var forward(const Var& x, size_t seq_len = 0) const;
    std::shared_ptr<Linear> c_attn, c_proj;
    size_t H, dk, window, sinks;
};

class KimiLinearAttention : public Module {  // Phase 3a: O(n*d²) linear-time attention
public:
    KimiLinearAttention(size_t d, size_t n_heads, unsigned seed = 0, bool causal = true);
    // seq_len > 0: stacked mini-batch; prefix sums reset per block.
    Var forward(const Var& x, size_t seq_len = 0) const;  // x: [T, d] -> [T, d]
    std::shared_ptr<Linear> c_attn, c_proj;
    size_t H, dk;
    bool causal;
    // Note: forward() internally calls ops::kimi_attention() instead of
    // standard scaled-dot-product; q,k,v projections are identical to
    // CausalSelfAttention, only the attention mechanism differs
};

class MLP : public Module {
public:
    MLP(size_t d, size_t hidden, unsigned seed = 0);
    Var forward(const Var& x) const;
    std::shared_ptr<Linear> c_fc, c_proj;
};

class Block : public Module {  // pre-LN transformer block, GPT-2 wiring
public:
    Block(size_t d, size_t n_heads, unsigned seed = 0);
    Var forward(const Var& x, size_t seq_len = 0) const;
    std::shared_ptr<LayerNorm> ln_1, ln_2;
    std::shared_ptr<CausalSelfAttention> attn;
    std::shared_ptr<MLP> mlp;
};

struct GPT2Config {
    size_t vocab = 50257, n_ctx = 1024, d = 768, n_layers = 12, n_heads = 12;
};

class GPT2 : public Module {
public:
    explicit GPT2(const GPT2Config& cfg, unsigned seed = 0);
    // ids: one sequence, or B sequences of seq_len concatenated
    // (positions restart every seq_len; attention is block-isolated).
    // -> logits [T, vocab] / [B*T, vocab]
    Var forward(const std::vector<int>& ids, size_t seq_len = 0) const;
    GPT2Config cfg;
    // Activation checkpointing per transformer block: intermediates are
    // rematerialized on backward instead of stored (see autograd.hpp).
    bool checkpoint_blocks = false;
    std::shared_ptr<Embedding> wte, wpe;
    std::vector<std::shared_ptr<Block>> h;
    std::shared_ptr<LayerNorm> ln_f;
};

// ---- optimizers (over the parameter list, matrices and rows alike) ----
class SGD {
public:
    SGD(std::vector<Var> params, float lr, float momentum = 0.0f);
    void step();
    void zero_grad();
    float lr;

private:
    std::vector<Var> params_;
    std::vector<Matrix> vel_;
    float mu_;
};

class AdamW {
public:
    AdamW(std::vector<Var> params, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float weight_decay = 0.01f);
    void step();
    void zero_grad();
    float lr;

private:
    std::vector<Var> params_;
    std::vector<Matrix> m_, v_;
    float b1_, b2_, eps_, wd_;
    long t_ = 0;
};

// Attention Residuals (AttnRes) — TECH_TRANSFER item 1; Kimi K3 arXiv
// 2607.24653 section 2.2 Eq. 8-10; reference anchor python/attn_res_reference.py.
//
// A residual stream compresses every earlier layer into one running sum —
// an RNN over depth. AttnRes replaces that with ATTENTION over depth:
// layer l's input is a softmax mixture of the embedding and all preceding
// layer outputs under a learnable per-layer pseudo-query w_l. Keys are
// parameter-free-RMSNormed sources; values are the raw sources
// (Eq. 9). Block form (Eq. 10): outputs SUM within blocks of block_size
// layers and attention runs across banked block representations only,
// dropping memory from O(L d) to O(N d).
//
// Composed entirely from gradchecked tape ops (rmsnorm, mul_row, matmul,
// softmax_row, mul_col, ...) so this class owns no new calculus.
// Pseudo-queries init to ZERO: depth-attention starts uniform — the
// nearest AttnRes analogue of a plain residual stream — and the softmax
// gradient is nonzero there (asserted in test_attn_res).
//
// Layers are passed twice, deliberately: `owned` registers them so their
// parameters appear in state_dict/parameters(), and `fns` is how they
// are called (Module has no virtual forward). Callers keep them aligned.
class AttnResStack : public Module {
public:
    // block_size 0 = full form (Eq. 8-9); >= 1 = block form (Eq. 10).
    // STRUCTURAL FACT the tests pin: block_size 1 must equal the full
    // form exactly (every block is one layer; the partial-sum branch
    // never fires).
    AttnResStack(std::vector<std::shared_ptr<Module>> owned,
                 std::vector<std::function<Var(const Var&)>> fns, size_t d, size_t block_size = 0);
    Var forward(const Var& h1) const;

    std::vector<Var> w;  // pseudo-queries [1, d]: one per layer + final
    size_t block_size;

private:
    std::vector<std::function<Var(const Var&)>> fns_;
    Var depth_attend(const Var& wq, const std::vector<Var>& sources) const;
    Var ones_gamma_, ones_col_;  // no-grad constants for the composition
};

// Quintic Newton-Schulz orthogonalization (Muon's core; coefficients per
// Keller Jordan's Muon). Drives the singular values of a 2-D matrix into
// a loose band around 1 while preserving the singular DIRECTIONS —
// approximately UV^T of the input's SVD. Iterates on the wide
// orientation internally; commutes with transpose. Reference anchor:
// python/muon_reference.py (golden-pinned in test_muon).
Matrix newton_schulz5(const Matrix& G, int steps = 5);

// Muon optimizer with the K3 per-head refinement (TECH_TRANSFER item 3;
// arXiv 2607.24653 section 2.5, on Keller Jordan's base Muon).
//
//   M_t = mu*M_{t-1} + G_t;  U = G_t + mu*M_t (nesterov) else M_t
//   O   = newton_schulz5(U);  W -= lr * sqrt(max(1, out/in)) * O
//
// LAYOUT NOTE, load-bearing: microtorch Linear stores W as [in, out]
// (x @ W), the transpose of the PyTorch reference's [out, in]. Heads
// therefore live along COLUMNS here, so n_heads partitions the COLUMN
// dimension, and the shape scale is sqrt(max(1, cols/rows)) per block —
// the exact mirror of the reference (newton_schulz5 commutes with
// transpose, so the two formulations produce transposed-identical
// updates). n_heads=1 is full-matrix Muon.
//
// Muon is for HIDDEN MATRIX parameters only. Vectors (biases, norm
// gains) are refused; embeddings/heads should go to AdamW by routing
// discipline, as in every deployment this is transferred from.
class Muon {
public:
    Muon(std::vector<Var> params, float lr = 0.02f, float momentum = 0.95f, bool nesterov = true,
         int ns_steps = 5, size_t n_heads = 1);
    void step();
    void zero_grad();
    float lr;

private:
    std::vector<Var> params_;
    std::vector<Matrix> buf_;
    float mu_;
    bool nesterov_;
    int ns_;
    size_t H_;
};

// Dropout as a module so `training()` decides train/eval behavior the way
// torch.nn.Dropout does; eval mode is the identity. Each forward draws a
// fresh op seed from the module's own counter, so two calls in one step
// get independent masks but a fixed module seed keeps runs reproducible.
class Dropout : public Module {
public:
    explicit Dropout(float p, unsigned long long seed = 0) : p_(p), next_seed_(seed) {}
    Var forward(const Var& x) const;

private:
    float p_;
    mutable unsigned long long next_seed_;
};

// ---- LR schedulers (mirror torch.optim.lr_scheduler; templated on the
// optimizer because SGD/AdamW share only a public `lr` field, not a base) --

// Linear warmup for `warmup` steps then cosine decay to min_lr over
// `total` steps -- the schedule every GPT-family training recipe uses.
template <typename Opt>
class CosineWarmupLR {
public:
    CosineWarmupLR(Opt& opt, size_t warmup, size_t total, float min_lr = 0.0f)
        : opt_(opt), base_lr_(opt.lr), warmup_(warmup), total_(total), min_lr_(min_lr) {}
    void step() {
        ++t_;
        if (warmup_ > 0 && t_ <= warmup_) {
            opt_.lr = base_lr_ * static_cast<float>(t_) / warmup_;
            return;
        }
        const float progress =
            total_ > warmup_ ? static_cast<float>(t_ - warmup_) / (total_ - warmup_) : 1.0f;
        const float clamped = progress > 1.0f ? 1.0f : progress;
        opt_.lr =
            min_lr_ + 0.5f * (base_lr_ - min_lr_) * (1.0f + std::cos(3.14159265358979f * clamped));
    }
    size_t current_step() const { return t_; }

private:
    Opt& opt_;
    float base_lr_;
    size_t warmup_, total_, t_ = 0;
    float min_lr_;
};

// Multiply lr by gamma every step_size steps (torch's StepLR).
template <typename Opt>
class StepLR {
public:
    StepLR(Opt& opt, size_t step_size, float gamma = 0.1f)
        : opt_(opt), base_lr_(opt.lr), step_size_(step_size), gamma_(gamma) {}
    void step() {
        ++t_;
        float lr = base_lr_;
        for (size_t k = 0; k < t_ / step_size_; ++k) lr *= gamma_;
        opt_.lr = lr;
    }

private:
    Opt& opt_;
    float base_lr_;
    size_t step_size_, t_ = 0;
    float gamma_;
};

}  // namespace nn
}  // namespace microtorch
