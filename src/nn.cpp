#include "microtorch/nn.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

#include "microtorch/device.hpp"

namespace microtorch {
namespace nn {

namespace {

Matrix randn(size_t r, size_t c, unsigned seed, float std) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, std);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(gen);
    return m;
}

}  // namespace

// ---- Module ---------------------------------------------------------------

Var Module::reg(const std::string& name, Matrix init) {
    Var v = make_var(std::move(init), /*requires_grad=*/true);
    params_.emplace_back(name, v);
    return v;
}

void Module::collect(const std::string& prefix,
                     std::vector<std::pair<std::string, Var>>& out) const {
    for (const auto& [n, p] : params_) out.emplace_back(prefix + n, p);
    // Recurse through the CHILD, not this. Calling collect() bare here
    // self-recurses forever -- found as a stack overflow on the first run.
    for (const auto& [n, c] : children_) c->collect(prefix + n + ".", out);
}

// A little indirection so collect() can recurse through the child pointer.
std::vector<std::pair<std::string, Var>> Module::named_parameters() const {
    std::vector<std::pair<std::string, Var>> out;
    collect("", out);
    return out;
}

std::vector<Var> Module::parameters() const {
    std::vector<Var> out;
    for (auto& [n, p] : named_parameters()) out.push_back(p);
    return out;
}

size_t Module::parameter_count() const {
    size_t n = 0;
    for (const auto& p : parameters()) n += p->data.rows() * p->data.cols();
    return n;
}

std::map<std::string, Matrix> Module::state_dict() const {
    std::map<std::string, Matrix> sd;
    for (auto& [n, p] : named_parameters()) sd.emplace(n, p->data);
    return sd;
}

void Module::load_state_dict(const std::map<std::string, Matrix>& sd, bool strict,
                             const std::vector<std::string>& missing_ok) {
    auto named = named_parameters();
    std::map<std::string, Var> byname(named.begin(), named.end());
    size_t hit = 0;
    for (const auto& [name, m] : sd) {
        auto it = byname.find(name);
        if (it == byname.end()) {
            if (strict) throw std::runtime_error("load_state_dict: unexpected key " + name);
            continue;
        }
        if (m.rows() != it->second->data.rows() || m.cols() != it->second->data.cols()) {
            throw std::runtime_error("load_state_dict: shape mismatch at " + name);
        }
        it->second->data = m;
        ++hit;
    }
    if (strict) {
        for (auto& [name, p] : named) {
            bool ok = sd.count(name) > 0;
            for (const auto& allow : missing_ok) ok = ok || name == allow;
            if (!ok) throw std::runtime_error("load_state_dict: missing key " + name);
        }
        (void)hit;
    }
}

void Module::set_training(bool t) {
    training_ = t;
    for (auto& [n, c] : children_) c->set_training(t);
}

// ---- layers ---------------------------------------------------------------

Linear::Linear(size_t in, size_t out, bool bias, unsigned seed) {
    // 0.02 init, the GPT-2 convention; overwritten anyway when loading.
    W = reg("weight", randn(in, out, seed * 7919u + 1u, 0.02f));
    if (bias) b = reg("bias", Matrix(1, out));
}

Var Linear::forward(const Var& x) const {
    Var y = ops::matmul(x, W);
    return b ? ops::add_bias(y, b) : y;
}

LayerNorm::LayerNorm(size_t d, float e) : eps(e) {
    weight = reg("weight", Matrix(1, d, 1.0f));
    bias = reg("bias", Matrix(1, d));
}

Var LayerNorm::forward(const Var& x) const {
    return ops::layernorm(x, weight, bias, eps);
}

Embedding::Embedding(size_t vocab, size_t d, unsigned seed) {
    weight = reg("weight", randn(vocab, d, seed * 104729u + 3u, 0.02f));
}

Var Embedding::forward(const std::vector<int>& ids) const {
    return ops::embedding(weight, ids);
}

CausalSelfAttention::CausalSelfAttention(size_t d, size_t n_heads, unsigned seed, bool causal_)
    : H(n_heads), dk(d / n_heads), causal(causal_) {
    if (d % n_heads != 0) {
        throw std::runtime_error("attention: d must divide by n_heads");
    }
    c_attn = mod<Linear>("c_attn", d, 3 * d, true, seed + 11);
    c_proj = mod<Linear>("c_proj", d, d, true, seed + 13);
}

Var CausalSelfAttention::forward(const Var& x, size_t seq_len) const {
    const size_t d = H * dk;
    Var qkv = c_attn->forward(x);  // [T, 3d]
    // One fused tape node per head: GEMMs on device::matmul,
    // scale+mask+softmax as a single in-place pass — the mask matrix is
    // never materialized (see ops::fused_attention; FD-checked).
    const float sc = 1.0f / std::sqrt(static_cast<float>(dk));
    std::vector<Var> heads;
    for (size_t h = 0; h < H; ++h) {
        Var q = ops::slice_cols(qkv, h * dk, (h + 1) * dk);
        Var k = ops::slice_cols(qkv, d + h * dk, d + (h + 1) * dk);
        Var v = ops::slice_cols(qkv, 2 * d + h * dk, 2 * d + (h + 1) * dk);
        heads.push_back(ops::fused_attention(q, k, v, sc, seq_len, causal));
    }
    return c_proj->forward(ops::concat_cols(heads));
}

// Sparse-phase S1 baseline: identical layout and seeds to
// CausalSelfAttention — the ONLY difference is the attention op, so any
// behavioural difference is the sparsity pattern and nothing else.
SlidingWindowAttention::SlidingWindowAttention(size_t d, size_t n_heads, size_t window_,
                                               size_t sinks_, unsigned seed)
    : H(n_heads), dk(d / n_heads), window(window_), sinks(sinks_) {
    if (d % n_heads != 0) {
        throw std::runtime_error("attention: d must divide by n_heads");
    }
    c_attn = mod<Linear>("c_attn", d, 3 * d, true, seed + 11);
    c_proj = mod<Linear>("c_proj", d, d, true, seed + 13);
}

Var SlidingWindowAttention::forward(const Var& x, size_t seq_len) const {
    const size_t d = H * dk;
    Var qkv = c_attn->forward(x);
    const float sc = 1.0f / std::sqrt(static_cast<float>(dk));
    std::vector<Var> heads;
    for (size_t h = 0; h < H; ++h) {
        Var q = ops::slice_cols(qkv, h * dk, (h + 1) * dk);
        Var k = ops::slice_cols(qkv, d + h * dk, d + (h + 1) * dk);
        Var v = ops::slice_cols(qkv, 2 * d + h * dk, 2 * d + (h + 1) * dk);
        heads.push_back(ops::swa_attention(q, k, v, sc, window, sinks, seq_len));
    }
    return c_proj->forward(ops::concat_cols(heads));
}

// Phase 3a: Kimi Linear Attention (O(n*d²) vs O(n²*d) standard attention)
KimiLinearAttention::KimiLinearAttention(size_t d, size_t n_heads, unsigned seed, bool causal_)
    : H(n_heads), dk(d / n_heads), causal(causal_) {
    if (d % n_heads != 0) {
        throw std::runtime_error("attention: d must divide by n_heads");
    }
    // Identical to CausalSelfAttention: same q,k,v projection + output projection
    c_attn = mod<Linear>("c_attn", d, 3 * d, true, seed + 11);
    c_proj = mod<Linear>("c_proj", d, d, true, seed + 13);
}

Var KimiLinearAttention::forward(const Var& x, size_t seq_len) const {
    const size_t d = H * dk;
    Var qkv = c_attn->forward(x);  // [T, 3d]

    std::vector<Var> heads;
    for (size_t h = 0; h < H; ++h) {
        Var q = ops::slice_cols(qkv, h * dk, (h + 1) * dk);
        Var k = ops::slice_cols(qkv, d + h * dk, d + (h + 1) * dk);
        Var v = ops::slice_cols(qkv, 2 * d + h * dk, 2 * d + (h + 1) * dk);
        // Only difference from CausalSelfAttention: use kimi_attention instead of
        // scaled-dot-product (matmul -> softmax -> matmul)
        // Kimi Linear: Feature map (elu+1) -> cumsum numerator/denominator
        heads.push_back(ops::kimi_attention(q, k, v, causal, seq_len));
    }
    return c_proj->forward(ops::concat_cols(heads));
}

MLP::MLP(size_t d, size_t hidden, unsigned seed) {
    c_fc = mod<Linear>("c_fc", d, hidden, true, seed + 17);
    c_proj = mod<Linear>("c_proj", hidden, d, true, seed + 19);
}

Var MLP::forward(const Var& x) const {
    return c_proj->forward(ops::gelu(c_fc->forward(x)));
}

Block::Block(size_t d, size_t n_heads, unsigned seed) {
    ln_1 = mod<LayerNorm>("ln_1", d);
    attn = mod<CausalSelfAttention>("attn", d, n_heads, seed + 23);
    ln_2 = mod<LayerNorm>("ln_2", d);
    mlp = mod<MLP>("mlp", d, 4 * d, seed + 29);
}

Var Block::forward(const Var& x, size_t seq_len) const {
    Var y = ops::add(x, attn->forward(ln_1->forward(x), seq_len));
    return ops::add(y, mlp->forward(ln_2->forward(y)));
}

GPT2::GPT2(const GPT2Config& c, unsigned seed) : cfg(c) {
    wte = mod<Embedding>("wte", cfg.vocab, cfg.d, seed + 31);
    wpe = mod<Embedding>("wpe", cfg.n_ctx, cfg.d, seed + 37);
    for (size_t l = 0; l < cfg.n_layers; ++l) {
        h.push_back(mod<Block>("h." + std::to_string(l), cfg.d, cfg.n_heads,
                               seed + 41 + static_cast<unsigned>(l)));
    }
    ln_f = mod<LayerNorm>("ln_f", cfg.d);
}

Var GPT2::forward(const std::vector<int>& ids, size_t seq_len) const {
    // Positions restart every seq_len so each sequence in a stacked batch
    // sees positions 0..seq_len-1 (seq_len 0 = one sequence).
    const size_t sl = seq_len == 0 ? ids.size() : seq_len;
    if (sl > cfg.n_ctx) throw std::runtime_error("sequence too long");
    if (ids.size() % sl != 0) throw std::runtime_error("ids not a multiple of seq_len");
    std::vector<int> pos(ids.size());
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i % sl);
    Var x = ops::add(wte->forward(ids), wpe->forward(pos));
    for (const auto& blk : h) {
        x = checkpoint_blocks && grad_enabled()
                ? checkpoint([blk, seq_len](const Var& in) { return blk->forward(in, seq_len); }, x)
                : blk->forward(x, seq_len);
    }
    x = ln_f->forward(x);
    // Weight-tied head: logits = h wte^T (GPT-2 has no separate lm_head).
    return ops::matmul(x, ops::transpose(wte->weight));
}

// ---- optimizers -----------------------------------------------------------

SGD::SGD(std::vector<Var> params, float lr_, float momentum)
    : lr(lr_), params_(std::move(params)), mu_(momentum) {
    // Velocity only when momentum asks for it: plain SGD on GPT-2-sized
    // models must not carry 500 MB of zeros.
    if (mu_ != 0.0f) {
        for (const auto& p : params_) {
            vel_.emplace_back(p->data.rows(), p->data.cols());
        }
    }
}

void SGD::step() {
    // B2.3b: persistent device velocity, created once on the first step
    // (zeroed = the host init). nullptr pins the host path for the run —
    // NEVER mix paths mid-run: the two states would silently diverge.
    if (!devtried_ && mu_ != 0.0f) {
        devtried_ = true;
        size_t total = 0;
        devoff_.clear();
        for (const auto& p : params_) {
            devoff_.push_back(total);
            total += p->data.rows() * p->data.cols();
        }
        if (float* s = device::devops::opt_state_new(total))
            devstate_.reset(s, device::devops::opt_state_free);
    }
    for (size_t k = 0; k < params_.size(); ++k) {
        Var& p = params_[k];
        if (p->grad.rows() == 0) continue;
        if (devstate_) {
            if (device::devops::sgd_step_dev(p->data, p->grad,
                                             devstate_.get() + devoff_[k],
                                             lr, mu_))
                continue;
            // Device ops were disabled mid-run: the velocity lives on
            // device and the host copy is stale zeros. Refuse loudly
            // rather than silently fork the trajectory.
            throw std::runtime_error(
                "SGD: persistent device state exists but device ops are "
                "off — do not toggle MICROTORCH_DEVICE_OPS mid-run");
        }
        // B2.3a seam for the momentum-free case (no state to persist).
        if (mu_ == 0.0f &&
            device::devops::sgd_step(p->data, p->grad, nullptr, lr, mu_))
            continue;
        for (size_t i = 0; i < p->data.rows(); ++i)
            for (size_t j = 0; j < p->data.cols(); ++j) {
                float g = p->grad(i, j);
                if (mu_ != 0.0f) {
                    float& v = vel_[k](i, j);
                    v = mu_ * v + g;
                    g = v;
                }
                p->data(i, j) -= lr * g;
            }
    }
}

void SGD::zero_grad() {
    microtorch::zero_grad(params_);
}

AdamW::AdamW(std::vector<Var> params, float lr_, float b1, float b2, float eps, float wd)
    : lr(lr_), params_(std::move(params)), b1_(b1), b2_(b2), eps_(eps), wd_(wd) {
    for (const auto& p : params_) {
        m_.emplace_back(p->data.rows(), p->data.cols());
        v_.emplace_back(p->data.rows(), p->data.cols());
    }
}

void AdamW::step() {
    // Same math as the audited lmhead_adam_update_kernel, plus decoupled
    // weight decay.
    ++t_;
    const float c1 = 1.0f - std::pow(b1_, static_cast<float>(t_));
    const float c2 = 1.0f - std::pow(b2_, static_cast<float>(t_));
    // B2.3b: persistent device m+v, created once on the first step
    // (zeroed = the host init, so the trajectory is the host one).
    // nullptr pins the host path for the whole run — never mix mid-run.
    if (!devtried_) {
        devtried_ = true;
        devtotal_ = 0;
        devoff_.clear();
        for (const auto& p : params_) {
            devoff_.push_back(devtotal_);
            devtotal_ += p->data.rows() * p->data.cols();
        }
        if (float* s = device::devops::opt_state_new(2 * devtotal_))
            devstate_.reset(s, device::devops::opt_state_free);
    }
    for (size_t k = 0; k < params_.size(); ++k) {
        Var& p = params_[k];
        if (p->grad.rows() == 0) continue;
        if (devstate_) {
            if (device::devops::adamw_step_dev(
                    p->data, p->grad, devstate_.get() + devoff_[k],
                    devstate_.get() + devtotal_ + devoff_[k], lr, b1_, b2_,
                    c1, c2, eps_, wd_))
                continue;
            // Same loud-failure rule as SGD: m/v live on device; the
            // host matrices are stale zeros. Never silently fork.
            throw std::runtime_error(
                "AdamW: persistent device state exists but device ops are "
                "off — do not toggle MICROTORCH_DEVICE_OPS mid-run");
        }
        for (size_t i = 0; i < p->data.rows(); ++i)
            for (size_t j = 0; j < p->data.cols(); ++j) {
                const float g = p->grad(i, j);
                float& m = m_[k](i, j);
                float& v = v_[k](i, j);
                m = b1_ * m + (1.0f - b1_) * g;
                v = b2_ * v + (1.0f - b2_) * g * g;
                const float update = (m / c1) / (std::sqrt(v / c2) + eps_);
                p->data(i, j) -= lr * (update + wd_ * p->data(i, j));
            }
    }
}

void AdamW::zero_grad() {
    microtorch::zero_grad(params_);
}

// ---- AttnRes ---------------------------------------------------------------

AttnResStack::AttnResStack(std::vector<std::shared_ptr<Module>> owned,
                           std::vector<std::function<Var(const Var&)>> fns, size_t d,
                           size_t block_size_)
    : block_size(block_size_), fns_(std::move(fns)) {
    if (owned.size() != fns_.size()) {
        throw std::runtime_error("AttnResStack: owned/fns size mismatch");
    }
    for (size_t i = 0; i < owned.size(); ++i) {
        adopt("f." + std::to_string(i), owned[i]);
    }
    // One pseudo-query per layer plus the final aggregation, zero-init
    // (uniform depth-attention at step 0; see header). Layer 0's query is
    // NOT allocated: its source list is always the single embedding, and
    // softmax over a singleton is constant, so that query is structurally
    // gradient-dead. (The reference stores all queries in one [L+1, d]
    // tensor, whose per-tensor dead-check cannot see the dead row; found
    // by this port's per-query registration.) w[l] is layer l's query for
    // l >= 1; w.back() is the final aggregation; layer 0 reads h1 direct.
    w.push_back(Var());  // slot 0 intentionally empty
    for (size_t i = 1; i <= fns_.size(); ++i) {
        w.push_back(reg("w." + std::to_string(i), Matrix(1, d)));
    }
    ones_gamma_ = make_var(Matrix(1, d, 1.0f));
    Matrix oc(d, 1, 1.0f);
    ones_col_ = make_var(std::move(oc));
}

Var AttnResStack::depth_attend(const Var& wq, const std::vector<Var>& sources) const {
    // Softmax over one source is the identity mixture regardless of the
    // query — return the source and record nothing.
    if (sources.size() == 1) return sources[0];
    // Eq. 9: logits_i = w^T RMSNorm(k_i) per position; softmax over the
    // SOURCE axis; values mixed raw.
    std::vector<Var> logits;
    logits.reserve(sources.size());
    for (const auto& s : sources) {
        logits.push_back(ops::matmul(ops::mul_row(ops::rmsnorm(s, ones_gamma_), wq), ones_col_));
    }
    Var alpha = ops::softmax_row(ops::concat_cols(logits));  // [T, S]
    Var out;
    for (size_t i = 0; i < sources.size(); ++i) {
        Var term = ops::mul_col(sources[i], ops::slice_cols(alpha, i, i + 1));
        out = out ? ops::add(out, term) : term;
    }
    return out;
}

Var AttnResStack::forward(const Var& h1) const {
    if (block_size == 0) {
        // Full form (Eq. 8-9): attend over embedding + every prior output.
        std::vector<Var> sources{h1};
        for (size_t l = 0; l < fns_.size(); ++l) {
            Var h = depth_attend(w[l], sources);
            sources.push_back(fns_[l](h));
        }
        return depth_attend(w.back(), sources);
    }
    // Block form (Eq. 10): sum within blocks, attend across banked blocks;
    // a partial final block banks too (K3's 9-block layout).
    std::vector<Var> banked{h1};
    Var partial;
    for (size_t l = 0; l < fns_.size(); ++l) {
        std::vector<Var> sources = banked;
        if (partial) sources.push_back(partial);
        Var h = depth_attend(w[l], sources);
        Var out = fns_[l](h);
        partial = partial ? ops::add(partial, out) : out;
        if ((l + 1) % block_size == 0 || l == fns_.size() - 1) {
            banked.push_back(partial);
            partial = Var();
        }
    }
    return depth_attend(w.back(), banked);
}

// ---- Muon ------------------------------------------------------------------

Matrix newton_schulz5(const Matrix& G, int steps) {
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    const bool transposed = G.rows() > G.cols();
    Matrix X = transposed ? G.transpose() : G;
    double fro = 0.0;
    for (size_t i = 0; i < X.rows(); ++i)
        for (size_t j = 0; j < X.cols(); ++j) fro += static_cast<double>(X(i, j)) * X(i, j);
    const float inv = 1.0f / (static_cast<float>(std::sqrt(fro)) + 1e-7f);
    for (size_t i = 0; i < X.rows(); ++i)
        for (size_t j = 0; j < X.cols(); ++j) X(i, j) *= inv;
    for (int s = 0; s < steps; ++s) {
        Matrix A = device::matmul(X, X.transpose());
        Matrix A2 = device::matmul(A, A);
        for (size_t i = 0; i < A.rows(); ++i)
            for (size_t j = 0; j < A.cols(); ++j) A(i, j) = b * A(i, j) + c * A2(i, j);
        Matrix BX = device::matmul(A, X);
        for (size_t i = 0; i < X.rows(); ++i)
            for (size_t j = 0; j < X.cols(); ++j) X(i, j) = a * X(i, j) + BX(i, j);
    }
    return transposed ? X.transpose() : X;
}

Muon::Muon(std::vector<Var> params, float lr_, float momentum, bool nesterov, int ns_steps,
           size_t n_heads)
    : lr(lr_),
      params_(std::move(params)),
      mu_(momentum),
      nesterov_(nesterov),
      ns_(ns_steps),
      H_(n_heads) {
    for (const auto& p : params_) {
        if (p->data.rows() <= 1 || p->data.cols() <= 1) {
            throw std::runtime_error("Muon is for 2-D hidden matrices; route vectors to AdamW/SGD");
        }
        if (p->data.cols() % H_ != 0) {
            throw std::runtime_error("Muon: cols not divisible by n_heads");
        }
        buf_.emplace_back(p->data.rows(), p->data.cols());
    }
}

void Muon::step() {
    for (size_t k = 0; k < params_.size(); ++k) {
        Var& p = params_[k];
        if (p->grad.rows() == 0) continue;
        Matrix& buf = buf_[k];
        const size_t R = p->data.rows(), C = p->data.cols();
        Matrix upd(R, C);
        for (size_t i = 0; i < R; ++i)
            for (size_t j = 0; j < C; ++j) {
                buf(i, j) = mu_ * buf(i, j) + p->grad(i, j);
                upd(i, j) = nesterov_ ? p->grad(i, j) + mu_ * buf(i, j) : buf(i, j);
            }
        // Per-head: partition the COLUMN (output) dimension — see the
        // layout note in nn.hpp — and orthogonalize each block alone.
        const size_t cols = C / H_;
        for (size_t h = 0; h < H_; ++h) {
            Matrix blk(R, cols);
            for (size_t i = 0; i < R; ++i)
                for (size_t j = 0; j < cols; ++j) blk(i, j) = upd(i, h * cols + j);
            Matrix O = newton_schulz5(blk, ns_);
            const float scale =
                std::sqrt(std::max(1.0f, static_cast<float>(cols) / static_cast<float>(R)));
            for (size_t i = 0; i < R; ++i)
                for (size_t j = 0; j < cols; ++j) p->data(i, h * cols + j) -= lr * scale * O(i, j);
        }
    }
}

void Muon::zero_grad() {
    microtorch::zero_grad(params_);
}

Var Dropout::forward(const Var& x) const {
    if (!training() || p_ == 0.0f) return x;  // eval mode: identity
    return ops::dropout(x, p_, next_seed_++);
}

}  // namespace nn
}  // namespace microtorch
