// The fused LM head + cross-entropy (ops::linear_cross_entropy; LOCAL_LLM_STACK.md
// section 11 item 5.1, 26 Oct 2026) against the unfused path it replaces:
// matmul (h W, or h E^T through ops::transpose for a tied embedding) [+ add_bias]
// -> cross_entropy or cross_entropy_masked -> backward.
//
// Host legs (every build):
//   1. Per case (tied and untied heads, a bias, masks: none, an SFT prompt/reply
//      mask, an explicit all-ones mask, scattered ignored rows, one kept row;
//      upstream 1 and 1/2) and per chunk size (1, 7, 32, the default, all rows):
//      the loss, dh and db bit-identical to the unfused path wherever
//      host_gemm_rows_free holds; dW (dE when tied) within the sequential-sum
//      bound (n + 2) u sum_k |h dl| of a float64 sum over the same dl, and the
//      unfused dW within the same bound; masked rows of dh exactly zero.
//   2. Chunk invariance: every chunk size gives the same bits (loss, dh, dW, db).
//   3. A shape where the vendored GEMM's rows are not free (M forces a split):
//      the loss and dh to float rounding.
//   4. Finite differences on h, W and b (central, h = 1e-2), tied and untied.
//   5. OpenMP threads 1..8: the fused results equal one thread's bit for bit (the
//      deep_swa lesson: no reduction whose order follows the thread split).
//   6. NoGrad: the same loss bits and no tape node.
//   7. An upstream gradient other than the stated one: the rescale path, within
//      rounding of the unfused gradients at that upstream.
//   8. nn::Llama: forward_hidden + the unfused head is forward() bit for bit;
//      forward_hidden + the fused op gives forward()'s loss bit for bit and every
//      parameter's gradient bit for bit except the head weight's (within the
//      bound), tied and untied, with and without an SFT mask.
// Device legs (CUDA build, MICROTORCH_DEVICE=cuda, run on a GPU only):
//   9. devops::linear_ce against the unfused device path (device ops on;
//      write-through and under step residency + deferred downloads). Inside a
//      step window with the tiled GEMM (MICROTORCH_CUBLAS unset), where both
//      paths multiply with step_resident_gemm's kernels: every output bit for
//      bit, at chunks of 32, of all rows and the default, and at chunks of 7
//      equal up to the sign of zero. Write-through (ops::matmul then takes the
//      Phase A cuBLAS GEMM) and under MICROTORCH_CUBLAS=1: float rounding.
//      Device vs host fused within float rounding. BF16 (cublasGemmEx) fused vs
//      unfused within BF16 rounding. 60 fused steps in windows: pool memory flat.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/llama.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;
using ops::HeadLayout;

namespace {

Matrix filled(size_t r, size_t c, unsigned seed, float a = 1.0f) {
    Matrix m(r, c);
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-a, a);
    for (size_t i = 0; i < r * c; ++i) m.get_data()[i] = u(g);
    return m;
}

bool bits(const Matrix& a, const Matrix& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() &&
           std::memcmp(a.get_data(), b.get_data(), a.rows() * a.cols() * sizeof(float)) == 0;
}
bool bits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }
// Equal as floats (+0 == -0): bit for bit up to the sign of zero.
bool same_values(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (size_t i = 0; i < a.rows() * a.cols(); ++i)
        if (!(a.get_data()[i] == b.get_data()[i])) return false;
    return true;
}
double max_abs(const Matrix& a) {
    double w = 0.0;
    for (size_t i = 0; i < a.rows() * a.cols(); ++i) w = std::max(w, std::fabs(static_cast<double>(a.get_data()[i])));
    return w;
}
double max_diff(const Matrix& a, const Matrix& b) {
    double w = 0.0;
    for (size_t i = 0; i < a.rows() * a.cols(); ++i)
        w = std::max(w, std::fabs(static_cast<double>(a.get_data()[i]) - b.get_data()[i]));
    return w;
}
// max |a - b| / max |b|: the normwise relative difference.
double rel_diff(const Matrix& a, const Matrix& b) { return max_diff(a, b) / std::max(max_abs(b), 1e-30); }
double rel(double a, double b) { return std::fabs(a - b) / std::max(std::fabs(b), 1e-30); }

enum class MaskKind { None, Sft, Ones, Scattered, Single };

std::vector<uint8_t> make_mask(MaskKind k, size_t R) {
    std::vector<uint8_t> m;
    switch (k) {
        case MaskKind::None:
            break;
        case MaskKind::Ones:
            m.assign(R, 1);
            break;
        case MaskKind::Sft:  // prompt, reply, prompt, reply, padding (loss policy "all")
            m.assign(R, 0);
            for (size_t i = R / 8; i < R / 3; ++i) m[i] = 1;
            for (size_t i = R / 2; i < (3 * R) / 4; ++i) m[i] = 1;
            break;
        case MaskKind::Scattered:  // state positions every fifth row, a padded tail
            m.assign(R, 1);
            for (size_t i = 0; i < R; i += 5) m[i] = 0;
            for (size_t i = R - R / 10; i < R; ++i) m[i] = 0;
            break;
        case MaskKind::Single:
            m.assign(R, 0);
            m[R / 2] = 1;
            break;
    }
    return m;
}

// Masked rows carry targets the op must never read (-1 and out of range).
std::vector<int> make_targets(const std::vector<uint8_t>& m, size_t R, size_t V, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<int> t(R);
    for (size_t i = 0; i < R; ++i) {
        const bool kept = m.empty() || m[i];
        t[i] = kept ? static_cast<int>(g() % V) : (i % 2 ? -1 : static_cast<int>(V + 3));
    }
    return t;
}

struct Out {
    float loss = 0.0f;
    Matrix dh, dw, db, dl;  // dl: the unfused path's logits gradient
};

struct Problem {
    Matrix H, W, B;  // W: [D, V] (InOut) or [V, D] (VocabMajor); B empty = no bias
    HeadLayout layout;
    std::vector<int> t;
    std::vector<uint8_t> m;
    size_t V() const { return layout == HeadLayout::VocabMajor ? W.rows() : W.cols(); }
};

Problem make_problem(size_t R, size_t D, size_t V, HeadLayout layout, bool bias, MaskKind mk, unsigned seed) {
    Problem p;
    p.H = filled(R, D, seed, 1.5f);
    p.W = layout == HeadLayout::VocabMajor ? filled(V, D, seed + 1, 0.5f) : filled(D, V, seed + 1, 0.5f);
    if (bias) p.B = filled(1, V, seed + 2, 0.5f);
    p.layout = layout;
    p.m = make_mask(mk, R);
    p.t = make_targets(p.m, R, V, seed + 3);
    return p;
}

// window: inside a step window (step residency + deferred downloads), closed
// before the gradients are read.
Out run_unfused(const Problem& p, float upstream, bool window = false) {
    Var h = make_var(p.H, true), w = make_var(p.W, true);
    Var b = p.B.rows() ? make_var(p.B, true) : nullptr;
    if (window) device::step_begin();
    Var logits = p.layout == HeadLayout::InOut ? ops::matmul(h, w) : ops::matmul(h, ops::transpose(w));
    if (b) logits = ops::add_bias(logits, b);
    Var loss = p.m.empty() ? ops::cross_entropy(logits, p.t) : ops::cross_entropy_masked(logits, p.t, p.m);
    backward(ops::scale(loss, upstream));
    if (window) device::step_end();
    Out o;
    o.loss = loss->data(0, 0);
    o.dh = h->grad;
    o.dw = w->grad;
    if (b) o.db = b->grad;
    o.dl = logits->grad;
    return o;
}

Out run_fused(const Problem& p, float stated, float actual, size_t chunk, bool window = false) {
    Var h = make_var(p.H, true), w = make_var(p.W, true);
    Var b = p.B.rows() ? make_var(p.B, true) : nullptr;
    if (window) device::step_begin();
    Var loss = ops::linear_cross_entropy(h, w, p.layout, b, p.t, p.m, stated, chunk);
    backward(ops::scale(loss, actual));
    if (window) device::step_end();
    Out o;
    o.loss = loss->data(0, 0);
    o.dh = h->grad;
    o.dw = w->grad;
    if (b) o.db = b->grad;
    return o;
}

size_t kept_rows(const Problem& p) {
    if (p.m.empty()) return p.H.rows();
    size_t k = 0;
    for (uint8_t v : p.m) k += v;
    return k;
}

// Whether the host fused op's loss, dh and db must equal the unfused path's
// bits: the vendored GEMM's rows free at the full row count and at every
// chunk height the fused op multiplies (full chunks and the remainder).
bool host_exact(const Problem& p, size_t chunk) {
    const size_t R = p.H.rows(), D = p.H.cols(), V = p.V(), kept = kept_rows(p);
    const size_t c = std::min(chunk ? chunk : ops::linear_ce_chunk_rows(R, D, V), R);
    std::vector<size_t> heights{std::min(c, kept)};
    if (kept % c) heights.push_back(kept % c);
    bool ok = ops::host_gemm_rows_free(R, V, D) && ops::host_gemm_rows_free(R, D, V);
    for (size_t n : heights) ok = ok && ops::host_gemm_rows_free(n, V, D) && ops::host_gemm_rows_free(n, D, V);
    return ok;
}

// dW against a float64 sum over the same dl (the unfused logits gradient):
// the worst |dW - ref| / ((n + 2) u sum_k |h(k, i) dl(k, j)|), n = rows. <= 1
// is the sequential-summation bound gamma_{n+2}.
double dw_bound_ratio(const Problem& p, const Matrix& dl, const Matrix& dw) {
    const size_t R = p.H.rows(), D = p.H.cols(), V = p.V();
    const bool vm = p.layout == HeadLayout::VocabMajor;
    const double u = std::ldexp(1.0, -24);
    double worst = 0.0;
    for (size_t i = 0; i < D; ++i)
        for (size_t j = 0; j < V; ++j) {
            double ref = 0.0, mag = 0.0;
            for (size_t k = 0; k < R; ++k) {
                const double x = static_cast<double>(p.H(k, i)) * dl(k, j);
                ref += x;
                mag += std::fabs(x);
            }
            const double got = vm ? dw(j, i) : dw(i, j);
            const double tol = (R + 2) * u * mag + 1e-30;
            worst = std::max(worst, std::fabs(got - ref) / tol);
        }
    return worst;
}

struct Case {
    const char* name;
    size_t R, D, V;
    HeadLayout layout;
    bool bias;
    MaskKind mask;
    float upstream;
};

void leg1_2_cases() {
    std::printf("-- 1-2. fused vs unfused, per case and chunk size --\n");
    const Case cases[] = {
        {"lm, tied", 96, 24, 1003, HeadLayout::VocabMajor, false, MaskKind::None, 1.0f},
        {"lm, untied + bias, upstream 1/2", 96, 24, 1003, HeadLayout::InOut, true, MaskKind::None, 0.5f},
        {"sft mask, tied, 130 rows", 130, 20, 517, HeadLayout::VocabMajor, false, MaskKind::Sft, 1.0f},
        {"explicit all-ones mask, untied", 64, 16, 256, HeadLayout::InOut, false, MaskKind::Ones, 1.0f},
        {"scattered ignored rows, tied + bias", 150, 32, 2000, HeadLayout::VocabMajor, true, MaskKind::Scattered, 0.5f},
        {"one kept row, untied", 40, 8, 300, HeadLayout::InOut, false, MaskKind::Single, 1.0f},
    };
    unsigned seed = 11;
    for (const Case& c : cases) {
        const Problem p = make_problem(c.R, c.D, c.V, c.layout, c.bias, c.mask, seed += 10);
        const Out u = run_unfused(p, c.upstream);
        const double ub = dw_bound_ratio(p, u.dl, u.dw);
        const size_t dflt = ops::linear_ce_chunk_rows(c.R, c.D, c.V);
        const size_t chunks[] = {1, 7, 32, 0, c.R};
        Out first;
        bool have_first = false;
        for (size_t ch : chunks) {
            const Out f = run_fused(p, c.upstream, c.upstream, ch);
            const bool exact = host_exact(p, ch);
            CHECK(exact);  // every case here is in the bit-identical regime
            CHECK(bits(f.loss, u.loss));
            CHECK(bits(f.dh, u.dh));
            if (c.bias) CHECK(bits(f.db, u.db));
            const double fb = dw_bound_ratio(p, u.dl, f.dw);
            CHECK(fb <= 1.0);
            CHECK(ub <= 1.0);
            for (size_t i = 0; i < c.R; ++i)
                if (!p.m.empty() && !p.m[i])
                    for (size_t j = 0; j < c.D; ++j) CHECK(f.dh(i, j) == 0.0f);
            if (!have_first) {
                first = f;
                have_first = true;
            } else {
                CHECK(bits(f.loss, first.loss));
                CHECK(bits(f.dh, first.dh));
                CHECK(bits(f.dw, first.dw));
                if (c.bias) CHECK(bits(f.db, first.db));
            }
            if (ch == 0)
                std::printf("  %-38s chunk %4zu: loss, dh%s bitwise; dW/bound %.3f (unfused %.3f), |dW - unfused| / max %.2e\n",
                            c.name, dflt, c.bias ? ", db" : "", fb, ub, rel_diff(f.dw, u.dw));
        }
    }
    std::printf("  every chunk size (1, 7, 32, default, all rows) gives the same bits\n");
}

void leg3_not_free() {
    std::printf("-- 3. vendored GEMM rows not free (M forces a split) --\n");
    // logits GEMM (600, 100, 8): N and K <= 128, M = 600 forces the halving.
    const Problem p = make_problem(600, 8, 100, HeadLayout::InOut, false, MaskKind::None, 301);
    CHECK(!ops::host_gemm_rows_free(600, 100, 8));
    const Out u = run_unfused(p, 1.0f);
    const Out f = run_fused(p, 1.0f, 1.0f, 0);
    const double dl = rel(f.loss, u.loss), dh = rel_diff(f.dh, u.dh), dw = rel_diff(f.dw, u.dw);
    CHECK(dl <= 1e-6);
    CHECK(dh <= 1e-5);
    CHECK(dw <= 1e-5);
    std::printf("  loss rel %.2e, dh %.2e, dW %.2e (normwise; float rounding)\n", dl, dh, dw);
}

double fd_worst(const std::function<float()>& f, Matrix& x, const Matrix& analytic, float step = 1e-2f) {
    double worst = 0.0;
    for (size_t i = 0; i < x.rows(); ++i)
        for (size_t j = 0; j < x.cols(); ++j) {
            const float keep = x(i, j);
            x(i, j) = keep + step;
            const float up = f();
            x(i, j) = keep - step;
            const float dn = f();
            x(i, j) = keep;
            const double fd = (static_cast<double>(up) - dn) / (2.0 * step);
            const double a = analytic(i, j);
            worst = std::max(worst, std::fabs(a - fd) / (1.0 + std::max(std::fabs(a), std::fabs(fd))));
        }
    return worst;
}

void leg4_fd() {
    std::printf("-- 4. finite differences --\n");
    for (HeadLayout lay : {HeadLayout::InOut, HeadLayout::VocabMajor}) {
        Problem p = make_problem(12, 6, 37, lay, true, MaskKind::Sft, lay == HeadLayout::InOut ? 401 : 402);
        const Out f = run_fused(p, 1.0f, 1.0f, 5);
        auto loss = [&]() {
            NoGrad ng;
            Var h = make_var(p.H), w = make_var(p.W), b = make_var(p.B);
            return ops::linear_cross_entropy(h, w, p.layout, b, p.t, p.m, 1.0f, 5)->data(0, 0);
        };
        const double eh = fd_worst(loss, p.H, f.dh), ew = fd_worst(loss, p.W, f.dw), eb = fd_worst(loss, p.B, f.db);
        CHECK(eh <= 2e-3);
        CHECK(ew <= 2e-3);
        CHECK(eb <= 2e-3);
        std::printf("  %s: worst |a - fd| / (1 + max) h %.2e, W %.2e, b %.2e\n",
                    lay == HeadLayout::InOut ? "untied" : "tied  ", eh, ew, eb);
    }
}

void leg5_threads() {
#ifdef _OPENMP
    std::printf("-- 5. OpenMP thread counts --\n");
    const Problem p = make_problem(150, 32, 2000, HeadLayout::VocabMajor, true, MaskKind::Scattered, 501);
    const Problem q = make_problem(96, 24, 1003, HeadLayout::InOut, true, MaskKind::None, 502);
    const int n0 = omp_get_max_threads();
    omp_set_num_threads(1);
    const Out a1 = run_fused(p, 0.5f, 0.5f, 0), b1 = run_fused(q, 1.0f, 1.0f, 7);
    const Out ua = run_unfused(p, 0.5f);
    for (int t = 1; t <= 8; ++t) {
        omp_set_num_threads(t);
        const Out a = run_fused(p, 0.5f, 0.5f, 0), b = run_fused(q, 1.0f, 1.0f, 7);
        CHECK(bits(a.loss, a1.loss) && bits(a.dh, a1.dh) && bits(a.dw, a1.dw) && bits(a.db, a1.db));
        CHECK(bits(b.loss, b1.loss) && bits(b.dh, b1.dh) && bits(b.dw, b1.dw) && bits(b.db, b1.db));
        const Out u = run_unfused(p, 0.5f);
        CHECK(bits(u.loss, ua.loss) && bits(a.loss, u.loss) && bits(a.dh, u.dh));
    }
    omp_set_num_threads(n0);
    std::printf("  1..8 threads: loss, dh, dW, db equal to one thread's bit for bit (tied and untied)\n");
#else
    std::printf("-- 5. OpenMP thread counts: SKIP (no OpenMP)\n");
#endif
}

void leg6_nograd() {
    std::printf("-- 6. NoGrad --\n");
    const Problem p = make_problem(64, 16, 300, HeadLayout::VocabMajor, false, MaskKind::Sft, 601);
    const Out f = run_fused(p, 1.0f, 1.0f, 0);
    NoGrad ng;
    Var h = make_var(p.H, true), w = make_var(p.W, true);
    Var l = ops::linear_cross_entropy(h, w, p.layout, nullptr, p.t, p.m);
    CHECK(bits(l->data(0, 0), f.loss));
    CHECK(!l->requires_grad && !l->backward_fn && l->parents.empty());
    std::printf("  loss bit for bit, no tape node\n");
}

void leg7_rescale() {
    std::printf("-- 7. an upstream gradient other than the stated one --\n");
    const Problem p = make_problem(96, 24, 1003, HeadLayout::InOut, true, MaskKind::Sft, 701);
    const Out u = run_unfused(p, 0.3f);
    const Out f = run_fused(p, 1.0f, 0.3f, 0);  // stated 1, actual 0.3: rescaled by 0.3
    const double dh = rel_diff(f.dh, u.dh), dw = rel_diff(f.dw, u.dw), db = rel_diff(f.db, u.db);
    CHECK(bits(f.loss, u.loss));
    CHECK(dh <= 1e-6 && dw <= 1e-5 && db <= 1e-6);
    std::printf("  loss bitwise; dh %.2e, dW %.2e, db %.2e (normwise)\n", dh, dw, db);
}

nn::LlamaConfig small_llama(bool tied) {
    nn::LlamaConfig c;
    c.vocab = 300;
    c.d = 32;
    c.n_layers = 2;
    c.n_heads = 4;
    c.d_ff = 64;
    c.n_ctx = 16;
    c.tie_embeddings = tied;
    return c;
}

void leg8_llama() {
    std::printf("-- 8. nn::Llama: forward_hidden + the fused op --\n");
    for (bool tied : {true, false})
        for (bool masked : {false, true}) {
            const nn::LlamaConfig cfg = small_llama(tied);
            nn::Llama a(cfg, 5), b(cfg, 5);
            std::mt19937 g(tied ? 801 : 802);
            const size_t T = 16, B = 3;
            std::vector<int> ids(T * B), y(T * B);
            for (auto& v : ids) v = static_cast<int>(g() % cfg.vocab);
            for (auto& v : y) v = static_cast<int>(g() % cfg.vocab);
            std::vector<uint8_t> m;
            if (masked) m = make_mask(MaskKind::Sft, T * B);
            // forward() is forward_hidden + the head.
            {
                NoGrad ng;
                Var full = a.forward(ids, T);
                Var hid = a.forward_hidden(ids, T);
                Var lg = a.lm_head ? a.lm_head->forward(hid) : ops::matmul(hid, ops::transpose(a.embed_tokens->weight));
                CHECK(bits(full->data, lg->data));
            }
            Var la = masked ? ops::cross_entropy_masked(a.forward(ids, T), y, m) : ops::cross_entropy(a.forward(ids, T), y);
            backward(ops::scale(la, 0.5f));
            Var lb = ops::linear_cross_entropy(b.forward_hidden(ids, T), b.head_weight(), b.head_layout(), nullptr, y,
                                               m, 0.5f);
            backward(ops::scale(lb, 0.5f));
            CHECK(bits(la->data(0, 0), lb->data(0, 0)));
            const auto pa = a.named_parameters(), pb = b.named_parameters();
            CHECK(pa.size() == pb.size());
            size_t same = 0;
            double head = 0.0;
            for (size_t i = 0; i < pa.size(); ++i) {
                CHECK(pa[i].first == pb[i].first);
                const bool is_head = pa[i].second.get() == a.head_weight().get();
                if (is_head) {
                    head = rel_diff(pb[i].second->grad, pa[i].second->grad);
                    CHECK(head <= 1e-5);
                } else {
                    CHECK(bits(pa[i].second->grad, pb[i].second->grad));
                    ++same;
                }
            }
            std::printf("  %s%s: loss bitwise, %zu of %zu parameter grads bitwise, head weight %.2e (normwise)\n",
                        tied ? "tied  " : "untied", masked ? ", sft mask" : "          ", same, pa.size(), head);
        }
}

// ---------------------------------------------------------------- device legs
bool cublas_env() {
    const char* v = std::getenv("MICROTORCH_CUBLAS");
    return v && std::strcmp(v, "1") == 0;
}

// mode 1: device ops, write-through; 2: + step residency and deferred downloads.
void set_mode(int mode) {
    device::set_device_ops(mode != 0);
    device::set_step_residency(mode == 2);
    device::set_defer_downloads(mode == 2);
}

Out dev_unfused(const Problem& p, float upstream, int mode) {
    set_mode(mode);
    Out o = run_unfused(p, upstream, mode == 2);
    set_mode(1);
    return o;
}

Out dev_fused(const Problem& p, float upstream, size_t chunk, int mode) {
    set_mode(mode);
    Out o = run_fused(p, upstream, upstream, chunk, mode == 2);
    set_mode(1);
    return o;
}

void leg9_device() {
    const bool cublas = cublas_env();
    std::printf("-- 9. device: devops::linear_ce vs the unfused device path (%s) --\n",
                cublas ? "cuBLAS: float rounding" : "tiled GEMM: bit for bit");
    const Case cases[] = {
        {"lm, tied", 96, 24, 1003, HeadLayout::VocabMajor, false, MaskKind::None, 1.0f},
        {"untied + bias, upstream 1/2", 96, 24, 1003, HeadLayout::InOut, true, MaskKind::None, 0.5f},
        {"sft mask, tied, 300 rows", 300, 40, 2049, HeadLayout::VocabMajor, false, MaskKind::Sft, 1.0f},
        {"scattered ignored rows, untied", 200, 64, 4096, HeadLayout::InOut, false, MaskKind::Scattered, 0.5f},
    };
    unsigned seed = 900;
    for (const Case& c : cases) {
        const Problem p = make_problem(c.R, c.D, c.V, c.layout, c.bias, c.mask, seed += 10);
        set_mode(0);
        device::set(device::Device::CPU);  // the host reference: host GEMMs too
        const Out host = run_fused(p, c.upstream, c.upstream, 0);
        device::set(device::Device::CUDA);
        for (int mode = 1; mode <= 2; ++mode) {
            const Out u = dev_unfused(p, c.upstream, mode);
            const Out f32 = dev_fused(p, c.upstream, 32, mode);
            const Out fall = dev_fused(p, c.upstream, c.R, mode);
            const Out f7 = dev_fused(p, c.upstream, 7, mode);
            const Out fd = dev_fused(p, c.upstream, 0, mode);
            // Bit for bit: inside a step window with the tiled GEMM, where both
            // paths multiply with step_resident_gemm's kernels. Write-through
            // (no window), ops::matmul takes the Phase A cuBLAS GEMM.
            if (mode == 2 && !cublas) {
                CHECK(bits(f32.loss, u.loss) && bits(f32.dh, u.dh) && bits(f32.dw, u.dw));
                if (c.bias) CHECK(bits(f32.db, u.db));
                CHECK(bits(fall.loss, u.loss) && bits(fall.dh, u.dh) && bits(fall.dw, u.dw));
                CHECK(bits(fd.loss, u.loss) && bits(fd.dh, u.dh) && bits(fd.dw, u.dw));
                CHECK(bits(f7.loss, u.loss) && bits(f7.dh, u.dh) && same_values(f7.dw, u.dw));
            } else {
                CHECK(rel(fd.loss, u.loss) <= 1e-6);
                CHECK(rel_diff(fd.dh, u.dh) <= 1e-5 && rel_diff(fd.dw, u.dw) <= 1e-5);
                CHECK(rel_diff(f7.dw, u.dw) <= 1e-5 && rel_diff(fall.dw, u.dw) <= 1e-5);
                if (c.bias) CHECK(rel_diff(fd.db, u.db) <= 1e-5);
            }
            const double hl = rel(fd.loss, host.loss), hh = rel_diff(fd.dh, host.dh), hw = rel_diff(fd.dw, host.dw);
            CHECK(hl <= 1e-5 && hh <= 1e-4 && hw <= 1e-4);
            for (size_t i = 0; i < c.R; ++i)
                if (!p.m.empty() && !p.m[i])
                    for (size_t j = 0; j < c.D; ++j) CHECK(fd.dh(i, j) == 0.0f);
            std::printf("  %-32s %s: vs unfused loss %.1e dh %.1e dW %.1e; vs host loss %.1e dh %.1e dW %.1e\n",
                        c.name, mode == 1 ? "write-through" : "deferred     ", rel(fd.loss, u.loss),
                        rel_diff(fd.dh, u.dh), rel_diff(fd.dw, u.dw), hl, hh, hw);
        }
    }
    // BF16 (cublasGemmEx in both paths): to bfloat16 input rounding.
    {
        const Problem p = make_problem(256, 64, 4096, HeadLayout::VocabMajor, false, MaskKind::Sft, 990);
        device::set_matmul_precision(device::Precision::BF16);
        const Out u = dev_unfused(p, 1.0f, 2), f = dev_fused(p, 1.0f, 0, 2);
        device::set_matmul_precision(device::Precision::FP32);
        const double l = rel(f.loss, u.loss), dh = rel_diff(f.dh, u.dh), dw = rel_diff(f.dw, u.dw);
        CHECK(l <= 1e-5 && dh <= 1e-4 && dw <= 1e-4);
        std::printf("  bf16, deferred: vs unfused loss %.1e dh %.1e dW %.1e\n", l, dh, dw);
    }
    // Endurance: 60 fused steps, each in its own window; pool memory flat.
    {
        const Problem p = make_problem(512, 64, 4096, HeadLayout::VocabMajor, false, MaskKind::None, 991);
        for (int i = 0; i < 5; ++i) dev_fused(p, 1.0f, 0, 2);
        const size_t base = device::pool_live_bytes();
        for (int i = 0; i < 60; ++i) dev_fused(p, 1.0f, 0, 2);
        const size_t after = device::pool_live_bytes();
        const double grew = (after > base ? static_cast<double>(after - base) : 0.0) / (1 << 20);
        CHECK(grew <= 8.0);
        std::printf("  60 fused steps under deferral: pool live grew %.2f MiB\n", grew);
    }
    set_mode(0);
}

}  // namespace

int main() {
    device::set_from_env();
    const bool on_device = device::cuda_compiled() && device::get() == device::Device::CUDA;
    // The host legs run on the host: the device legs set their own switches.
    if (on_device) device::set(device::Device::CPU);
    leg1_2_cases();
    leg3_not_free();
    leg4_fd();
    leg5_threads();
    leg6_nograd();
    leg7_rescale();
    leg8_llama();
    if (on_device) {
        device::set(device::Device::CUDA);
        leg9_device();
    } else {
        std::printf("-- 9. device legs: SKIP (CUDA build + MICROTORCH_DEVICE=cuda; run on a GPU only)\n");
    }
    std::printf("test_fused_head PASSED\n");
    return 0;
}
