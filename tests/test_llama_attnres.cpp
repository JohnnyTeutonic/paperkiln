// Attention Residuals on the llama block (nn::LlamaConfig::AttnRes;
// microtorch/llama_attnres.hpp) and the fused depth-attention op
// (ops::depth_attend, microtorch/depth_attend.hpp).
//   1. The fused op against the same function composed from gradchecked
//      tape ops (nn::AttnResStack's composition): outputs and every gradient,
//      unscaled and scaled.
//   2. Finite-difference gradcheck of the fused op, and of tiny llama_attnres
//      models (scale sources and none; fixed recursion with per-application
//      queries) on every query and a sample of the other tensors.
//   3. At initialisation (zero queries), scale "sources" is the plain llama of
//      the same seed: identical non-query parameters bit for bit, logits and
//      the plain parameters' gradients equal to rounding; scale "none" is not
//      (reported: rmsnorm(x / S) = x / sqrt(mean x^2 + S^2 eps)).
//   4. forward_cached (prefill then decode) against forward.
//   5. Activation checkpointing per sublayer: the same gradients.
//   6. Training steps as mtstudio runs them: on a CUDA build with
//      MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1, the device op set
//      (the depth_attend kernels included) with step residency and deferred
//      downloads against the host path; on a CPU build host vs host (bit for
//      bit). CUDA correctness is validated on Colab (tools/colab_cuda_validate.sh).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "microtorch/depth_attend.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/llama.hpp"
#include "microtorch/llama_attnres.hpp"

using namespace microtorch;

namespace {

nn::LlamaConfig tiny(size_t layers, bool attnres, const std::string& scale = "sources") {
    nn::LlamaConfig c;
    c.vocab = 37;
    c.d = 16;
    c.n_heads = 2;
    c.d_ff = 24;
    c.n_ctx = 12;
    c.n_layers = layers;
    c.attn_res.on = attnres;
    c.attn_res.scale = scale;
    return c;
}

nn::LlamaConfig tiny_rec(bool attnres) {
    nn::LlamaConfig c = tiny(4, attnres);
    c.recursion.prelude = 1;
    c.recursion.block = 2;
    c.recursion.repeats = 3;
    c.recursion.coda = 1;
    return c;
}

const std::vector<int> kIds = {5, 17, 3, 22, 22, 9, 31, 2, 14, 6};
const std::vector<int> kTargets = {17, 3, 22, 22, 9, 31, 2, 14, 6, 1};

size_t bits_differ(const Matrix& a, const Matrix& b) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols());
    size_t n = 0;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j) {
            const float x = a(i, j), y = b(i, j);
            n += std::memcmp(&x, &y, sizeof x) != 0;
        }
    return n;
}

double rel(const Matrix& a, const Matrix& b) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols());
    double s = 0, w = 0;
    for (size_t i = 0; i < b.rows(); ++i)
        for (size_t j = 0; j < b.cols(); ++j) s = std::max(s, (double)std::fabs(b(i, j)));
    for (size_t i = 0; i < b.rows(); ++i)
        for (size_t j = 0; j < b.cols(); ++j) w = std::max(w, std::fabs((double)a(i, j) - b(i, j)));
    return w / std::max(s, 1e-30);
}

Matrix rnd(size_t r, size_t c, std::mt19937& g, float scale) {
    std::normal_distribution<float> nd(0.0f, scale);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = nd(g);
    return m;
}

void randomise(nn::Module& m, unsigned seed, float scale = 0.3f, bool queries = true) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto& [name, p] : m.named_parameters()) {
        const bool norm = name.find("norm") != std::string::npos;
        const bool query = name.rfind("attn_res.", 0) == 0;
        for (size_t i = 0; i < p->data.rows(); ++i)
            for (size_t j = 0; j < p->data.cols(); ++j) {
                const float v = nd(g);
                if (query && !queries) continue;
                p->data(i, j) = norm ? 1.0f + 0.2f * v : v;
            }
    }
}

void clear_grads(nn::Module& m) {
    for (auto& p : m.parameters()) p->grad = Matrix();
}

std::map<std::string, Matrix> grads(const nn::Module& m) {
    std::map<std::string, Matrix> g;
    for (const auto& [n, p] : m.named_parameters()) g.emplace(n, p->grad);
    return g;
}

// nn::AttnResStack::depth_attend, composed from tape ops (the reference).
Var composed(const std::vector<Var>& src, const Var& w, float scale) {
    const size_t d = src[0]->data.cols();
    Var ones = make_var(Matrix(1, d, 1.0f)), col = make_var(Matrix(d, 1, 1.0f));
    std::vector<Var> logits;
    for (const auto& s : src) logits.push_back(ops::matmul(ops::mul_row(ops::rmsnorm(s, ones), w), col));
    Var a = ops::softmax_row(ops::concat_cols(logits));
    Var out;
    for (size_t i = 0; i < src.size(); ++i) {
        Var t = ops::mul_col(src[i], ops::slice_cols(a, i, i + 1));
        out = out ? ops::add(out, t) : t;
    }
    return scale == 1.0f ? out : ops::scale(out, scale);
}

void test_fused_vs_composed(float scale) {
    std::mt19937 g(3);
    const size_t S = 5, T = 7, d = 16;
    std::vector<Var> src;
    for (size_t i = 0; i < S; ++i) src.push_back(make_var(rnd(T, d, g, 0.5f + 0.3f * i), true));
    Var w = make_var(rnd(1, d, g, 0.7f), true);
    const Matrix wt = rnd(T, d, g, 1.0f);  // a random cotangent through sum(out .* wt)
    auto run = [&](bool fused) {
        for (auto& s : src) s->grad = Matrix();
        w->grad = Matrix();
        Var out = fused ? ops::depth_attend(src, w, ops::kRmsNormEps, scale) : composed(src, w, scale);
        backward(ops::mean(ops::mul(out, make_var(wt))));
        std::vector<Matrix> gs;
        for (auto& s : src) gs.push_back(s->grad);
        gs.push_back(w->grad);
        return std::make_pair(out->data, gs);
    };
    const auto [of, gf] = run(true);
    const auto [oc, gc] = run(false);
    double gw = 0;
    for (size_t i = 0; i < gf.size(); ++i) gw = std::max(gw, rel(gf[i], gc[i]));
    const double ow = rel(of, oc);
    std::printf("  fused depth_attend vs composed (S %zu, T %zu, d %zu, scale %.0f): output worst rel %.2e, "
                "gradients (sources and query) worst rel %.2e\n",
                S, T, d, scale, ow, gw);
    CHECK(ow < 1e-6 && gw < 1e-5);
}

void test_fused_fd() {
    std::mt19937 g(5);
    const size_t S = 4, T = 5, d = 8;
    std::vector<Var> src;
    for (size_t i = 0; i < S; ++i) src.push_back(make_var(rnd(T, d, g, 0.8f), true));
    Var w = make_var(rnd(1, d, g, 1.0f), true);
    const Matrix wt = rnd(T, d, g, 1.0f);
    auto loss = [&] { return ops::mean(ops::mul(ops::depth_attend(src, w, ops::kRmsNormEps, 3.0f), make_var(wt))); };
    backward(loss());
    const float eps = 1e-2f;
    double worst = 0;
    size_t probes = 0;
    std::vector<Var> all = src;
    all.push_back(w);
    for (auto& v : all)
        for (size_t i = 0; i < v->data.rows(); ++i)
            for (size_t j = 0; j < v->data.cols(); ++j) {
                double fp, fm;
                {
                    NoGrad ng;
                    const float keep = v->data(i, j);
                    v->data(i, j) = keep + eps;
                    fp = loss()->data(0, 0);
                    v->data(i, j) = keep - eps;
                    fm = loss()->data(0, 0);
                    v->data(i, j) = keep;
                }
                const double fd = (fp - fm) / (2.0 * eps), an = v->grad(i, j);
                worst = std::max(worst, std::fabs(fd - an) / std::max({std::fabs(fd), std::fabs(an), 1e-3}));
                ++probes;
            }
    std::printf("  FD gradcheck of the fused op (S %zu, scale 3): %zu entries, worst rel %.2e\n", S, probes, worst);
    CHECK(worst < 2e-2);
}

void fd_model(const nn::LlamaConfig& cfg, const char* label) {
    nn::Llama m(cfg, 5);
    randomise(m, 9);
    clear_grads(m);
    backward(ops::cross_entropy(m.forward(kIds), kTargets));
    std::mt19937 rng(17);
    const float eps = 1e-2f;
    double worst = 0;
    size_t probes = 0, tensors = 0;
    for (const auto& [n, p] : m.named_parameters()) {
        ++tensors;
        const bool query = n.rfind("attn_res.", 0) == 0;
        const int k = query ? 3 : 1;  // every query, a sample of the rest
        for (int t = 0; t < k; ++t) {
            const size_t i = rng() % p->data.rows(), j = rng() % p->data.cols();
            double fp, fm;
            {
                NoGrad ng;
                const float keep = p->data(i, j);
                p->data(i, j) = keep + eps;
                fp = ops::cross_entropy(m.forward(kIds), kTargets)->data(0, 0);
                p->data(i, j) = keep - eps;
                fm = ops::cross_entropy(m.forward(kIds), kTargets)->data(0, 0);
                p->data(i, j) = keep;
            }
            const double fd = (fp - fm) / (2.0 * eps), an = p->grad(i, j);
            worst = std::max(worst, std::fabs(fd - an) / std::max({std::fabs(fd), std::fabs(an), 1e-2}));
            ++probes;
        }
    }
    std::printf("  FD gradcheck %s: %zu entries over %zu tensors, worst rel %.2e\n", label, probes, tensors, worst);
    CHECK(worst < 2e-2);
}

void test_parameters(const nn::LlamaConfig& cfg, size_t queries) {
    nn::LlamaConfig pc = cfg;
    pc.attn_res.on = false;
    nn::Llama plain(pc, 42), ar(cfg, 42);
    const auto pp = plain.named_parameters(), ap = ar.named_parameters();
    size_t diff = 0, q = 0;
    std::map<std::string, Var> pm(pp.begin(), pp.end());
    for (const auto& [n, p] : ap) {
        if (n.rfind("attn_res.", 0) == 0) {
            ++q;
            for (size_t i = 0; i < p->data.cols(); ++i) CHECK(p->data(0, i) == 0.0f);
            continue;
        }
        CHECK(pm.count(n));
        diff += bits_differ(p->data, pm.at(n)->data);
    }
    std::printf("  parameters: the plain model's %zu tensors (%zu bits differ) plus %zu zero queries\n", pp.size(),
                diff, q);
    CHECK(diff == 0 && ap.size() == pp.size() + queries && q == queries);
}

// Zero queries against the plain llama of the same seed and weights.
void test_init_equivalence(const nn::LlamaConfig& cfg, bool randomised, const char* label, double* out_rel) {
    nn::LlamaConfig pc = cfg;
    pc.attn_res.on = false;
    nn::Llama plain(pc, 7), ar(cfg, 7);
    if (randomised) {
        randomise(plain, 3, 0.3f);
        randomise(ar, 3, 0.3f, /*queries=*/false);
    }
    clear_grads(plain);
    clear_grads(ar);
    Var lp = plain.forward(kIds), la = ar.forward(kIds);
    backward(ops::cross_entropy(lp, kTargets));
    backward(ops::cross_entropy(la, kTargets));
    const double r = rel(la->data, lp->data);
    const size_t bits = bits_differ(la->data, lp->data);
    const auto gp = grads(plain);
    double gw = 0;
    size_t qnz = 0;
    for (const auto& [n, g] : grads(ar)) {
        if (n.rfind("attn_res.", 0) == 0) {
            for (size_t j = 0; j < g.cols(); ++j) qnz += g(0, j) != 0.0f;
            continue;
        }
        gw = std::max(gw, rel(g, gp.at(n)));
    }
    std::printf("  init (%s, %s weights): logits worst rel %.2e (%zu/%zu bits differ); plain parameters' "
                "gradients worst rel %.2e; nonzero query-gradient entries %zu\n",
                label, randomised ? "randomised non-query" : "default", r, bits, la->data.rows() * la->data.cols(), gw,
                qnz);
    if (out_rel) *out_rel = r;
    if (cfg.attn_res.scale == "sources") CHECK(r < 1e-5 && gw < 1e-4 && qnz > 0);
}

void test_kv_cache(const nn::LlamaConfig& cfg, const char* label) {
    nn::Llama m(cfg, 21);
    randomise(m, 6);
    NoGrad ng;
    const Matrix full = m.forward(kIds)->data;
    nn::LlamaKV kv;
    const std::vector<int> pre(kIds.begin(), kIds.begin() + 6);
    Matrix got(kIds.size(), full.cols());
    Matrix lp = m.forward_cached(pre, kv)->data;
    for (size_t r = 0; r < pre.size(); ++r)
        for (size_t c = 0; c < full.cols(); ++c) got(r, c) = lp(r, c);
    for (size_t t = pre.size(); t < kIds.size(); ++t) {
        const Matrix l1 = m.forward_cached({kIds[t]}, kv)->data;
        for (size_t c = 0; c < full.cols(); ++c) got(t, c) = l1(0, c);
    }
    CHECK(kv.k.size() == m.schedule.size() && kv.len == kIds.size());
    const double r = rel(got, full);
    std::printf("  forward_cached (prefill 6, decode 4; %s) vs forward: worst rel %.2e\n", label, r);
    CHECK(r < 1e-5);
}

void test_checkpointing(const nn::LlamaConfig& cfg) {
    nn::Llama m(cfg, 13);
    randomise(m, 4);
    clear_grads(m);
    backward(ops::cross_entropy(m.forward(kIds), kTargets));
    const auto g0 = grads(m);
    clear_grads(m);
    m.checkpoint_blocks = true;
    backward(ops::cross_entropy(m.forward(kIds), kTargets));
    const auto g1 = grads(m);
    double worst = 0;
    size_t diff = 0;
    for (const auto& [n, g] : g0) {
        worst = std::max(worst, rel(g1.at(n), g));
        diff += bits_differ(g1.at(n), g);
    }
    std::printf("  checkpoint_blocks per sublayer: gradients %zu bits differ, worst rel %.2e\n", diff, worst);
    CHECK(worst < 1e-6);
}

bool on_device() { return device::device_ops_enabled() && device::get() == device::Device::CUDA; }

struct Run {
    std::vector<float> losses;
    std::map<std::string, Matrix> g1, params;
};

// tests/test_recursion.cpp's train(): mtstudio's step.
Run train(bool dev, size_t steps, const nn::LlamaConfig& cfg) {
    Run r;
    device::set_device_ops(dev);
    if (dev) {
        device::set_step_residency(true);
        device::set_defer_downloads(true);
    }
    nn::Llama m(cfg, 41);
    randomise(m, 12, 0.2f);
    m.train();
    nn::AdamW opt(m.parameters(), 3e-3f);
    std::mt19937 data(5);
    const size_t T = cfg.n_ctx, B = 2, accum = 2;
    for (size_t step = 1; step <= steps; ++step) {
        device::step_begin();
        opt.zero_grad();
        float lsum = 0;
        for (size_t k = 0; k < accum; ++k) {
            std::vector<int> x, y;
            for (size_t b = 0; b < B * T; ++b) {
                x.push_back(static_cast<int>(data() % cfg.vocab));
                y.push_back((x.back() * 7 + 3) % static_cast<int>(cfg.vocab));
            }
            Var loss = ops::cross_entropy(m.forward(x, T), y);
            backward(ops::scale(loss, 1.0f / accum));
            lsum += loss->data(0, 0) / accum;
        }
        if (!device::deferring()) device::step_end();
        if (step == 1) {
            device::materialize_all();
            for (const auto& [n, p] : m.named_parameters()) r.g1.emplace(n, p->grad);
        }
        ops::clip_grad_norm(m.parameters(), 1.0f);
        opt.step();
        if (device::deferring()) {
            const auto ps = m.parameters();
            for (const auto& p : ps) device::discard(p->grad);
            if (step < steps) {
                std::vector<const Matrix*> keep;
                for (const auto& p : ps) keep.push_back(&p->data);
                device::step_end_keeping(keep);
            } else {
                device::step_end();
            }
        }
        r.losses.push_back(lsum);
    }
    device::materialize_all();
    r.params = m.state_dict();
    if (dev) {
        device::set_defer_downloads(false);
        device::set_step_residency(false);
    }
    device::set_device_ops(true);
    return r;
}

void test_device_steps(const nn::LlamaConfig& cfg, const char* label) {
    const size_t steps = 4;
    const Run a = train(true, steps, cfg), b = train(false, steps, cfg);
    double lw = 0, gw = 0, pw = 0;
    size_t pbits = 0, ptotal = 0;
    for (size_t i = 0; i < steps; ++i)
        lw = std::max(lw, std::fabs((double)a.losses[i] - b.losses[i]) / std::fabs((double)b.losses[i]));
    for (const auto& [n, g] : b.g1) gw = std::max(gw, rel(a.g1.at(n), g));
    for (const auto& [n, p] : b.params) {
        for (size_t i = 0; i < p.rows(); ++i)
            for (size_t j = 0; j < p.cols(); ++j)
                pw = std::max(pw, std::fabs((double)a.params.at(n)(i, j) - p(i, j)));
        pbits += bits_differ(a.params.at(n), p);
        ptotal += p.rows() * p.cols();
    }
    std::printf("  %s, %zu steps, accum 2 (%s): loss worst rel %.2e, step-1 gradients worst rel %.2e, final "
                "parameters max |diff| %.2e (%zu/%zu bits differ); losses %.4f -> %.4f\n",
                label, steps, on_device() ? "device op set + residency + deferred downloads vs host" : "host vs host",
                lw, gw, pw, pbits, ptotal, b.losses.front(), b.losses.back());
    if (on_device()) {
        CHECK(lw < 1e-4 && gw < 1e-3 && pw < 4 * 3e-3);
    } else {
        CHECK(pbits == 0 && lw == 0 && gw == 0);
    }
}

// The fused op's device kernels against the host loop, at a 20m_L16-like
// width (d 320, S 33): the kernels' reductions run over 10 elements per lane.
void test_device_op() {
    std::mt19937 g(8);
    const size_t S = 33, T = 64, d = 320;
    std::vector<Var> src;
    for (size_t i = 0; i < S; ++i) src.push_back(make_var(rnd(T, d, g, 0.02f + 0.01f * (i % 4)), true));
    Var w = make_var(rnd(1, d, g, 0.5f), true);
    const Matrix wt = rnd(T, d, g, 1.0f);
    auto run = [&](bool dev) {
        device::set_device_ops(dev);
        for (auto& s : src) s->grad = Matrix();
        w->grad = Matrix();
        Var out = ops::depth_attend(src, w, ops::kRmsNormEps, static_cast<float>(S));
        backward(ops::mean(ops::mul(out, make_var(wt))));
        device::materialize_all();
        std::vector<Matrix> gs{out->data};
        for (auto& s : src) gs.push_back(s->grad);
        gs.push_back(w->grad);
        device::set_device_ops(true);
        return gs;
    };
    const auto a = run(true), b = run(false);
    double ow = rel(a[0], b[0]), gw = 0;
    for (size_t i = 1; i < a.size(); ++i) gw = std::max(gw, rel(a[i], b[i]));
    std::printf("  depth_attend S %zu, T %zu, d %zu (%s): output worst rel %.2e, gradients worst rel %.2e\n", S, T,
                d, on_device() ? "device kernels vs host" : "host vs host", ow, gw);
    if (on_device()) {
        CHECK(ow < 1e-5 && gw < 1e-4);
    } else {
        CHECK(ow == 0 && gw == 0);
    }
}

void test_refusals() {
    auto refused = [](nn::LlamaConfig c, const char* what) {
        try {
            nn::Llama m(c, 1);
        } catch (const std::exception& e) {
            CHECK(std::string(e.what()).find(what) != std::string::npos);
            return;
        }
        CHECK(false);
    };
    nn::LlamaConfig mor = tiny_rec(true);
    mor.recursion.schedule = "mor";
    refused(mor, "schedule mor is not implemented");
    nn::LlamaConfig inj = tiny_rec(true);
    inj.recursion.inject = "add";
    refused(inj, "inject add");
    nn::LlamaConfig moe = tiny(2, true);
    moe.moe.experts = 2;
    refused(moe, "MoE");
    refused(tiny(2, true, "softmax"), "scale must be");
    std::printf("  refused: schedule mor, inject add, MoE, an unknown scale\n");
}

}  // namespace

int main() {
    device::set_from_env();
    std::printf("attention residuals on the llama block (llama_attnres)\n");
    test_fused_vs_composed(1.0f);
    test_fused_vs_composed(5.0f);
    test_fused_fd();
    test_parameters(tiny(3, true), 2 * 3);         // ffn_q x 3, attn_q x 2, out_q
    test_parameters(tiny_rec(true), 2 * 8);        // 8 applications
    fd_model(tiny(3, true), "3 layers, scale sources");
    fd_model(tiny(3, true, "none"), "3 layers, scale none");
    fd_model(tiny_rec(true), "1 + [2] x 3 + 1, per-application queries");
    double r_none_default = 0;
    test_init_equivalence(tiny(3, true), false, "scale sources", nullptr);
    test_init_equivalence(tiny(3, true), true, "scale sources", nullptr);
    test_init_equivalence(tiny_rec(true), true, "scale sources, recursion", nullptr);
    test_init_equivalence(tiny(3, true, "none"), false, "scale none", &r_none_default);
    test_init_equivalence(tiny(3, true, "none"), true, "scale none", nullptr);
    test_init_equivalence(tiny(16, true, "none"), false, "scale none, 16 layers", nullptr);
    test_init_equivalence(tiny(16, true), false, "scale sources, 16 layers", nullptr);
    CHECK(r_none_default > 1e-3);  // the reason for the default
    test_kv_cache(tiny(3, true), "3 layers");
    test_kv_cache(tiny(3, true, "none"), "3 layers, scale none");
    test_kv_cache(tiny_rec(true), "1 + [2] x 3 + 1");
    test_checkpointing(tiny(3, true));
    test_refusals();
    std::printf("device (%s)\n", on_device() ? "device op set ON" : "CPU build or op set off");
    test_device_op();
    test_device_steps(tiny(3, true), "3 layers");
    test_device_steps(tiny_rec(true), "1 + [2] x 3 + 1");
    std::printf("\n[PASS] llama_attnres\n");
    return 0;
}
