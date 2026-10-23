// Tied-block recursion in the llama family (nn::LlamaConfig::Recursion;
// spec arch.custom.recursion): prelude layers, a block of B shared layers
// applied R times, coda layers.
//   1. R = 1 with no prelude or coda is the plain model bit for bit
//      (logits and every gradient).
//   2. Parameters: every stored tensor listed once (the plain model of the
//      stored layers' names exactly), so an optimiser sees each once.
//   3. Tied vs untied: the recursed model equals an unshared model whose
//      layers are copies of the shared ones, logits bit for bit, and each
//      shared gradient equals the sum of its copies' gradients.
//   4. Finite-difference gradcheck of a tiny recursed model (no injection
//      and inject "add"), on entries of shared, prelude and coda layers, the
//      norms and the embedding.
//   5. Activation checkpointing per application: the same gradients (bit
//      for bit without injection; with inject "add" the embedding's gradient
//      receives the same terms in another order, so it agrees to rounding).
//   6. forward_cached (one KV cache per application) against forward.
//   7. AdamW updates each shared tensor once per step: an AdamW step on the
//      recursed model equals one on the plain stored-layer model given the
//      same gradients, bit for bit.
//   8. Training steps as mtstudio runs them: on a CUDA build with
//      MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1, the device op set
//      with step residency and deferred downloads (zero_grad, accumulation
//      over two micro-batches, clip and AdamW inside the window, gradients
//      discarded and parameters kept on the device between steps) against
//      the host path: losses, first-step gradients and final parameters. On
//      a CPU build the legs run the host path against itself (bit for bit).
//      CUDA correctness is validated on Colab (tools/colab_cuda_validate.sh).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/ir.hpp"
#include "microtorch/llama.hpp"

using namespace microtorch;

namespace {

nn::LlamaConfig tiny(size_t prelude, size_t block, size_t repeats, size_t coda, const std::string& inject = "none") {
    nn::LlamaConfig c;
    c.vocab = 37;
    c.d = 16;
    c.n_heads = 2;
    c.d_ff = 24;
    c.n_ctx = 12;
    c.n_layers = prelude + block + coda;
    c.recursion.prelude = prelude;
    c.recursion.block = block;
    c.recursion.repeats = repeats;
    c.recursion.coda = coda;
    c.recursion.inject = inject;
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

// max |a - b| / max(1e-30, max |b|) over the tensor.
double rel(const Matrix& a, const Matrix& b) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols());
    double s = 0, w = 0;
    for (size_t i = 0; i < b.rows(); ++i)
        for (size_t j = 0; j < b.cols(); ++j) s = std::max(s, (double)std::fabs(b(i, j)));
    for (size_t i = 0; i < b.rows(); ++i)
        for (size_t j = 0; j < b.cols(); ++j) w = std::max(w, std::fabs((double)a(i, j) - b(i, j)));
    return w / std::max(s, 1e-30);
}

// Random parameters at a scale where the logits are not flat (the default
// init is small) and finite differences in f32 are well conditioned.
void randomise(nn::Module& m, unsigned seed, float scale = 0.3f) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto& [name, p] : m.named_parameters()) {
        const bool norm = name.find("norm") != std::string::npos;
        for (size_t i = 0; i < p->data.rows(); ++i)
            for (size_t j = 0; j < p->data.cols(); ++j) p->data(i, j) = norm ? 1.0f + 0.2f * nd(g) : nd(g);
    }
}

void clear_grads(nn::Module& m) {
    for (auto& p : m.parameters()) p->grad = Matrix();
}

Var loss_of(const nn::Llama& m, const std::vector<int>& ids, const std::vector<int>& y) {
    return ops::cross_entropy(m.forward(ids), y);
}

std::map<std::string, Matrix> grads(const nn::Module& m) {
    std::map<std::string, Matrix> g;
    for (const auto& [n, p] : m.named_parameters()) g.emplace(n, p->grad);
    return g;
}

void test_r1_is_plain() {
    nn::LlamaConfig pc = tiny(0, 0, 1, 0);
    pc.n_layers = 3;
    pc.recursion.block = 0;
    nn::Llama plain(pc, 42);
    nn::Llama rec(tiny(0, 3, 1, 0), 42);  // recursion on, one repeat
    CHECK(rec.schedule == (std::vector<size_t>{0, 1, 2}));
    Var a = plain.forward(kIds), b = rec.forward(kIds);
    CHECK(bits_differ(a->data, b->data) == 0);
    backward(ops::cross_entropy(a, kTargets));
    backward(ops::cross_entropy(b, kTargets));
    const auto ga = grads(plain), gb = grads(rec);
    CHECK(ga.size() == gb.size());
    size_t diff = 0;
    for (const auto& [n, g] : ga) diff += bits_differ(g, gb.at(n));
    // Its IR is the plain llama's (no weight_index, hint llama).
    CHECK(plain.export_ir() == rec.export_ir());
    std::printf("  R=1: logits and %zu gradients bit for bit (%zu bits differ); IR identical\n", ga.size(), diff);
    CHECK(diff == 0);
}

void test_parameters_once() {
    nn::Llama rec(tiny(1, 2, 3, 1), 7);
    nn::LlamaConfig pc = tiny(0, 0, 1, 0);
    pc.n_layers = 4;
    nn::Llama plain(pc, 7);
    CHECK(rec.schedule == (std::vector<size_t>{0, 1, 2, 1, 2, 1, 2, 3}));
    const auto np = rec.named_parameters(), pp = plain.named_parameters();
    CHECK(np.size() == pp.size());
    for (size_t i = 0; i < np.size(); ++i) CHECK(np[i].first == pp[i].first);
    auto ps = rec.parameters();
    std::vector<Variable*> raw;
    for (auto& p : ps) raw.push_back(p.get());
    std::sort(raw.begin(), raw.end());
    CHECK(std::adjacent_find(raw.begin(), raw.end()) == raw.end());
    CHECK(rec.parameter_count() == plain.parameter_count());
    std::printf("  1 + [2] x 3 + 1: %zu applications of 4 stored layers; %zu tensors listed once, %zu parameters "
                "(the plain 4-layer model's names and count)\n",
                rec.schedule.size(), np.size(), rec.parameter_count());
}

// The unshared model: one stored layer per application, each a copy of the
// shared layer that application runs.
nn::Llama unrolled(const nn::Llama& rec) {
    nn::LlamaConfig uc = rec.cfg;
    uc.recursion = nn::LlamaConfig::Recursion{};
    uc.n_layers = rec.schedule.size();
    nn::Llama u(uc, 1);
    const auto sd = rec.state_dict();
    std::map<std::string, Matrix> usd;
    for (const auto& [n, m] : sd) {
        if (n.rfind("layers.", 0) != 0) {
            usd.emplace(n, m);
            continue;
        }
        const size_t dot = n.find('.', 7);
        const size_t stored = std::stoul(n.substr(7, dot - 7));
        for (size_t a = 0; a < rec.schedule.size(); ++a)
            if (rec.schedule[a] == stored) usd.emplace("layers." + std::to_string(a) + n.substr(dot), m);
    }
    u.load_state_dict(usd, /*strict=*/true);
    return u;
}

void test_tied_vs_untied() {
    nn::Llama rec(tiny(1, 2, 3, 1), 11);
    randomise(rec, 3);
    nn::Llama u = unrolled(rec);
    Var a = rec.forward(kIds), b = u.forward(kIds);
    const size_t fwd = bits_differ(a->data, b->data);
    backward(ops::cross_entropy(a, kTargets));
    backward(ops::cross_entropy(b, kTargets));
    const auto gr = grads(rec), gu = grads(u);
    // Sum the copies in descending application order: the order backward
    // reaches them (the last application's gradient arrives first).
    double worst = 0;
    size_t diff = 0, total = 0;
    for (const auto& [n, g] : gr) {
        Matrix sum;
        if (n.rfind("layers.", 0) != 0) {
            sum = gu.at(n);
        } else {
            const size_t dot = n.find('.', 7);
            const size_t stored = std::stoul(n.substr(7, dot - 7));
            for (size_t a2 = rec.schedule.size(); a2-- > 0;)
                if (rec.schedule[a2] == stored) {
                    const Matrix& c = gu.at("layers." + std::to_string(a2) + n.substr(dot));
                    if (sum.rows() == 0) sum = c;
                    else sum += c;
                }
        }
        worst = std::max(worst, rel(g, sum));
        diff += bits_differ(g, sum);
        total += g.rows() * g.cols();
    }
    std::printf("  tied vs untied (1 + [2] x 3 + 1 vs 8 unshared copies): logits %zu bits differ; shared gradient vs "
                "the sum of its copies' worst rel %.2e (%zu/%zu bits differ)\n",
                fwd, worst, diff, total);
    CHECK(fwd == 0);
    CHECK(worst < 1e-6);
}

void fd_check(const nn::LlamaConfig& cfg, const char* label) {
    nn::Llama m(cfg, 5);
    randomise(m, 9);
    clear_grads(m);
    backward(loss_of(m, kIds, kTargets));
    std::mt19937 rng(17);
    const float eps = 1e-2f;
    double worst = 0;
    size_t probes = 0;
    std::vector<std::string> names;
    for (const auto& [n, p] : m.named_parameters()) names.push_back(n);
    for (const auto& [n, p] : m.named_parameters()) {
        // Every tensor of the shared block, and a sample of the rest.
        const bool shared = n.rfind("layers.", 0) == 0 && [&] {
            const size_t st = std::stoul(n.substr(7, n.find('.', 7) - 7));
            return st >= cfg.recursion.prelude && st < cfg.recursion.prelude + cfg.recursion.block;
        }();
        const int k = shared ? 3 : 1;
        for (int t = 0; t < k; ++t) {
            const size_t i = rng() % p->data.rows(), j = rng() % p->data.cols();
            double fp, fm;
            {
                NoGrad ng;
                const float keep = p->data(i, j);
                p->data(i, j) = keep + eps;
                fp = loss_of(m, kIds, kTargets)->data(0, 0);
                p->data(i, j) = keep - eps;
                fm = loss_of(m, kIds, kTargets)->data(0, 0);
                p->data(i, j) = keep;
            }
            const double fd = (fp - fm) / (2.0 * eps), an = p->grad(i, j);
            const double r = std::fabs(fd - an) / std::max({std::fabs(fd), std::fabs(an), 1e-2});
            worst = std::max(worst, r);
            ++probes;
        }
    }
    std::printf("  FD gradcheck %s: %zu entries over %zu tensors, worst rel %.2e\n", label, probes, names.size(),
                worst);
    CHECK(worst < 2e-2);
}

void test_checkpointing(const nn::LlamaConfig& cfg, const char* label) {
    nn::Llama m(cfg, 13);
    randomise(m, 4);
    clear_grads(m);
    backward(loss_of(m, kIds, kTargets));
    const auto g0 = grads(m);
    clear_grads(m);
    m.checkpoint_blocks = true;
    backward(loss_of(m, kIds, kTargets));
    const auto g1 = grads(m);
    size_t diff = 0;
    double worst = 0;
    std::string where;
    for (const auto& [n, g] : g0) {
        const size_t d = bits_differ(g, g1.at(n));
        if (d && where.size() < 200) where += " " + n;
        diff += d;
        worst = std::max(worst, rel(g1.at(n), g));
    }
    std::printf("  checkpoint_blocks per application (%s): gradients %zu bits differ, worst rel %.2e%s\n", label, diff,
                worst, where.c_str());
    CHECK(worst < 1e-6);
}

void test_kv_cache(const std::string& inject) {
    nn::Llama m(tiny(1, 2, 3, 1, inject), 21);
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
    std::printf("  forward_cached (prefill 6, decode 4; %zu caches, inject %s) vs forward: worst rel %.2e\n",
                kv.k.size(), inject.c_str(), r);
    CHECK(r < 1e-5);
}

void test_adamw_once() {
    nn::Llama rec(tiny(1, 2, 3, 1), 31);
    randomise(rec, 8);
    nn::LlamaConfig pc = tiny(0, 0, 1, 0);
    pc.n_layers = 4;
    nn::Llama plain(pc, 1);
    plain.load_state_dict(rec.state_dict(), true);
    clear_grads(rec);
    backward(loss_of(rec, kIds, kTargets));
    // The plain model gets the recursed model's gradients verbatim.
    const auto gr = grads(rec);
    for (auto& [n, p] : plain.named_parameters()) p->grad = gr.at(n);
    nn::AdamW o1(rec.parameters(), 1e-2f), o2(plain.parameters(), 1e-2f);
    o1.step();
    o2.step();
    const auto a = rec.state_dict(), b = plain.state_dict();
    size_t diff = 0, moved = 0;
    for (const auto& [n, m] : a) diff += bits_differ(m, b.at(n));
    for (const auto& [n, g] : gr) moved += g.rows() * g.cols();
    std::printf("  AdamW on the recursed model vs on the plain stored-layer model with its gradients: %zu bits "
                "differ over %zu parameters\n",
                diff, moved);
    CHECK(diff == 0);
}

bool on_device() { return device::device_ops_enabled() && device::get() == device::Device::CUDA; }

struct Run {
    std::vector<float> losses;
    std::map<std::string, Matrix> g1, params;
};

// mtstudio's step (tools/mtstudio.cpp run(): AdamW alone with deferred
// downloads keeps clip and the update inside the window).
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
        // Reductions differ in order on the device; AdamW's first steps move
        // each entry by about lr, so a few lr bounds a sign flip on a
        // near-zero gradient entry.
        CHECK(lw < 1e-4 && gw < 1e-3 && pw < 4 * 3e-3);
    } else {
        CHECK(pbits == 0 && lw == 0 && gw == 0);
    }
}

}  // namespace

int main() {
    device::set_from_env();
    std::printf("tied-block recursion (llama family)\n");
    test_r1_is_plain();
    test_parameters_once();
    test_tied_vs_untied();
    fd_check(tiny(1, 2, 3, 1), "1 + [2] x 3 + 1");
    fd_check(tiny(0, 2, 4, 0), "[2] x 4");
    fd_check(tiny(1, 2, 3, 1, "add"), "1 + [2] x 3 + 1, inject add");
    {
        nn::LlamaConfig pc = tiny(0, 0, 1, 0);
        pc.n_layers = 4;
        test_checkpointing(pc, "plain 4 layers, the reference");
    }
    test_checkpointing(tiny(1, 2, 3, 1), "1 + [2] x 3 + 1");
    test_checkpointing(tiny(1, 2, 3, 1, "add"), "1 + [2] x 3 + 1, inject add");
    test_kv_cache("none");
    test_kv_cache("add");
    test_adamw_once();
    std::printf("device steps (%s)\n", on_device() ? "device op set ON" : "CPU build or op set off");
    test_device_steps(tiny(0, 2, 4, 0), "[2] x 4");
    test_device_steps(tiny(1, 2, 3, 1, "add"), "1 + [2] x 3 + 1, inject add");
    std::printf("\n[PASS] recursion\n");
    return 0;
}
