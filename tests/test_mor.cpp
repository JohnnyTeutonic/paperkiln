// Mixture-of-Recursions, causal token choice (nn::LlamaConfig::MoR; spec
// arch.custom.recursion schedule "mor"; tiny-chat amendment 4).
//   1. Forced to the largest depth with gate "none", the routed model is
//      fixed recursion bit for bit (logits and every shared gradient), and
//      forced to depth 2 it is fixed recursion with two repeats; with gate
//      "score" each step is x + g (S(x) - x), checked against that formula.
//   2. Routing: depths 1 .. N_r all occur on a randomised model; the gate is
//      the chosen depth's softmax probability; the balancing loss equals
//      N_r sum_i f_i P_i and the z-loss mean logsumexp^2.
//   3. Causality: a token's depth and logits do not depend on later tokens
//      (every prefix run alone); prefill equals token-by-token decode and a
//      prefill + decode split (logits to rounding, depths exactly); prefix
//      reuse after truncate equals a fresh cache; two stacked sequences
//      equal each alone (no leakage across sequences).
//   4. kv_rows "active": each application's cache holds exactly the rows of
//      the tokens active there (counts and positions).
//   5. Finite-difference gradcheck of the routed loss (task + 0.1 balance +
//      1e-3 z-loss), router included; probes whose perturbation would
//      change a depth are skipped and counted.
//   6. Activation checkpointing per application: the same gradients.
//   7. Training steps as mtstudio runs them (see test_recursion.cpp): device
//      op set + residency + deferred downloads against the host path on a
//      CUDA build with MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1; host
//      against host on CPU (bit for bit).
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

nn::LlamaConfig tiny(size_t repeats = 3, const std::string& schedule = "mor", const std::string& gate = "score",
                     size_t force = 0) {
    nn::LlamaConfig c;
    c.vocab = 37;
    c.d = 16;
    c.n_heads = 2;
    c.d_ff = 24;
    c.n_ctx = 16;
    c.n_layers = 4;
    c.recursion.prelude = 1;
    c.recursion.block = 2;
    c.recursion.repeats = repeats;
    c.recursion.coda = 1;
    c.recursion.schedule = schedule;
    c.recursion.mor.gate = gate;
    c.recursion.mor.force_depth = force;
    return c;
}

const std::vector<int> kIds = {5, 17, 3, 22, 22, 9, 31, 2, 14, 6, 11, 28};
const std::vector<int> kTargets = {17, 3, 22, 22, 9, 31, 2, 14, 6, 1, 28, 4};

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

// As test_recursion.cpp; the router gets a larger scale so that every depth
// occurs on 12 tokens.
void randomise(nn::Module& m, unsigned seed, float scale = 0.3f) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto& [name, p] : m.named_parameters()) {
        const bool norm = name.find("norm") != std::string::npos;
        const float sc = name.find("router") != std::string::npos ? 4.0f : 1.0f;
        for (size_t i = 0; i < p->data.rows(); ++i)
            for (size_t j = 0; j < p->data.cols(); ++j) p->data(i, j) = norm ? 1.0f + 0.2f * nd(g) : sc * nd(g);
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

std::vector<int> histogram(const std::vector<int>& depth, size_t R);

// randomise() from the first seed >= `seed` under which kIds (as one
// sequence and as two of 6) takes every depth, so each test exercises
// exited tokens at every step.
unsigned diverse(nn::Llama& m, unsigned seed, float scale = 0.3f) {
    for (unsigned s = seed; s < seed + 200; ++s) {
        randomise(m, s, scale);
        NoGrad ng;
        bool ok = true;
        for (size_t sl : {size_t(0), size_t(6)}) {
            m.forward(kIds, sl);
            for (int c : histogram(m.last_depth, m.cfg.recursion.repeats)) ok = ok && c > 0;
        }
        if (ok) return s;
    }
    CHECK(false);
    return 0;
}

// The routed loss mtstudio trains with: task + 0.1 balance + 1e-3 z-loss.
Var routed_loss(const nn::Llama& m, const std::vector<int>& ids, const std::vector<int>& y, size_t seq_len = 0) {
    nn::MoEAux aux;
    Var l = ops::cross_entropy(m.forward(ids, seq_len, &aux), y);
    return ops::add(l, ops::add(ops::scale(aux.lb, 0.1f), ops::scale(aux.z, 1e-3f)));
}

std::vector<int> histogram(const std::vector<int>& depth, size_t R) {
    std::vector<int> h(R, 0);
    for (int d : depth) ++h[static_cast<size_t>(d - 1)];
    return h;
}

void test_forced_is_fixed() {
    for (size_t force : {3, 2}) {
        nn::Llama mor(tiny(3, "mor", "none", force), 42);
        randomise(mor, 3);
        nn::Llama fixed(tiny(force, "fixed"), 42);
        auto sd = mor.state_dict();
        sd.erase("recursion.router.weight");
        fixed.load_state_dict(sd, /*strict=*/true);
        // One sequence and two stacked ones.
        for (size_t sl : {size_t(0), size_t(6)}) {
            clear_grads(mor);
            clear_grads(fixed);
            Var a = mor.forward(kIds, sl), b = fixed.forward(kIds, sl);
            const size_t fwd = bits_differ(a->data, b->data);
            backward(ops::cross_entropy(a, kTargets));
            backward(ops::cross_entropy(b, kTargets));
            const auto ga = grads(mor), gb = grads(fixed);
            size_t diff = 0, n = 0;
            for (const auto& [name, g] : gb) {
                diff += bits_differ(ga.at(name), g);
                n += g.rows() * g.cols();
            }
            CHECK(std::all_of(mor.last_depth.begin(), mor.last_depth.end(),
                              [&](int d) { return d == static_cast<int>(force); }));
            std::printf("  forced depth %zu, gate none, %s vs fixed 1 + [2] x %zu + 1: logits %zu bits differ, "
                        "%zu gradients (%zu entries) %zu bits differ\n",
                        force, sl ? "2 stacked sequences" : "one sequence", force, fwd, gb.size(), n, diff);
            CHECK(fwd == 0 && diff == 0);
        }
    }
    // gate "score", every token at depth 3: each step is x + g (S(x) - x)
    // with g the depth-3 probability, so it equals the fixed model only
    // where g = 1. Check the formula on the first step against a direct
    // computation over the same layers.
    nn::Llama mor(tiny(3, "mor", "score", 3), 7);
    randomise(mor, 5);
    NoGrad ng;
    const Matrix got = mor.forward(kIds)->data;
    std::vector<int> pos(kIds.size());
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i);
    Var h = mor.embed_tokens->forward(kIds);
    h = mor.blocks[0]->forward(h, pos);
    Var p = ops::softmax_row(mor.rec_router->forward(h));
    std::vector<int> rows(kIds.size()), cols(kIds.size(), 2);
    for (size_t t = 0; t < rows.size(); ++t) rows[t] = static_cast<int>(t);
    Var g = ops::gather_elems(p, rows, cols, 1);
    for (size_t k = 0; k < 3; ++k) {
        Var y = mor.blocks[2]->forward(mor.blocks[1]->forward(h, pos), pos);
        h = ops::add(h, ops::mul_col(ops::sub(y, h), g));
    }
    h = mor.blocks[3]->forward(h, pos);
    h = ops::rmsnorm(h, mor.norm_w);
    const Matrix want = ops::matmul(h, ops::transpose(mor.embed_tokens->weight))->data;
    std::printf("  gate score, forced depth 3: logits vs x + g (S(x) - x) per step, %zu bits differ\n",
                bits_differ(got, want));
    CHECK(bits_differ(got, want) == 0);
}

void test_routing_and_losses() {
    nn::Llama m(tiny(), 11);
    diverse(m, 9);
    nn::MoEAux aux;
    Var logits = m.forward(kIds, 0, &aux);
    const auto hist = histogram(m.last_depth, 3);
    std::printf("  routed depths over %zu tokens: %d at 1, %d at 2, %d at 3\n", kIds.size(), hist[0], hist[1],
                hist[2]);
    CHECK(hist[0] > 0 && hist[1] > 0 && hist[2] > 0);
    // Recompute the routing by hand from the prelude's output.
    NoGrad ng;
    std::vector<int> pos(kIds.size());
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i);
    Var h = m.blocks[0]->forward(m.embed_tokens->forward(kIds), pos);
    const Matrix l = m.rec_router->forward(h)->data;
    const Matrix p = ops::softmax_row(m.rec_router->forward(h))->data;
    double lb = 0, z = 0;
    std::vector<double> f(3, 0.0), P(3, 0.0);
    for (size_t t = 0; t < kIds.size(); ++t) {
        size_t best = 0;
        for (size_t j = 1; j < 3; ++j)
            if (l(t, j) > l(t, best)) best = j;
        CHECK(m.last_depth[t] == static_cast<int>(best + 1));
        f[best] += 1.0 / kIds.size();
        double mx = std::max({l(t, 0), l(t, 1), l(t, 2)}), s = 0;
        for (size_t j = 0; j < 3; ++j) {
            s += std::exp((double)l(t, j) - mx);
            P[j] += p(t, j) / kIds.size();
        }
        z += std::pow(mx + std::log(s), 2) / kIds.size();
    }
    for (size_t i = 0; i < 3; ++i) lb += 3.0 * f[i] * P[i];
    const double elb = std::fabs(aux.lb->data(0, 0) - lb), ez = std::fabs(aux.z->data(0, 0) - z) / z;
    std::printf("  balance %.6f (by hand %.6f, |diff| %.1e), z-loss %.6f (rel diff %.1e); depth fractions "
                "%.3f %.3f %.3f\n",
                aux.lb->data(0, 0), lb, elb, aux.z->data(0, 0), ez, aux.frac[0][0], aux.frac[0][1], aux.frac[0][2]);
    CHECK(elb < 1e-5 && ez < 1e-5);
    CHECK(aux.layers == 1 && aux.frac.size() == 1);
}

void test_causality() {
    nn::Llama m(tiny(), 21);
    diverse(m, 6);
    NoGrad ng;
    const Matrix full = m.forward(kIds)->data;
    const std::vector<int> depth = m.last_depth;
    // Every prefix alone: its depths and its last row match the full run.
    size_t dmis = 0;
    double worst_prefix = 0;
    for (size_t t = 1; t <= kIds.size(); ++t) {
        const std::vector<int> pre(kIds.begin(), kIds.begin() + t);
        const Matrix lp = m.forward(pre)->data;
        for (size_t i = 0; i < t; ++i) dmis += m.last_depth[i] != depth[i];
        Matrix a(1, full.cols()), b(1, full.cols());
        for (size_t c = 0; c < full.cols(); ++c) {
            a(0, c) = lp(t - 1, c);
            b(0, c) = full(t - 1, c);
        }
        worst_prefix = std::max(worst_prefix, rel(a, b));
    }
    std::printf("  every prefix alone vs the full sequence: %zu depth mismatches, last-row logits worst rel %.2e\n",
                dmis, worst_prefix);
    CHECK(dmis == 0 && worst_prefix < 1e-5);

    // Token-by-token decode and a prefill 7 + decode split.
    for (size_t pre : {size_t(1), size_t(7)}) {
        nn::LlamaKV kv;
        Matrix got(kIds.size(), full.cols());
        std::vector<int> dgot;
        const Matrix lp = m.forward_cached(std::vector<int>(kIds.begin(), kIds.begin() + pre), kv)->data;
        dgot = m.last_depth;
        for (size_t r = 0; r < pre; ++r)
            for (size_t c = 0; c < full.cols(); ++c) got(r, c) = lp(r, c);
        for (size_t t = pre; t < kIds.size(); ++t) {
            const Matrix l1 = m.forward_cached({kIds[t]}, kv)->data;
            dgot.push_back(m.last_depth[0]);
            for (size_t c = 0; c < full.cols(); ++c) got(t, c) = l1(0, c);
        }
        // kv_rows "active": application a holds the rows of the tokens whose
        // depth reaches its step, at their positions.
        size_t bad_rows = 0;
        for (size_t a = 0; a < m.schedule.size(); ++a) {
            const int step = a < 1 || a >= 7 ? -1 : static_cast<int>((a - 1) / 2);
            std::vector<int> want;
            for (size_t t = 0; t < kIds.size(); ++t)
                if (step < 0 || depth[t] > step) want.push_back(static_cast<int>(t));
            bad_rows += kv.pos[a] != want || kv.k[a].size() != want.size() * 16;
        }
        const double r = rel(got, full);
        std::printf("  prefill %zu + decode %zu vs forward: logits worst rel %.2e, depths %s, per-application rows "
                    "%s (%zu/%zu/%zu rows at steps 0/1/2)\n",
                    pre, kIds.size() - pre, r, dgot == depth ? "identical" : "DIFFER",
                    bad_rows ? "WRONG" : "exactly the active tokens'", kv.pos[1].size(), kv.pos[3].size(),
                    kv.pos[5].size());
        CHECK(r < 1e-5 && dgot == depth && bad_rows == 0);
    }

    // Prefix reuse: truncate to 5 and continue, against a fresh cache.
    {
        nn::LlamaKV kv, fresh;
        m.forward_cached(kIds, kv);
        kv.truncate(5);
        const Matrix a = m.forward_cached(std::vector<int>(kIds.begin() + 5, kIds.end()), kv)->data;
        m.forward_cached(std::vector<int>(kIds.begin(), kIds.begin() + 5), fresh);
        const Matrix b = m.forward_cached(std::vector<int>(kIds.begin() + 5, kIds.end()), fresh)->data;
        std::printf("  truncate(5) and continue vs a fresh cache: %zu bits differ\n", bits_differ(a, b));
        CHECK(bits_differ(a, b) == 0);
    }

    // Two stacked sequences equal each alone (rows and depths).
    {
        const std::vector<int> s1(kIds.begin(), kIds.begin() + 6), s2(kIds.begin() + 6, kIds.end());
        const Matrix both = m.forward(kIds, 6)->data;
        const std::vector<int> dboth = m.last_depth;
        const Matrix a = m.forward(s1)->data;
        std::vector<int> d12 = m.last_depth;
        const Matrix b = m.forward(s2)->data;
        d12.insert(d12.end(), m.last_depth.begin(), m.last_depth.end());
        Matrix sep(kIds.size(), both.cols());
        for (size_t r = 0; r < 6; ++r)
            for (size_t c = 0; c < both.cols(); ++c) {
                sep(r, c) = a(r, c);
                sep(r + 6, c) = b(r, c);
            }
        const double r = rel(both, sep);
        std::printf("  2 stacked sequences vs each alone: logits worst rel %.2e, depths %s (%d, %d, %d)\n", r,
                    dboth == d12 ? "identical" : "DIFFER", histogram(dboth, 3)[0], histogram(dboth, 3)[1],
                    histogram(dboth, 3)[2]);
        CHECK(r < 1e-5 && dboth == d12);
    }
}

void fd_check(const nn::LlamaConfig& cfg, size_t seq_len, const char* label) {
    nn::Llama m(cfg, 5);
    diverse(m, 9);
    clear_grads(m);
    backward(routed_loss(m, kIds, kTargets, seq_len));
    const std::vector<int> depth = m.last_depth;
    std::mt19937 rng(17);
    const float eps = 5e-3f;
    double worst = 0;
    size_t probes = 0, skipped = 0;
    std::string worst_at;
    for (const auto& [n, p] : m.named_parameters()) {
        const bool many = n.find("router") != std::string::npos || n.rfind("layers.1.", 0) == 0 ||
                          n.rfind("layers.2.", 0) == 0;
        const int k = many ? 4 : 1;
        for (int t = 0; t < k; ++t) {
            const size_t i = rng() % p->data.rows(), j = rng() % p->data.cols();
            double fp, fm;
            bool same = true;
            {
                NoGrad ng;
                const float keep = p->data(i, j);
                p->data(i, j) = keep + eps;
                fp = routed_loss(m, kIds, kTargets, seq_len)->data(0, 0);
                same = same && m.last_depth == depth;
                p->data(i, j) = keep - eps;
                fm = routed_loss(m, kIds, kTargets, seq_len)->data(0, 0);
                same = same && m.last_depth == depth;
                p->data(i, j) = keep;
            }
            if (!same) {
                ++skipped;  // a depth flips inside +-eps: the loss is not differentiable there
                continue;
            }
            const double fd = (fp - fm) / (2.0 * eps), an = p->grad(i, j);
            const double r = std::fabs(fd - an) / std::max({std::fabs(fd), std::fabs(an), 1e-2});
            if (r > worst) {
                worst = r;
                worst_at = n;
            }
            ++probes;
        }
    }
    const auto hist = histogram(depth, cfg.recursion.repeats);
    std::printf("  FD gradcheck %s (depths %d/%d/%d): %zu entries, %zu skipped (a depth flips), worst rel %.2e (%s)\n",
                label, hist[0], hist[1], hist[2], probes, skipped, worst, worst_at.c_str());
    CHECK(probes > 40 && worst < 2e-2);
}

void test_checkpointing() {
    nn::Llama m(tiny(), 13);
    diverse(m, 4);
    clear_grads(m);
    backward(routed_loss(m, kIds, kTargets, 6));
    const auto g0 = grads(m);
    clear_grads(m);
    m.checkpoint_blocks = true;
    backward(routed_loss(m, kIds, kTargets, 6));
    const auto g1 = grads(m);
    size_t diff = 0;
    double worst = 0;
    for (const auto& [n, g] : g0) {
        diff += bits_differ(g, g1.at(n));
        worst = std::max(worst, rel(g1.at(n), g));
    }
    std::printf("  checkpoint_blocks per application (routed, 2 stacked sequences): gradients %zu bits differ, "
                "worst rel %.2e\n",
                diff, worst);
    CHECK(worst < 1e-6);
}

void test_ir() {
    nn::Llama m(tiny(), 3);
    const ir::json doc = m.export_ir();
    CHECK(doc["family_hint"] == "llama_mor");
    CHECK(doc["block"]["repeat"] == 8 && doc["model"]["n_layers"] == 8);
    CHECK(doc["block"]["weight_index"]["per_layer"] == ir::json::array({0, 1, 2, 1, 2, 1, 2, 3}));
    const auto& r = doc["block"]["nodes"][0];
    CHECK(r["op"] == "depth_route" && r["weight"] == "rec.0.router.weight" && r["n_rec"] == 3);
    CHECK(r["step"]["per_layer"] == ir::json::array({-1, 0, 0, 1, 1, 2, 2, -1}));
    CHECK(doc["block"]["state"][0]["kv_rows"] == "active");
    nn::Llama forced(tiny(3, "mor", "score", 2), 3);
    bool threw = false;
    try {
        forced.export_ir();
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
    std::printf("  IR: family llama_mor, depth_route first (rec.0.router.weight, steps -1 0 0 1 1 2 2 -1), kv_rows "
                "active; force_depth has no IR form (refused)\n");
}

bool on_device() { return device::device_ops_enabled() && device::get() == device::Device::CUDA; }

struct Run {
    std::vector<float> losses;
    std::map<std::string, Matrix> g1, params;
    std::vector<int> depth1;
};

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
            Var loss = routed_loss(m, x, y, T);
            if (step == 1 && k == 0) r.depth1 = m.last_depth;
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

void test_device_steps() {
    const size_t steps = 4;
    const nn::LlamaConfig cfg = tiny();
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
    const auto h = histogram(b.depth1, 3);
    std::printf("  routed 1 + [2] x {1..3} + 1, %zu steps, accum 2 (%s): first micro-batch depths %d/%d/%d (step-1 "
                "depths %s), loss worst rel %.2e, step-1 gradients worst rel %.2e, final parameters max |diff| %.2e "
                "(%zu/%zu bits differ); losses %.4f -> %.4f\n",
                steps, on_device() ? "device op set + residency + deferred downloads vs host" : "host vs host", h[0],
                h[1], h[2], a.depth1 == b.depth1 ? "identical" : "DIFFER", lw, gw, pw, pbits, ptotal,
                b.losses.front(), b.losses.back());
    if (on_device()) {
        CHECK(a.depth1 == b.depth1);
        CHECK(lw < 1e-4 && gw < 1e-3 && pw < 4 * 3e-3);
    } else {
        CHECK(pbits == 0 && lw == 0 && gw == 0);
    }
}

}  // namespace

int main() {
    device::set_from_env();
    std::printf("Mixture-of-Recursions, causal token choice (llama family)\n");
    test_forced_is_fixed();
    test_routing_and_losses();
    test_causality();
    fd_check(tiny(), 0, "1 + [2] x {1..3} + 1, gate score, one sequence");
    fd_check(tiny(), 6, "the same, 2 stacked sequences");
    test_checkpointing();
    test_ir();
    std::printf("device steps (%s)\n", on_device() ? "device op set ON" : "CPU build or op set off");
    test_device_steps();
    std::printf("\n[PASS] mor\n");
    return 0;
}
