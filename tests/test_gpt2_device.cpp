// The gpt2 family (ParityLM: exact, sliding-window, Kimi and SRD attention;
// FlexLM with LayerNorm) under the device op set, step residency and
// deferred downloads: defect D1 (docs/CUDA_PHASE_B2.md) and the device SFT
// evaluation (mtstudio, LOCAL_LLM_STACK.md section 11, items 5.2 and 5.4).
//   1. NoGrad forwards inside a deferring step window (mtstudio's device SFT
//      evaluation) against the same forward outside one: masked
//      cross-entropy within 1e-6 relative, logits within 1e-5; and once every
//      Var of the forward has died, no value-cache entry is left stale. A
//      forward cache that dies without discard() (LayerNorm's xhat did) leaves
//      one keyed by freed storage, which step_end()'s materialize_all then
//      writes into: D1's heap corruption. The llama family is checked too.
//   2. Training steps as mtstudio runs them under deferral (clip and AdamW in
//      the window, the grads discarded, a plain step_end: the gpt2 family never
//      keeps parameters on the device) against the host path: losses 1e-4,
//      step-1 gradients 1e-3, final parameters 4 x lr. Kimi and SRD pull their
//      operands home for their host loops, which this exercises as well.
// On a CPU build every leg runs host against host and must agree bit for bit.
// CUDA correctness is validated on a cloud GPU (tools/colab_cuda_validate.sh).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../tools/parity_model.hpp"
#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/llama.hpp"

using namespace microtorch;

namespace {

constexpr size_t kVocab = 37, kD = 16, kHeads = 2, kT = 12;

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

// Every parameter drawn afresh (LayerNorm gains and biases included), so no
// path hides behind an identity initialisation.
void randomise(nn::Module& m, unsigned seed, float scale = 0.3f) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto& p : m.parameters())
        for (size_t i = 0; i < p->data.rows(); ++i)
            for (size_t j = 0; j < p->data.cols(); ++j) p->data(i, j) = nd(g) + (p->data(i, j) == 1.0f ? 1.0f : 0.0f);
}

bool on_device() { return device::device_ops_enabled() && device::get() == device::Device::CUDA; }

struct Model {
    std::string label;
    std::function<std::shared_ptr<nn::Module>(unsigned)> make;
    std::function<Var(nn::Module&, const std::vector<int>&, size_t)> fwd;
};

std::vector<Model> models() {
    using parity::AttnKind;
    std::vector<Model> ms;
    auto parity_model = [&](const char* label, AttnKind k) {
        ms.push_back({label,
                      [k](unsigned s) { return std::make_shared<parity::ParityLM>(k, kVocab, kD, kHeads, 2 * kT, s, 4, 1); },
                      [](nn::Module& m, const std::vector<int>& x, size_t sl) {
                          return static_cast<parity::ParityLM&>(m).forward(x, sl);
                      }});
    };
    parity_model("gpt2 exact", AttnKind::EXACT);
    parity_model("gpt2 swa (window 4, 1 sink)", AttnKind::SWA);
    parity_model("gpt2 kimi", AttnKind::KIMI);
    parity_model("gpt2 srd", AttnKind::SRD);
    ms.push_back({"flex layernorm/gelu, 3 layers",
                  [](unsigned s) {
                      parity::FlexConfig c;
                      c.vocab = kVocab;
                      c.d = kD;
                      c.n_layers = 3;
                      c.n_heads = kHeads;
                      c.d_ff = 24;
                      c.n_ctx = 2 * kT;
                      return std::make_shared<parity::FlexLM>(c, s);
                  },
                  [](nn::Module& m, const std::vector<int>& x, size_t sl) {
                      return static_cast<parity::FlexLM&>(m).forward(x, sl);
                  }});
    ms.push_back({"llama 2 layers",
                  [](unsigned s) {
                      nn::LlamaConfig c;
                      c.vocab = kVocab;
                      c.d = kD;
                      c.n_heads = kHeads;
                      c.d_ff = 24;
                      c.n_ctx = 2 * kT;
                      c.n_layers = 2;
                      return std::make_shared<nn::Llama>(c, s);
                  },
                  [](nn::Module& m, const std::vector<int>& x, size_t sl) {
                      return static_cast<nn::Llama&>(m).forward(x, sl);
                  }});
    return ms;
}

void deferral(bool on) {
    device::set_step_residency(on);
    device::set_defer_downloads(on);
}

// Two stacked sequences with a completion mask, as mtstudio's eval_batch > 1.
struct Batch {
    std::vector<int> x, y;
    std::vector<uint8_t> m;
};
Batch batch(unsigned seed) {
    std::mt19937 g(seed);
    Batch b;
    for (size_t i = 0; i < 2 * kT; ++i) {
        b.x.push_back(static_cast<int>(g() % kVocab));
        b.y.push_back(static_cast<int>(g() % kVocab));
        b.m.push_back(i % kT >= kT / 2 ? 1 : 0);
    }
    return b;
}

void test_eval_in_window(const Model& md) {
    auto m = md.make(23);
    randomise(*m, 7);
    m->eval();
    const Batch b = batch(3);
    struct Out {
        double loss = 0;
        Matrix logits;
        size_t left_stale = 0;
    };
    auto run = [&](bool window) {
        Out o;
        if (window) {
            deferral(true);
            device::step_begin();
        }
        {
            NoGrad ng;
            Var lg = md.fwd(*m, b.x, kT);
            o.loss = ops::cross_entropy_masked(lg, b.y, b.m)->data(0, 0);
            device::materialize(lg->data);
            o.logits = lg->data;
        }
        o.left_stale = device::stale_count();
        if (window) {
            device::step_end();
            deferral(false);
        }
        return o;
    };
    const Out host = run(false), dev = run(true);
    const double lr = std::fabs(dev.loss - host.loss) / std::fabs(host.loss), gr = rel(dev.logits, host.logits);
    std::printf("  %s: eval forward in a window vs outside (%s): loss rel %.2e, logits rel %.2e, stale entries left %zu\n",
                md.label.c_str(), on_device() ? "deferred vs write-through" : "host vs host", lr, gr, dev.left_stale);
    CHECK(dev.left_stale == 0 && host.left_stale == 0);
    if (on_device()) {
        CHECK(lr <= 1e-6 && gr < 1e-5);
    } else {
        CHECK(lr == 0 && bits_differ(dev.logits, host.logits) == 0);
    }
}

struct Run {
    std::vector<float> losses;
    std::map<std::string, Matrix> g1, params;
};

Run train(const Model& md, bool dev, size_t steps) {
    Run r;
    device::set_device_ops(dev);
    if (dev) deferral(true);
    auto m = md.make(41);
    randomise(*m, 12, 0.2f);
    m->train();
    nn::AdamW opt(m->parameters(), 3e-3f);
    std::mt19937 data(5);
    const size_t accum = 2;
    for (size_t step = 1; step <= steps; ++step) {
        device::step_begin();
        opt.zero_grad();
        float lsum = 0;
        for (size_t k = 0; k < accum; ++k) {
            std::vector<int> x, y;
            for (size_t i = 0; i < 2 * kT; ++i) {
                x.push_back(static_cast<int>(data() % kVocab));
                y.push_back((x.back() * 7 + 3) % static_cast<int>(kVocab));
            }
            Var loss = ops::cross_entropy(md.fwd(*m, x, kT), y);
            backward(ops::scale(loss, 1.0f / accum));
            lsum += loss->data(0, 0) / accum;
        }
        const bool in_window = device::deferring();
        if (!in_window) device::step_end();
        if (step == 1) {
            device::materialize_all();
            for (const auto& [n, p] : m->named_parameters()) r.g1.emplace(n, p->grad);
        }
        ops::clip_grad_norm(m->parameters(), 1.0f);
        opt.step();
        if (in_window) {
            for (const auto& p : m->parameters()) device::discard(p->grad);
            device::step_end();
        }
        r.losses.push_back(lsum);
    }
    device::materialize_all();
    r.params = m->state_dict();
    if (dev) deferral(false);
    device::set_device_ops(true);
    return r;
}

void test_device_steps(const Model& md) {
    const size_t steps = 4;
    const Run a = train(md, true, steps), b = train(md, false, steps);
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
                md.label.c_str(), steps,
                on_device() ? "device op set + residency + deferred downloads vs host" : "host vs host", lw, gw, pw,
                pbits, ptotal, b.losses.front(), b.losses.back());
    if (on_device()) {
        CHECK(lw < 1e-4 && gw < 1e-3 && pw < 4 * 3e-3);
    } else {
        CHECK(pbits == 0 && lw == 0 && gw == 0);
    }
}

}  // namespace

int main() {
    device::set_from_env();
    // The legs switch residency and deferral themselves; the op set is the
    // environment's (MICROTORCH_DEVICE_OPS=1 on the device run).
    deferral(false);
    std::printf("gpt2 family on the device (%s)\n", on_device() ? "device op set ON" : "CPU build or op set off");
    for (const Model& md : models()) test_eval_in_window(md);
    for (const Model& md : models()) test_device_steps(md);
    std::printf("\n[PASS] gpt2_device\n");
    return 0;
}
