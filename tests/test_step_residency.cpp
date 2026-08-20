// Phase B2.0 gate (docs/CUDA_PHASE_B2.md): training-step residency
// plumbing. What the write-through/epoch-scope contract can get wrong,
// checked:
//   1. NO-OP LEG (every build): the gemm-flags refactor moved ops::matmul
//      onto device::gemm — with the switch OFF and with it ON under a
//      CPU device, N training steps of a FlexLM must be BITWISE identical
//      (the switch touches nothing off the CUDA step path, and the CPU
//      fallback runs the exact pre-B2 ops).
//   2. CUDA TRAINING PIN (T4 only; skipped on CPU-only builds): the same
//      training run, CPU reference vs Device::CUDA with step residency
//      ON — per-step loss within rtol 1e-4 and final-weights max abs
//      diff <= 1e-4. NOT bitwise: different kernels, different reduction
//      order (same pin tiers as test_resident_parity).
//   3. STALENESS PROBE (T4 only): the FD-suite pattern — forward in a
//      window, mutate a host weight BETWEEN windows, forward in a fresh
//      window. The second forward must see the mutation (match a
//      no-residency reference), i.e. the epoch scope forbids serving the
//      first window's cached weights. This is the class B1 refused to
//      create and B2.0 must not reintroduce.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../tools/parity_model.hpp"
#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/nn.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

namespace {
const size_t V = 31, D = 32, H = 4, T = 16;

std::vector<int> ids_mod(size_t n, size_t vocab, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<int> v(n);
    for (auto& x : v) x = static_cast<int>(g() % vocab);
    return v;
}

parity::FlexConfig tiny_cfg() {
    parity::FlexConfig fc;
    fc.vocab = V;
    fc.d = D;
    fc.n_heads = H;
    fc.n_ctx = T;
    fc.n_layers = 2;
    fc.d_ff = 4 * D;
    return fc;
}

// First 2-D parameter under the first block's attention (the same probe
// target the FD suites use).
Var probe_param(parity::FlexLM& m) {
    for (const auto& [name, p] : m.named_parameters())
        if (name.find("layers.0.attn") != std::string::npos && p->data.rows() > 1)
            return p;
    throw std::runtime_error("probe param not found");
}

struct RunResult {
    std::vector<float> losses;
    std::vector<float> final_w;
};

// N AdamW steps on a tiny FlexLM, synthetic data, fixed seeds. `windows`
// opens a step window per step (mtstudio's placement: closed before
// opt.step() mutates host params).
RunResult train(bool residency, bool windows, int steps) {
    parity::FlexLM m(tiny_cfg(), 7);
    nn::AdamW opt(m.parameters(), 1e-3f);
    auto ids = ids_mod(T, V, 3);
    std::vector<int> y(ids.rbegin(), ids.rend());

    device::set_step_residency(residency);
    RunResult r;
    for (int s = 0; s < steps; ++s) {
        if (windows) device::step_begin();
        opt.zero_grad();
        Var loss = ops::cross_entropy(m.forward(ids), y);
        backward(loss);
        if (windows) device::step_end();
        opt.step();
        r.losses.push_back(loss->data(0, 0));
    }
    device::set_step_residency(false);
    Var w = probe_param(m);
    for (size_t i = 0; i < w->data.rows(); ++i)
        for (size_t j = 0; j < w->data.cols(); ++j)
            r.final_w.push_back(w->data(i, j));
    return r;
}

// forward (window) -> host poke BETWEEN windows -> forward (new window).
// Returns the two losses. Any stale cached weight makes the second loss
// wrong; the caller compares against a no-residency reference.
std::pair<float, float> poke_probe(bool residency, bool windows) {
    parity::FlexLM m(tiny_cfg(), 7);
    auto ids = ids_mod(T, V, 3);
    std::vector<int> y(ids.rbegin(), ids.rend());
    device::set_step_residency(residency);

    if (windows) device::step_begin();
    const float l0 = ops::cross_entropy(m.forward(ids), y)->data(0, 0);
    if (windows) device::step_end();

    probe_param(m)->data(0, 0) += 0.25f;  // between windows, host-side

    if (windows) device::step_begin();
    const float l1 = ops::cross_entropy(m.forward(ids), y)->data(0, 0);
    if (windows) device::step_end();

    device::set_step_residency(false);
    return {l0, l1};
}
}  // namespace

int main() {
    const int STEPS = 8;

    // 1. no-op leg: switch off vs on (CPU device) — bitwise.
    {
        auto off = train(false, false, STEPS);
        auto on = train(true, true, STEPS);
        CHECK(off.losses.size() == on.losses.size());
        for (size_t i = 0; i < off.losses.size(); ++i)
            CHECK(off.losses[i] == on.losses[i]);
        CHECK(!off.final_w.empty() && off.final_w == on.final_w);
        std::printf("1. no-op leg: residency+windows on CPU device is bitwise inert\n");
    }

    if (!device::cuda_compiled()) {
        std::printf("2-3. CUDA legs: SKIPPED (CPU-only build) — run on T4\n");
        std::printf("all step-residency checks passed\n");
        return 0;
    }

    // 2. CUDA training pin vs CPU reference.
    {
        auto ref = train(false, false, STEPS);
        device::set(device::Device::CUDA);
        auto gpu = train(true, true, STEPS);
        device::set(device::Device::CPU);
        for (size_t i = 0; i < ref.losses.size(); ++i) {
            const float a = ref.losses[i], b = gpu.losses[i];
            CHECK(std::fabs(a - b) <= 1e-4f + 1e-4f * std::fabs(a));
        }
        CHECK(ref.final_w.size() == gpu.final_w.size());
        float md = 0;
        for (size_t i = 0; i < ref.final_w.size(); ++i)
            md = std::max(md, std::fabs(ref.final_w[i] - gpu.final_w[i]));
        CHECK(md <= 1e-4f);
        std::printf("2. CUDA pin: step-resident training matches CPU (max w diff %.2e)\n", md);
    }

    // 3. staleness probe: the second forward must see the host poke.
    {
        auto ref = poke_probe(false, false);  // CPU truth
        device::set(device::Device::CUDA);
        auto gpu = poke_probe(true, true);
        device::set(device::Device::CPU);
        CHECK(std::fabs(ref.first - gpu.first) <=
              1e-4f + 1e-4f * std::fabs(ref.first));
        CHECK(std::fabs(ref.second - gpu.second) <=
              1e-4f + 1e-4f * std::fabs(ref.second));
        CHECK(ref.first != ref.second);  // the poke must matter at all
        std::printf("3. staleness probe: between-window host poke visible on device\n");
    }

    std::printf("all step-residency checks passed\n");
    return 0;
}
