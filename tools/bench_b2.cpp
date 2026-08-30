// B2 adoption-gate benchmark (docs/CUDA_PHASE_B2.md, B2.3 item 5):
// wall-clock per training step, CPU AVX vs the full B2 stack
// (DEVICE_OPS + STEP_RESIDENCY + DEFER_DOWNLOADS), at the Rung C shape.
// This is a MEASUREMENT, not a test: it prints ms/step and never fails.
//
//   bench_b2 <d> <T> <layers> <steps> cpu|b2
//
// Same training-loop shape as test_step_residency (mtstudio's window
// placement: step_end before opt.step mutates host params). Warmup
// steps excluded from timing; synthetic data, fixed seeds, so the two
// engines run identical work.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../tools/parity_model.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/nn.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

int main(int argc, char** argv) {
    if (argc != 6) {
        std::printf("usage: bench_b2 <d> <T> <layers> <steps> cpu|b2\n");
        return 2;
    }
    const size_t d = std::strtoul(argv[1], nullptr, 10);
    const size_t T = std::strtoul(argv[2], nullptr, 10);
    const size_t L = std::strtoul(argv[3], nullptr, 10);
    const int steps = std::atoi(argv[4]);
    // Engine ladder for bisection (2026-08-30: the first bench run's b2
    // arm sat at exactly ln(vocab) — uniform logits, not learning — at
    // shapes the test suite never reaches):
    //   cpu  = host reference
    //   gpu  = CUDA gemm only (Phase A/B1 path, no B2 switches)
    //   ops  = + MICROTORCH_DEVICE_OPS
    //   res  = + step residency
    //   b2   = + deferred downloads (the full stack)
    const std::string eng = argv[5];
    const bool cuda = eng != "cpu";

    if (cuda) {
        device::set(device::Device::CUDA);
        if (device::get() != device::Device::CUDA) {
            std::printf("bench_b2: CUDA unavailable in this build\n");
            return 2;
        }
        if (eng == "ops" || eng == "res" || eng == "b2")
            device::set_device_ops(true);
        if (eng == "res" || eng == "b2") device::set_step_residency(true);
        if (eng == "b2") device::set_defer_downloads(true);
    } else {
        device::set(device::Device::CPU);
    }

    parity::FlexConfig fc;
    fc.vocab = 4096;
    fc.d = d;
    fc.n_heads = d / 32;
    fc.n_ctx = T;
    fc.n_layers = L;
    fc.d_ff = 4 * d;
    parity::FlexLM m(fc, 7);
    nn::AdamW opt(m.parameters(), 1e-3f);

    std::mt19937 g(3);
    std::vector<int> ids(T), y(T);
    for (size_t i = 0; i < T; ++i) {
        ids[i] = static_cast<int>(g() % fc.vocab);
        y[i] = static_cast<int>(g() % fc.vocab);
    }

    auto step = [&]() {
        device::step_begin();
        opt.zero_grad();
        Var loss = ops::cross_entropy(m.forward(ids), y);
        backward(loss);
        device::step_end();
        opt.step();
        return loss->data(0, 0);
    };

    const int warmup = 3;
    float last = 0.0f;
    for (int s = 0; s < warmup; ++s) {
        last = step();
        std::printf("  warm %d loss %.4f\n", s, last);
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        last = step();
        std::printf("  step %d loss %.4f\n", s, last);
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / steps;
    std::printf("BENCH d=%zu T=%zu L=%zu engine=%s steps=%d  "
                "ms_per_step=%.2f  tok_per_s=%.0f  (final loss %.4f)\n",
                d, T, L, eng.c_str(), steps, ms,
                1000.0 * static_cast<double>(T) / ms, last);
    return 0;
}
