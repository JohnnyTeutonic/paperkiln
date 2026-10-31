// A computed bias can be device-fresh while its host storage is stale.
#include <cmath>
#include <cstdio>
#include <string>
#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
int main(int argc, char** argv) {
    const bool gpu = argc == 2 && std::string(argv[1]) == "--require-cuda";
    if (argc > 2 || (argc == 2 && !gpu)) return 2;
#ifdef MICROTORCH_DEVCHECK
    constexpr bool checks = true;
#else
    constexpr bool checks = false;
#endif
    if (gpu && (!device::cuda_compiled() || !checks)) {
        std::fprintf(stderr, "requires actual CUDA and MICROTORCH_DEVCHECK=ON\n");
        return 2;
    }
    Matrix x(2, 3), b(1, 3);
    const double bias_values[3] = {-1.5, 0.75, 2.0};
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j) x(i, j) = float(0.25 + i + 0.125 * j);
    for (size_t j = 0; j < 3; ++j) b(0, j) = float(bias_values[j]);
    Var vx = make_var(std::move(x), true), vb = make_var(std::move(b), true);
    if (gpu) {
        device::set(device::Device::CUDA);
        device::set_device_ops(true);
        device::set_step_residency(true);
        device::set_defer_downloads(true);
        device::step_begin();
        CHECK(device::deferring());
    }
    Var computed_bias = ops::gelu(vb);
    if (gpu) {
        CHECK(device::host_stale(computed_bias->data));
        // Poison only the stale host backing storage. The real CUDA-produced
        // authoritative value remains cached; a correct materialize replaces it.
        // This makes the pre-fix wrong-host-read failure deterministic instead
        // of depending on arbitrary contents of an uninitialized allocation.
        for (size_t j = 0; j < 3; ++j) computed_bias->data(0, j) = -7777.0f;
        CHECK(device::host_stale(computed_bias->data));
        std::puts("{\"actual_cuda\":true,\"stale_computed_bias\":true,\"host_poison\":true}");
        std::fflush(stdout);
    }
    Var y = ops::add_bias(vx, computed_bias);
    const double k = std::sqrt(2.0 / std::acos(-1.0));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j) {
            const double v = bias_values[j], u = k * (v + 0.044715 * v * v * v);
            const double expected = 0.25 + i + 0.125 * j + 0.5 * v * (1 + std::tanh(u));
            CHECK(std::fabs(y->data(i, j) - expected) / (1 + std::fabs(expected)) < 2e-5);
        }
    backward(ops::mean(y));
    device::materialize(vx->grad);
    device::materialize(vb->grad);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j) CHECK(std::fabs(vx->grad(i, j) - 1.0 / 6) < 2e-5);
    for (size_t j = 0; j < 3; ++j) {
        const double v = bias_values[j], u = k * (v + 0.044715 * v * v * v), t = std::tanh(u);
        const double derivative =
            0.5 * (1 + t) + 0.5 * v * (1 - t * t) * k * (1 + 3 * 0.044715 * v * v);
        CHECK(std::fabs(vb->grad(0, j) - derivative / 3) / (1 + std::fabs(derivative / 3)) < 2e-5);
    }
    if (gpu) {
        device::step_end();
        device::set_defer_downloads(false);
        device::set_step_residency(false);
        device::set_device_ops(false);
        device::evict_all();
        device::set(device::Device::CPU);
    }
    std::printf(
        "{\"pass\":true,\"cuda_integration\":%s,\"devcheck_enabled\":%s,\"forward_and_backward_"
        "reference\":true}\n",
        gpu ? "true" : "false", checks ? "true" : "false");
}
