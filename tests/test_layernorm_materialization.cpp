// Actual fallback coherence, not a model or throughput test.
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
constexpr size_t R = 2, C = 4;
constexpr float eps = 1e-5f;
const std::array<double, 8> xv{.2, -.6, 1.2, 2., -.7, .8, 1.7, -1.2};
const std::array<double, 4> gv{.7, -1.1, .4, 1.8}, bv{-.2, .1, .3, -.7};
const std::array<double, 8> upstream{.2, -.4, .7, 1.3, -.9, .6, .1, -.3};
Matrix make(size_t rows, const double* values) {
    Matrix out(rows, C);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < C; ++j) out(i, j) = static_cast<float>(values[i * C + j]);
    return out;
}
void cell(const std::string& name, bool gpu) {
    auto x0 = make_var(make(R, xv.data()), true), g0 = make_var(make(1, gv.data()), true),
         b0 = make_var(make(1, bv.data()), true);
    Var x = x0, g = g0, b = b0;
    if (gpu) {
        device::set(device::Device::CUDA);
        device::set_device_ops(true);
        device::set_step_residency(true);
        device::set_defer_downloads(true);
        device::step_begin();
        CHECK(device::deferring());
    }
    Var produced;
    if (name == "forward_x")
        x = produced = ops::slice_cols(x0, 0, C);
    else if (name == "forward_beta")
        b = produced = ops::slice_cols(b0, 0, C);
    else
        g = produced = ops::slice_cols(g0, 0, C);
    if (gpu) {
        CHECK(device::host_stale(produced->data));
        for (size_t i = 0; i < produced->data.rows(); ++i)
            for (size_t j = 0; j < C; ++j) produced->data(i, j) = -7777.f;
        CHECK(device::host_stale(produced->data));
    }
    // For backward_gamma the GPU forward must consume authoritative gamma first.
    if (gpu && name != "backward_gamma") device::set_device_ops(false);
    auto y = ops::layernorm(x, g, b, eps);
    if (gpu) {
        if (name == "backward_gamma") {
            CHECK(device::host_stale(g->data));
            device::set_device_ops(false);
        }
        CHECK(!device::device_ops_enabled());
        device::materialize(y->data);
        std::printf(
            "{\"case\":\"%s\",\"actual_cuda\":true,\"stale_operand_proved\":true,\"host_poison\":"
            "true,\"ops_disabled_before_fallback\":true}\n",
            name.c_str());
        std::fflush(stdout);
    }
    std::array<double, 8> expected{}, dx{};
    std::array<double, 4> dg{}, db{};
    for (size_t i = 0; i < R; ++i) {
        double mean = 0, var = 0;
        for (size_t j = 0; j < C; ++j) mean += static_cast<float>(xv[i * C + j]);
        mean /= C;
        for (size_t j = 0; j < C; ++j) {
            double d = static_cast<float>(xv[i * C + j]) - mean;
            var += d * d;
        }
        var /= C;
        double rs = 1 / std::sqrt(var + eps), m1 = 0, m2 = 0;
        for (size_t j = 0; j < C; ++j) {
            double xh = (static_cast<float>(xv[i * C + j]) - mean) * rs;
            double dy = static_cast<float>(upstream[i * C + j]) / (R * C);
            expected[i * C + j] = static_cast<float>(gv[j]) * xh + static_cast<float>(bv[j]);
            dg[j] += dy * xh;
            db[j] += dy;
            m1 += dy * static_cast<float>(gv[j]);
            m2 += dy * static_cast<float>(gv[j]) * xh;
        }
        m1 /= C;
        m2 /= C;
        for (size_t j = 0; j < C; ++j) {
            double xh = (static_cast<float>(xv[i * C + j]) - mean) * rs,
                   dy = static_cast<float>(upstream[i * C + j]) / (R * C);
            dx[i * C + j] = rs * (dy * static_cast<float>(gv[j]) - m1 - xh * m2);
        }
    }
    for (size_t i = 0; i < R; ++i)
        for (size_t j = 0; j < C; ++j)
            CHECK(std::fabs(y->data(i, j) - expected[i * C + j]) /
                      (1 + std::fabs(expected[i * C + j])) <
                  2e-5);
    backward(ops::mean(ops::mul(y, make_var(make(R, upstream.data())))));
    device::materialize(x0->grad);
    device::materialize(g0->grad);
    device::materialize(b0->grad);
    for (size_t i = 0; i < R; ++i)
        for (size_t j = 0; j < C; ++j)
            CHECK(std::fabs(x0->grad(i, j) - dx[i * C + j]) / (1 + std::fabs(dx[i * C + j])) <
                  2e-5);
    for (size_t j = 0; j < C; ++j) {
        CHECK(std::fabs(g0->grad(0, j) - dg[j]) / (1 + std::fabs(dg[j])) < 2e-5);
        CHECK(std::fabs(b0->grad(0, j) - db[j]) / (1 + std::fabs(db[j])) < 2e-5);
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
        "{\"pass\":true,\"case\":\"%s\",\"cuda_integration\":%s,\"forward_backward_reference\":"
        "true}\n",
        name.c_str(), gpu ? "true" : "false");
}
int main(int argc, char** argv) {
    bool gpu = false;
    std::string which = "all";
    for (int i = 1; i < argc; ++i) {
        std::string v = argv[i];
        if (v == "--require-cuda")
            gpu = true;
        else if (v == "--case" && i + 1 < argc)
            which = argv[++i];
        else
            return 2;
    }
#ifdef MICROTORCH_DEVCHECK
    constexpr bool checks = true;
#else
    constexpr bool checks = false;
#endif
    if (gpu && (!device::cuda_compiled() || !checks)) {
        std::fprintf(stderr, "requires actual CUDA plus DEVCHECK build\n");
        return 2;
    }
    if (!gpu) {
        device::set(device::Device::CPU);
        device::set_device_ops(false);
        device::set_defer_downloads(false);
        device::set_step_residency(false);
    }
    const std::vector<std::string> cases{"forward_x", "forward_gamma", "forward_beta",
                                         "backward_gamma"};
    bool found = false;
    for (const auto& name : cases)
        if (which == "all" || which == name) {
            found = true;
            cell(name, gpu);
        }
    if (!found) return 2;
    std::printf("{\"pass\":true,\"cuda_integration\":%s,\"devcheck_enabled\":%s,\"cases\":%d}\n",
                gpu ? "true" : "false", checks ? "true" : "false", which == "all" ? 4 : 1);
}
