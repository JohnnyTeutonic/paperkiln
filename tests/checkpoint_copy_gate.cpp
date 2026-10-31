// Source-bound checkpoint host-copy gate; real CUDA is strictly opt-in.
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "microtorch/autograd.hpp"
#include "microtorch/device.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
using J = nlohmann::json;
#ifndef CHECKPOINT_VARIANT
#error CHECKPOINT_VARIANT required
#endif
void need(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
J dump(const Matrix& x) {
    device::materialize(x);
    J v = J::array();
    for (size_t i = 0; i < x.rows(); ++i)
        for (size_t j = 0; j < x.cols(); ++j) {
            need(std::isfinite(x(i, j)), "nonfinite");
            v.push_back(x(i, j));
        }
    return {{"shape", {x.rows(), x.cols()}}, {"values", v}};
}
Var fixture(size_t r, size_t c, int seed, bool grad = true) {
    Matrix x(r, c);
    for (size_t i = 0; i < r * c; ++i) x(i / c, i % c) = float(int((i * 7 + seed) % 19) - 9) / 8.f;
    return make_var(std::move(x), grad);
}
int main(int argc, char** argv) {
    bool gpu = false;
    std::string which, output;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--require-cuda")
            gpu = true;
        else if (a == "--case" && i + 1 < argc)
            which = argv[++i];
        else if (a == "--out" && i + 1 < argc)
            output = argv[++i];
        else
            return 2;
    }
    if ((which != "output" && which != "input") || output.empty() || std::ifstream(output).good())
        return 2;
    J report = {{"variant", CHECKPOINT_VARIANT},
                {"case", which},
                {"actual_cuda", gpu},
                {"pass", false},
                {"observations", J::array()}};
    try {
        if (gpu) {
            need(device::cuda_compiled(), "CUDA required");
            need(device::gpu_profile_on(), "profiler required");
#ifndef MICROTORCH_DEVCHECK
            throw std::runtime_error("DEVCHECK required");
#endif
        }
        device::set(gpu ? device::Device::CUDA : device::Device::CPU);
        device::set_device_ops(gpu);
        device::set_step_residency(gpu);
        device::set_defer_downloads(gpu);
        device::set_matmul_precision(device::Precision::FP32);
        device::step_begin();
        auto w = fixture(4, 2, 3);
        Var a, b, x;
        if (which == "input") {
            a = fixture(2, 3, 1);
            b = fixture(3, 4, 2);
            x = ops::matmul(a, b);
        } else {
            x = fixture(2, 4, 4);
        }
        report["input_stale_before_checkpoint"] = device::host_stale(x->data);
        auto y = checkpoint(
            [&](const Var& v) {
                auto z = ops::matmul(v, w);
                bool stale = device::host_stale(z->data);
                if (which == "input") device::materialize(z->data);
                report["observations"].push_back(
                    {{"grad_enabled", grad_enabled()},
                     {"input_stale", device::host_stale(v->data)},
                     {"output_stale_before_local_materialize", stale},
                     {"output_stale_on_return", device::host_stale(z->data)}});
                return z;
            },
            x);
        report["input_stale_before_backward"] = device::host_stale(x->data);
        report["logits"] = dump(y->data);
        Matrix u(2, 2);
        u(0, 0) = .5f;
        u(0, 1) = -.25f;
        u(1, 0) = .75f;
        u(1, 1) = 1.25f;
        auto weights = make_var(std::move(u), false);
        auto loss = ops::mean(ops::mul(y, weights));
        backward(loss);
        report["loss"] = dump(loss->data);
        J parameters = J::object(), gradients = J::object();
        auto add = [&](const char* n, const Var& v) {
            parameters[n] = dump(v->data);
            gradients[n] = dump(v->grad);
        };
        if (which == "input") {
            add("a", a);
            add("b", b);
        } else
            add("x", x);
        add("w", w);
        report["parameters"] = parameters;
        report["gradients"] = gradients;
        device::gpu_profile_take();
        J counts = J::object();
        for (const auto& entry : device::gpu_profile_counts()) counts[entry.first] = entry.second;
        report["profile"] = counts;
        report["pass"] = true;
    } catch (const std::exception& e) {
        report["error"] = e.what();
    }
    device::step_end();
    std::ofstream f(output);
    if (!f.good()) return 2;
    f << report.dump() << "\n";
    f.flush();
    return f.good() ? 0 : 2;
}
