// Public policy_gradient_kl reference-Matrix boundary; strict CUDA opt-in.
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "microtorch/autograd.hpp"
#include "microtorch/device.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
using J = nlohmann::json;
#ifndef POLICY_REFERENCE_VARIANT
#error POLICY_REFERENCE_VARIANT required
#endif
void need(bool b, const char* m) {
    if (!b) throw std::runtime_error(m);
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
int main(int argc, char** argv) {
    bool gpu = false;
    std::string which, out;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--require-cuda")
            gpu = true;
        else if (a == "--case" && i + 1 < argc)
            which = argv[++i];
        else if (a == "--out" && i + 1 < argc)
            out = argv[++i];
        else
            return 2;
    }
    if ((which != "mixed" && which != "policy_only" && which != "kl_only") || out.empty() ||
        std::ifstream(out).good())
        return 2;
    J row = {{"variant", POLICY_REFERENCE_VARIANT},
             {"case", which},
             {"actual_cuda", gpu},
             {"pass", false}};
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
        Matrix lm(2, 3);
        float lv[] = {-.75f, .125f, .5f, .25f, -.375f, 1.f};
        for (size_t i = 0; i < 6; ++i) lm(i / 3, i % 3) = lv[i];
        auto logits = make_var(std::move(lm), true);
        Matrix truth(2, 3), identity(3, 3);
        float probabilities[] = {.2f, .3f, .5f, .6f, .3f, .1f};
        for (size_t i = 0; i < 6; ++i) truth(i / 3, i % 3) = std::log(probabilities[i]);
        for (size_t i = 0; i < 3; ++i) identity(i, i) = 1.f;
        Matrix reference = device::gemm(truth, nullptr, device::Trans::N, identity, nullptr,
                                        device::Trans::N, device::Precision::FP32);
        row["reference_stale_before"] = device::host_stale(reference);
        row["poisoned"] = gpu;
        if (gpu) {
            need(device::host_stale(reference), "reference must be deferred");
            for (size_t i = 0; i < 6; ++i) reference.get_data()[i] = 9.f;
        }
        std::vector<float> w = {.75f, -.25f}, k = {.5f, 1.f};
        if (which == "policy_only") k = {0.f, 0.f};
        if (which == "kl_only") w = {0.f, 0.f};
        std::vector<int> targets = {2, 0};
        row["w"] = w;
        row["k"] = k;
        row["targets"] = targets;
        row["upstream"] = 1.5f;
        auto loss = ops::policy_gradient_kl(logits, targets, w, k, reference);
        row["reference_stale_after_op"] = device::host_stale(reference);
        // Restore/materialize caller storage only AFTER the closure's by-value copy.
        row["reference"] = dump(reference);
        row["logits"] = dump(logits->data);
        row["loss"] = dump(loss->data);
        auto weighted = ops::scale(loss, 1.5f);
        backward(weighted);
        row["gradient"] = dump(logits->grad);
        device::gpu_profile_take();
        J counts = J::object();
        for (const auto& x : device::gpu_profile_counts()) counts[x.first] = x.second;
        row["profile"] = counts;
        row["pass"] = true;
    } catch (const std::exception& e) {
        row["error"] = e.what();
    }
    device::step_end();
    std::ofstream f(out);
    if (!f.good()) return 2;
    f << row.dump() << "\n";
    f.flush();
    return f.good() ? 0 : 2;
}
