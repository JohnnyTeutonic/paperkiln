// Computed device table -> embedding fallback coherence; ordinary leaf tables are controls.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
using json = nlohmann::json;
const std::vector<std::string> names = {
    "deferred_fallback", "deferred_device", "write_through_fallback", "leaf_fallback", "empty_ids",
    "negative_id",       "large_id"};
void require(bool x, const char* m) {
    if (!x) throw std::runtime_error(m);
}
std::vector<float> flat(const Matrix& m) {
    std::vector<float> v;
    for (size_t r = 0; r < m.rows(); ++r)
        for (size_t c = 0; c < m.cols(); ++c) {
            require(std::isfinite(m(r, c)), "nonfinite raw tensor");
            v.push_back(m(r, c));
        }
    return v;
}
bool run(const std::string& name, bool gpu) {
    const bool computed = name != "leaf_fallback", deferred = name != "write_through_fallback";
    const bool invalid = name == "negative_id" || name == "large_id", empty = name == "empty_ids";
    Matrix input(4, 3);
    for (size_t r = 0; r < 4; ++r)
        for (size_t c = 0; c < 3; ++c) input(r, c) = float(int(r * 3 + c) - 6) / 8.f;
    auto x = make_var(input, true);
    Var table = x;
    device::set(device::Device::CPU);
    device::set_device_ops(false);
    if (gpu) {
        device::set(device::Device::CUDA);
        device::set_device_ops(true);
        device::set_step_residency(true);
        device::set_defer_downloads(deferred);
        device::step_begin();
    }
    if (computed) table = ops::slice_cols(x, 0, 3);
    const bool stale = gpu && computed && deferred;
    if (stale) {
        require(device::host_stale(table->data), "computed table not device-fresh");
        for (size_t r = 0; r < 4; ++r)
            for (size_t c = 0; c < 3; ++c) table->data(r, c) = 0;
        require(device::host_stale(table->data), "host poison invalidated device cache");
    }
    if (gpu) device::set_device_ops(name == "deferred_device");
    std::vector<int> ids = empty     ? std::vector<int>{}
                           : invalid ? std::vector<int>{name == "negative_id" ? -1 : 4}
                                     : std::vector<int>{2, 0, 2};
    json row = {{"case", name},
                {"actual_cuda", gpu},
                {"computed_table", computed},
                {"deferred", deferred},
                {"stale_operand_proved", stale},
                {"host_poison", stale},
                {"ops_enabled", gpu && name == "deferred_device"},
                {"input", flat(input)},
                {"input_shape", {4, 3}},
                {"ids", ids}};
    bool ok = true;
    if (invalid || empty) {
        bool rejected = false;
        try {
            ops::embedding(table, ids);
        } catch (const std::runtime_error& e) {
            rejected = std::string(e.what()) ==
                       (empty ? "Matrix dimensions cannot be zero" : "embedding: id out of range");
        }
        require(rejected, "bounds were not rejected by embedding");
        row[empty ? "empty_rejected" : "bounds_rejected"] = true;
    } else {
        auto y = ops::embedding(table, ids);
        row["output_shape"] = {y->data.rows(), y->data.cols()};
        Matrix weights(3, 3);
        for (size_t r = 0; r < 3; ++r)
            for (size_t c = 0; c < 3; ++c) weights(r, c) = float(r * 3 + c + 1) / 16.f;
        auto loss = ops::mean(ops::mul(y, make_var(weights)));
        backward(loss);
        device::materialize(y->data);
        device::materialize(x->grad);
        row["upstream"] = flat(weights);
        row["output"] = flat(y->data);
        row["input_gradient"] = flat(x->grad);
        double worst = 0;
        std::vector<double> expected(9), gradient(12, 0);
        for (size_t r = 0; r < 3; ++r)
            for (size_t c = 0; c < 3; ++c) {
                expected[r * 3 + c] = input(ids[r], c);
                gradient[size_t(ids[r]) * 3 + c] += double(weights(r, c)) / 9.;
            }
        for (size_t i = 0; i < 9; ++i)
            worst = std::max(worst, std::abs(double(row["output"][i].get<float>()) - expected[i]));
        for (size_t i = 0; i < 12; ++i)
            worst = std::max(worst,
                             std::abs(double(row["input_gradient"][i].get<float>()) - gradient[i]));
        row["max_abs_error"] = worst;
        ok = std::isfinite(worst) && worst <= 2e-5;
    }
    row["pass"] = ok;
    std::puts(row.dump().c_str());
    std::fflush(stdout);
    if (gpu) {
        device::step_end();
        device::set_defer_downloads(false);
        device::set_step_residency(false);
        device::set_device_ops(false);
        device::evict_all();
        device::set(device::Device::CPU);
    }
    return ok;
}
int main(int argc, char** argv) {
    bool gpu = false;
    std::string selected = "all";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--require-cuda")
            gpu = true;
        else if (a == "--case" && i + 1 < argc)
            selected = argv[++i];
        else
            return 2;
    }
#ifdef MICROTORCH_DEVCHECK
    constexpr bool checks = true;
#else
    constexpr bool checks = false;
#endif
    if (gpu && (!device::cuda_compiled() || !checks)) {
        std::fputs("requires actual CUDA plus DEVCHECK build\n", stderr);
        return 2;
    }
    try {
        size_t count = 0;
        bool ok = true;
        for (const auto& n : names)
            if (selected == "all" || selected == n) {
                ok = run(n, gpu) && ok;
                ++count;
            }
        require(count > 0, "unknown case");
        std::puts(json({{"summary", true},
                        {"pass", ok},
                        {"cases", count},
                        {"actual_cuda", gpu},
                        {"devcheck_enabled", checks}})
                      .dump()
                      .c_str());
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "embedding gate: %s\n", e.what());
        return 2;
    }
}
