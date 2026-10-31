#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
using J = nlohmann::json;
Matrix mat(size_t rows, size_t cols, const std::vector<float>& v) {
    CHECK(v.size() == rows * cols);
    Matrix m(rows, cols);
    for (size_t i = 0; i < v.size(); ++i) m(i / cols, i % cols) = v[i];
    return m;
}
J dump(const Matrix& m) {
    J v = J::array();
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j) {
            CHECK(std::isfinite(m(i, j)));
            v.push_back(m(i, j));
        }
    return J{{"shape", {m.rows(), m.cols()}}, {"values", v}};
}
void poison(Matrix& m) {
    CHECK(device::host_stale(m));
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j) m(i, j) = std::numeric_limits<float>::quiet_NaN();
    CHECK(device::host_stale(m));
}
J run(const std::string& op, bool gpu) {
    device::set(gpu ? device::Device::CUDA : device::Device::CPU);
    device::set_device_ops(gpu);
    device::set_step_residency(gpu);
    device::set_defer_downloads(gpu);
    if (gpu) {
        device::step_begin();
        CHECK(device::deferring());
    }
    const bool broad = op == "broadcast_cols",
               inplace = op == "relu_" || op == "sigmoid_" || op == "scale_";
    auto a0 = make_var(
        broad ? mat(2, 1, {.2f, .8f}) : mat(2, 3, {.2f, -.6f, 1.1f, .8f, -1.3f, .4f}), true);
    Var a = a0, b, c, w;
    std::map<std::string, Var> leaves{{"a", a0}};
    J inputs{{"a", dump(a0->data)}};
    int stale_inputs = 0;
    auto fresh = [&](const Var& x) {
        if (!gpu) return x;
        auto z = ops::slice_cols(x, 0, x->data.cols());
        poison(z->data);
        ++stale_inputs;
        return z;
    };
    a = fresh(a0);
    if (op == "div" || op == "where") {
        auto b0 = make_var(mat(2, 3, {.7f, 1.2f, .9f, 1.4f, .8f, 1.1f}), true);
        leaves["b"] = b0;
        inputs["b"] = dump(b0->data);
        b = fresh(b0);
    }
    if (op == "where") {
        auto c0 = make_var(mat(2, 3, {-1.f, 2.f, -3.f, 4.f, -5.f, 6.f}));
        inputs["c"] = dump(c0->data);
        c = fresh(c0);
    }
    Var y;
    if (inplace) {
        w = make_var(mat(3, 3, {1, 0, 0, 0, 1, 0, 0, 0, 1}), true);
        inputs["w"] = dump(w->data);
        leaves["w"] = w;
        y = ops::matmul(a, w);
        CHECK(y->dev == nullptr);
        if (gpu) {
            poison(y->data);
            ++stale_inputs;
        }
        if (op == "relu_")
            y = ops::relu_(y);
        else if (op == "sigmoid_")
            y = ops::sigmoid_(y);
        else
            y = ops::scale_(y, 1.7f);
    } else if (op == "exp")
        y = ops::exp(a);
    else if (op == "div")
        y = ops::div(a, b);
    else if (op == "where")
        y = ops::where(c, a, b);
    else if (op == "rowsum")
        y = ops::rowsum(a);
    else if (op == "rowmax")
        y = ops::rowmax(a);
    else if (op == "broadcast_cols")
        y = ops::broadcast_cols(a, 3);
    else
        throw std::runtime_error("unknown case");
    Matrix weights =
        y->data.cols() == 1 ? mat(2, 1, {-.4f, .7f}) : mat(2, 3, {-.4f, .7f, 1.2f, -.9f, .3f, .8f});
    inputs["upstream_weights"] = dump(weights);
    bool stale_gradient = false;
    auto original = std::move(y->backward_fn);
    Variable* self = y.get();
    y->backward_fn = [&, self, original]() {
        if (gpu) {
            poison(self->grad);
            stale_gradient = true;
        }
        original();
    };
    auto loss = ops::mean(ops::mul(y, make_var(weights)));
    backward(loss);
    device::materialize(y->data);
    J gradients = J::object();
    for (auto& item : leaves) {
        device::materialize(item.second->grad);
        gradients[item.first] = dump(item.second->grad);
    }
    J result{{"op", op},
             {"actual_cuda", gpu},
             {"stale_inputs_poisoned", stale_inputs},
             {"stale_gradient_poisoned", stale_gradient},
             {"inputs", inputs},
             {"output", dump(y->data)},
             {"gradients", gradients}};
    if (gpu) {
        device::step_end();
        device::set_defer_downloads(false);
        device::set_step_residency(false);
        device::set_device_ops(false);
        device::evict_all();
        device::set(device::Device::CPU);
    }
    return result;
}
int main(int argc, char** argv) {
    try {
        bool gpu = false;
        std::string output;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--require-cuda")
                gpu = true;
            else if (arg == "--out" && i + 1 < argc)
                output = argv[++i];
            else
                throw std::runtime_error("unknown argument");
        }
        if (gpu) {
            CHECK(device::cuda_compiled());
#ifndef MICROTORCH_DEVCHECK
            throw std::runtime_error("DEVCHECK required");
#endif
        }
        J cases = J::array();
        for (const char* op : {"exp", "div", "where", "rowsum", "rowmax", "broadcast_cols", "relu_",
                               "sigmoid_", "scale_"})
            cases.push_back(run(op, gpu));
        const std::string raw = J{{"pass", true}, {"actual_cuda", gpu}, {"cases", cases}}.dump();
        if (output.empty())
            std::puts(raw.c_str());
        else {
            std::ifstream existing(output);
            CHECK(!existing.good());
            std::ofstream f(output);
            CHECK(f.good());
            f << raw << "\n";
            f.flush();
            CHECK(f.good());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
