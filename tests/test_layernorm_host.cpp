// Shape/refusal and alias regressions for checked contiguous host LayerNorm.
#include <cmath>
#include <string>
#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
namespace {
Var row(size_t c, bool grad) {
    Matrix m(1, c);
    for (size_t j = 0; j < c; ++j) m(0, j) = float(j) - 2.5f;
    return make_var(m, grad);
}
}  // namespace
int main() {
    device::set(device::Device::CPU);
    // The existing Matrix cache constructor refuses empty outputs before any
    // element access. External-buffer constructors allow expressing those shapes.
    float storage[7]{};
    for (auto shape : std::vector<std::pair<size_t, size_t>>{{0, 7}, {1, 0}}) {
        auto x = make_var(Matrix(shape.first, shape.second, storage));
        auto g = make_var(Matrix(1, shape.second, storage));
        auto b = make_var(Matrix(1, shape.second, storage));
        bool refused = false;
        try {
            (void)ops::layernorm(x, g, b);
        } catch (const std::runtime_error& e) {
            refused = std::string(e.what()).find("zero") != std::string::npos;
        }
        CHECK(refused);
    }
    for (auto shape : std::vector<std::pair<size_t, size_t>>{{1, 6}, {2, 7}, {0, 0}}) {
        auto x = row(7, true), g = row(7, true), b = row(7, true);
        auto y = ops::layernorm(x, g, b);
        y->grad = shape.first ? Matrix(shape.first, shape.second) : Matrix();
        bool refused = false;
        try {
            y->backward_fn();
        } catch (const std::exception&) {
            refused = true;
        }
        CHECK(refused);
    }
    // Both nonzero smaller/larger row counts and smaller/larger widths fail
    // before a raw-pointer read. Only malformed gradient metadata changes.
    for (auto shape : std::vector<std::pair<size_t, size_t>>{{2, 7}, {4, 7}, {3, 6}, {3, 8}}) {
        auto rx = make_var(Matrix(3, 7, 1.25f), true), rg = row(7, true), rb = row(7, true);
        auto ry = ops::layernorm(rx, rg, rb);
        ry->grad = Matrix(shape.first, shape.second);
        bool refused = false;
        try {
            ry->backward_fn();
        } catch (const std::exception&) {
            refused = true;
        }
        CHECK(refused);
    }
    for (bool parameter_grad : {false, true}) {
        auto rx = row(7, true), rg = row(7, parameter_grad), rb = row(7, false);
        auto ry = ops::layernorm(rx, rg, rb);
        rg->data = Matrix(1, 6);  // Invalid mutation after the forward graph.
        ry->grad = Matrix(1, 7);
        bool refused = false;
        try {
            ry->backward_fn();
        } catch (const std::exception&) {
            refused = true;
        }
        CHECK(refused);
    }
    auto x = row(7, true), g = row(7, true);
    auto y = ops::layernorm(x, g, g);
    backward(ops::mean(y));
    CHECK(x->grad.rows() == 1 && x->grad.cols() == 7 && g->grad.rows() == 1 && g->grad.cols() == 7);
    for (size_t j = 0; j < 7; ++j) {
        CHECK(std::isfinite(x->grad(0, j)));
        CHECK(std::isfinite(g->grad(0, j)));
    }
    auto all = row(7, true);
    auto aliased = ops::layernorm(all, all, all);
    backward(ops::mean(aliased));
    for (size_t j = 0; j < 7; ++j) CHECK(std::isfinite(all->grad(0, j)));
    auto scalar = row(1, true), gain = make_var(Matrix(1, 1, 2.f), true),
         bias = make_var(Matrix(1, 1, .25f), true);
    auto one = ops::layernorm(scalar, gain, bias);
    CHECK(one->data(0, 0) == .25f);
    backward(ops::mean(one));
    CHECK(scalar->grad(0, 0) == 0.f && gain->grad(0, 0) == 0.f && bias->grad(0, 0) == 1.f);
    bool refused = false;
    try {
        (void)ops::layernorm(row(7, false), make_var(Matrix(2, 7)), row(7, false));
    } catch (const std::exception&) {
        refused = true;
    }
    CHECK(refused);
    std::puts(
        "PASS host LayerNorm empty/refusal, malformed gradients, parameter aliases, scalar "
        "gradient");
}
