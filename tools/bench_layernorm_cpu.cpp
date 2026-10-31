// Fixed CPU layernorm comparison harness: dumps native float bits or times
// complete forward / forward+backward calls. No alternative numerical oracle.
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include "microtorch/ops.hpp"
#include "parity_model.hpp"
using namespace microtorch;
using json = nlohmann::json;
namespace {
json bits(const Matrix& m) {
    json a = json::array();
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j) {
            uint32_t u;
            float f = m(i, j);
            std::memcpy(&u, &f, 4);
            a.push_back(u);
        }
    return a;
}
Matrix input(size_t r, size_t c, int pattern) {
    Matrix x(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) {
            float v = float(int((i * c + j) * 131 % 997) - 498) / 31.f;
            if (pattern == 1) v = 1.25f;
            if (pattern == 2) v = 32.f + float(int(j % 5) - 2) * 1e-5f;
            if (pattern == 3) v = j % 2 ? -0.f : 0.f;
            if (pattern == 4) v = j % 2 ? 1e9f : -1e-9f;
            if (pattern == 5) v = j == 0 ? std::numeric_limits<float>::quiet_NaN() : float(j);
            x(i, j) = v;
        }
    return x;
}
struct Inputs {
    Var x, g, b;
};
Inputs make(size_t r, size_t c, int pattern, int mask, int alias = 0) {
    auto x = make_var(input(r, c, pattern), mask & 1);
    Matrix gm(1, c), bm(1, c);
    for (size_t j = 0; j < c; ++j) {
        gm(0, j) = 1.f + .1f * float(j % 5);
        bm(0, j) = .01f * float(j % 7);
    }
    auto g = make_var(gm, mask & 2), b = make_var(bm, mask & 4);
    if (alias == 1) {
        g = x;
        x->requires_grad = bool(mask & 3);
    }
    if (alias == 2) {
        b = x;
        x->requires_grad = bool(mask & 5);
    }
    if (alias == 3) {
        b = g;
        g->requires_grad = bool(mask & 6);
    }
    if (alias == 4) {
        g = b = x;
        x->requires_grad = bool(mask);
    }
    return {x, g, b};
}
Var evaluate(Inputs& a, float eps, bool grad) {
    auto y = ops::layernorm(a.x, a.g, a.b, eps);
    if (grad) {
        Matrix dy(y->data.rows(), y->data.cols());
        for (size_t i = 0; i < dy.rows(); ++i)
            for (size_t j = 0; j < dy.cols(); ++j) dy(i, j) = 1.f + float(j % 11) / 13.f;
        backward(ops::mean(ops::mul(y, make_var(dy))));
    }
    return y;
}
void dump() {
    json cases = json::array();
    auto one = [&](size_t r, size_t c, int p, int mask, int alias, float eps) {
        auto a = make(r, c, p, mask, alias);
        auto y = evaluate(a, eps, mask != 0);
        cases.push_back({{"shape", {r, c}},
                         {"pattern", p},
                         {"mask", mask},
                         {"alias", alias},
                         {"eps", eps},
                         {"out", bits(y->data)},
                         {"dx", bits(a.x->grad)},
                         {"dg", bits(a.g->grad)},
                         {"db", bits(a.b->grad)}});
    };
    for (auto shape : std::vector<std::pair<size_t, size_t>>{
             {1, 1}, {1, 7}, {3, 16}, {2, 64}, {1, 320}, {2, 768}})
        for (int p = 0; p < 6; ++p)
            for (int mask = 0; mask < 8; ++mask) one(shape.first, shape.second, p, mask, 0, 1e-5f);
    for (int a = 1; a <= 4; ++a)
        for (int mask = 0; mask < 8; ++mask) one(1, 17, 0, mask, a, 1e-3f);
    one(16, 768, 0, 7, 0, 1e-5f);
    json models = json::array();
    for (const char* activation : {"gelu", "gelu_erf", "relu"}) {
        parity::FlexConfig c;
        c.vocab = 17;
        c.d = 16;
        c.n_heads = 2;
        c.n_layers = 2;
        c.d_ff = 24;
        c.n_ctx = 16;
        c.act = activation;
        parity::FlexLM m(c, 19);
        auto y = m.forward({1, 2, 3, 4, 5, 6, 7, 8});
        backward(ops::mean(y));
        json grads = json::object(), updated = json::object();
        for (auto& kv : m.named_parameters()) {
            grads[kv.first] = bits(kv.second->grad);
            for (size_t i = 0; i < kv.second->data.rows(); ++i)
                for (size_t j = 0; j < kv.second->data.cols(); ++j)
                    kv.second->data(i, j) -= .001f * kv.second->grad(i, j);
            updated[kv.first] = bits(kv.second->data);
        }
        models.push_back({{"activation", activation},
                          {"logits", bits(y->data)},
                          {"grads", grads},
                          {"sgd_step", updated}});
    }
    json refused = json::array();
    for (auto bad : std::vector<std::pair<size_t, size_t>>{{1, 6}, {2, 7}, {0, 0}}) {
        auto a = make(1, 7, 0, 7);
        auto y = ops::layernorm(a.x, a.g, a.b);
        y->grad = bad.first ? Matrix(bad.first, bad.second) : Matrix();
        bool fail = false;
        try {
            y->backward_fn();
        } catch (const std::exception&) {
            fail = true;
        }
        refused.push_back(fail);
    }
    std::cout
        << json({{"cases", cases}, {"models", models}, {"malformed_dy_refused", refused}}).dump()
        << '\n';
}
void bench() {
    json rows = json::array();
    constexpr int warm = 32, iterations = 1024;
    for (auto shape : std::vector<std::pair<size_t, size_t>>{
             {1, 16}, {16, 16}, {1, 128}, {16, 128}, {1, 320}, {16, 320}, {1, 768}, {16, 768}})
        for (bool grad : {false, true}) {
            auto a = make(shape.first, shape.second, 0, grad ? 7 : 0);
            float checksum = 0;
            auto call = [&]() {
                zero_grad({a.x, a.g, a.b});
                auto y = evaluate(a, 1e-5f, grad);
                checksum += y->data(0, 0);
            };
            for (int n = 0; n < warm; ++n) call();
            auto start = std::chrono::steady_clock::now();
            for (int n = 0; n < iterations; ++n) call();
            double ns =
                std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
                    .count() /
                iterations;
            rows.push_back({{"rows", shape.first},
                            {"cols", shape.second},
                            {"backward", grad},
                            {"ns_per_call", ns},
                            {"iterations", iterations},
                            {"warmup", warm},
                            {"checksum", checksum}});
        }
    std::cout << rows.dump() << '\n';
}
}  // namespace
int main(int argc, char** argv) {
    device::set(device::Device::CPU);
    if (argc == 2 && std::string(argv[1]) == "dump")
        dump();
    else if (argc == 2 && std::string(argv[1]) == "bench")
        bench();
    else
        return 2;
}
