// Independent staged double oracle for explicit BF16 attention training op.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
namespace {
using Grid = std::vector<std::vector<double>>;
Grid zeros(size_t r, size_t c) {
    return Grid(r, std::vector<double>(c));
}
double round_operand(double x, bool bf) {
    float f = static_cast<float>(x);
    if (!bf) return f;
    uint32_t u;
    std::memcpy(&u, &f, 4);
    // Independent ties-to-even, finite fixture values only.
    const uint32_t low = u & 65535U, high = u >> 16;
    u = (high + ((low > 32768U) || (low == 32768U && (high & 1U)))) << 16;
    std::memcpy(&f, &u, 4);
    return f;
}
Grid product(const Grid& a, const Grid& b, bool ta, bool tb, bool bf) {
    size_t m = ta ? a[0].size() : a.size(), n = tb ? b.size() : b[0].size(),
           k = ta ? a.size() : a[0].size();
    Grid y = zeros(m, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            double z = 0;
            for (size_t t = 0; t < k; ++t)
                z += round_operand(ta ? a[t][i] : a[i][t], bf) *
                     round_operand(tb ? b[j][t] : b[t][j], bf);
            y[i][j] = static_cast<float>(z);
        }
    return y;
}
struct Ref {
    Grid y, dq, dk, dv;
};
Ref reference(const Grid& q, const Grid& k, const Grid& v, const Grid& g, size_t sl, bool causal,
              bool bf, float scale) {
    const size_t n = q.size();
    Grid p = product(q, k, false, true, bf);
    for (size_t i = 0; i < n; ++i) {
        size_t lo = (i / sl) * sl, hi = causal ? i + 1 : lo + sl;
        double mx = -1e30;
        for (size_t j = lo; j < hi; ++j) mx = std::max(mx, double(float(p[i][j] * scale)));
        double z = 0;
        for (size_t j = lo; j < hi; ++j) {
            p[i][j] = std::exp(double(float(p[i][j] * scale)) - mx);
            z += p[i][j];
        }
        for (size_t j = 0; j < n; ++j) p[i][j] = (j < lo || j >= hi) ? 0 : float(p[i][j] / z);
    }
    Ref r;
    r.y = product(p, v, false, false, bf);
    r.dv = product(p, g, true, false, bf);
    Grid ds = product(g, v, false, true, bf);
    for (size_t i = 0; i < n; ++i) {
        double dot = 0;
        for (size_t j = 0; j < n; ++j) dot += ds[i][j] * p[i][j];
        for (size_t j = 0; j < n; ++j) ds[i][j] = float(scale * p[i][j] * (ds[i][j] - dot));
    }
    r.dq = product(ds, k, false, false, bf);
    r.dk = product(ds, q, true, false, bf);
    return r;
}
Matrix mat(const Grid& a) {
    Matrix m(a.size(), a[0].size());
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[0].size(); ++j) m(i, j) = float(a[i][j]);
    return m;
}
Grid fixture(size_t r, size_t c, int seed) {
    Grid a = zeros(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            a[i][j] = float(0.37 * std::sin(double((i + 1) * (j + 3) + seed)) +
                            0.21 * std::cos(double(i * 3 + j + seed)));
    return a;
}
double check(Matrix& m, const Grid& r, double bound) {
    device::materialize(m);
    double err = 0;
    for (size_t i = 0; i < r.size(); ++i)
        for (size_t j = 0; j < r[0].size(); ++j) {
            CHECK(std::isfinite(m(i, j)));
            err = std::max(err, std::abs(m(i, j) - r[i][j]) / (1 + std::abs(r[i][j])));
        }
    CHECK(err < bound);
    return err;
}
void print_grid(const char* name, const Grid& a) {
    std::printf(",\"%s\":[", name);
    for (size_t i = 0; i < a.size(); ++i) {
        if (i) std::printf(",");
        std::printf("[");
        for (size_t j = 0; j < a[i].size(); ++j) {
            if (j) std::printf(",");
            std::printf("%.9g", a[i][j]);
        }
        std::printf("]");
    }
    std::printf("]");
}
Grid actual_grid(Matrix& m) {
    device::materialize(m);
    Grid a = zeros(m.rows(), m.cols());
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j) a[i][j] = m(i, j);
    return a;
}
void run(size_t sl, size_t batch, size_t d, size_t dv, bool causal, bool bf, unsigned mask,
         bool gpu) {
    const size_t n = sl * batch;
    auto Q = fixture(n, d, 1), K = fixture(n, d, 5), V = fixture(n, dv, 9), G = fixture(n, dv, 13);
    // mean(mul(y,G)) supplies this independently known upstream gradient.
    Grid normalized = G;
    for (auto& row : normalized)
        for (auto& x : row) x = float(x / float(n * dv));
    Ref ref = reference(Q, K, V, normalized, sl, causal, bf, 0.5f);
    Var q = make_var(mat(Q), mask & 1), k = make_var(mat(K), mask & 2),
        v = make_var(mat(V), mask & 4);
    if (gpu) {
        device::gpu_profile_take();
        device::step_begin();
        CHECK(device::deferring());
    }
    Var y = ops::fused_attention_precision(q, k, v, 0.5f, sl, causal,
                                           bf ? device::Precision::BF16 : device::Precision::FP32);
    // Changing global projection precision must not change captured backward mode.
    device::set_matmul_precision(bf ? device::Precision::FP32 : device::Precision::BF16);
    if (mask) backward(ops::mean(ops::mul(y, make_var(mat(G)))));
    double worst = check(y->data, ref.y, bf ? 5e-3 : 3e-5);
    if (mask & 1) worst = std::max(worst, check(q->grad, ref.dq, bf ? 5e-3 : 3e-5));
    if (mask & 2) worst = std::max(worst, check(k->grad, ref.dk, bf ? 5e-3 : 3e-5));
    if (mask & 4) worst = std::max(worst, check(v->grad, ref.dv, bf ? 5e-3 : 3e-5));
    size_t measured_gemms = 0;
    const size_t expected_gemms =
        2 + bool(mask & 4) + (bool(mask & 3) ? 1 + bool(mask & 1) + bool(mask & 2) : 0);
    if (gpu) {
        device::gpu_profile_take();
        for (const auto& count : device::gpu_profile_counts()) {
            if (count.first == (bf ? "gemm_bf16" : "gemm")) measured_gemms += count.second;
            CHECK(count.first != (bf ? "gemm" : "gemm_bf16"));
        }
        CHECK(measured_gemms == expected_gemms);
        device::step_end();
    }
    std::printf(
        "{\"raw_case\":true,\"seq\":%zu,\"batch\":%zu,\"causal\":%s,\"bf16\":%s,\"grad_mask\":%u,"
        "\"expected_gemms\":%zu,\"measured_device_gemms\":%zu",
        sl, batch, causal ? "true" : "false", bf ? "true" : "false", mask, expected_gemms,
        measured_gemms);
    print_grid("q", Q);
    print_grid("k", K);
    print_grid("v", V);
    print_grid("upstream_before_mean", G);
    print_grid("y", actual_grid(y->data));
    if (mask & 1) print_grid("dq", actual_grid(q->grad));
    if (mask & 2) print_grid("dk", actual_grid(k->grad));
    if (mask & 4) print_grid("dv", actual_grid(v->grad));
    std::printf("}\n");
    device::set_matmul_precision(device::Precision::FP32);
    std::printf(
        "{\"case\":true,\"seq\":%zu,\"batch\":%zu,\"d\":%zu,\"dv\":%zu,\"causal\":%s,\"bf16\":%s,"
        "\"grad_mask\":%u,\"worst\":%.9g}\n",
        sl, batch, d, dv, causal ? "true" : "false", bf ? "true" : "false", mask, worst);
}
void default_equivalence() {
    for (bool causal : {false, true}) {
        auto Q = fixture(7, 5, 2), K = fixture(7, 5, 4), V = fixture(7, 3, 6);
        auto q = make_var(mat(Q), true), k = make_var(mat(K), true), v = make_var(mat(V), true);
        auto y = ops::fused_attention(q, k, v, 0.5f, 7, causal);
        backward(ops::mean(y));
        auto a = make_var(mat(Q), true), b = make_var(mat(K), true), c = make_var(mat(V), true);
        auto z = ops::fused_attention_precision(a, b, c, 0.5f, 7, causal, device::Precision::FP32);
        backward(ops::mean(z));
        for (auto pair : {std::make_pair(&y->data, &z->data), std::make_pair(&q->grad, &a->grad),
                          std::make_pair(&k->grad, &b->grad), std::make_pair(&v->grad, &c->grad)})
            CHECK(std::memcmp(pair.first->get_data(), pair.second->get_data(),
                              pair.first->rows() * pair.first->cols() * sizeof(float)) == 0);
    }
}
}  // namespace
int main(int argc, char** argv) {
    bool gpu = argc == 2 && std::string(argv[1]) == "--require-cuda";
    if (argc > 2 || (argc == 2 && !gpu)) return 2;
#ifdef MICROTORCH_DEVCHECK
    constexpr bool checks = true;
#else
    constexpr bool checks = false;
#endif
    if (gpu && (!checks || !device::cuda_compiled() || !device::gpu_profile_on())) return 2;
    device::set(device::Device::CPU);
    default_equivalence();
    if (gpu) {
        device::set(device::Device::CUDA);
        device::set_device_ops(true);
        device::set_step_residency(true);
        device::set_defer_downloads(true);
    }
    size_t cases = 0;
    for (bool bf : {false, true})
        for (bool causal : {false, true}) {
            for (auto shape :
                 {std::array<size_t, 4>{1, 1, 3, 2}, {7, 1, 5, 3}, {4, 2, 8, 5}, {17, 2, 9, 7}}) {
                run(shape[0], shape[1], shape[2], shape[3], causal, bf, 7, gpu);
                ++cases;
            }
            for (unsigned mask = 0; mask < 7; ++mask) {
                run(3, 2, 5, 4, causal, bf, mask, gpu);
                ++cases;
            }
        }
    if (gpu) {
        device::set_defer_downloads(false);
        device::set_step_residency(false);
        device::set_device_ops(false);
        device::evict_all();
        device::set(device::Device::CPU);
    }
    auto q = make_var(Matrix(2, 3)), k = make_var(Matrix(2, 3)), v = make_var(Matrix(2, 4));
    bool refused = false;
    try {
        ops::fused_attention_precision(q, k, v, 0.5f, 2, true, static_cast<device::Precision>(99));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    CHECK(refused);
    std::printf(
        "{\"pass\":true,\"cases\":%zu,\"actual_cuda\":%s,\"devcheck\":%s,\"default_fp32_bit_"
        "equal\":true}\n",
        cases, gpu ? "true" : "false", checks ? "true" : "false");
}
