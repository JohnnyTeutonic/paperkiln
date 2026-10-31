// Independent double recurrence/backpropagation and finite-difference oracle.
#include <algorithm>
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
using Vec = std::vector<double>;
struct Problem {
    size_t sl, batch, d, n;
    bool softplus, skip;
    std::array<Vec, 6> a;
};
Problem fixture(size_t sl, size_t batch, size_t d, size_t n, bool softplus, bool skip) {
    Problem p{sl, batch, d, n, softplus, skip, {}};
    size_t t = sl * batch;
    size_t sizes[] = {t * d, t * d, t * n, t * n, d * n, d};
    for (size_t v = 0; v < 6; ++v)
        for (size_t i = 0; i < sizes[v]; ++i) {
            double x = double(int((i * 7 + v * 5 + 3) % 23) - 11) / 19;
            if (v == 1) {
                const double vals[] = {-20, -.7, 0, 19.75, 20, 20.25};
                x = softplus ? vals[i % 6] : double(int(i % 5) - 2) / 10;
            }
            if (v == 4) x = -.2 - double(i % 7) / 25;
            p.a[v].push_back(float(x));
        }
    return p;
}
struct Ref {
    Vec y;
    std::array<Vec, 6> grad;
};
Vec upstream(const Problem& p) {
    Vec g(p.sl * p.batch * p.d);
    for (size_t i = 0; i < g.size(); ++i) g[i] = float(double(int((i * 3 + 2) % 17) - 8) / 13);
    return g;
}
Ref reference(const Problem& p, bool gradients = true) {
    size_t T = p.sl * p.batch, D = p.d, N = p.n, DN = D * N;
    const auto& x = p.a[0];
    const auto& dt = p.a[1];
    const auto& B = p.a[2];
    const auto& C = p.a[3];
    const auto& A = p.a[4];
    const auto& skip = p.a[5];
    Ref r;
    r.y.resize(T * D);
    for (size_t v = 0; v < 6; ++v) r.grad[v].resize(p.a[v].size());
    Vec states((T + p.batch) * DN), delta(T * D), g = upstream(p);
    for (size_t row = 0; row < T; ++row)
        for (size_t c = 0; c < D; ++c) {
            size_t q = row / p.sl, t = row % p.sl, base = q * (p.sl + 1) * DN, i = row * D + c;
            double dl = p.softplus && dt[i] <= 20 ? std::log1p(std::exp(dt[i])) : dt[i];
            delta[i] = dl;
            double y = 0;
            for (size_t n = 0; n < N; ++n) {
                size_t k = c * N + n;
                double s =
                    states[base + t * DN + k] * std::exp(dl * A[k]) + B[row * N + n] * x[i] * dl;
                states[base + (t + 1) * DN + k] = s;
                y += s * C[row * N + n];
            }
            r.y[i] = y + (p.skip ? x[i] * skip[c] : 0);
        }
    if (!gradients) return r;
    for (size_t q = 0; q < p.batch; ++q) {
        Vec carry(DN);
        size_t base = q * (p.sl + 1) * DN;
        for (size_t t = p.sl; t-- > 0;)
            for (size_t c = 0; c < D; ++c) {
                size_t row = q * p.sl + t, i = row * D + c;
                double dl = delta[i], dy = g[i] / double(T * D), d_delta = 0, d_xdl = 0;
                for (size_t n = 0; n < N; ++n) {
                    size_t k = c * N + n;
                    double ds = dy * C[row * N + n] + carry[k], a = std::exp(dl * A[k]);
                    double dpre = ds * states[base + t * DN + k] * a;
                    r.grad[3][row * N + n] += dy * states[base + (t + 1) * DN + k];
                    r.grad[4][k] += dpre * dl;
                    d_delta += dpre * A[k];
                    r.grad[2][row * N + n] += ds * x[i] * dl;
                    d_xdl += ds * B[row * N + n];
                    carry[k] = ds * a;
                }
                d_delta += d_xdl * x[i];
                r.grad[0][i] = d_xdl * dl + (p.skip ? dy * skip[c] : 0);
                r.grad[1][i] =
                    d_delta * (p.softplus && dt[i] <= 20 ? 1 / (1 + std::exp(-dt[i])) : 1);
                if (p.skip) r.grad[5][c] += dy * x[i];
            }
    }
    return r;
}
void finite_difference() {
    for (bool softplus : {false, true})
        for (bool skip : {false, true}) {
            Problem p = fixture(2, 2, 2, 2, softplus, skip);
            // FD stays away from the tiny mathematical jump in the threshold approximation.
            // The main matrix separately checks exact v=20 and both adjacent branches.
            for (auto& value : p.a[1])
                if (value == 20) value = 20.05;
            auto r = reference(p);
            auto g = upstream(p);
            auto loss = [&]() {
                auto y = reference(p, false).y;
                double z = 0;
                for (size_t i = 0; i < y.size(); ++i) z += y[i] * g[i] / y.size();
                return z;
            };
            for (size_t v = 0; v < (skip ? 6 : 5); ++v)
                for (size_t i = 0; i < p.a[v].size(); ++i) {
                    double saved = p.a[v][i], h = 1e-5;
                    p.a[v][i] = saved + h;
                    double plus = loss();
                    p.a[v][i] = saved - h;
                    double minus = loss();
                    p.a[v][i] = saved;
                    CHECK(std::fabs((plus - minus) / (2 * h) - r.grad[v][i]) /
                              (1 + std::fabs(r.grad[v][i])) <
                          2e-7);
                }
        }
}
Matrix matrix(size_t rows, size_t cols, const Vec& v) {
    Matrix m(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) m(i, j) = float(v[i * cols + j]);
    return m;
}
void poison(Matrix& m) {
    CHECK(device::host_stale(m));
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j) m(i, j) = 0;
    CHECK(device::host_stale(m));
}
double compare(Matrix& m, const Vec& ref) {
    device::materialize(m);
    CHECK(m.rows() * m.cols() == ref.size());
    double e = 0;
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j) {
            double a = m(i, j), b = ref[i * m.cols() + j];
            CHECK(std::isfinite(a));
            e = std::max(e, std::fabs(a - b) / (1 + std::fabs(b)));
        }
    CHECK(e < 3e-5);
    return e;
}
void dump(const char* key, Matrix& m) {
    device::materialize(m);
    std::printf(",\"%s\":[", key);
    for (size_t i = 0; i < m.rows(); ++i) {
        if (i) std::printf(",");
        std::printf("[");
        for (size_t j = 0; j < m.cols(); ++j) {
            if (j) std::printf(",");
            std::printf("%.9g", m(i, j));
        }
        std::printf("]");
    }
    std::printf("]");
}
void run(const Problem& p, unsigned mask, const std::string& mode, bool gpu) {
    size_t T = p.sl * p.batch, D = p.d, N = p.n;
    size_t rows[] = {T, T, T, T, D, 1}, cols[] = {D, D, N, N, N, D};
    std::array<Var, 6> leaf, args;
    for (size_t v = 0; v < 6; ++v)
        leaf[v] = args[v] = make_var(matrix(rows[v], cols[v], p.a[v]), mask & (1u << v));
    if (gpu) {
        device::gpu_profile_take();
        device::set_device_ops(true);
        device::set_step_residency(true);
        device::set_defer_downloads(mode != "write_through");
        device::step_begin();
        if (mode != "write_through") {
            CHECK(device::deferring());
            for (size_t v = 0; v < (p.skip ? 6 : 5); ++v) {
                args[v] = ops::slice_cols(leaf[v], 0, cols[v]);
                poison(args[v]->data);
            }
        }
        if (mode == "host_forward") device::set_device_ops(false);
    }
    auto y = ops::selective_scan(args[0], args[1], args[2], args[3], args[4],
                                 p.skip ? args[5] : Var{}, p.softplus, p.sl);
    if (gpu && mode == "host_forward") device::set_device_ops(true);
    if (mask) {
        auto loss = ops::mean(ops::mul(y, make_var(matrix(T, D, upstream(p)))));
        if (gpu && mode == "host_backward") device::set_device_ops(false);
        if (gpu && mode == "deferred") {
            auto callback = y->backward_fn;
            Variable* ptr = y.get();
            y->backward_fn = [callback, ptr]() {
                poison(ptr->grad);
                callback();
            };
        }
        backward(loss);
    }
    auto r = reference(p);
    double worst = compare(y->data, r.y);
    for (size_t v = 0; v < (p.skip ? 6 : 5); ++v) {
        if (mask & (1u << v))
            worst = std::max(worst, compare(leaf[v]->grad, r.grad[v]));
        else
            CHECK(leaf[v]->grad.rows() == 0);
    }
    size_t fwd = 0, bwd = 0;
    if (gpu) {
        device::gpu_profile_take();
        for (auto& item : device::gpu_profile_counts()) {
            if (item.first == "selective_scan_fwd") fwd += item.second;
            if (item.first == "selective_scan_bwd") bwd += item.second;
        }
        CHECK(fwd == (mode == "host_forward" ? 0u : 1u));
        CHECK(bwd == ((mask && mode != "host_backward") ? 1u : 0u));
        device::set_device_ops(true);
        device::step_end();
        CHECK(device::stale_count() == 0);
    }
    std::printf(
        "{\"raw_case\":true,\"seq\":%zu,\"batch\":%zu,\"d\":%zu,\"n\":%zu,\"softplus\":%s,\"skip\":"
        "%s,\"mask\":%u,\"mode\":\"%s\",\"actual_cuda\":%s,\"device_fwd\":%zu,\"device_bwd\":%zu,"
        "\"worst\":%.17g",
        p.sl, p.batch, D, N, p.softplus ? "true" : "false", p.skip ? "true" : "false", mask,
        mode.c_str(), gpu ? "true" : "false", fwd, bwd, worst);
    const char* keys[] = {"x", "dt", "B", "C", "A", "Dskip"};
    const char* grads[] = {"dx", "ddt", "dB", "dC", "dA", "dD"};
    for (size_t v = 0; v < (p.skip ? 6 : 5); ++v) {
        dump(keys[v], leaf[v]->data);
        if (mask & (1u << v)) dump(grads[v], leaf[v]->grad);
    }
    dump("y", y->data);
    std::printf("}\n");
    if (gpu) {
        device::set_defer_downloads(false);
        device::set_step_residency(false);
        device::set_device_ops(false);
        device::evict_all();
        device::set(device::Device::CPU);
        device::set(device::Device::CUDA);
    }
}
int main(int argc, char** argv) {
    bool gpu = argc == 2 && std::string(argv[1]) == "--require-cuda";
    if (argc > 2 || (argc == 2 && !gpu)) return 2;
#ifdef MICROTORCH_DEVCHECK
    constexpr bool checks = true;
#else
    constexpr bool checks = false;
#endif
    if (gpu && (!device::cuda_compiled() || !checks || !device::gpu_profile_on())) return 2;
    device::set(device::Device::CPU);
    device::set_device_ops(false);
    finite_difference();
    if (gpu) device::set(device::Device::CUDA);
    size_t count = 0;
    for (bool sp : {false, true})
        for (bool skip : {false, true}) {
            unsigned full = skip ? 63 : 31;
            for (unsigned mask = 0; mask <= full; ++mask) {
                run(fixture(2, 2, 3, 2, sp, skip), mask, "deferred", gpu);
                ++count;
            }
            for (auto shape :
                 {std::array<size_t, 4>{1, 1, 1, 1}, {3, 2, 3, 2}, {9, 2, 5, 3}, {4, 2, 2, 5}}) {
                run(fixture(shape[0], shape[1], shape[2], shape[3], sp, skip), full, "deferred",
                    gpu);
                ++count;
            }
            for (const char* mode : {"write_through", "host_forward", "host_backward"}) {
                run(fixture(3, 2, 3, 2, sp, skip), full, mode, gpu);
                ++count;
            }
        }
    if (gpu) {
        device::set_device_ops(false);
        device::evict_all();
        device::set(device::Device::CPU);
    }
    std::printf(
        "{\"pass\":true,\"cases\":%zu,\"actual_cuda\":%s,\"devcheck\":%s,\"finite_difference\":"
        "true}\n",
        count, gpu ? "true" : "false", checks ? "true" : "false");
    return 0;
}
