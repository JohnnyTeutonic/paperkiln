// Independent binary64 selective-scan forward and backward, shared by the host test
// (tests/test_selective_scan_chunked_backward.cpp) and the device check (tools/scan_bwd_bench.cu).
// It uses none of the code under test. For each gradient it returns ref, the binary64 value, and
// mag, the same sum evaluated in absolute values (every term replaced by its magnitude, so
// mag >= |ref|, with equality for a sign-definite sum). Errors are measured as
// |v - ref| / max(1, mag) (experiments/mamba_parallel_scan_2026-11-03/BACKWARD_DESIGN.md).
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ss_reference {

struct Shape {
    size_t sl, batch, D, N;
    bool softplus, skip;
};

struct Inputs64 {
    std::vector<double> x, dt, B, C, A, skip, dy;  // the op's row-major layouts; dy is [T, D]
};

template <class V>
Inputs64 widen(const V& x, const V& dt, const V& B, const V& C, const V& A, const V& skip,
               const V& dy) {
    auto w = [](const V& v) { return std::vector<double>(v.begin(), v.end()); };
    return {w(x), w(dt), w(B), w(C), w(A), w(skip), w(dy)};
}

// gradient j: 0 dx [T, D], 1 ddt [T, D], 2 dB [T, N], 3 dC [T, N], 4 dA [D, N], 5 dD [D] (empty
// without skip)
inline void sizes(const Shape& s, size_t out[6]) {
    const size_t T = s.sl * s.batch;
    out[0] = out[1] = T * s.D;
    out[2] = out[3] = T * s.N;
    out[4] = s.D * s.N;
    out[5] = s.skip ? s.D : 0;
}

// sum over every row and channel of y dy
inline double loss(const Shape& s, const Inputs64& in) {
    const size_t T = s.sl * s.batch, D = s.D, N = s.N;
    std::vector<double> st(D * N);
    double z = 0;
    for (size_t row = 0; row < T; ++row) {
        if (row % s.sl == 0) std::fill(st.begin(), st.end(), 0.0);
        for (size_t c = 0; c < D; ++c) {
            const size_t cell = row * D + c;
            const double v = in.dt[cell];
            const double dl = s.softplus && v <= 20 ? std::log1p(std::exp(v)) : v;
            double y = s.skip ? in.x[cell] * in.skip[c] : 0.0;
            for (size_t n = 0; n < N; ++n) {
                double& sv = st[c * N + n];
                sv = sv * std::exp(dl * in.A[c * N + n]) + in.B[row * N + n] * in.x[cell] * dl;
                y += sv * in.C[row * N + n];
            }
            z += y * in.dy[cell];
        }
    }
    return z;
}

struct Backward {
    std::array<std::vector<double>, 6> ref, mag;
};

// The forward states of one sequence at a time are kept (2 (sl + 1) D N doubles), the adjoint runs
// backward over them.
inline Backward backward(const Shape& s, const Inputs64& in) {
    const size_t D = s.D, N = s.N, DN = D * N, sl = s.sl;
    size_t sz[6];
    sizes(s, sz);
    Backward r;
    for (size_t j = 0; j < 6; ++j) r.ref[j].assign(sz[j], 0.0), r.mag[j].assign(sz[j], 0.0);
    std::vector<double> st((sl + 1) * DN), sa((sl + 1) * DN), dl(sl * D), carry(DN), carry_a(DN);
    for (size_t q = 0; q < s.batch; ++q) {
        std::fill(st.begin(), st.begin() + DN, 0.0);
        std::fill(sa.begin(), sa.begin() + DN, 0.0);
        for (size_t t = 0; t < sl; ++t)
            for (size_t c = 0; c < D; ++c) {
                const size_t row = q * sl + t, cell = row * D + c;
                const double v = in.dt[cell];
                const double d = s.softplus && v <= 20 ? std::log1p(std::exp(v)) : v;
                dl[t * D + c] = d;
                for (size_t n = 0; n < N; ++n) {
                    const size_t i = c * N + n;
                    const double a = std::exp(d * in.A[i]), b = in.B[row * N + n] * in.x[cell] * d;
                    st[(t + 1) * DN + i] = st[t * DN + i] * a + b;
                    sa[(t + 1) * DN + i] = sa[t * DN + i] * a + std::fabs(b);
                }
            }
        std::fill(carry.begin(), carry.end(), 0.0);
        std::fill(carry_a.begin(), carry_a.end(), 0.0);
        for (size_t t = sl; t-- > 0;)
            for (size_t c = 0; c < D; ++c) {
                const size_t row = q * sl + t, cell = row * D + c;
                const double d = dl[t * D + c], xv = in.x[cell], dyv = in.dy[cell];
                double ddelta = 0, ddelta_a = 0, dxd = 0, dxd_a = 0;
                for (size_t n = 0; n < N; ++n) {
                    const size_t i = c * N + n, bn = row * N + n;
                    const double a = std::exp(d * in.A[i]);
                    const double g = dyv * in.C[bn] + carry[i];  // d loss / d s_t
                    const double ga = std::fabs(dyv * in.C[bn]) + carry_a[i];
                    const double dpre = g * st[t * DN + i] * a, dpre_a = ga * sa[t * DN + i] * a;
                    r.ref[4][i] += dpre * d, r.mag[4][i] += dpre_a * std::fabs(d);
                    ddelta += dpre * in.A[i], ddelta_a += dpre_a * std::fabs(in.A[i]);
                    dxd += g * in.B[bn], dxd_a += ga * std::fabs(in.B[bn]);
                    r.ref[2][bn] += g * xv * d, r.mag[2][bn] += ga * std::fabs(xv * d);
                    r.ref[3][bn] += dyv * st[(t + 1) * DN + i];
                    r.mag[3][bn] += std::fabs(dyv) * sa[(t + 1) * DN + i];
                    carry[i] = g * a, carry_a[i] = ga * a;
                }
                ddelta += dxd * xv, ddelta_a += dxd_a * std::fabs(xv);
                const double sk = s.skip ? dyv * in.skip[c] : 0.0;
                r.ref[0][cell] = dxd * d + sk;
                r.mag[0][cell] = dxd_a * std::fabs(d) + std::fabs(sk);
                const double v = in.dt[cell];
                const double sg = s.softplus && v <= 20 ? 1 / (1 + std::exp(-v)) : 1.0;
                r.ref[1][cell] = ddelta * sg, r.mag[1][cell] = ddelta_a * sg;
                if (s.skip) r.ref[5][c] += dyv * xv, r.mag[5][c] += std::fabs(dyv * xv);
            }
    }
    return r;
}

// worst |a - b| / max(1, mag) over gradient j's entries (b the reference, or another result);
// a nonfinite value counts as infinite
template <class VA, class VB>
double worst(const VA& a, const VB& b, const std::vector<double>& mag) {
    double w = 0;
    for (size_t i = 0; i < mag.size(); ++i) {
        const double e = std::fabs(double(a[i]) - double(b[i])) / std::max(1.0, mag[i]);
        w = std::isfinite(e) ? std::max(w, e) : INFINITY;
    }
    return w;
}

}  // namespace ss_reference
