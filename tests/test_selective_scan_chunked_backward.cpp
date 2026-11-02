// Host test of the chunked selective-scan backward (src/selective_scan_chunked_backward.cuh),
// through the host emulation built from the same functions the CUDA kernels call. It establishes
// the algorithm, indexing and chunk boundaries on the host; device numerics are checked on the GPU
// by tools/scan_bwd_bench.cu. Tolerances: experiments/mamba_parallel_scan_2026-11-03/
// BACKWARD_DESIGN.md. The forward test's 173 shapes, each with an upstream gradient dy, and for
// every case:
//   1. the host emulation of the ordered backward kernel equals the CPU op's backward
//      (ops::selective_scan, the CPU path) bit for bit;
//   2. the chunked dx, ddt, dB, dC, dA and dD, from the ordered forward's states, within 3e-5 of an
//      independent binary64 backward, under |v - ref| / max(1, mag), where mag is the binary64 sum
//      of the absolute values of the terms that make up ref (mag >= |ref|; equal for sign-definite
//      sums); the ordered backward under the same bound; chunked against ordered under it too;
//   3. one chunk per sequence (L = sl, sl + 1, 4 sl + 7): dx, ddt, dB, dC bit-identical to the
//      ordered backward, and dA and dD as well for one sequence (with several sequences the
//      per-sequence partial sums of dA and dD regroup the ordered running sum: within the bound);
//   4. repeats bit-identical; every output written (pre-filled with NaN); each output computed
//      alone bit-identical to the same output computed with all the others (sl <= 129);
//   5. the chunked forward's states into the chunked backward (the opt-in path end to end) within
//      the bound of binary64.
// Plus: the binary64 backward against binary64 central differences on small cases.
// Prints one JSON summary line; exit 1 on any failure.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "../src/selective_scan_chunked_backward.cuh"
#include "selective_scan_backward_reference.hpp"
#include "microtorch/device.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

namespace {

constexpr double kBound = 3e-5;
constexpr double kFdBound = 1e-6;

struct Case {
    size_t sl, batch, d, n, L;
    bool softplus, skip;
    int regime;  // 0 mixed decay, 1 long memory (a near 1), 2 fast decay (a near 0)
};

// x, dt, B, C, A, skip, dy
struct Inputs {
    std::vector<float> x, dt, B, C, A, skip, dy;
};

uint64_t mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

float unit(uint64_t seed, size_t i) {
    return float(double(mix(seed * 1000003ULL + i) >> 11) / 9007199254740992.0 * 2 - 1);
}

// the forward test's inputs, plus dy
Inputs make(const Case& k) {
    const size_t T = k.sl * k.batch, D = k.d, N = k.n;
    Inputs in;
    for (size_t i = 0; i < T * D; ++i) in.x.push_back(unit(1, i));
    for (size_t i = 0; i < T * D; ++i) {
        const float u = unit(2, i);
        float v = k.regime == 1   ? 0.001f + 0.004f * (u + 1)
                  : k.regime == 2 ? 2.0f + 2.0f * u
                                  : 0.5f * (u + 1.2f);
        if (k.softplus) v = k.regime == 1 ? -6.0f + u : k.regime == 2 ? 2.0f + 2.0f * u : 1.5f * u;
        in.dt.push_back(v);
    }
    for (size_t i = 0; i < T * N; ++i) in.B.push_back(unit(3, i));
    for (size_t i = 0; i < T * N; ++i) in.C.push_back(unit(4, i));
    for (size_t i = 0; i < D * N; ++i)
        in.A.push_back(
            -std::exp(float(k.regime == 1 ? -3.0 + 0.5 * unit(5, i) : 1.5 * unit(5, i))));
    for (size_t i = 0; i < D; ++i) in.skip.push_back(unit(6, i));
    for (size_t i = 0; i < T * D; ++i) in.dy.push_back(unit(7, i));
    return in;
}

// gradients in the order dx, ddt, dB, dC, dA, dD
using Grads = std::array<std::vector<float>, 6>;
const char* const kNames[6] = {"dx", "ddt", "dB", "dC", "dA", "dD"};

void sizes(const Case& k, size_t out[6]) {
    const size_t T = k.sl * k.batch;
    out[0] = out[1] = T * k.d;
    out[2] = out[3] = T * k.n;
    out[4] = k.d * k.n;
    out[5] = k.skip ? k.d : 0;
}

// the ordered forward kernel (k_selective_scan_fwd) with the host's rounded operations: S, delta
void ordered_forward(const Case& k, const Inputs& in, std::vector<float>& S,
                     std::vector<float>& delta) {
    using namespace ss_chunked;
    const size_t T = k.sl * k.batch, D = k.d, N = k.n, DN = D * N;
    S.assign((T + k.batch) * DN, 0.0f);
    delta.assign(T * D, 0.0f);
    for (size_t q = 0; q < k.batch; ++q)
        for (size_t c = 0; c < D; ++c) {
            float* base = S.data() + (q * k.sl + q) * DN;
            for (size_t t = 0; t < k.sl; ++t) {
                const size_t row = q * k.sl + t, cell = row * D + c;
                const float v = in.dt[cell];
                const float dl = k.softplus && v <= 20.0f ? softplus20(v) : v;
                delta[cell] = dl;
                const float xd = mul(in.x[cell], dl);
                for (size_t n = 0; n < N; ++n) {
                    const size_t i = c * N + n;
                    const float a = ex(mul(dl, in.A[i]));
                    base[(t + 1) * DN + i] =
                        add(mul(base[t * DN + i], a), mul(in.B[row * N + n], xd));
                }
            }
        }
}

// the ordered backward kernels (k_selective_scan_bwd_channels, then k_selective_scan_bwd_BC) with
// the host's rounded operations
Grads ordered_backward(const Case& k, const Inputs& in, const std::vector<float>& S,
                       const std::vector<float>& delta) {
    using namespace ss_chunked;
    const size_t T = k.sl * k.batch, D = k.d, N = k.n, DN = D * N, sl = k.sl;
    size_t sz[6];
    sizes(k, sz);
    Grads o;
    for (size_t j = 0; j < 6; ++j) o[j].assign(sz[j], 0.0f);
    std::vector<float> G(DN), hist(T * DN);
    for (size_t c = 0; c < D; ++c) {
        float skip_gradient = 0.0f;
        for (size_t r0 = 0, q = 0; r0 < T; r0 += sl, ++q) {
            const float* base = S.data() + (r0 + q) * DN;
            for (size_t n = 0; n < N; ++n) G[c * N + n] = 0.0f;
            for (size_t t = sl; t-- > 0;) {
                const size_t row = r0 + t, cell = row * D + c;
                const float dl = delta[cell], xv = in.x[cell], dyv = in.dy[cell];
                float ddelta = 0.0f, dxd = 0.0f;
                for (size_t n = 0; n < N; ++n) {
                    const size_t i = c * N + n;
                    const float g = add(mul(dyv, in.C[row * N + n]), G[i]);
                    hist[row * DN + i] = g;
                    const float Acn = in.A[i], a = ex(mul(dl, Acn));
                    const float dpre = mul(mul(g, base[t * DN + i]), a);
                    o[4][i] = add(o[4][i], mul(dpre, dl));
                    ddelta = add(ddelta, mul(dpre, Acn));
                    dxd = add(dxd, mul(g, in.B[row * N + n]));
                    G[i] = mul(g, a);
                }
                ddelta = add(ddelta, mul(dxd, xv));
                o[0][cell] = add(mul(dxd, dl), k.skip ? mul(dyv, in.skip[c]) : 0.0f);
                if (k.skip) skip_gradient = add(skip_gradient, mul(dyv, xv));
                const float v = in.dt[cell];
                o[1][cell] = k.softplus && v <= 20.0f ? mul(ddelta, 1.0f / add(1.0f, ex(-v)))
                                                      : ddelta;
            }
        }
        if (k.skip) o[5][c] = skip_gradient;
    }
    for (size_t idx = 0; idx < T * N; ++idx) {
        const size_t row = idx / N, n = idx % N, q = row / sl, t = row % sl;
        const float* sn = S.data() + (q * (sl + 1) + t + 1) * DN;
        float gb = 0.0f, gc = 0.0f;
        for (size_t c = 0; c < D; ++c) {
            const size_t cell = row * D + c, i = c * N + n;
            gb = add(gb, mul(hist[row * DN + i], mul(in.x[cell], delta[cell])));
            gc = add(gc, mul(in.dy[cell], sn[i]));
        }
        o[2][idx] = gb;
        o[3][idx] = gc;
    }
    return o;
}

// the CPU op's backward (ops::selective_scan, host path), seeded with dy
Grads cpu_op_backward(const Case& k, const Inputs& in) {
    const size_t T = k.sl * k.batch, D = k.d, N = k.n;
    auto mat = [](size_t r, size_t c, const std::vector<float>& v) {
        Matrix m(r, c);
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < c; ++j) m(i, j) = v[i * c + j];
        return m;
    };
    Var x = make_var(mat(T, D, in.x), true), dt = make_var(mat(T, D, in.dt), true),
        B = make_var(mat(T, N, in.B), true), C = make_var(mat(T, N, in.C), true),
        A = make_var(mat(D, N, in.A), true), sk = make_var(mat(1, D, in.skip), true);
    Var y = ops::selective_scan(x, dt, B, C, A, k.skip ? sk : Var{}, k.softplus, k.sl);
    y->grad = mat(T, D, in.dy);
    y->backward_fn();
    Grads o;
    const Var* vars[6] = {&x, &dt, &B, &C, &A, &sk};
    for (size_t j = 0; j < (k.skip ? 6u : 5u); ++j) {
        const Matrix& g = (*vars[j])->grad;
        for (size_t r = 0; r < g.rows(); ++r)
            for (size_t c = 0; c < g.cols(); ++c) o[j].push_back(g(r, c));
    }
    return o;
}

// The independent binary64 reference (tests/selective_scan_backward_reference.hpp), shared with
// the device check
using Ref = ss_reference::Backward;
using InD = ss_reference::Inputs64;
ss_reference::Shape shape(const Case& k) { return {k.sl, k.batch, k.d, k.n, k.softplus, k.skip}; }
InD widen(const Inputs& in) {
    return ss_reference::widen(in.x, in.dt, in.B, in.C, in.A, in.skip, in.dy);
}
double loss64(const Case& k, const InD& in) { return ss_reference::loss(shape(k), in); }
Ref reference(const Case& k, const InD& in) { return ss_reference::backward(shape(k), in); }

// worst |v - ref| / max(1, mag) over every output; nonfinite counts as infinite
double worst(const Grads& got, const Ref& r) {
    double w = 0;
    for (size_t j = 0; j < 6; ++j)
        for (size_t i = 0; i < r.ref[j].size(); ++i) {
            const double e =
                std::fabs(double(got[j][i]) - r.ref[j][i]) / std::max(1.0, r.mag[j][i]);
            w = std::isfinite(e) ? std::max(w, e) : INFINITY;
        }
    return w;
}
// reported, not gated: the forward test's metric |v - ref| / max(1, |ref|), which a long sum that
// cancels can fail through rounding alone (BACKWARD_DESIGN.md)
double worst_plain(const Grads& got, const Ref& r) {
    double w = 0;
    for (size_t j = 0; j < 6; ++j)
        for (size_t i = 0; i < r.ref[j].size(); ++i) {
            const double e =
                std::fabs(double(got[j][i]) - r.ref[j][i]) / std::max(1.0, std::fabs(r.ref[j][i]));
            w = std::isfinite(e) ? std::max(w, e) : INFINITY;
        }
    return w;
}
// the same metric between two float results (b standing for the reference)
double worst_between(const Grads& a, const Grads& b, const Ref& r) {
    double w = 0;
    for (size_t j = 0; j < 6; ++j)
        for (size_t i = 0; i < r.ref[j].size(); ++i) {
            const double e =
                std::fabs(double(a[j][i]) - double(b[j][i])) / std::max(1.0, r.mag[j][i]);
            w = std::isfinite(e) ? std::max(w, e) : INFINITY;
        }
    return w;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// chunked backward through the host emulation; outputs pre-filled with NaN. want selects outputs
// (bit j for gradient j); unwanted ones stay empty.
Grads chunked_backward(const Case& k, const Inputs& in, const std::vector<float>& S,
                       const std::vector<float>& delta, size_t L, unsigned want = 63) {
    size_t sz[6];
    sizes(k, sz);
    Grads o;
    float* p[6];
    for (size_t j = 0; j < 6; ++j) {
        const bool on = (want >> j & 1) && sz[j];
        if (on) o[j].assign(sz[j], std::numeric_limits<float>::quiet_NaN());
        p[j] = on ? o[j].data() : nullptr;
    }
    ss_chunked::selective_scan_chunked_backward_host(
        in.x.data(), in.dt.data(), in.B.data(), in.C.data(), in.A.data(),
        k.skip ? in.skip.data() : nullptr, in.dy.data(), S.data(), delta.data(), k.sl * k.batch,
        k.d, k.n, k.sl, L, k.softplus, p[0], p[1], p[2], p[3], p[4], p[5]);
    return o;
}

bool all_finite(const Grads& g) {
    for (const auto& v : g)
        for (float f : v)
            if (!std::isfinite(f)) return false;
    return true;
}

struct Js {
    char s[32];
    explicit Js(double v) {
        if (std::isfinite(v))
            std::snprintf(s, sizeof s, "%.3g", v);
        else
            std::snprintf(s, sizeof s, "null");
    }
};

// the binary64 backward against binary64 central differences of sum(y dy)
double finite_difference_check(size_t& checked) {
    double w = 0;
    for (bool sp : {false, true})
        for (bool skip : {false, true})
            for (int regime : {0, 1, 2}) {
                const Case k{7, 2, 3, 4, 3, sp, skip, regime};
                InD in = widen(make(k));
                const Ref r = reference(k, in);
                std::vector<double>* vars[6] = {&in.x, &in.dt, &in.B, &in.C, &in.A, &in.skip};
                for (size_t j = 0; j < (skip ? 6u : 5u); ++j)
                    for (size_t i = 0; i < vars[j]->size(); ++i) {
                        double& v = (*vars[j])[i];
                        const double saved = v, h = 1e-5 * std::max(1.0, std::fabs(saved));
                        v = saved + h;
                        const double plus = loss64(k, in);
                        v = saved - h;
                        const double minus = loss64(k, in);
                        v = saved;
                        const double fd = (plus - minus) / (2 * h);
                        w = std::max(w, std::fabs(fd - r.ref[j][i]) / (1 + std::fabs(r.ref[j][i])));
                        ++checked;
                    }
            }
    return w;
}

}  // namespace

int main() {
    device::set(device::Device::CPU);
    device::set_device_ops(false);
    std::vector<Case> cases;
    for (size_t sl : {1, 2, 7, 63, 64, 65, 127, 128, 129, 1000, 2048})
        for (size_t L : {1, 2, 16, 64, 256})
            for (int regime : {0, 1, 2}) {
                const bool sp = (sl + L + regime) % 2, skip = (sl + regime) % 3 != 0;
                cases.push_back({sl, 2, 3, 4, L, sp, skip, regime});
            }
    for (size_t batch : {1, 3})
        for (int regime : {0, 1, 2})
            cases.push_back({1500, batch, 8, 16, 64, regime == 1, true, regime});
    cases.push_back({4096, 1, 2, 16, 128, false, true, 1});
    cases.push_back({4096, 1, 2, 16, 128, true, false, 2});

    size_t failures = 0, cpu_identical = 0, one_chunk_checks = 0, one_chunk_identical = 0,
           one_chunk_regrouped = 0, repeats = 0, finite = 0, mask_checks = 0, mask_identical = 0;
    double worst_chunked = 0, worst_ordered = 0, worst_vs_ordered = 0, worst_pipeline = 0,
           plain_chunked = 0, plain_ordered = 0;
    for (const Case& k : cases) {
        const Inputs in = make(k);
        const Ref r = reference(k, widen(in));
        std::vector<float> S, delta;
        ordered_forward(k, in, S, delta);
        const Grads ord = ordered_backward(k, in, S, delta);
        const Grads op = cpu_op_backward(k, in);
        bool cpu_same = true;
        for (size_t j = 0; j < 6; ++j) cpu_same = cpu_same && same_bits(ord[j], op[j]);
        cpu_identical += cpu_same;

        const Grads ch = chunked_backward(k, in, S, delta, k.L);
        const Grads ch2 = chunked_backward(k, in, S, delta, k.L);
        bool repeat = true;
        for (size_t j = 0; j < 6; ++j) repeat = repeat && same_bits(ch[j], ch2[j]);
        repeats += repeat;
        const bool fin = all_finite(ch);
        finite += fin;
        const double ec = worst(ch, r), eo = worst(ord, r), eco = worst_between(ch, ord, r);
        worst_chunked = std::max(worst_chunked, ec);
        worst_ordered = std::max(worst_ordered, eo);
        worst_vs_ordered = std::max(worst_vs_ordered, eco);
        plain_chunked = std::max(plain_chunked, worst_plain(ch, r));
        plain_ordered = std::max(plain_ordered, worst_plain(ord, r));
        bool ok = cpu_same && repeat && fin && ec <= kBound && eo <= kBound && eco <= kBound;

        // one chunk per sequence
        bool one = true;
        for (size_t L1 : {k.sl, k.sl + 1, 4 * k.sl + 7}) {
            const Grads w = chunked_backward(k, in, S, delta, L1);
            for (size_t j = 0; j < 6; ++j) {
                const bool same = same_bits(w[j], ord[j]);
                if (j >= 4 && k.batch > 1) {  // regrouped partial sums: within the bound
                    ++one_chunk_regrouped;
                    continue;
                }
                ++one_chunk_checks;
                one_chunk_identical += same;
                one = one && same;
            }
            one = one && worst_between(w, ord, r) <= kBound;
        }
        ok = ok && one;

        // each output alone
        bool masks = true;
        if (k.sl <= 129)
            for (unsigned j = 0; j < 6; ++j) {
                if (j == 5 && !k.skip) continue;
                const Grads w = chunked_backward(k, in, S, delta, k.L, 1u << j);
                ++mask_checks;
                const bool same = same_bits(w[j], ch[j]);
                mask_identical += same;
                masks = masks && same;
            }
        ok = ok && masks;

        // the opt-in path end to end: chunked forward states into the chunked backward
        std::vector<float> cS((k.sl + 1) * k.batch * k.d * k.n), cdelta(k.sl * k.batch * k.d),
            cy(k.sl * k.batch * k.d);
        ss_chunked::selective_scan_chunked_host(
            in.x.data(), in.dt.data(), in.B.data(), in.C.data(), in.A.data(),
            k.skip ? in.skip.data() : nullptr, cy.data(), cS.data(), cdelta.data(),
            k.sl * k.batch, k.d, k.n, k.sl, k.L, k.softplus);
        const double ep = worst(chunked_backward(k, in, cS, cdelta, k.L), r);
        worst_pipeline = std::max(worst_pipeline, ep);
        ok = ok && ep <= kBound;

        if (!ok) {
            ++failures;
            std::printf(
                "{\"fail\": true, \"sl\": %zu, \"batch\": %zu, \"d\": %zu, \"n\": %zu, \"L\": %zu, "
                "\"softplus\": %d, \"skip\": %d, \"regime\": %d, \"chunked\": %s, \"ordered\": %s, "
                "\"chunked_vs_ordered\": %s, \"pipeline\": %s, \"cpu_op_identical\": %d, "
                "\"one_chunk\": %d, \"repeat\": %d, \"finite\": %d, \"masks\": %d}\n",
                k.sl, k.batch, k.d, k.n, k.L, k.softplus, k.skip, k.regime, Js(ec).s, Js(eo).s,
                Js(eco).s, Js(ep).s, cpu_same, one, repeat, fin, masks);
        }
    }
    size_t fd_checked = 0;
    const double fd = finite_difference_check(fd_checked);
    const bool fd_ok = fd <= kFdBound;
    std::printf(
        "{\"pass\": %s, \"cases\": %zu, \"failures\": %zu, \"bound\": %.0e, "
        "\"worst_chunked_vs_binary64\": %s, \"worst_ordered_vs_binary64\": %s, "
        "\"worst_chunked_vs_ordered\": %s, \"worst_pipeline_vs_binary64\": %s, "
        "\"reported_forward_metric_chunked\": %s, \"reported_forward_metric_ordered\": %s, "
        "\"ordered_emulation_equals_cpu_op\": \"%zu/%zu\", \"one_chunk_bit_identical\": "
        "\"%zu/%zu\", \"one_chunk_regrouped_within_bound\": %zu, \"repeat_bit_identical\": "
        "\"%zu/%zu\", \"all_written_finite\": \"%zu/%zu\", \"single_output_bit_identical\": "
        "\"%zu/%zu\", \"fd_checked\": %zu, \"fd_worst\": %s, \"fd_bound\": %.0e}\n",
        failures || !fd_ok ? "false" : "true", cases.size(), failures, kBound, Js(worst_chunked).s,
        Js(worst_ordered).s, Js(worst_vs_ordered).s, Js(worst_pipeline).s, Js(plain_chunked).s,
        Js(plain_ordered).s, cpu_identical,
        cases.size(), one_chunk_identical, one_chunk_checks, one_chunk_regrouped, repeats,
        cases.size(), finite, cases.size(), mask_identical, mask_checks, fd_checked, Js(fd).s,
        kFdBound);
    return failures || !fd_ok ? 1 : 0;
}
