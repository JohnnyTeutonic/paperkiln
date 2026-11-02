// Host test of the chunked selective-scan forward (src/selective_scan_chunked.cuh), through the host emulation
// built from the same functions the CUDA kernels call. It establishes the algorithm, indexing and chunk
// boundaries on the host; device numerics (CUDA expf/log1pf, kernel execution) are checked on the GPU by
// tools/scan_bench.cu. For every case:
//   1. y and every state within 3e-5 of an independent binary64 recurrence, with the metric
//      |v - ref| / max(1, |ref|) (stricter, by up to 2x, than the frozen CUDA gate's |v - ref| / (1 + |ref|));
//   2. one chunk per sequence (L = sl and L > sl) gives the host ordered recurrence bit for bit;
//   3. two runs are bit-identical (measured and reported).
// Prints one JSON summary line; exit 1 on any failure.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/selective_scan_chunked.cuh"

namespace {

constexpr double kBound = 3e-5;

struct Case {
    size_t sl, batch, d, n, L;
    bool softplus, skip;
    int regime;  // 0 mixed decay, 1 long memory (a near 1), 2 fast decay (a near 0)
};

struct Inputs {
    std::vector<float> x, dt, B, C, A, skip;
};

uint64_t mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// deterministic value in [-1, 1)
float unit(uint64_t seed, size_t i) { return float(double(mix(seed * 1000003ULL + i) >> 11) / 9007199254740992.0 * 2 - 1); }

Inputs make(const Case& k) {
    const size_t T = k.sl * k.batch, D = k.d, N = k.n;
    Inputs in;
    for (size_t i = 0; i < T * D; ++i) in.x.push_back(unit(1, i));
    for (size_t i = 0; i < T * D; ++i) {
        const float u = unit(2, i);
        // delta = softplus(dt) or dt; long memory wants small delta, fast decay large
        float v = k.regime == 1 ? 0.001f + 0.004f * (u + 1) : k.regime == 2 ? 2.0f + 2.0f * u : 0.5f * (u + 1.2f);
        if (k.softplus) v = k.regime == 1 ? -6.0f + u : k.regime == 2 ? 2.0f + 2.0f * u : 1.5f * u;
        in.dt.push_back(v);
    }
    for (size_t i = 0; i < T * N; ++i) in.B.push_back(unit(3, i));
    for (size_t i = 0; i < T * N; ++i) in.C.push_back(unit(4, i));
    for (size_t i = 0; i < D * N; ++i)  // A = -exp(log range), as Mamba's A_log parametrisation
        in.A.push_back(-std::exp(float(k.regime == 1 ? -3.0 + 0.5 * unit(5, i) : 1.5 * unit(5, i))));
    for (size_t i = 0; i < D; ++i) in.skip.push_back(unit(6, i));
    return in;
}

// independent binary64 recurrence: y [T, D], states [(T + batch) D N] in the op's layout
void reference(const Case& k, const Inputs& in, std::vector<double>& y, std::vector<double>& S) {
    const size_t T = k.sl * k.batch, D = k.d, N = k.n, DN = D * N;
    y.assign(T * D, 0.0);
    S.assign((T + k.batch) * DN, 0.0);
    for (size_t row = 0; row < T; ++row) {
        const size_t q = row / k.sl, t = row % k.sl;
        double* base = S.data() + (q * k.sl + q) * DN;
        for (size_t c = 0; c < D; ++c) {
            const size_t cell = row * D + c;
            const double v = in.dt[cell];
            const double dl = k.softplus && v <= 20 ? std::log1p(std::exp(v)) : v;
            double acc = 0;
            for (size_t n = 0; n < N; ++n) {
                const size_t i = c * N + n;
                const double s = base[t * DN + i] * std::exp(dl * double(in.A[i])) + double(in.B[row * N + n]) * double(in.x[cell]) * dl;
                base[(t + 1) * DN + i] = s;
                acc += s * in.C[row * N + n];
            }
            y[cell] = acc + (k.skip ? double(in.x[cell]) * in.skip[c] : 0.0);
        }
    }
}

// the ordered kernel's recurrence (k_selective_scan_fwd), with the host's rounded operations
void ordered(const Case& k, const Inputs& in, std::vector<float>& y, std::vector<float>& S) {
    using namespace ss_chunked;
    const size_t T = k.sl * k.batch, D = k.d, N = k.n, DN = D * N;
    y.assign(T * D, 0.0f);
    S.assign((T + k.batch) * DN, 0.0f);
    for (size_t q = 0; q < k.batch; ++q)
        for (size_t c = 0; c < D; ++c) {
            float* base = S.data() + (q * k.sl + q) * DN;
            for (size_t t = 0; t < k.sl; ++t) {
                const size_t row = q * k.sl + t, cell = row * D + c;
                const float v = in.dt[cell];
                const float dl = k.softplus && v <= 20.0f ? softplus20(v) : v;
                const float xd = mul(in.x[cell], dl);
                float acc = 0.0f;
                for (size_t n = 0; n < N; ++n) {
                    const size_t i = c * N + n;
                    const float a = ex(mul(dl, in.A[i]));
                    const float s = add(mul(base[t * DN + i], a), mul(in.B[row * N + n], xd));
                    base[(t + 1) * DN + i] = s;
                    acc = add(acc, mul(s, in.C[row * N + n]));
                }
                y[cell] = k.skip ? add(acc, mul(in.x[cell], in.skip[c])) : acc;
            }
        }
}

void chunked(const Case& k, const Inputs& in, size_t L, std::vector<float>& y, std::vector<float>& S) {
    const size_t T = k.sl * k.batch, D = k.d, N = k.n;
    y.assign(T * D, 0.0f);
    S.assign((T + k.batch) * D * N, -1.0f);  // every slot must be written
    std::vector<float> delta(T * D);
    ss_chunked::selective_scan_chunked_host(in.x.data(), in.dt.data(), in.B.data(), in.C.data(), in.A.data(),
                                            k.skip ? in.skip.data() : nullptr, y.data(), S.data(), delta.data(),
                                            T, D, N, k.sl, L, k.softplus);
}

template <class V>
double worst(const V& got, const std::vector<double>& ref) {
    double w = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = std::fabs(double(got[i]) - ref[i]) / std::max(1.0, std::fabs(ref[i]));
        w = std::isfinite(e) ? std::max(w, e) : INFINITY;
    }
    return w;
}

// a JSON number, or null for a nonfinite value
struct Js {
    char s[32];
    explicit Js(double v) {
        if (std::isfinite(v)) std::snprintf(s, sizeof s, "%.3g", v);
        else std::snprintf(s, sizeof s, "null");
    }
};

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

int main() {
    std::vector<Case> cases;
    for (size_t sl : {1, 2, 7, 63, 64, 65, 127, 128, 129, 1000, 2048})
        for (size_t L : {1, 2, 16, 64, 256})
            for (int regime : {0, 1, 2}) {
                const bool sp = (sl + L + regime) % 2, skip = (sl + regime) % 3 != 0;
                cases.push_back({sl, 2, 3, 4, L, sp, skip, regime});
            }
    // wider channels and states at a Mamba-like state size; one sequence and several
    for (size_t batch : {1, 3})
        for (int regime : {0, 1, 2}) cases.push_back({1500, batch, 8, 16, 64, regime == 1, true, regime});
    cases.push_back({4096, 1, 2, 16, 128, false, true, 1});
    cases.push_back({4096, 1, 2, 16, 128, true, false, 2});

    size_t failures = 0, ordered_identical = 0, ordered_checks = 0, repeats_identical = 0;
    double worst_chunked = 0, worst_ordered = 0;
    for (const Case& k : cases) {
        const Inputs in = make(k);
        std::vector<double> ry, rS;
        reference(k, in, ry, rS);
        std::vector<float> oy, oS, cy, cS, cy2, cS2;
        ordered(k, in, oy, oS);
        chunked(k, in, k.L, cy, cS);
        chunked(k, in, k.L, cy2, cS2);
        const double ey = worst(cy, ry), eS = worst(cS, rS), oey = worst(oy, ry), oeS = worst(oS, rS);
        worst_chunked = std::max({worst_chunked, ey, eS});
        worst_ordered = std::max({worst_ordered, oey, oeS});
        const bool repeat = same_bits(cy, cy2) && same_bits(cS, cS2);
        repeats_identical += repeat;
        bool ok = ey <= kBound && eS <= kBound && repeat;
        // one chunk per sequence (L = sl, and L > sl, clamped) is the ordered recurrence exactly
        bool one_chunk = true;
        for (size_t L1 : {k.sl, k.sl + 1, 4 * k.sl + 7}) {
            std::vector<float> wy, wS;
            chunked(k, in, L1, wy, wS);
            ++ordered_checks;
            const bool same = same_bits(wy, oy) && same_bits(wS, oS);
            ordered_identical += same;
            one_chunk = one_chunk && same;
        }
        ok = ok && one_chunk;
        if (!ok) {
            ++failures;
            std::printf("{\"fail\": true, \"sl\": %zu, \"batch\": %zu, \"d\": %zu, \"n\": %zu, \"L\": %zu, \"softplus\": %d, "
                        "\"skip\": %d, \"regime\": %d, \"y\": %s, \"states\": %s, \"ordered_y\": %s, "
                        "\"one_chunk_identical\": %d}\n",
                        k.sl, k.batch, k.d, k.n, k.L, k.softplus, k.skip, k.regime, Js(ey).s, Js(eS).s, Js(oey).s,
                        one_chunk);
        }
    }
    std::printf("{\"pass\": %s, \"cases\": %zu, \"failures\": %zu, \"bound\": %.0e, \"worst_chunked\": %s, "
                "\"worst_ordered\": %s, \"one_chunk_bit_identical\": \"%zu/%zu\", \"repeat_bit_identical\": \"%zu/%zu\"}\n",
                failures ? "false" : "true", cases.size(), failures, kBound, Js(worst_chunked).s, Js(worst_ordered).s,
                ordered_identical, ordered_checks, repeats_identical, cases.size());
    return failures ? 1 : 0;
}
