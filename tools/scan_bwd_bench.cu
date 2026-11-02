// Ordered against chunked selective-scan backward on one GPU, on device-resident buffers (no host
// transfers in the timed region). PROTOCOL_BACKWARD.md gate 4 and the timings. For each shape, from
// the ordered forward kernel's states and an upstream gradient dy, all six gradients (dx, ddt, dB,
// dC, dA, dD) requested:
//   - the ordered backward kernels and the chunked backward (for each L) against the binary64
//     backward (tests/selective_scan_backward_reference.hpp), |v - ref| / max(1, mag) <= 3e-5, and
//     chunked against ordered under the same metric and bound; any nonfinite value a failure;
//   - every output written (pre-filled with NaN); repeat runs bit-identical;
//   - one chunk per sequence (L = sl and sl + 5): dx, ddt, dB, dC bit-identical to the ordered
//     kernels on the device, and dA and dD as well when there is one sequence;
//   - the chunked forward's states (same L) into the chunked backward within the bound of binary64;
//   - timings: ordered and chunked alternate within each repetition; median of the per-repetition
//     elapsed stream time (CUDA events on the default stream, spanning the two ordered launches or
//     the chunked path's eight, idle gaps included). Scratch buffers are allocated outside the timed
//     region.
// A component timing of the scan backward, not a model-speed claim.
// build: nvcc -O3 -std=c++17 -arch=sm_89 -I src -I tests tools/scan_bwd_bench.cu -o scan_bwd_bench
// run:   ./scan_bwd_bench [reps=20]   (one JSON line per shape and L, then a summary line; exit 1 on
//        any failure)
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ordered {
struct Matrix {  // stand-in: the header's host geometry helper takes paperkiln's Matrix, unused here
    size_t rows() const { return 0; }
    size_t cols() const { return 0; }
};
#include "selective_scan_kernels.cuh"
}  // namespace ordered
#include "selective_scan_backward_reference.hpp"
#include "selective_scan_chunked_backward.cuh"

#define CK(x)                                                                               \
    do {                                                                                    \
        cudaError_t e_ = (x);                                                               \
        if (e_ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

namespace {

constexpr int kThreads = 256;
constexpr double kBound = 3e-5;
int grid(size_t n) {
    const size_t g = (n + kThreads - 1) / kThreads;
    return static_cast<int>(g > 4096 ? 4096 : g);
}

uint64_t mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
float unit(uint64_t seed, size_t i) {
    return float(double(mix(seed * 1000003ULL + i) >> 11) / 9007199254740992.0 * 2 - 1);
}

struct Shape {
    size_t sl, batch, D, N;
    bool softplus;
};

struct Host {
    std::vector<float> x, dt, B, C, A, skip, dy;
};

// the forward bench's inputs (tools/scan_bench.cu), plus dy
Host make(const Shape& s) {
    const size_t T = s.sl * s.batch;
    Host h;
    for (size_t i = 0; i < T * s.D; ++i) h.x.push_back(unit(1, i));
    for (size_t i = 0; i < T * s.D; ++i)
        h.dt.push_back(s.softplus ? -4.0f + 3.0f * unit(2, i) : 0.02f * (unit(2, i) + 1.1f));
    for (size_t i = 0; i < T * s.N; ++i) h.B.push_back(unit(3, i));
    for (size_t i = 0; i < T * s.N; ++i) h.C.push_back(unit(4, i));
    for (size_t i = 0; i < s.D * s.N; ++i) h.A.push_back(-std::exp(float(1.5 * unit(5, i))));
    for (size_t i = 0; i < s.D; ++i) h.skip.push_back(unit(6, i));
    for (size_t i = 0; i < T * s.D; ++i) h.dy.push_back(unit(7, i));
    return h;
}

struct Js {
    char s[32];
    explicit Js(double v) {
        if (std::isfinite(v))
            std::snprintf(s, sizeof s, "%.4g", v);
        else
            std::snprintf(s, sizeof s, "null");
    }
};

float* up(const std::vector<float>& v) {
    float* d;
    CK(cudaMalloc(&d, v.size() * sizeof(float)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}
float* zeros(size_t n) {
    float* d;
    CK(cudaMalloc(&d, (n ? n : 1) * sizeof(float)));
    CK(cudaMemset(d, 0, (n ? n : 1) * sizeof(float)));
    return d;
}
std::vector<float> down(const float* d, size_t n) {
    std::vector<float> v(n);
    CK(cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

using Grads = std::array<std::vector<float>, 6>;

struct Dev {
    float *x, *dt, *B, *C, *A, *skip, *dy, *y, *S, *delta, *S2, *delta2;
    float* g[6];                                       // dx, ddt, dB, dC, dA, dD
    float *G, *history;                                // ordered scratch
    float *end, *prod, *start, *hist, *dA_part, *dD_part;  // chunked scratch (largest L)
    size_t n[6];
};

void prefill_nan(const Dev& d) {
    for (size_t j = 0; j < 6; ++j) CK(cudaMemset(d.g[j], 0xff, d.n[j] * sizeof(float)));
}
Grads fetch(const Dev& d) {
    Grads o;
    for (size_t j = 0; j < 6; ++j) o[j] = down(d.g[j], d.n[j]);
    return o;
}

void run_ordered(const Shape& s, const Dev& d) {
    const size_t T = s.sl * s.batch;
    ordered::k_selective_scan_bwd_channels<<<grid(s.D), kThreads>>>(
        d.x, d.dt, d.B, d.C, d.A, d.skip, d.dy, d.S, d.delta, d.G, d.history, d.g[0], d.g[1],
        d.g[4], d.g[5], T, s.D, s.N, s.sl, s.softplus);
    ordered::k_selective_scan_bwd_BC<<<grid(T * s.N), kThreads>>>(
        d.x, d.dy, d.S, d.delta, d.history, d.g[2], d.g[3], T, s.D, s.N, s.sl);
}
void run_chunked(const Shape& s, const Dev& d, size_t L, const float* S, const float* delta) {
    const ss_chunked::Geom g = ss_chunked::geom(s.sl * s.batch, s.D, s.N, s.sl, L);
    ss_chunked::launch_backward(g, d.x, d.dt, d.B, d.C, d.A, d.skip, d.dy, S, delta, d.g[0],
                                d.g[1], d.g[2], d.g[3], d.g[4], d.g[5], d.end, d.prod, d.start,
                                d.hist, d.dA_part, d.dD_part, s.softplus, grid, kThreads);
}

// errors of every gradient against the binary64 reference, and between two results
double vs_ref(const Grads& v, const ss_reference::Backward& r) {
    double w = 0;
    for (size_t j = 0; j < 6; ++j) w = std::max(w, ss_reference::worst(v[j], r.ref[j], r.mag[j]));
    return w;
}
double between(const Grads& a, const Grads& b, const ss_reference::Backward& r) {
    double w = 0;
    for (size_t j = 0; j < 6; ++j) w = std::max(w, ss_reference::worst(a[j], b[j], r.mag[j]));
    return w;
}
bool finite(const Grads& g) {
    for (const auto& v : g)
        for (float f : v)
            if (!std::isfinite(f)) return false;
    return true;
}
bool same_all(const Grads& a, const Grads& b, size_t upto) {
    for (size_t j = 0; j < upto; ++j)
        if (!same(a[j], b[j])) return false;
    return true;
}

// alternate a and b within each repetition after 3 warm-ups of each; medians in ms
template <class A, class B>
void paired_ms(A a, B b, int reps, double& ma, double& mb) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int r = 0; r < 3; ++r) a(), b();
    CK(cudaDeviceSynchronize());
    std::vector<float> ta, tb;
    auto one = [&](auto f, std::vector<float>& out) {
        CK(cudaEventRecord(e0));
        f();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float m;
        CK(cudaEventElapsedTime(&m, e0, e1));
        out.push_back(m);
    };
    for (int r = 0; r < reps; ++r) {
        if (r % 2)
            one(b, tb), one(a, ta);
        else
            one(a, ta), one(b, tb);
    }
    CK(cudaGetLastError());
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    auto med = [](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        return n % 2 ? double(v[n / 2]) : 0.5 * (double(v[n / 2 - 1]) + double(v[n / 2]));
    };
    ma = med(ta);
    mb = med(tb);
}

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    if (reps < 1) {
        std::fprintf(stderr, "reps must be >= 1\n");
        return 2;
    }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    // the forward bench's shapes: Mamba-130m (d_inner 1536, N 16) and Mamba-370m-like (d_inner
    // 2048), prefill lengths and batches
    const Shape shapes[] = {{512, 1, 1536, 16, true},  {2048, 1, 1536, 16, true},
                            {2048, 4, 1536, 16, true}, {4096, 1, 2048, 16, true},
                            {1024, 8, 768, 16, false}, {64, 16, 1536, 16, true}};
    const size_t chunks[] = {32, 64, 128, 256};
    bool all_ok = true;
    size_t rows = 0;
    for (const Shape& s : shapes) {
        const size_t T = s.sl * s.batch, DN = s.D * s.N, states = (T + s.batch) * DN;
        const Host h = make(s);
        const ss_reference::Shape rs{s.sl, s.batch, s.D, s.N, s.softplus, true};
        const ss_reference::Backward ref =
            ss_reference::backward(rs, ss_reference::widen(h.x, h.dt, h.B, h.C, h.A, h.skip, h.dy));
        Dev d{};
        d.x = up(h.x), d.dt = up(h.dt), d.B = up(h.B), d.C = up(h.C), d.A = up(h.A);
        d.skip = up(h.skip), d.dy = up(h.dy);
        d.y = zeros(T * s.D), d.S = zeros(states), d.delta = zeros(T * s.D);
        d.S2 = zeros(states), d.delta2 = zeros(T * s.D);
        const size_t n[6] = {T * s.D, T * s.D, T * s.N, T * s.N, DN, s.D};
        for (size_t j = 0; j < 6; ++j) d.n[j] = n[j], d.g[j] = zeros(n[j]);
        d.G = zeros(DN), d.history = zeros(T * DN);
        // the largest chunk scratch any L below needs (the smallest L here is 32)
        const ss_chunked::BwdSizes z = ss_chunked::bwd_sizes(ss_chunked::geom(T, s.D, s.N, s.sl, 32));
        d.end = zeros(z.items), d.prod = zeros(z.items), d.start = zeros(z.items);
        d.dA_part = zeros(z.items), d.hist = zeros(z.hist), d.dD_part = zeros(z.dD_parts);

        // the ordered forward's states and deltas: the backward's inputs
        ordered::k_selective_scan_fwd<<<grid(s.batch * s.D), kThreads>>>(
            d.x, d.dt, d.B, d.C, d.A, d.skip, d.y, d.S, d.delta, T, s.D, s.N, s.sl, s.softplus);
        CK(cudaGetLastError());
        prefill_nan(d);
        run_ordered(s, d);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        const Grads og = fetch(d);
        const double oe = vs_ref(og, ref);
        const bool ordered_ok = finite(og) && oe <= kBound;

        // one chunk per sequence: the ordered kernels' bits, on the device
        bool one_chunk = true;
        for (size_t L1 : {s.sl, s.sl + 5}) {
            prefill_nan(d);
            run_chunked(s, d, L1, d.S, d.delta);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            const Grads w = fetch(d);
            one_chunk = one_chunk && same_all(w, og, s.batch == 1 ? 6 : 4) &&
                        between(w, og, ref) <= kBound;
        }

        for (size_t L : chunks) {
            prefill_nan(d);  // every output must be rewritten
            run_chunked(s, d, L, d.S, d.delta);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            const Grads cg = fetch(d);
            prefill_nan(d);
            run_chunked(s, d, L, d.S, d.delta);
            CK(cudaDeviceSynchronize());
            const bool repeat = same_all(fetch(d), cg, 6);
            const bool written = finite(cg);
            const double ce = vs_ref(cg, ref), co = between(cg, og, ref);
            // the opt-in path end to end: the chunked forward's states into the chunked backward
            const ss_chunked::Geom gf = ss_chunked::geom(T, s.D, s.N, s.sl, L);
            ss_chunked::launch(gf, d.x, d.dt, d.B, d.C, d.A, d.skip, d.y, d.S2, d.delta2, d.end,
                               d.prod, d.start, s.softplus, grid, kThreads);
            prefill_nan(d);
            run_chunked(s, d, L, d.S2, d.delta2);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            const Grads pg = fetch(d);
            const double pe = vs_ref(pg, ref);
            const bool ok = ordered_ok && one_chunk && repeat && written && finite(pg) &&
                            ce <= kBound && co <= kBound && pe <= kBound;
            all_ok = all_ok && ok;
            double t_ord, t_ch;
            paired_ms([&] { run_ordered(s, d); }, [&] { run_chunked(s, d, L, d.S, d.delta); },
                      reps, t_ord, t_ch);
            ++rows;
            std::printf(
                "{\"gpu\": \"%s\", \"sl\": %zu, \"batch\": %zu, \"D\": %zu, \"N\": %zu, \"L\": "
                "%zu, \"ok\": %s, \"ordered_err\": %s, \"chunked_err\": %s, "
                "\"chunked_vs_ordered\": %s, \"pipeline_err\": %s, \"all_written\": %s, "
                "\"repeat_identical\": %s, \"one_chunk_identical_to_ordered\": %s, "
                "\"ordered_ms\": %s, \"chunked_ms\": %s, \"speedup\": %s, \"reps\": %d}\n",
                prop.name, s.sl, s.batch, s.D, s.N, L, ok ? "true" : "false", Js(oe).s, Js(ce).s,
                Js(co).s, Js(pe).s, written ? "true" : "false", repeat ? "true" : "false",
                one_chunk ? "true" : "false", Js(t_ord).s, Js(t_ch).s, Js(t_ord / t_ch).s, reps);
            std::fflush(stdout);
        }
        for (float* p : {d.x, d.dt, d.B, d.C, d.A, d.skip, d.dy, d.y, d.S, d.delta, d.S2, d.delta2,
                         d.G, d.history, d.end, d.prod, d.start, d.hist, d.dA_part, d.dD_part})
            cudaFree(p);
        for (float* p : d.g) cudaFree(p);
    }
    std::printf("{\"summary\": true, \"pass\": %s, \"rows\": %zu}\n", all_ok ? "true" : "false",
                rows);
    return all_ok ? 0 : 1;
}
