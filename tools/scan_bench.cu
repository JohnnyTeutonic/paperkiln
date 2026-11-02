// Ordered against chunked selective-scan forward on one GPU, on device-resident buffers (no host transfers in
// the timed region). For each shape:
//   - the ordered kernel and the chunked one (for each L) against a binary64 recurrence on the host: y and
//     every state, |v - ref| / max(1, |ref|) <= 3e-5, any nonfinite value a failure;
//   - every state slot written (pre-filled with NaN) and finite; repeat runs bit-identical;
//   - one chunk per sequence (L = sl and L > sl) bit-identical to the ordered kernel on the device;
//   - timings: ordered and chunked alternate within each repetition; median of the per-repetition elapsed stream
//     time (CUDA events on the default stream; for the chunked path this spans its five launches, including any
//     idle gaps between them). Chunk buffers are allocated outside the timed region.
// A component timing of the scan forward, not a model-speed claim.
// build: nvcc -O3 -std=c++17 -arch=sm_89 -I src tools/scan_bench.cu -o scan_bench
// run:   ./scan_bench [reps=20]        (one JSON line per shape and L, then a summary line; exit 1 on any failure)
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
#include "selective_scan_chunked.cuh"

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                                  \
        }                                                                                  \
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
float unit(uint64_t seed, size_t i) { return float(double(mix(seed * 1000003ULL + i) >> 11) / 9007199254740992.0 * 2 - 1); }

struct Shape {
    size_t sl, batch, D, N;
    bool softplus;
};

struct Host {
    std::vector<float> x, dt, B, C, A, skip;
};

Host make(const Shape& s) {
    const size_t T = s.sl * s.batch;
    Host h;
    for (size_t i = 0; i < T * s.D; ++i) h.x.push_back(unit(1, i));
    for (size_t i = 0; i < T * s.D; ++i) h.dt.push_back(s.softplus ? -4.0f + 3.0f * unit(2, i) : 0.02f * (unit(2, i) + 1.1f));
    for (size_t i = 0; i < T * s.N; ++i) h.B.push_back(unit(3, i));
    for (size_t i = 0; i < T * s.N; ++i) h.C.push_back(unit(4, i));
    for (size_t i = 0; i < s.D * s.N; ++i) h.A.push_back(-std::exp(float(1.5 * unit(5, i))));
    for (size_t i = 0; i < s.D; ++i) h.skip.push_back(unit(6, i));
    return h;
}

// a JSON number, or null for a nonfinite value (which has already failed its gate)
struct Js {
    char s[32];
    explicit Js(double v) {
        if (std::isfinite(v)) std::snprintf(s, sizeof s, "%.4g", v);
        else std::snprintf(s, sizeof s, "null");
    }
};

double err(double v, double ref) {
    const double e = std::fabs(v - ref) / std::max(1.0, std::fabs(ref));
    return std::isfinite(e) ? e : INFINITY;  // a nonfinite value or reference fails the bound
}

// worst errors of y and of every state (the zero start included) against a binary64 recurrence
void reference_errors(const Shape& s, const Host& h, const std::vector<float>& y, const std::vector<float>& S,
                      double& ey, double& es) {
    const size_t D = s.D, N = s.N, DN = D * N;
    ey = es = 0;
    std::vector<double> st(DN);
    for (size_t q = 0; q < s.batch; ++q) {
        std::fill(st.begin(), st.end(), 0.0);
        const float* base = S.data() + (q * s.sl + q) * DN;
        for (size_t i = 0; i < DN; ++i) es = std::max(es, err(base[i], 0.0));
        for (size_t t = 0; t < s.sl; ++t) {
            const size_t row = q * s.sl + t;
            for (size_t c = 0; c < D; ++c) {
                const size_t cell = row * D + c;
                const double v = h.dt[cell], dl = s.softplus && v <= 20 ? std::log1p(std::exp(v)) : v;
                double acc = 0;
                for (size_t n = 0; n < N; ++n) {
                    const size_t i = c * N + n;
                    st[i] = st[i] * std::exp(dl * double(h.A[i])) + double(h.B[row * N + n]) * double(h.x[cell]) * dl;
                    es = std::max(es, err(base[(t + 1) * DN + i], st[i]));
                    acc += st[i] * h.C[row * N + n];
                }
                ey = std::max(ey, err(y[cell], acc + double(h.x[cell]) * h.skip[c]));
            }
        }
    }
}

struct Dev {
    float *x, *dt, *B, *C, *A, *skip, *y, *S, *delta, *end = nullptr, *prod = nullptr, *start = nullptr;
};

float* up(const std::vector<float>& v) {
    float* d;
    CK(cudaMalloc(&d, v.size() * sizeof(float)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}
float* zeros(size_t n) {
    float* d;
    CK(cudaMalloc(&d, n * sizeof(float)));
    CK(cudaMemset(d, 0, n * sizeof(float)));
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

void run_ordered(const Shape& s, const Dev& d) {
    const size_t T = s.sl * s.batch;
    ordered::k_selective_scan_fwd<<<grid(s.batch * s.D), kThreads>>>(d.x, d.dt, d.B, d.C, d.A, d.skip, d.y, d.S,
                                                                     d.delta, T, s.D, s.N, s.sl, s.softplus);
}
void run_chunked(const Shape& s, const Dev& d, size_t L) {
    const ss_chunked::Geom g = ss_chunked::geom(s.sl * s.batch, s.D, s.N, s.sl, L);
    ss_chunked::launch(g, d.x, d.dt, d.B, d.C, d.A, d.skip, d.y, d.S, d.delta, d.end, d.prod, d.start, s.softplus,
                       grid, kThreads);
}
void prefill_nan(const Dev& d, size_t states) { CK(cudaMemset(d.S, 0xff, states * sizeof(float))); }

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
        if (r % 2) one(b, tb), one(a, ta);
        else one(a, ta), one(b, tb);
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
    // Mamba-130m (d_inner 1536, N 16) and Mamba-370m-like (d_inner 2048) shapes, prefill lengths and batches
    const Shape shapes[] = {{512, 1, 1536, 16, true}, {2048, 1, 1536, 16, true}, {2048, 4, 1536, 16, true},
                            {4096, 1, 2048, 16, true}, {1024, 8, 768, 16, false}, {64, 16, 1536, 16, true}};
    const size_t chunks[] = {32, 64, 128, 256};
    bool all_ok = true;
    size_t rows = 0;
    for (const Shape& s : shapes) {
        const size_t T = s.sl * s.batch, DN = s.D * s.N, states = (T + s.batch) * DN;
        const Host h = make(s);
        Dev d{up(h.x), up(h.dt), up(h.B), up(h.C), up(h.A), up(h.skip), zeros(T * s.D), zeros(states), zeros(T * s.D)};
        // the largest chunk buffers any L below needs (L = 1 is not timed; the smallest L here is 32)
        const size_t max_items = s.batch * ss_chunked::geom(T, s.D, s.N, s.sl, 32).nchunk * DN;
        d.end = zeros(max_items), d.prod = zeros(max_items), d.start = zeros(max_items);

        prefill_nan(d, states);
        run_ordered(s, d);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        const std::vector<float> oy = down(d.y, T * s.D), oS = down(d.S, states);
        double oey, oes;
        reference_errors(s, h, oy, oS, oey, oes);
        const bool ordered_ok = oey <= kBound && oes <= kBound;

        // one chunk per sequence: the ordered kernel's bits, on the device
        bool one_chunk = true;
        for (size_t L1 : {s.sl, s.sl + 5}) {
            prefill_nan(d, states);
            run_chunked(s, d, L1);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            one_chunk = one_chunk && same(down(d.y, T * s.D), oy) && same(down(d.S, states), oS);
        }

        for (size_t L : chunks) {
            prefill_nan(d, states);  // every state must be rewritten
            run_chunked(s, d, L);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            const std::vector<float> cy = down(d.y, T * s.D), cS = down(d.S, states);
            prefill_nan(d, states);
            run_chunked(s, d, L);
            CK(cudaDeviceSynchronize());
            const bool repeat = same(down(d.y, T * s.D), cy) && same(down(d.S, states), cS);
            double vs_ordered = 0;
            for (size_t i = 0; i < cS.size(); ++i) vs_ordered = std::max(vs_ordered, err(cS[i], oS[i]));
            double cey, ces;
            reference_errors(s, h, cy, cS, cey, ces);
            const bool ok = ordered_ok && one_chunk && repeat && cey <= kBound && ces <= kBound;
            all_ok = all_ok && ok;
            double t_ord, t_ch;
            paired_ms([&] { run_ordered(s, d); }, [&] { run_chunked(s, d, L); }, reps, t_ord, t_ch);
            ++rows;
            std::printf("{\"gpu\": \"%s\", \"sl\": %zu, \"batch\": %zu, \"D\": %zu, \"N\": %zu, \"L\": %zu, \"ok\": %s, "
                        "\"ordered_err_y\": %s, \"ordered_err_states\": %s, \"chunked_err_y\": %s, "
                        "\"chunked_err_states\": %s, \"chunked_vs_ordered_states\": %s, \"repeat_identical\": %s, "
                        "\"one_chunk_identical_to_ordered\": %s, \"ordered_ms\": %s, \"chunked_ms\": %s, "
                        "\"speedup\": %s, \"reps\": %d}\n",
                        prop.name, s.sl, s.batch, s.D, s.N, L, ok ? "true" : "false", Js(oey).s, Js(oes).s, Js(cey).s,
                        Js(ces).s, Js(vs_ordered).s, repeat ? "true" : "false", one_chunk ? "true" : "false",
                        Js(t_ord).s, Js(t_ch).s, Js(t_ord / t_ch).s, reps);
            std::fflush(stdout);
        }
        for (float* p : {d.x, d.dt, d.B, d.C, d.A, d.skip, d.y, d.S, d.delta, d.end, d.prod, d.start}) cudaFree(p);
    }
    std::printf("{\"summary\": true, \"pass\": %s, \"rows\": %zu}\n", all_ok ? "true" : "false", rows);
    return all_ok ? 0 : 1;
}
