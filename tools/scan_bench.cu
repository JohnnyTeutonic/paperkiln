// Ordered against chunked selective-scan forward on one GPU, on device-resident buffers (no host transfers in
// the timed region). For each shape: both outputs against a binary64 recurrence on the host (the frozen
// 3e-5 normalised bound), chunked repeat bit-identity, then median kernel time over repetitions (CUDA events).
// This is a component timing of the scan forward, not a model-speed claim.
// build: nvcc -O3 -std=c++17 -arch=sm_89 -I src tools/scan_bench.cu -o scan_bench
// run:   ./scan_bench [reps=20]        (one JSON line per shape and chunk length, then a summary line)
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

// worst normalised error of y against a binary64 recurrence (states checked through y and the last state)
void reference_errors(const Shape& s, const Host& h, const std::vector<float>& y, const std::vector<float>& S,
                      double& ey, double& es) {
    const size_t D = s.D, N = s.N, DN = D * N;
    ey = es = 0;
    std::vector<double> st(DN);
    for (size_t q = 0; q < s.batch; ++q) {
        std::fill(st.begin(), st.end(), 0.0);
        for (size_t t = 0; t < s.sl; ++t) {
            const size_t row = q * s.sl + t;
            for (size_t c = 0; c < D; ++c) {
                const size_t cell = row * D + c;
                const double v = h.dt[cell], dl = s.softplus && v <= 20 ? std::log1p(std::exp(v)) : v;
                double acc = 0;
                for (size_t n = 0; n < N; ++n) {
                    const size_t i = c * N + n;
                    st[i] = st[i] * std::exp(dl * double(h.A[i])) + double(h.B[row * N + n]) * double(h.x[cell]) * dl;
                    acc += st[i] * h.C[row * N + n];
                }
                const double ref = acc + double(h.x[cell]) * h.skip[c];
                ey = std::max(ey, std::fabs(double(y[cell]) - ref) / std::max(1.0, std::fabs(ref)));
            }
        }
        const float* last = S.data() + (q * s.sl + q + s.sl) * DN;
        for (size_t i = 0; i < DN; ++i) es = std::max(es, std::fabs(double(last[i]) - st[i]) / std::max(1.0, std::fabs(st[i])));
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

template <class F>
double median_ms(F f, int reps) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<float> ms;
    for (int r = 0; r < 3; ++r) f();  // warm
    CK(cudaDeviceSynchronize());
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(a));
        f();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float m;
        CK(cudaEventElapsedTime(&m, a, b));
        ms.push_back(m);
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    // Mamba-130m (d_inner 1536, N 16) and Mamba-370m-like (d_inner 2048) shapes, prefill lengths and batches
    const Shape shapes[] = {{512, 1, 1536, 16, true}, {2048, 1, 1536, 16, true}, {2048, 4, 1536, 16, true},
                            {4096, 1, 2048, 16, true}, {1024, 8, 768, 16, false}, {64, 16, 1536, 16, true}};
    const size_t chunks[] = {32, 64, 128, 256};
    bool all_ok = true;
    for (const Shape& s : shapes) {
        const size_t T = s.sl * s.batch, DN = s.D * s.N, states = (T + s.batch) * DN;
        const Host h = make(s);
        Dev d{up(h.x), up(h.dt), up(h.B), up(h.C), up(h.A), up(h.skip), zeros(T * s.D), zeros(states), zeros(T * s.D)};
        run_ordered(s, d);
        CK(cudaDeviceSynchronize());
        const std::vector<float> oy = down(d.y, T * s.D), oS = down(d.S, states);
        double oey, oes;
        reference_errors(s, h, oy, oS, oey, oes);
        const double t_ord = median_ms([&] { run_ordered(s, d); }, reps);
        for (size_t L : chunks) {
            const size_t items = s.batch * ((s.sl + L - 1) / L) * DN;
            d.end = zeros(items), d.prod = zeros(items), d.start = zeros(items);
            CK(cudaMemset(d.S, 0xff, states * sizeof(float)));  // every state must be rewritten
            run_chunked(s, d, L);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            const std::vector<float> cy = down(d.y, T * s.D), cS = down(d.S, states);
            run_chunked(s, d, L);
            CK(cudaDeviceSynchronize());
            const std::vector<float> cy2 = down(d.y, T * s.D), cS2 = down(d.S, states);
            const bool repeat = std::memcmp(cy.data(), cy2.data(), cy.size() * 4) == 0 &&
                                std::memcmp(cS.data(), cS2.data(), cS.size() * 4) == 0;
            bool finite = true;
            double vs_ordered = 0;
            for (size_t i = 0; i < cS.size(); ++i) {
                finite = finite && std::isfinite(cS[i]);
                vs_ordered = std::max(vs_ordered, std::fabs(double(cS[i]) - oS[i]) / std::max(1.0, std::fabs(double(oS[i]))));
            }
            double cey, ces;
            reference_errors(s, h, cy, cS, cey, ces);
            const bool ok = repeat && finite && cey <= 3e-5 && ces <= 3e-5;
            all_ok = all_ok && ok && oey <= 3e-5 && oes <= 3e-5;
            const double t_ch = median_ms([&] { run_chunked(s, d, L); }, reps);
            std::printf("{\"gpu\": \"%s\", \"sl\": %zu, \"batch\": %zu, \"D\": %zu, \"N\": %zu, \"L\": %zu, \"ok\": %s, "
                        "\"ordered_err_y\": %.3g, \"ordered_err_last_state\": %.3g, \"chunked_err_y\": %.3g, "
                        "\"chunked_err_last_state\": %.3g, \"chunked_vs_ordered_states\": %.3g, \"repeat_identical\": %s, "
                        "\"ordered_ms\": %.4f, \"chunked_ms\": %.4f, \"speedup\": %.3f, \"reps\": %d}\n",
                        prop.name, s.sl, s.batch, s.D, s.N, L, ok ? "true" : "false", oey, oes, cey, ces, vs_ordered,
                        repeat ? "true" : "false", t_ord, t_ch, t_ord / t_ch, reps);
            std::fflush(stdout);
            cudaFree(d.end), cudaFree(d.prod), cudaFree(d.start);
        }
        for (float* p : {d.x, d.dt, d.B, d.C, d.A, d.skip, d.y, d.S, d.delta}) cudaFree(p);
    }
    std::printf("{\"summary\": true, \"pass\": %s}\n", all_ok ? "true" : "false");
    return all_ok ? 0 : 1;
}
