// Chunked (parallel-in-time) selective-scan forward. Opt-in with MICROTORCH_SCAN_CHUNK=L (L >= 2) or
// MICROTORCH_SCAN_CHUNK=auto (L = 128; scan_chunk_switch.hpp); unset, the ordered kernels in
// selective_scan_kernels.cuh run unchanged.
//
// Per (sequence, channel c, state n) the recurrence is s_t = s_{t-1} a_t + b_t, with a_t =
// exp(delta_t A_cn) and b_t = B_tn (x_tc delta_t), s starting at zero for each sequence. Rows of a
// sequence are split into chunks of L rows (the last may be shorter):
//   1. local:   each chunk scans from zero and keeps its end value and the product of its a's;
//   2. carry:   one pass over a sequence's chunks gives each chunk its true start;
//   3. final:   each chunk rescans from its start and writes every state;
//   4. readout: y_tc = sum over ascending n of s_tcn C_tn, plus x_tc skip_c, as the ordered kernel.
// Steps 1, 3 and 4 run with (sequences x chunks x D x N) or (T x D) independent threads, where the
// ordered kernel has (sequences x D). Rounding differs from the ordered scan only through the chunk
// starts of step 2. Every multiply and add is explicitly rounded (no contraction), so results are
// deterministic.
//
// The same functions build the host emulation (selective_scan_chunked_host) that the host test
// runs, so the indexing and chunk boundaries the device executes are the ones the host test checks.
// Device numerics (CUDA expf/log1pf and the kernels) are checked on the GPU by tools/scan_bench.cu.
#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

#ifdef __CUDACC__
#define SS_HD __host__ __device__
#else
#define SS_HD
#endif

namespace ss_chunked {

SS_HD inline float mul(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fmul_rn(a, b);
#else
    volatile float r = a * b;  // rounded to binary32, never fused with a following add
    return r;
#endif
}
SS_HD inline float add(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fadd_rn(a, b);
#else
    volatile float r = a + b;
    return r;
#endif
}
SS_HD inline float ex(float v) {
#ifdef __CUDA_ARCH__
    return expf(v);
#else
    return std::exp(v);
#endif
}
SS_HD inline float softplus20(float v) {
#ifdef __CUDA_ARCH__
    return log1pf(expf(v));
#else
    return std::log1p(std::exp(v));
#endif
}

struct Geom {
    size_t T, D, N, sl, L, nchunk;  // nchunk = ceil(sl / L) chunks per sequence
};

// L is clamped to [1, sl]: any L >= sl is one chunk per sequence (the ordered recurrence), and the
// chunk count is computed without the overflow of (sl + L - 1) / L. sl must be >= 1 (the op's
// geometry check requires it).
SS_HD inline Geom geom(size_t T, size_t D, size_t N, size_t sl, size_t L) {
    const size_t Lc = L < 1 ? 1 : (L > sl ? sl : L);
    return Geom{T, D, N, sl, Lc, 1 + (sl - 1) / Lc};
}

// delta for one cell, as the ordered kernel computes it
SS_HD inline void delta_cell(const float* dt, float* delta, size_t cell, bool softplus) {
    const float v = dt[cell];
    delta[cell] = softplus && v <= 20.0f ? softplus20(v) : v;
}

// rows [r0, r1) of chunk k of sequence q
SS_HD inline void chunk_rows(const Geom& g, size_t q, size_t k, size_t& r0, size_t& r1) {
    r0 = q * g.sl + k * g.L;
    const size_t end = q * g.sl + g.sl;
    r1 = r0 + g.L < end ? r0 + g.L : end;
}

// work item index -> (q, k, c, n), n fastest so that consecutive threads write consecutive states
SS_HD inline void unpack(const Geom& g, size_t idx, size_t& q, size_t& k, size_t& c, size_t& n) {
    n = idx % g.N;
    idx /= g.N;
    c = idx % g.D;
    idx /= g.D;
    k = idx % g.nchunk;
    q = idx / g.nchunk;
}

// slot of (q, k, c, n) in the per-chunk buffers
SS_HD inline size_t slot(const Geom& g, size_t q, size_t k, size_t c, size_t n) {
    return ((q * g.nchunk + k) * g.D + c) * g.N + n;
}

// step 1: the chunk's scan from zero; its end value and the product of its decays
SS_HD inline void local(const Geom& g, const float* x, const float* delta, const float* B,
                        const float* A, size_t idx, float* end, float* prod) {
    size_t q, k, c, n, r0, r1;
    unpack(g, idx, q, k, c, n);
    chunk_rows(g, q, k, r0, r1);
    const float Acn = A[c * g.N + n];
    float s = 0.0f, p = 1.0f;
    for (size_t row = r0; row < r1; ++row) {
        const size_t cell = row * g.D + c;
        const float dl = delta[cell];
        const float a = ex(mul(dl, Acn));
        s = add(mul(s, a), mul(B[row * g.N + n], mul(x[cell], dl)));
        p = mul(p, a);
    }
    end[idx] = s;
    prod[idx] = p;
}

// step 2: true chunk starts for one (q, c, n), chunks in order
SS_HD inline void carry(const Geom& g, size_t qcn, const float* end, const float* prod,
                        float* start) {
    const size_t n = qcn % g.N, c = (qcn / g.N) % g.D, q = qcn / (g.N * g.D);
    float s = 0.0f;
    for (size_t k = 0; k < g.nchunk; ++k) {
        const size_t i = slot(g, q, k, c, n);
        start[i] = s;
        s = add(mul(prod[i], s), end[i]);
    }
}

// step 3: rescan the chunk from its start, writing the state after each row (S[(t + 1) D N] of
// sequence q, the ordered layout) and, for the first chunk, the zero state before row 0
SS_HD inline void final_states(const Geom& g, const float* x, const float* delta, const float* B,
                               const float* A, const float* start, size_t idx, float* S) {
    size_t q, k, c, n, r0, r1;
    unpack(g, idx, q, k, c, n);
    chunk_rows(g, q, k, r0, r1);
    const size_t DN = g.D * g.N, i = c * g.N + n;
    float* base = S + (q * g.sl + q) * DN;
    if (k == 0) base[i] = 0.0f;
    const float Acn = A[i];
    float s = start[idx];
    for (size_t row = r0; row < r1; ++row) {
        const size_t cell = row * g.D + c, t = row - q * g.sl;
        const float dl = delta[cell];
        const float a = ex(mul(dl, Acn));
        s = add(mul(s, a), mul(B[row * g.N + n], mul(x[cell], dl)));
        base[(t + 1) * DN + i] = s;
    }
}

// step 4: y for one (row, c), ascending n as the ordered kernel
SS_HD inline void readout(const Geom& g, const float* x, const float* C, const float* skip,
                          const float* S, size_t cell, float* y) {
    const size_t row = cell / g.D, c = cell % g.D, q = row / g.sl, t = row % g.sl, DN = g.D * g.N;
    const float* sn = S + (q * g.sl + q + t + 1) * DN + c * g.N;
    float acc = 0.0f;
    for (size_t n = 0; n < g.N; ++n) acc = add(acc, mul(sn[n], C[row * g.N + n]));
    y[cell] = skip ? add(acc, mul(x[cell], skip[c])) : acc;
}

#ifdef __CUDACC__
__global__ void k_delta(const float* dt, float* delta, size_t cells, bool softplus) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < cells;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        delta_cell(dt, delta, i, softplus);
}
__global__ void k_local(Geom g, const float* x, const float* delta, const float* B, const float* A,
                        float* end, float* prod, size_t items) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < items;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        local(g, x, delta, B, A, i, end, prod);
}
__global__ void k_carry(Geom g, const float* end, const float* prod, float* start, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        carry(g, i, end, prod, start);
}
__global__ void k_final(Geom g, const float* x, const float* delta, const float* B, const float* A,
                        const float* start, float* S, size_t items) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < items;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        final_states(g, x, delta, B, A, start, i, S);
}
__global__ void k_readout(Geom g, const float* x, const float* C, const float* skip, const float* S,
                          float* y, size_t cells) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < cells;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        readout(g, x, C, skip, S, i, y);
}

// The four steps on the device. end, prod and start hold (T / sl) nchunk D N floats each; grid(n)
// gives the block count for n grid-stride work items.
template <class Grid>
inline void launch(const Geom& g, const float* x, const float* dt, const float* B, const float* C,
                   const float* A, const float* skip, float* y, float* S, float* delta, float* end,
                   float* prod, float* start, bool softplus, Grid grid, int threads) {
    const size_t cells = g.T * g.D, items = (g.T / g.sl) * g.nchunk * g.D * g.N,
                 carries = (g.T / g.sl) * g.D * g.N;
    k_delta<<<grid(cells), threads>>>(dt, delta, cells, softplus);
    k_local<<<grid(items), threads>>>(g, x, delta, B, A, end, prod, items);
    k_carry<<<grid(carries), threads>>>(g, end, prod, start, carries);
    k_final<<<grid(items), threads>>>(g, x, delta, B, A, start, S, items);
    k_readout<<<grid(cells), threads>>>(g, x, C, skip, S, y, cells);
}
#endif

// Host emulation of the four device steps, in the same phases. S must hold (T + T / sl) D N floats.
inline void selective_scan_chunked_host(const float* x, const float* dt, const float* B,
                                        const float* C, const float* A, const float* skip, float* y,
                                        float* S, float* delta, size_t T, size_t D, size_t N,
                                        size_t sl, size_t L, bool softplus) {
    const Geom g = geom(T, D, N, sl, L);
    const size_t items = (T / sl) * g.nchunk * D * N;
    std::vector<float> end(items), prod(items), start(items);
    for (size_t cell = 0; cell < T * D; ++cell) delta_cell(dt, delta, cell, softplus);
    for (size_t i = 0; i < items; ++i) local(g, x, delta, B, A, i, end.data(), prod.data());
    for (size_t qcn = 0; qcn < (T / sl) * D * N; ++qcn)
        carry(g, qcn, end.data(), prod.data(), start.data());
    for (size_t i = 0; i < items; ++i) final_states(g, x, delta, B, A, start.data(), i, S);
    for (size_t cell = 0; cell < T * D; ++cell) readout(g, x, C, skip, S, cell, y);
}

}  // namespace ss_chunked
