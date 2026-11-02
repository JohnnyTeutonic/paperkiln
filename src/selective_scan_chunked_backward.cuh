// Chunked (parallel-in-time) selective-scan backward. Opt-in with MICROTORCH_SCAN_CHUNK=L (L >= 1),
// the same switch as the chunked forward (selective_scan_chunked.cuh); unset, the ordered kernels
// in selective_scan_kernels.cuh run unchanged. Design, recurrences and tolerances:
// experiments/mamba_parallel_scan_2026-11-03/BACKWARD_DESIGN.md.
//
// Per (sequence, channel c, state n) the adjoint g_t = d loss / d s_t satisfies, backward in time,
//   g_t = e_t + G_{t+1},   G_{t+1} = g_{t+1} a_{t+1},   e_t = dy_tc C_tn,   G after the last row 0,
// a linear recurrence of the forward's form run in reverse. Rows of a sequence are split into the
// forward's chunks of L rows (the last may be shorter):
//   1. local:   each chunk scans backward from G = 0; it keeps the G it passes to the chunk before
//               (G after its first row) and the product of its a's;
//   2. carry:   one pass over a sequence's chunks, last to first, gives each chunk its true
//   incoming G;
//   3. final:   each chunk rescans backward from that G, writes g_t for every row (the history) and
//               its partial sum of dA's terms;
//   4. reduce:  dA (per c, n) and dD (per c) add their chunk partials in sequence order, chunks
//   last
//               to first, as the ordered kernel visits rows;
//   5. cell:    dx and ddt per (row, c) from the history, ascending n, as the ordered kernel;
//   6. B/C:     dB and dC per (row, n), ascending c, as the ordered kernel.
// The forward states S (and delta) are read, never recomputed; the decays a_t are recomputed from
// delta and A wherever they are needed, with the forward's exact expression. Given the same g,
// steps 5 and 6 are the ordered kernel's arithmetic, so rounding differs from the ordered backward
// only through the chunk boundaries of step 2 (the g's) and the regrouped sums of step 4 (dA and
// dD). With one chunk per sequence (L >= sl) g, dx, ddt, dB and dC are the ordered kernel's bits,
// and so are dA and dD for a single sequence. Every multiply and add is explicitly rounded; results
// are deterministic.
//
// The same functions build the host emulation (selective_scan_chunked_backward_host) that
// tests/test_selective_scan_chunked_backward.cpp runs. Device numerics are checked on the GPU by
// tools/scan_bwd_bench.cu.
#pragma once
#include "selective_scan_chunked.cuh"

namespace ss_chunked {

// step 1: the chunk's backward scan from G = 0; the G it passes on and the product of its decays
SS_HD inline void bwd_local(const Geom& g, const float* dy, const float* C, const float* delta,
                            const float* A, size_t idx, float* end, float* prod) {
    size_t q, k, c, n, r0, r1;
    unpack(g, idx, q, k, c, n);
    chunk_rows(g, q, k, r0, r1);
    const float Acn = A[c * g.N + n];
    float G = 0.0f, p = 1.0f;
    for (size_t row = r1; row-- > r0;) {
        const size_t cell = row * g.D + c;
        const float a = ex(mul(delta[cell], Acn));
        const float gt = add(mul(dy[cell], C[row * g.N + n]), G);
        G = mul(gt, a);
        p = mul(p, a);
    }
    end[idx] = G;
    prod[idx] = p;
}

// step 2: the true incoming G of each chunk for one (q, c, n), chunks last to first
SS_HD inline void bwd_carry(const Geom& g, size_t qcn, const float* end, const float* prod,
                            float* start) {
    const size_t n = qcn % g.N, c = (qcn / g.N) % g.D, q = qcn / (g.N * g.D);
    float G = 0.0f;
    for (size_t k = g.nchunk; k-- > 0;) {
        const size_t i = slot(g, q, k, c, n);
        start[i] = G;
        G = add(mul(prod[i], G), end[i]);
    }
}

// step 3: rescan the chunk backward from its incoming G; g_t to hist[row D N + c N + n] (the
// ordered kernel's g_history layout) and, if dA_part, the chunk's partial sum of dA's terms, rows
// descending
SS_HD inline void bwd_final(const Geom& g, const float* dy, const float* C, const float* delta,
                            const float* A, const float* S, const float* start, size_t idx,
                            float* hist, float* dA_part) {
    size_t q, k, c, n, r0, r1;
    unpack(g, idx, q, k, c, n);
    chunk_rows(g, q, k, r0, r1);
    const size_t DN = g.D * g.N, i = c * g.N + n;
    const float* base = S + (q * g.sl + q) * DN;
    const float Acn = A[i];
    float G = start[idx], da = 0.0f;
    for (size_t row = r1; row-- > r0;) {
        const size_t cell = row * g.D + c, t = row - q * g.sl;
        const float dl = delta[cell];
        const float gt = add(mul(dy[cell], C[row * g.N + n]), G);
        hist[row * DN + i] = gt;
        const float a = ex(mul(dl, Acn));
        const float dpre = mul(mul(gt, base[t * DN + i]), a);
        da = add(da, mul(dpre, dl));
        G = mul(gt, a);
    }
    if (dA_part) dA_part[idx] = da;
}

// dD's partial sum over chunk k of sequence q for channel c (work item (q k) D + c), rows
// descending
SS_HD inline void bwd_skip_part(const Geom& g, const float* dy, const float* x, size_t qkc,
                                float* dD_part) {
    const size_t c = qkc % g.D, k = (qkc / g.D) % g.nchunk, q = qkc / (g.D * g.nchunk);
    size_t r0, r1;
    chunk_rows(g, q, k, r0, r1);
    float s = 0.0f;
    for (size_t row = r1; row-- > r0;) s = add(s, mul(dy[row * g.D + c], x[row * g.D + c]));
    dD_part[qkc] = s;
}

// step 4: dA for one (c, n) from its chunk partials, sequences ascending, chunks descending
SS_HD inline void bwd_reduce_A(const Geom& g, size_t cn, const float* dA_part, float* dA) {
    const size_t n = cn % g.N, c = cn / g.N;
    float v = 0.0f;
    for (size_t q = 0; q < g.T / g.sl; ++q)
        for (size_t k = g.nchunk; k-- > 0;) v = add(v, dA_part[slot(g, q, k, c, n)]);
    dA[cn] = v;
}
SS_HD inline void bwd_reduce_D(const Geom& g, size_t c, const float* dD_part, float* dD) {
    float v = 0.0f;
    for (size_t q = 0; q < g.T / g.sl; ++q)
        for (size_t k = g.nchunk; k-- > 0;) v = add(v, dD_part[(q * g.nchunk + k) * g.D + c]);
    dD[c] = v;
}

// step 5: dx and ddt for one (row, c), ascending n, the ordered kernel's arithmetic given g
SS_HD inline void bwd_cell(const Geom& g, const float* x, const float* dt, const float* B,
                           const float* A, const float* skip, const float* dy, const float* S,
                           const float* delta, const float* hist, size_t cell, float* dx,
                           float* ddt, bool softplus) {
    const size_t row = cell / g.D, c = cell % g.D, q = row / g.sl, t = row % g.sl, DN = g.D * g.N;
    const float* sp = S + (q * g.sl + q + t) * DN;  // the state before row t
    const float dl = delta[cell], xv = x[cell], dyv = dy[cell];
    float ddelta = 0.0f, dxd = 0.0f;
    for (size_t n = 0; n < g.N; ++n) {
        const size_t i = c * g.N + n;
        const float gt = hist[row * DN + i];
        const float Acn = A[i], a = ex(mul(dl, Acn));
        const float dpre = mul(mul(gt, sp[i]), a);
        ddelta = add(ddelta, mul(dpre, Acn));
        dxd = add(dxd, mul(gt, B[row * g.N + n]));
    }
    ddelta = add(ddelta, mul(dxd, xv));
    if (dx) dx[cell] = add(mul(dxd, dl), skip ? mul(dyv, skip[c]) : 0.0f);
    const float v = dt[cell];
    if (ddt) ddt[cell] = softplus && v <= 20.0f ? mul(ddelta, 1.0f / add(1.0f, ex(-v))) : ddelta;
}

// step 6: dB and dC for one (row, n), ascending c, the ordered kernel's arithmetic given g
SS_HD inline void bwd_bc(const Geom& g, const float* x, const float* dy, const float* S,
                         const float* delta, const float* hist, size_t idx, float* dB, float* dC) {
    const size_t DN = g.D * g.N, row = idx / g.N, n = idx % g.N, q = row / g.sl, t = row % g.sl;
    const float* sn = S + (q * (g.sl + 1) + t + 1) * DN;
    float gb = 0.0f, gc = 0.0f;
    for (size_t c = 0; c < g.D; ++c) {
        const size_t cell = row * g.D + c, i = c * g.N + n;
        if (dB) gb = add(gb, mul(hist[row * DN + i], mul(x[cell], delta[cell])));
        if (dC) gc = add(gc, mul(dy[cell], sn[i]));
    }
    if (dB) dB[idx] = gb;
    if (dC) dC[idx] = gc;
}

// Scratch the chunked backward needs, in floats: end, prod and start (items each), the history
// (T D N), dA partials (items) and dD partials ((T / sl) nchunk D).
struct BwdSizes {
    size_t items, hist, dD_parts;
};
SS_HD inline BwdSizes bwd_sizes(const Geom& g) {
    const size_t seqs = g.T / g.sl;
    return BwdSizes{seqs * g.nchunk * g.D * g.N, g.T * g.D * g.N, seqs * g.nchunk * g.D};
}

#ifdef __CUDACC__
__global__ void k_bwd_local(Geom g, const float* dy, const float* C, const float* delta,
                            const float* A, float* end, float* prod, size_t items) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < items;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        bwd_local(g, dy, C, delta, A, i, end, prod);
}
__global__ void k_bwd_carry(Geom g, const float* end, const float* prod, float* start,
                            size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        bwd_carry(g, i, end, prod, start);
}
__global__ void k_bwd_final(Geom g, const float* dy, const float* C, const float* delta,
                            const float* A, const float* S, const float* start, float* hist,
                            float* dA_part, size_t items) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < items;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        bwd_final(g, dy, C, delta, A, S, start, i, hist, dA_part);
}
__global__ void k_bwd_skip_part(Geom g, const float* dy, const float* x, float* dD_part,
                                size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        bwd_skip_part(g, dy, x, i, dD_part);
}
__global__ void k_bwd_reduce_A(Geom g, const float* dA_part, float* dA, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        bwd_reduce_A(g, i, dA_part, dA);
}
__global__ void k_bwd_reduce_D(Geom g, const float* dD_part, float* dD, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        bwd_reduce_D(g, i, dD_part, dD);
}
__global__ void k_bwd_cell(Geom g, const float* x, const float* dt, const float* B, const float* A,
                           const float* skip, const float* dy, const float* S, const float* delta,
                           const float* hist, float* dx, float* ddt, bool softplus, size_t cells) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < cells;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        bwd_cell(g, x, dt, B, A, skip, dy, S, delta, hist, i, dx, ddt, softplus);
}
__global__ void k_bwd_bc(Geom g, const float* x, const float* dy, const float* S,
                         const float* delta, const float* hist, float* dB, float* dC,
                         size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        bwd_bc(g, x, dy, S, delta, hist, i, dB, dC);
}

// The backward steps on the device. Any of dx, ddt, dB, dC, dA, dD may be null (not wanted); dD
// needs skip. Scratch sizes from bwd_sizes: end, prod, start and dA_part hold items floats, hist
// T D N, dD_part dD_parts; dA_part may be null without dA, dD_part without dD, and end, prod,
// start and hist without dx, ddt, dB and dA (dC and dD need no adjoint). grid(n) gives the block
// count for n grid-stride work items.
template <class Grid>
inline void launch_backward(const Geom& g, const float* x, const float* dt, const float* B,
                            const float* C, const float* A, const float* skip, const float* dy,
                            const float* S, const float* delta, float* dx, float* ddt, float* dB,
                            float* dC, float* dA, float* dD, float* end, float* prod, float* start,
                            float* hist, float* dA_part, float* dD_part, bool softplus, Grid grid,
                            int threads) {
    const BwdSizes z = bwd_sizes(g);
    const size_t cells = g.T * g.D, carries = (g.T / g.sl) * g.D * g.N;
    if (dx || ddt || dB || dA) {
        k_bwd_local<<<grid(z.items), threads>>>(g, dy, C, delta, A, end, prod, z.items);
        k_bwd_carry<<<grid(carries), threads>>>(g, end, prod, start, carries);
        k_bwd_final<<<grid(z.items), threads>>>(g, dy, C, delta, A, S, start, hist,
                                                dA ? dA_part : nullptr, z.items);
    }
    if (dA) k_bwd_reduce_A<<<grid(g.D * g.N), threads>>>(g, dA_part, dA, g.D * g.N);
    if (dD) {
        k_bwd_skip_part<<<grid(z.dD_parts), threads>>>(g, dy, x, dD_part, z.dD_parts);
        k_bwd_reduce_D<<<grid(g.D), threads>>>(g, dD_part, dD, g.D);
    }
    if (dx || ddt)
        k_bwd_cell<<<grid(cells), threads>>>(g, x, dt, B, A, skip, dy, S, delta, hist, dx, ddt,
                                             softplus, cells);
    if (dB || dC)
        k_bwd_bc<<<grid(g.T * g.N), threads>>>(g, x, dy, S, delta, hist, dB, dC, g.T * g.N);
}
#endif

// Host emulation of the device steps, in the same phases. S and delta are the forward's (S holds
// (T + T / sl) D N floats). Null outputs are not computed, as on the device; dD needs skip.
inline void selective_scan_chunked_backward_host(const float* x, const float* dt, const float* B,
                                                 const float* C, const float* A, const float* skip,
                                                 const float* dy, const float* S,
                                                 const float* delta, size_t T, size_t D, size_t N,
                                                 size_t sl, size_t L, bool softplus, float* dx,
                                                 float* ddt, float* dB, float* dC, float* dA,
                                                 float* dD) {
    const Geom g = geom(T, D, N, sl, L);
    const BwdSizes z = bwd_sizes(g);
    const bool adjoint = dx || ddt || dB || dA;
    std::vector<float> end, prod, start, hist, dA_part, dD_part;
    if (adjoint) {
        end.resize(z.items), prod.resize(z.items), start.resize(z.items), hist.resize(z.hist);
        if (dA) dA_part.resize(z.items);
        for (size_t i = 0; i < z.items; ++i)
            bwd_local(g, dy, C, delta, A, i, end.data(), prod.data());
        for (size_t qcn = 0; qcn < (T / sl) * D * N; ++qcn)
            bwd_carry(g, qcn, end.data(), prod.data(), start.data());
        for (size_t i = 0; i < z.items; ++i)
            bwd_final(g, dy, C, delta, A, S, start.data(), i, hist.data(),
                      dA ? dA_part.data() : nullptr);
    }
    if (dA)
        for (size_t cn = 0; cn < D * N; ++cn) bwd_reduce_A(g, cn, dA_part.data(), dA);
    if (dD) {
        dD_part.resize(z.dD_parts);
        for (size_t i = 0; i < z.dD_parts; ++i) bwd_skip_part(g, dy, x, i, dD_part.data());
        for (size_t c = 0; c < D; ++c) bwd_reduce_D(g, c, dD_part.data(), dD);
    }
    if (dx || ddt)
        for (size_t cell = 0; cell < T * D; ++cell)
            bwd_cell(g, x, dt, B, A, skip, dy, S, delta, hist.data(), cell, dx, ddt, softplus);
    if (dB || dC)
        for (size_t i = 0; i < T * N; ++i) bwd_bc(g, x, dy, S, delta, hist.data(), i, dB, dC);
}

}  // namespace ss_chunked
