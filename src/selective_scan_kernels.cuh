// Included in cuda_ops.cu's anonymous namespace. No time-parallel scan or atomics.
// Explicit rounded multiply/add keeps the public host recurrence's reduction order.
__global__ void k_selective_scan_fwd(const float* x, const float* dt, const float* B,
                                     const float* C, const float* A, const float* skip, float* y,
                                     float* S, float* delta, size_t T, size_t D, size_t N,
                                     size_t sl, bool softplus) {
    const size_t DN = D * N, count = (T / sl) * D;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t q = idx / D, c = idx % D, r0 = q * sl;
        float* base = S + (r0 + q) * DN;
        for (size_t n = 0; n < N; ++n) base[c * N + n] = 0.0f;
        for (size_t t = 0; t < sl; ++t) {
            const size_t row = r0 + t, cell = row * D + c;
            const float v = dt[cell];
            const float dl = softplus && v <= 20.0f ? log1pf(expf(v)) : v;
            delta[cell] = dl;
            const float xd = __fmul_rn(x[cell], dl);
            float acc = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                const size_t i = c * N + n;
                const float a = expf(__fmul_rn(dl, A[i]));
                const float s =
                    __fadd_rn(__fmul_rn(base[t * DN + i], a), __fmul_rn(B[row * N + n], xd));
                base[(t + 1) * DN + i] = s;
                acc = __fadd_rn(acc, __fmul_rn(s, C[row * N + n]));
            }
            y[cell] = skip ? __fadd_rn(acc, __fmul_rn(x[cell], skip[c])) : acc;
        }
    }
}

// One channel owns all sequence/time contributions to its A and skip gradients.
// g_history is required only for B's channel reduction, and is never atomic.
__global__ void k_selective_scan_bwd_channels(const float* x, const float* dt, const float* B,
                                              const float* C, const float* A, const float* skip,
                                              const float* dy, const float* S, const float* delta,
                                              float* G, float* g_history, float* dx, float* ddt,
                                              float* dA, float* dD, size_t T, size_t D, size_t N,
                                              size_t sl, bool softplus) {
    const size_t DN = D * N;
    for (size_t c = blockIdx.x * blockDim.x + threadIdx.x; c < D;
         c += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (dA)
            for (size_t n = 0; n < N; ++n) dA[c * N + n] = 0.0f;
        float skip_gradient = 0.0f;
        for (size_t r0 = 0, q = 0; r0 < T; r0 += sl, ++q) {
            const float* base = S + (r0 + q) * DN;
            for (size_t n = 0; n < N; ++n) G[c * N + n] = 0.0f;
            for (size_t t = sl; t-- > 0;) {
                const size_t row = r0 + t, cell = row * D + c;
                const float dl = delta[cell], xv = x[cell], dyv = dy[cell];
                float ddelta = 0.0f, dxd = 0.0f;
                for (size_t n = 0; n < N; ++n) {
                    const size_t i = c * N + n;
                    const float g = __fadd_rn(__fmul_rn(dyv, C[row * N + n]), G[i]);
                    if (g_history) g_history[row * DN + i] = g;
                    const float Acn = A[i], a = expf(__fmul_rn(dl, Acn));
                    const float dpre = __fmul_rn(__fmul_rn(g, base[t * DN + i]), a);
                    if (dA) dA[i] = __fadd_rn(dA[i], __fmul_rn(dpre, dl));
                    ddelta = __fadd_rn(ddelta, __fmul_rn(dpre, Acn));
                    dxd = __fadd_rn(dxd, __fmul_rn(g, B[row * N + n]));
                    G[i] = __fmul_rn(g, a);
                }
                ddelta = __fadd_rn(ddelta, __fmul_rn(dxd, xv));
                if (dx)
                    dx[cell] = __fadd_rn(__fmul_rn(dxd, dl), skip ? __fmul_rn(dyv, skip[c]) : 0.0f);
                if (dD) skip_gradient = __fadd_rn(skip_gradient, __fmul_rn(dyv, xv));
                const float v = dt[cell];
                if (ddt)
                    ddt[cell] = softplus && v <= 20.0f
                                    ? __fmul_rn(ddelta, 1.0f / __fadd_rn(1.0f, expf(-v)))
                                    : ddelta;
            }
        }
        if (dD) dD[c] = skip_gradient;
    }
}

// Each B/C cell owns the ascending-channel sum used by the CPU loop.
__global__ void k_selective_scan_bwd_BC(const float* x, const float* dy, const float* S,
                                        const float* delta, const float* history, float* dB,
                                        float* dC, size_t T, size_t D, size_t N, size_t sl) {
    const size_t DN = D * N;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < T * N;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t row = idx / N, n = idx % N, q = row / sl, t = row % sl;
        const float* sn = S + (q * (sl + 1) + t + 1) * DN;
        float gb = 0.0f, gc = 0.0f;
        for (size_t c = 0; c < D; ++c) {
            const size_t cell = row * D + c, i = c * N + n;
            if (dB)
                gb = __fadd_rn(gb,
                               __fmul_rn(history[row * DN + i], __fmul_rn(x[cell], delta[cell])));
            if (dC) gc = __fadd_rn(gc, __fmul_rn(dy[cell], sn[i]));
        }
        if (dB) dB[idx] = gb;
        if (dC) dC[idx] = gc;
    }
}

bool selective_scan_geometry(const Matrix& x, const Matrix& dt, const Matrix& B, const Matrix& C,
                             const Matrix& A, const Matrix* skip, size_t seq_len,
                             const Matrix& states, const Matrix& delta, size_t& sl) {
    const size_t T = x.rows(), D = x.cols(), N = A.cols();
    sl = seq_len ? seq_len : T;
    return T && D && N && sl && T % sl == 0 && A.rows() == D && dt.rows() == T && dt.cols() == D &&
           B.rows() == T && B.cols() == N && C.rows() == T && C.cols() == N &&
           (!skip || (skip->rows() == 1 && skip->cols() == D)) &&
           states.rows() == (T + T / sl) * D && states.cols() == N && delta.rows() == T &&
           delta.cols() == D;
}
