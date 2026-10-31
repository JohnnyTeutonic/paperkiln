// Included in cuda_ops.cu's devops anonymous namespace. Every element owns
// its complete reduction; neither backward nor forward uses atomics.
__global__ void k_conv1d_causal(const float* x, const float* w, const float* b, float* y,
                               size_t T, size_t C, size_t K, size_t sl) {
    const size_t n = T * C;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t row = idx / C, c = idx % C, t = row % sl;
        const size_t padding = t < K - 1 ? K - 1 - t : 0;
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            const float xv = k >= padding ? x[(row - (K - 1 - k)) * C + c] : 0.0f;
            // Padding products are still evaluated, including nonfinite w*0.
            acc = __fadd_rn(acc, __fmul_rn(w[c * K + k], xv));
        }
        y[idx] = b ? __fadd_rn(acc, b[c]) : acc;
    }
}

__global__ void k_conv1d_causal_dx(const float* w, const float* dy, float* dx,
                                  size_t T, size_t C, size_t K, size_t sl) {
    const size_t n = T * C;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t row = idx / C, c = idx % C, u = row % sl;
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            const size_t shift = K - 1 - k;
            if (shift < sl - u)
                acc = __fadd_rn(acc, __fmul_rn(dy[(row + shift) * C + c], w[c * K + k]));
        }
        dx[idx] = acc;
    }
}

__global__ void k_conv1d_causal_dw(const float* x, const float* dy, float* dw,
                                  size_t T, size_t C, size_t K, size_t sl) {
    const size_t n = C * K;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t c = idx / K, k = idx % K, shift = K - 1 - k;
        float acc = 0.0f;
        for (size_t row = 0; row < T; ++row)
            if (row % sl >= shift)
                acc = __fadd_rn(acc, __fmul_rn(dy[row * C + c], x[(row - shift) * C + c]));
        dw[idx] = acc;
    }
}

__global__ void k_conv1d_causal_db(const float* dy, float* db, size_t T, size_t C) {
    for (size_t c = blockIdx.x * blockDim.x + threadIdx.x; c < C;
         c += static_cast<size_t>(gridDim.x) * blockDim.x) {
        float acc = 0.0f;
        for (size_t row = 0; row < T; ++row) acc = __fadd_rn(acc, dy[row * C + c]);
        db[c] = acc;
    }
}

bool conv1d_geometry(const Matrix& x, const Matrix& w, size_t seq_len, size_t& sl) {
    const size_t T = x.rows(), C = x.cols(), K = w.cols();
    sl = seq_len ? seq_len : T;
    return T && C && K >= 2 && w.rows() == C && sl && T % sl == 0 &&
           C <= std::numeric_limits<size_t>::max() / T && K <= std::numeric_limits<size_t>::max() / C;
}
