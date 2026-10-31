// Exact kernel region from public revision 87ccf28cf src/cuda_resident.cu; names only changed for test isolation.
// Transpose-flag GEMM: logical C(M,N) = op(A) * op(B). TA/TB pick the
// index math (TA: A stored (K,M); TB: B stored (N,K)). Same 32x32
// shared-memory tiling as gemm_tiled; the transposed loads are strided
// rather than coalesced — correctness first, coalescing is a measured
// optimization for later (docs/CUDA_PHASE_B2.md non-goals).
template <bool TA, bool TB>
__global__ void old_gemm_tiled_ops(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int ak = t * TILE + threadIdx.x;
        const int bk = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && ak < K)
                ? (TA ? A[static_cast<size_t>(ak) * M + row]
                      : A[static_cast<size_t>(row) * K + ak])
                : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (bk < K && col < N)
                ? (TB ? B[static_cast<size_t>(col) * K + bk]
                      : B[static_cast<size_t>(bk) * N + col])
                : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// C(M,N) += op(A) * op(B): old_gemm_tiled_ops with the accumulator starting at
// C's current value instead of 0, so a product split along K into chunks
// continues each element's FMA chain where the previous chunk left it. With
// chunks of whole tiles (a multiple of TILE rows) it is old_gemm_tiled_ops over
// the whole K bit for bit; otherwise the padded zeros in between only add
// +0 (ops::linear_cross_entropy's dW, 26 Oct 2026).
template <bool TA, bool TB>
__global__ void old_gemm_tiled_acc(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = (row < M && col < N) ? C[static_cast<size_t>(row) * N + col] : 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int ak = t * TILE + threadIdx.x;
        const int bk = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && ak < K)
                ? (TA ? A[static_cast<size_t>(ak) * M + row]
                      : A[static_cast<size_t>(row) * K + ak])
                : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (bk < K && col < N)
                ? (TB ? B[static_cast<size_t>(col) * K + bk]
                      : B[static_cast<size_t>(bk) * N + col])
                : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[static_cast<size_t>(row) * N + col] = acc;
}
