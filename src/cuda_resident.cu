// Phase B1: resident-tensor GEMM (docs/CUDA_PHASE_B.md).
//
// Self-contained: raw CUDA runtime API + one tiled kernel. Deliberately
// no cuBLAS (the zero-dependency rule extends to the GPU path) and no
// use of the vendored Matrix's gpu_data_ fields (sync_vendor.sh owns
// that tree; its semantics are transformer_cpp's, not ours).
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "microtorch/device_cache.hpp"

namespace microtorch {
namespace device {

// Phase B2.0 (docs/CUDA_PHASE_B2.md): Variable-owned device state.
// Epoch-scoped: the buffer is trusted only inside the step window that
// stamped it. Buffers are REUSED across windows when dims match (params
// reach a realloc-free steady state; only the H2D refresh remains until
// B2.3 moves the optimizer on-device).
struct DevState {
    float* d = nullptr;
    size_t rows = 0, cols = 0;
    unsigned long long epoch = 0;
};

namespace {

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + what + ": " +
                                 cudaGetErrorString(e));
}

struct DevBuf {
    float* d = nullptr;
    size_t rows = 0, cols = 0;
};

bool g_residency = false;
std::unordered_map<const float*, DevBuf> g_table;  // key: host data ptr

bool g_step_residency = false;         // B2 master switch
bool g_in_step = false;                // inside step_begin()/step_end()
unsigned long long g_epoch = 0;        // window stamp; 0 = never opened

constexpr int TILE = 32;

// C(M,N) = A(M,K) * B(K,N), row-major, bounds-checked edge tiles.
__global__ void gemm_tiled(const float* __restrict__ A,
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
            (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

float* upload(const float* host, size_t bytes) {
    float* d = nullptr;
    cuda_check(cudaMalloc(&d, bytes), "malloc");
    cuda_check(cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice), "H2D");
    return d;
}

// Device pointer for an operand: table hit, or temp upload (owned=true).
float* operand(const Matrix& m, bool& owned) {
    auto it = g_table.find(m.get_data());
    if (it != g_table.end() && it->second.rows == m.rows() &&
        it->second.cols == m.cols()) {
        owned = false;
        return it->second.d;
    }
    owned = true;
    return upload(m.get_data(), m.rows() * m.cols() * sizeof(float));
}

// B2 operand under an open window. Cache hit iff the slot was stamped
// by THIS window; otherwise upload and (when a slot is given) cache +
// stamp it. Slotless operands (grads, non-Var intermediates) are temp
// uploads (owned=true). Falls back to the B1 table before uploading so
// explicitly-resident params are never duplicated.
float* window_operand(const Matrix& m, DevState** slot, bool& owned) {
    const size_t bytes = m.rows() * m.cols() * sizeof(float);
    if (slot == nullptr) {
        auto it = g_table.find(m.get_data());
        if (it != g_table.end() && it->second.rows == m.rows() &&
            it->second.cols == m.cols()) {
            owned = false;
            return it->second.d;
        }
        owned = true;
        return upload(m.get_data(), bytes);
    }
    DevState*& s = *slot;
    if (s != nullptr && s->epoch == g_epoch && s->rows == m.rows() &&
        s->cols == m.cols()) {
        owned = false;
        return s->d;  // stamped by this window: host has not mutated
    }
    if (s == nullptr) s = new DevState{};
    if (s->d != nullptr && (s->rows != m.rows() || s->cols != m.cols())) {
        cudaFree(s->d);
        s->d = nullptr;
    }
    if (s->d == nullptr)
        cuda_check(cudaMalloc(&s->d, bytes), "malloc devstate");
    cuda_check(cudaMemcpy(s->d, m.get_data(), bytes, cudaMemcpyHostToDevice),
               "H2D devstate");
    s->rows = m.rows();
    s->cols = m.cols();
    s->epoch = g_epoch;
    owned = false;
    return s->d;
}

// Transpose-flag GEMM: logical C(M,N) = op(A) * op(B). TA/TB pick the
// index math (TA: A stored (K,M); TB: B stored (N,K)). Same 32x32
// shared-memory tiling as gemm_tiled; the transposed loads are strided
// rather than coalesced — correctness first, coalescing is a measured
// optimization for later (docs/CUDA_PHASE_B2.md non-goals).
template <bool TA, bool TB>
__global__ void gemm_tiled_ops(const float* __restrict__ A,
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

}  // namespace

void set_residency(bool on) { g_residency = on; }
bool residency_enabled() { return g_residency; }

void make_resident(const Matrix& m) {
    const float* key = m.get_data();
    const size_t bytes = m.rows() * m.cols() * sizeof(float);
    auto it = g_table.find(key);
    if (it != g_table.end()) {
        // Refresh in place if the shape still matches; else realloc.
        if (it->second.rows == m.rows() && it->second.cols == m.cols()) {
            cuda_check(cudaMemcpy(it->second.d, key, bytes,
                                  cudaMemcpyHostToDevice), "H2D refresh");
            return;
        }
        cudaFree(it->second.d);
        g_table.erase(it);
    }
    g_table[key] = DevBuf{upload(key, bytes), m.rows(), m.cols()};
}

void invalidate(const Matrix& m) {
    auto it = g_table.find(m.get_data());
    if (it == g_table.end()) return;
    cudaFree(it->second.d);
    g_table.erase(it);
}

void evict_all() {
    for (auto& kv : g_table) cudaFree(kv.second.d);
    g_table.clear();
}

size_t resident_count() { return g_table.size(); }

// ---- Phase B2.0 ----

void set_step_residency(bool on) { g_step_residency = on; }
bool step_residency_enabled() { return g_step_residency; }
void step_begin() {
    ++g_epoch;
    g_in_step = true;
}
void step_end() { g_in_step = false; }

namespace detail {
void release_devstate_impl(DevState* s) {
    if (s->d != nullptr) cudaFree(s->d);
    delete s;
}
}  // namespace detail

bool step_resident_gemm(const Matrix& A, DevState** devA, Trans tA,
                        const Matrix& B, DevState** devB, Trans tB,
                        Matrix& C) {
    if (!g_step_residency || !g_in_step) return false;
    const int M = static_cast<int>(C.rows());
    const int N = static_cast<int>(C.cols());
    const int K = static_cast<int>((tA == Trans::T) ? A.rows() : A.cols());

    bool own_a = false, own_b = false;
    float* dA = window_operand(A, devA, own_a);
    float* dB = window_operand(B, devB, own_b);
    float* dC = nullptr;
    cuda_check(cudaMalloc(&dC, static_cast<size_t>(M) * N * sizeof(float)),
               "malloc C");

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    const bool ta = (tA == Trans::T), tb = (tB == Trans::T);
    if (!ta && !tb)
        gemm_tiled_ops<false, false><<<grid, block>>>(dA, dB, dC, M, N, K);
    else if (ta && !tb)
        gemm_tiled_ops<true, false><<<grid, block>>>(dA, dB, dC, M, N, K);
    else if (!ta && tb)
        gemm_tiled_ops<false, true><<<grid, block>>>(dA, dB, dC, M, N, K);
    else
        gemm_tiled_ops<true, true><<<grid, block>>>(dA, dB, dC, M, N, K);
    cuda_check(cudaGetLastError(), "gemm_ops launch");
    // Write-through (B2.0): C lands in host storage, so host data is
    // never stale. Deferred download is B2.1's contract change.
    cuda_check(cudaMemcpy(C.get_data(), dC,
                          static_cast<size_t>(M) * N * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H");

    cudaFree(dC);
    if (own_a) cudaFree(dA);
    if (own_b) cudaFree(dB);
    return true;
}

bool resident_matmul(const Matrix& a, const Matrix& b, Matrix& c) {
    if (!g_residency) return false;
    if (a.cols() != b.rows())
        throw std::runtime_error("resident_matmul: inner dims mismatch");
    const int M = static_cast<int>(a.rows());
    const int K = static_cast<int>(a.cols());
    const int N = static_cast<int>(b.cols());

    bool own_a = false, own_b = false;
    float* dA = operand(a, own_a);
    float* dB = operand(b, own_b);
    float* dC = nullptr;
    cuda_check(cudaMalloc(&dC, static_cast<size_t>(M) * N * sizeof(float)),
               "malloc C");

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    cuda_check(cudaGetLastError(), "gemm launch");
    cuda_check(cudaMemcpy(c.get_data(), dC,
                          static_cast<size_t>(M) * N * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H");

    cudaFree(dC);
    if (own_a) cudaFree(dA);
    if (own_b) cudaFree(dB);
    return true;
}

}  // namespace device
}  // namespace microtorch
