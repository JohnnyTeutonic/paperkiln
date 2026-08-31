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

// ---- Phase B2.1b: the value cache -----------------------------------
// Device-fresh op outputs, keyed by host data pointer. `stale` means the
// device copy is FRESHER than host storage. Buffers are retained across
// windows for reuse (dims permitting); staleness never survives a window
// because step_end() materializes when defer is on.
bool g_defer_downloads = false;
struct VBuf {
    float* d = nullptr;
    size_t n = 0;
    unsigned long long epoch = 0;
    bool stale = false;
};
std::unordered_map<const float*, VBuf> g_vcache;

// Deferral requires the device op set to be live: with devops off, the
// tape's CPU loops are the compute path, and they read host storage —
// deferring gemm outputs under them would serve stale host bytes.
bool defer_active() {
    return g_defer_downloads && g_in_step && device_ops_enabled();
}

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

// B2.1b: stale-value hit — the device copy of this host storage is
// fresher than host memory, so it is the ONLY correct operand source.
// The epoch guard kills the recycled-address class across windows: a
// host allocation reusing a dead matrix's address cannot inherit its
// stale entry, because step_end() materialized (un-staled) everything
// and a prior window's stamp no longer matches. Within a window, tape
// temporaries that die early are always materialized first by the
// accumulate() choke point, so no stale entry outlives its matrix.
float* vcache_hit(const float* key, size_t n) {
    auto it = g_vcache.find(key);
    if (it != g_vcache.end() && it->second.stale &&
        it->second.epoch == g_epoch && it->second.n == n)
        return it->second.d;
    return nullptr;
}

// B2 operand under an open window. THE VALUE CACHE OUTRANKS EVERYTHING:
// a stale hit means the device holds the authoritative value and host is
// behind, so uploading host would inject staleness. This check is
// UNIVERSAL — slotted or not.
//
// It used to sit inside the slotless branch only, and that was the
// B2.3 flat-loss bug (30 Aug 2026): every Variable's data passes a slot
// (ops call gemm with &var->dev), so under deferral a gemm on a deferred
// activation took the slot path and uploaded the untouched host buffer —
// zeros. Logits came out exactly 0, loss exactly ln(vocab), at every
// width and depth, while gpu/ops/residency-without-defer were perfect.
// The suite missed it because its shapes are tiny and its legs test ops
// in isolation, not a deferred activation feeding a slotted gemm.
//
// Otherwise: slot hit iff stamped by THIS window; else upload and (when
// a slot is given) cache + stamp it. Falls back to the B1 table before
// uploading so explicitly-resident params are never duplicated.
float* window_operand(const Matrix& m, DevState** slot, bool& owned) {
    const size_t bytes = m.rows() * m.cols() * sizeof(float);
    if (float* v = vcache_hit(m.get_data(), m.rows() * m.cols())) {
        owned = false;
        return v;
    }
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
    // B2.1b value cache goes with it. Anything still stale is lost by
    // explicit request — evict_all() is the caller saying "device state
    // is disposable"; materialize first if the values matter.
    for (auto& kv : g_vcache) cudaFree(kv.second.d);
    g_vcache.clear();
}

size_t resident_count() { return g_table.size(); }

size_t device_bytes_in_use() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return total_b - free_b;
}

// ---- Phase B2.0 ----

void set_step_residency(bool on) { g_step_residency = on; }
bool step_residency_enabled() { return g_step_residency; }
void step_begin() {
    ++g_epoch;
    g_in_step = true;
}
void step_end() {
    // B2.1b: the window edge is a materialize boundary — the optimizer,
    // eval, and checkpointing still read host storage between windows.
    if (g_defer_downloads) materialize_all();
    g_in_step = false;
}

// ---- Phase B2.1b ----

void set_defer_downloads(bool on) { g_defer_downloads = on; }
bool defer_downloads_enabled() { return g_defer_downloads; }

bool host_stale(const Matrix& m) {
    auto it = g_vcache.find(m.get_data());
    return it != g_vcache.end() && it->second.stale;
}

void materialize(const Matrix& m) {
    auto it = g_vcache.find(m.get_data());
    if (it == g_vcache.end() || !it->second.stale) return;
    cuda_check(cudaMemcpy(const_cast<float*>(m.get_data()), it->second.d,
                          it->second.n * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H materialize");
    it->second.stale = false;
}

void materialize_all() {
    for (auto& kv : g_vcache) {
        if (!kv.second.stale) continue;
        cuda_check(cudaMemcpy(const_cast<float*>(kv.first), kv.second.d,
                              kv.second.n * sizeof(float),
                              cudaMemcpyDeviceToHost), "D2H materialize_all");
        kv.second.stale = false;
    }
}

void discard(const Matrix& m) {
    auto it = g_vcache.find(m.get_data());
    if (it == g_vcache.end()) return;
    cudaFree(it->second.d);
    g_vcache.erase(it);
}

void devcheck_host_read(const Matrix& m, const char* where) {
    if (host_stale(m))
        throw std::runtime_error(
            std::string("DEVCHECK: host read of device-stale tensor at ") +
            where);
}

namespace detail {

float* vc_operand(const float* key, size_t n, bool& owned) {
    if (float* v = vcache_hit(key, n)) {
        owned = false;
        return v;
    }
    auto it = g_table.find(key);
    if (it != g_table.end() && it->second.rows * it->second.cols == n) {
        owned = false;
        return it->second.d;
    }
    owned = true;
    return upload(key, n * sizeof(float));
}

float* vc_output(const float* key, size_t n, bool& deferred,
                 bool need_current, const float* host_src) {
    if (!defer_active()) {
        deferred = false;
        return nullptr;
    }
    VBuf& v = g_vcache[key];
    const bool had_fresh =
        (v.d != nullptr && v.stale && v.epoch == g_epoch && v.n == n);
    if (v.d != nullptr && v.n != n) {
        cudaFree(v.d);
        v.d = nullptr;
    }
    if (v.d == nullptr) {
        cuda_check(cudaMalloc(&v.d, n * sizeof(float)), "malloc vcache");
        v.n = n;
    }
    if (need_current && !had_fresh)
        cuda_check(cudaMemcpy(v.d, host_src, n * sizeof(float),
                              cudaMemcpyHostToDevice), "H2D vcache inout");
    v.epoch = g_epoch;
    v.stale = true;
    deferred = true;
    return v.d;
}

}  // namespace detail

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
    // B2.1b: with defer active, C's buffer lives in the value cache and
    // the download is skipped (host goes stale until materialize).
    // Otherwise the B2.0 write-through path runs unchanged.
    const size_t nC = static_cast<size_t>(M) * N;
    bool deferred = false;
    float* dC = detail::vc_output(C.get_data(), nC, deferred,
                                  /*need_current=*/false, nullptr);
    if (!deferred)
        cuda_check(cudaMalloc(&dC, nC * sizeof(float)), "malloc C");

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
    if (!deferred) {
        cuda_check(cudaMemcpy(C.get_data(), dC, nC * sizeof(float),
                              cudaMemcpyDeviceToHost), "D2H");
        cudaFree(dC);
    }
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
