// Phase B1: resident-tensor GEMM (docs/CUDA_PHASE_B.md).
//
// Self-contained: raw CUDA runtime API + one tiled kernel, and no use of
// the vendored Matrix's gpu_data_ fields (sync_vendor.sh owns that tree;
// its semantics are transformer_cpp's, not ours). The zero-dependency
// rule was relaxed for training GEMMs on 18 Oct 2026 (Jonathan's decision,
// relayed by the ember.cpp session): step-window GEMMs may run as plain
// cublasSgemm, FP32 in and out, opt-in by MICROTORCH_CUBLAS=1, with TF32
// tensor-core math scoped to the call by MICROTORCH_TF32=1 (ember's
// pattern, tinyllama.cpp model.cpp). BF16 autocast (24 Oct 2026, opt-in per
// spec, device_cache.hpp Precision) is the one GemmEx: bfloat16 inputs,
// FP32 accumulate and output.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "device_pool.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "microtorch/device_cache.hpp"

// transformer_core's shared cuBLAS handle (src/cuda/matrix_ops.cu, namespace
// cuda; loss_kernels.cu declares it the same way). pooled_matmul runs its
// GEMM on this handle, so it computes what cuda::matmul computed.
namespace cuda {
cublasHandle_t get_cublas_handle();
}

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
    bool kept = false;  // stale across a step_end_keeping() boundary (see there)
    bool released = false;  // device-only value freed by release(): reads throw
};

[[noreturn]] void released_read(const char* where) {
    throw std::runtime_error(std::string("device: read of a released activation at ") + where +
                             " (device::release was called on a tensor that is still read)");
}
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
    cuda_check(pool::dmalloc(&d, bytes), "malloc");
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
    if (it != g_vcache.end() && it->second.released && it->second.n == n) released_read("device operand");
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
        // A parameter the optimizer updated on the device and kept across
        // the step boundary (step_end_keeping) lives here from now on: its
        // slot copy, uploaded on the first step, is dead weight.
        if (slot != nullptr && *slot != nullptr && (*slot)->d != nullptr) {
            pool::dfree((*slot)->d);
            (*slot)->d = nullptr;
            (*slot)->epoch = 0;
        }
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
        pool::dfree(s->d);
        s->d = nullptr;
    }
    if (s->d == nullptr)
        cuda_check(pool::dmalloc(&s->d, bytes), "malloc devstate");
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
        pool::dfree(it->second.d);
        g_table.erase(it);
    }
    g_table[key] = DevBuf{upload(key, bytes), m.rows(), m.cols()};
}

void invalidate(const Matrix& m) {
    auto it = g_table.find(m.get_data());
    if (it == g_table.end()) return;
    pool::dfree(it->second.d);
    g_table.erase(it);
}

void evict_all() {
    for (auto& kv : g_table) pool::dfree(kv.second.d);
    g_table.clear();
    // B2.1b value cache goes with it. Anything still stale is lost by
    // explicit request — evict_all() is the caller saying "device state
    // is disposable"; materialize first if the values matter.
    for (auto& kv : g_vcache) pool::dfree(kv.second.d);
    g_vcache.clear();
}

size_t resident_count() { return g_table.size(); }

size_t device_bytes_in_use() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return total_b - free_b;
}

size_t pool_live_bytes() { return pool::live_bytes(); }
size_t pool_cached_bytes() { return pool::cached_bytes(); }
size_t pool_peak_bytes() { return pool::peak_bytes(); }
void pool_reset_peak() { pool::reset_peak(); }
void synchronize() { cuda_check(cudaDeviceSynchronize(), "synchronize"); }

// ---- MICROTORCH_GPU_PROFILE: per-op GPU time from CUDA events ----------
namespace {
struct GpuMark {
    const char* op;
    cudaEvent_t a, b;
};
std::vector<GpuMark> g_marks;
std::vector<cudaEvent_t> g_free_events;
const char* g_open_op = nullptr;
cudaEvent_t g_open_ev = nullptr;
cudaEvent_t take_event() {
    if (!g_free_events.empty()) {
        cudaEvent_t e = g_free_events.back();
        g_free_events.pop_back();
        return e;
    }
    cudaEvent_t e = nullptr;
    cuda_check(cudaEventCreate(&e), "event create");
    return e;
}
}  // namespace

void set_blocking_sync() { cuda_check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync), "set blocking sync"); }

bool gpu_profile_on() {
    static const bool on = [] {
        const char* v = std::getenv("MICROTORCH_GPU_PROFILE");
        return v && std::strcmp(v, "1") == 0;
    }();
    return on;
}

namespace detail {
void gpu_profile_begin(const char* op) {
    if (!gpu_profile_on() || g_open_op) return;
    g_open_op = op;
    g_open_ev = take_event();
    cuda_check(cudaEventRecord(g_open_ev), "event record");
}
void gpu_profile_end(const char* op) {
    if (!gpu_profile_on() || g_open_op != op) return;
    cudaEvent_t e = take_event();
    cuda_check(cudaEventRecord(e), "event record");
    g_marks.push_back({op, g_open_ev, e});
    g_open_op = nullptr;
}
}  // namespace detail

std::vector<std::pair<std::string, double>> gpu_profile_take() {
    std::vector<std::pair<std::string, double>> out;
    if (!gpu_profile_on()) return out;
    cuda_check(cudaDeviceSynchronize(), "profile sync");
    std::unordered_map<std::string, double> sum;
    for (const auto& m : g_marks) {
        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, m.a, m.b), "event elapsed");
        sum[m.op] += ms;
        g_free_events.push_back(m.a);
        g_free_events.push_back(m.b);
    }
    g_marks.clear();
    for (const auto& kv : sum) out.emplace_back(kv.first, kv.second);
    return out;
}

// ---- Phase B2.0 ----

void set_step_residency(bool on) { g_step_residency = on; }
bool step_residency_enabled() { return g_step_residency; }
void step_begin() {
    ++g_epoch;
    g_in_step = true;
    // Entries kept across the last boundary join this window as its own
    // device-fresh values, so every epoch check downstream is unchanged.
    for (auto& kv : g_vcache)
        if (kv.second.kept) {
            kv.second.kept = false;
            if (kv.second.stale) kv.second.epoch = g_epoch;
        }
}
void step_end() {
    // B2.1b: the window edge is a materialize boundary — the optimizer,
    // eval, and checkpointing still read host storage between windows.
    if (g_defer_downloads) materialize_all();
    g_in_step = false;
}
void step_end_keeping(const std::vector<const Matrix*>& keep) {
    if (g_defer_downloads) {
        for (const Matrix* m : keep) {
            auto it = g_vcache.find(m->get_data());
            if (it != g_vcache.end() && it->second.stale) it->second.kept = true;
        }
        for (auto& kv : g_vcache) {
            if (!kv.second.stale || kv.second.kept) continue;
            cuda_check(cudaMemcpy(const_cast<float*>(kv.first), kv.second.d, kv.second.n * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "D2H step_end_keeping");
            kv.second.stale = false;
        }
    }
    g_in_step = false;
}

// ---- Phase B2.1b ----

void set_defer_downloads(bool on) { g_defer_downloads = on; }
bool defer_downloads_enabled() { return g_defer_downloads; }

bool deferring() { return defer_active(); }

bool host_stale(const Matrix& m) {
    auto it = g_vcache.find(m.get_data());
    return it != g_vcache.end() && it->second.stale;
}

void materialize(const Matrix& m) {
    auto it = g_vcache.find(m.get_data());
    if (it != g_vcache.end() && it->second.released) released_read("materialize");
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
        kv.second.kept = false;
    }
}

void discard(const Matrix& m) {
    auto it = g_vcache.find(m.get_data());
    if (it == g_vcache.end()) return;
    pool::dfree(it->second.d);
    g_vcache.erase(it);
}

void release(const Matrix& m) {
    if (!defer_active()) return;
    auto it = g_vcache.find(m.get_data());
    if (it == g_vcache.end() || it->second.released) return;
    if (!it->second.stale) {  // host holds the value: the device copy is a spare
        pool::dfree(it->second.d);
        g_vcache.erase(it);
        return;
    }
    pool::dfree(it->second.d);
    it->second.d = nullptr;
    it->second.stale = false;
    it->second.kept = false;
    it->second.released = true;  // kept (n too) until the Variable's destructor discards it
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
    v.released = false;  // written again: a value once more
    const bool had_fresh =
        (v.d != nullptr && v.stale && v.epoch == g_epoch && v.n == n);
    if (v.d != nullptr && v.n != n) {
        pool::dfree(v.d);
        v.d = nullptr;
    }
    if (v.d == nullptr) {
        cuda_check(pool::dmalloc(&v.d, n * sizeof(float)), "malloc vcache");
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
    if (s->d != nullptr) pool::dfree(s->d);
    delete s;
}
}  // namespace detail

namespace {

bool env_on(const char* k) {
    const char* v = std::getenv(k);
    return v && std::strcmp(v, "1") == 0;
}

bool cublas_on() {
    static const bool on = env_on("MICROTORCH_CUBLAS");
    return on;
}

cublasHandle_t cublas_handle() {
    static cublasHandle_t h = [] {
        cublasHandle_t x = nullptr;
        if (cublasCreate(&x) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        return x;
    }();
    return h;
}

// Row-major C[M,N] = op(A) op(B), every operand device-resident. cuBLAS is
// column-major, and a row-major X[r,c] read column-major is X^T with
// leading dimension c, so compute C^T = op(B)^T op(A)^T:
//   B: stored [K,N] (tb false) -> N with ldb N; stored [N,K] (tb) -> T with ldb K
//   A: stored [M,K] (ta false) -> N with lda K; stored [K,M] (ta) -> T with lda M
void cublas_gemm_rowmajor(const float* dA, bool ta, const float* dB, bool tb, float* dC, int M, int N, int K) {
    cublasHandle_t h = cublas_handle();
    static const bool tf32 = env_on("MICROTORCH_TF32");
    if (tf32) cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    const float one = 1.0f, zero = 0.0f;
    const cublasStatus_t s = cublasSgemm(h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K,
                                         &one, dB, tb ? K : N, dA, ta ? M : K, &zero, dC, N);
    if (tf32) cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSgemm failed: " + std::to_string(s));
}

__global__ void k_to_bf16(const float* x, __nv_bfloat16* y, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        y[i] = __float2bfloat16_rn(x[i]);
}

__nv_bfloat16* to_bf16(const float* d, size_t n) {
    __nv_bfloat16* h = nullptr;
    cuda_check(pool::dmalloc(&h, n * sizeof(__nv_bfloat16)), "malloc bf16");
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((n + 255) / 256, 4096));
    k_to_bf16<<<blocks, 256>>>(d, h, n);
    cuda_check(cudaGetLastError(), "bf16 convert");
    return h;
}

// The BF16 GEMM wants tensor cores with bfloat16 support (sm80+: A100,
// L4, H100). Older devices are refused loudly rather than emulated slowly.
void require_bf16_device() {
    static const bool ok = [] {
        int dev = 0, major = 0;
        cuda_check(cudaGetDevice(&dev), "get device");
        cuda_check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "device attribute");
        return major >= 8;
    }();
    if (!ok) throw std::runtime_error("precision bf16 needs compute capability 8.0 or newer (an L4 is 8.9)");
}

// Row-major C[M,N] = op(A) op(B) with A and B rounded to bfloat16 on the
// device and FP32 accumulation and output (cublas_gemm_rowmajor's operand
// mapping). The rounded copies are temporaries, freed before return.
void cublas_gemm_rowmajor_bf16(const float* dA, bool ta, const float* dB, bool tb, float* dC, int M, int N,
                               int K, const char* gemm_name) {
    require_bf16_device();
    detail::gpu_profile_begin("bf16_convert");
    __nv_bfloat16* a16 = to_bf16(dA, static_cast<size_t>(M) * K);
    __nv_bfloat16* b16 = to_bf16(dB, static_cast<size_t>(K) * N);
    detail::gpu_profile_end("bf16_convert");
    detail::gpu_profile_begin(gemm_name);
    const float one = 1.0f, zero = 0.0f;
    const cublasStatus_t s =
        cublasGemmEx(cublas_handle(), tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &one,
                     b16, CUDA_R_16BF, tb ? K : N, a16, CUDA_R_16BF, ta ? M : K, &zero, dC, CUDA_R_32F, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    detail::gpu_profile_end(gemm_name);
    pool::dfree(a16);
    pool::dfree(b16);
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx (bf16) failed: " + std::to_string(s));
}

// MICROTORCH_GPU_PROFILE names: the ProfileTag's name (or "gemm"), with a
// _bf16 twin for the mixed-precision path, and the conversion's own name.
const char* gemm_profile_name(bool bf16) {
    const char* t = profile_tag();
    const bool head = t != nullptr && std::strcmp(t, "gemm_head") == 0;
    if (bf16) return head ? "gemm_head_bf16" : "gemm_bf16";
    return head ? "gemm_head" : "gemm";
}

}  // namespace

bool step_resident_gemm(const Matrix& A, DevState** devA, Trans tA,
                        const Matrix& B, DevState** devB, Trans tB,
                        Matrix& C, Precision prec) {
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
        cuda_check(pool::dmalloc(&dC, nC * sizeof(float)), "malloc C");

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    const bool ta = (tA == Trans::T), tb = (tB == Trans::T);
    const bool bf16 = prec == Precision::BF16;
    const char* pname = gemm_profile_name(bf16);
    if (bf16) {
        cublas_gemm_rowmajor_bf16(dA, ta, dB, tb, dC, M, N, K, pname);
    } else {
        detail::gpu_profile_begin(pname);
        if (cublas_on())
            cublas_gemm_rowmajor(dA, ta, dB, tb, dC, M, N, K);
        else if (!ta && !tb)
            gemm_tiled_ops<false, false><<<grid, block>>>(dA, dB, dC, M, N, K);
        else if (ta && !tb)
            gemm_tiled_ops<true, false><<<grid, block>>>(dA, dB, dC, M, N, K);
        else if (!ta && tb)
            gemm_tiled_ops<false, true><<<grid, block>>>(dA, dB, dC, M, N, K);
        else
            gemm_tiled_ops<true, true><<<grid, block>>>(dA, dB, dC, M, N, K);
        cuda_check(cudaGetLastError(), "gemm_ops launch");
        detail::gpu_profile_end(pname);
    }
    if (!deferred) {
        cuda_check(cudaMemcpy(C.get_data(), dC, nC * sizeof(float),
                              cudaMemcpyDeviceToHost), "D2H");
        pool::dfree(dC);
    }
    if (own_a) pool::dfree(dA);
    if (own_b) pool::dfree(dB);
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
    cuda_check(pool::dmalloc(&dC, static_cast<size_t>(M) * N * sizeof(float)),
               "malloc C");

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    cuda_check(cudaGetLastError(), "gemm launch");
    cuda_check(cudaMemcpy(c.get_data(), dC,
                          static_cast<size_t>(M) * N * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H");

    pool::dfree(dC);
    if (own_a) pool::dfree(dA);
    if (own_b) pool::dfree(dB);
    return true;
}

namespace {
// A pool block for one call, returned on every exit (a throw included).
struct PoolBlock {
    float* p = nullptr;
    ~PoolBlock() { pool::dfree(p); }
};
}  // namespace

void pooled_matmul(const Matrix& a, const Matrix& b, Matrix& c) {
    if (a.cols() != b.rows() || c.rows() != a.rows() || c.cols() != b.cols())
        throw std::runtime_error("pooled_matmul: shapes disagree");
    // cuda::matmul's call, argument for argument: C^T = B^T A^T in cuBLAS's
    // column-major view (m = C.cols, n = C.rows, k = A.cols; ld B.cols,
    // A.cols, C.cols), alpha 1, beta 0, on transformer_core's handle.
    const size_t nA = a.rows() * a.cols(), nB = b.rows() * b.cols(), nC = c.rows() * c.cols();
    PoolBlock dA, dB, dC;
    cuda_check(pool::dmalloc(&dA.p, nA * sizeof(float)), "malloc pooled_matmul A");
    cuda_check(pool::dmalloc(&dB.p, nB * sizeof(float)), "malloc pooled_matmul B");
    cuda_check(pool::dmalloc(&dC.p, nC * sizeof(float)), "malloc pooled_matmul C");
    cuda_check(cudaMemcpy(dA.p, a.get_data(), nA * sizeof(float), cudaMemcpyHostToDevice), "H2D pooled_matmul A");
    cuda_check(cudaMemcpy(dB.p, b.get_data(), nB * sizeof(float), cudaMemcpyHostToDevice), "H2D pooled_matmul B");
    cublasHandle_t h = ::cuda::get_cublas_handle();
    cudaStream_t prev = nullptr;
    cublasGetStream(h, &prev);
    cublasSetStream(h, nullptr);  // the default stream, as every op in this library
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t s =
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(c.cols()), static_cast<int>(c.rows()),
                    static_cast<int>(a.cols()), &alpha, dB.p, static_cast<int>(b.cols()), dA.p,
                    static_cast<int>(a.cols()), &beta, dC.p, static_cast<int>(c.cols()));
    cublasSetStream(h, prev);
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("pooled_matmul: cublasSgemm failed: " + std::to_string(s));
    cuda_check(cudaMemcpy(c.get_data(), dC.p, nC * sizeof(float), cudaMemcpyDeviceToHost), "D2H pooled_matmul");
}

}  // namespace device
}  // namespace microtorch
