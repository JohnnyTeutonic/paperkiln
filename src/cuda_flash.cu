// Flash attention on the device (ops::attention_heads / attention_heads_turn_state
// with attention_kernel Flash; ops.hpp): forward and backward over a fused
// [T, 3d] qkv, every head in one launch, FP32 throughout, tiled through shared
// memory with an online softmax (FlashAttention-2's forward), so the [H T, sl]
// weights of the tiled path are never stored. The forward keeps the per-row
// log-sum-exp L_i = m_i + log l_i ([H T], head-major); the backward recomputes
// P_ij = exp(scale q_i . k_j - L_i) tile by tile:
//   D_i  = dO_i . O_i                          (k_flash_dot)
//   dV_j = sum_i P_ij dO_i,  dK_j = scale sum_i dS_ij q_i   (k_flash_bwd_kv, one block per key tile)
//   dQ_i = scale sum_j dS_ij k_j               (k_flash_bwd_q, one block per query tile)
//   dS_ij = P_ij (dO_i . v_j - D_i)
// dK/dV and dQ come from two kernels instead of FlashAttention-2's atomic dQ,
// so a run is reproducible bit for bit on one GPU and grid. Masks: 0 full
// (every key of the row's sequence block), 1 causal (j <= i), 2 turn_state
// (j <= i and ops::turn_state_visible(code[i], code[j])). Rows are B stacked
// sequences of seq_len; tiles never straddle two.
//
// Numerics against the tiled path: the same formulas, a different summation
// order (online rescaling, per-tile partial sums), so equal to float
// rounding, not bit for bit (tests/test_cuda_ops.cpp leg 10 states the
// bound). Same operand/output seam as cuda_ops.cu (detail::vc_operand /
// vc_output), so under deferred downloads q, k, v, O, L and the gradients
// stay on the device.
#include <cuda_runtime.h>

#include "device_pool.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"

namespace microtorch {
namespace device {
namespace devops {

namespace {

void fl_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA flash attention ") + what + ": " + cudaGetErrorString(e));
}

bool fl_active() { return device_ops_enabled() && get() == Device::CUDA; }

// cuda_ops.cu's In / Out, verbatim in behaviour (`owned` assigned in the body:
// see the note there).
struct FIn {
    float* d = nullptr;
    bool owned = false;
    FIn(const float* h, size_t n) { d = detail::vc_operand(h, n, owned); }
    ~FIn() {
        if (owned) pool::dfree(d);
    }
    FIn(const FIn&) = delete;
    FIn& operator=(const FIn&) = delete;
};
struct FOut {
    float* d = nullptr;
    float* host = nullptr;
    size_t n = 0;
    bool deferred = false;
    FOut(float* h, size_t n_) : host(h), n(n_) {
        d = detail::vc_output(h, n_, deferred, false, h);
        if (!deferred) fl_check(pool::dmalloc(&d, n_ * sizeof(float)), "malloc");
    }
    void finish() {
        if (!deferred) fl_check(cudaMemcpy(host, d, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H");
    }
    ~FOut() {
        if (!deferred && d) pool::dfree(d);
    }
    FOut(const FOut&) = delete;
    FOut& operator=(const FOut&) = delete;
};
struct FBuf {
    void* d = nullptr;
    explicit FBuf(size_t bytes) { fl_check(pool::dmalloc(&d, bytes), "malloc"); }
    ~FBuf() {
        if (d) pool::dfree(d);
    }
    FBuf(const FBuf&) = delete;
    FBuf& operator=(const FBuf&) = delete;
};
// Profile span (MICROTORCH_GPU_PROFILE) over a whole entry point.
struct FProf {
    const char* op;
    explicit FProf(const char* o) : op(o) { detail::gpu_profile_begin(op); }
    ~FProf() { detail::gpu_profile_end(op); }
};

constexpr int BQ = 64;   // forward query tile
constexpr int BK = 32;   // key tile (forward and backward); backward query tile
constexpr int NT = 256;  // threads: 16 x 16

struct Geo {
    const float* qkv;  // [T, 3d]
    int T, d, dk, ld, sl, mode;
    const int* code;   // [T] for mode 2, else null
};

__device__ __forceinline__ bool fl_visible(const Geo& g, int b0, int li, int lj) {
    if (lj >= g.sl || li >= g.sl) return false;
    if (g.mode == 0) return true;
    if (lj > li) return false;
    if (g.mode == 1) return true;
    const int ci = g.code[b0 + li], cj = g.code[b0 + lj];
    return (cj & 1) != 0 || (cj >> 1) == (ci >> 1);
}

// Max and sum over the 16 lanes that share a row (tx = lane % 16).
__device__ __forceinline__ float row16_max(float v) {
    for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float row16_sum(float v) {
    for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Stage rows [r0, r0 + R) of the head's column block (col0 .. col0 + dk) of a
// row-major matrix with row stride ld into S[R][DK + 1], zero past the
// sequence and past dk.
template <int R, int DK>
__device__ __forceinline__ void stage(float (*S)[DK + 1], const float* X, int ld, int b0, int r0, int sl, int col0,
                                      int dk) {
    for (int idx = threadIdx.x; idx < R * DK; idx += NT) {
        const int r = idx / DK, c = idx % DK;
        S[r][c] = (r0 + r < sl && c < dk) ? X[static_cast<size_t>(b0 + r0 + r) * ld + col0 + c] : 0.0f;
    }
}

// Forward: one block per (sequence, 64-row query tile, head). Thread (ty, tx)
// owns rows ty + 16 a (a < 4); for a 32-key tile it scores keys tx + 16 c
// (c < 2) and accumulates output columns tx + 16 b (b < DK / 16).
template <int DK>
__global__ void k_flash_fwd(Geo g, float scale, float* __restrict__ y, float* __restrict__ lse) {
    __shared__ float Qs[BQ][DK + 1];
    __shared__ float Ks[BK][DK + 1];
    __shared__ float Vs[BK][DK + 1];
    __shared__ float Ps[BQ][BK + 1];
    const int h = blockIdx.z, nt = (g.sl + BQ - 1) / BQ;
    const int b0 = (blockIdx.x / nt) * g.sl, q0 = (blockIdx.x % nt) * BQ;
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    constexpr int NB = DK / 16;
    stage<BQ, DK>(Qs, g.qkv, g.ld, b0, q0, g.sl, h * g.dk, g.dk);
    float m[4], l[4], o[4][NB];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        m[a] = -INFINITY;
        l[a] = 0.0f;
#pragma unroll
        for (int b = 0; b < NB; ++b) o[a][b] = 0.0f;
    }
    const int kend = g.mode == 0 ? g.sl : min(g.sl, q0 + BQ);
    for (int k0 = 0; k0 < kend; k0 += BK) {
        __syncthreads();  // the previous tile's Ks/Vs/Ps are spent (and Qs is staged)
        stage<BK, DK>(Ks, g.qkv, g.ld, b0, k0, g.sl, g.d + h * g.dk, g.dk);
        stage<BK, DK>(Vs, g.qkv, g.ld, b0, k0, g.sl, 2 * g.d + h * g.dk, g.dk);
        __syncthreads();
        // Register-blocked scores: k outermost, so each staged value feeds
        // several FMAs; every output still sums k ascending from 0 (the same
        // bits as one dot product at a time).
        float sc[4][2];
#pragma unroll
        for (int a = 0; a < 4; ++a) sc[a][0] = sc[a][1] = 0.0f;
#pragma unroll 8
        for (int k = 0; k < DK; ++k) {
            const float k0v = Ks[tx][k], k1v = Ks[tx + 16][k];
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const float qv = Qs[ty + 16 * a][k];
                sc[a][0] = fmaf(qv, k0v, sc[a][0]);
                sc[a][1] = fmaf(qv, k1v, sc[a][1]);
            }
        }
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            const int li = q0 + ty + 16 * a;
            float s[2];
#pragma unroll
            for (int c = 0; c < 2; ++c)
                s[c] = fl_visible(g, b0, li, k0 + tx + 16 * c) ? sc[a][c] * scale : -INFINITY;
            const float mn = fmaxf(m[a], row16_max(fmaxf(s[0], s[1])));
            const float mu = mn == -INFINITY ? 0.0f : mn;
            const float p0 = expf(s[0] - mu), p1 = expf(s[1] - mu);
            const float alpha = expf(m[a] - mu);
            l[a] = l[a] * alpha + row16_sum(p0 + p1);
            m[a] = mn;
#pragma unroll
            for (int b = 0; b < NB; ++b) o[a][b] *= alpha;
            Ps[ty + 16 * a][tx] = p0;
            Ps[ty + 16 * a][tx + 16] = p1;
        }
        __syncthreads();
#pragma unroll 4
        for (int j = 0; j < BK; ++j) {
            float vv[NB];
#pragma unroll
            for (int b = 0; b < NB; ++b) vv[b] = Vs[j][tx + 16 * b];
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const float p = Ps[ty + 16 * a][j];
#pragma unroll
                for (int b = 0; b < NB; ++b) o[a][b] = fmaf(p, vv[b], o[a][b]);
            }
        }
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int li = q0 + ty + 16 * a;
        if (li >= g.sl) continue;
        const size_t row = static_cast<size_t>(b0 + li);
        const float inv = 1.0f / l[a];
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int c = tx + 16 * b;
            if (c < g.dk) y[row * g.d + h * g.dk + c] = o[a][b] * inv;
        }
        if (tx == 0) lse[static_cast<size_t>(h) * g.T + row] = m[a] + logf(l[a]);
    }
}

// D[h T + i] = dO_i . O_i over the head's columns: one warp per (row, head).
__global__ void k_flash_dot(const float* __restrict__ dy, const float* __restrict__ yv, float* __restrict__ D, int T,
                            int d, int dk, int H) {
    const int w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (w >= T * H) return;
    const int h = w / T, i = w % T;
    const size_t o = static_cast<size_t>(i) * d + static_cast<size_t>(h) * dk;
    float s = 0.0f;
    for (int c = lane; c < dk; c += 32) s = fmaf(dy[o + c], yv[o + c], s);
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    if (lane == 0) D[w] = s;
}

// dK, dV: one block per (sequence, 32-key tile, head). Thread (ty, tx) owns
// keys ty + 16 a (a < 2) and output columns tx + 16 b; per 32-query tile it
// forms P and dS for keys ty + 16 a and queries tx + 16 c (c < 2).
template <int DK>
__global__ void k_flash_bwd_kv(Geo g, const float* __restrict__ dy, const float* __restrict__ lse,
                               const float* __restrict__ D, float scale, float* __restrict__ dqkv) {
    __shared__ float Ks[BK][DK + 1];
    __shared__ float Vs[BK][DK + 1];
    __shared__ float Qs[BK][DK + 1];
    __shared__ float Gs[BK][DK + 1];  // dO
    __shared__ float Ps[BK][BK + 1];  // [key][query]
    __shared__ float Ss[BK][BK + 1];  // dS, [key][query]
    __shared__ float Ls[BK], Ds[BK];
    const int h = blockIdx.z, nt = (g.sl + BK - 1) / BK;
    const int b0 = (blockIdx.x / nt) * g.sl, k0 = (blockIdx.x % nt) * BK;
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    constexpr int NB = DK / 16;
    stage<BK, DK>(Ks, g.qkv, g.ld, b0, k0, g.sl, g.d + h * g.dk, g.dk);
    stage<BK, DK>(Vs, g.qkv, g.ld, b0, k0, g.sl, 2 * g.d + h * g.dk, g.dk);
    float dk_acc[2][NB], dv_acc[2][NB];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < NB; ++b) dk_acc[a][b] = dv_acc[a][b] = 0.0f;
    const int qstart = g.mode == 0 ? 0 : k0;  // causal masks: queries before the tile see none of its keys
    for (int i0 = qstart; i0 < g.sl; i0 += BK) {
        __syncthreads();
        stage<BK, DK>(Qs, g.qkv, g.ld, b0, i0, g.sl, h * g.dk, g.dk);
        stage<BK, DK>(Gs, dy, g.d, b0, i0, g.sl, h * g.dk, g.dk);
        if (threadIdx.x < BK) {
            const int li = i0 + threadIdx.x;
            const size_t r = static_cast<size_t>(h) * g.T + b0 + li;
            Ls[threadIdx.x] = li < g.sl ? lse[r] : 0.0f;
            Ds[threadIdx.x] = li < g.sl ? D[r] : 0.0f;
        }
        __syncthreads();
        {
            // Register-blocked: k outermost (8 loads per 8 FMAs, was 2 per FMA),
            // each output summing k ascending as before.
            float s[2][2] = {}, dp[2][2] = {};
#pragma unroll 8
            for (int k = 0; k < DK; ++k) {
                const float q0v = Qs[tx][k], q1v = Qs[tx + 16][k], g0v = Gs[tx][k], g1v = Gs[tx + 16][k];
#pragma unroll
                for (int a = 0; a < 2; ++a) {
                    const float kv = Ks[ty + 16 * a][k], vv = Vs[ty + 16 * a][k];
                    s[a][0] = fmaf(q0v, kv, s[a][0]);
                    s[a][1] = fmaf(q1v, kv, s[a][1]);
                    dp[a][0] = fmaf(g0v, vv, dp[a][0]);
                    dp[a][1] = fmaf(g1v, vv, dp[a][1]);
                }
            }
#pragma unroll
            for (int a = 0; a < 2; ++a) {
                const int kj = ty + 16 * a, lj = k0 + kj;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int qi = tx + 16 * c, li = i0 + qi;
                    const float p = fl_visible(g, b0, li, lj) ? expf(s[a][c] * scale - Ls[qi]) : 0.0f;
                    Ps[kj][qi] = p;
                    Ss[kj][qi] = p * (dp[a][c] - Ds[qi]);
                }
            }
        }
        __syncthreads();
#pragma unroll 4
        for (int qi = 0; qi < BK; ++qi) {
            float gv[NB], qv[NB];
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                gv[b] = Gs[qi][tx + 16 * b];
                qv[b] = Qs[qi][tx + 16 * b];
            }
#pragma unroll
            for (int a = 0; a < 2; ++a) {
                const int kj = ty + 16 * a;
                const float p = Ps[kj][qi], ds = Ss[kj][qi];
#pragma unroll
                for (int b = 0; b < NB; ++b) {
                    dv_acc[a][b] = fmaf(p, gv[b], dv_acc[a][b]);
                    dk_acc[a][b] = fmaf(ds, qv[b], dk_acc[a][b]);
                }
            }
        }
    }
#pragma unroll
    for (int a = 0; a < 2; ++a) {
        const int lj = k0 + ty + 16 * a;
        if (lj >= g.sl) continue;
        const size_t row = static_cast<size_t>(b0 + lj) * g.ld;
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int c = tx + 16 * b;
            if (c >= g.dk) continue;
            dqkv[row + g.d + h * g.dk + c] = scale * dk_acc[a][b];
            dqkv[row + 2 * g.d + h * g.dk + c] = dv_acc[a][b];
        }
    }
}

// dQ: one block per (sequence, 32-query tile, head). Thread (ty, tx) owns
// queries ty + 16 a and output columns tx + 16 b; per 32-key tile it forms
// dS for queries ty + 16 a and keys tx + 16 c.
template <int DK>
__global__ void k_flash_bwd_q(Geo g, const float* __restrict__ dy, const float* __restrict__ lse,
                              const float* __restrict__ D, float scale, float* __restrict__ dqkv) {
    __shared__ float Qs[BK][DK + 1];
    __shared__ float Gs[BK][DK + 1];
    __shared__ float Ks[BK][DK + 1];
    __shared__ float Vs[BK][DK + 1];
    __shared__ float Ss[BK][BK + 1];  // dS, [query][key]
    const int h = blockIdx.z, nt = (g.sl + BK - 1) / BK;
    const int b0 = (blockIdx.x / nt) * g.sl, i0 = (blockIdx.x % nt) * BK;
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    constexpr int NB = DK / 16;
    stage<BK, DK>(Qs, g.qkv, g.ld, b0, i0, g.sl, h * g.dk, g.dk);
    stage<BK, DK>(Gs, dy, g.d, b0, i0, g.sl, h * g.dk, g.dk);
    float Lr[2], Dr[2], dq_acc[2][NB];
#pragma unroll
    for (int a = 0; a < 2; ++a) {
        const int li = i0 + ty + 16 * a;
        const size_t r = static_cast<size_t>(h) * g.T + b0 + li;
        Lr[a] = li < g.sl ? lse[r] : 0.0f;
        Dr[a] = li < g.sl ? D[r] : 0.0f;
#pragma unroll
        for (int b = 0; b < NB; ++b) dq_acc[a][b] = 0.0f;
    }
    const int kend = g.mode == 0 ? g.sl : min(g.sl, i0 + BK);
    for (int k0 = 0; k0 < kend; k0 += BK) {
        __syncthreads();
        stage<BK, DK>(Ks, g.qkv, g.ld, b0, k0, g.sl, g.d + h * g.dk, g.dk);
        stage<BK, DK>(Vs, g.qkv, g.ld, b0, k0, g.sl, 2 * g.d + h * g.dk, g.dk);
        __syncthreads();
        {
            float s[2][2] = {}, dp[2][2] = {};
#pragma unroll 8
            for (int k = 0; k < DK; ++k) {
                const float k0v = Ks[tx][k], k1v = Ks[tx + 16][k], v0v = Vs[tx][k], v1v = Vs[tx + 16][k];
#pragma unroll
                for (int a = 0; a < 2; ++a) {
                    const float qv = Qs[ty + 16 * a][k], gv = Gs[ty + 16 * a][k];
                    s[a][0] = fmaf(qv, k0v, s[a][0]);
                    s[a][1] = fmaf(qv, k1v, s[a][1]);
                    dp[a][0] = fmaf(gv, v0v, dp[a][0]);
                    dp[a][1] = fmaf(gv, v1v, dp[a][1]);
                }
            }
#pragma unroll
            for (int a = 0; a < 2; ++a) {
                const int qi = ty + 16 * a, li = i0 + qi;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int kj = tx + 16 * c, lj = k0 + kj;
                    const float p = fl_visible(g, b0, li, lj) ? expf(s[a][c] * scale - Lr[a]) : 0.0f;
                    Ss[qi][kj] = p * (dp[a][c] - Dr[a]);
                }
            }
        }
        __syncthreads();
#pragma unroll 4
        for (int kj = 0; kj < BK; ++kj) {
            float kv[NB];
#pragma unroll
            for (int b = 0; b < NB; ++b) kv[b] = Ks[kj][tx + 16 * b];
#pragma unroll
            for (int a = 0; a < 2; ++a) {
                const float ds = Ss[ty + 16 * a][kj];
#pragma unroll
                for (int b = 0; b < NB; ++b) dq_acc[a][b] = fmaf(ds, kv[b], dq_acc[a][b]);
            }
        }
    }
#pragma unroll
    for (int a = 0; a < 2; ++a) {
        const int li = i0 + ty + 16 * a;
        if (li >= g.sl) continue;
        const size_t row = static_cast<size_t>(b0 + li) * g.ld;
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int c = tx + 16 * b;
            if (c < g.dk) dqkv[row + h * g.dk + c] = scale * dq_acc[a][b];
        }
    }
}

// The head width's template: 16, 32 or 64 (zero-padded up from dk).
int dk_tile(int dk) { return dk <= 16 ? 16 : dk <= 32 ? 32 : dk <= 64 ? 64 : 0; }

bool flash_shapes(const Matrix& qkv, size_t heads, size_t seq_len, int& T, int& d, int& dk, int& sl) {
    T = static_cast<int>(qkv.rows());
    sl = static_cast<int>(seq_len == 0 ? qkv.rows() : seq_len);
    if (heads == 0 || qkv.cols() % (3 * heads) != 0 || sl <= 0 || T % sl != 0) return false;
    d = static_cast<int>(qkv.cols() / 3);
    dk = d / static_cast<int>(heads);
    return dk_tile(dk) != 0;
}

}  // namespace

bool flash_attention_supported(size_t head_dim) { return dk_tile(static_cast<int>(head_dim)) != 0; }

bool attn_flash_fwd(const Matrix& qkv, size_t heads, float scale, size_t seq_len, int mode,
                    const std::vector<int>* code, Matrix& lse, Matrix& y) {
    if (!fl_active()) return false;
    int T, d, dk, sl;
    if (!flash_shapes(qkv, heads, seq_len, T, d, dk, sl) || mode < 0 || mode > 2) return false;
    const int H = static_cast<int>(heads);
    if ((mode == 2) != (code != nullptr) || (code && code->size() != qkv.rows())) return false;
    if (lse.rows() * lse.cols() != heads * qkv.rows() || y.rows() != qkv.rows() || y.cols() != static_cast<size_t>(d))
        return false;
    FIn iqkv(qkv.get_data(), static_cast<size_t>(T) * 3 * d);
    FBuf dcode(code ? code->size() * sizeof(int) : sizeof(int));
    if (code)
        fl_check(cudaMemcpy(dcode.d, code->data(), code->size() * sizeof(int), cudaMemcpyHostToDevice), "code H2D");
    FOut oy(y.get_data(), static_cast<size_t>(T) * d), ol(lse.get_data(), static_cast<size_t>(H) * T);
    {
        FProf prof("attn_flash_fwd");
        const Geo g{iqkv.d, T, d, dk, 3 * d, sl, mode, code ? static_cast<const int*>(dcode.d) : nullptr};
        const dim3 grid(static_cast<unsigned>((T / sl) * ((sl + BQ - 1) / BQ)), 1, static_cast<unsigned>(H));
        switch (dk_tile(dk)) {
            case 16: k_flash_fwd<16><<<grid, NT>>>(g, scale, oy.d, ol.d); break;
            case 32: k_flash_fwd<32><<<grid, NT>>>(g, scale, oy.d, ol.d); break;
            default: k_flash_fwd<64><<<grid, NT>>>(g, scale, oy.d, ol.d); break;
        }
        fl_check(cudaGetLastError(), "fwd");
    }
    oy.finish();
    ol.finish();
    return true;
}

bool attn_flash_bwd(const Matrix& qkv, size_t heads, const Matrix& lse, const Matrix& y, const Matrix& dY, float scale,
                    size_t seq_len, int mode, const std::vector<int>* code, Matrix& dqkv) {
    if (!fl_active()) return false;
    int T, d, dk, sl;
    if (!flash_shapes(qkv, heads, seq_len, T, d, dk, sl) || mode < 0 || mode > 2) return false;
    const int H = static_cast<int>(heads);
    if ((mode == 2) != (code != nullptr) || (code && code->size() != qkv.rows())) return false;
    if (lse.rows() * lse.cols() != heads * qkv.rows() || y.rows() != qkv.rows() || dY.rows() != qkv.rows() ||
        y.cols() != static_cast<size_t>(d) || dY.cols() != static_cast<size_t>(d) || dqkv.rows() != qkv.rows() ||
        dqkv.cols() != qkv.cols())
        return false;
    const size_t nTd = static_cast<size_t>(T) * d;
    FIn iqkv(qkv.get_data(), 3 * nTd), il(lse.get_data(), static_cast<size_t>(H) * T), iy(y.get_data(), nTd),
        igy(dY.get_data(), nTd);
    FBuf dcode(code ? code->size() * sizeof(int) : sizeof(int));
    if (code)
        fl_check(cudaMemcpy(dcode.d, code->data(), code->size() * sizeof(int), cudaMemcpyHostToDevice), "code H2D");
    FBuf Dbuf(static_cast<size_t>(H) * T * sizeof(float));
    FOut og(dqkv.get_data(), 3 * nTd);  // every element written below
    {
        FProf prof("attn_flash_bwd");
        float* D = static_cast<float*>(Dbuf.d);
        const int warps = 8;
        k_flash_dot<<<(T * H + warps - 1) / warps, warps * 32>>>(igy.d, iy.d, D, T, d, dk, H);
        fl_check(cudaGetLastError(), "bwd dot");
        const Geo g{iqkv.d, T, d, dk, 3 * d, sl, mode, code ? static_cast<const int*>(dcode.d) : nullptr};
        const dim3 grid(static_cast<unsigned>((T / sl) * ((sl + BK - 1) / BK)), 1, static_cast<unsigned>(H));
        switch (dk_tile(dk)) {
            case 16:
                k_flash_bwd_kv<16><<<grid, NT>>>(g, igy.d, il.d, D, scale, og.d);
                k_flash_bwd_q<16><<<grid, NT>>>(g, igy.d, il.d, D, scale, og.d);
                break;
            case 32:
                k_flash_bwd_kv<32><<<grid, NT>>>(g, igy.d, il.d, D, scale, og.d);
                k_flash_bwd_q<32><<<grid, NT>>>(g, igy.d, il.d, D, scale, og.d);
                break;
            default:
                k_flash_bwd_kv<64><<<grid, NT>>>(g, igy.d, il.d, D, scale, og.d);
                k_flash_bwd_q<64><<<grid, NT>>>(g, igy.d, il.d, D, scale, og.d);
                break;
        }
        fl_check(cudaGetLastError(), "bwd");
    }
    og.finish();
    return true;
}

}  // namespace devops
}  // namespace device
}  // namespace microtorch
