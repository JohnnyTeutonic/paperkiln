// Phase B2.1a: the device-side op set (docs/CUDA_PHASE_B2.md section 4).
//
// Self-contained: raw CUDA runtime API, our own kernels, no cuBLAS/cuDNN
// (the zero-dependency rule extends to the GPU path). One canonical entry
// point per op — the Phase 0 audit's duplicate-kernel landmine stays
// behind the seam. Every kernel implements EXACTLY the formula the CPU
// tape op implements (ops.cpp is the reference; the parity test pins the
// two against each other, and the same FD oracle that gates the CPU path
// gates this one on T4).
//
// B2.1a semantics were WRITE-THROUGH: host in, host out, device
// round-trip inside, so host data was never stale (B2.0's contract).
// B2.1b (this revision) adds the deferred path: with
// MICROTORCH_DEFER_DOWNLOADS=1 and a step window open, outputs stay in
// the value cache (cuda_resident.cu), chained ops fetch them there, and
// host storage is stale until device::materialize()/step_end(). With the
// switch off every op behaves exactly as B2.1a. Per-call H2D/D2H
// traffic makes these SLOWER than the CPU loops at small dims — that is
// expected, measured, and irrelevant: the wall-clock claim is only ever
// made by the Rung C benchmark after chaining exists.
#include <cuda_runtime.h>

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

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA devops ") + what + ": " +
                                 cudaGetErrorString(e));
}

// Gate shared by every entry point: the master switch (env
// MICROTORCH_DEVICE_OPS=1 / set_device_ops) AND the process device.
// Returning false makes the caller fall through to today's CPU loop,
// so a CPU run is bit-identical to the pre-B2.1 tape.
bool active() { return device_ops_enabled() && get() == Device::CUDA; }

constexpr int NTHREADS = 256;

float* dalloc(size_t n) {
    float* d = nullptr;
    cuda_check(cudaMalloc(&d, n * sizeof(float)), "malloc");
    return d;
}
void h2d(float* d, const float* h, size_t n) {
    cuda_check(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice),
               "H2D");
}
void d2h(float* h, const float* d, size_t n) {
    cuda_check(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost),
               "D2H");
}

// RAII so a throwing cuda_check cannot leak the temporaries.
struct DBuf {
    float* d = nullptr;
    explicit DBuf(size_t n) : d(dalloc(n)) {}
    DBuf(const float* h, size_t n) : d(dalloc(n)) { h2d(d, h, n); }
    ~DBuf() {
        if (d) cudaFree(d);
    }
    DBuf(const DBuf&) = delete;
    DBuf& operator=(const DBuf&) = delete;
};

// ---- B2.1b seams (docs/CUDA_PHASE_B2.md checklist steps 2-3) ---------
// In: operand fetch through the residency stack — stale-value hit or
// B1-resident hit (owned=false), else temp upload (owned=true).
// Out: with defer active (switch on + step window open) the result stays
// in the value cache and host storage goes stale until materialize();
// otherwise a temp buffer is written through on finish(), which is
// exactly B2.1a's behaviour. Aux outputs (per-row stat vectors) remain
// write-through: small, and consumed host-side by the CPU reference
// formulas either way.
struct In {
    float* d = nullptr;
    bool owned = false;
    In(const float* h, size_t n)
        : d(detail::vc_operand(h, n, owned)) {}
    ~In() {
        if (owned) cudaFree(d);
    }
    In(const In&) = delete;
    In& operator=(const In&) = delete;
};
struct Out {
    float* d = nullptr;
    float* host = nullptr;
    size_t n = 0;
    bool deferred = false;
    Out(float* h, size_t n_, bool need_current = false) : host(h), n(n_) {
        d = detail::vc_output(h, n_, deferred, need_current, h);
        if (!deferred) {
            d = dalloc(n_);
            if (need_current) h2d(d, h, n_);
        }
    }
    void finish() {
        if (!deferred) d2h(host, d, n);
    }
    ~Out() {
        if (!deferred && d) cudaFree(d);
    }
    Out(const Out&) = delete;
    Out& operator=(const Out&) = delete;
};

// ---- elementwise (grid-stride) --------------------------------------

__global__ void k_add(const float* a, const float* b, float* y, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        y[i] = a[i] + b[i];
}
__global__ void k_sub(const float* a, const float* b, float* y, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        y[i] = a[i] - b[i];
}
__global__ void k_mul(const float* a, const float* b, float* y, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        y[i] = a[i] * b[i];
}
__global__ void k_scale(const float* a, float s, float* y, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        y[i] = a[i] * s;
}
// y += a*x — the grad-accumulate shape (accumulate() in autograd.cpp).
__global__ void k_axpy(float* y, float a, const float* x, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        y[i] += a * x[i];
}
__global__ void k_fill(float* y, float v, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        y[i] = v;
}

// ---- activations -----------------------------------------------------

__global__ void k_sigmoid_fwd(const float* x, float* y, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        y[i] = 1.0f / (1.0f + expf(-x[i]));
}
// dx = dy * s * (1 - s), s = the SAVED forward output (ops.cpp sigmoid).
__global__ void k_sigmoid_bwd(const float* s, const float* dy, float* dx,
                              size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        dx[i] = dy[i] * s[i] * (1.0f - s[i]);
}

// tanh-GELU forward, the same approximation Matrix::apply_gelu computes:
//   y = 0.5 x (1 + tanh(k (x + 0.044715 x^3))), k = sqrt(2/pi)
__global__ void k_gelu_fwd(const float* x, float* y, size_t n) {
    constexpr float k = 0.7978845608028654f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const float v = x[i];
        y[i] = 0.5f * v * (1.0f + tanhf(k * (v + 0.044715f * v * v * v)));
    }
}
// The CORRECT tanh-GELU derivative (ops.cpp gelu backward; NOT the
// vendored Matrix::apply_gelu_derivative — see primitives.hpp):
//   u   = k (x + 0.044715 x^3)
//   d   = cdf(u) + x * 0.5 sech^2(u) * u'
__global__ void k_gelu_bwd(const float* x, const float* dy, float* dx,
                           size_t n) {
    constexpr float k = 0.7978845608028654f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const float v = x[i];
        const float u = k * (v + 0.044715f * v * v * v);
        const float t = tanhf(u);
        const float cdf = 0.5f * (1.0f + t);
        const float pdf =
            0.5f * (1.0f - t * t) * k * (1.0f + 0.134145f * v * v);
        dx[i] = dy[i] * (cdf + v * pdf);
    }
}

// ---- rowwise reductions ---------------------------------------------
//
// One block per row; threads stride the columns; two-phase shared-memory
// reduction. Reduction ORDER differs from the CPU's serial loop, so
// rowwise results carry fp32 tolerance (the parity test's 1e-5), never
// bitwise claims — same rule test_step_residency already states.

__global__ void k_softmax_fwd(const float* x, float* y, int C) {
    extern __shared__ float sh[];
    const float* xr = x + static_cast<size_t>(blockIdx.x) * C;
    float* yr = y + static_cast<size_t>(blockIdx.x) * C;

    float mx = -3.402823466e+38f;
    for (int j = threadIdx.x; j < C; j += blockDim.x)
        mx = fmaxf(mx, xr[j]);
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mx = sh[0];
    __syncthreads();

    float z = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const float e = expf(xr[j] - mx);
        yr[j] = e;
        z += e;
    }
    sh[threadIdx.x] = z;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    z = sh[0];
    __syncthreads();

    for (int j = threadIdx.x; j < C; j += blockDim.x) yr[j] /= z;
}

// dX = S .* (dY - rowsum(dY .* S)) — softmax_row's backward and the same
// formula the composed attention backward uses.
__global__ void k_softmax_bwd(const float* S, const float* dY, float* dX,
                              int C) {
    extern __shared__ float sh[];
    const float* Sr = S + static_cast<size_t>(blockIdx.x) * C;
    const float* dYr = dY + static_cast<size_t>(blockIdx.x) * C;
    float* dXr = dX + static_cast<size_t>(blockIdx.x) * C;

    float dot = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) dot += dYr[j] * Sr[j];
    sh[threadIdx.x] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    dot = sh[0];
    __syncthreads();

    for (int j = threadIdx.x; j < C; j += blockDim.x)
        dXr[j] = Sr[j] * (dYr[j] - dot);
}

// ---- masked attention softmax (B2.2) --------------------------------
//
// In place over the raw scores A [T,T]: one block per row i; the
// visible range is computed exactly as ops.cpp's host loops compute it
// (fused: [lo, hi); swa: [b0, sink_hi) U [win_lo, i]). Masked entries
// are written as hard zeros, never exponentiated. The scale multiply is
// folded into the reads (the host stores A*=scale first — same value,
// one rounding difference, inside the rowwise fp32 tolerance).

__global__ void k_attn_masked_softmax(float* A, float scale, int sl,
                                      int causal, int T) {
    extern __shared__ float sh[];
    const int i = blockIdx.x;
    float* Ar = A + static_cast<size_t>(i) * T;
    const int b0 = (i / sl) * sl;
    const int lo = b0, hi = causal ? i + 1 : b0 + sl;

    float mx = -1e30f;
    for (int j = lo + threadIdx.x; j < hi; j += blockDim.x)
        mx = fmaxf(mx, Ar[j] * scale);
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mx = sh[0];
    __syncthreads();

    float z = 0.0f;
    for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) {
        const float e = expf(Ar[j] * scale - mx);
        Ar[j] = e;
        z += e;
    }
    sh[threadIdx.x] = z;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    z = sh[0];
    __syncthreads();

    for (int j = threadIdx.x; j < T; j += blockDim.x)
        Ar[j] = (j >= lo && j < hi) ? Ar[j] / z : 0.0f;
}

__global__ void k_swa_masked_softmax(float* A, float scale, int sl,
                                     int window, int sinks, int T) {
    extern __shared__ float sh[];
    const int i = blockIdx.x;
    float* Ar = A + static_cast<size_t>(i) * T;
    const int b0 = (i / sl) * sl;
    const int ii = i - b0;
    const int win_lo = b0 + (ii + 1 > window ? ii + 1 - window : 0);
    int sink_hi = b0 + (sinks < ii + 1 ? sinks : ii + 1);
    if (sink_hi > win_lo) sink_hi = win_lo;

    float mx = -1e30f;
    for (int j = b0 + threadIdx.x; j < sink_hi; j += blockDim.x)
        mx = fmaxf(mx, Ar[j] * scale);
    for (int j = win_lo + threadIdx.x; j <= i; j += blockDim.x)
        mx = fmaxf(mx, Ar[j] * scale);
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mx = sh[0];
    __syncthreads();

    float z = 0.0f;
    for (int j = b0 + threadIdx.x; j < sink_hi; j += blockDim.x) {
        const float e = expf(Ar[j] * scale - mx);
        Ar[j] = e;
        z += e;
    }
    for (int j = win_lo + threadIdx.x; j <= i; j += blockDim.x) {
        const float e = expf(Ar[j] * scale - mx);
        Ar[j] = e;
        z += e;
    }
    sh[threadIdx.x] = z;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    z = sh[0];
    __syncthreads();

    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        const bool vis =
            (j >= b0 && j < sink_hi) || (j >= win_lo && j <= i);
        Ar[j] = vis ? Ar[j] / z : 0.0f;
    }
}

// ---- embedding + cross-entropy (B2.2) -------------------------------
//
// embed gather: out[i,:] = table[ids[i],:]. Bounds are checked by the
// HOST caller before launch (a kernel cannot throw); ids arrive as a
// device int buffer.
__global__ void k_embed_gather(const float* tab, const int* ids, float* out,
                               size_t n, int d) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = idx / d;
        const int j = static_cast<int>(idx % d);
        out[idx] = tab[static_cast<size_t>(ids[i]) * d + j];
    }
}

// CE forward: one block per row. P = softmax(logits) written for the
// backward (the same (P - onehot)/N contract as the host op), per-row
// nll[i] = -log(max(P[i, tgt[i]], 1e-12)) — the exact host formula,
// clamp included.
__global__ void k_ce_fwd(const float* x, const int* tgt, float* P,
                         float* nll, int C) {
    extern __shared__ float sh[];
    const int i = blockIdx.x;
    const float* xr = x + static_cast<size_t>(i) * C;
    float* Pr = P + static_cast<size_t>(i) * C;

    float mx = -3.402823466e+38f;
    for (int j = threadIdx.x; j < C; j += blockDim.x)
        mx = fmaxf(mx, xr[j]);
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mx = sh[0];
    __syncthreads();

    float z = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const float e = expf(xr[j] - mx);
        Pr[j] = e;
        z += e;
    }
    sh[threadIdx.x] = z;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    z = sh[0];
    __syncthreads();

    for (int j = threadIdx.x; j < C; j += blockDim.x) Pr[j] /= z;
    __syncthreads();
    if (threadIdx.x == 0)
        nll[i] = -logf(fmaxf(Pr[tgt[i]], 1e-12f));
}

// Single-block vector sum so the loss leaves the device as ONE float.
__global__ void k_vec_sum(const float* v, float* out, int n) {
    extern __shared__ float sh[];
    float acc = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) acc += v[j];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = sh[0];
}

// CE backward: dl = (P - onehot(tgt)) * g, elementwise.
__global__ void k_ce_bwd(const float* P, const int* tgt, float g, float* dl,
                         int C, size_t n) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = idx / C;
        const int j = static_cast<int>(idx % C);
        dl[idx] = (P[idx] - (j == tgt[i] ? 1.0f : 0.0f)) * g;
    }
}

// ds = A != 0 ? scale * A * (ds - rowdot(ds .* A)) : 0, in place on ds.
// One kernel serves BOTH attention flavors: masked entries carry A == 0
// from the forward, so the full-row dot equals the visible-range dot
// and masked outputs vanish without mask bookkeeping (the same argument
// ops.cpp's host loops rely on).
__global__ void k_attn_softmax_bwd(float* ds, const float* A, float scale,
                                   int T) {
    extern __shared__ float sh[];
    const int i = blockIdx.x;
    float* dr = ds + static_cast<size_t>(i) * T;
    const float* Ar = A + static_cast<size_t>(i) * T;

    float dot = 0.0f;
    for (int j = threadIdx.x; j < T; j += blockDim.x) dot += dr[j] * Ar[j];
    sh[threadIdx.x] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    dot = sh[0];
    __syncthreads();

    for (int j = threadIdx.x; j < T; j += blockDim.x)
        dr[j] = Ar[j] != 0.0f ? scale * Ar[j] * (dr[j] - dot) : 0.0f;
}

// ---- optimizers (B2.3a: write-through parity seam) ------------------
//
// Elementwise, in place on p (and m/v/vel). The bias-correction factors
// c1/c2 are computed HOST-side (std::pow once per step) and passed in,
// so the per-element math is bit-for-bit the nn.cpp loop's formula.
__global__ void k_adamw_step(float* p, const float* g, float* m, float* v,
                             float lr, float b1, float b2, float c1,
                             float c2, float eps, float wd, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float gi = g[i];
        const float mi = b1 * m[i] + (1.0f - b1) * gi;
        const float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        const float update = (mi / c1) / (sqrtf(vi / c2) + eps);
        p[i] -= lr * (update + wd * p[i]);
    }
}

__global__ void k_sgd_step(float* p, const float* g, float* vel, float lr,
                           float mu, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        float gi = g[i];
        if (vel) {
            const float vi = mu * vel[i] + gi;
            vel[i] = vi;
            gi = vi;
        }
        p[i] -= lr * gi;
    }
}

// LayerNorm forward: per row mu, var, rstd; xhat cached for backward
// exactly as the CPU op caches it (ops.cpp layernorm).
__global__ void k_layernorm_fwd(const float* x, const float* gamma,
                                const float* beta, float eps, float* y,
                                float* xhat, float* rstd, int C) {
    extern __shared__ float sh[];
    const float* xr = x + static_cast<size_t>(blockIdx.x) * C;
    float* yr = y + static_cast<size_t>(blockIdx.x) * C;
    float* xh = xhat + static_cast<size_t>(blockIdx.x) * C;

    float acc = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) acc += xr[j];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float mu = sh[0] / static_cast<float>(C);
    __syncthreads();

    acc = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const float d = xr[j] - mu;
        acc += d * d;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float rs =
        rsqrtf(sh[0] / static_cast<float>(C) + eps);
    if (threadIdx.x == 0) rstd[blockIdx.x] = rs;
    __syncthreads();

    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const float h = (xr[j] - mu) * rs;
        xh[j] = h;
        yr[j] = gamma[j] * h + beta[j];
    }
}

// dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat .* xhat)),
// dxhat = dY .* gamma — the exact CPU backward.
__global__ void k_layernorm_bwd_dx(const float* dY, const float* xhat,
                                   const float* rstd, const float* gamma,
                                   float* dx, int C) {
    extern __shared__ float sh[];
    float* sh1 = sh;
    float* sh2 = sh + blockDim.x;
    const float* dYr = dY + static_cast<size_t>(blockIdx.x) * C;
    const float* xh = xhat + static_cast<size_t>(blockIdx.x) * C;
    float* dxr = dx + static_cast<size_t>(blockIdx.x) * C;

    float a1 = 0.0f, a2 = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const float dxh = dYr[j] * gamma[j];
        a1 += dxh;
        a2 += dxh * xh[j];
    }
    sh1[threadIdx.x] = a1;
    sh2[threadIdx.x] = a2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh1[threadIdx.x] += sh1[threadIdx.x + s];
            sh2[threadIdx.x] += sh2[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float m1 = sh1[0] / static_cast<float>(C);
    const float m2 = sh2[0] / static_cast<float>(C);
    const float rs = rstd[blockIdx.x];
    __syncthreads();

    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const float dxh = dYr[j] * gamma[j];
        dxr[j] = rs * (dxh - m1 - xh[j] * m2);
    }
}

// dgamma_j = sum_i dY_ij * xhat_ij ; dbeta_j = sum_i dY_ij.
// One thread per column looping rows: correctness-first (the column
// count bounds parallelism; a tiled transpose-reduce is a measured
// optimization for later, same rule as the strided gemm loads).
__global__ void k_colsum_dgb(const float* dY, const float* xhat, float* dg,
                             float* db, int R, int C) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= C) return;
    float g = 0.0f, b = 0.0f;
    for (int i = 0; i < R; ++i) {
        const float d = dY[static_cast<size_t>(i) * C + j];
        g += d * xhat[static_cast<size_t>(i) * C + j];
        b += d;
    }
    dg[j] = g;
    db[j] = b;
}

// RMSNorm forward: rms_inv_i = 1/sqrt(mean(x_i^2) + eps); y = x*rinv*w.
__global__ void k_rmsnorm_fwd(const float* x, const float* w, float eps,
                              float* y, float* rinv, int C) {
    extern __shared__ float sh[];
    const float* xr = x + static_cast<size_t>(blockIdx.x) * C;
    float* yr = y + static_cast<size_t>(blockIdx.x) * C;

    float acc = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) acc += xr[j] * xr[j];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float ri = rsqrtf(sh[0] / static_cast<float>(C) + eps);
    if (threadIdx.x == 0) rinv[blockIdx.x] = ri;
    __syncthreads();

    for (int j = threadIdx.x; j < C; j += blockDim.x)
        yr[j] = xr[j] * ri * w[j];
}

// dx_ik = rinv_i * (dY_ik w_k - x_ik rinv_i^2 * sum_j(dY_ij w_j x_ij)/C)
__global__ void k_rmsnorm_bwd_dx(const float* dY, const float* x,
                                 const float* rinv, const float* w,
                                 float* dx, int C) {
    extern __shared__ float sh[];
    const float* dYr = dY + static_cast<size_t>(blockIdx.x) * C;
    const float* xr = x + static_cast<size_t>(blockIdx.x) * C;
    float* dxr = dx + static_cast<size_t>(blockIdx.x) * C;

    float acc = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x)
        acc += dYr[j] * w[j] * xr[j];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float term = sh[0];
    const float ri = rinv[blockIdx.x];
    const float n_inv = 1.0f / static_cast<float>(C);
    __syncthreads();

    for (int j = threadIdx.x; j < C; j += blockDim.x)
        dxr[j] = ri * (dYr[j] * w[j] - xr[j] * ri * ri * term * n_inv);
}

// dw_j = sum_i dY_ij * x_ij * rinv_i
__global__ void k_rmsnorm_bwd_dw(const float* dY, const float* x,
                                 const float* rinv, float* dw, int R,
                                 int C) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= C) return;
    float acc = 0.0f;
    for (int i = 0; i < R; ++i)
        acc += dY[static_cast<size_t>(i) * C + j] *
               x[static_cast<size_t>(i) * C + j] * rinv[i];
    dw[j] = acc;
}

int ew_grid(size_t n) {
    const size_t g = (n + NTHREADS - 1) / NTHREADS;
    return static_cast<int>(g > 4096 ? 4096 : g);
}

// Rowwise launches use a power-of-two block (the reductions assume it).
constexpr int ROW_THREADS = 256;

}  // namespace

// ---- entry points ----------------------------------------------------
// Each returns false unless (switch on && device==CUDA); the caller then
// runs today's CPU loop. On the device path the full formula runs in the
// kernel and the result is written through to host storage.

bool add(const Matrix& a, const Matrix& b, Matrix& y) {
    if (!active()) return false;
    const size_t n = a.rows() * a.cols();
    In da(a.get_data(), n), db(b.get_data(), n);
    Out dy(y.get_data(), n);
    k_add<<<ew_grid(n), NTHREADS>>>(da.d, db.d, dy.d, n);
    cuda_check(cudaGetLastError(), "add");
    dy.finish();
    return true;
}

bool sub(const Matrix& a, const Matrix& b, Matrix& y) {
    if (!active()) return false;
    const size_t n = a.rows() * a.cols();
    In da(a.get_data(), n), db(b.get_data(), n);
    Out dy(y.get_data(), n);
    k_sub<<<ew_grid(n), NTHREADS>>>(da.d, db.d, dy.d, n);
    cuda_check(cudaGetLastError(), "sub");
    dy.finish();
    return true;
}

bool mul(const Matrix& a, const Matrix& b, Matrix& y) {
    if (!active()) return false;
    const size_t n = a.rows() * a.cols();
    In da(a.get_data(), n), db(b.get_data(), n);
    Out dy(y.get_data(), n);
    k_mul<<<ew_grid(n), NTHREADS>>>(da.d, db.d, dy.d, n);
    cuda_check(cudaGetLastError(), "mul");
    dy.finish();
    return true;
}

bool scale(const Matrix& a, float s, Matrix& y) {
    if (!active()) return false;
    const size_t n = a.rows() * a.cols();
    In da(a.get_data(), n);
    Out dy(y.get_data(), n);
    k_scale<<<ew_grid(n), NTHREADS>>>(da.d, s, dy.d, n);
    cuda_check(cudaGetLastError(), "scale");
    dy.finish();
    return true;
}

bool axpy(Matrix& y, float a, const Matrix& x) {
    if (!active()) return false;
    const size_t n = y.rows() * y.cols();
    In dx(x.get_data(), n);
    Out dy(y.get_data(), n, /*need_current=*/true);
    k_axpy<<<ew_grid(n), NTHREADS>>>(dy.d, a, dx.d, n);
    cuda_check(cudaGetLastError(), "axpy");
    dy.finish();
    return true;
}

bool fill(Matrix& y, float v) {
    if (!active()) return false;
    const size_t n = y.rows() * y.cols();
    Out dy(y.get_data(), n);
    k_fill<<<ew_grid(n), NTHREADS>>>(dy.d, v, n);
    cuda_check(cudaGetLastError(), "fill");
    dy.finish();
    return true;
}

bool sigmoid_fwd(const Matrix& x, Matrix& y) {
    if (!active()) return false;
    const size_t n = x.rows() * x.cols();
    In dx(x.get_data(), n);
    Out dy(y.get_data(), n);
    k_sigmoid_fwd<<<ew_grid(n), NTHREADS>>>(dx.d, dy.d, n);
    cuda_check(cudaGetLastError(), "sigmoid_fwd");
    dy.finish();
    return true;
}

bool sigmoid_bwd(const Matrix& s, const Matrix& dy, Matrix& dx) {
    if (!active()) return false;
    const size_t n = s.rows() * s.cols();
    In ds(s.get_data(), n), dgy(dy.get_data(), n);
    Out dgx(dx.get_data(), n);
    k_sigmoid_bwd<<<ew_grid(n), NTHREADS>>>(ds.d, dgy.d, dgx.d, n);
    cuda_check(cudaGetLastError(), "sigmoid_bwd");
    dgx.finish();
    return true;
}

bool gelu_fwd(const Matrix& x, Matrix& y) {
    if (!active()) return false;
    const size_t n = x.rows() * x.cols();
    In dx(x.get_data(), n);
    Out dy(y.get_data(), n);
    k_gelu_fwd<<<ew_grid(n), NTHREADS>>>(dx.d, dy.d, n);
    cuda_check(cudaGetLastError(), "gelu_fwd");
    dy.finish();
    return true;
}

bool gelu_bwd(const Matrix& x, const Matrix& dy, Matrix& dx) {
    if (!active()) return false;
    const size_t n = x.rows() * x.cols();
    In dxi(x.get_data(), n), dgy(dy.get_data(), n);
    Out dgx(dx.get_data(), n);
    k_gelu_bwd<<<ew_grid(n), NTHREADS>>>(dxi.d, dgy.d, dgx.d, n);
    cuda_check(cudaGetLastError(), "gelu_bwd");
    dgx.finish();
    return true;
}

bool softmax_fwd(const Matrix& x, Matrix& y) {
    if (!active()) return false;
    const int R = static_cast<int>(x.rows());
    const int C = static_cast<int>(x.cols());
    const size_t n = static_cast<size_t>(R) * C;
    In dx(x.get_data(), n);
    Out dy(y.get_data(), n);
    k_softmax_fwd<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(dx.d, dy.d,
                                                                   C);
    cuda_check(cudaGetLastError(), "softmax_fwd");
    dy.finish();
    return true;
}

bool softmax_bwd(const Matrix& S, const Matrix& dY, Matrix& dX) {
    if (!active()) return false;
    const int R = static_cast<int>(S.rows());
    const int C = static_cast<int>(S.cols());
    const size_t n = static_cast<size_t>(R) * C;
    In ds(S.get_data(), n), dgy(dY.get_data(), n);
    Out dgx(dX.get_data(), n);
    k_softmax_bwd<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(
        ds.d, dgy.d, dgx.d, C);
    cuda_check(cudaGetLastError(), "softmax_bwd");
    dgx.finish();
    return true;
}

// B2.2 masked attention softmax family. A is IN-PLACE (raw scores in,
// weights out): Out with need_current — under deferral the value cache
// already holds the gemm's device-fresh scores, so no H2D happens; the
// weights then stay resident for the A@V gemm and the backward, and the
// [T,T] matrix never crosses the bus inside a step.
bool attn_masked_softmax(Matrix& A, float scale, size_t seq_len,
                         bool causal) {
    if (!active()) return false;
    const int T = static_cast<int>(A.rows());
    if (A.cols() != A.rows()) return false;
    const size_t n = static_cast<size_t>(T) * T;
    Out dA(A.get_data(), n, /*need_current=*/true);
    k_attn_masked_softmax<<<T, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(
        dA.d, scale, static_cast<int>(seq_len), causal ? 1 : 0, T);
    cuda_check(cudaGetLastError(), "attn_masked_softmax");
    dA.finish();
    return true;
}

bool swa_masked_softmax(Matrix& A, float scale, size_t seq_len,
                        size_t window, size_t sinks) {
    if (!active()) return false;
    const int T = static_cast<int>(A.rows());
    if (A.cols() != A.rows()) return false;
    const size_t n = static_cast<size_t>(T) * T;
    Out dA(A.get_data(), n, /*need_current=*/true);
    k_swa_masked_softmax<<<T, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(
        dA.d, scale, static_cast<int>(seq_len), static_cast<int>(window),
        static_cast<int>(sinks), T);
    cuda_check(cudaGetLastError(), "swa_masked_softmax");
    dA.finish();
    return true;
}

// Int upload RAII (ids/targets), mirroring DBuf.
struct IBuf {
    int* d = nullptr;
    IBuf(const int* h, size_t n) {
        cuda_check(cudaMalloc(&d, n * sizeof(int)), "int malloc");
        cuda_check(cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice),
                   "int H2D");
    }
    ~IBuf() {
        if (d) cudaFree(d);
    }
    IBuf(const IBuf&) = delete;
    IBuf& operator=(const IBuf&) = delete;
};

// B2.2 embedding gather: the first activation of every forward is born
// on-device (table via the residency stack — B1-resident hit for a
// parameter — output through the vcache). Backward (scatter-add into
// the host-authoritative table grad) stays host until B2.3.
bool embed_gather(const Matrix& table, const int* ids, size_t n_ids,
                  Matrix& out) {
    if (!active()) return false;
    const int d = static_cast<int>(table.cols());
    const size_t n = n_ids * static_cast<size_t>(d);
    In dt(table.get_data(), table.rows() * table.cols());
    IBuf dids(ids, n_ids);
    Out dout(out.get_data(), n);
    k_embed_gather<<<ew_grid(n), NTHREADS>>>(dt.d, dids.d, dout.d, n, d);
    cuda_check(cudaGetLastError(), "embed_gather");
    dout.finish();
    return true;
}

// B2.2 CE forward: softmax + nll fully on-device; the host receives ONE
// float. P is written through the vcache (deferred: stays resident for
// the backward; write-through: host P is filled, matching the host op).
bool ce_fwd(const Matrix& logits, const int* targets, Matrix& P,
            float& loss) {
    if (!active()) return false;
    const int R = static_cast<int>(logits.rows());
    const int C = static_cast<int>(logits.cols());
    const size_t n = static_cast<size_t>(R) * C;
    In dx(logits.get_data(), n);
    IBuf dt(targets, static_cast<size_t>(R));
    Out dP(P.get_data(), n);
    DBuf dnll(static_cast<size_t>(R));
    DBuf dsum(1);
    k_ce_fwd<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(dx.d, dt.d,
                                                              dP.d, dnll.d,
                                                              C);
    cuda_check(cudaGetLastError(), "ce_fwd");
    k_vec_sum<<<1, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(dnll.d,
                                                               dsum.d, R);
    cuda_check(cudaGetLastError(), "ce_fwd sum");
    float s = 0.0f;
    d2h(&s, dsum.d, 1);
    loss = s / static_cast<float>(R);
    dP.finish();
    return true;
}

// B2.2 CE backward: dl = (P - onehot) * g on-device; dl rides the
// vcache, so the [R,vocab] gradient first touches host at accumulate()
// (host-authoritative until B2.3 moves accumulation on-device).
bool ce_bwd(const Matrix& P, const int* targets, float g, Matrix& dl) {
    if (!active()) return false;
    const int C = static_cast<int>(P.cols());
    const size_t n = static_cast<size_t>(P.rows()) * C;
    In dP(P.get_data(), n);
    IBuf dt(targets, P.rows());
    Out ddl(dl.get_data(), n);
    k_ce_bwd<<<ew_grid(n), NTHREADS>>>(dP.d, dt.d, g, ddl.d, C, n);
    cuda_check(cudaGetLastError(), "ce_bwd");
    ddl.finish();
    return true;
}

// B2.3a optimizer steps: write-through parity seam (p/m/v round-trip
// the bus every step — SLOWER than host on purpose for now; persistent
// device optimizer state and resident params are B2.3b, exactly the
// B2.1a -> B2.1b staging this codebase already follows).
bool adamw_step(Matrix& p, const Matrix& g, Matrix& m, Matrix& v, float lr,
                float b1, float b2, float c1, float c2, float eps,
                float wd) {
    if (!active()) return false;
    const size_t n = p.rows() * p.cols();
    In dg(g.get_data(), n);
    Out dp(p.get_data(), n, /*need_current=*/true);
    Out dm(m.get_data(), n, /*need_current=*/true);
    Out dv(v.get_data(), n, /*need_current=*/true);
    k_adamw_step<<<ew_grid(n), NTHREADS>>>(dp.d, dg.d, dm.d, dv.d, lr, b1,
                                           b2, c1, c2, eps, wd, n);
    cuda_check(cudaGetLastError(), "adamw_step");
    dp.finish();
    dm.finish();
    dv.finish();
    return true;
}

bool sgd_step(Matrix& p, const Matrix& g, Matrix* vel, float lr, float mu) {
    if (!active()) return false;
    const size_t n = p.rows() * p.cols();
    In dg(g.get_data(), n);
    Out dp(p.get_data(), n, /*need_current=*/true);
    if (vel) {
        Out dvel(vel->get_data(), n, /*need_current=*/true);
        k_sgd_step<<<ew_grid(n), NTHREADS>>>(dp.d, dg.d, dvel.d, lr, mu, n);
        cuda_check(cudaGetLastError(), "sgd_step");
        dp.finish();
        dvel.finish();
    } else {
        k_sgd_step<<<ew_grid(n), NTHREADS>>>(dp.d, dg.d, nullptr, lr, mu, n);
        cuda_check(cudaGetLastError(), "sgd_step");
        dp.finish();
    }
    return true;
}

bool attn_softmax_bwd_inplace(Matrix& ds, const Matrix& A, float scale) {
    if (!active()) return false;
    const int T = static_cast<int>(ds.rows());
    if (ds.cols() != ds.rows() || A.rows() != ds.rows() ||
        A.cols() != ds.cols())
        return false;
    const size_t n = static_cast<size_t>(T) * T;
    In dA(A.get_data(), n);
    Out dds(ds.get_data(), n, /*need_current=*/true);
    k_attn_softmax_bwd<<<T, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(
        dds.d, dA.d, scale, T);
    cuda_check(cudaGetLastError(), "attn_softmax_bwd_inplace");
    dds.finish();
    return true;
}

bool layernorm_fwd(const Matrix& x, const Matrix& gamma, const Matrix& beta,
                   float eps, Matrix& y, Matrix& xhat,
                   std::vector<float>& rstd) {
    if (!active()) return false;
    const int R = static_cast<int>(x.rows());
    const int C = static_cast<int>(x.cols());
    const size_t n = static_cast<size_t>(R) * C;
    In dx(x.get_data(), n), dg(gamma.get_data(), C), db(beta.get_data(), C);
    Out dy(y.get_data(), n), dxh(xhat.get_data(), n);
    DBuf drs(static_cast<size_t>(R));
    k_layernorm_fwd<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(
        dx.d, dg.d, db.d, eps, dy.d, dxh.d, drs.d, C);
    cuda_check(cudaGetLastError(), "layernorm_fwd");
    dy.finish();
    dxh.finish();
    d2h(rstd.data(), drs.d, R);
    return true;
}

bool layernorm_bwd(const Matrix& dY, const Matrix& xhat,
                   const std::vector<float>& rstd, const Matrix& gamma,
                   bool want_dgb, Matrix* dg, Matrix* db, bool want_dx,
                   Matrix* dx) {
    if (!active()) return false;
    const int R = static_cast<int>(dY.rows());
    const int C = static_cast<int>(dY.cols());
    const size_t n = static_cast<size_t>(R) * C;
    In ddy(dY.get_data(), n), dxh(xhat.get_data(), n);
    if (want_dgb) {
        Out ddg(dg->get_data(), static_cast<size_t>(C));
        Out ddb(db->get_data(), static_cast<size_t>(C));
        const int grid = (C + NTHREADS - 1) / NTHREADS;
        k_colsum_dgb<<<grid, NTHREADS>>>(ddy.d, dxh.d, ddg.d, ddb.d, R, C);
        cuda_check(cudaGetLastError(), "layernorm_bwd dgb");
        ddg.finish();
        ddb.finish();
    }
    if (want_dx) {
        DBuf drs(rstd.data(), static_cast<size_t>(R));
        In dgm(gamma.get_data(), static_cast<size_t>(C));
        Out ddx(dx->get_data(), n);
        k_layernorm_bwd_dx<<<R, ROW_THREADS, 2 * ROW_THREADS * sizeof(float)>>>(
            ddy.d, dxh.d, drs.d, dgm.d, ddx.d, C);
        cuda_check(cudaGetLastError(), "layernorm_bwd dx");
        ddx.finish();
    }
    return true;
}

bool rmsnorm_fwd(const Matrix& x, const Matrix& w, float eps, Matrix& y,
                 std::vector<float>& rms_inv) {
    if (!active()) return false;
    const int R = static_cast<int>(x.rows());
    const int C = static_cast<int>(x.cols());
    const size_t n = static_cast<size_t>(R) * C;
    In dx(x.get_data(), n), dw(w.get_data(), C);
    Out dy(y.get_data(), n);
    DBuf dri(static_cast<size_t>(R));
    k_rmsnorm_fwd<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(
        dx.d, dw.d, eps, dy.d, dri.d, C);
    cuda_check(cudaGetLastError(), "rmsnorm_fwd");
    dy.finish();
    d2h(rms_inv.data(), dri.d, R);
    return true;
}

bool rmsnorm_bwd(const Matrix& dY, const Matrix& x,
                 const std::vector<float>& rms_inv, const Matrix& w,
                 bool want_dw, Matrix* dw, bool want_dx, Matrix* dx) {
    if (!active()) return false;
    const int R = static_cast<int>(dY.rows());
    const int C = static_cast<int>(dY.cols());
    const size_t n = static_cast<size_t>(R) * C;
    In ddy(dY.get_data(), n), dxi(x.get_data(), n);
    DBuf dri(rms_inv.data(), static_cast<size_t>(R));
    if (want_dw) {
        Out ddw(dw->get_data(), static_cast<size_t>(C));
        const int grid = (C + NTHREADS - 1) / NTHREADS;
        k_rmsnorm_bwd_dw<<<grid, NTHREADS>>>(ddy.d, dxi.d, dri.d, ddw.d, R, C);
        cuda_check(cudaGetLastError(), "rmsnorm_bwd dw");
        ddw.finish();
    }
    if (want_dx) {
        In dwv(w.get_data(), static_cast<size_t>(C));
        Out ddx(dx->get_data(), n);
        k_rmsnorm_bwd_dx<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(
            ddy.d, dxi.d, dri.d, dwv.d, ddx.d, C);
        cuda_check(cudaGetLastError(), "rmsnorm_bwd dx");
        ddx.finish();
    }
    return true;
}

}  // namespace devops
}  // namespace device
}  // namespace microtorch
