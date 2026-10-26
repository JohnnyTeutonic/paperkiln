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

#include "device_pool.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
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
    cuda_check(pool::dmalloc(&d, n * sizeof(float)), "malloc");
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
        if (d) pool::dfree(d);
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
// NOTE the assignment in the BODY, not a mem-init list. `owned` is an
// out-parameter of vc_operand, and members initialize in DECLARATION
// order regardless of mem-init order: with `: d(vc_operand(h, n, owned))`
// the compiler initialized d (setting owned=true), then ran `owned`'s
// default member initializer and reset it to false. ~In therefore never
// freed, and every operand of every device op leaked — 156 MiB/step,
// OOM at step 95 (31 Aug 2026). Assigning in the body is immune to
// declaration order; do not "tidy" this back into an init list.
struct In {
    float* d = nullptr;
    bool owned = false;
    In(const float* h, size_t n) { d = detail::vc_operand(h, n, owned); }
    ~In() {
        if (owned) pool::dfree(d);
    }
    In(const In&) = delete;
    In& operator=(const In&) = delete;
};
struct Out {
    float* d = nullptr;
    float* host = nullptr;
    size_t n = 0;
    bool deferred = false;
    const char* op = nullptr;  // the entry point's name, for MICROTORCH_GPU_PROFILE
    Out(float* h, size_t n_, bool need_current = false, const char* op_ = __builtin_FUNCTION())
        : host(h), n(n_), op(op_) {
        d = detail::vc_output(h, n_, deferred, need_current, h);
        if (!deferred) {
            d = dalloc(n_);
            if (need_current) h2d(d, h, n_);
        }
        detail::gpu_profile_begin(op);
    }
    void finish() {
        detail::gpu_profile_end(op);
        if (!deferred) d2h(host, d, n);
    }
    ~Out() {
        if (!deferred && d) pool::dfree(d);
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
// SiLU (the llama MLP gate), ops.cpp silu: y = x s(x), dy/dx = s (1 + x (1 - s)).
__global__ void k_silu_fwd(const float* x, float* y, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
}
__global__ void k_silu_bwd(const float* x, const float* dy, float* dx, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = x[i];
        const float s = 1.0f / (1.0f + expf(-v));
        dx[i] = dy[i] * (s * (1.0f + v * (1.0f - s)));
    }
}
// SwiGLU (ops::swiglu): k_silu_fwd's silu, then k_mul's product.
__global__ void k_swiglu_fwd(const float* g, const float* u, float* y, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = g[i];
        const float sv = v / (1.0f + expf(-v));
        y[i] = sv * u[i];
    }
}
// du = dy * silu(g) (k_mul on the saved silu), dg = (dy * u) * silu'(g)
// (k_mul, then k_silu_bwd on that product). Either output may be null.
__global__ void k_swiglu_bwd(const float* g, const float* u, const float* dy, float* dg, float* du, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = g[i];
        if (du) du[i] = dy[i] * (v / (1.0f + expf(-v)));
        if (dg) {
            const float da = dy[i] * u[i];
            const float s = 1.0f / (1.0f + expf(-v));
            dg[i] = da * (s * (1.0f + v * (1.0f - s)));
        }
    }
}

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
// kMasked (ops::cross_entropy_masked, 24 Oct 2026): a row whose target is
// negative is masked out: P row and nll written as zeros, no softmax. The
// test is per block (one row), so the whole block leaves together before
// any __syncthreads. kMasked = false compiles to the unmasked kernel
// exactly (the branch is a compile-time constant), so ops::cross_entropy's
// numerics are untouched; a kept row runs the same instructions in both
// instantiations, so its P and nll are bit-identical to the unmasked op's.
template <bool kMasked>
__global__ void k_ce_fwd(const float* x, const int* tgt, float* P,
                         float* nll, int C) {
    extern __shared__ float sh[];
    const int i = blockIdx.x;
    const float* xr = x + static_cast<size_t>(i) * C;
    float* Pr = P + static_cast<size_t>(i) * C;

    if (kMasked && tgt[i] < 0) {
        for (int j = threadIdx.x; j < C; j += blockDim.x) Pr[j] = 0.0f;
        if (threadIdx.x == 0) nll[i] = 0.0f;
        return;
    }

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

// CE backward: dl = (P - onehot(tgt)) * g, elementwise. kMasked: a row
// with a negative target gets an exact zero gradient (the host op's rule).
template <bool kMasked>
__global__ void k_ce_bwd(const float* P, const int* tgt, float g, float* dl,
                         int C, size_t n) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = idx / C;
        const int j = static_cast<int>(idx % C);
        if (kMasked && tgt[i] < 0) {
            dl[idx] = 0.0f;
            continue;
        }
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

bool silu_fwd(const Matrix& x, Matrix& y) {
    if (!active()) return false;
    const size_t n = x.rows() * x.cols();
    In dx(x.get_data(), n);
    Out dy(y.get_data(), n);
    k_silu_fwd<<<ew_grid(n), NTHREADS>>>(dx.d, dy.d, n);
    cuda_check(cudaGetLastError(), "silu_fwd");
    dy.finish();
    return true;
}

bool silu_bwd(const Matrix& x, const Matrix& dy, Matrix& dx) {
    if (!active()) return false;
    const size_t n = x.rows() * x.cols();
    In dxi(x.get_data(), n), dgy(dy.get_data(), n);
    Out dgx(dx.get_data(), n);
    k_silu_bwd<<<ew_grid(n), NTHREADS>>>(dxi.d, dgy.d, dgx.d, n);
    cuda_check(cudaGetLastError(), "silu_bwd");
    dgx.finish();
    return true;
}

bool swiglu_fwd(const Matrix& g, const Matrix& u, Matrix& y) {
    if (!active()) return false;
    const size_t n = g.rows() * g.cols();
    In dg(g.get_data(), n), du(u.get_data(), n);
    Out dy(y.get_data(), n);
    k_swiglu_fwd<<<ew_grid(n), NTHREADS>>>(dg.d, du.d, dy.d, n);
    cuda_check(cudaGetLastError(), "swiglu_fwd");
    dy.finish();
    return true;
}

bool swiglu_bwd(const Matrix& g, const Matrix& u, const Matrix& dy, Matrix* dg, Matrix* du) {
    if (!active()) return false;
    const size_t n = g.rows() * g.cols();
    In ig(g.get_data(), n), iu(u.get_data(), n), idy(dy.get_data(), n);
    // Outputs: an Out per wanted gradient (finish() downloads it when not deferring).
    std::unique_ptr<Out> og, ou;
    if (dg) og.reset(new Out(dg->get_data(), n, false, "swiglu_bwd"));
    if (du) ou.reset(new Out(du->get_data(), n, false, "swiglu_bwd"));
    k_swiglu_bwd<<<ew_grid(n), NTHREADS>>>(ig.d, iu.d, idy.d, og ? og->d : nullptr, ou ? ou->d : nullptr, n);
    cuda_check(cudaGetLastError(), "swiglu_bwd");
    if (og) og->finish();
    if (ou) ou->finish();
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
        cuda_check(pool::dmalloc(&d, n * sizeof(int)), "int malloc");
        cuda_check(cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice),
                   "int H2D");
    }
    ~IBuf() {
        if (d) pool::dfree(d);
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
    k_ce_fwd<false><<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(
        dx.d, dt.d, dP.d, dnll.d, C);
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

// Masked CE forward (ops::cross_entropy_masked, 24 Oct 2026): targets
// carry -1 on masked rows. Masked rows add an exact 0.0f to the sum, and
// the mean is over the `kept` rows. Before this the masked op had no
// device path: every SFT step downloaded the [R, vocab] logits, ran the
// softmax on the host and uploaded the gradient (1 GiB each way at
// batch 32 x T 512 x 16k vocab).
bool ce_fwd_masked(const Matrix& logits, const int* targets, size_t kept,
                   Matrix& P, float& loss) {
    if (!active()) return false;
    const int R = static_cast<int>(logits.rows());
    const int C = static_cast<int>(logits.cols());
    const size_t n = static_cast<size_t>(R) * C;
    In dx(logits.get_data(), n);
    IBuf dt(targets, static_cast<size_t>(R));
    Out dP(P.get_data(), n);
    DBuf dnll(static_cast<size_t>(R));
    DBuf dsum(1);
    k_ce_fwd<true><<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(
        dx.d, dt.d, dP.d, dnll.d, C);
    cuda_check(cudaGetLastError(), "ce_fwd_masked");
    k_vec_sum<<<1, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(dnll.d,
                                                               dsum.d, R);
    cuda_check(cudaGetLastError(), "ce_fwd_masked sum");
    float s = 0.0f;
    d2h(&s, dsum.d, 1);
    loss = s / static_cast<float>(kept);
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
    k_ce_bwd<false><<<ew_grid(n), NTHREADS>>>(dP.d, dt.d, g, ddl.d, C, n);
    cuda_check(cudaGetLastError(), "ce_bwd");
    ddl.finish();
    return true;
}

// Masked CE backward: as ce_bwd, rows with target -1 written as zeros.
bool ce_bwd_masked(const Matrix& P, const int* targets, float g, Matrix& dl) {
    if (!active()) return false;
    const int C = static_cast<int>(P.cols());
    const size_t n = static_cast<size_t>(P.rows()) * C;
    In dP(P.get_data(), n);
    IBuf dt(targets, P.rows());
    Out ddl(dl.get_data(), n);
    k_ce_bwd<true><<<ew_grid(n), NTHREADS>>>(dP.d, dt.d, g, ddl.d, C, n);
    cuda_check(cudaGetLastError(), "ce_bwd_masked");
    ddl.finish();
    return true;
}

namespace {
// ops::add_bias's forward on a chunk of logits: L[i, j] += b[j].
__global__ void k_bias_rows(float* L, const float* b, size_t n, int V) {
    for (size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; idx < n;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x)
        L[idx] += b[idx % V];
}
// ops::add_bias's backward over a chunk, continued across chunks: per column,
// the rows added in ascending order to the running sum (first chunk: from 0),
// one float add each, the host loop's sequence.
__global__ void k_colsum_acc(const float* dl, float* db, int rows, int V, int first) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < V; j += gridDim.x * blockDim.x) {
        float s = first ? 0.0f : db[j];
        for (int r = 0; r < rows; ++r) s += dl[static_cast<size_t>(r) * V + j];
        db[j] = s;
    }
}
// A step_operand fetch, freed on every exit.
struct StepIn {
    float* d = nullptr;
    bool owned = false;
    StepIn(const Matrix& m, DevState** slot) { d = detail::step_operand(m, slot, owned); }
    ~StepIn() {
        if (owned) pool::dfree(d);
    }
    StepIn(const StepIn&) = delete;
    StepIn& operator=(const StepIn&) = delete;
};
}  // namespace

// The fused LM head + cross-entropy (device_cache.hpp; ops::linear_cross_entropy).
// Not yet run on a GPU (26 Oct 2026): tests/test_fused_head.cpp's device legs
// are its gate.
bool linear_ce(const Matrix& h, DevState** h_slot, const Matrix& w, DevState** w_slot, bool vocab_major,
               const Matrix* bias, const int* targets, size_t kept, float g, size_t chunk_rows, Precision prec,
               Matrix* dh, Matrix* dw, Matrix* db, float& loss) {
    if (!active()) return false;
    static const char* const kOp = "linear_ce";  // MICROTORCH_GPU_PROFILE's name for the whole op
    const size_t R = h.rows(), D = h.cols();
    const size_t V = vocab_major ? w.rows() : w.cols();
    const size_t chunk = std::max<size_t>(1, std::min(chunk_rows, R));
    const bool masked = kept != R;
    const bool grads = dh || dw || db;
    detail::gpu_profile_begin(kOp);
    StepIn dH(h, h_slot);
    StepIn dW(w, w_slot);
    std::unique_ptr<In> dB;
    if (bias) dB.reset(new In(bias->get_data(), V));
    IBuf dt(targets, R);
    DBuf L(chunk * V);
    DBuf dnll(R);
    DBuf dsum(1);
    std::unique_ptr<Out> odh, odw, odb;
    if (dh) odh.reset(new Out(dh->get_data(), R * D, false, kOp));
    if (dw) odw.reset(new Out(dw->get_data(), V * D, false, kOp));
    if (db) odb.reset(new Out(db->get_data(), V, false, kOp));
    const int iV = static_cast<int>(V), iD = static_cast<int>(D);
    for (size_t c0 = 0; c0 < R; c0 += chunk) {
        const size_t n = std::min(chunk, R - c0);
        const int in = static_cast<int>(n);
        const float* hc = dH.d + c0 * D;
        // logits [n, V] = h_c W (W [D, V]: B stored [K, N]) or h_c E^T (E [V, D]: B stored [N, K])
        detail::gemm_raw(hc, false, dW.d, vocab_major, L.d, in, iV, iD, prec, false);
        if (bias) {
            k_bias_rows<<<ew_grid(n * V), NTHREADS>>>(L.d, dB->d, n * V, iV);
            cuda_check(cudaGetLastError(), "linear_ce bias");
        }
        // The unfused CE kernels, in place on the chunk (each thread reads an
        // element before it writes it): P and nll, then (P - onehot) g.
        if (masked)
            k_ce_fwd<true><<<in, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(L.d, dt.d + c0, L.d, dnll.d + c0, iV);
        else
            k_ce_fwd<false><<<in, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(L.d, dt.d + c0, L.d, dnll.d + c0, iV);
        cuda_check(cudaGetLastError(), "linear_ce softmax");
        if (!grads) continue;
        if (masked)
            k_ce_bwd<true><<<ew_grid(n * V), NTHREADS>>>(L.d, dt.d + c0, g, L.d, iV, n * V);
        else
            k_ce_bwd<false><<<ew_grid(n * V), NTHREADS>>>(L.d, dt.d + c0, g, L.d, iV, n * V);
        cuda_check(cudaGetLastError(), "linear_ce dlogits");
        // dh rows = dl W^T (W [D, V] read as [N, K]) or dl E (E [V, D] read as [K, N])
        if (odh) detail::gemm_raw(L.d, false, dW.d, !vocab_major, odh->d + c0 * D, in, iD, iV, prec, false);
        // dW [D, V] += h_c^T dl, or dE [V, D] += dl^T h_c; the first chunk writes.
        if (odw) {
            if (!vocab_major)
                detail::gemm_raw(hc, true, L.d, false, odw->d, iD, iV, in, prec, c0 > 0);
            else
                detail::gemm_raw(L.d, true, hc, false, odw->d, iV, iD, in, prec, c0 > 0);
        }
        if (odb) {
            k_colsum_acc<<<ew_grid(V), NTHREADS>>>(L.d, odb->d, in, iV, c0 == 0 ? 1 : 0);
            cuda_check(cudaGetLastError(), "linear_ce bias grad");
        }
    }
    k_vec_sum<<<1, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(dnll.d, dsum.d, static_cast<int>(R));
    cuda_check(cudaGetLastError(), "linear_ce sum");
    float s = 0.0f;
    d2h(&s, dsum.d, 1);
    loss = s / static_cast<float>(kept);
    if (odh) odh->finish();
    if (odw) odw->finish();
    if (odb) odb->finish();
    detail::gpu_profile_end(kOp);
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

// B2.3b: persistent device optimizer state — owned, zeroed, freed by
// the optimizer (shared_ptr deleter), NEVER the value cache.
float* opt_state_new(size_t n_elems) {
    if (!active()) return nullptr;
    float* d = dalloc(n_elems);
    cuda_check(cudaMemset(d, 0, n_elems * sizeof(float)), "opt_state memset");
    return d;
}

void opt_state_free(float* s) {
    if (s) pool::dfree(s);
}

void opt_state_download(float* host, const float* dev, size_t n_elems) {
    if (dev && host && n_elems) d2h(host, dev, n_elems);
}

void opt_state_upload(float* dev, const float* host, size_t n_elems) {
    if (dev && host && n_elems) h2d(dev, host, n_elems);
}

bool adamw_step_dev(Matrix& p, const Matrix& g, float* m_dev, float* v_dev,
                    float lr, float b1, float b2, float c1, float c2,
                    float eps, float wd) {
    if (!active()) return false;
    const size_t n = p.rows() * p.cols();
    In dg(g.get_data(), n);
    Out dp(p.get_data(), n, /*need_current=*/true);
    k_adamw_step<<<ew_grid(n), NTHREADS>>>(dp.d, dg.d, m_dev, v_dev, lr, b1,
                                           b2, c1, c2, eps, wd, n);
    cuda_check(cudaGetLastError(), "adamw_step_dev");
    dp.finish();
    return true;
}

bool sgd_step_dev(Matrix& p, const Matrix& g, float* vel_dev, float lr,
                  float mu) {
    if (!active()) return false;
    const size_t n = p.rows() * p.cols();
    In dg(g.get_data(), n);
    Out dp(p.get_data(), n, /*need_current=*/true);
    k_sgd_step<<<ew_grid(n), NTHREADS>>>(dp.d, dg.d, vel_dev, lr, mu, n);
    cuda_check(cudaGetLastError(), "sgd_step_dev");
    dp.finish();
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

namespace {
// The shared body of both rmsnorm_bwd overloads: rms_inv already on device.
void rmsnorm_bwd_dev(const float* ddy, const float* dxi, const float* dri, const Matrix& dY, const Matrix& w,
                     bool want_dw, Matrix* dw, bool want_dx, Matrix* dx);
}  // namespace

bool rmsnorm_fwd(const Matrix& x, const Matrix& w, float eps, Matrix& y, Matrix& rms_inv) {
    if (!active()) return false;
    const int R = static_cast<int>(x.rows());
    const int C = static_cast<int>(x.cols());
    if (rms_inv.rows() != x.rows() || rms_inv.cols() != 1) return false;
    const size_t n = static_cast<size_t>(R) * C;
    In dx(x.get_data(), n), dw(w.get_data(), C);
    Out dy(y.get_data(), n), dri(rms_inv.get_data(), static_cast<size_t>(R));
    k_rmsnorm_fwd<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(dx.d, dw.d, eps, dy.d, dri.d, C);
    cuda_check(cudaGetLastError(), "rmsnorm_fwd");
    dy.finish();
    dri.finish();
    return true;
}

bool rmsnorm_bwd(const Matrix& dY, const Matrix& x, const Matrix& rms_inv, const Matrix& w, bool want_dw,
                 Matrix* dw, bool want_dx, Matrix* dx) {
    if (!active()) return false;
    if (rms_inv.rows() != dY.rows() || rms_inv.cols() != 1) return false;
    const size_t n = dY.rows() * dY.cols();
    In ddy(dY.get_data(), n), dxi(x.get_data(), n), dri(rms_inv.get_data(), dY.rows());
    rmsnorm_bwd_dev(ddy.d, dxi.d, dri.d, dY, w, want_dw, dw, want_dx, dx);
    return true;
}

// ops::rmsnorm_group: the rmsnorm kernels over the same row-major buffer
// read as [rows * groups, group] (a group is a contiguous run of a row).
bool rmsnorm_group_fwd(const Matrix& x, const Matrix& w, size_t group, float eps, Matrix& y, Matrix& rms_inv) {
    if (!active()) return false;
    if (group == 0 || x.cols() % group != 0 || w.cols() != group) return false;
    const size_t n = x.rows() * x.cols(), Rg = n / group;
    if (rms_inv.rows() != Rg || rms_inv.cols() != 1) return false;
    In dx(x.get_data(), n), dw(w.get_data(), group);
    Out dy(y.get_data(), n), dri(rms_inv.get_data(), Rg);
    k_rmsnorm_fwd<<<static_cast<int>(Rg), ROW_THREADS, ROW_THREADS * sizeof(float)>>>(dx.d, dw.d, eps, dy.d, dri.d,
                                                                                         static_cast<int>(group));
    cuda_check(cudaGetLastError(), "rmsnorm_group_fwd");
    dy.finish();
    dri.finish();
    return true;
}

bool rmsnorm_group_bwd(const Matrix& dY, const Matrix& x, const Matrix& rms_inv, const Matrix& w, size_t group,
                       bool want_dw, Matrix* dw, bool want_dx, Matrix* dx) {
    if (!active()) return false;
    if (group == 0 || dY.cols() % group != 0 || w.cols() != group) return false;
    const size_t n = dY.rows() * dY.cols(), Rg = n / group;
    if (rms_inv.rows() != Rg || rms_inv.cols() != 1) return false;
    In ddy(dY.get_data(), n), dxi(x.get_data(), n), dri(rms_inv.get_data(), Rg);
    const int R = static_cast<int>(Rg), C = static_cast<int>(group);
    if (want_dw) {
        Out ddw(dw->get_data(), group, false, "rmsnorm_group_bwd");
        k_rmsnorm_bwd_dw<<<(C + NTHREADS - 1) / NTHREADS, NTHREADS>>>(ddy.d, dxi.d, dri.d, ddw.d, R, C);
        cuda_check(cudaGetLastError(), "rmsnorm_group_bwd dw");
        ddw.finish();
    }
    if (want_dx) {
        In dwv(w.get_data(), group);
        Out ddx(dx->get_data(), n, false, "rmsnorm_group_bwd");
        k_rmsnorm_bwd_dx<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(ddy.d, dxi.d, dri.d, dwv.d, ddx.d, C);
        cuda_check(cudaGetLastError(), "rmsnorm_group_bwd dx");
        ddx.finish();
    }
    return true;
}

bool rmsnorm_bwd(const Matrix& dY, const Matrix& x,
                 const std::vector<float>& rms_inv, const Matrix& w,
                 bool want_dw, Matrix* dw, bool want_dx, Matrix* dx) {
    if (!active()) return false;
    const size_t n = dY.rows() * dY.cols();
    In ddy(dY.get_data(), n), dxi(x.get_data(), n);
    DBuf dri(rms_inv.data(), dY.rows());
    rmsnorm_bwd_dev(ddy.d, dxi.d, dri.d, dY, w, want_dw, dw, want_dx, dx);
    return true;
}

namespace {
void rmsnorm_bwd_dev(const float* ddy, const float* dxi, const float* dri, const Matrix& dY, const Matrix& w,
                     bool want_dw, Matrix* dw, bool want_dx, Matrix* dx) {
    const int R = static_cast<int>(dY.rows());
    const int C = static_cast<int>(dY.cols());
    const size_t n = static_cast<size_t>(R) * C;
    if (want_dw) {
        Out ddw(dw->get_data(), static_cast<size_t>(C), false, "rmsnorm_bwd");
        const int grid = (C + NTHREADS - 1) / NTHREADS;
        k_rmsnorm_bwd_dw<<<grid, NTHREADS>>>(ddy, dxi, dri, ddw.d, R, C);
        cuda_check(cudaGetLastError(), "rmsnorm_bwd dw");
        ddw.finish();
    }
    if (want_dx) {
        In dwv(w.get_data(), static_cast<size_t>(C));
        Out ddx(dx->get_data(), n, false, "rmsnorm_bwd");
        k_rmsnorm_bwd_dx<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(ddy, dxi, dri, dwv.d, ddx.d, C);
        cuda_check(cudaGetLastError(), "rmsnorm_bwd dx");
        ddx.finish();
    }
}
}  // namespace

// ---- blocked attention (18 Oct 2026) ---------------------------------
// The device side of ops::fused_attention_blocked. The rows of q, k, v
// are B stacked sequences of length sl; row i attends only within its own
// block (b0 = i - i % sl), causally if asked. A is kept BLOCK-LOCAL,
// [T, sl]: A(i, j) is the weight of key row b0 + j. Same formulas as the
// host loops in ops.cpp (scores * scale, max-shifted exp, masked entries
// hard zeros; backward dV = A^T dY, dA = dY V^T, ds = scale * A .* (dA -
// rowsum(dA .* A)), dQ = ds K, dK = ds^T Q), summed in a different order,
// so equal to rounding, not bit for bit. Measured reason (gdb samples of
// the chat run on an L4): the host path was ~44% of a training step.
// The tiled kernels below (20 Oct 2026) keep the first kernels' bits.

namespace {

__device__ inline int blk_hi(int li, int sl, int causal) { return causal ? li + 1 : sl; }

// Block reduction over blockDim.x (a power of two) in red[]; returns the
// result to every thread.
__device__ inline float blk_reduce(float v, float* red, bool is_max) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            red[threadIdx.x] = is_max ? fmaxf(red[threadIdx.x], red[threadIdx.x + s])
                                      : red[threadIdx.x] + red[threadIdx.x + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

// Tiled products (20 Oct 2026). The first kernels gave every query row its
// own block and read keys/values with one thread per key row: 690 ms of a
// 1.37 s d512 L8 step on the GPU went to attention. These tile 64 x 64
// outputs per 256-thread block through shared memory, and every output
// element still runs the first kernels' exact chain, acc = fmaf(x, y, acc)
// from 0.0f over the same index range in ascending order, so results are
// unchanged bit for bit. Tiles never straddle two sequences: blockIdx.x
// enumerates (sequence, 64-row tile within it).
constexpr int AT = 64;  // output tile edge
constexpr int AK = 16;  // reduction chunk staged per pass

// S(b0 + li, lj) = scale * sum_d P(b0 + li, d) Q(b0 + lj, d) for lj < hi(li);
// masked entries are not written. P, Q are [T, D] with row strides ldp, ldq.
// blockIdx.z is the head: each pointer advances by its head stride (hs*),
// so one launch covers every head of a fused [T, 3d] qkv (ops::attention_heads);
// gridDim.z = 1 with zero strides is the single-head call.
__global__ void k_blk_scores(const float* P, int ldp, const float* Q, int ldq, float* S, float scale, int sl,
                             int causal, int D, size_t hsP, size_t hsQ, size_t hsS) {
    __shared__ float Ps[AK][AT];
    __shared__ float Qs[AK][AT];
    P += blockIdx.z * hsP;
    Q += blockIdx.z * hsQ;
    S += blockIdx.z * hsS;
    const int nt = (sl + AT - 1) / AT;
    const int b0 = (blockIdx.x / nt) * sl, q0 = (blockIdx.x % nt) * AT, k0 = blockIdx.y * AT;
    if (causal && k0 > q0 + AT - 1) return;  // the whole tile is above the diagonal
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    float acc[4][4] = {};
    for (int d0 = 0; d0 < D; d0 += AK) {
        for (int idx = threadIdx.x; idx < AT * AK; idx += blockDim.x) {
            const int r = idx / AK, dd = idx % AK, d = d0 + dd;
            Ps[dd][r] = (q0 + r < sl && d < D) ? P[static_cast<size_t>(b0 + q0 + r) * ldp + d] : 0.0f;
            Qs[dd][r] = (k0 + r < sl && d < D) ? Q[static_cast<size_t>(b0 + k0 + r) * ldq + d] : 0.0f;
        }
        __syncthreads();
        const int dn = min(AK, D - d0);
        for (int dd = 0; dd < dn; ++dd)
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const float p = Ps[dd][ty + 16 * a];
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[a][c] = fmaf(p, Qs[dd][tx + 16 * c], acc[a][c]);
            }
        __syncthreads();
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int li = q0 + ty + 16 * a;
        if (li >= sl) continue;
        const int hi = blk_hi(li, sl, causal);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int lj = k0 + tx + 16 * c;
            if (lj < hi) S[static_cast<size_t>(b0 + li) * sl + lj] = acc[a][c] * scale;
        }
    }
}

// C(b0 + li, n) = sum_{j < hi(li)} X(b0 + li, j) Y(b0 + j, n): A V and ds K.
// X is block-local [T, sl]; Y and C are [T, N] with row strides ldy, ldc.
// Head strides as k_blk_scores.
__global__ void k_blk_rowprod(const float* X, const float* Y, int ldy, float* C, int ldc, int sl, int causal, int N,
                              size_t hsX, size_t hsY, size_t hsC) {
    __shared__ float Xs[AK][AT];
    __shared__ float Ys[AK][AT];
    X += blockIdx.z * hsX;
    Y += blockIdx.z * hsY;
    C += blockIdx.z * hsC;
    const int nt = (sl + AT - 1) / AT;
    const int b0 = (blockIdx.x / nt) * sl, q0 = (blockIdx.x % nt) * AT, n0 = blockIdx.y * AT;
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    int hi[4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int li = q0 + ty + 16 * a;
        hi[a] = li < sl ? blk_hi(li, sl, causal) : 0;
    }
    const int jmax = causal ? min(sl, q0 + AT) : sl;
    float acc[4][4] = {};
    for (int j0 = 0; j0 < jmax; j0 += AK) {
        for (int idx = threadIdx.x; idx < AT * AK; idx += blockDim.x) {
            const int r = idx / AK, jj = idx % AK, j = j0 + jj;
            Xs[jj][r] = (q0 + r < sl && j < sl) ? X[static_cast<size_t>(b0 + q0 + r) * sl + j] : 0.0f;
            const int jy = idx / AT, c = idx % AT;
            Ys[jy][c] = (j0 + jy < sl && n0 + c < N) ? Y[static_cast<size_t>(b0 + j0 + jy) * ldy + n0 + c] : 0.0f;
        }
        __syncthreads();
        const int jn = min(AK, jmax - j0);
        for (int jj = 0; jj < jn; ++jj) {
            const int j = j0 + jj;
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                if (j >= hi[a]) continue;
                const float x = Xs[jj][ty + 16 * a];
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[a][c] = fmaf(x, Ys[jj][tx + 16 * c], acc[a][c]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int li = q0 + ty + 16 * a;
        if (li >= sl) continue;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int n = n0 + tx + 16 * c;
            if (n < N) C[static_cast<size_t>(b0 + li) * ldc + n] = acc[a][c];
        }
    }
}

// C(b0 + lj, n) = sum_{li = lo(lj)}^{sl - 1} X(b0 + li, lj) Y(b0 + li, n),
// lo = lj when causal else 0: ds^T Q and A^T dY, over the queries that see
// key lj, ascending. Strides as k_blk_rowprod.
__global__ void k_blk_colprod(const float* X, const float* Y, int ldy, float* C, int ldc, int sl, int causal, int N,
                              size_t hsX, size_t hsY, size_t hsC) {
    __shared__ float Xs[AK][AT];
    __shared__ float Ys[AK][AT];
    X += blockIdx.z * hsX;
    Y += blockIdx.z * hsY;
    C += blockIdx.z * hsC;
    const int nt = (sl + AT - 1) / AT;
    const int b0 = (blockIdx.x / nt) * sl, k0 = (blockIdx.x % nt) * AT, n0 = blockIdx.y * AT;
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    int lo[4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int lj = k0 + ty + 16 * a;
        lo[a] = lj < sl ? (causal ? lj : 0) : sl;  // rows past the sequence take nothing
    }
    float acc[4][4] = {};
    for (int i0 = causal ? k0 : 0; i0 < sl; i0 += AK) {
        for (int idx = threadIdx.x; idx < AT * AK; idx += blockDim.x) {
            const int ii = idx / AT, r = idx % AT, li = i0 + ii;
            Xs[ii][r] = (li < sl && k0 + r < sl) ? X[static_cast<size_t>(b0 + li) * sl + k0 + r] : 0.0f;
            Ys[ii][r] = (li < sl && n0 + r < N) ? Y[static_cast<size_t>(b0 + li) * ldy + n0 + r] : 0.0f;
        }
        __syncthreads();
        const int in = min(AK, sl - i0);
        for (int ii = 0; ii < in; ++ii) {
            const int li = i0 + ii;
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                if (li < lo[a]) continue;
                const float x = Xs[ii][ty + 16 * a];
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[a][c] = fmaf(x, Ys[ii][tx + 16 * c], acc[a][c]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int lj = k0 + ty + 16 * a;
        if (lj >= sl) continue;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int n = n0 + tx + 16 * c;
            if (n < N) C[static_cast<size_t>(b0 + lj) * ldc + n] = acc[a][c];
        }
    }
}

// Row i of the scaled scores (already in A for j < hi) -> softmax weights:
// max-shifted exp, masked entries hard zeros. One block per row, the first
// kernel's thread-to-column map and reduction tree, so the same bits.
__global__ void k_blk_softmax_rows(float* A, int sl, int causal) {
    extern __shared__ float red[];
    const int i = blockIdx.x, li = i % sl, hi = blk_hi(li, sl, causal);
    float* Ar = A + static_cast<size_t>(i) * sl;
    float mx = -1e30f;
    for (int j = threadIdx.x; j < hi; j += blockDim.x) mx = fmaxf(mx, Ar[j]);
    mx = blk_reduce(mx, red, true);
    float z = 0.0f;
    for (int j = threadIdx.x; j < hi; j += blockDim.x) {
        const float e = expf(Ar[j] - mx);
        Ar[j] = e;
        z += e;
    }
    z = blk_reduce(z, red, false);
    for (int j = threadIdx.x; j < sl; j += blockDim.x) Ar[j] = j < hi ? Ar[j] / z : 0.0f;
}

// ds(i, j) = scale * A(i, j) * (dA(i, j) - sum_j' A(i, j') dA(i, j')), with
// dA = dY V^T already in ds for j < hi. Same map and tree as the first kernel.
__global__ void k_blk_ds_rows(const float* A, float* ds, float scale, int sl, int causal) {
    extern __shared__ float red[];
    const int i = blockIdx.x, li = i % sl, hi = blk_hi(li, sl, causal);
    const float* Ar = A + static_cast<size_t>(i) * sl;
    float* dr = ds + static_cast<size_t>(i) * sl;
    float part = 0.0f;
    for (int j = threadIdx.x; j < hi; j += blockDim.x) part += dr[j] * Ar[j];
    const float dot = blk_reduce(part, red, false);
    for (int j = threadIdx.x; j < sl; j += blockDim.x) dr[j] = j < hi ? scale * Ar[j] * (dr[j] - dot) : 0.0f;
}

// The turn_state mask (ops::attention_heads_turn_state, ir.hpp): row i of
// the head-major [H T, sl] scores, key j <= i visible iff code[j] is odd (a
// state position) or code[j] / 2 == code[i] / 2 (the same segment). The
// scores kernel wrote every j <= i (causal); this is k_blk_softmax_rows with
// the visibility test, masked entries hard zeros, so the causal products and
// the causal backward are the turn_state ones.
__global__ void k_blk_softmax_rows_ts(float* A, int sl, int T, const int* code) {
    extern __shared__ float red[];
    const int i = blockIdx.x, r = i % T, li = r % sl, b0 = r - li, hi = li + 1;
    const int ci = code[r] >> 1;
    float* Ar = A + static_cast<size_t>(i) * sl;
    float mx = -1e30f;
    for (int j = threadIdx.x; j < hi; j += blockDim.x) {
        const int cj = code[b0 + j];
        if ((cj & 1) || (cj >> 1) == ci) mx = fmaxf(mx, Ar[j]);
    }
    mx = blk_reduce(mx, red, true);
    float z = 0.0f;
    for (int j = threadIdx.x; j < hi; j += blockDim.x) {
        const int cj = code[b0 + j];
        if ((cj & 1) || (cj >> 1) == ci) {
            const float e = expf(Ar[j] - mx);
            Ar[j] = e;
            z += e;
        }
    }
    z = blk_reduce(z, red, false);
    for (int j = threadIdx.x; j < sl; j += blockDim.x) {
        const int cj = j < hi ? code[b0 + j] : 0;
        Ar[j] = j < hi && ((cj & 1) || (cj >> 1) == ci) ? Ar[j] / z : 0.0f;
    }
}

dim3 blk_grid(int T, int sl, int N, int heads = 1) {
    const int nt = (sl + AT - 1) / AT;
    return dim3(static_cast<unsigned>((T / sl) * nt), static_cast<unsigned>((N + AT - 1) / AT),
                static_cast<unsigned>(heads));
}

constexpr int AT_THREADS = 256;  // 16 x 16, four by four outputs each

int col_threads(int n) {
    int t = 32;
    while (t < n && t < 256) t <<= 1;
    return t;
}

}  // namespace

bool attn_blocked_fwd(const Matrix& q, const Matrix& k, const Matrix& v, float scale, size_t seq_len,
                      bool causal, Matrix& A, Matrix& y) {
    if (!active()) return false;
    const int T = static_cast<int>(q.rows()), dk = static_cast<int>(q.cols()), dv = static_cast<int>(v.cols());
    const int sl = static_cast<int>(seq_len);
    if (sl <= 0 || T % sl != 0 || k.rows() != q.rows() || k.cols() != q.cols() || v.rows() != q.rows() ||
        A.rows() != q.rows() || A.cols() != seq_len || y.rows() != q.rows() || y.cols() != v.cols())
        return false;
    In dq(q.get_data(), static_cast<size_t>(T) * dk), dkk(k.get_data(), static_cast<size_t>(T) * dk),
        dvv(v.get_data(), static_cast<size_t>(T) * dv);
    Out dA(A.get_data(), static_cast<size_t>(T) * sl), dy(y.get_data(), static_cast<size_t>(T) * dv);
    const int c = causal ? 1 : 0;
    k_blk_scores<<<blk_grid(T, sl, sl), AT_THREADS>>>(dq.d, dk, dkk.d, dk, dA.d, scale, sl, c, dk, 0, 0, 0);
    cuda_check(cudaGetLastError(), "attn_blocked_fwd scores");
    k_blk_softmax_rows<<<T, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(dA.d, sl, c);
    cuda_check(cudaGetLastError(), "attn_blocked_fwd softmax");
    k_blk_rowprod<<<blk_grid(T, sl, dv), AT_THREADS>>>(dA.d, dvv.d, dv, dy.d, dv, sl, c, dv, 0, 0, 0);
    cuda_check(cudaGetLastError(), "attn_blocked_fwd av");
    dA.finish();
    dy.finish();
    return true;
}

bool attn_blocked_bwd(const Matrix& q, const Matrix& k, const Matrix& v, const Matrix& A, const Matrix& dY,
                      float scale, size_t seq_len, bool causal, Matrix& dq, Matrix& dk, Matrix& dv) {
    if (!active()) return false;
    const int T = static_cast<int>(q.rows()), dkd = static_cast<int>(q.cols()), dvd = static_cast<int>(v.cols());
    const int sl = static_cast<int>(seq_len);
    if (sl <= 0 || T % sl != 0 || A.rows() != q.rows() || A.cols() != seq_len || dY.rows() != q.rows() ||
        dY.cols() != v.cols() || dq.rows() != q.rows() || dq.cols() != q.cols() || dk.rows() != k.rows() ||
        dk.cols() != k.cols() || dv.rows() != v.rows() || dv.cols() != v.cols())
        return false;
    const size_t nq = static_cast<size_t>(T) * dkd, nv = static_cast<size_t>(T) * dvd;
    In iq(q.get_data(), nq), ik(k.get_data(), nq), iv(v.get_data(), nv), iA(A.get_data(), static_cast<size_t>(T) * sl),
        igy(dY.get_data(), nv);
    DBuf ds(static_cast<size_t>(T) * sl);
    Out odq(dq.get_data(), nq), odk(dk.get_data(), nq), odv(dv.get_data(), nv);
    const int c = causal ? 1 : 0;
    k_blk_scores<<<blk_grid(T, sl, sl), AT_THREADS>>>(igy.d, dvd, iv.d, dvd, ds.d, 1.0f, sl, c, dvd, 0, 0,
                                                      0);  // dA = dY V^T
    cuda_check(cudaGetLastError(), "attn_blocked_bwd dA");
    k_blk_ds_rows<<<T, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(iA.d, ds.d, scale, sl, c);
    cuda_check(cudaGetLastError(), "attn_blocked_bwd ds");
    k_blk_rowprod<<<blk_grid(T, sl, dkd), AT_THREADS>>>(ds.d, ik.d, dkd, odq.d, dkd, sl, c, dkd, 0, 0, 0);
    cuda_check(cudaGetLastError(), "attn_blocked_bwd dq");
    k_blk_colprod<<<blk_grid(T, sl, dkd), AT_THREADS>>>(ds.d, iq.d, dkd, odk.d, dkd, sl, c, dkd, 0, 0, 0);
    cuda_check(cudaGetLastError(), "attn_blocked_bwd dk");
    k_blk_colprod<<<blk_grid(T, sl, dvd), AT_THREADS>>>(iA.d, igy.d, dvd, odv.d, dvd, sl, c, dvd, 0, 0, 0);
    cuda_check(cudaGetLastError(), "attn_blocked_bwd dv");
    odq.finish();
    odk.finish();
    odv.finish();
    return true;
}

// ---- all heads in one launch (ops::attention_heads, 20 Oct 2026) -------
// The same kernels over a fused [T, 3d] qkv (q | k | v, head h in columns
// h dk .. (h + 1) dk of each part), blockIdx.z = head: no per-head slice,
// concat or column accumulate, and H times the blocks per launch (one
// head's d768 b16 launch was 64 blocks, under three per SM on an L4). Each
// output element runs the per-head kernels' fmaf chain over the same
// operands, so y, A and the gradients are those of the per-head path. A is
// [H T, sl] (head-major), y [T, d], dqkv [T, 3d].
bool attn_heads_fwd(const Matrix& qkv, size_t heads, float scale, size_t seq_len, bool causal, Matrix& A,
                    Matrix& y) {
    if (!active()) return false;
    const int T = static_cast<int>(qkv.rows()), H = static_cast<int>(heads), sl = static_cast<int>(seq_len);
    if (H <= 0 || qkv.cols() % (3 * heads) != 0 || sl <= 0 || T % sl != 0) return false;
    const int d = static_cast<int>(qkv.cols() / 3), dk = d / H, ld = 3 * d;
    if (A.rows() != heads * qkv.rows() || A.cols() != seq_len || y.rows() != qkv.rows() ||
        y.cols() != static_cast<size_t>(d))
        return false;
    In iqkv(qkv.get_data(), static_cast<size_t>(T) * ld);
    Out oA(A.get_data(), static_cast<size_t>(H) * T * sl), oy(y.get_data(), static_cast<size_t>(T) * d);
    const int c = causal ? 1 : 0;
    const size_t hsA = static_cast<size_t>(T) * sl;
    k_blk_scores<<<blk_grid(T, sl, sl, H), AT_THREADS>>>(iqkv.d, ld, iqkv.d + d, ld, oA.d, scale, sl, c, dk, dk, dk,
                                                         hsA);
    cuda_check(cudaGetLastError(), "attn_heads_fwd scores");
    k_blk_softmax_rows<<<H * T, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(oA.d, sl, c);
    cuda_check(cudaGetLastError(), "attn_heads_fwd softmax");
    k_blk_rowprod<<<blk_grid(T, sl, dk, H), AT_THREADS>>>(oA.d, iqkv.d + 2 * d, ld, oy.d, d, sl, c, dk, hsA, dk, dk);
    cuda_check(cudaGetLastError(), "attn_heads_fwd av");
    oA.finish();
    oy.finish();
    return true;
}

bool attn_heads_bwd(const Matrix& qkv, size_t heads, const Matrix& A, const Matrix& dY, float scale, size_t seq_len,
                    bool causal, Matrix& dqkv) {
    if (!active()) return false;
    const int T = static_cast<int>(qkv.rows()), H = static_cast<int>(heads), sl = static_cast<int>(seq_len);
    if (H <= 0 || qkv.cols() % (3 * heads) != 0 || sl <= 0 || T % sl != 0) return false;
    const int d = static_cast<int>(qkv.cols() / 3), dk = d / H, ld = 3 * d;
    if (A.rows() != heads * qkv.rows() || A.cols() != seq_len || dY.rows() != qkv.rows() ||
        dY.cols() != static_cast<size_t>(d) || dqkv.rows() != qkv.rows() || dqkv.cols() != qkv.cols())
        return false;
    const size_t hsA = static_cast<size_t>(T) * sl;
    In iqkv(qkv.get_data(), static_cast<size_t>(T) * ld), iA(A.get_data(), H * hsA),
        igy(dY.get_data(), static_cast<size_t>(T) * d);
    DBuf ds(H * hsA);
    Out og(dqkv.get_data(), static_cast<size_t>(T) * ld);  // every element written below
    const int c = causal ? 1 : 0;
    const float* q = iqkv.d;
    const float* k = iqkv.d + d;
    const float* v = iqkv.d + 2 * d;
    k_blk_scores<<<blk_grid(T, sl, sl, H), AT_THREADS>>>(igy.d, d, v, ld, ds.d, 1.0f, sl, c, dk, dk, dk,
                                                         hsA);  // dA = dY V^T
    cuda_check(cudaGetLastError(), "attn_heads_bwd dA");
    k_blk_ds_rows<<<H * T, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(iA.d, ds.d, scale, sl, c);
    cuda_check(cudaGetLastError(), "attn_heads_bwd ds");
    k_blk_rowprod<<<blk_grid(T, sl, dk, H), AT_THREADS>>>(ds.d, k, ld, og.d, ld, sl, c, dk, hsA, dk, dk);
    cuda_check(cudaGetLastError(), "attn_heads_bwd dq");
    k_blk_colprod<<<blk_grid(T, sl, dk, H), AT_THREADS>>>(ds.d, q, ld, og.d + d, ld, sl, c, dk, hsA, dk, dk);
    cuda_check(cudaGetLastError(), "attn_heads_bwd dk");
    k_blk_colprod<<<blk_grid(T, sl, dk, H), AT_THREADS>>>(iA.d, igy.d, d, og.d + 2 * d, ld, sl, c, dk, hsA, dk,
                                                          dk);
    cuda_check(cudaGetLastError(), "attn_heads_bwd dv");
    og.finish();
    return true;
}

bool attn_heads_fwd_ts(const Matrix& qkv, size_t heads, float scale, size_t seq_len, const std::vector<int>& code,
                       Matrix& A, Matrix& y) {
    if (!active()) return false;
    const int T = static_cast<int>(qkv.rows()), H = static_cast<int>(heads), sl = static_cast<int>(seq_len);
    if (H <= 0 || qkv.cols() % (3 * heads) != 0 || sl <= 0 || T % sl != 0 || code.size() != qkv.rows()) return false;
    const int d = static_cast<int>(qkv.cols() / 3), dk = d / H, ld = 3 * d;
    if (A.rows() != heads * qkv.rows() || A.cols() != seq_len || y.rows() != qkv.rows() ||
        y.cols() != static_cast<size_t>(d))
        return false;
    In iqkv(qkv.get_data(), static_cast<size_t>(T) * ld);
    IBuf dcode(code.data(), code.size());
    Out oA(A.get_data(), static_cast<size_t>(H) * T * sl), oy(y.get_data(), static_cast<size_t>(T) * d);
    const size_t hsA = static_cast<size_t>(T) * sl;
    k_blk_scores<<<blk_grid(T, sl, sl, H), AT_THREADS>>>(iqkv.d, ld, iqkv.d + d, ld, oA.d, scale, sl, 1, dk, dk, dk,
                                                         hsA);
    cuda_check(cudaGetLastError(), "attn_heads_fwd_ts scores");
    k_blk_softmax_rows_ts<<<H * T, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(oA.d, sl, T, dcode.d);
    cuda_check(cudaGetLastError(), "attn_heads_fwd_ts softmax");
    k_blk_rowprod<<<blk_grid(T, sl, dk, H), AT_THREADS>>>(oA.d, iqkv.d + 2 * d, ld, oy.d, d, sl, 1, dk, hsA, dk, dk);
    cuda_check(cudaGetLastError(), "attn_heads_fwd_ts av");
    oA.finish();
    oy.finish();
    return true;
}

// ---- RoPE and column slice/concat (the llama block's head plumbing) ----
//
// ops::apply_rope, ops::slice_cols and ops::concat_cols are the reference.
// Slice/concat/place are pure copies: bit for bit. RoPE: the cos/sin table
// is built on the HOST with the CPU op's exact float expressions
// (inv_freq = 1 / pow(base, dim / head_dim), theta = pos * inv_freq,
// std::cos/std::sin) and cached on the device keyed by (positions,
// head_dim, base), so the device never recomputes a transcendental and
// the table uploads once per distinct position vector (every training
// step reuses the same 0..sl-1 pattern). The backward uses the same table:
// cos(-t) == cos(t) and sin(-t) == -sin(t) exactly. The rotation rounds
// each product and the sum explicitly (__fmul_rn / __fadd_rn / __fsub_rn),
// as the host loop does, so RoPE is bit for bit too. That assumes ops.cpp
// is built without FMA contraction (the default microtorch flags have no
// -march); a build that adds -march=native to microtorch would contract
// the host loop into fma(x0, ct, -(x1 * st)) and lose bit equality.

namespace {

struct RopeTable {
    std::vector<int> pos;
    size_t head_dim = 0;
    float base = 0.0f;
    float* d = nullptr;  // [T * half] cos, then [T * half] sin
};

// One live table; rebuilt when the key changes. Held for the process
// (never freed at exit: a static dtor may run after context teardown).
// head_dim here is the rotated width n_rot (the head width unless partial):
// the angle of pair j is pos * 1 / pow(base, f32(2j) / n_rot).
const float* rope_table(const std::vector<int>& pos, size_t head_dim, float base) {
    static RopeTable t;
    if (t.d && t.pos == pos && t.head_dim == head_dim && t.base == base) return t.d;
    const size_t T = pos.size(), half = head_dim / 2;
    std::vector<float> h(2 * T * half);
    for (size_t i = 0; i < T; ++i) {
        const float m = static_cast<float>(pos[i]);
        for (size_t dim = 0; dim < head_dim; dim += 2) {
            const float inv_freq = 1.0f / std::pow(base, static_cast<float>(dim) / head_dim);
            const float theta = m * inv_freq;
            h[i * half + dim / 2] = std::cos(theta);
            h[T * half + i * half + dim / 2] = std::sin(theta);
        }
    }
    if (t.d) pool::dfree(t.d);
    t.d = nullptr;
    t.d = dalloc(h.size());
    h2d(t.d, h.data(), h.size());
    t.pos = pos, t.head_dim = head_dim, t.base = base;
    return t.d;
}

// y = rotate(x) over the [T, 3d] fused qkv layout: q cols [0, d), k cols
// [d, 2d), v untouched; heads h < n_rot rotate the first rot dims of the
// head (rot = head_dim unless partial) in adjacent pairs (x[2j], x[2j+1]),
// or with neox in the pairs (x[j], x[j + rot/2]), by the table angle (sign
// = -1: the inverse rotation); the other dims are copied.
__global__ void k_rope(const float* x, float* y, const float* tab, size_t T, size_t d3, size_t d,
                       size_t head_dim, size_t n_rot, float sign, size_t rot, bool neox) {
    const size_t n = T * d3, half = rot / 2;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
        const size_t i = idx / d3, c = idx % d3;
        const size_t cc = c % d;
        const size_t off = cc % head_dim;
        if (c >= 2 * d || cc / head_dim >= n_rot || off >= rot) {
            y[idx] = x[idx];
            continue;
        }
        const bool second = neox ? off >= half : (off & 1) != 0;
        const size_t p = neox ? off % half : off / 2;
        const size_t c0 = neox ? c - (second ? half : 0) : c - (off & 1);
        const size_t c1 = neox ? c0 + half : c0 + 1;
        const float x0 = x[i * d3 + c0], x1 = x[i * d3 + c1];
        const float ct = tab[i * half + p], st = sign * tab[T * half + i * half + p];
        // Explicitly rounded products and sum: the host loop (ops.cpp,
        // built without -march, so no FMA) rounds each product, and nvcc
        // would otherwise contract to FMA. This makes RoPE bit for bit.
        y[idx] = second ? __fadd_rn(__fmul_rn(x0, st), __fmul_rn(x1, ct))
                        : __fsub_rn(__fmul_rn(x0, ct), __fmul_rn(x1, st));
    }
}

// ops::repeat_heads: out head h = in head h / group.
__global__ void k_repeat_heads(const float* x, float* y, size_t T, size_t heads, size_t group, size_t hd) {
    const size_t W = heads * group * hd, n = T * W;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
        const size_t i = idx / W, c = idx % W;
        y[idx] = x[i * heads * hd + (c / hd / group) * hd + c % hd];
    }
}

// Its backward: in head k = the sum of out heads k*group .. k*group+group-1,
// in that order (as the host loop), each sum rounded.
__global__ void k_repeat_heads_bwd(const float* dy, float* dx, size_t T, size_t heads, size_t group, size_t hd) {
    const size_t w = heads * hd, W = w * group, n = T * w;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
        const size_t i = idx / w, c = idx % w, k = c / hd, e = c % hd;
        float acc = dy[i * W + (k * group) * hd + e];
        for (size_t r = 1; r < group; ++r) acc = __fadd_rn(acc, dy[i * W + (k * group + r) * hd + e]);
        dx[idx] = acc;
    }
}

__global__ void k_slice_cols(const float* x, float* y, size_t R, size_t C, size_t c0, size_t w) {
    const size_t n = R * w;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x)
        y[idx] = x[(idx / w) * C + c0 + idx % w];
}

// dx = 0 everywhere except columns [c0, c0 + w), which take g.
__global__ void k_place_cols(const float* g, float* dx, size_t R, size_t C, size_t c0, size_t w) {
    const size_t n = R * C;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
        const size_t j = idx % C;
        dx[idx] = (j >= c0 && j < c0 + w) ? g[(idx / C) * w + (j - c0)] : 0.0f;
    }
}

__global__ void k_put_cols(const float* src, float* dst, size_t R, size_t w, size_t C, size_t off) {
    const size_t n = R * w;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x)
        dst[(idx / w) * C + off + idx % w] = src[idx];
}

bool rope_launch(const Matrix& x, const std::vector<int>& pos, size_t head_dim, size_t n_rot, float base,
                 float sign, Matrix& y, const char* what, bool neox, size_t rot) {
    const size_t T = x.rows(), d3 = x.cols();
    if (d3 == 0 || d3 % 3 != 0 || pos.size() != T || head_dim == 0 || head_dim % 2 != 0) return false;
    const size_t d = d3 / 3;
    if (rot == 0) rot = head_dim;
    if (rot % 2 != 0 || rot > head_dim) return false;
    if (n_rot == 0 || n_rot * head_dim > d || y.rows() != T || y.cols() != d3) return false;
    const float* tab = rope_table(pos, rot, base);
    const size_t n = T * d3;
    In dx(x.get_data(), n);
    Out dy(y.get_data(), n);
    k_rope<<<ew_grid(n), NTHREADS>>>(dx.d, dy.d, tab, T, d3, d, head_dim, n_rot, sign, rot, neox);
    cuda_check(cudaGetLastError(), what);
    dy.finish();
    return true;
}

}  // namespace

bool rope_fwd(const Matrix& x, const std::vector<int>& pos, size_t head_dim, size_t n_rot_heads, float base,
              Matrix& y, bool neox, size_t n_rot_dims) {
    if (!active()) return false;
    return rope_launch(x, pos, head_dim, n_rot_heads, base, 1.0f, y, "rope_fwd", neox, n_rot_dims);
}

bool rope_bwd(const Matrix& dy, const std::vector<int>& pos, size_t head_dim, size_t n_rot_heads, float base,
              Matrix& dx, bool neox, size_t n_rot_dims) {
    if (!active()) return false;
    return rope_launch(dy, pos, head_dim, n_rot_heads, base, -1.0f, dx, "rope_bwd", neox, n_rot_dims);
}

bool repeat_heads_fwd(const Matrix& x, size_t heads, size_t group, size_t head_dim, Matrix& y) {
    if (!active()) return false;
    const size_t T = x.rows();
    if (x.cols() != heads * head_dim || y.rows() != T || y.cols() != heads * group * head_dim) return false;
    In ix(x.get_data(), T * heads * head_dim);
    Out oy(y.get_data(), T * heads * group * head_dim);
    k_repeat_heads<<<ew_grid(T * heads * group * head_dim), NTHREADS>>>(ix.d, oy.d, T, heads, group, head_dim);
    cuda_check(cudaGetLastError(), "repeat_heads_fwd");
    oy.finish();
    return true;
}

bool repeat_heads_bwd(const Matrix& dy, size_t heads, size_t group, size_t head_dim, Matrix& dx) {
    if (!active()) return false;
    const size_t T = dy.rows();
    if (dy.cols() != heads * group * head_dim || dx.rows() != T || dx.cols() != heads * head_dim) return false;
    In idy(dy.get_data(), T * heads * group * head_dim);
    Out odx(dx.get_data(), T * heads * head_dim);
    k_repeat_heads_bwd<<<ew_grid(T * heads * head_dim), NTHREADS>>>(idy.d, odx.d, T, heads, group, head_dim);
    cuda_check(cudaGetLastError(), "repeat_heads_bwd");
    odx.finish();
    return true;
}

bool slice_cols(const Matrix& x, size_t c0, size_t n, Matrix& y) {
    if (!active()) return false;
    const size_t R = x.rows(), C = x.cols();
    if (n == 0 || c0 + n > C || y.rows() != R || y.cols() != n) return false;
    In dx(x.get_data(), R * C);
    Out dy(y.get_data(), R * n);
    k_slice_cols<<<ew_grid(R * n), NTHREADS>>>(dx.d, dy.d, R, C, c0, n);
    cuda_check(cudaGetLastError(), "slice_cols");
    dy.finish();
    return true;
}

bool place_cols(const Matrix& g, size_t c0, Matrix& dx) {
    if (!active()) return false;
    const size_t R = dx.rows(), C = dx.cols(), w = g.cols();
    if (w == 0 || g.rows() != R || c0 + w > C) return false;
    In ig(g.get_data(), R * w);
    Out od(dx.get_data(), R * C);
    k_place_cols<<<ew_grid(R * C), NTHREADS>>>(ig.d, od.d, R, C, c0, w);
    cuda_check(cudaGetLastError(), "place_cols");
    od.finish();
    return true;
}

bool concat_cols(const std::vector<const Matrix*>& xs, Matrix& y) {
    if (!active()) return false;
    if (xs.empty()) return false;
    const size_t R = y.rows(), C = y.cols();
    size_t total = 0;
    for (const Matrix* x : xs) {
        if (!x || x->rows() != R || x->cols() == 0) return false;
        total += x->cols();
    }
    if (total != C) return false;
    Out dy(y.get_data(), R * C);
    size_t off = 0;
    for (const Matrix* x : xs) {
        const size_t w = x->cols();
        In dx(x->get_data(), R * w);
        k_put_cols<<<ew_grid(R * w), NTHREADS>>>(dx.d, dy.d, R, w, C, off);
        cuda_check(cudaGetLastError(), "concat_cols");
        off += w;
    }
    dy.finish();
    return true;
}

namespace {
// y[:, c0 : c0 + w) += g over the range only.
__global__ void k_add_cols(const float* g, float* y, size_t R, size_t C, size_t c0, size_t w) {
    const size_t n = R * w;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x)
        y[(idx / w) * C + c0 + idx % w] += g[idx];
}
}  // namespace

bool accumulate_cols(Matrix& y, size_t c0, const Matrix& g, bool y_is_zero) {
    if (!active()) return false;
    const size_t R = y.rows(), C = y.cols(), w = g.cols();
    if (w == 0 || g.rows() != R || c0 + w > C) return false;
    In ig(g.get_data(), R * w);
    if (y_is_zero) {
        // Known-zero destination: write it whole, never upload it.
        Out oy(y.get_data(), R * C);
        k_place_cols<<<ew_grid(R * C), NTHREADS>>>(ig.d, oy.d, R, C, c0, w);
        cuda_check(cudaGetLastError(), "accumulate_cols place");
        oy.finish();
    } else {
        Out oy(y.get_data(), R * C, /*need_current=*/true);
        k_add_cols<<<ew_grid(R * w), NTHREADS>>>(ig.d, oy.d, R, C, c0, w);
        cuda_check(cudaGetLastError(), "accumulate_cols add");
        oy.finish();
    }
    return true;
}

// ---- mixture-of-experts dispatch ---------------------------------------
// The MoE layer routes each token's row to k experts, runs every expert on
// its gathered rows and scatter-adds the gated outputs back. On the host
// these loops forced the activations home once per expert per layer. Here
// they stay in the value cache. Scatters are deterministic and atomic-free:
// the host builds, per destination row, its sources in ascending order
// (one int upload: offsets then sources), and one thread per destination
// element adds them in that order, which is the host loop's add sequence.

namespace {

// y[i, :] = x[idx[i], :] over n = rows(y) * C elements.
__global__ void k_gather_rows(const float* x, const int* idx, float* y, size_t n, size_t C) {
    for (size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x)
        y[t] = x[static_cast<size_t>(idx[t / C]) * C + t % C];
}

// Destination row r sums x rows src[off[r] .. off[r + 1]) in that order.
// SC_COPY: the first source is copied (scatter_add_rows' rule, keeping a
// -0.0), rows with no source are zero. SC_ZERO: the sum starts from 0.0f
// (a gather backward into a zero-filled matrix). SC_INTO: that sum is then
// added to y, and rows with no source keep their value (dx, then grad +=
// dx, without the dense dx).
enum { SC_COPY = 0, SC_ZERO = 1, SC_INTO = 2 };
__global__ void k_scatter_rows(const float* x, const int* off, const int* src, float* y, size_t n, size_t C,
                               int mode) {
    for (size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) {
        const size_t r = t / C, j = t % C;
        const int b = off[r], e = off[r + 1];
        if (mode == SC_INTO && b == e) continue;
        float acc = 0.0f;
        if (b < e) {
            const float v = x[static_cast<size_t>(src[b]) * C + j];
            acc = mode == SC_COPY ? v : acc + v;
            for (int p = b + 1; p < e; ++p) acc += x[static_cast<size_t>(src[p]) * C + j];
        }
        y[t] = mode == SC_INTO ? y[t] + acc : acc;
    }
}

// Counting sort of sources by destination row: stable, so each row's
// sources stay ascending. Packed [offsets (n_rows + 1) | sources] for one
// upload.
std::vector<int> scatter_index(const std::vector<int>& idx, size_t n_rows) {
    std::vector<int> pack(n_rows + 1 + idx.size(), 0);
    int* off = pack.data();
    int* src = off + n_rows + 1;
    for (int r : idx) ++off[r + 1];
    for (size_t r = 0; r < n_rows; ++r) off[r + 1] += off[r];
    std::vector<int> fill(off, off + n_rows);
    for (size_t i = 0; i < idx.size(); ++i) src[fill[idx[i]]++] = static_cast<int>(i);
    return pack;
}

// One thread per row, the host loop verbatim: running max, then a double
// sum of exp(x - max) in column order.
__global__ void k_logsumexp_fwd(const float* x, float* lse, int R, int C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    const float* xr = x + static_cast<size_t>(i) * C;
    float mx = xr[0];
    for (int j = 1; j < C; ++j) mx = mx < xr[j] ? xr[j] : mx;
    double s = 0.0;
    for (int j = 0; j < C; ++j) s += exp(static_cast<double>(xr[j]) - mx);
    lse[i] = mx + static_cast<float>(log(s));
}

__global__ void k_logsumexp_bwd(const float* x, const float* lse, const float* g, float* dx, size_t n, size_t C) {
    for (size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) {
        const size_t i = t / C;
        // The host's std::exp(float) is glibc's expf, correctly rounded in
        // practice; the double exp rounded once to float matches it where
        // the device expf (2 ulp) would not.
        dx[t] = g[i] * static_cast<float>(exp(static_cast<double>(x[t] - lse[i])));
    }
}

// y = x .* c[row]: mul_col's forward and its dx (dy .* c[row]).
__global__ void k_mul_col(const float* x, const float* c, float* y, size_t n, size_t C) {
    for (size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x)
        y[t] = x[t] * c[t / C];
}

// dc[i] = sum_j dy[i,j] x[i,j], one thread per row in column order. The
// host library is built without FMA (no -march), so the loop rounds the
// product and then the sum; the _rn intrinsics stop nvcc contracting them.
__global__ void k_mul_col_dc(const float* dy, const float* x, float* dc, int R, int C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    const size_t o = static_cast<size_t>(i) * C;
    float s = 0.0f;
    for (int j = 0; j < C; ++j) s = __fadd_rn(s, __fmul_rn(dy[o + j], x[o + j]));
    dc[i] = s;
}

}  // namespace

bool gather_rows(const Matrix& x, const std::vector<int>& idx, size_t C, Matrix& y) {
    if (!active()) return false;
    const size_t n_src = x.rows() * x.cols(), n = idx.size() * C;
    if (idx.empty() || C == 0 || n_src % C != 0 || y.rows() * y.cols() != n) return false;
    In dx(x.get_data(), n_src);
    IBuf di(idx.data(), idx.size());
    Out dy(y.get_data(), n);
    k_gather_rows<<<ew_grid(n), NTHREADS>>>(dx.d, di.d, dy.d, n, C);
    cuda_check(cudaGetLastError(), "gather_rows");
    dy.finish();
    return true;
}

bool scatter_rows(const Matrix& x, const std::vector<int>& idx, size_t n_rows, size_t C, bool first_copies,
                  Matrix& y) {
    if (!active()) return false;
    const size_t n_src = idx.size() * C, n = n_rows * C;
    if (idx.empty() || C == 0 || x.rows() * x.cols() != n_src || y.rows() * y.cols() != n) return false;
    const std::vector<int> pack = scatter_index(idx, n_rows);
    In dx(x.get_data(), n_src);
    IBuf dp(pack.data(), pack.size());
    Out dy(y.get_data(), n);
    k_scatter_rows<<<ew_grid(n), NTHREADS>>>(dx.d, dp.d, dp.d + n_rows + 1, dy.d, n, C,
                                             first_copies ? SC_COPY : SC_ZERO);
    cuda_check(cudaGetLastError(), "scatter_rows");
    dy.finish();
    return true;
}

bool scatter_rows_multi(const std::vector<const Matrix*>& xs, const std::vector<int>& idx, size_t n_rows,
                        size_t C, Matrix& y) {
    if (!active()) return false;
    const size_t n = n_rows * C;
    size_t rows = 0;
    for (const Matrix* x : xs) {
        if (!x || x->cols() != C) return false;
        rows += x->rows();
    }
    if (xs.empty() || C == 0 || rows != idx.size() || y.rows() * y.cols() != n) return false;
    // Stack the inputs in one device-only buffer (no host shadow), then
    // one scatter over the concatenated sources.
    DBuf cat(rows * C);
    size_t at = 0;
    for (const Matrix* x : xs) {
        const size_t m = x->rows() * C;
        In dx(x->get_data(), m);
        cuda_check(cudaMemcpyAsync(cat.d + at, dx.d, m * sizeof(float), cudaMemcpyDeviceToDevice, 0),
                   "scatter_rows_multi stack");
        at += m;
    }
    const std::vector<int> pack = scatter_index(idx, n_rows);
    IBuf dp(pack.data(), pack.size());
    Out dy(y.get_data(), n);
    k_scatter_rows<<<ew_grid(n), NTHREADS>>>(cat.d, dp.d, dp.d + n_rows + 1, dy.d, n, C, SC_COPY);
    cuda_check(cudaGetLastError(), "scatter_rows_multi");
    dy.finish();
    return true;
}

bool scatter_rows_into(const Matrix& g, const std::vector<int>& idx, Matrix& y, bool y_is_zero) {
    if (!active()) return false;
    const size_t n_rows = y.rows(), C = y.cols(), n = n_rows * C;
    if (idx.empty() || C == 0 || g.rows() != idx.size() || g.cols() != C) return false;
    const std::vector<int> pack = scatter_index(idx, n_rows);
    In ig(g.get_data(), idx.size() * C);
    IBuf dp(pack.data(), pack.size());
    Out oy(y.get_data(), n, /*need_current=*/!y_is_zero);
    k_scatter_rows<<<ew_grid(n), NTHREADS>>>(ig.d, dp.d, dp.d + n_rows + 1, oy.d, n, C,
                                             y_is_zero ? SC_ZERO : SC_INTO);
    cuda_check(cudaGetLastError(), "scatter_rows_into");
    oy.finish();
    return true;
}

bool logsumexp_fwd(const Matrix& x, Matrix& lse) {
    if (!active()) return false;
    const int R = static_cast<int>(x.rows()), C = static_cast<int>(x.cols());
    if (R == 0 || C == 0 || lse.rows() != x.rows() || lse.cols() != 1) return false;
    In dx(x.get_data(), x.rows() * x.cols());
    Out dl(lse.get_data(), x.rows());
    k_logsumexp_fwd<<<(R + NTHREADS - 1) / NTHREADS, NTHREADS>>>(dx.d, dl.d, R, C);
    cuda_check(cudaGetLastError(), "logsumexp_fwd");
    dl.finish();
    return true;
}

bool logsumexp_bwd(const Matrix& x, const Matrix& lse, const Matrix& g, Matrix& dx) {
    if (!active()) return false;
    const size_t R = x.rows(), C = x.cols(), n = R * C;
    if (n == 0 || lse.rows() != R || g.rows() != R || lse.cols() != 1 || g.cols() != 1 || dx.rows() != R ||
        dx.cols() != C)
        return false;
    In ix(x.get_data(), n), il(lse.get_data(), R), ig(g.get_data(), R);
    Out od(dx.get_data(), n);
    k_logsumexp_bwd<<<ew_grid(n), NTHREADS>>>(ix.d, il.d, ig.d, od.d, n, C);
    cuda_check(cudaGetLastError(), "logsumexp_bwd");
    od.finish();
    return true;
}

bool mul_col_fwd(const Matrix& x, const Matrix& c, Matrix& y) {
    if (!active()) return false;
    const size_t R = x.rows(), C = x.cols(), n = R * C;
    if (n == 0 || c.rows() != R || c.cols() != 1 || y.rows() != R || y.cols() != C) return false;
    In ix(x.get_data(), n), ic(c.get_data(), R);
    Out oy(y.get_data(), n);
    k_mul_col<<<ew_grid(n), NTHREADS>>>(ix.d, ic.d, oy.d, n, C);
    cuda_check(cudaGetLastError(), "mul_col_fwd");
    oy.finish();
    return true;
}

bool mul_col_dx(const Matrix& dy, const Matrix& c, Matrix& dx) {
    return mul_col_fwd(dy, c, dx);  // the same broadcast product
}

bool mul_col_dc(const Matrix& dy, const Matrix& x, Matrix& dc) {
    if (!active()) return false;
    const int R = static_cast<int>(x.rows()), C = static_cast<int>(x.cols());
    const size_t n = x.rows() * x.cols();
    if (n == 0 || dy.rows() != x.rows() || dy.cols() != x.cols() || dc.rows() != x.rows() || dc.cols() != 1)
        return false;
    In idy(dy.get_data(), n), ix(x.get_data(), n);
    Out odc(dc.get_data(), x.rows());
    k_mul_col_dc<<<(R + NTHREADS - 1) / NTHREADS, NTHREADS>>>(idy.d, ix.d, odc.d, R, C);
    cuda_check(cudaGetLastError(), "mul_col_dc");
    odc.finish();
    return true;
}

// ---- dense training step: the tied head, the embedding backward and the
// grad-norm clip on the device (20 Oct 2026). Transpose and the scatter are
// exact (copies; each element's adds in the host loop's order). The sum of
// squares is double like the host loop's, but summed in a different order.

namespace {

constexpr int TR_TILE = 32;

// y[C, R] = x[R, C]^T through a padded shared tile (both sides coalesced).
__global__ void k_transpose(const float* x, float* y, int R, int C) {
    __shared__ float t[TR_TILE][TR_TILE + 1];
    const int bx = blockIdx.x * TR_TILE, by = blockIdx.y * TR_TILE;
    for (int k = threadIdx.y; k < TR_TILE; k += blockDim.y) {
        const int r = by + k, c = bx + threadIdx.x;
        if (r < R && c < C) t[k][threadIdx.x] = x[static_cast<size_t>(r) * C + c];
    }
    __syncthreads();
    for (int k = threadIdx.y; k < TR_TILE; k += blockDim.y) {
        const int c = bx + k, r = by + threadIdx.x;
        if (c < C && r < R) y[static_cast<size_t>(c) * R + r] = t[threadIdx.x][k];
    }
}

// Embedding backward: block u owns table row rows[u] and adds the gradient
// rows src[off[u] .. off[u + 1]) into it one after another, ascending, as
// the host loop does (t->grad(ids[i], j) += g(i, j) for i = 0, 1, ...).
__global__ void k_embed_scatter(const float* g, const int* rows, const int* off, const int* src, float* y,
                                int d) {
    const int u = blockIdx.x;
    float* yr = y + static_cast<size_t>(rows[u]) * d;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float acc = yr[j];
        for (int k = off[u]; k < off[u + 1]; ++k) acc += g[static_cast<size_t>(src[k]) * d + j];
        yr[j] = acc;
    }
}

constexpr int SQ_BLOCKS = 256;

// Per-block partial sums of x^2 in double; k_sumsq_fold adds the partials
// in block order, so the result is the same run to run.
__global__ void k_sumsq_part(const float* x, size_t n, double* part) {
    __shared__ double sh[NTHREADS];
    double acc = 0.0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const double v = x[i];
        acc += v * v;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = sh[0];
}

__global__ void k_sumsq_fold(const double* part, int n, double* out) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += part[i];
    *out = s;
}

}  // namespace

bool transpose(const Matrix& x, Matrix& y) {
    if (!active()) return false;
    const int R = static_cast<int>(x.rows()), C = static_cast<int>(x.cols());
    if (y.rows() != x.cols() || y.cols() != x.rows()) return false;
    const size_t n = x.rows() * x.cols();
    In ix(x.get_data(), n);
    Out oy(y.get_data(), n);
    const dim3 grid((C + TR_TILE - 1) / TR_TILE, (R + TR_TILE - 1) / TR_TILE), block(TR_TILE, 8);
    k_transpose<<<grid, block>>>(ix.d, oy.d, R, C);
    cuda_check(cudaGetLastError(), "transpose");
    oy.finish();
    return true;
}

bool embed_scatter_add(const Matrix& g, const std::vector<int>& ids, Matrix& table_grad) {
    if (!active()) return false;
    const int d = static_cast<int>(table_grad.cols());
    if (g.rows() != ids.size() || g.cols() != table_grad.cols() || ids.empty()) return false;
    // Group source rows by table row (ids were bounds-checked by the caller):
    // a counting sort keeps each row's sources in ascending order.
    const size_t V = table_grad.rows();
    std::vector<int> count(V, 0);
    for (int id : ids) ++count[id];
    std::vector<int> rows, off{0}, start(V, 0);
    for (size_t r = 0; r < V; ++r)
        if (count[r]) {
            start[r] = off.back();
            rows.push_back(static_cast<int>(r));
            off.push_back(off.back() + count[r]);
        }
    std::vector<int> src(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) src[start[ids[i]]++] = static_cast<int>(i);
    In ig(g.get_data(), g.rows() * g.cols());
    IBuf drows(rows.data(), rows.size()), doff(off.data(), off.size()), dsrc(src.data(), src.size());
    Out oy(table_grad.get_data(), table_grad.rows() * table_grad.cols(), /*need_current=*/true);
    k_embed_scatter<<<static_cast<int>(rows.size()), col_threads(d)>>>(ig.d, drows.d, doff.d, dsrc.d, oy.d, d);
    cuda_check(cudaGetLastError(), "embed_scatter_add");
    oy.finish();
    return true;
}

bool sumsq(const std::vector<const Matrix*>& xs, std::vector<double>& out) {
    if (!active()) return false;
    out.assign(xs.size(), 0.0);
    if (xs.empty()) return true;
    double* part = nullptr;
    double* res = nullptr;
    cuda_check(pool::dmalloc(&part, SQ_BLOCKS * sizeof(double)), "sumsq malloc");
    cuda_check(pool::dmalloc(&res, xs.size() * sizeof(double)), "sumsq malloc");
    for (size_t k = 0; k < xs.size(); ++k) {
        const size_t n = xs[k]->rows() * xs[k]->cols();
        In ix(xs[k]->get_data(), n);
        const int blocks = static_cast<int>(std::min<size_t>(SQ_BLOCKS, (n + NTHREADS - 1) / NTHREADS));
        k_sumsq_part<<<blocks, NTHREADS>>>(ix.d, n, part);
        k_sumsq_fold<<<1, 1>>>(part, blocks, res + k);
        cuda_check(cudaGetLastError(), "sumsq");
    }
    cuda_check(cudaMemcpy(out.data(), res, xs.size() * sizeof(double), cudaMemcpyDeviceToHost), "sumsq D2H");
    pool::dfree(part);
    pool::dfree(res);
    return true;
}

}  // namespace devops
}  // namespace device
}  // namespace microtorch
