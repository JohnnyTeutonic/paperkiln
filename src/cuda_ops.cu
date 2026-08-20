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
// B2.1a semantics are WRITE-THROUGH: host in, host out, device round-trip
// inside, so host data is never stale (B2.0's contract). This stage is
// about landing the kernels under test; B2.1b defers the downloads and
// activates the validity-flag contract + DEVCHECK. Per-call H2D/D2H
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
    DBuf da(a.get_data(), n), db(b.get_data(), n), dy(n);
    k_add<<<ew_grid(n), NTHREADS>>>(da.d, db.d, dy.d, n);
    cuda_check(cudaGetLastError(), "add");
    d2h(y.get_data(), dy.d, n);
    return true;
}

bool sub(const Matrix& a, const Matrix& b, Matrix& y) {
    if (!active()) return false;
    const size_t n = a.rows() * a.cols();
    DBuf da(a.get_data(), n), db(b.get_data(), n), dy(n);
    k_sub<<<ew_grid(n), NTHREADS>>>(da.d, db.d, dy.d, n);
    cuda_check(cudaGetLastError(), "sub");
    d2h(y.get_data(), dy.d, n);
    return true;
}

bool mul(const Matrix& a, const Matrix& b, Matrix& y) {
    if (!active()) return false;
    const size_t n = a.rows() * a.cols();
    DBuf da(a.get_data(), n), db(b.get_data(), n), dy(n);
    k_mul<<<ew_grid(n), NTHREADS>>>(da.d, db.d, dy.d, n);
    cuda_check(cudaGetLastError(), "mul");
    d2h(y.get_data(), dy.d, n);
    return true;
}

bool scale(const Matrix& a, float s, Matrix& y) {
    if (!active()) return false;
    const size_t n = a.rows() * a.cols();
    DBuf da(a.get_data(), n), dy(n);
    k_scale<<<ew_grid(n), NTHREADS>>>(da.d, s, dy.d, n);
    cuda_check(cudaGetLastError(), "scale");
    d2h(y.get_data(), dy.d, n);
    return true;
}

bool axpy(Matrix& y, float a, const Matrix& x) {
    if (!active()) return false;
    const size_t n = y.rows() * y.cols();
    DBuf dy(y.get_data(), n), dx(x.get_data(), n);
    k_axpy<<<ew_grid(n), NTHREADS>>>(dy.d, a, dx.d, n);
    cuda_check(cudaGetLastError(), "axpy");
    d2h(y.get_data(), dy.d, n);
    return true;
}

bool fill(Matrix& y, float v) {
    if (!active()) return false;
    const size_t n = y.rows() * y.cols();
    DBuf dy(n);
    k_fill<<<ew_grid(n), NTHREADS>>>(dy.d, v, n);
    cuda_check(cudaGetLastError(), "fill");
    d2h(y.get_data(), dy.d, n);
    return true;
}

bool sigmoid_fwd(const Matrix& x, Matrix& y) {
    if (!active()) return false;
    const size_t n = x.rows() * x.cols();
    DBuf dx(x.get_data(), n), dy(n);
    k_sigmoid_fwd<<<ew_grid(n), NTHREADS>>>(dx.d, dy.d, n);
    cuda_check(cudaGetLastError(), "sigmoid_fwd");
    d2h(y.get_data(), dy.d, n);
    return true;
}

bool sigmoid_bwd(const Matrix& s, const Matrix& dy, Matrix& dx) {
    if (!active()) return false;
    const size_t n = s.rows() * s.cols();
    DBuf ds(s.get_data(), n), dgy(dy.get_data(), n), dgx(n);
    k_sigmoid_bwd<<<ew_grid(n), NTHREADS>>>(ds.d, dgy.d, dgx.d, n);
    cuda_check(cudaGetLastError(), "sigmoid_bwd");
    d2h(dx.get_data(), dgx.d, n);
    return true;
}

bool gelu_fwd(const Matrix& x, Matrix& y) {
    if (!active()) return false;
    const size_t n = x.rows() * x.cols();
    DBuf dx(x.get_data(), n), dy(n);
    k_gelu_fwd<<<ew_grid(n), NTHREADS>>>(dx.d, dy.d, n);
    cuda_check(cudaGetLastError(), "gelu_fwd");
    d2h(y.get_data(), dy.d, n);
    return true;
}

bool gelu_bwd(const Matrix& x, const Matrix& dy, Matrix& dx) {
    if (!active()) return false;
    const size_t n = x.rows() * x.cols();
    DBuf dxi(x.get_data(), n), dgy(dy.get_data(), n), dgx(n);
    k_gelu_bwd<<<ew_grid(n), NTHREADS>>>(dxi.d, dgy.d, dgx.d, n);
    cuda_check(cudaGetLastError(), "gelu_bwd");
    d2h(dx.get_data(), dgx.d, n);
    return true;
}

bool softmax_fwd(const Matrix& x, Matrix& y) {
    if (!active()) return false;
    const int R = static_cast<int>(x.rows());
    const int C = static_cast<int>(x.cols());
    const size_t n = static_cast<size_t>(R) * C;
    DBuf dx(x.get_data(), n), dy(n);
    k_softmax_fwd<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(dx.d, dy.d,
                                                                   C);
    cuda_check(cudaGetLastError(), "softmax_fwd");
    d2h(y.get_data(), dy.d, n);
    return true;
}

bool softmax_bwd(const Matrix& S, const Matrix& dY, Matrix& dX) {
    if (!active()) return false;
    const int R = static_cast<int>(S.rows());
    const int C = static_cast<int>(S.cols());
    const size_t n = static_cast<size_t>(R) * C;
    DBuf ds(S.get_data(), n), dgy(dY.get_data(), n), dgx(n);
    k_softmax_bwd<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(
        ds.d, dgy.d, dgx.d, C);
    cuda_check(cudaGetLastError(), "softmax_bwd");
    d2h(dX.get_data(), dgx.d, n);
    return true;
}

bool layernorm_fwd(const Matrix& x, const Matrix& gamma, const Matrix& beta,
                   float eps, Matrix& y, Matrix& xhat,
                   std::vector<float>& rstd) {
    if (!active()) return false;
    const int R = static_cast<int>(x.rows());
    const int C = static_cast<int>(x.cols());
    const size_t n = static_cast<size_t>(R) * C;
    DBuf dx(x.get_data(), n), dg(gamma.get_data(), C), db(beta.get_data(), C);
    DBuf dy(n), dxh(n), drs(R);
    k_layernorm_fwd<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(
        dx.d, dg.d, db.d, eps, dy.d, dxh.d, drs.d, C);
    cuda_check(cudaGetLastError(), "layernorm_fwd");
    d2h(y.get_data(), dy.d, n);
    d2h(xhat.get_data(), dxh.d, n);
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
    DBuf ddy(dY.get_data(), n), dxh(xhat.get_data(), n);
    if (want_dgb) {
        DBuf ddg(static_cast<size_t>(C)), ddb(static_cast<size_t>(C));
        const int grid = (C + NTHREADS - 1) / NTHREADS;
        k_colsum_dgb<<<grid, NTHREADS>>>(ddy.d, dxh.d, ddg.d, ddb.d, R, C);
        cuda_check(cudaGetLastError(), "layernorm_bwd dgb");
        d2h(dg->get_data(), ddg.d, C);
        d2h(db->get_data(), ddb.d, C);
    }
    if (want_dx) {
        DBuf drs(rstd.data(), static_cast<size_t>(R));
        DBuf dgm(gamma.get_data(), static_cast<size_t>(C)), ddx(n);
        k_layernorm_bwd_dx<<<R, ROW_THREADS, 2 * ROW_THREADS * sizeof(float)>>>(
            ddy.d, dxh.d, drs.d, dgm.d, ddx.d, C);
        cuda_check(cudaGetLastError(), "layernorm_bwd dx");
        d2h(dx->get_data(), ddx.d, n);
    }
    return true;
}

bool rmsnorm_fwd(const Matrix& x, const Matrix& w, float eps, Matrix& y,
                 std::vector<float>& rms_inv) {
    if (!active()) return false;
    const int R = static_cast<int>(x.rows());
    const int C = static_cast<int>(x.cols());
    const size_t n = static_cast<size_t>(R) * C;
    DBuf dx(x.get_data(), n), dw(w.get_data(), C), dy(n), dri(R);
    k_rmsnorm_fwd<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(
        dx.d, dw.d, eps, dy.d, dri.d, C);
    cuda_check(cudaGetLastError(), "rmsnorm_fwd");
    d2h(y.get_data(), dy.d, n);
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
    DBuf ddy(dY.get_data(), n), dxi(x.get_data(), n);
    DBuf dri(rms_inv.data(), static_cast<size_t>(R));
    if (want_dw) {
        DBuf ddw(static_cast<size_t>(C));
        const int grid = (C + NTHREADS - 1) / NTHREADS;
        k_rmsnorm_bwd_dw<<<grid, NTHREADS>>>(ddy.d, dxi.d, dri.d, ddw.d, R, C);
        cuda_check(cudaGetLastError(), "rmsnorm_bwd dw");
        d2h(dw->get_data(), ddw.d, C);
    }
    if (want_dx) {
        DBuf dwv(w.get_data(), static_cast<size_t>(C)), ddx(n);
        k_rmsnorm_bwd_dx<<<R, ROW_THREADS, ROW_THREADS * sizeof(float)>>>(
            ddy.d, dxi.d, dri.d, dwv.d, ddx.d, C);
        cuda_check(cudaGetLastError(), "rmsnorm_bwd dx");
        d2h(dx->get_data(), ddx.d, n);
    }
    return true;
}

}  // namespace devops
}  // namespace device
}  // namespace microtorch
