// Device kernels for ops::depth_attend (include/microtorch/depth_attend.hpp):
// attention over depth, one warp per row. Its own translation unit, with the
// same operand/output seam as cuda_ops.cu (the value cache of
// cuda_resident.cu through detail::vc_operand / vc_output), so the B2.1b
// deferred path keeps every source, weight table and gradient on the device.
//
// Numerics against the host loop: the per-element products and the
// ascending-i sums of the output and of the softmax denominator are the
// host's (no fused multiply-add in those); ss_i, l_i and g_i are reduced
// across a warp (another summation order) and l_i is formed as r_i (s_i . w)
// rather than sum (s_i r_i) w: ROUNDING, as every other reduction kernel.
// The query gradient sums per-warp partials in a fixed order (no atomics),
// so a run is reproducible on one GPU and grid.
#include <cuda_runtime.h>

#include "device_pool.cuh"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "microtorch/depth_attend.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"

namespace microtorch {
namespace device {
namespace devops {

namespace {

void da_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA depth_attend ") + what + ": " + cudaGetErrorString(e));
}

bool da_active() { return device_ops_enabled() && get() == Device::CUDA; }

float* da_alloc(size_t n) {
    float* d = nullptr;
    da_check(pool::dmalloc(&d, n * sizeof(float)), "malloc");
    return d;
}

// cuda_ops.cu's In / Out, verbatim in behaviour (see the note there on why
// `owned` is assigned in the body).
struct DIn {
    float* d = nullptr;
    bool owned = false;
    DIn(const float* h, size_t n) { d = detail::vc_operand(h, n, owned); }
    ~DIn() {
        if (owned) pool::dfree(d);
    }
    DIn(const DIn&) = delete;
    DIn& operator=(const DIn&) = delete;
};
struct DOut {
    float* d = nullptr;
    float* host = nullptr;
    size_t n = 0;
    bool deferred = false;
    const char* op = nullptr;
    DOut(float* h, size_t n_, const char* op_) : host(h), n(n_), op(op_) {
        d = detail::vc_output(h, n_, deferred, false, h);
        if (!deferred) d = da_alloc(n_);
        detail::gpu_profile_begin(op);
    }
    void finish() {
        detail::gpu_profile_end(op);
        if (!deferred)
            da_check(cudaMemcpy(host, d, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H");
    }
    ~DOut() {
        if (!deferred && d) pool::dfree(d);
    }
    DOut(const DOut&) = delete;
    DOut& operator=(const DOut&) = delete;
};

constexpr int kMax = static_cast<int>(kDepthAttendMaxSources);
constexpr int kWarps = 4;  // rows per block (one warp each)

struct SrcPtrs {
    const float* p[kMax];
};
struct DstPtrs {
    float* p[kMax];
};

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__global__ void k_da_fwd(SrcPtrs src, int S, const float* __restrict__ w, float eps, float scale, float* __restrict__ out,
                         float* __restrict__ alpha, float* __restrict__ rinv, float* __restrict__ logit, int T,
                         int d) {
    __shared__ float sl[kWarps][kMax], sa[kWarps][kMax];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int t = blockIdx.x * kWarps + warp;
    if (t >= T) return;  // the whole warp leaves together
    const size_t o = static_cast<size_t>(t) * d;
    for (int i = 0; i < S; ++i) {
        const float* row = src.p[i] + o;
        float ss = 0.0f, dot = 0.0f;
        for (int j = lane; j < d; j += 32) {
            const float v = row[j];
            ss = __fadd_rn(ss, __fmul_rn(v, v));
            dot = __fadd_rn(dot, __fmul_rn(v, w[j]));
        }
        ss = warp_sum(ss);
        dot = warp_sum(dot);
        const float r = 1.0f / sqrtf(ss / static_cast<float>(d) + eps);
        const float l = __fmul_rn(r, dot);
        if (lane == 0) {
            sl[warp][i] = l;
            rinv[static_cast<size_t>(t) * S + i] = r;
            logit[static_cast<size_t>(t) * S + i] = l;
        }
    }
    __syncwarp();
    float mx = sl[warp][0];
    for (int i = 1; i < S; ++i) mx = fmaxf(mx, sl[warp][i]);
    float z = 0.0f;
    for (int i = 0; i < S; ++i) z = __fadd_rn(z, expf(sl[warp][i] - mx));  // ascending i, as the host
    for (int i = lane; i < S; i += 32) {
        const float a = expf(sl[warp][i] - mx) / z;
        sa[warp][i] = a;
        alpha[static_cast<size_t>(t) * S + i] = a;
    }
    __syncwarp();
    for (int j = lane; j < d; j += 32) {
        float acc = __fmul_rn(src.p[0][o + j], sa[warp][0]);
        for (int i = 1; i < S; ++i) acc = __fadd_rn(acc, __fmul_rn(src.p[i][o + j], sa[warp][i]));
        out[o + j] = scale != 1.0f ? __fmul_rn(acc, scale) : acc;
    }
}

// Dynamic shared memory: per warp, d floats (the query-gradient partial) and
// kMax floats (g_i).
__global__ void k_da_bwd(SrcPtrs src, DstPtrs ds, int S, const float* __restrict__ w, float scale,
                         const float* __restrict__ alpha, const float* __restrict__ rinv,
                         const float* __restrict__ logit, const float* __restrict__ dy, float* __restrict__ part,
                         int T, int d) {
    extern __shared__ float shm[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float* acc = shm + static_cast<size_t>(warp) * (d + kMax);
    float* sg = acc + d;
    for (int j = lane; j < d; j += 32) acc[j] = 0.0f;
    const int gw = blockIdx.x * kWarps + warp, nw = gridDim.x * kWarps;
    const float fd = static_cast<float>(d);
    for (int t = gw; t < T; t += nw) {
        const size_t o = static_cast<size_t>(t) * d, so = static_cast<size_t>(t) * S;
        for (int i = 0; i < S; ++i) {
            const float* row = src.p[i] + o;
            float g = 0.0f;
            for (int j = lane; j < d; j += 32) g = __fadd_rn(g, __fmul_rn(__fmul_rn(dy[o + j], scale), row[j]));
            g = warp_sum(g);
            if (lane == 0) sg[i] = g;
        }
        __syncwarp();
        float gbar = 0.0f;
        for (int i = 0; i < S; ++i) gbar = __fadd_rn(gbar, __fmul_rn(alpha[so + i], sg[i]));
        for (int i = 0; i < S; ++i) {
            const float a = alpha[so + i], r = rinv[so + i];
            const float dl = __fmul_rn(a, sg[i] - gbar);
            const float coef = __fmul_rn(dl, r);
            const float k = __fmul_rn(logit[so + i], r) / fd;
            const float* row = src.p[i] + o;
            float* out = ds.p[i];
            for (int j = lane; j < d; j += 32) {
                const float s = row[j];
                if (out)
                    out[o + j] = __fadd_rn(__fmul_rn(a, __fmul_rn(dy[o + j], scale)),
                                           __fmul_rn(coef, w[j] - __fmul_rn(k, s)));
                if (part) acc[j] = __fadd_rn(acc[j], __fmul_rn(coef, s));
            }
        }
        __syncwarp();
    }
    if (part)
        for (int j = lane; j < d; j += 32) part[static_cast<size_t>(gw) * d + j] = acc[j];
}

__global__ void k_da_dw(const float* __restrict__ part, int nw, int d, float* __restrict__ dw) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d) return;
    float s = 0.0f;
    for (int k = 0; k < nw; ++k) s = __fadd_rn(s, part[static_cast<size_t>(k) * d + j]);
    dw[j] = s;
}

}  // namespace

bool depth_attend_fwd(const std::vector<const Matrix*>& src, const Matrix& w, float eps, float scale, Matrix& out,
                      Matrix& alpha, Matrix& rinv, Matrix& logit) {
    if (!da_active()) return false;
    const size_t S = src.size();
    if (S < 2 || S > kDepthAttendMaxSources) return false;
    const size_t T = src[0]->rows(), d = src[0]->cols();
    if (T == 0 || d == 0) return false;
    std::vector<std::unique_ptr<DIn>> ins;
    SrcPtrs sp{};
    for (size_t i = 0; i < S; ++i) {
        ins.push_back(std::make_unique<DIn>(src[i]->get_data(), T * d));
        sp.p[i] = ins.back()->d;
    }
    DIn iw(w.get_data(), d);
    DOut oo(out.get_data(), T * d, "depth_attend_fwd"), oa(alpha.get_data(), T * S, "depth_attend_fwd.alpha"),
        orr(rinv.get_data(), T * S, "depth_attend_fwd.rinv"), ol(logit.get_data(), T * S, "depth_attend_fwd.logit");
    const int blocks = static_cast<int>((T + kWarps - 1) / kWarps);
    k_da_fwd<<<blocks, kWarps * 32>>>(sp, static_cast<int>(S), iw.d, eps, scale, oo.d, oa.d, orr.d, ol.d,
                                      static_cast<int>(T), static_cast<int>(d));
    da_check(cudaGetLastError(), "forward");
    oo.finish();
    oa.finish();
    orr.finish();
    ol.finish();
    return true;
}

bool depth_attend_bwd(const std::vector<const Matrix*>& src, const Matrix& w, float scale, const Matrix& alpha,
                      const Matrix& rinv, const Matrix& logit, const Matrix& dy, const std::vector<Matrix*>& ds,
                      Matrix* dw) {
    if (!da_active()) return false;
    const size_t S = src.size();
    if (S < 2 || S > kDepthAttendMaxSources || ds.size() != S) return false;
    const size_t T = src[0]->rows(), d = src[0]->cols();
    if (T == 0 || d == 0) return false;
    const size_t shm = static_cast<size_t>(kWarps) * (d + kMax) * sizeof(float);
    if (shm > 48 * 1024) return false;  // d > ~2.9K: the host loop
    std::vector<std::unique_ptr<DIn>> ins;
    SrcPtrs sp{};
    for (size_t i = 0; i < S; ++i) {
        ins.push_back(std::make_unique<DIn>(src[i]->get_data(), T * d));
        sp.p[i] = ins.back()->d;
    }
    DIn iw(w.get_data(), d), ia(alpha.get_data(), T * S), ir(rinv.get_data(), T * S), il(logit.get_data(), T * S),
        idy(dy.get_data(), T * d);
    std::vector<std::unique_ptr<DOut>> outs;
    DstPtrs dp{};
    for (size_t i = 0; i < S; ++i) {
        if (!ds[i]) continue;
        outs.push_back(std::make_unique<DOut>(ds[i]->get_data(), T * d, "depth_attend_bwd.ds"));
        dp.p[i] = outs.back()->d;
    }
    const size_t want = (T + kWarps - 1) / kWarps;
    const int blocks = static_cast<int>(want < 256 ? want : 256);
    const int nw = blocks * kWarps;
    float* part = nullptr;
    if (dw) part = da_alloc(static_cast<size_t>(nw) * d);
    k_da_bwd<<<blocks, kWarps * 32, shm>>>(sp, dp, static_cast<int>(S), iw.d, scale, ia.d, ir.d, il.d, idy.d, part,
                                           static_cast<int>(T), static_cast<int>(d));
    da_check(cudaGetLastError(), "backward");
    if (dw) {
        DOut odw(dw->get_data(), d, "depth_attend_bwd.dw");
        k_da_dw<<<static_cast<int>((d + 255) / 256), 256>>>(part, nw, static_cast<int>(d), odw.d);
        da_check(cudaGetLastError(), "query gradient");
        odw.finish();
        pool::dfree(part);
    }
    for (auto& o : outs) o->finish();
    return true;
}

}  // namespace devops
}  // namespace device
}  // namespace microtorch
