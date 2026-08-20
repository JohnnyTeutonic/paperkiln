#include "microtorch/device.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "microtorch/device_cache.hpp"

#ifdef MICROTORCH_CUDA
// transformer_core's REAL CUDA surface is namespace cuda in include/cuda/
// (cuda::matmul: host Matrix in/out, device round-trip inside). The
// global-namespace CudaMatrix in include/cuda_kernels.hpp is a stale
// declaration with no definitions in the archive -- linking against it
// was the round-2 Colab failure (2026-07-30).
#include "cuda/matrix_ops.cuh"
#endif

namespace microtorch {
namespace device {

namespace {
Device g_device = Device::CPU;
bool g_device_ops = false;  // B2.1a master switch (docs/CUDA_PHASE_B2.md)
}

void set_device_ops(bool on) { g_device_ops = on; }
bool device_ops_enabled() { return g_device_ops; }

Device get() {
    return g_device;
}

bool cuda_compiled() {
#ifdef MICROTORCH_CUDA
    return true;
#else
    return false;
#endif
}

void set(Device d) {
    if (d == Device::CUDA && !cuda_compiled()) {
        throw std::runtime_error("device::set(CUDA): built without -DMICROTORCH_CUDA=ON");
    }
    g_device = d;
}

void set_from_env() {
    // Phase B2: opt into step residency from the environment, so sweep
    // configs can flip it without a rebuild (no-op in CPU builds).
    if (const char* r = std::getenv("MICROTORCH_STEP_RESIDENCY")) {
        if (std::strcmp(r, "1") == 0) set_step_residency(true);
    }
    // B2.1a: opt the tape's op set onto the device from the environment,
    // same no-rebuild rule as step residency (no-op in CPU builds).
    if (const char* o = std::getenv("MICROTORCH_DEVICE_OPS")) {
        if (std::strcmp(o, "1") == 0) set_device_ops(true);
    }
    const char* v = std::getenv("MICROTORCH_DEVICE");
    if (!v) return;
    if (std::strcmp(v, "cuda") == 0)
        set(Device::CUDA);
    else if (std::strcmp(v, "cpu") == 0)
        set(Device::CPU);
    else
        throw std::runtime_error("MICROTORCH_DEVICE must be cpu or cuda");
}

Matrix matmul(const Matrix& a, const Matrix& b) {
#ifdef MICROTORCH_CUDA
    if (g_device == Device::CUDA) {
        Matrix c(a.rows(), b.cols());
        // Phase B1 first (no-op branch unless residency is enabled):
        // resident operands stay on device (docs/CUDA_PHASE_B.md).
        if (resident_matmul(a, b, c)) return c;
        // Phase A: host-resident tensors, per-call round trip. Correctness
        // is gradcheck-gated (the same suite runs under either device).
        cuda::matmul(a, b, c);
        return c;
    }
#endif
    return matmul_optimized(a, b);
}

// Phase B2.0 (docs/CUDA_PHASE_B2.md): gemm with transpose flags. The
// dispatch chain per call: step-resident path (flags native, operands
// cached in Variable-owned DevState within an open window) -> B1/Phase A
// with materialized transposes -> CPU with materialized transposes. The
// fallbacks run the exact ops today's code runs, so numerics off the B2
// path are bit-identical to the pre-B2 tape.
Matrix gemm(const Matrix& A, DevState** devA, Trans tA,
            const Matrix& B, DevState** devB, Trans tB) {
    const size_t M = (tA == Trans::T) ? A.cols() : A.rows();
    const size_t Ka = (tA == Trans::T) ? A.rows() : A.cols();
    const size_t Kb = (tB == Trans::T) ? B.cols() : B.rows();
    const size_t N = (tB == Trans::T) ? B.rows() : B.cols();
    if (Ka != Kb) throw std::runtime_error("gemm: inner dimensions disagree");
#ifdef MICROTORCH_CUDA
    if (g_device == Device::CUDA) {
        Matrix C(M, N);
        if (step_resident_gemm(A, devA, tA, B, devB, tB, C)) return C;
        auto run = [&C](const Matrix& a, const Matrix& b) {
            if (resident_matmul(a, b, C)) return;
            cuda::matmul(a, b, C);
        };
        if (tA == Trans::N && tB == Trans::N)
            run(A, B);
        else if (tA == Trans::T && tB == Trans::N)
            run(A.transpose(), B);
        else if (tA == Trans::N && tB == Trans::T)
            run(A, B.transpose());
        else
            run(A.transpose(), B.transpose());
        return C;
    }
#endif
    (void)devA;
    (void)devB;
    (void)M;
    (void)N;
    if (tA == Trans::N && tB == Trans::N) return matmul_optimized(A, B);
    if (tA == Trans::T && tB == Trans::N) return matmul_optimized(A.transpose(), B);
    if (tA == Trans::N && tB == Trans::T) return matmul_optimized(A, B.transpose());
    return matmul_optimized(A.transpose(), B.transpose());
}

}  // namespace device
}  // namespace microtorch
