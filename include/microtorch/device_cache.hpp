// Phase B1: explicit device residency for host tensors (docs/CUDA_PHASE_B.md).
//
// Opt-in table of host->device tensor copies, keyed by the host data
// pointer. Nothing becomes resident implicitly. Contract: a resident
// matrix must not resize/reallocate while resident, and host mutation
// requires invalidate() or a fresh make_resident() — there is no
// automatic staleness detection, by design (no silent wrongness class).
//
// CPU-only builds: every call is a no-op (residency_enabled() == false),
// so callers never need #ifdefs.
#pragma once
#include <vector>

#include "matrix.hpp"

namespace microtorch {
namespace device {

void set_residency(bool on);      // master switch (default off)
bool residency_enabled();

void make_resident(const Matrix& m);   // upload now (or refresh in place)
void invalidate(const Matrix& m);      // free the device copy, if any
void evict_all();                      // free everything
size_t resident_count();               // table size (diagnostics)

#ifdef MICROTORCH_CUDA
// B1 matmul path: uses device copies for resident operands, temp-uploads
// the rest, downloads C. Returns false (doing nothing) when residency is
// disabled, so the caller can fall through to the Phase A path.
bool resident_matmul(const Matrix& a, const Matrix& b, Matrix& c);
#endif

// ---- Phase B2.0: training-step residency plumbing (docs/CUDA_PHASE_B2.md)
//
// Write-through and epoch-scoped: gemm() results always land in host
// storage (host data is NEVER stale in B2.0); device copies of operands
// are cached in the owning Variable's DevState slot and trusted only
// inside the step_begin()/step_end() window that stamped them. Host
// mutations (optimizer, FD pokes, checkpoint load) fall between windows,
// so the staleness class is killed by scope, not by discipline — code
// that never opens a window gets today's behaviour unchanged.

struct DevState;  // opaque; defined by the CUDA TU, owned by a Variable

void set_step_residency(bool on);  // master switch (default off)
bool step_residency_enabled();
void step_begin();  // open a cache window: bump the epoch, enable caching
void step_end();    // close it: caches are ignored until the next begin

namespace detail {
void release_devstate_impl(DevState*);  // frees the buffer + the struct
}
// Variable-dtor hook. Inline null fast path: every Variable pays one
// branch; only Variables that were actually cached call out of line.
inline void release_devstate(DevState*& s) {
    if (s) {
        detail::release_devstate_impl(s);
        s = nullptr;
    }
}

enum class Trans { N, T };
// C = op(A) . op(B), op in {identity, transpose}. devA/devB are the
// owning Variables' DevState slots, nullable (grads and non-Var
// intermediates pass nullptr: temp upload, no caching). On the CUDA
// step-resident path the transpose is kernel index math — no
// materialized host transpose; the CPU / Phase-A fallbacks materialize
// and match today's numerics exactly.
Matrix gemm(const Matrix& A, DevState** devA, Trans tA,
            const Matrix& B, DevState** devB, Trans tB);

#ifdef MICROTORCH_CUDA
// B2 gemm path; returns false (doing nothing) unless step residency is
// on AND a window is open, so the caller falls through to B1/Phase A.
bool step_resident_gemm(const Matrix& A, DevState** devA, Trans tA,
                        const Matrix& B, DevState** devB, Trans tB,
                        Matrix& C);
#endif

// ---- Phase B2.1a: the device-side op set (docs/CUDA_PHASE_B2.md sec 4)
//
// One canonical entry point per op; kernels implement EXACTLY the CPU
// tape formulas in ops.cpp (which stay in place as the fallback AND the
// reference the parity test pins against). B2.1a is write-through: host
// in, host out, so host data is never stale; deferred downloads + the
// validity-flag contract + DEVCHECK are B2.1b.
//
// Every entry returns false unless (set_device_ops(true) or env
// MICROTORCH_DEVICE_OPS=1) AND device::get()==CUDA — the caller then
// falls through to today's CPU loop, so CPU numerics are untouched.
// CPU-only builds: stubs returning false (callers never need #ifdefs).

void set_device_ops(bool on);   // master switch (default off)
bool device_ops_enabled();

namespace devops {
// elementwise
bool add(const Matrix& a, const Matrix& b, Matrix& y);    // y = a + b
bool sub(const Matrix& a, const Matrix& b, Matrix& y);    // y = a - b
bool mul(const Matrix& a, const Matrix& b, Matrix& y);    // y = a .* b
bool scale(const Matrix& a, float s, Matrix& y);          // y = a * s
bool axpy(Matrix& y, float a, const Matrix& x);           // y += a * x
bool fill(Matrix& y, float v);                            // y = v
// activations (bwd formulas are ops.cpp's, incl. the CORRECT tanh-GELU
// derivative — never the vendored apply_gelu_derivative)
bool sigmoid_fwd(const Matrix& x, Matrix& y);
bool sigmoid_bwd(const Matrix& s, const Matrix& dy, Matrix& dx);
bool gelu_fwd(const Matrix& x, Matrix& y);
bool gelu_bwd(const Matrix& x, const Matrix& dy, Matrix& dx);
// rowwise (parallel reduction order != serial CPU order: results carry
// fp32 tolerance, never bitwise claims — test_step_residency's rule)
bool softmax_fwd(const Matrix& x, Matrix& y);
bool softmax_bwd(const Matrix& S, const Matrix& dY, Matrix& dX);
bool layernorm_fwd(const Matrix& x, const Matrix& gamma, const Matrix& beta,
                   float eps, Matrix& y, Matrix& xhat,
                   std::vector<float>& rstd);
bool layernorm_bwd(const Matrix& dY, const Matrix& xhat,
                   const std::vector<float>& rstd, const Matrix& gamma,
                   bool want_dgb, Matrix* dg, Matrix* db, bool want_dx,
                   Matrix* dx);
bool rmsnorm_fwd(const Matrix& x, const Matrix& w, float eps, Matrix& y,
                 std::vector<float>& rms_inv);
bool rmsnorm_bwd(const Matrix& dY, const Matrix& x,
                 const std::vector<float>& rms_inv, const Matrix& w,
                 bool want_dw, Matrix* dw, bool want_dx, Matrix* dx);
}  // namespace devops

}  // namespace device
}  // namespace microtorch
