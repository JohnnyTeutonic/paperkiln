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

// Device bytes currently allocated (total - free, via cudaMemGetInfo);
// 0 on CPU-only builds. Exists so a test can assert memory is FLAT over
// many steps: every CUDA leak this project has shipped was invisible to
// a suite that only ran 9-50 steps.
size_t device_bytes_in_use();

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

// ---- Phase B2.1b: deferred downloads (docs/CUDA_PHASE_B2.md checklist)
//
// With the switch on (set_defer_downloads / env MICROTORCH_DEFER_DOWNLOADS=1)
// AND a step window open, devops results and gemm outputs stay on-device in
// an epoch-stamped value cache instead of writing through; chained ops find
// them there and skip the H2D round-trip. Host storage for those buffers is
// STALE until materialize()/materialize_all() downloads them. step_end() is
// a materialize boundary in B2.1b (the optimizer still host-mutates between
// windows), so cross-window staleness cannot exist — the same
// killed-by-scope rule as B2.0. Outside a window, or with the switch off,
// every op writes through exactly as B2.1a.

// TWO INVARIANTS THE WHOLE CACHE RESTS ON — break either and every
// pointer key silently means something else:
//
// 1. THE CACHE IS KEYED BY Matrix::get_data(), AND THAT MUST BE THE HOST
//    POINTER. The vendored Matrix compiles get_data() as
//    `is_on_gpu_ ? gpu_data_ : data_.data()` under CUDA_AVAILABLE, and
//    `is_on_gpu_` is set only by to_gpu() and by copying an already-GPU
//    matrix. microtorch NEVER calls to_gpu() (audited 30 Aug 2026: zero
//    call sites in src/, include/, tools/) — so keys are host pointers
//    and materialize()'s D2H destination is host memory. If anything ever
//    hands a to_gpu() matrix into the tape, keys become device pointers
//    and materialize() memcpys device->device-as-host. Don't arm it.
//    (Related vendored defect, harmless while the above holds: Matrix's
//    move ctor moves data_ but not gpu_data_/is_on_gpu_.)
// 2. ONE STALE ENTRY IS AUTHORITATIVE OVER EVERY OTHER SOURCE for that
//    storage. A stale hit means host is behind; any path that instead
//    uploads from host injects the stale value. window_operand checked
//    the cache in only one of its two branches and that was the B2.3
//    flat-loss bug (30 Aug 2026) — see docs/CUDA_PHASE_B2.md.

void set_defer_downloads(bool on);  // master switch (default off)
bool defer_downloads_enabled();

// True iff the device holds a fresher copy of m's storage than the host.
bool host_stale(const Matrix& m);
// Download m's device-fresh copy into host storage (no-op if not stale).
void materialize(const Matrix& m);
// Download every stale buffer (step boundary, eval, checkpoint sweep).
void materialize_all();
// Drop m's value-cache entry WITHOUT downloading — for a deferred
// output that will never be read on host and is about to be destroyed.
// A stale entry keyed by a freed host pointer makes step_end()'s
// materialize_all() write into freed memory (glibc heap corruption —
// found by the B2.2 T4 valgrind run, 30 Aug 2026): every op whose
// deferred temporary dies before the step boundary MUST discard it.
void discard(const Matrix& m);
// Throws if m is host-stale — the DEVCHECK assert. Call sites compile to
// nothing unless MICROTORCH_DEVCHECK is defined at build time.
void devcheck_host_read(const Matrix& m, const char* where);
#ifdef MICROTORCH_DEVCHECK
#define MT_DEVCHECK_HOST_READ(m, where) \
    ::microtorch::device::devcheck_host_read((m), (where))
#else
#define MT_DEVCHECK_HOST_READ(m, where) ((void)0)
#endif

#ifdef MICROTORCH_CUDA
namespace detail {
// The devops/gemm seam onto the B2.1b value cache (cuda_resident.cu).
// vc_operand: device pointer for an input — stale-value hit (owned=false),
// B1-resident hit (owned=false), else temp upload (owned=true, caller
// frees). vc_output: with defer active, a retained cache buffer for host
// storage `key` (deferred=true, caller must NOT free and must NOT D2H);
// when need_current, the buffer is preloaded from host_src unless the
// cache already holds a fresh stale copy. Returns nullptr with
// deferred=false when defer is inactive (caller does its own temp+D2H).
float* vc_operand(const float* key, size_t n, bool& owned);
float* vc_output(const float* key, size_t n, bool& deferred,
                 bool need_current, const float* host_src);
}  // namespace detail
#endif

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
// B2.2: masked attention softmax on-device, in place over the raw
// score matrix A [T,T] — the host loops in ops.cpp fused_attention /
// swa_attention are the reference semantics (masked entries written as
// hard zeros, never exponentiated). seq_len arrives RESOLVED (sl, not
// 0). The shared backward serves both flavors: masked entries carry
// A == 0 from the forward, so the full-row dot equals the
// visible-range dot and masked outputs vanish with no bookkeeping.
bool attn_masked_softmax(Matrix& A, float scale, size_t seq_len,
                         bool causal);
bool swa_masked_softmax(Matrix& A, float scale, size_t seq_len,
                        size_t window, size_t sinks);
bool attn_softmax_bwd_inplace(Matrix& ds, const Matrix& A, float scale);
// B2.2: embedding gather (ids bounds-checked by the HOST caller first;
// backward scatter-add stays host until B2.3) and cross-entropy
// (softmax + nll on-device, host receives ONE float; P cached for the
// backward under the same (P - onehot)/N contract as the host op).
bool embed_gather(const Matrix& table, const int* ids, size_t n_ids,
                  Matrix& out);
bool ce_fwd(const Matrix& logits, const int* targets, Matrix& P,
            float& loss);
bool ce_bwd(const Matrix& P, const int* targets, float g, Matrix& dl);
// B2.3a: optimizer steps on device, write-through parity seam (state
// round-trips per step for now; persistent device state is B2.3b).
// c1/c2 are the host-computed bias corrections so the per-element math
// matches nn.cpp exactly.
bool adamw_step(Matrix& p, const Matrix& g, Matrix& m, Matrix& v, float lr,
                float b1, float b2, float c1, float c2, float eps,
                float wd);
bool sgd_step(Matrix& p, const Matrix& g, Matrix* vel, float lr, float mu);
// B2.3b: persistent device optimizer state — an OWNED zeroed device
// buffer (never the pointer-keyed value cache: the B2.2 lifetime rule).
// opt_state_new returns nullptr on CPU builds / devops off, which pins
// the optimizer to its host path for the whole run. The *_dev steps
// take raw device pointers into that buffer; p still round-trips
// (host-authoritative until B2.3c).
float* opt_state_new(size_t n_elems);
void opt_state_free(float* s);
bool adamw_step_dev(Matrix& p, const Matrix& g, float* m_dev, float* v_dev,
                    float lr, float b1, float b2, float c1, float c2,
                    float eps, float wd);
bool sgd_step_dev(Matrix& p, const Matrix& g, float* vel_dev, float lr,
                  float mu);
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
