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
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "matrix.hpp"

namespace microtorch {
namespace device {

void set_residency(bool on);  // master switch (default off)
bool residency_enabled();

void make_resident(const Matrix& m);  // upload now (or refresh in place)
void invalidate(const Matrix& m);     // free the device copy, if any
void evict_all();                     // free everything
size_t resident_count();              // table size (diagnostics)

#ifdef MICROTORCH_CUDA
// B1 matmul path: uses device copies for resident operands, temp-uploads
// the rest, downloads C. Returns false (doing nothing) when residency is
// disabled, so the caller can fall through to the Phase A path.
bool resident_matmul(const Matrix& a, const Matrix& b, Matrix& c);
// Phase A matmul (every GEMM outside a step window when residency is off:
// score, chat, eval, sample), host in and out, through this library's
// device pool: transformer_core's cuda::matmul call (its cuBLAS handle,
// cublasSgemm, the same operands and leading dimensions, so the same
// floats), but its buffers come from pool::dmalloc. cuda::matmul's own
// buffer cache keeps one block per exact element count for the life of
// the process, outside MICROTORCH_POOL_MB, and exit(1)s on a failed
// cudaMalloc with no unwinding; a score run over many context lengths
// (and a MoR model's per-step active-row counts) grew it to ~1 GB per
// process, and beside a training job on the same L4 that ended a rung's
// `mtstudio score` mid-chunk (25 Oct 2026). Here the cache is capped and
// returned to the driver before a retry, and a failure throws.
void pooled_matmul(const Matrix& a, const Matrix& b, Matrix& c);
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
// step_end() that leaves the device-fresh values of `keep` (parameters the
// optimizer just updated on the device) on the device instead of
// downloading them: the next step_begin() adopts them as that window's
// values, so the parameters never cross the bus between two training steps.
// Until then their host storage is STALE: the caller guarantees nothing
// reads it raw before the next window (materialize() still works, and a
// plain step_end()/materialize_all() brings them home). mtstudio uses it
// only when the next thing to run is another training step.
void step_end_keeping(const std::vector<const Matrix*>& keep);

// Device bytes currently allocated (total - free, via cudaMemGetInfo);
// 0 on CPU-only builds. Exists so a test can assert memory is FLAT over
// many steps: every CUDA leak this project has shipped was invisible to
// a suite that only ran 9-50 steps.
// It counts the caching pool's free list and every other process on the
// device too, so flat-memory assertions use pool_live_bytes() instead.
size_t device_bytes_in_use();
// Bytes this process holds through the caching device pool (handed out and
// not yet returned), excluding the pool's cache and other processes: the
// leak-sensitive number. pool_cached_bytes() is the cache itself (bounded
// by MICROTORCH_POOL_MB). Both 0 on CPU-only builds.
size_t pool_live_bytes();
size_t pool_cached_bytes();
// High-water mark of pool_live_bytes() since the last reset (the step's
// true device working set, which nvidia-smi cannot separate from the cache).
size_t pool_peak_bytes();
void pool_reset_peak();
// Wait for all queued device work (cudaDeviceSynchronize); no-op on CPU
// builds. For phase timers only (MICROTORCH_STEP_TIMING), never the
// training path: every copy back to host already orders itself.
void synchronize();
// MICROTORCH_GPU_PROFILE=1: CUDA events around every device op and gemm,
// summed per op name. gpu_profile_take() waits for the device and returns
// {op: ms} since the last call (empty when off and on CPU builds).
bool gpu_profile_on();
// cudaDeviceScheduleBlockingSync (env MICROTORCH_BLOCKING_SYNC=1, before
// the first CUDA call): waits sleep, so CPU time is host work alone.
void set_blocking_sync();
std::vector<std::pair<std::string, double>> gpu_profile_take();
namespace detail {
void gpu_profile_begin(const char* op);  // no-op unless on, or an op is already open
void gpu_profile_end(const char* op);    // closes the open op if it is `op`
}  // namespace detail

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
Matrix gemm(const Matrix& A, DevState** devA, Trans tA, const Matrix& B, DevState** devB, Trans tB);

// ---- Mixed precision (BF16 autocast for the projection GEMMs, 24 Oct 2026)
//
// Precision::BF16 rounds both GEMM inputs to bfloat16 (round to nearest
// even) and accumulates and returns FP32: master weights, optimiser state,
// gradients, norms, softmax, RoPE, residual adds and the loss all stay
// FP32, because only the GEMM INPUTS are rounded and every output is a
// float. No loss scaling: bfloat16 keeps float32's 8-bit exponent (normal
// range down to ~1.2e-38), so nothing a float holds underflows when
// rounded, and no gradient is ever stored in bfloat16 (FP16, with 5
// exponent bits and normals only above ~6.1e-5, is what needs scaling).
// On the CUDA step-resident path the inputs are converted on the device
// (__float2bfloat16_rn) and multiplied by cublasGemmEx (CUDA_R_16BF in,
// CUDA_R_32F out, CUBLAS_COMPUTE_32F; compute capability 8.0 or newer).
// Everywhere else (CPU, the Phase A/B1 device paths) the inputs are
// rounded on the host and the FP32 path multiplies them: an emulation
// whose only difference from the device is the FP32 summation order
// (a bfloat16 x bfloat16 product is exact in float32).
// Precision::FP32 is the existing gemm, unchanged bit for bit.
enum class Precision { FP32, BF16 };
// Process-wide precision of ops::matmul (default FP32; mtstudio sets it
// from the spec's train.precision). Only ops::matmul reads it: attention's
// internal products and every other op stay FP32.
void set_matmul_precision(Precision p);
Precision matmul_precision();
const char* precision_name(Precision p);  // "fp32" / "bf16"
// float -> nearest bfloat16 (ties to even), returned as the float it
// equals. NaN stays NaN (quiet), +-inf stays inf, a finite value past the
// largest bfloat16 rounds to inf: __float2bfloat16_rn's results.
inline float bf16_round(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0) {
        u = (u | 0x00400000u) & 0xffff0000u;  // quiet NaN, sign kept
    } else {
        u += 0x7fffu + ((u >> 16) & 1u);
        u &= 0xffff0000u;
    }
    float y;
    std::memcpy(&y, &u, 4);
    return y;
}
Matrix bf16_rounded(const Matrix& m);  // elementwise bf16_round copy
// gemm at a precision: FP32 is the 6-argument gemm exactly.
Matrix gemm(const Matrix& A, DevState** devA, Trans tA, const Matrix& B, DevState** devB, Trans tB,
            Precision prec);

// Profile tag (MICROTORCH_GPU_PROFILE): while a ProfileTag is alive, gemm
// reports its time under the tag's name (a string literal, e.g.
// "gemm_head") instead of "gemm"; ops::matmul carries the tag into its
// backward. Names only: numerics are untouched, and with the profile off
// nothing reads it.
const char* profile_tag();
void set_profile_tag(const char* tag);
struct ProfileTag {
    const char* prev;
    explicit ProfileTag(const char* t) : prev(profile_tag()) { set_profile_tag(t); }
    ~ProfileTag() { set_profile_tag(prev); }
    ProfileTag(const ProfileTag&) = delete;
    ProfileTag& operator=(const ProfileTag&) = delete;
};

#ifdef MICROTORCH_CUDA
// B2 gemm path; returns false (doing nothing) unless step residency is
// on AND a window is open, so the caller falls through to B1/Phase A.
bool step_resident_gemm(const Matrix& A, DevState** devA, Trans tA, const Matrix& B,
                        DevState** devB, Trans tB, Matrix& C, Precision prec = Precision::FP32);
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

void set_device_ops(bool on);  // master switch (default off)
bool device_ops_enabled();

// Storage for an op output that a device op will overwrite in full. With
// the device op set live (switch on, process device CUDA) its elements are
// left unwritten: the host used to zero-fill every output, and gemm, op
// and gradient outputs that stay deferred on the device never needed it
// (zero-fills were ~44% of a d512 forward's host samples, 20 Oct 2026).
// Otherwise, and when the vendored Matrix predates the uninitialised
// constructor, it is zero-filled exactly as Matrix(rows, cols). A caller
// whose device op declines (returns false) and whose host fallback reads
// the output before writing it must zero it first: zero_if_uninit().
Matrix out_matrix(size_t rows, size_t cols);
// True when out_matrix() hands out unwritten storage right now.
bool out_matrix_uninit();
inline void zero_if_uninit(Matrix& m) {
    if (out_matrix_uninit()) m.fill(0.0f);
}

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

// True while device op outputs defer (switch on, device ops live, window
// open): a devops result written now stays on the device.
bool deferring();
// True iff the device holds a fresher copy of m's storage than the host.
bool host_stale(const Matrix& m);
// Download m's device-fresh copy into host storage (no-op if not stale).
void materialize(const Matrix& m);
// Download every stale buffer (step boundary, eval, checkpoint sweep).
void materialize_all();
// Value-cache entries holding a device-fresher copy than host storage right
// now (diagnostics; 0 on CPU builds). After a NoGrad forward inside a
// deferring window, once every Var of it has died, this is 0: any entry
// left is a forward cache that died without discard(), the dying-temporary
// class behind defect D1 (tests/test_gpt2_device.cpp).
size_t stale_count();
// Drop m's value-cache entry WITHOUT downloading — for a deferred
// output that will never be read on host and is about to be destroyed.
// A stale entry keyed by a freed host pointer makes step_end()'s
// materialize_all() write into freed memory (glibc heap corruption —
// found by the B2.2 T4 valgrind run, 30 Aug 2026): every op whose
// deferred temporary dies before the step boundary MUST discard it.
void discard(const Matrix& m);
// Free the device copy of an activation that nothing reads again: its
// forward consumers have all run and no backward reads it (a projection
// feeding only a concat, an add or a column slice). The tape keeps the
// Variable, and with it this buffer, alive until the step ends; this hands
// the device memory back now. A device-only value is marked released: any
// later device read or materialize() throws instead of reading garbage
// (host storage was never written). No-op outside a deferring window and
// on CPU builds. LlamaBlock::forward uses it (~40% of a dense llama
// step's activations, 20 Oct 2026).
void release(const Matrix& m);
// Throws if m is host-stale — the DEVCHECK assert. Call sites compile to
// nothing unless MICROTORCH_DEVCHECK is defined at build time.
void devcheck_host_read(const Matrix& m, const char* where);
#ifdef MICROTORCH_DEVCHECK
#define MT_DEVCHECK_HOST_READ(m, where) ::microtorch::device::devcheck_host_read((m), (where))
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
float* vc_output(const float* key, size_t n, bool& deferred, bool need_current,
                 const float* host_src);
}  // namespace detail
#endif

namespace devops {
// elementwise
bool add(const Matrix& a, const Matrix& b, Matrix& y);  // y = a + b
bool sub(const Matrix& a, const Matrix& b, Matrix& y);  // y = a - b
bool mul(const Matrix& a, const Matrix& b, Matrix& y);  // y = a .* b
bool scale(const Matrix& a, float s, Matrix& y);        // y = a * s
bool axpy(Matrix& y, float a, const Matrix& x);         // y += a * x
bool fill(Matrix& y, float v);                          // y = v
// activations (bwd formulas are ops.cpp's, incl. the CORRECT tanh-GELU
// derivative — never the vendored apply_gelu_derivative)
bool sigmoid_fwd(const Matrix& x, Matrix& y);
bool sigmoid_bwd(const Matrix& s, const Matrix& dy, Matrix& dx);
bool silu_fwd(const Matrix& x, Matrix& y);
bool silu_bwd(const Matrix& x, const Matrix& dy, Matrix& dx);
// ops::swiglu: y = silu(g) .* u; backward dg = (dy .* u) silu'(g) and
// du = dy .* silu(g), silu recomputed (dg/du may be null: not wanted).
bool swiglu_fwd(const Matrix& g, const Matrix& u, Matrix& y);
bool swiglu_bwd(const Matrix& g, const Matrix& u, const Matrix& dy, Matrix* dg, Matrix* du);
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
bool attn_masked_softmax(Matrix& A, float scale, size_t seq_len, bool causal);
bool swa_masked_softmax(Matrix& A, float scale, size_t seq_len, size_t window, size_t sinks);
bool attn_softmax_bwd_inplace(Matrix& ds, const Matrix& A, float scale);
// Blocked attention (ops::fused_attention_blocked) on-device: rows are B
// stacked sequences of length seq_len, each attending only within its own
// block. A is block-local [T, seq_len] (A(i, j) weights key row
// i - i % seq_len + j), kept from forward for backward. Backward writes
// all three gradients.
bool attn_blocked_fwd(const Matrix& q, const Matrix& k, const Matrix& v, float scale, size_t seq_len,
                      bool causal, Matrix& A, Matrix& y);
bool attn_blocked_bwd(const Matrix& q, const Matrix& k, const Matrix& v, const Matrix& A, const Matrix& dY,
                      float scale, size_t seq_len, bool causal, Matrix& dq, Matrix& dk, Matrix& dv);
// The same for every head of a fused [T, 3d] qkv in one launch per kernel
// (ops::attention_heads): A is [heads T, seq_len] head-major, y [T, d] in
// the head-concat layout, dqkv [T, 3d] written whole.
bool attn_heads_fwd(const Matrix& qkv, size_t heads, float scale, size_t seq_len, bool causal, Matrix& A,
                    Matrix& y);
bool attn_heads_bwd(const Matrix& qkv, size_t heads, const Matrix& A, const Matrix& dY, float scale, size_t seq_len,
                    bool causal, Matrix& dqkv);
// attn_heads_fwd under the turn_state mask (ops::attention_heads_turn_state):
// code holds one entry per row of qkv (ops::turn_state_codes); key j <= i is
// visible iff ops::turn_state_visible(code[i], code[j]). Masked weights are
// zeros, so attn_heads_bwd with causal = true is its backward.
bool attn_heads_fwd_ts(const Matrix& qkv, size_t heads, float scale, size_t seq_len, const std::vector<int>& code,
                       Matrix& A, Matrix& y);
// Flash attention (src/cuda_flash.cu; ops::attention_heads* with the Flash
// kernel): the same fused qkv layout, mode 0 full (every key of the row's
// seq_len block), 1 causal, 2 turn_state (code: one entry per row, else
// null). The forward writes y [T, d] and lse [H T] (head-major per-row
// log-sum-exp of the scaled scores) and stores no weights; the backward
// recomputes them from qkv and lse and writes all of dqkv [T, 3d]. seq_len
// 0 = one sequence. Head widths up to 64 (flash_attention_supported).
bool flash_attention_supported(size_t head_dim);
bool attn_flash_fwd(const Matrix& qkv, size_t heads, float scale, size_t seq_len, int mode,
                    const std::vector<int>* code, Matrix& lse, Matrix& y);
bool attn_flash_bwd(const Matrix& qkv, size_t heads, const Matrix& lse, const Matrix& y, const Matrix& dY, float scale,
                    size_t seq_len, int mode, const std::vector<int>* code, Matrix& dqkv);
// B2.2: embedding gather (ids bounds-checked by the HOST caller first;
// backward scatter-add stays host until B2.3) and cross-entropy
// (softmax + nll on-device, host receives ONE float; P cached for the
// backward under the same (P - onehot)/N contract as the host op).
bool embed_gather(const Matrix& table, const int* ids, size_t n_ids, Matrix& out);
bool ce_fwd(const Matrix& logits, const int* targets, Matrix& P, float& loss);
bool ce_bwd(const Matrix& P, const int* targets, float g, Matrix& dl);
// Masked cross-entropy (ops::cross_entropy_masked): targets are -1 on masked
// rows (the caller bounds-checks the kept ones). Masked rows add nothing to
// the loss (mean over `kept` rows) and get an exact zero gradient; a kept
// row's P and gradient are the unmasked kernels' bit for bit (only the
// loss sum's reduction tree differs, so the loss carries fp32 tolerance).
bool ce_fwd_masked(const Matrix& logits, const int* targets, size_t kept, Matrix& P, float& loss);
bool ce_bwd_masked(const Matrix& P, const int* targets, float g, Matrix& dl);
// B2.3a: optimizer steps on device, write-through parity seam (state
// round-trips per step for now; persistent device state is B2.3b).
// c1/c2 are the host-computed bias corrections so the per-element math
// matches nn.cpp exactly.
bool adamw_step(Matrix& p, const Matrix& g, Matrix& m, Matrix& v, float lr, float b1, float b2,
                float c1, float c2, float eps, float wd);
bool sgd_step(Matrix& p, const Matrix& g, Matrix* vel, float lr, float mu);
// B2.3b: persistent device optimizer state — an OWNED zeroed device
// buffer (never the pointer-keyed value cache: the B2.2 lifetime rule).
// opt_state_new returns nullptr on CPU builds / devops off, which pins
// the optimizer to its host path for the whole run. The *_dev steps
// take raw device pointers into that buffer; p still round-trips
// (host-authoritative until B2.3c).
float* opt_state_new(size_t n_elems);
void opt_state_free(float* s);
// Checkpoint/resume for device-resident optimizer state: copy the whole
// owned buffer to/from a host array. No-ops on CPU builds (there is no
// device state to move; the host matrices are authoritative there).
void opt_state_download(float* host, const float* dev, size_t n_elems);
void opt_state_upload(float* dev, const float* host, size_t n_elems);
bool adamw_step_dev(Matrix& p, const Matrix& g, float* m_dev, float* v_dev, float lr, float b1,
                    float b2, float c1, float c2, float eps, float wd);
bool sgd_step_dev(Matrix& p, const Matrix& g, float* vel_dev, float lr, float mu);
bool layernorm_fwd(const Matrix& x, const Matrix& gamma, const Matrix& beta, float eps, Matrix& y,
                   Matrix& xhat, std::vector<float>& rstd);
bool layernorm_bwd(const Matrix& dY, const Matrix& xhat, const std::vector<float>& rstd,
                   const Matrix& gamma, bool want_dgb, Matrix* dg, Matrix* db, bool want_dx,
                   Matrix* dx);
bool rmsnorm_fwd(const Matrix& x, const Matrix& w, float eps, Matrix& y,
                 std::vector<float>& rms_inv);
bool rmsnorm_bwd(const Matrix& dY, const Matrix& x, const std::vector<float>& rms_inv,
                 const Matrix& w, bool want_dw, Matrix* dw, bool want_dx, Matrix* dx);
// The same with rms_inv an [R, 1] matrix: it rides the value cache like any
// output, so forward and backward never sync the host for it (ops::rmsnorm).
bool rmsnorm_fwd(const Matrix& x, const Matrix& w, float eps, Matrix& y, Matrix& rms_inv);
bool rmsnorm_bwd(const Matrix& dY, const Matrix& x, const Matrix& rms_inv, const Matrix& w, bool want_dw,
                 Matrix* dw, bool want_dx, Matrix* dx);
// The llama block's head plumbing (ops::apply_rope / slice_cols /
// concat_cols are the reference). RoPE runs over the fused [T, 3d] qkv
// layout: heads h < n_rot_heads of q and k rotate their first n_rot_dims
// dims (0 = head_dim) in adjacent pairs, or in NEOX pairs (j, j +
// n_rot_dims / 2) when neox; v is copied; the backward is the inverse
// rotation. Slice/concat/place are exact copies.
// place_cols writes g into columns [c0, c0 + g.cols()) of dx and zeros
// everywhere else (slice_cols' backward).
bool rope_fwd(const Matrix& x, const std::vector<int>& pos, size_t head_dim, size_t n_rot_heads, float base,
              Matrix& y, bool neox = false, size_t n_rot_dims = 0);
bool rope_bwd(const Matrix& dy, const std::vector<int>& pos, size_t head_dim, size_t n_rot_heads, float base,
              Matrix& dx, bool neox = false, size_t n_rot_dims = 0);
// ops::rmsnorm_group (rms_inv [rows * groups, 1]) and ops::repeat_heads.
bool rmsnorm_group_fwd(const Matrix& x, const Matrix& w, size_t group, float eps, Matrix& y, Matrix& rms_inv);
bool rmsnorm_group_bwd(const Matrix& dY, const Matrix& x, const Matrix& rms_inv, const Matrix& w, size_t group,
                       bool want_dw, Matrix* dw, bool want_dx, Matrix* dx);
bool repeat_heads_fwd(const Matrix& x, size_t heads, size_t group, size_t head_dim, Matrix& y);
bool repeat_heads_bwd(const Matrix& dy, size_t heads, size_t group, size_t head_dim, Matrix& dx);
bool slice_cols(const Matrix& x, size_t c0, size_t n, Matrix& y);
bool place_cols(const Matrix& g, size_t c0, Matrix& dx);
bool concat_cols(const std::vector<const Matrix*>& xs, Matrix& y);
// y[:, c0 : c0 + g.cols()) += g (Variable::accumulate_cols, the slice
// backward). y_is_zero: y was just created zero, so it is written whole
// (zeros outside the range) and never uploaded.
bool accumulate_cols(Matrix& y, size_t c0, const Matrix& g, bool y_is_zero);
// Mixture-of-experts dispatch (ops::gather_rows / scatter_add_rows /
// gather_elems / logsumexp_row / mul_col are the reference; the host
// caller bounds-checks every index first). Rows are C-wide; the element
// ops are the C = 1 case over flat (row * cols + col) indices. A scatter
// sums each destination row's sources in ascending source order from a
// host-built index (no atomics), first write copying when first_copies
// (scatter_add_rows) or adding to 0.0f otherwise (the gather backwards):
// the host loops' exact add sequence, so results agree bit for bit.
bool gather_rows(const Matrix& x, const std::vector<int>& idx, size_t C, Matrix& y);
bool scatter_rows(const Matrix& x, const std::vector<int>& idx, size_t n_rows, size_t C, bool first_copies,
                  Matrix& y);
// scatter_rows over several inputs at once (xs stacked in order; idx is
// their index lists concatenated), first write copying.
bool scatter_rows_multi(const std::vector<const Matrix*>& xs, const std::vector<int>& idx, size_t n_rows,
                        size_t C, Matrix& y);
// y += scatter(g by idx) in place, each row's sources summed from 0.0f in
// ascending order and then added (the host's dx-then-accumulate rounding).
// y_is_zero: y was just created zero, so it is written whole, never uploaded.
bool scatter_rows_into(const Matrix& g, const std::vector<int>& idx, Matrix& y, bool y_is_zero);
// lse[i] = log sum_j exp(x[i,j]) with the host's double accumulation;
// backward dx = g * exp(x - lse).
bool logsumexp_fwd(const Matrix& x, Matrix& lse);
bool logsumexp_bwd(const Matrix& x, const Matrix& lse, const Matrix& g, Matrix& dx);
// y = x .* c (c a column, broadcast over cols); backward dx = dy .* c and
// dc[i] = sum_j dy[i,j] x[i,j], serial per row in the host's order.
bool mul_col_fwd(const Matrix& x, const Matrix& c, Matrix& y);
bool mul_col_dx(const Matrix& dy, const Matrix& c, Matrix& dx);
bool mul_col_dc(const Matrix& dy, const Matrix& x, Matrix& dc);
// The dense step's last host loops. transpose: y = x^T, an exact copy (the
// tied head's E^T and its gradient). embed_scatter_add: table_grad(ids[i], :)
// += g(i, :), each table row's adds in ascending i as ops::embedding's host
// backward does, so bit for bit (ids bounds-checked by the caller). sumsq:
// out[k] = sum of squares of xs[k] in double (clip_grad_norm's accumulator
// type, a different summation order: equal to double rounding).
bool transpose(const Matrix& x, Matrix& y);
bool embed_scatter_add(const Matrix& g, const std::vector<int>& ids, Matrix& table_grad);
bool sumsq(const std::vector<const Matrix*>& xs, std::vector<double>& out);
}  // namespace devops

}  // namespace device
}  // namespace microtorch
