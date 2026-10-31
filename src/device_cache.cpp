// CPU-only stubs for the Phase B1 residency API (docs/CUDA_PHASE_B.md).
// The real implementation lives in src/cuda_resident.cu and is compiled
// only under -DMICROTORCH_CUDA=ON; this file compiles to the no-op
// surface otherwise so callers never need #ifdefs.
#ifndef MICROTORCH_CUDA

#include "microtorch/device_cache.hpp"

namespace microtorch {
namespace device {

void set_residency(bool) {}
bool residency_enabled() {
    return false;
}
void make_resident(const Matrix&) {}
void invalidate(const Matrix&) {}
void evict_all() {}
size_t resident_count() {
    return 0;
}

// Phase B2.0 stubs. CPU builds never allocate a DevState, so the
// release impl is unreachable — it exists so the inline header wrapper
// links.
void set_step_residency(bool) {}
bool step_residency_enabled() {
    return false;
}
void step_begin() {}
void step_end() {}
void step_end_keeping(const std::vector<const Matrix*>&) {}
size_t device_bytes_in_use() {
    return 0;
}
size_t pool_live_bytes() {
    return 0;
}
size_t pool_cached_bytes() {
    return 0;
}
size_t pool_peak_bytes() {
    return 0;
}
void pool_reset_peak() {}
void synchronize() {}
bool gpu_profile_on() {
    return false;
}
void set_blocking_sync() {}
std::vector<std::pair<std::string, double>> gpu_profile_take() {
    return {};
}
std::vector<std::pair<std::string, size_t>> gpu_profile_counts() {
    return {};
}
namespace detail {
void gpu_profile_begin(const char*) {}
void gpu_profile_end(const char*) {}
}  // namespace detail
namespace detail {
void release_devstate_impl(DevState*) {}
}  // namespace detail

// B2.1b stubs: no CUDA, nothing can defer, nothing is ever stale.
void set_defer_downloads(bool) {}
bool defer_downloads_enabled() {
    return false;
}
bool deferring() {
    return false;
}
bool host_stale(const Matrix&) {
    return false;
}
void materialize(const Matrix&) {}
void materialize_all() {}
size_t stale_count() {
    return 0;
}
void discard(const Matrix&) {}
void release(const Matrix&) {}
void devcheck_host_read(const Matrix&, const char*) {}

// CPU-only builds: the op-set entries are no-ops returning false, so the
// tape always runs its own loops and callers never need #ifdefs.
namespace devops {
bool conv1d_causal_fwd(const Matrix&, const Matrix&, const Matrix*, size_t, Matrix&) {
    return false;
}
bool conv1d_causal_bwd(const Matrix&, const Matrix&, const Matrix&, size_t, Matrix*, Matrix*, Matrix*) {
    return false;
}
bool selective_scan_fwd(const Matrix&, const Matrix&, const Matrix&, const Matrix&, const Matrix&,
                        const Matrix*, bool, size_t, Matrix&, Matrix&, Matrix&) { return false; }
bool selective_scan_bwd(const Matrix&, const Matrix&, const Matrix&, const Matrix&, const Matrix&,
                        const Matrix*, const Matrix&, const Matrix&, const Matrix&, bool, size_t,
                        Matrix*, Matrix*, Matrix*, Matrix*, Matrix*, Matrix*) { return false; }
bool add(const Matrix&, const Matrix&, Matrix&) {
    return false;
}
bool sub(const Matrix&, const Matrix&, Matrix&) {
    return false;
}
bool mul(const Matrix&, const Matrix&, Matrix&) {
    return false;
}
bool scale(const Matrix&, float, Matrix&) {
    return false;
}
bool axpy(Matrix&, float, const Matrix&) {
    return false;
}
bool fill(Matrix&, float) {
    return false;
}
bool sigmoid_fwd(const Matrix&, Matrix&) {
    return false;
}
bool sigmoid_bwd(const Matrix&, const Matrix&, Matrix&) {
    return false;
}
bool silu_fwd(const Matrix&, Matrix&) {
    return false;
}
bool silu_bwd(const Matrix&, const Matrix&, Matrix&) {
    return false;
}
bool swiglu_fwd(const Matrix&, const Matrix&, Matrix&) {
    return false;
}
bool swiglu_bwd(const Matrix&, const Matrix&, const Matrix&, Matrix*, Matrix*) {
    return false;
}
bool gelu_fwd(const Matrix&, Matrix&) {
    return false;
}
bool gelu_bwd(const Matrix&, const Matrix&, Matrix&) {
    return false;
}
bool softmax_fwd(const Matrix&, Matrix&) {
    return false;
}
bool softmax_bwd(const Matrix&, const Matrix&, Matrix&) {
    return false;
}
bool attn_masked_softmax(Matrix&, float, size_t, bool) {
    return false;
}
bool swa_masked_softmax(Matrix&, float, size_t, size_t, size_t) {
    return false;
}
bool attn_softmax_bwd_inplace(Matrix&, const Matrix&, float) {
    return false;
}
bool attn_blocked_fwd(const Matrix&, const Matrix&, const Matrix&, float, size_t, bool, Matrix&, Matrix&) {
    return false;
}
bool attn_blocked_bwd(const Matrix&, const Matrix&, const Matrix&, const Matrix&, const Matrix&, float, size_t,
                      bool, Matrix&, Matrix&, Matrix&) {
    return false;
}
bool attn_heads_fwd(const Matrix&, size_t, float, size_t, bool, Matrix&, Matrix&) {
    return false;
}
bool attn_heads_bwd(const Matrix&, size_t, const Matrix&, const Matrix&, float, size_t, bool, Matrix&) {
    return false;
}
bool attn_heads_fwd_ts(const Matrix&, size_t, float, size_t, const std::vector<int>&, Matrix&, Matrix&) {
    return false;
}
bool flash_attention_supported(size_t head_dim) {
    return head_dim > 0 && head_dim <= 64;  // the device kernels' limit, so CPU and CUDA builds agree
}
bool attn_flash_fwd(const Matrix&, size_t, float, size_t, int, const std::vector<int>*, Matrix&, Matrix&) {
    return false;
}
bool attn_flash_bwd(const Matrix&, size_t, const Matrix&, const Matrix&, const Matrix&, float, size_t, int,
                    const std::vector<int>*, Matrix&) {
    return false;
}
bool embed_gather(const Matrix&, const int*, size_t, Matrix&) {
    return false;
}
bool ce_fwd(const Matrix&, const int*, Matrix&, float&) {
    return false;
}
bool ce_bwd(const Matrix&, const int*, float, Matrix&) {
    return false;
}
bool ce_fwd_masked(const Matrix&, const int*, size_t, Matrix&, float&) {
    return false;
}
bool ce_bwd_masked(const Matrix&, const int*, float, Matrix&) {
    return false;
}
bool linear_ce(const Matrix&, DevState**, const Matrix&, DevState**, bool, const Matrix*, const int*, size_t, float,
               size_t, Precision, Matrix*, Matrix*, Matrix*, float&) {
    return false;
}
bool adamw_step(Matrix&, const Matrix&, Matrix&, Matrix&, float, float, float, float, float, float,
                float) {
    return false;
}
bool sgd_step(Matrix&, const Matrix&, Matrix*, float, float) {
    return false;
}
float* opt_state_new(size_t) {
    return nullptr;
}
void opt_state_free(float*) {}
void opt_state_download(float*, const float*, size_t) {}
void opt_state_upload(float*, const float*, size_t) {}
bool adamw_step_dev(Matrix&, const Matrix&, float*, float*, float, float, float, float, float,
                    float, float) {
    return false;
}
bool sgd_step_dev(Matrix&, const Matrix&, float*, float, float) {
    return false;
}
bool layernorm_fwd(const Matrix&, const Matrix&, const Matrix&, float, Matrix&, Matrix&,
                   std::vector<float>&) {
    return false;
}
bool layernorm_bwd(const Matrix&, const Matrix&, const std::vector<float>&, const Matrix&, bool,
                   Matrix*, Matrix*, bool, Matrix*) {
    return false;
}
bool rmsnorm_fwd(const Matrix&, const Matrix&, float, Matrix&, std::vector<float>&) {
    return false;
}
bool rmsnorm_bwd(const Matrix&, const Matrix&, const std::vector<float>&, const Matrix&, bool,
                 Matrix*, bool, Matrix*) {
    return false;
}
bool rmsnorm_fwd(const Matrix&, const Matrix&, float, Matrix&, Matrix&) {
    return false;
}
bool rmsnorm_bwd(const Matrix&, const Matrix&, const Matrix&, const Matrix&, bool, Matrix*, bool, Matrix*) {
    return false;
}
bool rope_fwd(const Matrix&, const std::vector<int>&, size_t, size_t, float, Matrix&, bool, size_t) {
    return false;
}
bool rope_bwd(const Matrix&, const std::vector<int>&, size_t, size_t, float, Matrix&, bool, size_t) {
    return false;
}
bool rmsnorm_group_fwd(const Matrix&, const Matrix&, size_t, float, Matrix&, Matrix&) { return false; }
bool rmsnorm_group_bwd(const Matrix&, const Matrix&, const Matrix&, const Matrix&, size_t, bool, Matrix*, bool,
                       Matrix*) {
    return false;
}
bool repeat_heads_fwd(const Matrix&, size_t, size_t, size_t, Matrix&) { return false; }
bool repeat_heads_bwd(const Matrix&, size_t, size_t, size_t, Matrix&) { return false; }
bool slice_cols(const Matrix&, size_t, size_t, Matrix&) {
    return false;
}
bool place_cols(const Matrix&, size_t, Matrix&) {
    return false;
}
bool concat_cols(const std::vector<const Matrix*>&, Matrix&) {
    return false;
}
bool accumulate_cols(Matrix&, size_t, const Matrix&, bool) {
    return false;
}
bool gather_rows(const Matrix&, const std::vector<int>&, size_t, Matrix&) {
    return false;
}
bool scatter_rows(const Matrix&, const std::vector<int>&, size_t, size_t, bool, Matrix&) {
    return false;
}
bool scatter_rows_multi(const std::vector<const Matrix*>&, const std::vector<int>&, size_t, size_t, Matrix&) {
    return false;
}
bool scatter_rows_into(const Matrix&, const std::vector<int>&, Matrix&, bool) {
    return false;
}
bool logsumexp_fwd(const Matrix&, Matrix&) {
    return false;
}
bool logsumexp_bwd(const Matrix&, const Matrix&, const Matrix&, Matrix&) {
    return false;
}
bool mul_col_fwd(const Matrix&, const Matrix&, Matrix&) {
    return false;
}
bool mul_col_dx(const Matrix&, const Matrix&, Matrix&) {
    return false;
}
bool mul_col_dc(const Matrix&, const Matrix&, Matrix&) {
    return false;
}
bool transpose(const Matrix&, Matrix&) {
    return false;
}
bool embed_scatter_add(const Matrix&, const std::vector<int>&, Matrix&) {
    return false;
}
bool sumsq(const std::vector<const Matrix*>&, std::vector<double>&) {
    return false;
}
bool ternary_fwd(const Matrix&, Matrix&) {
    return false;
}
bool act_quant_fwd(const Matrix&, int, Matrix&) {
    return false;
}
}  // namespace devops
}  // namespace device
}  // namespace microtorch

#endif  // !MICROTORCH_CUDA
