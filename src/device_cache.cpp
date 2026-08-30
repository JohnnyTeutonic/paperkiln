// CPU-only stubs for the Phase B1 residency API (docs/CUDA_PHASE_B.md).
// The real implementation lives in src/cuda_resident.cu and is compiled
// only under -DMICROTORCH_CUDA=ON; this file compiles to the no-op
// surface otherwise so callers never need #ifdefs.
#ifndef MICROTORCH_CUDA

#include "microtorch/device_cache.hpp"

namespace microtorch {
namespace device {

void set_residency(bool) {}
bool residency_enabled() { return false; }
void make_resident(const Matrix&) {}
void invalidate(const Matrix&) {}
void evict_all() {}
size_t resident_count() { return 0; }

// Phase B2.0 stubs. CPU builds never allocate a DevState, so the
// release impl is unreachable — it exists so the inline header wrapper
// links.
void set_step_residency(bool) {}
bool step_residency_enabled() { return false; }
void step_begin() {}
void step_end() {}
namespace detail {
void release_devstate_impl(DevState*) {}
}  // namespace detail

// B2.1b stubs: no CUDA, nothing can defer, nothing is ever stale.
void set_defer_downloads(bool) {}
bool defer_downloads_enabled() { return false; }
bool host_stale(const Matrix&) { return false; }
void materialize(const Matrix&) {}
void materialize_all() {}
void discard(const Matrix&) {}
void devcheck_host_read(const Matrix&, const char*) {}

// CPU-only builds: the op-set entries are no-ops returning false, so the
// tape always runs its own loops and callers never need #ifdefs.
namespace devops {
bool add(const Matrix&, const Matrix&, Matrix&) { return false; }
bool sub(const Matrix&, const Matrix&, Matrix&) { return false; }
bool mul(const Matrix&, const Matrix&, Matrix&) { return false; }
bool scale(const Matrix&, float, Matrix&) { return false; }
bool axpy(Matrix&, float, const Matrix&) { return false; }
bool fill(Matrix&, float) { return false; }
bool sigmoid_fwd(const Matrix&, Matrix&) { return false; }
bool sigmoid_bwd(const Matrix&, const Matrix&, Matrix&) { return false; }
bool gelu_fwd(const Matrix&, Matrix&) { return false; }
bool gelu_bwd(const Matrix&, const Matrix&, Matrix&) { return false; }
bool softmax_fwd(const Matrix&, Matrix&) { return false; }
bool softmax_bwd(const Matrix&, const Matrix&, Matrix&) { return false; }
bool attn_masked_softmax(Matrix&, float, size_t, bool) { return false; }
bool swa_masked_softmax(Matrix&, float, size_t, size_t, size_t) {
    return false;
}
bool attn_softmax_bwd_inplace(Matrix&, const Matrix&, float) {
    return false;
}
bool embed_gather(const Matrix&, const int*, size_t, Matrix&) {
    return false;
}
bool ce_fwd(const Matrix&, const int*, Matrix&, float&) { return false; }
bool ce_bwd(const Matrix&, const int*, float, Matrix&) { return false; }
bool adamw_step(Matrix&, const Matrix&, Matrix&, Matrix&, float, float,
                float, float, float, float, float) {
    return false;
}
bool sgd_step(Matrix&, const Matrix&, Matrix*, float, float) {
    return false;
}
float* opt_state_new(size_t) { return nullptr; }
void opt_state_free(float*) {}
bool adamw_step_dev(Matrix&, const Matrix&, float*, float*, float, float,
                    float, float, float, float, float) {
    return false;
}
bool sgd_step_dev(Matrix&, const Matrix&, float*, float, float) {
    return false;
}
bool layernorm_fwd(const Matrix&, const Matrix&, const Matrix&, float,
                   Matrix&, Matrix&, std::vector<float>&) { return false; }
bool layernorm_bwd(const Matrix&, const Matrix&, const std::vector<float>&,
                   const Matrix&, bool, Matrix*, Matrix*, bool, Matrix*) {
    return false;
}
bool rmsnorm_fwd(const Matrix&, const Matrix&, float, Matrix&,
                 std::vector<float>&) { return false; }
bool rmsnorm_bwd(const Matrix&, const Matrix&, const std::vector<float>&,
                 const Matrix&, bool, Matrix*, bool, Matrix*) { return false; }
}  // namespace devops
}  // namespace device
}  // namespace microtorch

#endif  // !MICROTORCH_CUDA
