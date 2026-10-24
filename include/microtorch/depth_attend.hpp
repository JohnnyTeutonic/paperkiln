#pragma once
// Attention over depth as ONE tape node (Attention Residuals, Kimi arXiv
// 2603.15031; the IR op depth_attend, include/microtorch/ir.hpp).
//
// nn::AttnResStack::depth_attend composes the same function from tape ops
// (rmsnorm, mul_row, matmul, softmax_row, mul_col, add): about five nodes
// and three [T, d] activations per source, so a sublayer-level stack of L
// llama blocks keeps ~3 (2L + 1)^2 / 2 activations of [T, d] alive until
// backward (1,600 at L = 16 against ~300 for the plain model). This op
// keeps three [T, S] tables (weights, inverse RMS, logits) and nothing
// else, has a device kernel pair (src/cuda_depth_attend.cu), and is what
// nn::Llama's attention-residual stream (LlamaConfig::AttnRes) runs.
//
// Forward, each row t on its own, S = sources.size() >= 2 (S = 1 returns
// the source itself, recording nothing, as the IR's S = 1 copy):
//   r_i  = 1 / sqrt(ss_i / d + eps), ss_i = sum_j s_i[t, j]^2 (ascending j)
//   l_i  = sum_j (s_i[t, j] * r_i) * w[j]                      (ascending j)
//   a    = softmax_i(l): e_i = exp(l_i - max l), z = e_0 + e_1 + ..., a_i = e_i / z
//   out  = s_0 a_0 + s_1 a_1 + ...  per element, ascending i, no fused multiply-add
//   out  = out * scale              when scale != 1 (the product rounded): the
//                                   IR's depth_attend followed by affine(scale, 0)
// which is the IR's normative order (the host loop is bit-reproducible;
// the device kernel reduces ss_i and l_i across a warp, ROUNDING).
// Backward (dY [T, d]; read as scale * dY when scale != 1):
//   g_i   = sum_j dY[t, j] s_i[t, j];  gbar = sum_i a_i g_i;  dl_i = a_i (g_i - gbar)
//   ds_i  = a_i dY + dl_i r_i (w - (l_i r_i / d) s_i)
//   dw   += sum_t sum_i dl_i r_i s_i[t, :]
// Finite-difference checked against the composed form in
// tests/test_llama_attnres.cpp.
#include <vector>

#include "microtorch/autograd.hpp"
#include "microtorch/ops.hpp"

namespace microtorch {
namespace ops {

// sources: S tensors [T, d] (S >= 1); w: the pseudo-query [1, d]; scale
// multiplies the mixture (nn::LlamaConfig::AttnRes scale "sources": S).
Var depth_attend(const std::vector<Var>& sources, const Var& w, float eps = kRmsNormEps, float scale = 1.0f);

}  // namespace ops

namespace device {
namespace devops {
// The device seam (false = the host loop runs). At most kDepthAttendMaxSources
// sources per call (the kernels take the pointers as a launch parameter).
constexpr size_t kDepthAttendMaxSources = 128;
// out [T, d]; alpha, rinv, logit [T, S] (written for backward).
bool depth_attend_fwd(const std::vector<const Matrix*>& src, const Matrix& w, float eps, float scale, Matrix& out,
                      Matrix& alpha, Matrix& rinv, Matrix& logit);
// ds[i] (nullptr: source i needs no gradient) [T, d] written whole; dw
// (nullptr: no gradient) [1, d] written whole.
bool depth_attend_bwd(const std::vector<const Matrix*>& src, const Matrix& w, float scale, const Matrix& alpha,
                      const Matrix& rinv, const Matrix& logit, const Matrix& dy, const std::vector<Matrix*>& ds,
                      Matrix* dw);
}  // namespace devops
}  // namespace device
}  // namespace microtorch
