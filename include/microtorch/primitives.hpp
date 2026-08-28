#pragma once
// Canonical transformer_core entry points for microtorch.
//
// docs/history/PHASE0_KERNEL_AUDIT.md section 3 found the same op defined in multiple
// translation units with DIVERGING contracts (gelu_backward three times,
// softmax three times). This header is therefore a whitelist, not a
// re-export: exactly one entry point per op family, and nothing else from
// transformer_cpp is part of microtorch's contract. If an op is not named
// here, microtorch does not call it.
//
//   Matrix              row-major float32; (rows, cols) ctor zero-fills;
//                       operator()(i,j), data(), transpose(), hadamard()
//   matmul_optimized    blocked AVX2 CPU matmul (matmul_optimized.cpp) --
//                       the phase-1a workhorse
//   Matrix::apply_gelu  tanh-approximation GELU forward (components.cpp)
//   Matrix::apply_softmax  row-wise, max-subtracted, in place
//
// DELIBERATELY NOT USED: Matrix::apply_gelu_derivative. Its formula is
// wrong -- it evaluates tanh at x instead of at the inner argument
// u = sqrt(2/pi)*(x + 0.044715 x^3) and drops the 0.5 sech^2 factor
// (components.cpp:220-224), ~13% error at x=1. The CUDA kernel
// (src/cuda/matrix_ops.cu gelu_backward_kernel) has the CORRECT formula,
// so the CPU and GPU training paths of transformer_cpp currently disagree;
// microtorch::ops implements the correct derivative itself and the
// gradcheck test measures both against finite differences.
#include "matmul_optimized.hpp"
#include "matrix.hpp"
