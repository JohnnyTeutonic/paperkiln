#pragma once
// The CPU/CUDA dispatch seam (docs/DESIGN.md section 8.2).
//
// Every matmul in the op set routes through device::matmul. Default build:
// the blocked-AVX2 CPU path, bit-identical to before this header existed.
// -DMICROTORCH_CUDA=ON: dispatch to transformer_core's CudaMatrix::matmul
// (cuda_kernels.hpp) when the runtime device is CUDA.
//
// Phase A is correctness-first: tensors live on the host and each CUDA
// matmul round-trips host->device->host. Resident device memory (upload
// parameters once, keep activations on-device) is the phase-B item and
// belongs on Colab hardware, not local ([[no-local-gpu]] discipline: the
// local machine never runs CUDA).
#include "microtorch/primitives.hpp"

namespace microtorch {
namespace device {

enum class Device { CPU, CUDA };

// Process-wide default device. CPU unless set otherwise; setting CUDA on a
// build without MICROTORCH_CUDA throws rather than silently running CPU.
Device get();
void set(Device d);
bool cuda_compiled();  // true iff built with -DMICROTORCH_CUDA=ON

// Honor MICROTORCH_DEVICE=cuda|cpu if present (no-op otherwise). Test mains
// call this so the Colab CUDA validation reruns the same suites on GPU.
void set_from_env();

// The one matmul entry point for the op set.
Matrix matmul(const Matrix& a, const Matrix& b);

}  // namespace device
}  // namespace microtorch
