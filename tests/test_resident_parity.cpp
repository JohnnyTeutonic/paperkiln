// Phase B1 gate (docs/CUDA_PHASE_B.md): the residency path must agree
// with the CPU reference and the Phase A round-trip path, including the
// mutate -> invalidate -> re-resident leg. Dims are deliberately not
// tile multiples so edge tiles are exercised.
//
// CPU-only builds: prints SKIP and exits 0 (the API is a no-op there).
#include <cmath>
#include <cstdio>
#include <random>

#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"

namespace device = microtorch::device;

namespace {

int g_failures = 0;

void check(bool ok, const char* label, double measured) {
    std::printf("  [%s] %-52s %.3e\n", ok ? "ok" : "FAIL", label, measured);
    if (!ok) ++g_failures;
}

Matrix filled(size_t r, size_t c, unsigned seed) {
    Matrix m(r, c);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (size_t i = 0; i < r * c; ++i) m.get_data()[i] = u(rng);
    return m;
}

double max_abs_diff(const Matrix& x, const Matrix& y) {
    double worst = 0.0;
    for (size_t i = 0; i < x.rows() * x.cols(); ++i)
        worst = std::max(worst,
                         static_cast<double>(std::fabs(x.get_data()[i] -
                                                       y.get_data()[i])));
    return worst;
}

}  // namespace

int main() {
    std::printf("test_resident_parity\n");
    if (!device::cuda_compiled()) {
        std::printf("  SKIP: built without MICROTORCH_CUDA\n");
        return 0;
    }
    device::set_from_env();  // honour MICROTORCH_DEVICE like the other suites

    // Non-tile-multiple dims: edge tiles do real work.
    Matrix A = filled(37, 53, 1);
    Matrix B = filled(53, 29, 2);

    device::set(device::Device::CPU);
    Matrix c_ref = device::matmul(A, B);

    device::set(device::Device::CUDA);
    Matrix c_phase_a = device::matmul(A, B);
    check(max_abs_diff(c_ref, c_phase_a) <= 1e-4,
          "phase A (round-trip) vs CPU reference",
          max_abs_diff(c_ref, c_phase_a));

    device::set_residency(true);
    device::make_resident(A);
    device::make_resident(B);
    check(device::resident_count() == 2, "resident_count after 2 uploads",
          static_cast<double>(device::resident_count()));

    Matrix c_b1 = device::matmul(A, B);
    check(max_abs_diff(c_ref, c_b1) <= 1e-4,
          "B1 (both operands resident) vs CPU reference",
          max_abs_diff(c_ref, c_b1));

    // Mixed: one resident, one temp-uploaded.
    device::invalidate(B);
    Matrix c_mixed = device::matmul(A, B);
    check(max_abs_diff(c_ref, c_mixed) <= 1e-4,
          "B1 (A resident, B temp) vs CPU reference",
          max_abs_diff(c_ref, c_mixed));

    // The contract leg: mutate host data, invalidate, re-resident, and the
    // result must match a FRESH CPU reference of the mutated operand.
    A.get_data()[0] += 0.5f;
    A.get_data()[37 * 53 - 1] -= 0.25f;
    device::invalidate(A);
    device::make_resident(A);
    device::make_resident(B);
    device::set(device::Device::CPU);
    Matrix c_ref2 = device::matmul(A, B);
    device::set(device::Device::CUDA);
    Matrix c_after = device::matmul(A, B);
    check(max_abs_diff(c_ref2, c_after) <= 1e-4,
          "mutate -> invalidate -> re-resident matches new ref",
          max_abs_diff(c_ref2, c_after));

    device::evict_all();
    check(device::resident_count() == 0, "evict_all empties the table",
          static_cast<double>(device::resident_count()));
    device::set_residency(false);

    if (g_failures) {
        std::printf("FAILURES: %d\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
