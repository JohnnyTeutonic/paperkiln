// Public scalar helper: host-coherent controls plus an explicitly selected
// synthetic CPU boundary or real CUDA producer. CPU emulation is not a GPU gate.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
namespace {
const std::vector<float> values{-.25f, .5f, 1.f, -1.5f, .75f, -.125f};
const float* pending = nullptr;
int materializations = 0;
Matrix fixture() {
    Matrix m(2, 3);
    std::copy(values.begin(), values.end(), m.get_data());
    return m;
}
float reference(const Matrix& m) {
    double sum = 0;
    for (size_t i = 0; i < m.rows() * m.cols(); ++i)
        sum += std::abs(static_cast<double>(m.get_data()[i]));
    const float mean = m.rows() * m.cols() ? static_cast<float>(sum / (m.rows() * m.cols())) : 0;
    return 1.f / std::max(mean, 1e-5f);
}
bool same_bits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}
}  // namespace
#ifdef TERNARY_BOUNDARY_WRAP
extern "C" void __real__ZN10microtorch6device11materializeERK6Matrix(const Matrix&);
extern "C" void __wrap__ZN10microtorch6device11materializeERK6Matrix(const Matrix& m) {
    if (m.get_data() == pending) {
        std::copy(values.begin(), values.end(), const_cast<float*>(m.get_data()));
        ++materializations;
        pending = nullptr;
    }
    __real__ZN10microtorch6device11materializeERK6Matrix(m);
}
#endif
int main(int argc, char** argv) {
    const bool gpu = argc == 2 && std::string(argv[1]) == "--cuda";
    CHECK(argc == 1 || gpu);
    auto m = fixture();
    const float expected = 16.f / 11.f;  // Independent exact-rational fixture mean: 11/16.
    CHECK(same_bits(reference(m), expected));
    CHECK(same_bits(ops::ternary_scale(m), expected));
    Matrix zero(2, 3);
    CHECK(same_bits(ops::ternary_scale(zero), reference(zero)));
    if (gpu) {
        device::set(device::Device::CUDA);
        device::set_device_ops(true);
        device::set_step_residency(true);
        device::set_defer_downloads(true);
        device::step_begin();
        CHECK(device::deferring());
        {
            auto x = make_var(fixture());
            auto y = ops::slice_cols(x, 0, 3);
            CHECK(device::host_stale(y->data));
            // Only overwrite non-authoritative host bytes; the GPU value stays intact.
            std::fill(y->data.get_data(), y->data.get_data() + 6, 0.f);
            CHECK(device::host_stale(y->data));
            const float actual = ops::ternary_scale(y->data);
            std::printf("actual_cuda=true expected=%.9g actual=%.9g\n", expected, actual);
            CHECK(same_bits(actual, expected));
            CHECK(!device::host_stale(y->data));
            for (size_t i = 0; i < 6; ++i) CHECK(y->data.get_data()[i] == values[i]);
        }
        device::step_end();
        device::set_defer_downloads(false);
        device::set_step_residency(false);
        device::set_device_ops(false);
        device::evict_all();
        device::set(device::Device::CPU);
    }
#ifdef TERNARY_BOUNDARY_WRAP
    else {
        pending = m.get_data();
        std::fill(m.get_data(), m.get_data() + 6, 0.f);
        const float actual = ops::ternary_scale(m);
        std::printf("synthetic_cpu_boundary=true expected=%.9g actual=%.9g materializations=%d\n",
                    expected, actual, materializations);
        CHECK(same_bits(actual, expected));
        CHECK(materializations == 1);
    }
#endif
    std::puts("ternary_scale boundary controls PASS");
}
