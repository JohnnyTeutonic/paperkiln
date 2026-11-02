
// Causal CPU authority emulation, not a CUDA execution result.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "microtorch/device.hpp"
#include "microtorch/nn.hpp"
using namespace microtorch;
namespace {
std::map<const float*, std::vector<float>> pending;
void poison(Matrix& m) {
    auto p = m.get_data();
    pending[p] = std::vector<float>(p, p + 4);
    std::fill(p, p + 4, 19.f);
}
bool run(int kind, int mode) {
    pending.clear();
    Matrix m(2, 2);
    const std::array<float, 4> init{.25f, -.5f, .75f, -1.f};
    std::copy(init.begin(), init.end(), m.get_data());
    auto p = make_var(std::move(m), true);
    nn::SGD sgd({p}, .05f, kind == 1 ? .9f : 0.f);
    nn::AdamW adam({p}, .05f, .9f, .99f, 1e-6f, .1f);
    std::array<double, 4> truth{}, moment{}, v{};
    std::copy(init.begin(), init.end(), truth.begin());
    bool correct = true;
    for (int step = 0; step < 2; ++step) {
        const std::array<float, 4> g = step == 0 ? std::array<float, 4>{.125f, -.25f, .375f, -.5f}
                                                 : std::array<float, 4>{-.2f, .1f, -.3f, .4f};
        p->grad = Matrix(2, 2);
        std::copy(g.begin(), g.end(), p->grad.get_data());
        if (mode & 1) poison(p->data);
        if (mode & 2) poison(p->grad);
        if (kind == 2)
            adam.step();
        else
            sgd.step();
        for (int i = 0; i < 4; ++i) {
            if (kind == 2) {
                moment[i] = .9 * moment[i] + .1 * double(g[i]);
                v[i] = .99 * v[i] + .01 * double(g[i]) * g[i];
                truth[i] -= .05 * ((moment[i] / (1 - std::pow(.9, step + 1))) /
                                       (std::sqrt(v[i] / (1 - std::pow(.99, step + 1))) + 1e-6) +
                                   .1 * truth[i]);
            } else {
                moment[i] = (kind == 1 ? .9 : 0) * moment[i] + double(g[i]);
                truth[i] -= .05 * moment[i];
            }
            correct = correct && std::isfinite(p->data.get_data()[i]) &&
                      std::abs(p->data.get_data()[i] - truth[i]) < 3e-6;
        }
        // Do not let synthetic authority survive destruction or change later inputs.
        pending.clear();
    }
    return correct;
}
}  // namespace
extern "C" void __real__ZN10microtorch6device11materializeERK6Matrix(const Matrix&);
extern "C" void __wrap__ZN10microtorch6device11materializeERK6Matrix(const Matrix& m) {
    auto i = pending.find(m.get_data());
    if (i != pending.end()) {
        std::copy(i->second.begin(), i->second.end(), const_cast<float*>(m.get_data()));
        pending.erase(i);
    }
    __real__ZN10microtorch6device11materializeERK6Matrix(m);
}
int main(int argc, char** argv) {
    bool deferred = argc == 2 && std::string(argv[1]) == "--deferred";
    int failures = 0, cases = 0, unexpected = 0;
    device::set(device::Device::CPU);
    device::set_device_ops(false);
    for (int kind = 0; kind < 3; ++kind)
        for (int mode = deferred ? 1 : 0; mode <= (deferred ? 3 : 0); ++mode) {
            bool okay = run(kind, mode);
            ++cases;
            if (!okay) ++failures;
#ifdef OPTIMIZER_HOST_BASELINE
            bool expected = !(deferred && (kind != 2 || (mode & 1)));
#else
            bool expected = true;
#endif
            if (okay != expected) ++unexpected;
            std::cout << "{\"optimizer\":" << kind << ",\"authority_mode\":" << mode
                      << ",\"numerical_match\":" << (okay ? "true" : "false") << "}\n";
        }
    std::cout << "{\"cases\":" << cases << ",\"numerical_failures\":" << failures
              << ",\"unexpected\":" << unexpected << ",\"cpu_emulation_only\":true}\n";
    return unexpected ? 1 : 0;
}
