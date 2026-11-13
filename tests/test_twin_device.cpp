// The cue-selective twin loss (include/microtorch/posttrain/twin.hpp; lane amendment A31) on the device against the host: the loss
// and its gradient with respect to the logits, element by element, with the device op set on (CUDA build, MICROTORCH_DEVICE=cuda
// MICROTORCH_DEVICE_OPS=1), write-through and inside a deferred step window. Claude Opus 5.5, 13 Nov 2026, after the A31 launch
// reviews (the gate compared gradient norms, not the twin gradient itself). The twin term's own backward is this gradient: past
// the logits it flows through the model's standard backward, which every SFT step on the device already runs. Run by the gate
// (tools/colab_chat/a31_gate.py) on the pod, which requires "device op set ON" in its output; on a CPU build it compares the host
// path with itself.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/posttrain/twin.hpp"

using namespace microtorch;
namespace pt = microtorch::posttrain;

namespace {

bool on_device() { return device::device_ops_enabled() && device::get() == device::Device::CUDA; }

template <class F>
void with_ops(bool dev, bool windowed, F fn) {
    device::set_device_ops(dev);
    if (windowed) {
        device::set_step_residency(true);
        device::set_defer_downloads(true);
        device::step_begin();
    }
    fn();
    if (windowed) {
        device::step_end();
        device::set_defer_downloads(false);
        device::set_step_residency(false);
    }
    device::set_device_ops(true);
}

// A batch shaped like the trainer's: G groups of 5 rows of L positions, a vocabulary of C, replies of 3 to 18 targets at the end of
// rows of different real lengths (a reply has the same targets in each context it is read in).
pt::TwinTargets make_targets(size_t G, size_t L, size_t C, unsigned seed) {
    pt::TwinTargets t;
    t.groups = G;
    t.per_group = 5;
    t.rows = G * 5;
    std::mt19937 g(seed);
    for (size_t gi = 0; gi < G; ++gi) {
        std::vector<int> a(3 + g() % 16), b(3 + g() % 16);
        for (auto& x : a) x = static_cast<int>(g() % C);
        for (auto& x : b) x = static_cast<int>(g() % C);
        t.n_a.push_back(static_cast<float>(a.size()));
        t.n_b.push_back(static_cast<float>(b.size()));
        for (size_t j = 0; j < 5; ++j) {
            const auto& y = (j == 0 || j == 1 || j == 4) ? a : b;
            const size_t r = gi * 5 + j, end = L - (g() % (L / 4));
            for (size_t i = 0; i < y.size(); ++i) {
                t.pos.push_back(static_cast<int>(r * L + end - y.size() + i));
                t.tgt.push_back(y[i]);
                t.pos_row.push_back(static_cast<int>(r));
            }
        }
    }
    return t;
}

Matrix randn(size_t r, size_t c, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, 2.0f);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(g);
    return m;
}

void leg(bool windowed) {
    const size_t G = 4, L = 96, C = 2048;
    const pt::TwinTargets t = make_targets(G, L, C, 7);
    const Matrix X = randn(t.rows * L, C, 11);
    float loss[2];
    Matrix grad[2];
    pt::TwinStats st[2];
    for (int dev = 0; dev < 2; ++dev) {
        Var x = make_var(X, true);
        with_ops(dev == 1, windowed, [&] {
            Var l = pt::twin_loss(x, t, 1.0f, 0.7f, 0.4f, &st[dev]);
            backward(l);
            device::materialize(l->data);
            loss[dev] = l->data(0, 0);
        });
        device::materialize(x->grad);
        grad[dev] = x->grad;
    }
    double scale = 0.0, worst = 0.0;
    size_t nonzero = 0;
    for (size_t i = 0; i < grad[0].rows(); ++i)
        for (size_t j = 0; j < grad[0].cols(); ++j) scale = std::max(scale, (double)std::abs(grad[0](i, j)));
    for (size_t i = 0; i < grad[0].rows(); ++i)
        for (size_t j = 0; j < grad[0].cols(); ++j) {
            worst = std::max(worst, std::abs((double)grad[1](i, j) - grad[0](i, j)) / scale);
            nonzero += grad[0](i, j) != 0.0f;
        }
    const double lrel = std::abs((double)loss[1] - loss[0]) / std::max(1e-12, std::abs((double)loss[0]));
    std::printf("  %s: loss host %.9g device %.9g (relative %.2e); logits gradient worst relative %.2e over %zu nonzero entries;"
                " d_a %.6f/%.6f d_n %.6f/%.6f\n",
                windowed ? "deferred window" : "write-through", loss[0], loss[1], lrel, worst, nonzero, st[0].d_a, st[1].d_a,
                st[0].d_n, st[1].d_n);
    CHECK(std::isfinite(loss[0]) && std::isfinite(loss[1]) && loss[0] > 0.0f);
    CHECK(nonzero > 0);
    CHECK(lrel < 1e-5);
    CHECK(worst < 1e-5);
    CHECK(std::abs(st[0].d_a - st[1].d_a) < 1e-4 && std::abs(st[0].d_n - st[1].d_n) < 1e-4);
}

}  // namespace

int main() {
    device::set_from_env();
    std::printf("twin loss: device vs host (%s)\n", on_device() ? "device op set ON" : "host vs host: CPU build or op set off");
    leg(false);
    leg(true);
    std::printf("test_twin_device: all checks passed\n");
    return 0;
}
