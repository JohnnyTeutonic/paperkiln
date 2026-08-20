// Phase B2.1a gate (docs/CUDA_PHASE_B2.md section 4): the device-side op
// set must agree with the CPU tape formulas in ops.cpp — which stay in
// place as both the fallback and the reference.
//
// Two legs:
//   1. KERNEL PARITY — each devops entry vs the CPU formula, same inputs.
//      Elementwise/activations are per-element (same fp ops, same order):
//      bound 1e-6. Rowwise ops reduce in parallel (order differs from the
//      CPU's serial loop): bound 1e-5. Bitwise claims are reserved for
//      same-backend pins (test_step_residency's rule).
//   2. TAPE PARITY — a composed graph (gelu -> layernorm -> softmax ->
//      rmsnorm -> sigmoid -> mean) run twice on identical inputs, device
//      ops OFF then ON: outputs and every leaf grad within 1e-4.
//
// Dims are deliberately not multiples of the 256-thread row block so the
// strided column loops and edge cases are exercised.
//
// CPU-only builds / MICROTORCH_DEVICE != cuda: prints SKIP and exits 0.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/ops.hpp"

namespace mt = microtorch;
namespace device = microtorch::device;
namespace devops = microtorch::device::devops;

namespace {

int g_failures = 0;

void check(bool ok, const char* label, double measured) {
    std::printf("  [%s] %-52s %.3e\n", ok ? "ok" : "FAIL", label, measured);
    if (!ok) ++g_failures;
}

Matrix filled(size_t r, size_t c, unsigned seed) {
    Matrix m(r, c);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.5f, 1.5f);
    for (size_t i = 0; i < r * c; ++i) m.get_data()[i] = u(rng);
    return m;
}

double max_abs_diff(const Matrix& x, const Matrix& y) {
    double worst = 0.0;
    for (size_t i = 0; i < x.rows() * x.cols(); ++i)
        worst = std::max(worst,
                         static_cast<double>(std::fabs(
                             x.get_data()[i] - y.get_data()[i])));
    return worst;
}

constexpr double EW_TOL = 1e-6;   // per-element: same ops, same order
constexpr double ROW_TOL = 1e-5;  // parallel reduction order differs

// Deliberately awkward shapes: rows/cols not tile or block multiples.
constexpr size_t R = 67, C = 133;

void leg1_kernels() {
    std::printf("-- leg 1: kernel parity vs the CPU formulas --\n");
    const Matrix a = filled(R, C, 1), b = filled(R, C, 2);

    {   // elementwise family
        Matrix dev(R, C), cpu(R, C);
        if (devops::add(a, b, dev)) {
            for (size_t i = 0; i < R * C; ++i)
                cpu.get_data()[i] = a.get_data()[i] + b.get_data()[i];
            check(max_abs_diff(dev, cpu) <= EW_TOL, "add", max_abs_diff(dev, cpu));
        }
        if (devops::sub(a, b, dev)) {
            for (size_t i = 0; i < R * C; ++i)
                cpu.get_data()[i] = a.get_data()[i] - b.get_data()[i];
            check(max_abs_diff(dev, cpu) <= EW_TOL, "sub", max_abs_diff(dev, cpu));
        }
        if (devops::mul(a, b, dev)) {
            for (size_t i = 0; i < R * C; ++i)
                cpu.get_data()[i] = a.get_data()[i] * b.get_data()[i];
            check(max_abs_diff(dev, cpu) <= EW_TOL, "mul", max_abs_diff(dev, cpu));
        }
        if (devops::scale(a, 0.37f, dev)) {
            for (size_t i = 0; i < R * C; ++i)
                cpu.get_data()[i] = a.get_data()[i] * 0.37f;
            check(max_abs_diff(dev, cpu) <= EW_TOL, "scale", max_abs_diff(dev, cpu));
        }
        Matrix ydev = filled(R, C, 3), ycpu = ydev;
        if (devops::axpy(ydev, 0.61f, a)) {
            for (size_t i = 0; i < R * C; ++i)
                ycpu.get_data()[i] += 0.61f * a.get_data()[i];
            check(max_abs_diff(ydev, ycpu) <= EW_TOL, "axpy", max_abs_diff(ydev, ycpu));
        }
        if (devops::fill(dev, -2.5f)) {
            for (size_t i = 0; i < R * C; ++i) cpu.get_data()[i] = -2.5f;
            check(max_abs_diff(dev, cpu) <= EW_TOL, "fill", max_abs_diff(dev, cpu));
        }
    }

    {   // sigmoid fwd/bwd
        Matrix dev(R, C), cpu(R, C);
        devops::sigmoid_fwd(a, dev);
        for (size_t i = 0; i < R * C; ++i)
            cpu.get_data()[i] = 1.0f / (1.0f + std::exp(-a.get_data()[i]));
        check(max_abs_diff(dev, cpu) <= EW_TOL, "sigmoid_fwd", max_abs_diff(dev, cpu));

        Matrix dxd(R, C), dxc(R, C);
        devops::sigmoid_bwd(cpu, b, dxd);
        for (size_t i = 0; i < R * C; ++i) {
            const float s = cpu.get_data()[i];
            dxc.get_data()[i] = b.get_data()[i] * s * (1.0f - s);
        }
        check(max_abs_diff(dxd, dxc) <= EW_TOL, "sigmoid_bwd", max_abs_diff(dxd, dxc));
    }

    {   // gelu fwd/bwd — the CORRECT derivative, per primitives.hpp
        Matrix dev(R, C), cpu = a;
        devops::gelu_fwd(a, dev);
        cpu.apply_gelu();  // the vendored forward is the CPU reference
        check(max_abs_diff(dev, cpu) <= EW_TOL, "gelu_fwd (vs apply_gelu)",
              max_abs_diff(dev, cpu));

        constexpr float k = 0.7978845608028654f;
        Matrix dxd(R, C), dxc(R, C);
        devops::gelu_bwd(a, b, dxd);
        for (size_t i = 0; i < R * C; ++i) {
            const float v = a.get_data()[i];
            const float u = k * (v + 0.044715f * v * v * v);
            const float t = std::tanh(u);
            const float cdf = 0.5f * (1.0f + t);
            const float pdf =
                0.5f * (1.0f - t * t) * k * (1.0f + 0.134145f * v * v);
            dxc.get_data()[i] = b.get_data()[i] * (cdf + v * pdf);
        }
        check(max_abs_diff(dxd, dxc) <= EW_TOL, "gelu_bwd", max_abs_diff(dxd, dxc));
    }

    {   // softmax fwd/bwd
        Matrix dev(R, C), cpu = a;
        devops::softmax_fwd(a, dev);
        cpu.apply_softmax();
        check(max_abs_diff(dev, cpu) <= ROW_TOL, "softmax_fwd (vs apply_softmax)",
              max_abs_diff(dev, cpu));

        Matrix dXd(R, C), dXc(R, C);
        devops::softmax_bwd(cpu, b, dXd);
        for (size_t i = 0; i < R; ++i) {
            float dot = 0.0f;
            for (size_t j = 0; j < C; ++j) dot += b(i, j) * cpu(i, j);
            for (size_t j = 0; j < C; ++j)
                dXc(i, j) = cpu(i, j) * (b(i, j) - dot);
        }
        check(max_abs_diff(dXd, dXc) <= ROW_TOL, "softmax_bwd", max_abs_diff(dXd, dXc));
    }

    {   // layernorm fwd/bwd
        const Matrix g = filled(1, C, 4), be = filled(1, C, 5);
        const float eps = 1e-5f;
        Matrix yd(R, C), xhd(R, C), yc(R, C), xhc(R, C);
        std::vector<float> rsd(R), rsc(R);
        devops::layernorm_fwd(a, g, be, eps, yd, xhd, rsd);
        for (size_t i = 0; i < R; ++i) {
            float mu = 0.0f;
            for (size_t j = 0; j < C; ++j) mu += a(i, j);
            mu /= static_cast<float>(C);
            float var = 0.0f;
            for (size_t j = 0; j < C; ++j) {
                const float d = a(i, j) - mu;
                var += d * d;
            }
            var /= static_cast<float>(C);
            rsc[i] = 1.0f / std::sqrt(var + eps);
            for (size_t j = 0; j < C; ++j) {
                xhc(i, j) = (a(i, j) - mu) * rsc[i];
                yc(i, j) = g(0, j) * xhc(i, j) + be(0, j);
            }
        }
        check(max_abs_diff(yd, yc) <= ROW_TOL, "layernorm_fwd y", max_abs_diff(yd, yc));
        check(max_abs_diff(xhd, xhc) <= ROW_TOL, "layernorm_fwd xhat",
              max_abs_diff(xhd, xhc));

        Matrix dgd(1, C), dbd(1, C), dxd(R, C);
        devops::layernorm_bwd(b, xhc, rsc, g, true, &dgd, &dbd, true, &dxd);
        Matrix dgc(1, C), dbc(1, C), dxc(R, C);
        for (size_t i = 0; i < R; ++i)
            for (size_t j = 0; j < C; ++j) {
                dgc(0, j) += b(i, j) * xhc(i, j);
                dbc(0, j) += b(i, j);
            }
        for (size_t i = 0; i < R; ++i) {
            float m1 = 0.0f, m2 = 0.0f;
            for (size_t j = 0; j < C; ++j) {
                const float dxh = b(i, j) * g(0, j);
                m1 += dxh;
                m2 += dxh * xhc(i, j);
            }
            m1 /= static_cast<float>(C);
            m2 /= static_cast<float>(C);
            for (size_t j = 0; j < C; ++j) {
                const float dxh = b(i, j) * g(0, j);
                dxc(i, j) = rsc[i] * (dxh - m1 - xhc(i, j) * m2);
            }
        }
        check(max_abs_diff(dgd, dgc) <= ROW_TOL, "layernorm_bwd dgamma",
              max_abs_diff(dgd, dgc));
        check(max_abs_diff(dbd, dbc) <= ROW_TOL, "layernorm_bwd dbeta",
              max_abs_diff(dbd, dbc));
        check(max_abs_diff(dxd, dxc) <= ROW_TOL, "layernorm_bwd dx",
              max_abs_diff(dxd, dxc));
    }

    {   // rmsnorm fwd/bwd
        const Matrix w = filled(1, C, 6);
        const float eps = 1e-5f;
        Matrix yd(R, C), yc(R, C);
        std::vector<float> rid(R), ric(R);
        devops::rmsnorm_fwd(a, w, eps, yd, rid);
        for (size_t i = 0; i < R; ++i) {
            float ss = 0.0f;
            for (size_t j = 0; j < C; ++j) ss += a(i, j) * a(i, j);
            ss /= static_cast<float>(C);
            ric[i] = 1.0f / std::sqrt(ss + eps);
            for (size_t j = 0; j < C; ++j) yc(i, j) = a(i, j) * ric[i] * w(0, j);
        }
        check(max_abs_diff(yd, yc) <= ROW_TOL, "rmsnorm_fwd", max_abs_diff(yd, yc));

        Matrix dwd(1, C), dxd(R, C);
        devops::rmsnorm_bwd(b, a, ric, w, true, &dwd, true, &dxd);
        Matrix dwc(1, C), dxc(R, C);
        for (size_t i = 0; i < R; ++i)
            for (size_t j = 0; j < C; ++j)
                dwc(0, j) += b(i, j) * a(i, j) * ric[i];
        for (size_t i = 0; i < R; ++i) {
            float term = 0.0f;
            for (size_t j = 0; j < C; ++j) term += b(i, j) * w(0, j) * a(i, j);
            const float ri2 = ric[i] * ric[i];
            const float n_inv = 1.0f / static_cast<float>(C);
            for (size_t j = 0; j < C; ++j)
                dxc(i, j) = ric[i] * (b(i, j) * w(0, j) - a(i, j) * ri2 * term * n_inv);
        }
        check(max_abs_diff(dwd, dwc) <= ROW_TOL, "rmsnorm_bwd dw", max_abs_diff(dwd, dwc));
        check(max_abs_diff(dxd, dxc) <= ROW_TOL, "rmsnorm_bwd dx", max_abs_diff(dxd, dxc));
    }
}

// One forward+backward of a graph that exercises every wired op through
// the ACTUAL ops.cpp call sites. Returns output value + all leaf grads.
struct TapeRun {
    float loss;
    Matrix gx, ggamma, gbeta, gw;
};

TapeRun run_tape(unsigned seed) {
    using mt::Var;
    Var x = mt::make_var(filled(R, C, seed), true);
    Var gamma = mt::make_var(filled(1, C, seed + 1), true);
    Var beta = mt::make_var(filled(1, C, seed + 2), true);
    Var w = mt::make_var(filled(1, C, seed + 3), true);

    Var h = mt::ops::gelu(x);
    h = mt::ops::layernorm(h, gamma, beta, 1e-5f);
    h = mt::ops::softmax_row(h);
    h = mt::ops::rmsnorm(h, w);
    h = mt::ops::sigmoid(h);
    Var loss = mt::ops::mean(h);
    mt::backward(loss);

    return TapeRun{loss->data(0, 0), x->grad, gamma->grad, beta->grad,
                   w->grad};
}

void leg2_tape() {
    std::printf("-- leg 2: composed tape, device ops OFF vs ON --\n");
    device::set_device_ops(false);
    TapeRun off = run_tape(42);
    device::set_device_ops(true);
    TapeRun on = run_tape(42);
    device::set_device_ops(false);

    const double dloss =
        std::fabs(static_cast<double>(off.loss) - on.loss);
    check(dloss <= 1e-5, "loss", dloss);
    check(max_abs_diff(off.gx, on.gx) <= 1e-4, "grad x",
          max_abs_diff(off.gx, on.gx));
    check(max_abs_diff(off.ggamma, on.ggamma) <= 1e-4, "grad gamma",
          max_abs_diff(off.ggamma, on.ggamma));
    check(max_abs_diff(off.gbeta, on.gbeta) <= 1e-4, "grad beta",
          max_abs_diff(off.gbeta, on.gbeta));
    check(max_abs_diff(off.gw, on.gw) <= 1e-4, "grad w",
          max_abs_diff(off.gw, on.gw));
}

}  // namespace

int main() {
    device::set_from_env();
    if (!device::cuda_compiled() || device::get() != device::Device::CUDA) {
        std::printf("SKIP test_cuda_ops: CUDA build + MICROTORCH_DEVICE=cuda "
                    "required (CPU-only build or device is cpu)\n");
        return 0;
    }
    device::set_device_ops(true);

    leg1_kernels();
    leg2_tape();

    if (g_failures == 0) {
        std::printf("test_cuda_ops PASSED (B2.1a op set: kernel + tape parity)\n");
        return 0;
    }
    std::printf("test_cuda_ops: %d FAILURE(S)\n", g_failures);
    return 1;
}
