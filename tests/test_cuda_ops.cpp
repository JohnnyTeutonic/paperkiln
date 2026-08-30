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
//   3. DEFERRED DOWNLOADS (B2.1b) — staleness-contract unit checks plus
//      the same composed tape under deferral vs plain ops-ON, at the
//      leg-2 bounds.
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
#include "microtorch/nn.hpp"
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

TapeRun run_tape(unsigned seed, bool windowed = false) {
    using mt::Var;
    Var x = mt::make_var(filled(R, C, seed), true);
    Var gamma = mt::make_var(filled(1, C, seed + 1), true);
    Var beta = mt::make_var(filled(1, C, seed + 2), true);
    Var w = mt::make_var(filled(1, C, seed + 3), true);

    if (windowed) device::step_begin();
    Var h = mt::ops::gelu(x);
    h = mt::ops::layernorm(h, gamma, beta, 1e-5f);
    h = mt::ops::softmax_row(h);
    h = mt::ops::rmsnorm(h, w);
    h = mt::ops::sigmoid(h);
    Var loss = mt::ops::mean(h);
    mt::backward(loss);
    if (windowed) device::step_end();  // B2.1b materialize boundary

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

// B2.1b: deferred downloads. Unit contract first (stale inside the
// window, materialized at step_end, chained op consumes the stale
// value on-device), then the full composed tape under deferral vs
// plain ops-ON, at the leg-2 tolerances.
void leg3_deferred() {
    std::printf("-- leg 3: deferred downloads (B2.1b) --\n");
    device::set_device_ops(true);
    device::set_step_residency(true);
    device::set_defer_downloads(true);

    const Matrix a = filled(R, C, 7);
    device::step_begin();
    Matrix y(R, C), y2(R, C);
    devops::gelu_fwd(a, y);
    check(device::host_stale(y), "output stale inside window",
          device::host_stale(y) ? 1.0 : 0.0);
    devops::sigmoid_fwd(y, y2);  // consumes the stale value on-device
    device::step_end();
    check(!device::host_stale(y) && !device::host_stale(y2),
          "materialized at step_end", 0.0);

    device::set_defer_downloads(false);
    device::step_begin();
    Matrix ry(R, C), ry2(R, C);
    devops::gelu_fwd(a, ry);
    devops::sigmoid_fwd(ry, ry2);
    device::step_end();
    check(max_abs_diff(y2, ry2) <= EW_TOL,
          "deferred chain == write-through", max_abs_diff(y2, ry2));

    TapeRun on = run_tape(43);
    device::set_defer_downloads(true);
    TapeRun def = run_tape(43, /*windowed=*/true);
    device::set_defer_downloads(false);
    device::set_step_residency(false);

    const double dloss = std::fabs(static_cast<double>(on.loss) - def.loss);
    check(dloss <= 1e-5, "loss (defer vs write-through)", dloss);
    check(max_abs_diff(on.gx, def.gx) <= 1e-4, "grad x",
          max_abs_diff(on.gx, def.gx));
    check(max_abs_diff(on.ggamma, def.ggamma) <= 1e-4, "grad gamma",
          max_abs_diff(on.ggamma, def.ggamma));
    check(max_abs_diff(on.gbeta, def.gbeta) <= 1e-4, "grad beta",
          max_abs_diff(on.gbeta, def.gbeta));
    check(max_abs_diff(on.gw, def.gw) <= 1e-4, "grad w",
          max_abs_diff(on.gw, def.gw));
}

// B2.2: masked attention softmax on-device — fused (causal and block)
// and swa, forward + backward, through the REAL ops so the devops-OFF
// run is the host-loop reference. Then the same contrast under deferral
// (the [T,T] scores/weights never crossing the bus inside the step).
struct AttnRun {
    float loss;
    Matrix gq, gk, gv;
};

AttnRun run_attn(unsigned seed, int flavor, bool windowed) {
    using mt::Var;
    const size_t T = 32, D = 16, SL = 16;
    Var q = mt::make_var(filled(T, D, seed), true);
    Var k = mt::make_var(filled(T, D, seed + 1), true);
    Var v = mt::make_var(filled(T, D, seed + 2), true);
    if (windowed) device::step_begin();
    Var y;
    switch (flavor) {
        case 0:  // fused causal
            y = mt::ops::fused_attention(q, k, v, 0.25f, SL, true);
            break;
        case 1:  // fused block (non-causal)
            y = mt::ops::fused_attention(q, k, v, 0.25f, SL, false);
            break;
        default:  // swa, window < SL, sinks on
            y = mt::ops::swa_attention(q, k, v, 0.25f, /*window=*/5,
                                       /*sinks=*/2, SL);
    }
    Var loss = mt::ops::mean(y);
    mt::backward(loss);
    if (windowed) device::step_end();
    return AttnRun{loss->data(0, 0), q->grad, k->grad, v->grad};
}

void leg4_attention() {
    std::printf("-- leg 4: masked attention softmax (B2.2) --\n");
    const char* names[] = {"fused-causal", "fused-block", "swa"};
    for (int f = 0; f < 3; ++f) {
        device::set_device_ops(false);
        AttnRun off = run_attn(44 + f, f, false);
        device::set_device_ops(true);
        AttnRun on = run_attn(44 + f, f, false);
        device::set_step_residency(true);
        device::set_defer_downloads(true);
        AttnRun def = run_attn(44 + f, f, /*windowed=*/true);
        device::set_defer_downloads(false);
        device::set_step_residency(false);
        device::set_device_ops(false);

        const double dl =
            std::fabs(static_cast<double>(off.loss) - on.loss);
        check(dl <= 1e-5, names[f], dl);
        check(max_abs_diff(off.gq, on.gq) <= 1e-4, "grad q",
              max_abs_diff(off.gq, on.gq));
        check(max_abs_diff(off.gk, on.gk) <= 1e-4, "grad k",
              max_abs_diff(off.gk, on.gk));
        check(max_abs_diff(off.gv, on.gv) <= 1e-4, "grad v",
              max_abs_diff(off.gv, on.gv));
        const double dld =
            std::fabs(static_cast<double>(off.loss) - def.loss);
        check(dld <= 1e-5, "defer loss", dld);
        check(max_abs_diff(off.gq, def.gq) <= 1e-4, "defer grad q",
              max_abs_diff(off.gq, def.gq));
        check(max_abs_diff(off.gk, def.gk) <= 1e-4, "defer grad k",
              max_abs_diff(off.gk, def.gk));
        check(max_abs_diff(off.gv, def.gv) <= 1e-4, "defer grad v",
              max_abs_diff(off.gv, def.gv));
    }
    device::set_device_ops(true);
}

// B2.2: embedding gather + cross-entropy on-device. Composed exactly as
// a model uses them (CE consuming the gathered rows as logits), so the
// devops-OFF run is the host reference for forward loss AND the
// scatter-add table gradient; then again under deferral, where the
// logits/P never cross the bus and the host receives one float.
struct EmbCeRun {
    float loss;
    Matrix gt;
};

EmbCeRun run_embed_ce(unsigned seed, bool windowed) {
    using mt::Var;
    const size_t V = 32, C = 20, N = 24;
    Var table = mt::make_var(filled(V, C, seed), true);
    std::vector<int> ids(N), targets(N);
    for (size_t i = 0; i < N; ++i) {
        ids[i] = static_cast<int>((seed + 3 * i) % V);
        targets[i] = static_cast<int>((seed + 5 * i) % C);
    }
    if (windowed) device::step_begin();
    Var logits = mt::ops::embedding(table, ids);
    Var loss = mt::ops::cross_entropy(logits, targets);
    mt::backward(loss);
    if (windowed) device::step_end();
    return EmbCeRun{loss->data(0, 0), table->grad};
}

void leg5_embed_ce() {
    std::printf("-- leg 5: embedding + cross-entropy (B2.2) --\n");
    device::set_device_ops(false);
    EmbCeRun off = run_embed_ce(46, false);
    device::set_device_ops(true);
    EmbCeRun on = run_embed_ce(46, false);
    device::set_step_residency(true);
    device::set_defer_downloads(true);
    EmbCeRun def = run_embed_ce(46, /*windowed=*/true);
    device::set_defer_downloads(false);
    device::set_step_residency(false);

    const double dl = std::fabs(static_cast<double>(off.loss) - on.loss);
    check(dl <= 1e-5, "ce loss", dl);
    check(max_abs_diff(off.gt, on.gt) <= 1e-4, "table grad",
          max_abs_diff(off.gt, on.gt));
    const double dld = std::fabs(static_cast<double>(off.loss) - def.loss);
    check(dld <= 1e-5, "defer ce loss", dld);
    check(max_abs_diff(off.gt, def.gt) <= 1e-4, "defer table grad",
          max_abs_diff(off.gt, def.gt));
}

// B2.3a: optimizer steps on device (write-through parity seam). Drive
// the REAL optimizers over multi-step trajectories with fresh grads per
// step — state (m/v/vel) must track bit-for-tolerance across steps, not
// just one update.
void leg6_optimizers() {
    std::printf("-- leg 6: optimizer steps (B2.3a) --\n");
    using mt::Var;
    const size_t R6 = 13, C6 = 17, STEPS = 5;

    auto drive = [&](bool adam) {
        Var p = mt::make_var(filled(R6, C6, 90), true);
        std::vector<mt::Var> ps{p};
        mt::nn::AdamW aw(ps, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);
        mt::nn::SGD sgd(ps, 0.01f, 0.9f);
        for (size_t s = 0; s < STEPS; ++s) {
            p->grad = filled(R6, C6, static_cast<unsigned>(91 + s));
            if (adam)
                aw.step();
            else
                sgd.step();
        }
        return p->data;
    };

    for (int adam = 0; adam < 2; ++adam) {
        device::set_device_ops(false);
        Matrix off = drive(adam == 1);
        device::set_device_ops(true);
        Matrix on = drive(adam == 1);
        device::set_device_ops(false);
        const char* name = adam ? "adamw 5-step trajectory"
                                : "sgd+momentum 5-step trajectory";
        check(max_abs_diff(off, on) <= 1e-5, name, max_abs_diff(off, on));
    }
    device::set_device_ops(true);
}

// REGRESSION LEG (B2.3 flat-loss bug, 30 Aug 2026). A gemm that reads a
// DEFERRED activation took window_operand's slot path, which consulted
// the value cache only for SLOTLESS operands and so uploaded the
// untouched (zero) host buffer. Real models produced exactly-zero logits
// and loss exactly ln(vocab); the whole 285-check suite passed anyway,
// because legs 2-3's tapes contain NO matmul after a device op and every
// unit leg materializes between stages. THE SHAPE THAT CATCHES IT:
// device op -> matmul -> loss with NO intervening materialize, plus the
// embedding -> matmul -> CE chain a real forward actually runs.
struct GemmRun {
    float loss;
    Matrix gw, gx;
};

GemmRun run_deferred_gemm(unsigned seed, bool defer, bool via_embedding) {
    using mt::Var;
    const size_t V = 40, D = 24, N = 18;
    Var w = mt::make_var(filled(D, D, seed), true);
    Var gam = mt::make_var(filled(1, D, seed + 1), true);
    Var bet = mt::make_var(filled(1, D, seed + 2), true);
    Var tbl = mt::make_var(filled(V, D, seed + 3), true);
    Var x = mt::make_var(filled(N, D, seed + 4), true);
    std::vector<int> ids(N), tgt(N);
    for (size_t i = 0; i < N; ++i) {
        ids[i] = static_cast<int>((seed + 3 * i) % V);
        tgt[i] = static_cast<int>((seed + 7 * i) % D);
    }

    device::set_defer_downloads(defer);
    device::step_begin();
    // The deferred producer: its output's host copy stays untouched.
    Var h = via_embedding ? mt::ops::embedding(tbl, ids)
                          : mt::ops::layernorm(x, gam, bet, 1e-5f);
    Var y = mt::ops::matmul(h, w);  // slotted gemm consumes it — the seam
    Var loss = via_embedding ? mt::ops::cross_entropy(y, tgt)
                             : mt::ops::mean(y);
    mt::backward(loss);
    device::step_end();
    device::set_defer_downloads(false);
    return GemmRun{loss->data(0, 0), w->grad,
                   via_embedding ? tbl->grad : x->grad};
}

void leg7_deferred_gemm() {
    std::printf("-- leg 7: gemm reading a deferred activation (B2.3 "
                "regression) --\n");
    device::set_device_ops(true);
    device::set_step_residency(true);
    for (int emb = 0; emb < 2; ++emb) {
        GemmRun wt = run_deferred_gemm(70 + emb, /*defer=*/false, emb == 1);
        GemmRun df = run_deferred_gemm(70 + emb, /*defer=*/true, emb == 1);
        const char* what = emb ? "embedding->matmul->CE"
                               : "layernorm->matmul->mean";
        const double dl = std::fabs(static_cast<double>(wt.loss) - df.loss);
        check(dl <= 1e-5, what, dl);
        check(max_abs_diff(wt.gw, df.gw) <= 1e-4, "grad W",
              max_abs_diff(wt.gw, df.gw));
        check(max_abs_diff(wt.gx, df.gx) <= 1e-4, "grad input",
              max_abs_diff(wt.gx, df.gx));
    }
    device::set_step_residency(false);
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
    leg3_deferred();
    leg4_attention();
    leg5_embed_ce();
    leg6_optimizers();
    leg7_deferred_gemm();

    if (g_failures == 0) {
        std::printf("test_cuda_ops PASSED (B2.1a kernels + tape parity + "
                    "B2.1b deferred downloads + B2.2 attention/embed/CE)\n");
        return 0;
    }
    std::printf("test_cuda_ops: %d FAILURE(S)\n", g_failures);
    return 1;
}
