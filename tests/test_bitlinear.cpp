// BitNet b1.58 quantisation-aware training (LOCAL_LLM_WORK R3; ops::ternary_ste,
// ops::act_quant_ste, nn::Linear::BitLinear, LlamaConfig::bitlinear).
//
// Legs:
//   1. the fake quantisers' forwards against the formulas written out here
//      (BitNet b1.58's reference code), bit for bit on the CPU;
//   2. their straight-through backwards: the gradient passes unchanged;
//   3. the BitLinear layer: its forward is the composition, its weight
//      gradient is h_q^T dY, its input gradient is the gradient of the
//      smooth surrogate sum(G .* (rmsnorm(x, 1) W_q)), whose finite
//      differences are checked (with act_bits 0 the layer itself is smooth
//      in x, so it is FD-checked directly); the true derivative of a
//      rounding is zero almost everywhere, so FD on the quantisers
//      themselves would check nothing;
//   4. a BitLinear llama: only the block projections are quantised, the
//      cached decode gives the full forward's logits, the state_dict is the
//      float model's, and a few AdamW steps lower the loss;
//   5. CUDA (MICROTORCH_DEVICE=cuda, a CUDA build): the device forwards
//      against the CPU's, and a BitLinear layer's forward and gradients with
//      the device op set off and on.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/llama.hpp"
#include "microtorch/nn.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;
namespace device = microtorch::device;

namespace {

int g_failures = 0;

void check(bool ok, const char* label, double measured) {
    std::printf("  [%s] %-60s %.3e\n", ok ? "ok" : "FAIL", label, measured);
    if (!ok) ++g_failures;
}

Matrix randn(size_t r, size_t c, unsigned seed, float scale = 1.0f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, scale);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(gen);
    return m;
}

double max_abs(const Matrix& a, const Matrix& b) {
    double w = 0.0;
    for (size_t i = 0; i < a.rows() * a.cols(); ++i)
        w = std::max(w, static_cast<double>(std::fabs(a.get_data()[i] - b.get_data()[i])));
    return w;
}

// The reference formulas, written out independently of ops.cpp.
Matrix ref_ternary(const Matrix& w, float* scale = nullptr) {
    double acc = 0.0;
    for (size_t i = 0; i < w.rows() * w.cols(); ++i) acc += std::fabs(static_cast<double>(w.get_data()[i]));
    const float mean = static_cast<float>(acc / static_cast<double>(w.rows() * w.cols()));
    const float s = 1.0f / std::max(mean, 1e-5f);
    if (scale) *scale = s;
    Matrix y(w.rows(), w.cols());
    for (size_t i = 0; i < w.rows() * w.cols(); ++i) {
        float q = std::nearbyint(w.get_data()[i] * s);
        q = q > 1.0f ? 1.0f : (q < -1.0f ? -1.0f : q);
        y.get_data()[i] = q / s;
    }
    return y;
}

Matrix ref_actq(const Matrix& x, int bits) {
    const float qmax = static_cast<float>((1 << (bits - 1)) - 1), qmin = -qmax - 1.0f;
    Matrix y(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        float m = 0.0f;
        for (size_t j = 0; j < x.cols(); ++j) m = std::max(m, std::fabs(x(i, j)));
        const float s = qmax / std::max(m, 1e-5f);
        for (size_t j = 0; j < x.cols(); ++j) {
            float q = std::nearbyint(x(i, j) * s);
            q = q > qmax ? qmax : (q < qmin ? qmin : q);
            y(i, j) = q / s;
        }
    }
    return y;
}

Matrix ref_rmsnorm1(const Matrix& x) {
    Matrix y(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        float ss = 0.0f;
        for (size_t j = 0; j < x.cols(); ++j) ss += x(i, j) * x(i, j);
        ss /= static_cast<float>(x.cols());
        const float r = 1.0f / std::sqrt(ss + ops::kRmsNormEps);
        for (size_t j = 0; j < x.cols(); ++j) y(i, j) = x(i, j) * r * 1.0f;
    }
    return y;
}

// sum(y .* G) on the tape: mean, then times the count (dY = G to rounding).
Var wsum(const Var& y, const Matrix& g) {
    return ops::scale(ops::mean(ops::mul(y, make_var(g))), static_cast<float>(g.rows() * g.cols()));
}

float weighted_sum(const Var& y, const Matrix& g) {
    double s = 0.0;
    for (size_t i = 0; i < g.rows() * g.cols(); ++i) s += static_cast<double>(y->data.get_data()[i]) * g.get_data()[i];
    return static_cast<float>(s);
}

double fd_vs_analytic(const std::function<float()>& f, Var leaf, const Matrix& analytic, float h = 1e-2f) {
    double worst = 0.0;
    for (size_t i = 0; i < leaf->data.rows(); ++i)
        for (size_t j = 0; j < leaf->data.cols(); ++j) {
            const float keep = leaf->data(i, j);
            NoGrad ng;
            leaf->data(i, j) = keep + h;
            const double up = f();
            leaf->data(i, j) = keep - h;
            const double dn = f();
            leaf->data(i, j) = keep;
            const double fd = (up - dn) / (2.0 * h);
            const double a = analytic(i, j);
            worst = std::max(worst, std::abs(a - fd) / (1.0 + std::max(std::abs(a), std::abs(fd))));
        }
    return worst;
}

void leg1_forwards() {
    std::printf("-- leg 1: forwards against the reference formulas --\n");
    const Matrix w = randn(37, 53, 1, 0.05f);
    float s = 0.0f;
    const Matrix want = ref_ternary(w, &s);
    const Var y = ops::ternary_ste(make_var(w));
    check(max_abs(y->data, want) == 0.0, "ternary_ste == clamp(round(W s), -1, 1) / s", max_abs(y->data, want));
    check(ops::ternary_scale(w) == s, "ternary_scale == 1 / mean|W|", std::fabs(ops::ternary_scale(w) - s));
    size_t bad = 0, zeros = 0;
    for (size_t i = 0; i < w.rows() * w.cols(); ++i) {
        const float q = y->data.get_data()[i] * s;
        if (q != -1.0f && q != 0.0f && q != 1.0f) ++bad;
        zeros += q == 0.0f;
    }
    check(bad == 0, "every value is a code in {-1, 0, 1} over s", static_cast<double>(bad));
    // Gaussian weights: |w| < mean|w| / 2 holds for about 31% of them.
    const double zf = static_cast<double>(zeros) / static_cast<double>(w.rows() * w.cols());
    check(zf > 0.2 && zf < 0.45, "zero share near a Gaussian's 31%", zf);
    const Matrix z(4, 6);
    const Var yz = ops::ternary_ste(make_var(z));
    check(max_abs(yz->data, z) == 0.0, "an all-zero tensor stays zero (the 1e-5 floor)", max_abs(yz->data, z));

    for (int bits : {8, 4}) {
        const Matrix x = randn(19, 41, 2 + bits, 2.0f);
        const Matrix wantx = ref_actq(x, bits);
        const Var yx = ops::act_quant_ste(make_var(x), bits);
        char lab[96];
        std::snprintf(lab, sizeof lab, "act_quant_ste(%d bits) == per-row absmax rounding", bits);
        check(max_abs(yx->data, wantx) == 0.0, lab, max_abs(yx->data, wantx));
        // The row's largest |x| lands on the grid's end exactly.
        double worst = 0.0;
        const float qmax = static_cast<float>((1 << (bits - 1)) - 1);
        for (size_t i = 0; i < x.rows(); ++i) {
            size_t a = 0;
            for (size_t j = 1; j < x.cols(); ++j)
                if (std::fabs(x(i, j)) > std::fabs(x(i, a))) a = j;
            float m = std::fabs(x(i, a));
            const float sc = qmax / m;
            worst = std::max(worst, static_cast<double>(std::fabs(std::fabs(yx->data(i, a) * sc) - qmax)));
        }
        std::snprintf(lab, sizeof lab, "act_quant_ste(%d bits): each row's absmax maps to +-%d", bits,
                      static_cast<int>(qmax));
        check(worst < 1e-3, lab, worst);
    }
}

void leg2_ste() {
    std::printf("-- leg 2: straight-through backwards --\n");
    const Matrix w = randn(11, 7, 3, 0.1f), g = randn(11, 7, 4);
    Var wv = make_var(w, true);
    Var y = ops::ternary_ste(wv);
    backward(wsum(y, g));
    check(max_abs(wv->grad, y->grad) == 0.0, "ternary_ste: dW == dY (bit for bit)", max_abs(wv->grad, y->grad));
    Var xv = make_var(randn(5, 9, 5), true);
    const Matrix gx = randn(5, 9, 6);
    Var yx = ops::act_quant_ste(xv, 8);
    backward(wsum(yx, gx));
    check(max_abs(xv->grad, yx->grad) == 0.0, "act_quant_ste: dX == dY (bit for bit)", max_abs(xv->grad, yx->grad));
}

void leg3_layer() {
    std::printf("-- leg 3: the BitLinear layer --\n");
    const size_t N = 6, IN = 24, OUT = 10;
    nn::Linear lin(IN, OUT, false, 9);
    lin.W->data = randn(IN, OUT, 10, 0.2f);
    nn::Linear::BitLinear q;
    q.on = true;
    lin.set_bitlinear(q);
    const Matrix x0 = randn(N, IN, 11, 1.5f), G = randn(N, OUT, 12);
    Var x = make_var(x0, true);
    Var y = lin.forward(x);
    const Matrix hq = ref_actq(ref_rmsnorm1(x0), 8), wq = ref_ternary(lin.W->data);
    const Matrix want = hq * wq;
    check(max_abs(y->data, want) < 1e-5, "forward == actq(rmsnorm(x, 1)) ternary(W)", max_abs(y->data, want));
    backward(wsum(y, G));
    const Matrix dw = hq.transpose() * G;
    check(max_abs(lin.W->grad, dw) < 1e-5, "dW == h_q^T dY (straight through the ternary)", max_abs(lin.W->grad, dw));
    // dx: the surrogate rmsnorm(x, 1) W_q with W_q held fixed, the activation
    // quantiser passed straight through.
    Var xs = make_var(x0, true);
    Var ones = make_var(Matrix(1, IN, 1.0f));
    Var wqv = make_var(wq);
    auto sur = [&]() { return weighted_sum(ops::matmul(ops::rmsnorm(xs, ones), wqv), G); };
    {
        Var ys = ops::matmul(ops::rmsnorm(xs, ones), wqv);
        backward(wsum(ys, G));
    }
    check(max_abs(x->grad, xs->grad) < 1e-6, "dx == the surrogate's gradient", max_abs(x->grad, xs->grad));
    const double fd = fd_vs_analytic(sur, xs, xs->grad);
    check(fd < 2e-3, "the surrogate's gradient vs finite differences", fd);

    // act_bits 0: the layer is smooth in x (W_q does not depend on x): FD on it directly.
    nn::Linear lin0(IN, OUT, false, 9);
    lin0.W->data = lin.W->data;
    q.act_bits = 0;
    lin0.set_bitlinear(q);
    Var x2 = make_var(x0, true);
    backward(wsum(lin0.forward(x2), G));
    const double fd0 = fd_vs_analytic([&]() { return weighted_sum(lin0.forward(x2), G); }, x2, x2->grad);
    check(fd0 < 2e-3, "act_bits 0: dx vs finite differences on the layer", fd0);
    // in_norm off, act_bits 0: y = x ternary(W), so dx = dY ternary(W)^T exactly.
    nn::Linear lin1(IN, OUT, false, 9);
    lin1.W->data = lin.W->data;
    q.in_norm = false;
    lin1.set_bitlinear(q);
    Var x3 = make_var(x0, true);
    backward(wsum(lin1.forward(x3), G));
    const Matrix dx3 = G * wq.transpose();
    check(max_abs(x3->grad, dx3) < 1e-5, "no norm, no act quant: dx == dY W_q^T", max_abs(x3->grad, dx3));
    // A float layer with the same weights is untouched by the option's existence.
    nn::Linear f(IN, OUT, false, 9);
    f.W->data = lin.W->data;
    const Var yf = f.forward(make_var(x0));
    const Var yref = ops::matmul(make_var(x0), make_var(lin.W->data));
    check(max_abs(yf->data, yref->data) == 0.0, "off: y == ops::matmul(x, W) bit for bit", max_abs(yf->data, yref->data));
}

nn::LlamaConfig tiny(bool bit) {
    nn::LlamaConfig c;
    c.vocab = 61;
    c.d = 32;
    c.n_layers = 2;
    c.n_heads = 4;
    c.d_ff = 48;
    c.n_ctx = 24;
    c.bitlinear.on = bit;
    return c;
}

void leg4_llama() {
    std::printf("-- leg 4: a BitLinear llama --\n");
    nn::Llama fp(tiny(false), 5), bit(tiny(true), 5);
    const auto sf = fp.state_dict(), sb = bit.state_dict();
    bool same = sf.size() == sb.size();
    for (const auto& [k, v] : sf) same = same && sb.count(k) && max_abs(v, sb.at(k)) == 0.0;
    check(same, "state_dict: the float model's keys and initial values", same ? 0.0 : 1.0);
    size_t on = 0;
    for (const auto& b : bit.blocks)
        for (const nn::Linear* p : {b->q_proj.get(), b->k_proj.get(), b->v_proj.get(), b->o_proj.get(),
                                    b->gate_proj.get(), b->up_proj.get(), b->down_proj.get()})
            on += p->bitlinear().on;
    check(on == 14, "every block projection is a BitLinear (7 x 2 layers)", static_cast<double>(on));
    std::mt19937 rng(1);
    std::vector<int> ids(20);
    for (int& t : ids) t = static_cast<int>(rng() % 61);
    Matrix full, ffp;
    {
        NoGrad ng;
        full = bit.forward(ids)->data;
        ffp = fp.forward(ids)->data;
    }
    check(max_abs(full, ffp) > 1e-3, "the quantised forward differs from the float one", max_abs(full, ffp));
    // Cached decode: prefill 4, then one token at a time.
    nn::LlamaKV kv;
    Matrix cached(ids.size(), 61);
    {
        NoGrad ng;
        const Matrix p = bit.forward_cached(std::vector<int>(ids.begin(), ids.begin() + 4), kv)->data;
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 61; ++j) cached(i, j) = p(i, j);
        for (size_t t = 4; t < ids.size(); ++t) {
            const Matrix r = bit.forward_cached({ids[t]}, kv)->data;
            for (size_t j = 0; j < 61; ++j) cached(t, j) = r(0, j);
        }
    }
    float lo = 1e30f, hi = -1e30f;
    for (size_t i = 0; i < full.rows() * full.cols(); ++i) {
        lo = std::min(lo, full.get_data()[i]);
        hi = std::max(hi, full.get_data()[i]);
    }
    const double rel = max_abs(full, cached) / (hi - lo);
    check(rel < 5e-3, "cached decode == full forward (over the logit range)", rel);
    // Training: AdamW on a repeating sequence lowers the loss.
    nn::Llama m(tiny(true), 6);
    nn::AdamW opt(m.parameters(), 3e-3f);
    std::vector<int> seq(24), tgt(23);
    for (size_t i = 0; i < seq.size(); ++i) seq[i] = static_cast<int>((i * 7) % 13);
    std::vector<int> in(seq.begin(), seq.end() - 1);
    for (size_t i = 0; i < tgt.size(); ++i) tgt[i] = seq[i + 1];
    float first = 0.0f, last = 0.0f;
    const Matrix w0 = m.blocks[0]->q_proj->W->data;
    for (int step = 0; step < 60; ++step) {
        opt.zero_grad();
        Var loss = ops::cross_entropy(m.forward(in), tgt);
        backward(loss);
        opt.step();
        if (step == 0) first = loss->data(0, 0);
        last = loss->data(0, 0);
    }
    std::printf("     loss %.4f -> %.4f\n", first, last);
    check(last < 0.5f * first, "60 AdamW steps halve the loss", last / first);
    check(max_abs(m.blocks[0]->q_proj->W->data, w0) > 0.0, "the latent weights train", max_abs(m.blocks[0]->q_proj->W->data, w0));
}

void leg5_cuda() {
    std::printf("-- leg 5: CUDA --\n");
    if (!device::cuda_compiled() || device::get() != device::Device::CUDA) {
        std::printf("  SKIP: a CUDA build with MICROTORCH_DEVICE=cuda runs this leg\n");
        return;
    }
    device::set_device_ops(true);
    const std::vector<std::pair<size_t, size_t>> shapes = {{67, 133}, {320, 864}, {864, 320}, {1, 5}};
    for (const auto& [r, c] : shapes) {
        const Matrix w = randn(r, c, static_cast<unsigned>(r + c), 0.03f);
        Matrix yd(r, c);
        const bool ran = device::devops::ternary_fwd(w, yd);
        device::materialize(yd);
        const Matrix want = ref_ternary(w);
        char lab[96];
        std::snprintf(lab, sizeof lab, "device ternary_fwd == CPU, %zux%zu", r, c);
        check(ran && max_abs(yd, want) == 0.0, lab, ran ? max_abs(yd, want) : -1.0);
        const Matrix x = randn(r, c, static_cast<unsigned>(r * 3 + c), 3.0f);
        Matrix xd(r, c);
        const bool ran2 = device::devops::act_quant_fwd(x, 8, xd);
        device::materialize(xd);
        const Matrix wantx = ref_actq(x, 8);
        std::snprintf(lab, sizeof lab, "device act_quant_fwd == CPU, %zux%zu", r, c);
        check(ran2 && max_abs(xd, wantx) == 0.0, lab, ran2 ? max_abs(xd, wantx) : -1.0);
    }
    // A BitLinear layer's forward and gradients with the op set off and on.
    const size_t N = 33, IN = 320, OUT = 864;
    const Matrix x0 = randn(N, IN, 21), w0 = randn(IN, OUT, 22, 0.05f), G = randn(N, OUT, 23);
    Matrix y[2], dx[2], dw[2];
    for (int k = 0; k < 2; ++k) {
        device::set_device_ops(k == 1);
        nn::Linear lin(IN, OUT, false, 1);
        lin.W->data = w0;
        nn::Linear::BitLinear q;
        q.on = true;
        lin.set_bitlinear(q);
        Var x = make_var(x0, true);
        Var out = lin.forward(x);
        backward(wsum(out, G));
        device::materialize(out->data);
        device::materialize(x->grad);
        device::materialize(lin.W->grad);
        y[k] = out->data;
        dx[k] = x->grad;
        dw[k] = lin.W->grad;
    }
    device::set_device_ops(true);
    check(max_abs(y[0], y[1]) < 1e-3, "BitLinear forward, ops off vs on", max_abs(y[0], y[1]));
    check(max_abs(dx[0], dx[1]) < 1e-3, "BitLinear dx, ops off vs on", max_abs(dx[0], dx[1]));
    check(max_abs(dw[0], dw[1]) < 1e-3, "BitLinear dW, ops off vs on", max_abs(dw[0], dw[1]));
}

}  // namespace

int main() {
    device::set_from_env();
    std::printf("test_bitlinear (BitNet b1.58 fake quantisers, straight-through estimators)\n");
    // Legs 1-4 are the CPU reference (bit-for-bit claims); leg 5 runs on the device when one is set.
    const device::Device dev = device::get();
    device::set(device::Device::CPU);
    leg1_forwards();
    leg2_ste();
    leg3_layer();
    leg4_llama();
    device::set(dev);
    leg5_cuda();
    if (g_failures) {
        std::printf("test_bitlinear FAILED (%d)\n", g_failures);
        return 1;
    }
    std::printf("test_bitlinear PASSED\n");
    return 0;
}
