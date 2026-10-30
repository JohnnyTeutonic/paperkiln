// The third-party ops paperkiln trains (LOCAL_LLM_STACK.md section 11 item 4.4; keramos IR_SPEC thirdparty
// profile): grouped-query attention (ops::repeat_heads + the fused attention), NEOX and partial RoPE
// (ops::apply_rope's pairing and n_rot), per-head RMSNorm (ops::rmsnorm_group), Mamba-1's conv1d_causal and
// selective_scan. Agreement with the numpy reference's own expected values is the conformance suite's
// single-op fixtures through tools/ir_eval (tests/test_thirdparty_ir.py); here:
//   1. each op against a float64 re-statement of its IR_SPEC formula, and the bit pins: apply_rope's
//      defaults are the original op, rmsnorm_group over the whole row is ops::rmsnorm, and the RoPE and
//      repeat results equal a float re-statement of the host loop bit for bit (on CUDA too: the device
//      kernels round as the host does);
//   2. every input's gradient against central finite differences;
//   3. the llama block with the options: the options at their default values are the plain block bit for
//      bit; a grouped-query model equals the multi-head model whose k/v projections repeat its columns,
//      bit for bit; KV-cached decode equals the full forward; a stacked batch equals its sequences; finite
//      differences through k_proj, the q/k norms and the loss;
//   4. Mamba-1 (parity::MambaLM, ssm mamba1): a stacked batch equals its sequences bit for bit, finite
//      differences through A_log, the conv and dt_proj.
// Run with MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 for the device legs (sections 1-3 then run the
// device kernels of rope, repeat_heads and rmsnorm_group).
#include <cmath>
#include <limits>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "../tools/parity_model.hpp"
#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/llama.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

namespace {

Matrix randn(size_t r, size_t c, unsigned seed, float scale = 1.0f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, scale);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(gen);
    return m;
}

Matrix host(const Var& v) {
    device::materialize(v->data);
    return v->data;
}

double max_abs(const Matrix& a, const std::vector<double>& b) {
    double w = 0;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j) w = std::max(w, std::fabs(a(i, j) - b[i * a.cols() + j]));
    return w;
}

bool bits_equal(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j)
            if (!(a(i, j) == b(i, j))) return false;
    return true;
}

// loss = sum(f() .* R), R fixed; worst |analytic - fd| / (1 + max(|a|, |fd|)) over every entry of leaf.
double gradcheck(const std::function<Var()>& f, const Var& leaf, unsigned seed, float h = 1e-2f) {
    Var y0;
    {
        NoGrad ng;
        y0 = f();
    }
    const Matrix R = randn(y0->data.rows(), y0->data.cols(), seed);
    const Var r = make_var(R);
    auto loss = [&]() {
        Var y = f();
        return ops::scale(ops::mean(ops::mul(y, r)), static_cast<float>(y->data.rows() * y->data.cols()));
    };
    leaf->grad = Matrix();
    backward(loss());
    const Matrix a = host(leaf->grad.rows() ? make_var(leaf->grad) : make_var(Matrix(leaf->data.rows(), leaf->data.cols())));
    double worst = 0;
    for (size_t i = 0; i < leaf->data.rows(); ++i)
        for (size_t j = 0; j < leaf->data.cols(); ++j) {
            const float keep = leaf->data(i, j);
            NoGrad ng;
            leaf->data(i, j) = keep + h;
            const double up = host(loss())(0, 0);
            leaf->data(i, j) = keep - h;
            const double dn = host(loss())(0, 0);
            leaf->data(i, j) = keep;
            const double fd = (up - dn) / (2.0 * h);
            worst = std::max(worst, std::fabs(a(i, j) - fd) / (1.0 + std::max(static_cast<double>(std::fabs(a(i, j))), std::fabs(fd))));
        }
    return worst;
}

// IR_SPEC 5.14 in float64 (the angle as the table states it, in f32, then cos/sin in double) over one
// [T, heads * hd] tensor.
std::vector<double> rope_ref(const Matrix& x, const std::vector<int>& pos, size_t heads, size_t hd, size_t n_rot,
                             bool neox, float theta) {
    std::vector<double> y(x.rows() * x.cols());
    for (size_t i = 0; i < x.rows(); ++i)
        for (size_t j = 0; j < x.cols(); ++j) y[i * x.cols() + j] = x(i, j);
    for (size_t i = 0; i < x.rows(); ++i)
        for (size_t p = 0; p < n_rot / 2; ++p) {
            const float f = 1.0f / std::pow(theta, static_cast<float>(2 * p) / static_cast<float>(n_rot));
            const double ang = static_cast<float>(pos[i]) * f;
            for (size_t h = 0; h < heads; ++h) {
                const size_t c0 = h * hd + (neox ? p : 2 * p), c1 = neox ? c0 + n_rot / 2 : c0 + 1;
                const double a = x(i, c0), b = x(i, c1);
                y[i * x.cols() + c0] = a * std::cos(ang) - b * std::sin(ang);
                y[i * x.cols() + c1] = a * std::sin(ang) + b * std::cos(ang);
            }
        }
    return y;
}

// The host loop of ops::apply_rope, re-stated in float (the bit pin).
Matrix rope_float(const Matrix& x, const std::vector<int>& pos, size_t heads, size_t hd, size_t n_rot, bool neox,
                  float theta) {
    Matrix m = x;
    for (size_t i = 0; i < x.rows(); ++i) {
        const float mp = static_cast<float>(pos[i]);
        for (size_t dim = 0; dim < n_rot; dim += 2) {
            const float inv = 1.0f / std::pow(theta, static_cast<float>(dim) / n_rot);
            const float t = mp * inv, c = std::cos(t), s = std::sin(t);
            for (size_t h = 0; h < heads; ++h) {
                const size_t c0 = h * hd + (neox ? dim / 2 : dim), c1 = neox ? c0 + n_rot / 2 : c0 + 1;
                const float a = m(i, c0), b = m(i, c1);
                m(i, c0) = a * c - b * s;
                m(i, c1) = a * s + b * c;
            }
        }
    }
    return m;
}

// q part of apply_rope on the fused layout [x | x | x].
Var rope_of(const Var& x, const std::vector<int>& pos, size_t hd, ops::RopePairing pr, size_t n_rot) {
    return ops::slice_cols(ops::apply_rope(ops::concat_cols({x, x, x}), pos, 10000.0f, hd, true, pr, n_rot), 0,
                           x->data.cols());
}

std::vector<int> ids_of(size_t n, size_t vocab, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<int> v(n);
    for (auto& x : v) x = static_cast<int>(g() % vocab);
    return v;
}

double logits_diff(const Matrix& a, size_t ra, const Matrix& b, size_t rb, size_t n) {
    double w = 0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < a.cols(); ++j) w = std::max(w, static_cast<double>(std::fabs(a(ra + i, j) - b(rb + i, j))));
    return w;
}

}  // namespace

int main() {
    device::set_from_env();
    const bool on_dev = device::get() == device::Device::CUDA;
    std::printf("third-party ops (%s)\n", on_dev ? "CUDA device ops" : "host");
    const std::vector<int> pos = {0, 1, 2, 5, 9, 17};
    const size_t T = pos.size();

    // ---- 1. RoPE: neox, partial, interleaved ----
    {
        const size_t H = 3, hd = 8;
        const Matrix x = randn(T, H * hd, 1);
        struct Case { const char* name; ops::RopePairing pr; size_t n_rot; };
        for (const Case& c : {Case{"interleaved", ops::RopePairing::Interleaved, 8},
                              Case{"neox", ops::RopePairing::Neox, 8}, Case{"neox n_rot 4", ops::RopePairing::Neox, 4},
                              Case{"interleaved n_rot 6", ops::RopePairing::Interleaved, 6}}) {
            const bool neox = c.pr == ops::RopePairing::Neox;
            const Matrix y = host(rope_of(make_var(x), pos, hd, c.pr, c.n_rot == hd ? 0 : c.n_rot));
            const double e = max_abs(y, rope_ref(x, pos, H, hd, c.n_rot, neox, 10000.0f));
            CHECK(e < 1e-5);
            CHECK(bits_equal(y, rope_float(x, pos, H, hd, c.n_rot, neox, 10000.0f)));
            Var leaf = make_var(x, true);
            const double g = gradcheck([&] { return rope_of(leaf, pos, hd, c.pr, c.n_rot == hd ? 0 : c.n_rot); }, leaf, 2);
            CHECK(g < 2e-3);
            std::printf("1. rope %-20s vs float64 %.2e, host-loop bits equal, fd %.2e\n", c.name, e, g);
        }
        // The defaults are the original op: interleaved over the whole head, bit for bit.
        const Var v = make_var(randn(T, 3 * H * hd, 3));
        CHECK(bits_equal(host(ops::apply_rope(v, pos, 10000.0f, hd, true)),
                         host(ops::apply_rope(v, pos, 10000.0f, hd, true, ops::RopePairing::Interleaved, hd))));
    }

    // ---- 1. rmsnorm_group ----
    {
        const size_t g = 4, G = 3;
        const Matrix x = randn(T, g * G, 4), w = randn(1, g, 5);
        const Matrix y = host(ops::rmsnorm_group(make_var(x), make_var(w), g));
        std::vector<double> ref(x.rows() * x.cols());
        for (size_t i = 0; i < x.rows(); ++i)
            for (size_t q = 0; q < G; ++q) {
                double ms = 0;
                for (size_t j = 0; j < g; ++j) ms += double(x(i, q * g + j)) * x(i, q * g + j);
                const double ri = 1.0 / std::sqrt(ms / g + ops::kRmsNormEps);
                for (size_t j = 0; j < g; ++j) ref[i * x.cols() + q * g + j] = x(i, q * g + j) * ri * w(0, j);
            }
        const double e = max_abs(y, ref);
        CHECK(e < 1e-5);
        Var xl = make_var(x, true), wl = make_var(w, true);
        const double gx = gradcheck([&] { return ops::rmsnorm_group(xl, wl, g); }, xl, 6);
        const double gw = gradcheck([&] { return ops::rmsnorm_group(xl, wl, g); }, wl, 7);
        CHECK(gx < 2e-3 && gw < 2e-3);
        // The whole row as one group is ops::rmsnorm (host: bit for bit; device: the same kernels).
        const Matrix w2 = randn(1, g * G, 8);
        const Matrix a = host(ops::rmsnorm(make_var(x), make_var(w2)));
        const Matrix b = host(ops::rmsnorm_group(make_var(x), make_var(w2), g * G));
        CHECK(bits_equal(a, b));
        std::printf("1. rmsnorm_group (4 x 3) vs float64 %.2e, fd x %.2e w %.2e, one group == rmsnorm bit for bit\n", e,
                    gx, gw);
    }

    // ---- 1. repeat_heads ----
    {
        const size_t KV = 2, grp = 3, hd = 4;
        const Matrix x = randn(T, KV * hd, 9);
        const Matrix y = host(ops::repeat_heads(make_var(x), KV, grp, hd));
        bool ok = y.cols() == KV * grp * hd;
        for (size_t i = 0; ok && i < T; ++i)
            for (size_t h = 0; h < KV * grp; ++h)
                for (size_t c = 0; c < hd; ++c) ok = ok && y(i, h * hd + c) == x(i, (h / grp) * hd + c);
        CHECK(ok);
        Var xl = make_var(x, true);
        const double g = gradcheck([&] { return ops::repeat_heads(xl, KV, grp, hd); }, xl, 10);
        CHECK(g < 2e-3);
        // The backward's sum in ascending head order, bit for bit.
        Var y2 = ops::repeat_heads(xl, KV, grp, hd);
        const Matrix Rg = randn(T, KV * grp * hd, 11);
        xl->grad = Matrix();
        backward(ops::scale(ops::mean(ops::mul(y2, make_var(Rg))), static_cast<float>(T * KV * grp * hd)));
        bool sum_ok = true;
        const Matrix gx = host(make_var(xl->grad));
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < KV; ++k)
                for (size_t c = 0; c < hd; ++c) {
                    float acc = Rg(i, (k * grp) * hd + c);
                    for (size_t r = 1; r < grp; ++r) acc += Rg(i, (k * grp + r) * hd + c);
                    sum_ok = sum_ok && std::fabs(gx(i, k * hd + c) - acc) <= 1e-6f * (1.0f + std::fabs(acc));
                }
        CHECK(sum_ok);
        std::printf("1. repeat_heads (2 kv heads x 3): copies exact, backward the group sums, fd %.2e\n", g);
    }

    // ---- 1. conv1d_causal ----
    {
        const size_t C = 5, K = 4, sl = 7;
        const Matrix x = randn(2 * sl, C, 12), w = randn(C, K, 13, 0.5f), b = randn(1, C, 14);
        const Matrix y = host(ops::conv1d_causal(make_var(x), make_var(w), make_var(b), sl));
        std::vector<double> ref(y.rows() * C);
        for (size_t r0 = 0; r0 < 2 * sl; r0 += sl)
            for (size_t t = 0; t < sl; ++t)
                for (size_t c = 0; c < C; ++c) {
                    double acc = 0;
                    for (size_t k = 0; k < K; ++k)
                        if (t + k + 1 >= K) acc += double(w(c, k)) * x(r0 + t + k + 1 - K, c);
                    ref[(r0 + t) * C + c] = acc + b(0, c);
                }
        const double e = max_abs(y, ref);
        CHECK(e < 1e-5);
        Var xl = make_var(x, true), wl = make_var(w, true), bl = make_var(b, true);
        auto f = [&] { return ops::conv1d_causal(xl, wl, bl, sl); };
        const double gx = gradcheck(f, xl, 15), gw = gradcheck(f, wl, 16), gb = gradcheck(f, bl, 17);
        CHECK(gx < 2e-3 && gw < 2e-3 && gb < 2e-3);
        std::printf("1. conv1d_causal (K 4, 2 sequences) vs float64 %.2e, fd x %.2e w %.2e b %.2e\n", e, gx, gw, gb);
    }

    // ---- 1. selective_scan ----
    {
        const size_t D = 4, N = 3, sl = 6;
        const Matrix x = randn(2 * sl, D, 18), Bm = randn(2 * sl, N, 20),
                     Cm = randn(2 * sl, N, 21), Dk = randn(1, D, 23);
        Matrix dt = randn(2 * sl, D, 19);
        Matrix A = randn(D, N, 22, 0.5f);
        for (size_t i = 0; i < D; ++i)
            for (size_t n = 0; n < N; ++n) A(i, n) = -std::fabs(A(i, n)) - 0.1f;
        dt(0, 0) = 25.0f;  // past softplus's threshold: delta = dt
        const Matrix y = host(ops::selective_scan(make_var(x), make_var(dt), make_var(Bm), make_var(Cm), make_var(A),
                                                  make_var(Dk), true, sl));
        std::vector<double> ref(y.rows() * D);
        for (size_t r0 = 0; r0 < 2 * sl; r0 += sl) {
            std::vector<double> s(D * N, 0.0);
            for (size_t t = 0; t < sl; ++t) {
                const size_t row = r0 + t;
                for (size_t c = 0; c < D; ++c) {
                    const double v = dt(row, c), dl = v <= 20 ? std::log1p(std::exp(v)) : v;
                    double acc = 0;
                    for (size_t n = 0; n < N; ++n) {
                        s[c * N + n] = s[c * N + n] * std::exp(dl * A(c, n)) + Bm(row, n) * (x(row, c) * dl);
                        acc += s[c * N + n] * Cm(row, n);
                    }
                    ref[row * D + c] = acc + Dk(0, c) * x(row, c);
                }
            }
        }
        double scale = 1;
        for (double v : ref) scale = std::max(scale, std::fabs(v));
        const double e = max_abs(y, ref) / scale;
        CHECK(e < 1e-5);
        Var xl = make_var(x, true), dtl = make_var(dt, true), Bl = make_var(Bm, true), Cl = make_var(Cm, true),
            Al = make_var(A, true), Dl = make_var(Dk, true);
        auto f = [&] { return ops::selective_scan(xl, dtl, Bl, Cl, Al, Dl, true, sl); };
        double worst = 0;
        unsigned sd = 24;
        for (const Var* v : {&xl, &dtl, &Bl, &Cl, &Al, &Dl}) worst = std::max(worst, gradcheck(f, *v, sd++, 5e-3f));
        CHECK(worst < 5e-3);
        // dt_softplus false: delta = dt.
        Var dpos = make_var(randn(2 * sl, D, 31, 0.1f), true);
        const double g2 = gradcheck([&] { return ops::selective_scan(xl, dpos, Bl, Cl, Al, Dl, false, sl); }, dpos, 32,
                                    5e-3f);
        CHECK(g2 < 5e-3);
        std::printf("1. selective_scan (D 4, N 3, 2 sequences) vs float64 %.2e of the range, fd worst %.2e "
                    "(no softplus %.2e)\n", e, worst, g2);
    }

    // ---- 3. the llama block with the options ----
    {
        nn::LlamaConfig base;
        base.vocab = 23;
        base.d = 24;
        base.n_layers = 2;
        base.n_heads = 4;
        base.d_ff = 40;
        base.n_ctx = 32;
        const auto ids = ids_of(12, base.vocab, 40);
        // The options at their default values: the plain block, bit for bit.
        nn::LlamaConfig same = base;
        same.n_kv_heads = 4;
        same.rope_dims = 6;
        CHECK(!same.thirdparty());
        nn::Llama m0(base, 3), m1(same, 3);
        CHECK(bits_equal(host(m0.forward(ids)), host(m1.forward(ids))));
        // Grouped-query attention = the multi-head model with the kv columns repeated, bit for bit.
        nn::LlamaConfig gq = base;
        gq.n_kv_heads = 2;
        nn::Llama mg(gq, 3), mh(base, 3);
        for (const auto& [name, p] : mh.named_parameters()) {
            Var src = nullptr;
            for (const auto& [n2, q] : mg.named_parameters())
                if (n2 == name) src = q;
            CHECK(src != nullptr);
            if (name.find("k_proj") != std::string::npos || name.find("v_proj") != std::string::npos) {
                const size_t dk = base.d / base.n_heads, grp = 2;
                for (size_t i = 0; i < p->data.rows(); ++i)
                    for (size_t h = 0; h < base.n_heads; ++h)
                        for (size_t c = 0; c < dk; ++c) p->data(i, h * dk + c) = src->data(i, (h / grp) * dk + c);
            } else {
                p->data = src->data;
            }
        }
        const double dg = logits_diff(host(mg.forward(ids)), 0, host(mh.forward(ids)), 0, ids.size());
        CHECK(on_dev ? dg < 1e-5 : dg == 0.0);  // cuBLAS may tile the two widths differently
        std::printf("3. defaults == plain llama bit for bit; GQA (4 q / 2 kv heads) == MHA with repeated kv columns, "
                    "max diff %.1e\n", dg);
        // Every option at once: GQA, NEOX, partial rotary, q/k norms.
        nn::LlamaConfig tp = base;
        tp.n_kv_heads = 2;
        tp.rope_pairing = "neox";
        tp.rope_dims = 4;
        tp.qk_norm = true;
        tp.rope_theta = 5000.0f;
        nn::Llama m(tp, 5);
        Var full;
        {
            NoGrad ng;
            full = m.forward(ids);
            nn::LlamaKV kv;
            Var pre = m.forward_cached(std::vector<int>(ids.begin(), ids.begin() + 5), kv);
            double dc = logits_diff(host(pre), 0, host(full), 0, 5);
            for (size_t t = 5; t < ids.size(); ++t) {
                Var one = m.forward_cached({ids[t]}, kv);
                dc = std::max(dc, logits_diff(host(one), 0, host(full), t, 1));
            }
            CHECK(dc < 1e-5);
            const auto ids2 = ids_of(12, base.vocab, 41);
            std::vector<int> both = ids;
            both.insert(both.end(), ids2.begin(), ids2.end());
            const Matrix b = host(m.forward(both, ids.size())), s1 = host(full), s2 = host(m.forward(ids2));
            const double db = std::max(logits_diff(b, 0, s1, 0, ids.size()), logits_diff(b, ids.size(), s2, 0, ids.size()));
            // The stacked attention runs its products over every row of the batch (masked weights exactly 0),
            // so the plain llama's batch agrees with its singles to rounding, not bit for bit: the same here.
            const Matrix pb = host(m0.forward(both, ids.size())), p1 = host(m0.forward(ids)), p2 = host(m0.forward(ids2));
            const double dp = std::max(logits_diff(pb, 0, p1, 0, ids.size()), logits_diff(pb, ids.size(), p2, 0, ids.size()));
            CHECK(db < 1e-5);
            std::printf("3. GQA + neox + rope_dims 4 + qk_norm: cached decode vs forward %.2e, batch vs singles %.1e "
                        "(the plain llama's %.1e)\n", dc, db, dp);
        }
        const std::vector<int> y(ids.rbegin(), ids.rend());
        auto loss_now = [&]() { return ops::cross_entropy(m.forward(ids), y); };
        backward(loss_now());
        int checked = 0;
        double worst = 0;
        const float h = 5e-3f;
        for (const auto& [name, p] : m.named_parameters()) {
            const bool want = name.find("layers.0.self_attn.k_proj") != std::string::npos ||
                              name.find("q_norm") != std::string::npos || name.find("k_norm") != std::string::npos;
            if (!want) continue;
            const Matrix g = host(make_var(p->grad));
            for (int k = 0; k < 3; ++k) {
                const size_t i = (k * 5 + 1) % p->data.rows(), j = (k * 3 + 1) % p->data.cols();
                const float w0 = p->data(i, j);
                NoGrad ng;
                p->data(i, j) = w0 + h;
                const double lp = host(loss_now())(0, 0);
                p->data(i, j) = w0 - h;
                const double lm = host(loss_now())(0, 0);
                p->data(i, j) = w0;
                const double fd = (lp - lm) / (2.0 * h);
                worst = std::max(worst, std::fabs(fd - g(i, j)) / (1.0 + std::fabs(fd)));
                ++checked;
            }
        }
        CHECK(checked == 3 * 5);  // k_proj of layer 0, q_norm and k_norm of both layers
        CHECK(worst < 2e-2);
        std::printf("3. finite differences through k_proj and the q/k norms (%d spots): worst %.2e\n", checked, worst);
    }

    // ---- 3b. qk_temp: a learned per-head temperature on the normed q ----
    {
        nn::LlamaConfig c;
        c.vocab = 23;
        c.d = 24;
        c.n_layers = 2;
        c.n_heads = 4;
        c.d_ff = 40;
        c.n_ctx = 32;
        c.qk_norm = true;
        const auto ids = ids_of(12, c.vocab, 42);
        // Temperature 1 is the plain qk_norm block bit for bit (x * (1 * 1) = x).
        nn::LlamaConfig one = c;
        one.qk_temp = 1.0f;
        nn::Llama mq(c, 7), m1(one, 7);
        CHECK(bits_equal(host(mq.forward(ids)), host(m1.forward(ids))));
        // Projection BF16 must not round the separate FP32 temperature multiply.
        // Otherwise an unrepresentable temperature and its BF16-rounded value
        // become identical, while exported IR retains the original FP32 value.
        {
            nn::LlamaConfig precise = c;
            precise.qk_temp = 1.236f;
            nn::LlamaConfig rounded = precise;
            rounded.qk_temp = device::bf16_round(precise.qk_temp);
            CHECK(precise.qk_temp != rounded.qk_temp);
            nn::Llama mp(precise, 7), mr(rounded, 7);
            const auto old_precision = device::matmul_precision();
            device::set_matmul_precision(device::Precision::BF16);
            const Matrix yp = host(mp.forward(ids)), yr = host(mr.forward(ids));
            device::set_matmul_precision(old_precision);
            CHECK(!bits_equal(yp, yr));
        }
        {
            nn::LlamaConfig bad = c;
            bad.qk_temp = std::numeric_limits<float>::infinity();
            bool refused = false;
            try {
                nn::Llama invalid(bad, 7);
            } catch (const std::runtime_error&) {
                refused = true;
            }
            CHECK(refused);
        }
        // Head h's q columns times t[h]: the composition (repeat_heads, the ones product, mul) against a
        // float re-statement, and its finite differences in t and in x.
        const size_t H = 4, dk = 6, T = 5;
        Var x = make_var(randn(T, H * dk, 60), true);
        Var t = make_var(randn(1, H, 61, 0.5f), true);
        auto f = [&]() {
            const Var row = ops::repeat_heads(t, H, dk, 1);
            return ops::mul(x, ops::matmul(make_var(Matrix(T, 1, 1.0f)), row));
        };
        Matrix want(T, H * dk);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < H * dk; ++j) want(i, j) = x->data(i, j) * t->data(0, j / dk);
        CHECK(bits_equal(host(f()), want));
        const double gt = gradcheck(f, t, 62), gx = gradcheck(f, x, 63);
        CHECK(gt < 1e-3 && gx < 1e-3);
        // The model: init value, KV-cached decode, a stacked batch, finite differences through q_temp.
        nn::LlamaConfig tc = c;
        tc.qk_temp = 2.75f;
        tc.n_kv_heads = 2;
        nn::Llama m(tc, 9);
        size_t n_temp = 0;
        for (const auto& [name, p] : m.named_parameters())
            if (name.find("q_temp") != std::string::npos) {
                CHECK(p->data.rows() == 1 && p->data.cols() == tc.n_heads && p->data(0, 3) == 2.75f);
                for (size_t h = 0; h < tc.n_heads; ++h) p->data(0, h) = 1.5f + 0.5f * static_cast<float>(h);
                ++n_temp;
            }
        CHECK(n_temp == tc.n_layers);
        {
            NoGrad ng;
            const Var full = m.forward(ids);
            nn::LlamaKV kv;
            Var pre = m.forward_cached(std::vector<int>(ids.begin(), ids.begin() + 5), kv);
            double dc = logits_diff(host(pre), 0, host(full), 0, 5);
            for (size_t s = 5; s < ids.size(); ++s)
                dc = std::max(dc, logits_diff(host(m.forward_cached({ids[s]}, kv)), 0, host(full), s, 1));
            CHECK(dc < 1e-5);
            const auto ids2 = ids_of(12, c.vocab, 43);
            std::vector<int> both = ids;
            both.insert(both.end(), ids2.begin(), ids2.end());
            const Matrix b = host(m.forward(both, ids.size())), s2 = host(m.forward(ids2));
            const double db = std::max(logits_diff(b, 0, host(full), 0, ids.size()),
                                       logits_diff(b, ids.size(), s2, 0, ids.size()));
            CHECK(db < 1e-5);
            std::printf("3b. qk_temp: t = 1 == qk_norm bit for bit; head scaling exact; fd t %.2e x %.2e; cached decode "
                        "%.2e, batch vs singles %.1e\n", gt, gx, dc, db);
        }
        const std::vector<int> y(ids.rbegin(), ids.rend());
        auto loss_now = [&]() { return ops::cross_entropy(m.forward(ids), y); };
        backward(loss_now());
        int checked = 0;
        double worst = 0;
        const float h = 5e-3f;
        for (const auto& [name, p] : m.named_parameters()) {
            if (name.find("q_temp") == std::string::npos) continue;
            const Matrix g = host(make_var(p->grad));
            for (size_t j = 0; j < p->data.cols(); ++j) {
                const float w0 = p->data(0, j);
                NoGrad ng;
                p->data(0, j) = w0 + h;
                const double lp = host(loss_now())(0, 0);
                p->data(0, j) = w0 - h;
                const double lm = host(loss_now())(0, 0);
                p->data(0, j) = w0;
                const double fd = (lp - lm) / (2.0 * h);
                worst = std::max(worst, std::fabs(fd - g(0, j)) / (1.0 + std::fabs(fd)));
                ++checked;
            }
        }
        CHECK(checked == 2 * 4);
        CHECK(worst < 2e-2);
        std::printf("3b. finite differences through q_temp (%d spots): worst %.2e\n", checked, worst);
    }

    // ---- 4. Mamba-1 ----
    {
        parity::Mamba1Config c;
        c.on = true;
        parity::MambaLM m(19, 16, 2, 4, 7, c);
        const auto a = ids_of(9, 19, 50), b = ids_of(9, 19, 51);
        std::vector<int> both = a;
        both.insert(both.end(), b.begin(), b.end());
        {
            NoGrad ng;
            const Matrix sb = host(m.forward(both, 9)), s1 = host(m.forward(a)), s2 = host(m.forward(b));
            const double db = std::max(logits_diff(sb, 0, s1, 0, 9), logits_diff(sb, 9, s2, 0, 9));
            CHECK(on_dev ? db < 1e-5 : db == 0.0);  // cuBLAS may tile the two row counts differently
            std::printf("4. mamba1 (d 16, Di 32, N 4, K 4): batch == singles %.1e\n", db);
        }
        const std::vector<int> y(a.rbegin(), a.rend());
        auto loss_now = [&]() { return ops::cross_entropy(m.forward(a), y); };
        backward(loss_now());
        double worst = 0;
        int checked = 0;
        const float h = 5e-3f;
        for (const auto& [name, p] : m.named_parameters()) {
            const bool want = name.find("layers.0.A_log") != std::string::npos ||
                              name.find("layers.0.conv1d.weight") != std::string::npos ||
                              name.find("layers.1.dt_proj.bias") != std::string::npos ||
                              name.find("layers.0.D") != std::string::npos ||
                              name.find("layers.1.x_proj") != std::string::npos;
            if (!want) continue;
            const Matrix g = host(make_var(p->grad));
            for (int k = 0; k < 3; ++k) {
                const size_t i = (k * 7 + 1) % p->data.rows(), j = (k * 5 + 2) % p->data.cols();
                const float w0 = p->data(i, j);
                NoGrad ng;
                p->data(i, j) = w0 + h;
                const double lp = host(loss_now())(0, 0);
                p->data(i, j) = w0 - h;
                const double lm = host(loss_now())(0, 0);
                p->data(i, j) = w0;
                const double fd = (lp - lm) / (2.0 * h);
                worst = std::max(worst, std::fabs(fd - g(i, j)) / (1.0 + std::fabs(fd)));
                ++checked;
            }
        }
        CHECK(checked == 15);
        CHECK(worst < 2e-2);
        std::printf("4. finite differences through A_log, conv1d, D, x_proj and dt_proj's bias (%d spots): worst %.2e\n",
                    checked, worst);
    }
    std::printf("all third-party op checks passed\n");
    return 0;
}
