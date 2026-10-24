// Mixed precision (BF16 GEMM inputs, device_cache.hpp Precision) and flash
// attention (ops::AttnKernel::Flash), 24 Oct 2026. On the CPU (the local gate):
//   1. bf16_round: edge cases and 1,000,000 random floats against a reference
//      that picks the nearer bfloat16 neighbour, ties to even.
//   2. gemm: the FP32 overload is the 6-argument gemm bit for bit; BF16 is the
//      FP32 gemm of bfloat16-rounded inputs bit for bit (all four transposes).
//   3. ops::matmul under BF16: forward and both gradients are the rounded
//      GEMMs bit for bit; under FP32 the tape is the pre-change one.
//   4. Flash vs the tiled attention on every mask (full blocks, causal,
//      turn_state), head widths 3 to 64, sequence lengths that are not tile
//      multiples: outputs and gradients to float rounding (bounds below).
//   5. Finite-difference gradchecks of the flash op on every mask.
//   6. Training (a two-layer llama, 2 x 2 stacked sequences, AdamW, clip):
//      flash vs tiled per-step losses; BF16 vs FP32 per-step losses within a
//      stated tolerance, both learning.
// With MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 (tools/colab_cuda_validate.sh)
// the same binary adds the device legs: the device flash kernels against the
// host flash reference and the device tiled path, the device BF16 GEMM
// (cublasGemmEx) against the host emulation, and device training runs (step
// residency + deferred downloads) against the host runs.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/llama.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;
using device::Precision;
using device::Trans;

namespace {

bool cuda_run() { return device::get() == device::Device::CUDA; }

uint32_t bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return u;
}
float from_bits(uint32_t u) {
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

size_t bits_differ(const Matrix& a, const Matrix& b) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols());
    size_t n = 0;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j) n += bits(a(i, j)) != bits(b(i, j));
    return n;
}

// max |a - b| / max |b|
double rel(const Matrix& a, const Matrix& b) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols());
    double s = 0, w = 0;
    for (size_t i = 0; i < b.rows(); ++i)
        for (size_t j = 0; j < b.cols(); ++j) {
            s = std::max(s, (double)std::fabs(b(i, j)));
            w = std::max(w, std::fabs((double)a(i, j) - b(i, j)));
        }
    return w / std::max(s, 1e-30);
}

Matrix randm(size_t r, size_t c, std::mt19937& g, float sd = 1.0f) {
    std::normal_distribution<float> nd(0.0f, sd);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = nd(g);
    return m;
}

// ---- 1. rounding ----
// The reference: the two bfloat16 values around x (truncation and the next
// one up in magnitude), the nearer one, ties to the even mantissa; overflow
// past the largest finite bfloat16 goes to inf.
float bf16_ref(float x) {
    if (std::isnan(x)) return x;
    const uint32_t u = bits(x);
    const uint32_t lo = u & 0xffff0000u, hi = lo + 0x10000u;
    const double dx = x, dlo = from_bits(lo);
    const float fhi = from_bits(hi);  // may be inf (or a NaN pattern past inf: not reached for finite x)
    if ((u & 0xffffu) == 0) return x;
    const double dhi = std::isinf(fhi) ? std::copysign(std::ldexp(1.0, 128), dx) : (double)fhi;
    const double elo = std::fabs(dx - dlo), ehi = std::fabs(dhi - dx);
    if (elo < ehi) return from_bits(lo);
    if (ehi < elo) return fhi;
    return ((lo >> 16) & 1u) ? fhi : from_bits(lo);
}

void test_round() {
    const float one = 1.0f;
    CHECK(device::bf16_round(one) == one);
    CHECK(device::bf16_round(1.0f + std::ldexp(1.0f, -8)) == 1.0f);                                  // tie -> even (down)
    CHECK(device::bf16_round(1.0f + 3 * std::ldexp(1.0f, -8)) == 1.0f + std::ldexp(1.0f, -6));       // tie -> even (up)
    CHECK(device::bf16_round(1.0f + std::ldexp(1.0f, -8) + std::ldexp(1.0f, -20)) == 1.0f + std::ldexp(1.0f, -7));
    CHECK(bits(device::bf16_round(-0.0f)) == bits(-0.0f));
    CHECK(std::isinf(device::bf16_round(std::numeric_limits<float>::max())));
    CHECK(std::isinf(device::bf16_round(-std::numeric_limits<float>::infinity())) &&
          device::bf16_round(-std::numeric_limits<float>::infinity()) < 0);
    CHECK(std::isnan(device::bf16_round(std::numeric_limits<float>::quiet_NaN())));
    CHECK(std::isnan(device::bf16_round(from_bits(0x7f800001u))));  // a signalling NaN whose payload rounds away
    CHECK(device::bf16_round(1e-40f) == bf16_ref(1e-40f));         // subnormals round like normals
    std::mt19937 g(3);
    size_t bad = 0;
    for (int k = 0; k < 1000000; ++k) {
        float x = from_bits(static_cast<uint32_t>(g()));
        if (std::isnan(x)) continue;
        if (bits(device::bf16_round(x)) != bits(bf16_ref(x))) ++bad;
        // and every rounded value is a bfloat16 (low half zero)
        if (!std::isnan(x) && (bits(device::bf16_round(x)) & 0xffffu) != 0) ++bad;
    }
    std::printf("  bf16_round: edge cases and 1e6 random floats vs the nearest-neighbour reference: %zu differ\n", bad);
    CHECK(bad == 0);
}

// ---- 2. gemm ----
void test_gemm() {
    std::mt19937 g(7);
    size_t diff_fp32 = 0, diff_bf16 = 0;
    for (int ta = 0; ta < 2; ++ta)
        for (int tb = 0; tb < 2; ++tb) {
            const size_t M = 37, K = 53, N = 29;
            const Matrix A = ta ? randm(K, M, g) : randm(M, K, g), B = tb ? randm(N, K, g) : randm(K, N, g);
            const Trans tA = ta ? Trans::T : Trans::N, tB = tb ? Trans::T : Trans::N;
            const Matrix c6 = device::gemm(A, nullptr, tA, B, nullptr, tB);
            diff_fp32 += bits_differ(device::gemm(A, nullptr, tA, B, nullptr, tB, Precision::FP32), c6);
            const Matrix ref = device::gemm(device::bf16_rounded(A), nullptr, tA, device::bf16_rounded(B), nullptr, tB);
            const Matrix c16 = device::gemm(A, nullptr, tA, B, nullptr, tB, Precision::BF16);
            diff_bf16 += bits_differ(c16, ref);
            CHECK(rel(c16, c6) > 0 && rel(c16, c6) < 2e-2);  // it did round, and only by bfloat16's precision
        }
    std::printf("  gemm: FP32 overload vs 6-argument gemm %zu bits differ; BF16 vs FP32 gemm of rounded inputs %zu "
                "bits differ (outside a step window: the host emulation on every build)\n",
                diff_fp32, diff_bf16);
    CHECK(diff_fp32 == 0 && diff_bf16 == 0);
}

// The device BF16 GEMM (cublasGemmEx) inside a step window vs the host emulation.
void test_gemm_device() {
    if (!cuda_run()) return;
    std::mt19937 g(8);
    double worst = 0;
    device::set_step_residency(true);
    for (int ta = 0; ta < 2; ++ta)
        for (int tb = 0; tb < 2; ++tb) {
            const size_t M = 300, K = 196, N = 260;
            const Matrix A = ta ? randm(K, M, g) : randm(M, K, g), B = tb ? randm(N, K, g) : randm(K, N, g);
            const Trans tA = ta ? Trans::T : Trans::N, tB = tb ? Trans::T : Trans::N;
            const Matrix host = device::gemm(device::bf16_rounded(A), nullptr, tA, device::bf16_rounded(B), nullptr, tB);
            device::step_begin();
            Matrix dev = device::gemm(A, nullptr, tA, B, nullptr, tB, Precision::BF16);
            device::step_end();
            device::materialize(dev);
            worst = std::max(worst, rel(dev, host));
        }
    device::set_step_residency(false);
    // bfloat16 x bfloat16 products are exact in float32: only the summation order differs.
    std::printf("  device BF16 gemm (cublasGemmEx) vs host emulation, K = 196: worst rel %.2e (bound 1e-5)\n", worst);
    CHECK(worst < 1e-5);
}

// ---- 3. ops::matmul under BF16 ----
void test_matmul_op() {
    std::mt19937 g(11);
    // y is 8 x 4: the mean's 1/32 is a power of two, so dL/dy = R / 32 exactly.
    Var a = make_var(randm(8, 14, g), true), b = make_var(randm(14, 4, g), true);
    const Matrix R = randm(8, 4, g);
    device::set_matmul_precision(Precision::BF16);
    Var y = ops::matmul(a, b);
    backward(ops::mean(ops::mul(y, make_var(R))));
    device::set_matmul_precision(Precision::FP32);
    Matrix R32 = R;
    for (size_t i = 0; i < R32.rows(); ++i)
        for (size_t j = 0; j < R32.cols(); ++j) R32(i, j) *= 1.0f / 32.0f;
    const Matrix ra = device::bf16_rounded(a->data), rb = device::bf16_rounded(b->data), rR = device::bf16_rounded(R32);
    const Matrix y_ref = device::gemm(ra, nullptr, Trans::N, rb, nullptr, Trans::N);
    // da = (R / 32) b^T and db = a^T (R / 32), both inputs rounded.
    const Matrix da_ref = device::gemm(rR, nullptr, Trans::N, rb, nullptr, Trans::T);
    const Matrix db_ref = device::gemm(ra, nullptr, Trans::T, rR, nullptr, Trans::N);
    const size_t n = bits_differ(y->data, y_ref) + bits_differ(a->grad, da_ref) + bits_differ(b->grad, db_ref);
    std::printf("  ops::matmul under BF16: forward and both gradients vs the rounded GEMMs: %zu bits differ\n", n);
    CHECK(n == 0);
}

// ---- 4/5. flash attention ----
struct AttnCase {
    size_t H, dk, sl, nseq;
    int mode;  // 0 full, 1 causal, 2 turn_state
};

std::vector<int> ts_codes(size_t T, size_t sl, std::mt19937& g) {
    // Ids with complete k = 3 state blocks at random places (vocab 37, ids 37..39).
    std::vector<int> ids;
    for (size_t i = 0; i < T; ++i) ids.push_back(4 + static_cast<int>(g() % 30));
    for (size_t p = 3; p + 3 < T; p += 7 + g() % 9) {
        ids[p] = 37;
        ids[p + 1] = 38;
        ids[p + 2] = 39;
    }
    return ops::turn_state_codes(ids, 37, 3, sl);
}

Var attend(const Var& qkv, const AttnCase& c, float sc, const std::vector<int>& code, ops::AttnKernel k) {
    ops::set_attention_kernel(k);
    Var y = c.mode == 2 ? ops::attention_heads_turn_state(qkv, c.H, sc, c.sl, code)
                        : ops::attention_heads(qkv, c.H, sc, c.nseq > 1 ? c.sl : 0, c.mode == 1);
    ops::set_attention_kernel(ops::AttnKernel::Tiled);
    return y;
}

struct AttnOut {
    Matrix y, g;
};

AttnOut attn_run(const Matrix& x, const Matrix& R, const AttnCase& c, const std::vector<int>& code, ops::AttnKernel k) {
    Var qkv = make_var(x, true);
    const float sc = 1.0f / std::sqrt(static_cast<float>(c.dk));
    device::step_begin();
    Var y = attend(qkv, c, sc, code, k);
    backward(ops::mean(ops::mul(y, make_var(R))));
    device::step_end();
    device::materialize(y->data);
    device::materialize(qkv->grad);
    return {y->data, qkv->grad};
}

const char* mode_name(int m) { return m == 0 ? "full" : m == 1 ? "causal" : "turn_state"; }

void test_flash_vs_tiled() {
    // Head widths 3 .. 64 (the device pads to 16 / 32 / 64), sequence lengths
    // that are not multiples of the 32- or 64-row tiles, one and several
    // sequences, every mask.
    const AttnCase cases[] = {{2, 3, 37, 2, 1}, {2, 8, 37, 2, 0}, {3, 16, 70, 1, 1}, {2, 33, 45, 2, 1},
                              {1, 64, 97, 2, 1}, {2, 64, 64, 2, 0}, {2, 8, 37, 2, 2}, {2, 64, 75, 2, 2},
                              {4, 16, 129, 1, 1}};
    std::mt19937 g(21);
    double wy = 0, wg = 0;
    for (const auto& c : cases) {
        const size_t T = c.sl * c.nseq, d = c.H * c.dk;
        const Matrix x = randm(T, 3 * d, g, 0.8f), R = randm(T, d, g);
        const std::vector<int> code = c.mode == 2 ? ts_codes(T, c.sl, g) : std::vector<int>();
        const AttnOut t = attn_run(x, R, c, code, ops::AttnKernel::Tiled);
        const AttnOut f = attn_run(x, R, c, code, ops::AttnKernel::Flash);
        const double ry = rel(f.y, t.y), rg = rel(f.g, t.g);
        wy = std::max(wy, ry);
        wg = std::max(wg, rg);
        std::printf("    flash vs tiled %-10s H %zu dk %2zu sl %3zu x %zu: y rel %.2e, dqkv rel %.2e\n",
                    mode_name(c.mode), c.H, c.dk, c.sl, c.nseq, ry, rg);
    }
    // FP32 throughout; the online softmax rescales and sums in another order.
    std::printf("  flash vs tiled (%s): worst y rel %.2e (bound 2e-6), worst dqkv rel %.2e (bound 1e-5)\n",
                cuda_run() && device::device_ops_enabled() ? "device kernels" : "host loops", wy, wg);
    CHECK(wy < 2e-6 && wg < 1e-5);
}

// The device flash kernels against the host flash reference (same algorithm).
void test_flash_device_vs_host() {
    if (!(cuda_run() && device::device_ops_enabled())) return;
    const AttnCase cases[] = {{2, 8, 37, 2, 0}, {3, 16, 70, 1, 1}, {2, 33, 45, 2, 1}, {12, 64, 512, 2, 1},
                              {2, 64, 75, 2, 2}};
    std::mt19937 g(31);
    double wy = 0, wg = 0;
    for (const auto& c : cases) {
        const size_t T = c.sl * c.nseq, d = c.H * c.dk;
        const Matrix x = randm(T, 3 * d, g, 0.8f), R = randm(T, d, g);
        const std::vector<int> code = c.mode == 2 ? ts_codes(T, c.sl, g) : std::vector<int>();
        const AttnOut dev = attn_run(x, R, c, code, ops::AttnKernel::Flash);
        device::set_device_ops(false);
        const AttnOut host = attn_run(x, R, c, code, ops::AttnKernel::Flash);
        device::set_device_ops(true);
        wy = std::max(wy, rel(dev.y, host.y));
        wg = std::max(wg, rel(dev.g, host.g));
    }
    std::printf("  device flash kernels vs host flash loops (incl. H 12 dk 64 sl 512): worst y rel %.2e (bound 2e-6), "
                "dqkv rel %.2e (bound 1e-5)\n",
                wy, wg);
    CHECK(wy < 2e-6 && wg < 1e-5);
}

void test_flash_fd() {
    const AttnCase cases[] = {{2, 3, 13, 2, 0}, {2, 3, 13, 2, 1}, {2, 3, 13, 2, 2}, {1, 5, 40, 1, 1}};
    std::mt19937 g(41);
    for (const auto& c : cases) {
        const size_t T = c.sl * c.nseq, d = c.H * c.dk;
        Var qkv = make_var(randm(T, 3 * d, g, 0.7f), true);
        const Matrix R = randm(T, d, g);
        const std::vector<int> code = c.mode == 2 ? ts_codes(T, c.sl, g) : std::vector<int>();
        auto loss = [&] { return ops::mean(ops::mul(attend(qkv, c, 0.6f, code, ops::AttnKernel::Flash), make_var(R))); };
        qkv->grad = Matrix();
        device::step_begin();
        backward(loss());
        device::step_end();
        device::materialize(qkv->grad);
        const Matrix an = qkv->grad;
        double worst = 0;
        const float eps = 1e-2f;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < 3 * d; ++j) {
                double fp, fm;
                {
                    NoGrad ng;
                    const float keep = qkv->data(i, j);
                    qkv->data(i, j) = keep + eps;
                    fp = loss()->data(0, 0);
                    qkv->data(i, j) = keep - eps;
                    fm = loss()->data(0, 0);
                    qkv->data(i, j) = keep;
                }
                const double fd = (fp - fm) / (2.0 * eps), a = an(i, j);
                worst = std::max(worst, std::fabs(fd - a) / std::max({std::fabs(fd), std::fabs(a), 1e-3}));
            }
        std::printf("  FD gradcheck flash %-10s (H %zu dk %zu, %zu x %zu rows, every entry of q, k, v): worst rel %.2e\n",
                    mode_name(c.mode), c.H, c.dk, c.nseq, c.sl, worst);
        CHECK(worst < 2e-2);  // the bound of the tiled op's gradcheck (tests/test_state_tokens.cpp)
    }
}

// ---- 6. training ----
nn::LlamaConfig tiny() {
    nn::LlamaConfig c;
    c.vocab = 37;
    c.d = 32;
    c.n_heads = 2;
    c.d_ff = 48;
    c.n_ctx = 64;
    c.n_layers = 2;
    return c;
}

void randomise(nn::Module& m, unsigned seed, float scale) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto& [name, p] : m.named_parameters()) {
        const bool norm = name.find("norm") != std::string::npos;
        for (size_t i = 0; i < p->data.rows(); ++i)
            for (size_t j = 0; j < p->data.cols(); ++j) p->data(i, j) = norm ? 1.0f + 0.1f * nd(g) : nd(g);
    }
}

// mtstudio's step: B stacked sequences per micro-batch, two micro-batches,
// clip, AdamW; dev = the device op set with step residency and deferred
// downloads (as every production run), else the host loops. The data is a
// learnable pattern (a repeated random sequence with noise), so losses fall.
std::vector<float> train(Precision prec, ops::AttnKernel k, bool dev, size_t steps) {
    const bool was_ops = device::device_ops_enabled();
    device::set_device_ops(dev);
    if (dev) {
        device::set_step_residency(true);
        device::set_defer_downloads(true);
    }
    device::set_matmul_precision(prec);
    ops::set_attention_kernel(k);
    nn::Llama m(tiny(), 41);
    randomise(m, 12, 0.15f);
    m.train();
    nn::AdamW opt(m.parameters(), 3e-3f);
    std::mt19937 data(5);
    const size_t T = 40, B = 2, accum = 2;
    std::vector<int> base(T + 1);
    for (auto& t : base) t = static_cast<int>(data() % 37);
    std::vector<float> losses;
    for (size_t step = 1; step <= steps; ++step) {
        device::step_begin();
        opt.zero_grad();
        float lsum = 0;
        for (size_t kk = 0; kk < accum; ++kk) {
            std::vector<int> x, y;
            for (size_t b = 0; b < B; ++b) {
                const size_t off = data() % 4;
                for (size_t t = 0; t < T; ++t) {
                    const bool noise = data() % 10 == 0;
                    x.push_back(noise ? static_cast<int>(data() % 37) : base[(t + off) % T]);
                    y.push_back(base[(t + off + 1) % T]);
                }
            }
            Var loss = ops::cross_entropy(m.forward(x, T), y);
            backward(ops::scale(loss, 1.0f / accum));
            lsum += loss->data(0, 0) / accum;
        }
        device::step_end();
        ops::clip_grad_norm(m.parameters(), 1.0f);
        opt.step();
        losses.push_back(lsum);
    }
    device::materialize_all();
    device::set_matmul_precision(Precision::FP32);
    ops::set_attention_kernel(ops::AttnKernel::Tiled);
    if (dev) {
        device::set_defer_downloads(false);
        device::set_step_residency(false);
    }
    device::set_device_ops(was_ops);
    return losses;
}

double worst_rel(const std::vector<float>& a, const std::vector<float>& b) {
    double w = 0;
    for (size_t i = 0; i < a.size(); ++i) w = std::max(w, std::fabs((double)a[i] - b[i]) / std::fabs((double)b[i]));
    return w;
}

void test_training() {
    const bool dev = cuda_run() && device::device_ops_enabled();
    const size_t steps = 60;
    const auto fp32 = train(Precision::FP32, ops::AttnKernel::Tiled, dev, steps);
    const auto again = train(Precision::FP32, ops::AttnKernel::Tiled, dev, steps);
    const auto flash = train(Precision::FP32, ops::AttnKernel::Flash, dev, steps);
    const auto bf16 = train(Precision::BF16, ops::AttnKernel::Tiled, dev, steps);
    const auto both = train(Precision::BF16, ops::AttnKernel::Flash, dev, steps);
    size_t rep = 0;
    for (size_t i = 0; i < steps; ++i) rep += bits(fp32[i]) != bits(again[i]);
    const double rf = worst_rel(flash, fp32), rb = worst_rel(bf16, fp32), rbf = worst_rel(both, fp32);
    std::printf("  training (%s, %zu steps, loss %.4f -> %.4f FP32): FP32 rerun %zu steps differ; flash vs tiled worst "
                "rel %.2e (bound 1e-4); BF16 vs FP32 worst rel %.2e, BF16 + flash %.2e (bound 2e-2); final BF16 %.4f\n",
                dev ? "device step" : "host", steps, fp32.front(), fp32.back(), rep, rf, rb, rbf, bf16.back());
    CHECK(rep == 0);
    CHECK(rf < 1e-4);
    CHECK(rb < 2e-2 && rbf < 2e-2);
    CHECK(fp32.back() < 0.6f * fp32.front() && bf16.back() < 0.6f * bf16.front());
    if (dev) {
        // The device runs against the host runs of the same configuration.
        const auto h32 = train(Precision::FP32, ops::AttnKernel::Tiled, false, steps);
        const auto hfl = train(Precision::FP32, ops::AttnKernel::Flash, false, steps);
        const auto h16 = train(Precision::BF16, ops::AttnKernel::Tiled, false, steps);
        const double d32 = worst_rel(fp32, h32), dfl = worst_rel(flash, hfl), d16 = worst_rel(bf16, h16);
        std::printf("  device vs host training: FP32 tiled %.2e, flash %.2e (bounds 1e-4), BF16 %.2e (bound 5e-3)\n",
                    d32, dfl, d16);
        CHECK(d32 < 1e-4 && dfl < 1e-4 && d16 < 5e-3);
    }
}

}  // namespace

int main() {
    device::set_from_env();
    std::printf("mixed precision + flash attention (%s%s)\n", cuda_run() ? "CUDA" : "CPU",
                device::device_ops_enabled() ? ", device op set" : "");
    test_round();
    test_gemm();
    test_gemm_device();
    test_matmul_op();
    test_flash_vs_tiled();
    test_flash_device_vs_host();
    test_flash_fd();
    test_training();
    std::printf("mixed precision + flash attention: all checks passed\n");
    return 0;
}
