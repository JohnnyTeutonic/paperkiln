// Phase A GEMM through this library's device pool (device_cache.hpp
// pooled_matmul; 25 Oct 2026). Every GEMM outside a step window (mtstudio
// score, chat, eval, sample) used to go through transformer_core's
// cuda::matmul, whose buffer cache keeps one block per exact element count
// for the life of the process, outside MICROTORCH_POOL_MB, and exit(1)s on
// a failed cudaMalloc. A Mixture-of-Recursions rung's `mtstudio score`
// beside a training job on the same L4 died that way mid-chunk.
//
//   1. pooled_matmul against cuda::matmul, bit for bit, over score shapes
//      (rows 1 .. 155 by d 640 -> 640 / 1760 / 16384 / 3, and 1760 -> 640).
//   2. device::gemm with every transpose flag against cuda::matmul on the
//      host-transposed operands, bit for bit (the Phase A dispatch).
//   3. Memory: 300 distinct row counts through device::matmul leave nothing
//      live and at most MICROTORCH_POOL_MB cached (64 here, unless set).
//   4. A MoR model scored as `mtstudio score` scores it (one sequence per
//      call, NoGrad, no step window, lengths 1 .. n_ctx): logits on the
//      device (op set on) against the CPU within 1e-4, depth traces equal.
// CPU builds (or MICROTORCH_DEVICE unset): legs 1-4 are skipped; leg 4's
// forward still runs on the CPU.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/llama.hpp"

#ifdef MICROTORCH_CUDA
#include "cuda/matrix_ops.cuh"
#endif

using namespace microtorch;

namespace {

Matrix rnd(size_t r, size_t c, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = nd(g);
    return m;
}

size_t bits_differ(const Matrix& a, const Matrix& b) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols());
    size_t n = 0;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j) {
            const float x = a(i, j), y = b(i, j);
            n += std::memcmp(&x, &y, sizeof x) != 0;
        }
    return n;
}

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

bool on_cuda() { return device::get() == device::Device::CUDA; }

#ifdef MICROTORCH_CUDA
void test_bitwise() {
    size_t calls = 0, bits = 0;
    for (size_t M : {size_t(1), size_t(2), size_t(37), size_t(155)})
        for (auto [K, N] : std::vector<std::pair<size_t, size_t>>{{640, 640}, {640, 1760}, {640, 16384}, {640, 3},
                                                                 {1760, 640}}) {
            const Matrix a = rnd(M, K, 1 + static_cast<unsigned>(M)), b = rnd(K, N, 7 + static_cast<unsigned>(N));
            Matrix c1(M, N), c2(M, N);
            ::cuda::matmul(a, b, c1);
            device::pooled_matmul(a, b, c2);
            bits += bits_differ(c1, c2);
            ++calls;
        }
    std::printf("  1. pooled_matmul vs cuda::matmul: %zu shapes, %zu bits differ\n", calls, bits);
    CHECK(bits == 0);
}

void test_gemm_flags() {
    using device::Trans;
    const size_t M = 23, K = 64, N = 40;
    size_t bits = 0;
    for (int f = 0; f < 4; ++f) {
        const bool ta = f & 1, tb = f & 2;
        const Matrix A = ta ? rnd(K, M, 3) : rnd(M, K, 3), B = tb ? rnd(N, K, 5) : rnd(K, N, 5);
        const Matrix got = device::gemm(A, nullptr, ta ? Trans::T : Trans::N, B, nullptr, tb ? Trans::T : Trans::N);
        Matrix want(M, N);
        ::cuda::matmul(ta ? A.transpose() : A, tb ? B.transpose() : B, want);
        bits += bits_differ(got, want);
    }
    std::printf("  2. device::gemm (N/T flags) vs cuda::matmul on host transposes: %zu bits differ\n", bits);
    CHECK(bits == 0);
}

void test_memory() {
    const size_t cap = static_cast<size_t>(std::atol(std::getenv("MICROTORCH_POOL_MB"))) << 20;
    const Matrix w = rnd(640, 1760, 11);
    for (size_t M = 1; M <= 300; ++M) {
        const Matrix a = rnd(M, 640, static_cast<unsigned>(M));
        const Matrix c = device::matmul(a, w);
        CHECK(c.rows() == M && c.cols() == 1760);
    }
    const size_t live = device::pool_live_bytes(), cached = device::pool_cached_bytes();
    std::printf("  3. 300 row counts x (640 -> 1760): pool live %zu B, cached %.1f MiB (cap %zu MiB)\n", live,
                cached / 1048576.0, cap >> 20);
    CHECK(live == 0);
    CHECK(cached <= cap);
}
#endif

nn::LlamaConfig mor_cfg() {
    nn::LlamaConfig c;
    c.vocab = 97;
    c.d = 64;
    c.n_heads = 4;
    c.d_ff = 96;
    c.n_ctx = 64;
    c.n_layers = 4;
    c.rope_all_heads = true;
    c.recursion.prelude = 1;
    c.recursion.block = 2;
    c.recursion.repeats = 3;
    c.recursion.coda = 1;
    c.recursion.schedule = "mor";
    return c;
}

// test_mor.cpp's randomise: the router scaled up so that every depth occurs.
void randomise(nn::Module& m, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    for (auto& [name, p] : m.named_parameters()) {
        const bool norm = name.find("norm") != std::string::npos;
        const float sc = name.find("router") != std::string::npos ? 4.0f : 1.0f;
        for (size_t i = 0; i < p->data.rows(); ++i)
            for (size_t j = 0; j < p->data.cols(); ++j) p->data(i, j) = norm ? 1.0f + 0.2f * nd(g) : sc * nd(g);
    }
}

void test_mor_score() {
    const nn::LlamaConfig cfg = mor_cfg();
    nn::Llama m(cfg, 3);
    randomise(m, 21);
    m.eval();
    std::mt19937 g(9);
    const bool dev = on_cuda();
    const device::Device was = device::get();
    const bool ops_was = device::device_ops_enabled();
    double worst = 0;
    size_t lens = 0, depth_mismatch = 0, depth_counts[3] = {0, 0, 0};
    NoGrad ng;
    for (size_t L : {size_t(1), size_t(2), size_t(5), size_t(17), size_t(31), size_t(48), size_t(63), size_t(64)}) {
        std::vector<int> ids(L);
        for (int& t : ids) t = static_cast<int>(g() % cfg.vocab);
        const Matrix got = m.forward(ids)->data;
        const std::vector<int> dgot = m.last_depth;
        for (int d : dgot) ++depth_counts[d - 1];
        if (dev) {
            device::set(device::Device::CPU);
            device::set_device_ops(false);
            const Matrix want = m.forward(ids)->data;
            depth_mismatch += m.last_depth != dgot;
            worst = std::max(worst, rel(got, want));
            device::set(was);
            device::set_device_ops(ops_was);
        }
        CHECK(got.rows() == L && got.cols() == cfg.vocab);
        ++lens;
    }
    std::printf("  4. MoR 1 + [2] x {1..3} + 1, one sequence per call (as mtstudio score), %zu lengths 1 .. 64, "
                "depths %zu/%zu/%zu: %s\n",
                lens, depth_counts[0], depth_counts[1], depth_counts[2],
                dev ? (std::string(ops_was ? "device op set" : "device GEMMs") + " vs CPU, logits worst rel " +
                       std::to_string(worst) + ", depth traces differing " + std::to_string(depth_mismatch))
                          .c_str()
                    : "CPU only");
    CHECK(depth_counts[0] > 0 && depth_counts[1] > 0 && depth_counts[2] > 0);
    if (dev) {
        CHECK(depth_mismatch == 0);
        CHECK(worst < 1e-4);
    }
}

}  // namespace

int main() {
    // Leg 3's cap, before the pool's first use reads it (a value already set wins).
    setenv("MICROTORCH_POOL_MB", "64", 0);
    device::set_from_env();
    std::printf("Phase A GEMM through the device pool (%s)\n", on_cuda() ? "CUDA" : "CPU: device legs skipped");
#ifdef MICROTORCH_CUDA
    if (on_cuda()) {
        test_bitwise();
        test_gemm_flags();
        test_memory();
    }
#endif
    test_mor_score();
    std::printf("\n[PASS] pooled_gemm\n");
    return 0;
}
