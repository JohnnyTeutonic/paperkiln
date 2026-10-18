// The llama block's head plumbing on the device: ops::apply_rope,
// ops::slice_cols and ops::concat_cols with the device op set on (CUDA
// build, MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1) against the host
// loops, bit for bit: slice/concat are copies, and RoPE reads a host-built
// cos/sin table and rounds each product and sum as the host loop does
// (test_rope_heads' bit-equality leg depends on it). Legs: the fused qkv
// rotation alone (all heads and
// legacy first-head), the whole block chain (concat -> rope -> q/k/v
// slices -> per-head slices -> concat heads) forward and backward, an FD
// gradcheck, the chain under deferred downloads in a step window, and
// device memory flat over repeated steps. On a CPU-only build (or with the
// op set off) the legs run the host path against itself.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/ops.hpp"

using microtorch::make_var;
using microtorch::Var;
namespace ops = microtorch::ops;
namespace device = microtorch::device;

namespace {

Matrix randn(size_t r, size_t c, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(g);
    return m;
}

bool on_device() { return device::device_ops_enabled() && device::get() == device::Device::CUDA; }

// Positions of B stacked sequences of length sl (what the trainer passes).
std::vector<int> stacked_pos(size_t B, size_t sl) {
    std::vector<int> p;
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < sl; ++i) p.push_back(static_cast<int>(i));
    return p;
}

struct Diff {
    double worst = 0.0;  // max |a - b| / max |b| over the tensor
    size_t mismatched = 0, total = 0;
};

void cmp(const Matrix& a, const Matrix& b, Diff& d) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols());
    double scale = 0.0;
    for (size_t i = 0; i < b.rows(); ++i)
        for (size_t j = 0; j < b.cols(); ++j) scale = std::max(scale, (double)std::abs(b(i, j)));
    for (size_t i = 0; i < b.rows(); ++i)
        for (size_t j = 0; j < b.cols(); ++j) {
            const float x = a(i, j), y = b(i, j);
            d.worst = std::max(d.worst, std::abs((double)x - y) / (scale + 1e-30));
            d.mismatched += std::memcmp(&x, &y, sizeof x) != 0;
            ++d.total;
        }
}

// The llama attention plumbing (LlamaBlock::forward up to the attention
// call, with the attention itself replaced by the identity so only the
// ops under test are on the tape): returns the concatenated heads.
Var block_chain(const Var& q, const Var& k, const Var& v, const std::vector<int>& pos, size_t H, size_t dk,
                bool all_heads) {
    const size_t d = H * dk;
    Var qkv = ops::apply_rope(ops::concat_cols({q, k, v}), pos, 10000.0f, dk, all_heads);
    Var qq = ops::slice_cols(qkv, 0, d), kk = ops::slice_cols(qkv, d, 2 * d), vv = ops::slice_cols(qkv, 2 * d, 3 * d);
    std::vector<Var> heads;
    for (size_t h = 0; h < H; ++h) {
        Var qh = ops::slice_cols(qq, h * dk, (h + 1) * dk);
        Var kh = ops::slice_cols(kk, h * dk, (h + 1) * dk);
        Var vh = ops::slice_cols(vv, h * dk, (h + 1) * dk);
        heads.push_back(ops::add(ops::add(qh, kh), vh));
    }
    return ops::concat_cols(heads);
}

struct ChainOut {
    Matrix y, gq, gk, gv;
};

ChainOut run_chain(bool dev, size_t B, size_t sl, size_t H, size_t dk, bool all_heads, bool windowed) {
    device::set_device_ops(dev);
    const size_t T = B * sl, d = H * dk;
    const Matrix Q = randn(T, d, 21), K = randn(T, d, 22), V = randn(T, d, 23), G = randn(T, dk * H, 24);
    if (windowed) {
        device::set_step_residency(true);
        device::set_defer_downloads(true);
        device::step_begin();
    }
    Var q = make_var(Q, true), k = make_var(K, true), v = make_var(V, true);
    Var y = block_chain(q, k, v, stacked_pos(B, sl), H, dk, all_heads);
    microtorch::backward(ops::mean(ops::mul(y, make_var(G))));
    if (windowed) {
        device::step_end();  // materialize boundary
        device::set_defer_downloads(false);
        device::set_step_residency(false);
    }
    ChainOut o{y->data, q->grad, k->grad, v->grad};
    device::set_device_ops(true);
    return o;
}

// The fused rotation alone over [T, 3d], forward and backward.
void test_rope_alone(bool all_heads) {
    const size_t B = 4, sl = 256, H = 6, dk = 64, T = B * sl, d3 = 3 * H * dk;
    const Matrix X = randn(T, d3, 31), G = randn(T, d3, 32);
    const std::vector<int> pos = stacked_pos(B, sl);
    auto run = [&](bool dev, Matrix& y, Matrix& gx) {
        device::set_device_ops(dev);
        Var x = make_var(X, true);
        Var out = ops::apply_rope(x, pos, 10000.0f, dk, all_heads);
        microtorch::backward(ops::mean(ops::mul(out, make_var(G))));
        y = out->data, gx = x->grad;
    };
    Matrix y1, g1, y0, g0;
    run(true, y1, g1);
    run(false, y0, g0);
    device::set_device_ops(true);
    Diff dy, dg;
    cmp(y1, y0, dy), cmp(g1, g0, dg);
    std::printf("  rope alone, B=4 sl=256 H=6 dk=64 (all_heads=%d): y worst %.3e (%zu/%zu bits differ), "
                "grad worst %.3e (%zu/%zu)\n",
                all_heads, dy.worst, dy.mismatched, dy.total, dg.worst, dg.mismatched, dg.total);
    CHECK(dy.mismatched == 0 && dg.mismatched == 0);
}

void test_chain(bool all_heads, bool windowed) {
    const size_t B = 4, sl = 256, H = 6, dk = 64;
    const ChainOut a = run_chain(true, B, sl, H, dk, all_heads, windowed);
    const ChainOut b = run_chain(false, B, sl, H, dk, all_heads, false);
    Diff d;
    cmp(a.y, b.y, d), cmp(a.gq, b.gq, d), cmp(a.gk, b.gk, d), cmp(a.gv, b.gv, d);
    std::printf("  block chain%s (all_heads=%d): output + q/k/v grads worst %.3e (%zu/%zu bits differ)\n",
                windowed ? " under deferred downloads" : "", all_heads, d.worst, d.mismatched, d.total);
    CHECK(d.mismatched == 0);
}

// Slice and concat alone must be exact on the device.
void test_copies_exact() {
    const size_t T = 1024, d = 384;
    const Matrix A = randn(T, d, 41), Bm = randn(T, 128, 42), G = randn(T, d + 128, 43);
    auto run = [&](bool dev, Matrix& y, Matrix& ga, Matrix& gb) {
        device::set_device_ops(dev);
        Var a = make_var(A, true), b = make_var(Bm, true);
        Var cat = ops::concat_cols({ops::slice_cols(a, 64, 384), b, ops::slice_cols(a, 0, 64)});
        microtorch::backward(ops::mean(ops::mul(cat, make_var(G))));
        y = cat->data, ga = a->grad, gb = b->grad;
    };
    Matrix y1, a1, b1, y0, a0, b0;
    run(true, y1, a1, b1);
    run(false, y0, a0, b0);
    device::set_device_ops(true);
    Diff df;
    cmp(y1, y0, df), cmp(a1, a0, df), cmp(b1, b0, df);
    std::printf("  slice/concat copies: %zu/%zu bits differ\n", df.mismatched, df.total);
    CHECK(df.mismatched == 0);
}

// Variable::accumulate_cols, the slice backward: overlapping slices of one
// parent, plus a direct use of the parent so its grad already exists when
// the slices land (both the fresh and the add-into-range branches), in
// write-through and under deferred downloads. Same adds in the same tape
// order on both sides, so bit for bit.
void test_accumulate_cols() {
    const size_t T = 1024, d = 384;
    const Matrix A = randn(T, d, 71), G = randn(T, 100 + 150 + 64 + d, 72);
    auto run = [&](bool dev, bool windowed, Matrix& ga) {
        device::set_device_ops(dev);
        if (windowed) {
            device::set_step_residency(true);
            device::set_defer_downloads(true);
            device::step_begin();
        }
        Var a = make_var(A, true);
        Var cat = ops::concat_cols(
            {ops::slice_cols(a, 0, 100), ops::slice_cols(a, 50, 200), ops::slice_cols(a, 320, 384), a});
        microtorch::backward(ops::mean(ops::mul(cat, make_var(G))));
        if (windowed) {
            device::step_end();
            device::set_defer_downloads(false);
            device::set_step_residency(false);
        }
        ga = a->grad;
    };
    Matrix g0, g1, g2;
    run(false, false, g0);
    run(true, false, g1);
    run(true, true, g2);
    device::set_device_ops(true);
    Diff wt, df;
    cmp(g1, g0, wt), cmp(g2, g0, df);
    std::printf("  accumulate_cols (overlapping slices + direct use): write-through %zu/%zu, deferred %zu/%zu bits differ\n",
                wt.mismatched, wt.total, df.mismatched, df.total);
    CHECK(wt.mismatched == 0 && df.mismatched == 0);
}

void test_gradcheck() {
    const size_t B = 2, sl = 4, H = 2, dk = 4, d = H * dk;
    Var q = make_var(randn(B * sl, d, 51), true), k = make_var(randn(B * sl, d, 52), true),
        v = make_var(randn(B * sl, d, 53), true);
    const Var W = make_var(randn(B * sl, d, 54));
    const std::vector<int> pos = stacked_pos(B, sl);
    auto f = [&]() { return ops::mean(ops::mul(block_chain(q, k, v, pos, H, dk, true), W))->data(0, 0); };
    microtorch::backward(ops::mean(ops::mul(block_chain(q, k, v, pos, H, dk, true), W)));
    double worst = 0.0;
    for (Var leaf : {q, k, v})
        for (size_t i = 0; i < leaf->data.rows(); ++i)
            for (size_t j = 0; j < leaf->data.cols(); ++j) {
                const float keep = leaf->data(i, j), h = 1e-2f;
                microtorch::NoGrad ng;
                leaf->data(i, j) = keep + h;
                const double up = f();
                leaf->data(i, j) = keep - h;
                const double dn = f();
                leaf->data(i, j) = keep;
                const double fd = (up - dn) / (2.0 * h), a = leaf->grad(i, j);
                worst = std::max(worst, std::abs(a - fd) / (1.0 + std::max(std::abs(a), std::abs(fd))));
            }
    std::printf("  FD gradcheck, block chain: worst relative error %.3e\n", worst);
    CHECK(worst < 5e-3);
}

// Pool-held device memory must not grow over steps (the table is cached
// once; every temporary goes back to the pool). pool_live_bytes, not
// cudaMemGetInfo: the latter counts the pool's cache and other processes.
void test_memory_flat() {
    if (!on_device()) return;
    const size_t B = 4, sl = 256, H = 6, dk = 64;
    run_chain(true, B, sl, H, dk, true, true);
    const size_t before = device::pool_live_bytes();
    for (int s = 0; s < 30; ++s) run_chain(true, B, sl, H, dk, true, true);
    const size_t after = device::pool_live_bytes();
    std::printf("  pool live bytes over 30 steps: %zu -> %zu (cache %zu)\n", before, after,
                device::pool_cached_bytes());
    CHECK(after <= before);
}

// Wall time of the chain forward + backward at the chat training shape
// (batch 16 x 256, d384, 6 heads), host loops vs the device op set inside a
// deferred step window. Informational (no check): leaves are host tensors
// here, whereas in a resident step q/k/v arrive on the device from the
// projections.
void bench_chain() {
    if (!on_device()) return;
    const size_t B = 16, sl = 256, H = 6, dk = 64, T = B * sl, d = H * dk;
    const Matrix Q = randn(T, d, 61), K = randn(T, d, 62), V = randn(T, d, 63), G = randn(T, d, 64);
    const std::vector<int> pos = stacked_pos(B, sl);
    auto time = [&](bool dev, bool windowed) {
        device::set_device_ops(dev);
        double best = 1e30;
        for (int r = 0; r < 6; ++r) {
            Var q = make_var(Q, true), k = make_var(K, true), v = make_var(V, true), g = make_var(G);
            const auto t0 = std::chrono::steady_clock::now();
            if (windowed) {
                device::set_step_residency(true);
                device::set_defer_downloads(true);
                device::step_begin();
            }
            microtorch::backward(ops::mean(ops::mul(block_chain(q, k, v, pos, H, dk, true), g)));
            if (windowed) {
                device::step_end();
                device::set_defer_downloads(false);
                device::set_step_residency(false);
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (r > 0) best = std::min(best, ms);  // r = 0 warms up
        }
        return best;
    };
    const double host = time(false, false), wt = time(true, false), win = time(true, true);
    device::set_device_ops(true);
    std::printf("  chain fwd+bwd at B=16 sl=256 d384 H6 (best of 5): host %.1f ms, device write-through %.1f ms, "
                "device deferred %.1f ms\n",
                host, wt, win);
}

}  // namespace

int main() {
    device::set_from_env();  // MICROTORCH_DEVICE / _DEVICE_OPS select the kernels
    std::printf("device rope/slice/concat%s\n", on_device() ? " (device op set)" : " (host path)");
    const bool dev = on_device();
    test_copies_exact();
    test_accumulate_cols();
    device::set_device_ops(dev);
    test_rope_alone(true);
    test_rope_alone(false);
    test_chain(true, false);
    test_chain(false, false);
    if (dev) test_chain(true, true);
    device::set_device_ops(dev);
    test_gradcheck();
    test_memory_flat();
    bench_chain();
    std::printf("test_device_rope: all checks passed\n");
    return 0;
}
