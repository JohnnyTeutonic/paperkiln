// AttnRes receipts, mirroring python/attn_res_reference.py's self-test suite
// (the correctness anchor — TECH_TRANSFER item 1):
//   1. shape + uniform-at-init (w=0: depth-attention over one source is
//      exactly the identity)
//   2. THE EQUIVALENCE PIN: Block(S=1) == Full exactly — every block is
//      one layer and the partial-sum branch never fires, so the two code
//      paths must agree token-for-token (a transcription error has to be
//      made twice, identically, to pass)
//   3. Block(S=2) must DIFFER from Full (summing is not attending); if
//      they agree the block path is silently running the full path
//   4. partial final block (K3's 9-block layout analogue) runs
//   5. every parameter — pseudo-queries included — gets nonzero grad at
//      the zero-init point, exactly where a softmax gradient COULD die
//   6. finite differences through both wirings, off the uniform point
//   7. training receipt: the stack learns
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "microtorch/nn.hpp"
#include "microtorch/ops.hpp"

#include "check.hpp"

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

// A stack of L d->d MLP layers with deterministic per-layer seeds, so two
// stacks built with the same seeds have identical weights (the pin's
// substitute for weight sharing).
std::shared_ptr<nn::AttnResStack> make_stack(size_t L, size_t d, size_t block_size,
                                             unsigned seed_base = 100) {
    std::vector<std::shared_ptr<nn::Module>> owned;
    std::vector<std::function<Var(const Var&)>> fns;
    for (size_t i = 0; i < L; ++i) {
        auto mlp = std::make_shared<nn::MLP>(d, d, seed_base + static_cast<unsigned>(i));
        owned.push_back(mlp);
        fns.push_back([mlp](const Var& x) { return mlp->forward(x); });
    }
    return std::make_shared<nn::AttnResStack>(owned, fns, d, block_size);
}

double max_abs_diff(const Matrix& a, const Matrix& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j)
            worst = std::max(worst, std::fabs(static_cast<double>(a(i, j)) - b(i, j)));
    return worst;
}

double fd_vs_analytic(const std::function<float()>& forward, Var leaf, const Matrix& analytic,
                      float h = 1e-2f) {
    double worst = 0.0;
    for (size_t i = 0; i < leaf->data.rows(); ++i)
        for (size_t j = 0; j < leaf->data.cols(); ++j) {
            const float keep = leaf->data(i, j);
            NoGrad ng;
            leaf->data(i, j) = keep + h;
            const float up = forward();
            leaf->data(i, j) = keep - h;
            const float dn = forward();
            leaf->data(i, j) = keep;
            const double fd = (static_cast<double>(up) - dn) / (2.0 * h);
            const double a = analytic(i, j);
            const double err = std::abs(a - fd) / (1.0 + std::max(std::abs(a), std::abs(fd)));
            worst = std::max(worst, err);
        }
    return worst;
}

}  // namespace

int main() {
    const size_t T = 5, d = 16, L = 6;
    Var x = make_var(randn(T, d, 7));

    // 1. shape + uniform-at-init identity ------------------------------------
    {
        auto full = make_stack(L, d, /*block_size=*/0);
        Var y = full->forward(x);
        CHECK(y->data.rows() == T && y->data.cols() == d);
        // With w = 0 and ONE source, depth-attention must return the
        // source bit-for-bit (softmax over a single logit is 1.0).
        auto one = make_stack(0, d, 0);
        Var same = one->forward(x);
        const double di = max_abs_diff(same->data, x->data);
        std::printf("  [attnres] one-source identity max diff   %.3e\n", di);
        CHECK(di == 0.0);
    }

    // 2. the equivalence pin: Block(S=1) == Full -----------------------------
    {
        auto full = make_stack(L, d, 0);
        auto blk1 = make_stack(L, d, 1);
        const double diff = max_abs_diff(full->forward(x)->data, blk1->forward(x)->data);
        std::printf("  [attnres] Block(S=1) vs Full max diff    %.3e\n", diff);
        CHECK(diff < 1e-6);
    }

    // 3. Block(S=2) differs from Full ----------------------------------------
    {
        // Off the uniform point, where the wirings can actually diverge
        // (at w=0 both mix uniformly and agree by construction).
        auto full = make_stack(L, d, 0);
        auto blk2 = make_stack(L, d, 2);
        for (auto* s : {&full, &blk2})
            for (size_t i = 1; i < (*s)->w.size(); ++i)  // slot 0 is empty by design
                (*s)->w[i]->data = randn(1, d, 300 + static_cast<unsigned>(i), 0.5f);
        const double diff = max_abs_diff(full->forward(x)->data, blk2->forward(x)->data);
        std::printf("  [attnres] Block(S=2) vs Full max diff    %.3e\n", diff);
        CHECK(diff > 1e-6);
    }

    // 4. partial final block --------------------------------------------------
    {
        auto blk = make_stack(5, d, 2);  // L=5, S=2 -> blocks 2+2+1
        Var y = blk->forward(x);
        CHECK(y->data.rows() == T && y->data.cols() == d);
        std::printf("  [attnres] partial final block (L=5,S=2)  ok\n");
    }

    // 5. no gradient-dead parameters at zero init ----------------------------
    {
        for (size_t bs : {size_t(0), size_t(2)}) {
            auto stack = make_stack(L, d, bs);
            auto params = stack->parameters();
            zero_grad(params);
            Var y = stack->forward(x);
            backward(ops::mean(ops::mul(y, y)));
            size_t dead = 0;
            for (const auto& p : params) {
                double s = 0.0;
                if (p->grad.rows() != 0)
                    for (size_t i = 0; i < p->grad.rows(); ++i)
                        for (size_t j = 0; j < p->grad.cols(); ++j) s += std::fabs(p->grad(i, j));
                if (s == 0.0) ++dead;
            }
            std::printf("  [attnres] grad-dead params at init (bs=%zu)  %zu/%zu\n", bs, dead,
                        params.size());
            CHECK(dead == 0);
        }
    }

    // 6. finite differences through both wirings, off the uniform point ------
    {
        const double TOL = 5e-3;
        for (size_t bs : {size_t(0), size_t(2)}) {
            auto stack = make_stack(3, 8, bs, 200);
            for (size_t i = 1; i < stack->w.size(); ++i)  // slot 0 is empty by design
                stack->w[i]->data = randn(1, 8, 400 + static_cast<unsigned>(i), 0.5f);
            Var xs = make_var(randn(3, 8, 9), true);
            auto f = [&] { return ops::mean(stack->forward(xs))->data(0, 0); };
            zero_grad({xs});
            backward(ops::mean(stack->forward(xs)));
            const double ex = fd_vs_analytic(f, xs, xs->grad);
            std::printf("  [attnres] FD dx (bs=%zu)                  %.3e\n", bs, ex);
            CHECK(ex < TOL);
            // and through a pseudo-query
            Var wq = stack->w[1];
            zero_grad({xs, wq});
            backward(ops::mean(stack->forward(xs)));
            const double ew = fd_vs_analytic(f, wq, wq->grad);
            std::printf("  [attnres] FD dw1 (bs=%zu)                 %.3e\n", bs, ew);
            CHECK(ew < TOL);
        }
    }

    // 7. training receipt -----------------------------------------------------
    {
        auto stack = make_stack(3, d, 2, 500);
        Var X = make_var(randn(32, d, 20));
        Matrix ym(32, d);
        for (size_t i = 0; i < 32; ++i)
            for (size_t j = 0; j < d; ++j)
                ym(i, j) = 0.5f * X->data(i, (j + 1) % d) - 0.2f * X->data(i, j);
        Var y = make_var(std::move(ym));
        nn::AdamW opt(stack->parameters(), 3e-3f);
        float first = 0, last = 0;
        for (int s = 0; s < 150; ++s) {
            opt.zero_grad();
            Var err = ops::sub(stack->forward(X), y);
            Var loss = ops::mean(ops::mul(err, err));
            if (s == 0) first = loss->data(0, 0);
            last = loss->data(0, 0);
            backward(loss);
            opt.step();
        }
        std::printf("  [attnres] training loss %.4f -> %.4f (150 steps)\n", first, last);
        CHECK(last < 0.5f * first);
    }

    std::printf("[PASS] all AttnRes tests\n");
    return 0;
}
