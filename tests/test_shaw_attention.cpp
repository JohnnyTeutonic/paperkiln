// ops::shaw_attention: Shaw-style relative positions (Shaw, Uszkoreit and Vaswani 2018, arXiv 1803.02155,
// eqs. 3-4; keramos IR_SPEC 5.15 attention.relative_kv, revision 1.4). LOCAL_LLM_STACK.md section 5.4.
//   1. zero tables are plain causal attention (ops::fused_attention), to float rounding
//   2. the op against the formula in binary64 (clipped distances, two sequences of seq_len rows, vectors
//      shared by the heads and one set per head, keys only)
//   3. the clamp: rows i < K never read table row K, bit for bit
//   4. gradients of q, k, v and both tables against central finite differences, shared and per-head
//      tables and keys only; a head never writes another head's columns
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "check.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

namespace {
std::mt19937 g(1803);

Var rnd(size_t r, size_t c, bool grad, float sd = 1.0f) {
    std::normal_distribution<float> nd(0.0f, sd);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = nd(g);
    return make_var(m, grad);
}

// Shaw's eqs. 3-4 in binary64 for one head: e_ij = (q_i . (k_j + rk[n])) * scale, n = min(i - j, K),
// softmax over the visible keys of the block, z_i = sum_j a_ij (v_j + rv[n]).
std::vector<double> reference(const Matrix& q, const Matrix& k, const Matrix& v, const Matrix& rk, const Matrix* rv,
                              size_t col0, size_t K, double scale, size_t sl) {
    const size_t T = q.rows(), dk = q.cols(), dv = v.cols();
    std::vector<double> y(T * dv, 0.0);
    for (size_t i = 0; i < T; ++i) {
        const size_t b0 = (i / sl) * sl;
        std::vector<double> e(i - b0 + 1);
        double mx = -1e300;
        for (size_t j = b0; j <= i; ++j) {
            const size_t n = std::min(i - j, K);
            double s = 0.0;
            for (size_t c = 0; c < dk; ++c) s += double(q(i, c)) * (double(k(j, c)) + double(rk(n, col0 + c)));
            e[j - b0] = s * scale;
            mx = std::max(mx, e[j - b0]);
        }
        double z = 0.0;
        for (auto& x : e) z += (x = std::exp(x - mx));
        for (size_t j = b0; j <= i; ++j) {
            const size_t n = std::min(i - j, K);
            for (size_t c = 0; c < dv; ++c)
                y[i * dv + c] += e[j - b0] / z * (double(v(j, c)) + (rv ? double((*rv)(n, col0 + c)) : 0.0));
        }
    }
    return y;
}

double gradcheck(const std::vector<Var>& leaves, const std::function<Var()>& loss) {
    for (const Var& l : leaves) l->grad = Matrix();
    backward(loss());
    double worst = 0.0;
    for (const Var& l : leaves)
        for (size_t i = 0; i < l->data.rows(); ++i)
            for (size_t j = 0; j < l->data.cols(); ++j) {
                const float keep = l->data(i, j), h = 1e-2f;
                NoGrad ng;
                l->data(i, j) = keep + h;
                const double up = loss()->data(0, 0);
                l->data(i, j) = keep - h;
                const double dn = loss()->data(0, 0);
                l->data(i, j) = keep;
                const double fd = (up - dn) / (2.0 * h), a = l->grad.rows() ? l->grad(i, j) : 0.0;
                worst = std::max(worst, std::abs(a - fd) / (1.0 + std::max(std::abs(a), std::abs(fd))));
            }
    return worst;
}
}  // namespace

int main() {
    const size_t T = 12, dk = 8, K = 3, sl = 6;
    const float sc = 0.35f;

    // 1. zero tables ------------------------------------------------------------------------------------
    Var q = rnd(T, dk, true), k = rnd(T, dk, true), v = rnd(T, dk, true);
    {
        Var z = make_var(Matrix(K + 1, dk));
        Var a = ops::fused_attention(q, k, v, sc, sl, true);
        Var b = ops::shaw_attention(q, k, v, z, z, 0, K, sc, sl);
        float d = 0;
        for (size_t i = 0; i < T; ++i)
            for (size_t c = 0; c < dk; ++c) d = std::max(d, std::fabs(a->data(i, c) - b->data(i, c)));
        CHECK(d < 1e-6f);
        std::printf("1. zero tables = causal attention within %.2e\n", d);
    }

    // 2. the formula in binary64 --------------------------------------------------------------------------
    Var rk = rnd(K + 1, 2 * dk, true, 0.5f), rv = rnd(K + 1, 2 * dk, true, 0.5f);
    Var rk1 = rnd(K + 1, dk, true, 0.5f), rv1 = rnd(K + 1, dk, true, 0.5f);
    double worst_ref = 0.0;
    struct Case {
        Var rk, rv;
        size_t col0;
    };
    const std::vector<Case> cases = {{rk1, rv1, 0}, {rk, rv, dk}, {rk, nullptr, 0}, {rk1, nullptr, 0}};
    for (const auto& cs : cases) {
        Var y = ops::shaw_attention(q, k, v, cs.rk, cs.rv, cs.col0, K, sc, sl);
        const auto ref = reference(q->data, k->data, v->data, cs.rk->data, cs.rv ? &cs.rv->data : nullptr, cs.col0, K,
                                   double(sc), sl);
        for (size_t i = 0; i < T; ++i)
            for (size_t c = 0; c < dk; ++c)
                worst_ref = std::max(worst_ref, std::abs(double(y->data(i, c)) - ref[i * dk + c]));
    }
    CHECK(worst_ref < 2e-6);
    std::printf("2. against Shaw's eqs. 3-4 in binary64 (shared, per head, keys only): worst %.2e\n", worst_ref);

    // 3. the clamp ----------------------------------------------------------------------------------------
    {
        Var a = ops::shaw_attention(q, k, v, rk1, rv1, 0, K, sc, 0);
        Matrix tk = rk1->data, tv = rv1->data;
        for (size_t c = 0; c < dk; ++c) {
            tk(K, c) += 1.0f;
            tv(K, c) -= 1.0f;
        }
        Var b = ops::shaw_attention(q, k, v, make_var(tk), make_var(tv), 0, K, sc, 0);
        bool same_below = true, moved_above = false;
        for (size_t i = 0; i < T; ++i)
            for (size_t c = 0; c < dk; ++c) {
                if (i < K) same_below = same_below && a->data(i, c) == b->data(i, c);
                else moved_above = moved_above || a->data(i, c) != b->data(i, c);
            }
        CHECK(same_below && moved_above);
        std::printf("3. rows i < K = %zu never read table row K (bit for bit); later rows do\n", K);
    }

    // 4. gradients ----------------------------------------------------------------------------------------
    Var R = rnd(T, dk, false);
    double worst = 0.0;
    for (const auto& cs : cases) {
        std::vector<Var> leaves = {q, k, v, cs.rk};
        if (cs.rv) leaves.push_back(cs.rv);
        auto loss = [&]() {
            return ops::scale(ops::mean(ops::mul(ops::shaw_attention(q, k, v, cs.rk, cs.rv, cs.col0, K, sc, sl), R)),
                              static_cast<float>(T * dk));
        };
        worst = std::max(worst, gradcheck(leaves, loss));
        // The other head's columns of a per-head table get no gradient.
        if (cs.rk->data.cols() == 2 * dk)
            for (size_t n = 0; n <= K; ++n)
                for (size_t c = 0; c < dk; ++c) CHECK(cs.rk->grad(n, (cs.col0 + dk + c) % (2 * dk)) == 0.0f);
    }
    CHECK(worst < 2e-3);
    std::printf("4. gradients of q, k, v and the tables against central differences (4 layouts): worst %.2e\n", worst);
    std::printf("SHAW-OK 4/4\n");
    return 0;
}
