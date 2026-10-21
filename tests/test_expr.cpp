// Runtime-defined ops (the IR's expr op; microtorch/expr.hpp) receipts:
//   1. the parser accepts the grammar and rejects everything else, naming
//      the column
//   2. values: each primitive, precedence and constant folding against
//      hand-computed f32 values, bit for bit; relu and sigmoid equal the
//      relu and sigmoid ops
//   3. shapes: a result without inputs fills [T, w] with no gradient; the
//      input count and shapes are checked
//   4. gradients: every primitive's backward against central finite
//      differences on random inputs, through the expressions papers use
//      (squared ReLU, GELU's sigmoid form, leaky ReLU, softsign, a
//      layernorm written out, a two-input gate)
//   5. flex: act "expr:relu(x)" equals act "relu" bit for bit, a squared-
//      ReLU flex model trains (loss falls), its IR has the expr node, and a
//      bad expression fails at construction
// tests/test_ir_expr.py compares ops::expr with the numpy reference
// (tools/ir_expr.py) through tools/expr_eval.cpp.
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../tools/parity_model.hpp"
#include "check.hpp"
#include "microtorch/expr.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

namespace {

Matrix randn(size_t r, size_t c, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(gen);
    return m;
}

// Random entries at least 0.2 from 0, so a finite difference never steps
// across the kink of abs, relu, max or where.
Matrix away_from_zero(size_t r, size_t c, unsigned seed) {
    Matrix m = randn(r, c, seed);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = (m(i, j) < 0 ? -0.2f : 0.2f) + m(i, j);
    return m;
}

// The column the parser names in its message, or 0 when it accepts.
size_t reject_column(const std::string& text, size_t n_inputs) {
    try {
        Expr::parse(text, n_inputs);
    } catch (const std::invalid_argument& e) {
        const std::string m = e.what();
        const size_t at = m.find(" at column ");
        CHECK(at != std::string::npos);
        return std::stoul(m.substr(at + 11));
    }
    return 0;
}

Var leaf(const Matrix& m) { return make_var(m, true); }

bool same_bits(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j)
            if (!(a(i, j) == b(i, j) || (std::isnan(a(i, j)) && std::isnan(b(i, j))))) return false;
    return true;
}

}  // namespace

int main() {
    // 1. parser ---------------------------------------------------------------
    {
        const std::vector<std::pair<std::string, size_t>> good = {
            {"x", 1},
            {"a", 2},
            {"  x * sigmoid(1.702*x) ", 1},
            {"max(x, 0)^2", 1},
            {"where(x, x, 0.01*x)", 1},
            {"x/(1+abs(x))", 1},
            {"(x-rowmean(x))/sqrt(rowmean((x-rowmean(x))^2)+1e-5)", 1},
            {"x*tanh(a)", 2},
            {"where(b, x, a) - rowmax(x) + rowsum(a)", 3},
            {"relu(a)*x", 2},  // ReGLU, as papers/llm_extract.py writes it
            {"-x^-2 + 2^3^2 - -1", 1},
            {".5 + 2. + 1.5E+3 + 1e-5 + pi", 1},
            {"exp(log(sqrt(erf(min(x, 1)))))", 1},
            {"relu(x)\t", 1},
            {"3", 1},
        };
        for (const auto& [text, n] : good) {
            if (reject_column(text, n) != 0) std::fprintf(stderr, "rejected: %s\n", text.c_str());
            CHECK(reject_column(text, n) == 0);
        }
        // (text, inputs, the column the message names)
        const std::vector<std::tuple<std::string, size_t, size_t>> bad = {
            {"", 1, 1},                // empty
            {"x +", 1, 4},             // operand missing at the end
            {"x + * 2", 1, 5},         // operand missing
            {"2x", 1, 2},              // no implicit multiplication
            {"x ** 2", 1, 4},          // ** is not a token pair
            {"y", 1, 1},               // unknown name
            {"a", 1, 1},               // only one input: x
            {"b", 2, 1},               // x and a only
            {"c", 3, 1},               // x, a, b only
            {"foo(x)", 1, 1},          // unknown function
            {"exp x", 1, 5},           // a function needs '('
            {"max(x)", 1, 1},          // arity
            {"where(x, 1)", 1, 1},     // arity
            {"exp(x, x)", 1, 1},       // arity
            {"(x", 1, 3},              // unclosed
            {"x)", 1, 2},              // stray ')'
            {"1e", 1, 1},              // exponent without digits
            {".", 1, 1},               // malformed number
            {"1e999", 1, 1},           // not a finite float
            {"x % 2", 1, 3},           // unknown character
            {"x = 1", 1, 3},           // unknown character
            {"pi(x)", 1, 1},           // pi is a constant
            {"sigmoid", 1, 8},         // a function name alone
            {"max(x,,0)", 1, 7},       // empty argument
        };
        for (const auto& [text, n, col] : bad) {
            const size_t got = reject_column(text, n);
            if (got != col) std::fprintf(stderr, "\"%s\": column %zu, want %zu\n", text.c_str(), got, col);
            CHECK(got == col);
        }
        std::string deep(300, '(');
        deep += "x" + std::string(300, ')');
        CHECK(reject_column(deep, 1) != 0);  // nesting limit, not a stack overflow
        bool threw = false;
        try {
            Expr::parse("x", 0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
        std::printf("1. parser: %zu accepted, %zu rejected at the right column, nesting capped\n", good.size(),
                    bad.size());
    }

    // 2. values ---------------------------------------------------------------
    {
        const size_t T = 3, w = 4;
        Matrix X(T, w), B(T, w);
        const float xs[] = {-2.5f, -1.0f, -0.25f, 0.0f, 0.3f, 1.0f, 2.0f, 3.7f, -0.0f, 0.5f, -3.0f, 1e-3f};
        for (size_t k = 0; k < T * w; ++k) {
            X(k / w, k % w) = xs[k];
            B(k / w, k % w) = 0.37f * static_cast<float>(k) - 1.9f;
        }
        const Var x = make_var(X), b = make_var(B);
        size_t n_checked = 0;
        auto pin = [&](const std::string& text, const std::function<float(size_t, size_t)>& want,
                       size_t inputs = 1) {
            const Var y = inputs == 1 ? ops::expr({x}, text) : ops::expr({x, b}, text);
            CHECK(y->data.rows() == T && y->data.cols() == w);
            for (size_t i = 0; i < T; ++i)
                for (size_t j = 0; j < w; ++j) {
                    const float got = y->data(i, j), ref = want(i, j);
                    const bool ok = got == ref || (std::isnan(got) && std::isnan(ref));
                    if (!ok) std::fprintf(stderr, "%s at (%zu,%zu): %.9g, want %.9g\n", text.c_str(), i, j, got, ref);
                    CHECK(ok);
                }
            ++n_checked;
        };
        auto X_ = [&](size_t i, size_t j) { return X(i, j); };
        pin("x", X_);
        pin("x + 0.5", [&](size_t i, size_t j) { return X(i, j) + 0.5f; });
        pin("0.5 - x", [&](size_t i, size_t j) { return 0.5f - X(i, j); });
        pin("x - 0.1", [&](size_t i, size_t j) { return X(i, j) - 0.1f; });
        pin("x * 1.702", [&](size_t i, size_t j) { return X(i, j) * 1.702f; });
        pin("x / 3", [&](size_t i, size_t j) { return X(i, j) / 3.0f; });
        pin("3 / x", [&](size_t i, size_t j) { return 3.0f / X(i, j); });
        // powf itself: a constant 2 lets the compiler turn pow into x * x.
        volatile float two_v = 2.0f;
        const float two = two_v;
        pin("x ^ 2", [&](size_t i, size_t j) { return std::pow(X(i, j), two); });
        pin("2 ^ x", [&](size_t i, size_t j) { return std::pow(two, X(i, j)); });
        pin("-x", [&](size_t i, size_t j) { return -X(i, j); });
        pin("exp(x)", [&](size_t i, size_t j) { return std::exp(X(i, j)); });
        pin("log(x)", [&](size_t i, size_t j) { return std::log(X(i, j)); });  // NaN and -inf included
        pin("sqrt(x)", [&](size_t i, size_t j) { return std::sqrt(X(i, j)); });
        pin("tanh(x)", [&](size_t i, size_t j) { return std::tanh(X(i, j)); });
        pin("erf(x)", [&](size_t i, size_t j) { return std::erf(X(i, j)); });
        pin("abs(x)", [&](size_t i, size_t j) { return std::fabs(X(i, j)); });
        pin("sigmoid(x)", [&](size_t i, size_t j) { return 1.0f / (1.0f + std::exp(-X(i, j))); });
        pin("relu(x)", [&](size_t i, size_t j) { return std::max(0.0f, X(i, j)); });
        pin("max(x, 0.5)", [&](size_t i, size_t j) { return X(i, j) >= 0.5f ? X(i, j) : 0.5f; });
        pin("min(x, 0.5)", [&](size_t i, size_t j) { return X(i, j) <= 0.5f ? X(i, j) : 0.5f; });
        pin("where(x, x, 0.01*x)", [&](size_t i, size_t j) { return X(i, j) > 0 ? X(i, j) : 0.01f * X(i, j); });
        pin("rowsum(x)", [&](size_t i, size_t) {
            float s = 0;
            for (size_t k = 0; k < w; ++k) s += X(i, k);
            return s;
        });
        pin("rowmean(x)", [&](size_t i, size_t) {
            float s = 0;
            for (size_t k = 0; k < w; ++k) s += X(i, k);
            return s / static_cast<float>(w);
        });
        pin("rowmax(x)", [&](size_t i, size_t) {
            float m = X(i, 0);
            for (size_t k = 1; k < w; ++k) m = std::max(m, X(i, k));
            return m;
        });
        pin("x - rowmean(x)", [&](size_t i, size_t j) {
            float s = 0;
            for (size_t k = 0; k < w; ++k) s += X(i, k);
            return X(i, j) - s / static_cast<float>(w);
        });
        pin("rowsum(2)", [&](size_t, size_t) { return 2.0f * static_cast<float>(w); });
        // two inputs: x the first, a the second
        pin("a", [&](size_t i, size_t j) { return B(i, j); }, 2);
        pin("x * tanh(a)", [&](size_t i, size_t j) { return X(i, j) * std::tanh(B(i, j)); }, 2);
        pin("min(x, a)", [&](size_t i, size_t j) { return X(i, j) <= B(i, j) ? X(i, j) : B(i, j); }, 2);
        pin("where(a, x, a)", [&](size_t i, size_t j) { return B(i, j) > 0 ? X(i, j) : B(i, j); }, 2);
        pin("x / a", [&](size_t i, size_t j) { return X(i, j) / B(i, j); }, 2);
        // precedence, associativity, folding (every constant step in f32)
        pin("1 + 2 * 3", [](size_t, size_t) { return 7.0f; });
        pin("(1 + 2) * 3", [](size_t, size_t) { return 9.0f; });
        pin("8 / 2 / 2", [](size_t, size_t) { return 2.0f; });
        pin("1 - 2 - 3", [](size_t, size_t) { return -4.0f; });
        pin("2 ^ 3 ^ 2", [](size_t, size_t) { return 512.0f; });
        pin("-2 ^ 2", [](size_t, size_t) { return -4.0f; });
        pin("2 ^ -1", [](size_t, size_t) { return 0.5f; });
        pin("-x ^ 2", [&](size_t i, size_t j) { return -std::pow(X(i, j), two); });
        pin("2 * pi", [](size_t, size_t) { return 2.0f * static_cast<float>(M_PI); });
        pin("0.1 + 0.2", [](size_t, size_t) { return 0.1f + 0.2f; });
        pin("sqrt(2) * x", [&](size_t i, size_t j) { return std::sqrt(2.0f) * X(i, j); });
        pin("where(-1, 1, 2)", [](size_t, size_t) { return 2.0f; });
        pin("1e-5", [](size_t, size_t) { return 1e-5f; });
        // the IR's relu and sigmoid ops are these functions, bit for bit
        CHECK(same_bits(ops::expr({x}, "relu(x)")->data, ops::relu(x)->data));
        CHECK(same_bits(ops::expr({x}, "sigmoid(x)")->data, ops::sigmoid(x)->data));
        std::printf("2. values: %zu expressions bit-exact against hand-computed f32\n", n_checked);
    }

    // 3. shapes ---------------------------------------------------------------
    {
        const Var x = leaf(randn(2, 5, 1));
        const Var c = ops::expr({x}, "2 * pi");
        CHECK(c->data.rows() == 2 && c->data.cols() == 5 && !c->requires_grad);
        CHECK(ops::expr({x}, "x * 1")->requires_grad);
        bool threw = false;
        try {
            ops::expr({x, make_var(randn(2, 4, 2))}, "x + a");  // shapes differ
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        const Expr one = Expr::parse("x", 1);
        try {
            ops::expr({x, x}, one);  // parsed for one input
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
        std::printf("3. shapes: constants fill [T, w] without a gradient; inputs checked\n");
    }

    // 4. gradients ------------------------------------------------------------
    {
        const size_t T = 3, w = 5;
        const std::vector<std::pair<std::string, size_t>> exprs = {
            {"max(x, 0)^2", 1},          // squared ReLU
            {"x*sigmoid(1.702*x)", 1},   // GELU's sigmoid approximation
            {"where(x, x, 0.01*x)", 1},  // leaky ReLU
            {"x/(1+abs(x))", 1},         // softsign
            {"(x-rowmean(x))/sqrt(rowmean((x-rowmean(x))^2)+1e-5)", 1},  // layernorm, no weight
            {"x*tanh(a)", 2},  // a two-input gate
            {"exp(-x^2) + log(1 + x^2)", 1},
            {"erf(x/sqrt(2)) * 0.5 * x", 1},
            {"min(x, 0.5*x) - rowmax(x)", 1},
            {"sqrt(abs(x) + 1) + abs(x)^1.5", 1},
            {"2^x - pi/(1 + x^2) + relu(x)", 1},
            {"where(a, x^2, -x) + max(x, a) * min(a, 1)", 2},
            {"(x - a) / (1 + exp(rowsum(x*a)/5))", 2},
            {"relu(a)*x", 2},  // ReGLU
        };
        double worst_all = 0;
        for (const auto& [text, n_in] : exprs) {
            std::vector<Var> ins;
            for (size_t k = 0; k < n_in; ++k) ins.push_back(leaf(away_from_zero(T, w, 100 + 7 * k)));
            const Matrix R = randn(T, w, 55);
            const Var r = make_var(R);
            const Expr e = Expr::parse(text, n_in);
            // loss = sum(expr(ins) .* R), as mean * T*w
            auto loss_of = [&]() {
                return ops::scale(ops::mean(ops::mul(ops::expr(ins, e), r)), static_cast<float>(T * w));
            };
            Var loss = loss_of();
            backward(loss);
            const float h = 1e-2f;
            double worst = 0;
            for (const Var& p : ins) {
                CHECK(p->grad.rows() == T);
                for (size_t i = 0; i < T; ++i)
                    for (size_t j = 0; j < w; ++j) {
                        const float keep = p->data(i, j);
                        NoGrad ng;
                        p->data(i, j) = keep + h;
                        const double up = loss_of()->data(0, 0);
                        p->data(i, j) = keep - h;
                        const double dn = loss_of()->data(0, 0);
                        p->data(i, j) = keep;
                        const double fd = (up - dn) / (2.0 * h);
                        const double a = p->grad(i, j);
                        worst = std::max(worst, std::abs(a - fd) / (1.0 + std::max(std::abs(a), std::abs(fd))));
                    }
            }
            std::printf("   %-58s rel err %.2e\n", text.c_str(), worst);
            CHECK(worst < 2e-3);
            worst_all = std::max(worst_all, worst);
        }
        std::printf("4. gradients: %zu expressions FD-checked, worst %.2e\n", exprs.size(), worst_all);
    }

    // 5. flex -----------------------------------------------------------------
    {
        const size_t V = 31, d = 32, H = 4, T = 16;
        auto ids_mod = [](size_t n, size_t vocab, unsigned seed) {
            std::mt19937 g(seed);
            std::vector<int> v(n);
            for (auto& t : v) t = static_cast<int>(g() % vocab);
            return v;
        };
        parity::FlexConfig fc;
        fc.vocab = V;
        fc.d = d;
        fc.n_heads = H;
        fc.n_ctx = T;
        fc.d_ff = 4 * d;
        fc.act = "relu";
        parity::FlexLM relu(fc, 13);
        fc.act = "expr:relu(x)";
        parity::FlexLM as_expr(fc, 13);
        CHECK(relu.parameter_count() == as_expr.parameter_count());
        const auto ids = ids_mod(T, V, 9);
        CHECK(same_bits(relu.forward(ids)->data, as_expr.forward(ids)->data));

        fc.act = "expr:max(x, 0)^2";  // squared ReLU (Primer, arXiv 2109.08668)
        fc.norm = "rmsnorm";
        parity::FlexLM m(fc, 11);
        nn::AdamW opt(m.parameters(), 3e-3f);
        const auto seq = ids_mod(T + 1, V, 5);
        std::vector<int> x(seq.begin(), seq.end() - 1), y(seq.begin() + 1, seq.end());
        float first = 0, last = 0;
        for (int step = 0; step < 30; ++step) {
            Var loss = ops::cross_entropy(m.forward(x), y);
            if (step == 0) first = loss->data(0, 0);
            last = loss->data(0, 0);
            opt.zero_grad();
            backward(loss);
            opt.step();
        }
        CHECK(last < first * 0.8f);

        const auto fx = m.family_export();
        CHECK(fx.has_value() && fx->arch == "paperkiln");
        size_t n_expr = 0, n_relu = 0;
        for (const auto& node : fx->ir["block"]["nodes"]) {
            if (node["op"] == "expr") {
                ++n_expr;
                CHECK(node["expr"] == "max(x, 0)^2");
                CHECK(node["in"] == ir::json::array({"u"}) && node["out"] == "u");
            }
            if (node["op"] == "relu") ++n_relu;
        }
        CHECK(n_expr == 1 && n_relu == 0);

        bool threw = false;
        fc.act = "expr:max(x";
        try {
            parity::FlexLM bad(fc, 1);
        } catch (const std::invalid_argument& e) {
            threw = std::string(e.what()).find("column") != std::string::npos;
        }
        CHECK(threw);
        std::printf("5. flex: expr:relu(x) == relu bitwise; squared ReLU trains %.3f -> %.3f; IR has the expr node\n",
                    first, last);
    }

    std::printf("EXPR-OK 5/5\n");
    return 0;
}
