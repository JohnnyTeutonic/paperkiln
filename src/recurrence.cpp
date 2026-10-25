#include "microtorch/recurrence.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include "microtorch/device.hpp"
#include "microtorch/ops.hpp"

namespace microtorch {

namespace {

using Expr = Recurrence::Expr;
using Def = Recurrence::Def;
using json = nlohmann::ordered_json;

[[noreturn]] void refuse(int rule, const std::string& msg) {
    throw std::invalid_argument("IR_SPEC 6.10 (rule " + std::to_string(rule) + "): " + msg);
}

struct Fn {
    const char* name;
    Expr::Kind kind;
    size_t arity;
};
constexpr Fn kFns[] = {
    {"exp", Expr::EXP, 1},   {"log", Expr::LOG, 1}, {"sqrt", Expr::SQRT, 1}, {"tanh", Expr::TANH, 1},
    {"sigmoid", Expr::SIGMOID, 1}, {"erf", Expr::ERF, 1}, {"abs", Expr::ABS, 1}, {"relu", Expr::RELU, 1},
    {"max", Expr::MAX, 2},   {"min", Expr::MIN, 2}, {"where", Expr::WHERE, 3},
};
constexpr float kPi = 3.14159265358979323846f;
constexpr size_t kMaxDepth = 200;

bool reserved(const std::string& n) {
    for (const Fn& f : kFns)
        if (n == f.name) return true;
    return n == "pi" || n == "sum" || n == "rowsum" || n == "rowmean" || n == "rowmax" || n == "per_layer";
}

bool identifier(const std::string& n) {
    if (n.empty() || !(std::isalpha(static_cast<unsigned char>(n[0])) || n[0] == '_')) return false;
    for (char c : n)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
    return true;
}

enum class Tok { NUM, NAME, OP, LPAREN, RPAREN, LBRACK, RBRACK, COMMA, END };
struct Token {
    Tok tok;
    std::string text;
    size_t column;
};

class Parser {
public:
    Parser(const std::string& text, const std::string& where) : s_(text), where_(where) { lex(); }

    // name '[' index (',' index)* ']'
    std::pair<std::string, std::vector<std::string>> decl() {
        const Token t = expect(Tok::NAME, "a name");
        expect(Tok::LBRACK, "'['");
        std::vector<std::string> idx{expect(Tok::NAME, "an index name").text};
        while (peek().tok == Tok::COMMA) {
            next();
            idx.push_back(expect(Tok::NAME, "an index name").text);
        }
        expect(Tok::RBRACK, "']'");
        return {t.text, idx};
    }
    Expr assignment_rhs() {
        const Token t = next();
        if (t.tok != Tok::OP || t.text != "=") fail("expected '=', got '" + t.text + "'", t.column);
        return expr(0);
    }
    void end() {
        if (peek().tok != Tok::END) fail("expected an operator or the end, got '" + peek().text + "'", peek().column);
    }

private:
    [[noreturn]] void fail(const std::string& what, size_t column) const {
        refuse(31, where_ + ": " + what + " at column " + std::to_string(column) + " in \"" + s_ + "\"");
    }

    void lex() {
        size_t i = 0;
        while (i < s_.size()) {
            const char ch = s_[i];
            const size_t col = i + 1;
            if (ch == ' ' || ch == '\t') {
                ++i;
            } else if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '.') {
                size_t j = i, mant = 0;
                while (j < s_.size() && std::isdigit(static_cast<unsigned char>(s_[j]))) ++j, ++mant;
                if (j < s_.size() && s_[j] == '.') {
                    ++j;
                    while (j < s_.size() && std::isdigit(static_cast<unsigned char>(s_[j]))) ++j, ++mant;
                }
                if (mant == 0) fail("malformed number", col);
                if (j < s_.size() && (s_[j] == 'e' || s_[j] == 'E')) {
                    size_t k = j + 1;
                    if (k < s_.size() && (s_[k] == '+' || s_[k] == '-')) ++k;
                    const size_t d0 = k;
                    while (k < s_.size() && std::isdigit(static_cast<unsigned char>(s_[k]))) ++k;
                    if (k == d0) fail("malformed number (exponent without digits)", col);
                    j = k;
                }
                const std::string lit = s_.substr(i, j - i);
                if (!std::isfinite(std::strtof(lit.c_str(), nullptr))) fail("number " + lit + " is out of float range", col);
                toks_.push_back({Tok::NUM, lit, col});
                i = j;
            } else if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_') {
                size_t j = i;
                while (j < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[j])) || s_[j] == '_')) ++j;
                toks_.push_back({Tok::NAME, s_.substr(i, j - i), col});
                i = j;
            } else if (std::string("+-*/^=").find(ch) != std::string::npos) {
                toks_.push_back({Tok::OP, std::string(1, ch), col});
                ++i;
            } else if (ch == '(' || ch == ')' || ch == '[' || ch == ']' || ch == ',') {
                const Tok k = ch == '(' ? Tok::LPAREN : ch == ')' ? Tok::RPAREN : ch == '[' ? Tok::LBRACK
                            : ch == ']' ? Tok::RBRACK : Tok::COMMA;
                toks_.push_back({k, std::string(1, ch), col});
                ++i;
            } else {
                fail(std::string("unexpected character '") + ch + "'", col);
            }
        }
        toks_.push_back({Tok::END, "end", s_.size() + 1});
    }

    const Token& peek() const { return toks_[pos_]; }
    const Token& next() { return toks_[pos_++]; }
    bool at_op(char op) const { return peek().tok == Tok::OP && peek().text[0] == op; }
    Token expect(Tok k, const std::string& what) {
        const Token t = next();
        if (t.tok != k) fail("expected " + what + ", got '" + t.text + "'", t.column);
        return t;
    }
    void enter(size_t depth) const {
        if (depth > kMaxDepth) fail("nesting deeper than " + std::to_string(kMaxDepth), peek().column);
    }
    static Expr node(Expr::Kind k, size_t col, std::vector<Expr> args = {}) {
        Expr e;
        e.kind = k;
        e.column = col;
        e.args = std::move(args);
        return e;
    }

    Expr expr(size_t depth) {
        enter(depth);
        Expr left = term(depth + 1);
        while (at_op('+') || at_op('-')) {
            const Token op = next();
            Expr right = term(depth + 1);
            left = node(op.text[0] == '+' ? Expr::ADD : Expr::SUB, op.column, {std::move(left), std::move(right)});
        }
        return left;
    }
    Expr term(size_t depth) {
        enter(depth);
        Expr left = unary(depth + 1);
        while (at_op('*') || at_op('/')) {
            const Token op = next();
            Expr right = unary(depth + 1);
            left = node(op.text[0] == '*' ? Expr::MUL : Expr::DIV, op.column, {std::move(left), std::move(right)});
        }
        return left;
    }
    Expr unary(size_t depth) {
        enter(depth);
        if (at_op('-')) {
            const Token op = next();
            return node(Expr::NEG, op.column, {unary(depth + 1)});
        }
        return power(depth + 1);
    }
    Expr power(size_t depth) {
        enter(depth);
        Expr base = primary(depth + 1);
        if (at_op('^')) {
            const Token op = next();
            Expr ex = unary(depth + 1);
            return node(Expr::POW, op.column, {std::move(base), std::move(ex)});
        }
        return base;
    }
    std::pair<std::string, int> ix() {
        const Token t = next();
        auto integer = [](const std::string& s) {
            return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; });
        };
        if (t.tok == Tok::NUM) {
            if (!integer(t.text)) fail("an index constant is an integer, not " + t.text, t.column);
            return {"", std::stoi(t.text)};
        }
        if (t.tok != Tok::NAME) fail("expected an index, got '" + t.text + "'", t.column);
        int off = 0;
        if (at_op('+') || at_op('-')) {
            const int sign = next().text[0] == '+' ? 1 : -1;
            const Token c = next();
            if (c.tok != Tok::NUM || !integer(c.text)) fail("an index offset is an integer", c.column);
            off = sign * std::stoi(c.text);
        }
        return {t.text, off};
    }
    Expr primary(size_t depth) {
        enter(depth);
        const Token t = next();
        if (t.tok == Tok::NUM) {
            Expr e = node(Expr::NUM, t.column);
            e.value = std::strtof(t.text.c_str(), nullptr);  // the nearest float (C locale)
            return e;
        }
        if (t.tok == Tok::LPAREN) {
            Expr inner = expr(depth + 1);
            expect(Tok::RPAREN, "')'");
            return inner;
        }
        if (t.tok != Tok::NAME)
            fail(t.tok == Tok::END ? "expected an operand, got the end" : "expected an operand, got '" + t.text + "'",
                 t.column);
        if (t.text == "rowsum" || t.text == "rowmean" || t.text == "rowmax")
            fail("'" + t.text + "' is not defined in a recurrence body (write sum(index, ...))", t.column);
        if (t.text == "sum") {
            expect(Tok::LPAREN, "'('");
            Expr e = node(Expr::SUM, t.column);
            e.name = expect(Tok::NAME, "the summed index").text;
            expect(Tok::COMMA, "','");
            e.args.push_back(expr(depth + 1));
            expect(Tok::RPAREN, "')'");
            return e;
        }
        for (const Fn& f : kFns) {
            if (t.text != f.name) continue;
            if (peek().tok != Tok::LPAREN) fail("'" + t.text + "' is a function: expected '('", peek().column);
            next();
            Expr e = node(f.kind, t.column);
            e.args.push_back(expr(depth + 1));
            while (peek().tok == Tok::COMMA) {
                next();
                e.args.push_back(expr(depth + 1));
            }
            expect(Tok::RPAREN, "',' or ')'");
            if (e.args.size() != f.arity)
                fail("'" + t.text + "' takes " + std::to_string(f.arity) + " argument" + (f.arity == 1 ? "" : "s") +
                         ", got " + std::to_string(e.args.size()),
                     t.column);
            return e;
        }
        if (t.text == "pi") return node(Expr::PI, t.column);
        if (peek().tok == Tok::LPAREN) fail("unknown function '" + t.text + "'", t.column);
        if (peek().tok == Tok::LBRACK) {
            next();
            Expr e = node(Expr::READ, t.column);
            e.name = t.text;
            e.ix.push_back(ix());
            while (peek().tok == Tok::COMMA) {
                next();
                e.ix.push_back(ix());
            }
            expect(Tok::RBRACK, "']'");
            return e;
        }
        Expr e = node(Expr::INDEX, t.column);
        e.name = t.text;
        return e;
    }

    const std::string s_;
    std::string where_;
    std::vector<Token> toks_;
    size_t pos_ = 0;
};

bool uses_libm(const Expr& e) {
    if (e.kind == Expr::EXP || e.kind == Expr::LOG || e.kind == Expr::TANH || e.kind == Expr::ERF ||
        e.kind == Expr::SIGMOID || e.kind == Expr::POW)
        return true;
    for (const Expr& a : e.args)
        if (uses_libm(a)) return true;
    return false;
}

// The checker and compiler: names, scopes, then the gather offsets of every read.
struct Compiler {
    std::map<std::string, size_t> index;  // index name -> size
    std::map<std::string, std::string> kind;
    std::map<std::string, const Recurrence::Tensor*> tensors;
    std::map<std::string, int> slot;

    void declare(const std::string& name, const std::string& k) {
        if (!identifier(name) || reserved(name))
            refuse(31, "'" + name + "' is not a name (an identifier, not a function, pi or sum)");
        if (kind.count(name)) refuse(31, name + " is declared twice (" + kind[name] + " and " + k + ")");
        kind[name] = k;
    }
    // declaration: inputs and weights may repeat an index (W[c,c], IR_SPEC 6.10.1); assignments may not.
    std::vector<size_t> sizes(const std::vector<std::string>& idx, const std::string& where,
                              bool declaration = false) const {
        std::vector<size_t> out;
        for (size_t a = 0; a < idx.size(); ++a) {
            for (size_t b = 0; b < a && !declaration; ++b)
                if (idx[a] == idx[b]) refuse(32, where + ": an assignment repeats an index");
            auto it = index.find(idx[a]);
            if (it == index.end()) refuse(32, where + ": " + idx[a] + " is not a declared index");
            out.push_back(it->second);
        }
        return out;
    }

    void check(const Expr& e, const std::vector<std::string>& scope, const std::set<std::string>& readable,
               const std::string& where) const {
        auto in_scope = [&](const std::string& n) { return std::find(scope.begin(), scope.end(), n) != scope.end(); };
        if (e.kind == Expr::READ) {
            if (!readable.count(e.name)) refuse(32, where + ": reads " + e.name + ", which this body cannot read");
            const auto* t = tensors.at(e.name);
            if (e.ix.size() != t->shape.size())
                refuse(32, where + ": " + e.name + " has rank " + std::to_string(t->shape.size()) + ", read with " +
                               std::to_string(e.ix.size()) + " indices");
            for (const auto& [n, off] : e.ix)
                if (!n.empty() && !in_scope(n)) refuse(32, where + ": index " + n + " is not in scope");
        } else if (e.kind == Expr::INDEX) {
            if (!in_scope(e.name))
                refuse(32, where + ": " + e.name + (tensors.count(e.name) ? " is a tensor (write it with its indices)"
                                                                          : " is not an index in scope"));
        } else if (e.kind == Expr::SUM) {
            if (!index.count(e.name)) refuse(32, where + ": sum over " + e.name + ", which is not a declared index");
            if (in_scope(e.name)) refuse(32, where + ": sum over " + e.name + ", which is already in scope");
            std::vector<std::string> inner = scope;
            inner.push_back(e.name);
            check(e.args[0], inner, readable, where);
            return;
        }
        for (const Expr& a : e.args) check(a, scope, readable, where);
    }

    // Fills points, slot, gather, values and terms for e evaluated in `scope`.
    void compile(Expr& e, const std::vector<std::string>& scope) const {
        std::vector<size_t> sz;
        for (const auto& n : scope) sz.push_back(index.at(n));
        size_t P = 1;
        for (size_t s : sz) P *= s;
        e.points = P;
        auto coord = [&](size_t p, size_t axis) {  // the value of scope[axis] at point p (row-major)
            for (size_t a = sz.size(); a-- > axis + 1;) p /= sz[a];
            return p % sz[axis];
        };
        if (e.kind == Expr::READ) {
            const auto* t = tensors.at(e.name);
            e.slot = slot.at(e.name);
            e.gather.assign(P, -1);
            for (size_t p = 0; p < P; ++p) {
                long flat = 0;
                bool inside = true;
                for (size_t k = 0; k < e.ix.size(); ++k) {
                    long c = e.ix[k].second;
                    if (!e.ix[k].first.empty()) {
                        const size_t axis = static_cast<size_t>(
                            std::find(scope.begin(), scope.end(), e.ix[k].first) - scope.begin());
                        c += static_cast<long>(coord(p, axis));
                    }
                    if (c < 0 || c >= static_cast<long>(t->shape[k])) inside = false;
                    flat = flat * static_cast<long>(t->shape[k]) + c;
                }
                if (inside) e.gather[p] = static_cast<int>(flat);
            }
        } else if (e.kind == Expr::INDEX) {
            const size_t axis = static_cast<size_t>(std::find(scope.begin(), scope.end(), e.name) - scope.begin());
            e.values.resize(P);
            for (size_t p = 0; p < P; ++p) e.values[p] = static_cast<float>(coord(p, axis));
        } else if (e.kind == Expr::SUM) {
            std::vector<std::string> inner = scope;
            inner.push_back(e.name);
            e.terms = index.at(e.name);
            compile(e.args[0], inner);
            return;
        }
        for (Expr& a : e.args) compile(a, scope);
    }
};

// ---- the two tape ops a recurrence adds to the elementwise ones ----

// out[r, p] = src's element idx[p] (row r of src, or the flat element when src
// is one tensor broadcast over the rows), +0 where idx[p] < 0.
Var gather_cols(const Var& src, const std::vector<int>& idx, size_t rows, bool broadcast) {
    device::materialize(src->data);
    const size_t P = idx.size(), sc = src->data.cols();
    Matrix out(rows, P);
    for (size_t r = 0; r < rows; ++r)
        for (size_t p = 0; p < P; ++p)
            if (idx[p] >= 0) {
                const size_t k = static_cast<size_t>(idx[p]);
                out(r, p) = broadcast ? src->data(k / sc, k % sc) : src->data(r, k);
            }
    Var out_v = make_var(std::move(out), false);
    if (grad_enabled() && src->requires_grad) {
        out_v->requires_grad = true;
        out_v->parents = {src};
        Variable* self = out_v.get();
        out_v->backward_fn = [self, idx, broadcast]() {
            const Var& s = self->parents[0];
            device::materialize(self->grad);
            Matrix d(s->data.rows(), s->data.cols());
            const size_t c = d.cols();
            for (size_t p = 0; p < idx.size(); ++p) {
                if (idx[p] < 0) continue;
                const size_t k = static_cast<size_t>(idx[p]);
                for (size_t r = 0; r < self->grad.rows(); ++r) {
                    if (broadcast)
                        d(k / c, k % c) += self->grad(r, p);
                    else
                        d(r, k) += self->grad(r, p);
                }
            }
            s->accumulate(std::move(d));
        };
    }
    return out_v;
}

// [R, P * J] -> [R, P]: out[r, p] = sum over j ascending of v[r, p J + j], from +0.
Var sum_groups(const Var& v, size_t J) {
    device::materialize(v->data);
    const size_t R = v->data.rows(), P = v->data.cols() / J;
    Matrix out(R, P);
    for (size_t r = 0; r < R; ++r)
        for (size_t p = 0; p < P; ++p) {
            float s = 0.0f;
            for (size_t j = 0; j < J; ++j) s += v->data(r, p * J + j);
            out(r, p) = s;
        }
    Var out_v = make_var(std::move(out), false);
    if (grad_enabled() && v->requires_grad) {
        out_v->requires_grad = true;
        out_v->parents = {v};
        Variable* self = out_v.get();
        out_v->backward_fn = [self, J]() {
            const Var& x = self->parents[0];
            device::materialize(self->grad);
            Matrix d(x->data.rows(), x->data.cols());
            for (size_t r = 0; r < d.rows(); ++r)
                for (size_t p = 0; p < self->grad.cols(); ++p)
                    for (size_t j = 0; j < J; ++j) d(r, p * J + j) = self->grad(r, p);
            x->accumulate(std::move(d));
        };
    }
    return out_v;
}

// ---- evaluation ----

float fold_unary(Expr::Kind k, float a) {
    switch (k) {
        case Expr::NEG: return -a;
        case Expr::EXP: return std::exp(a);
        case Expr::LOG: return std::log(a);
        case Expr::SQRT: return std::sqrt(a);
        case Expr::TANH: return std::tanh(a);
        case Expr::SIGMOID: return 1.0f / (1.0f + std::exp(-a));
        case Expr::ERF: return std::erf(a);
        case Expr::ABS: return std::fabs(a);
        case Expr::RELU: return std::max(0.0f, a);
        default: throw std::logic_error("recurrence: fold_unary");
    }
}

float fold_binary(Expr::Kind k, float a, float b) {
    switch (k) {
        case Expr::ADD: return a + b;
        case Expr::SUB: return a - b;
        case Expr::MUL: return a * b;
        case Expr::DIV: return a / b;
        case Expr::POW: return std::pow(a, b);
        case Expr::MAX: return a >= b ? a : b;
        case Expr::MIN: return a <= b ? a : b;
        default: throw std::logic_error("recurrence: fold_binary");
    }
}

struct Val {
    bool scalar = true;
    float c = 0.0f;
    Var v;
};

class Evaluator {
public:
    Evaluator(const std::vector<Var>& env, const std::vector<bool>& broadcast) : env_(env), broadcast_(broadcast) {}

    Var define(const Def& d, size_t rows) {
        Val r = eval(d.e, rows);
        return full(r, rows, d.e.points);
    }

private:
    static Var full(const Val& r, size_t rows, size_t points) {
        if (!r.scalar) return r.v;
        return make_var(Matrix(rows, points, r.c), false);
    }
    static Val tensor(Var v) {
        Val out;
        out.scalar = false;
        out.v = std::move(v);
        return out;
    }
    static Val constant(float c) {
        Val out;
        out.c = c;
        return out;
    }

    Val eval(const Expr& e, size_t rows) {
        switch (e.kind) {
            case Expr::NUM: return constant(e.value);
            case Expr::PI: return constant(kPi);
            case Expr::INDEX: {
                Matrix m(rows, e.points);
                for (size_t r = 0; r < rows; ++r)
                    for (size_t p = 0; p < e.points; ++p) m(r, p) = e.values[p];
                return tensor(make_var(std::move(m), false));
            }
            case Expr::READ:
                return tensor(gather_cols(env_[static_cast<size_t>(e.slot)], e.gather, rows,
                                          broadcast_[static_cast<size_t>(e.slot)]));
            case Expr::NEG: {
                Val a = eval(e.args[0], rows);
                if (a.scalar) return constant(fold_unary(e.kind, a.c));
                return tensor(ops::scale(a.v, -1.0f));  // a sign flip, exact
            }
            case Expr::EXP: case Expr::LOG: case Expr::SQRT: case Expr::TANH: case Expr::SIGMOID:
            case Expr::ERF: case Expr::ABS: case Expr::RELU: {
                Val a = eval(e.args[0], rows);
                if (a.scalar) return constant(fold_unary(e.kind, a.c));
                switch (e.kind) {
                    case Expr::EXP: return tensor(ops::exp(a.v));
                    case Expr::LOG: return tensor(ops::log(a.v));
                    case Expr::SQRT: return tensor(ops::sqrt(a.v));
                    case Expr::TANH: return tensor(ops::tanh(a.v));
                    case Expr::SIGMOID: return tensor(ops::sigmoid(a.v));
                    case Expr::ERF: return tensor(ops::erf(a.v));
                    case Expr::ABS: return tensor(ops::abs(a.v));
                    default: return tensor(ops::relu(a.v));
                }
            }
            case Expr::ADD: case Expr::SUB: case Expr::MUL: case Expr::DIV: case Expr::POW:
            case Expr::MAX: case Expr::MIN: {
                Val a = eval(e.args[0], rows), b = eval(e.args[1], rows);
                if (a.scalar && b.scalar) return constant(fold_binary(e.kind, a.c, b.c));
                // Tensor-and-constant forms that are the same IEEE operation per element.
                if (e.kind == Expr::MUL && a.scalar) return tensor(ops::scale(b.v, a.c));
                if (e.kind == Expr::MUL && b.scalar) return tensor(ops::scale(a.v, b.c));
                if (e.kind == Expr::ADD && a.scalar) return tensor(ops::add_scalar(b.v, a.c));
                if (e.kind == Expr::ADD && b.scalar) return tensor(ops::add_scalar(a.v, b.c));
                if (e.kind == Expr::SUB && b.scalar) return tensor(ops::add_scalar(a.v, -b.c));
                const Var x = full(a, rows, e.points), y = full(b, rows, e.points);
                switch (e.kind) {
                    case Expr::ADD: return tensor(ops::add(x, y));
                    case Expr::SUB: return tensor(ops::sub(x, y));
                    case Expr::MUL: return tensor(ops::mul(x, y));
                    case Expr::DIV: return tensor(ops::div(x, y));
                    case Expr::POW: return tensor(ops::pow(x, y));
                    case Expr::MAX: return tensor(ops::maximum(x, y));
                    default: return tensor(ops::minimum(x, y));
                }
            }
            case Expr::WHERE: {
                Val c = eval(e.args[0], rows), a = eval(e.args[1], rows), b = eval(e.args[2], rows);
                if (c.scalar && a.scalar && b.scalar) return constant(c.c > 0.0f ? a.c : b.c);
                return tensor(ops::where(full(c, rows, e.points), full(a, rows, e.points), full(b, rows, e.points)));
            }
            case Expr::SUM: {
                Val body = eval(e.args[0], rows);
                return tensor(sum_groups(full(body, rows, e.points * e.terms), e.terms));
            }
        }
        throw std::logic_error("recurrence: unhandled node");
    }

    const std::vector<Var>& env_;
    const std::vector<bool>& broadcast_;
};

}  // namespace

bool Recurrence::initialised(size_t k) const {
    for (const Def& d : init_)
        if (d.name == state_[k].name) return true;
    return false;
}

Recurrence Recurrence::from_json(const json& nd) {
    Recurrence r;
    Compiler c;
    if (!nd.contains("index") || !nd["index"].is_object() || nd["index"].empty())
        refuse(31, "index must map index names to sizes");
    for (const auto& [k, v] : nd["index"].items()) {
        c.declare(k, "index");
        if (!v.is_number_integer() || v.get<long long>() < 1)
            refuse(31, "index " + k + ": size " + v.dump() + " is not a positive integer");
        c.index[k] = v.get<size_t>();
    }
    const size_t n_in = nd.contains("in") && nd["in"].is_array() ? nd["in"].size() : 0;
    if (!nd.contains("inputs") || !nd["inputs"].is_array() || nd["inputs"].size() != n_in || n_in == 0)
        refuse(31, "inputs must declare each of the " + std::to_string(n_in) + " values of in (at least one)");
    auto tensor_of = [&](const std::string& text, const std::string& where, const std::string& k) {
        Parser p(text, where);
        auto [name, idx] = p.decl();
        p.end();
        c.declare(name, k);
        Tensor t;
        t.name = name;
        t.index = idx;
        t.shape = c.sizes(idx, where, true);
        for (size_t s : t.shape) t.numel *= s;
        return t;
    };
    for (size_t j = 0; j < n_in; ++j) {
        if (!nd["inputs"][j].is_string()) refuse(31, "inputs[" + std::to_string(j) + "] is not a declaration string");
        r.inputs_.push_back(tensor_of(nd["inputs"][j].get<std::string>(), "inputs[" + std::to_string(j) + "]", "input"));
    }
    if (nd.contains("weights")) {
        if (!nd["weights"].is_object()) refuse(31, "weights must map declarations to tensor names");
        for (const auto& [k, v] : nd["weights"].items()) {
            Tensor t = tensor_of(k, "weights '" + k + "'", "weight");
            if (!v.is_string() || v.get<std::string>().empty())
                refuse(31, "weight " + t.name + ": the tensor name must be a string");
            t.tensor = v.get<std::string>();
            r.weights_.push_back(t);
        }
    }
    auto defs = [&](const char* key, bool required) {
        std::vector<Def> out;
        if (!nd.contains(key)) {
            if (required) refuse(31, std::string(key) + " must be a list of assignments (at least one)");
            return out;
        }
        if (!nd[key].is_array() || (required && nd[key].empty()))
            refuse(31, std::string(key) + " must be a list of assignments" + (required ? " (at least one)" : ""));
        for (size_t j = 0; j < nd[key].size(); ++j) {
            const std::string where = std::string(key) + "[" + std::to_string(j) + "]";
            if (!nd[key][j].is_string()) refuse(31, where + " is not an assignment string");
            Parser p(nd[key][j].get<std::string>(), where);
            Def d;
            auto [name, idx] = p.decl();
            d.name = name;
            d.index = idx;
            d.e = p.assignment_rhs();
            p.end();
            d.where = where;
            out.push_back(std::move(d));
        }
        return out;
    };
    r.update_ = defs("update", true);
    for (const Def& d : r.update_) {
        c.declare(d.name, "state");
        Tensor t;
        t.name = d.name;
        t.index = d.index;
        t.shape = c.sizes(d.index, d.where);
        for (size_t s : t.shape) t.numel *= s;
        r.state_.push_back(t);
    }
    r.let_ = defs("let", false);
    r.init_ = defs("init", false);
    if (!nd.contains("readout") || !nd["readout"].is_string()) refuse(31, "readout must be one assignment string");
    {
        Parser p(nd["readout"].get<std::string>(), "readout");
        auto [name, idx] = p.decl();
        r.readout_.name = name;
        r.readout_.index = idx;
        r.readout_.e = p.assignment_rhs();
        p.end();
        r.readout_.where = "readout";
    }
    // Slots, in the order inputs, weights, state, lets, readout; tensors by name.
    int next_slot = 0;
    for (const Tensor& t : r.inputs_) c.slot[t.name] = next_slot++, c.tensors[t.name] = &t;
    for (const Tensor& t : r.weights_) c.slot[t.name] = next_slot++, c.tensors[t.name] = &t;
    for (const Tensor& t : r.state_) c.slot[t.name] = next_slot++, c.tensors[t.name] = &t;
    std::set<std::string> ins, ws, st, lets;
    for (const Tensor& t : r.inputs_) ins.insert(t.name);
    for (const Tensor& t : r.weights_) ws.insert(t.name);
    for (const Tensor& t : r.state_) st.insert(t.name);
    std::set<std::string> seen;
    for (const Def& d : r.init_) {
        if (!st.count(d.name)) refuse(33, d.where + ": init assigns " + d.name + ", which is not a state tensor");
        if (!seen.insert(d.name).second) refuse(33, d.where + ": " + d.name + " is initialised twice");
        const auto* t = c.tensors.at(d.name);
        if (c.sizes(d.index, d.where) != t->shape) refuse(33, d.where + ": " + d.name + "'s sizes differ from its update's");
        c.check(d.e, d.index, ws, d.where);
    }
    r.lets_.reserve(r.let_.size());
    for (Def& d : r.let_) {
        std::set<std::string> readable = ins;
        readable.insert(ws.begin(), ws.end());
        readable.insert(st.begin(), st.end());
        readable.insert(lets.begin(), lets.end());
        c.check(d.e, d.index, readable, d.where);
        c.declare(d.name, "let");
        Tensor t;
        t.name = d.name;
        t.index = d.index;
        t.shape = c.sizes(d.index, d.where);
        for (size_t s : t.shape) t.numel *= s;
        r.lets_.push_back(t);
        c.tensors[d.name] = &r.lets_.back();
        c.slot[d.name] = next_slot++;
        lets.insert(d.name);
    }
    std::set<std::string> all = ins;
    all.insert(ws.begin(), ws.end());
    all.insert(st.begin(), st.end());
    all.insert(lets.begin(), lets.end());
    for (const Def& d : r.update_) c.check(d.e, d.index, all, d.where);
    c.check(r.readout_.e, r.readout_.index, all, "readout");
    c.declare(r.readout_.name, "readout");
    r.out_.name = r.readout_.name;
    r.out_.index = r.readout_.index;
    r.out_.shape = c.sizes(r.readout_.index, "readout");
    for (size_t s : r.out_.shape) r.out_.numel *= s;
    r.readout_.slot = next_slot++;
    r.n_slots_ = static_cast<size_t>(next_slot);
    // Compile: every body over its own index list.
    for (Def& d : r.init_) d.slot = c.slot.at(d.name), c.compile(d.e, d.index);
    for (Def& d : r.let_) d.slot = c.slot.at(d.name), c.compile(d.e, d.index);
    for (Def& d : r.update_) d.slot = c.slot.at(d.name), c.compile(d.e, d.index);
    c.compile(r.readout_.e, r.readout_.index);
    r.exact_ = true;
    for (const auto* v : {&r.init_, &r.let_, &r.update_})
        for (const Def& d : *v) r.exact_ = r.exact_ && !uses_libm(d.e);
    r.exact_ = r.exact_ && !uses_libm(r.readout_.e);
    return r;
}

namespace ops {

Var recurrence(const Recurrence& r, const std::vector<Var>& ins, const std::vector<Var>& weights, size_t seq_len,
               const std::vector<Matrix>* state_in, std::vector<Matrix>* state_out) {
    if (ins.size() != r.inputs().size())
        throw std::invalid_argument("recurrence: " + std::to_string(ins.size()) + " inputs for " +
                                    std::to_string(r.inputs().size()) + " declarations");
    if (weights.size() != r.weights().size()) throw std::invalid_argument("recurrence: one Var per declared weight");
    const size_t T = ins[0]->data.rows();
    const size_t sl = seq_len == 0 ? T : seq_len;
    if (sl == 0 || T % sl != 0) throw std::invalid_argument("recurrence: rows are not a multiple of seq_len");
    const size_t B = T / sl;
    for (size_t k = 0; k < ins.size(); ++k)
        if (ins[k]->data.rows() != T || ins[k]->data.cols() != r.inputs()[k].numel)
            throw std::invalid_argument("IR_SPEC 6.10 (rule 34): input " + r.inputs()[k].name + " is [" +
                                        std::to_string(ins[k]->data.rows()) + ", " +
                                        std::to_string(ins[k]->data.cols()) + "], its declaration [T, " +
                                        std::to_string(r.inputs()[k].numel) + "]");
    for (size_t k = 0; k < weights.size(); ++k)
        if (weights[k]->data.rows() * weights[k]->data.cols() != r.weights()[k].numel)
            throw std::invalid_argument("IR_SPEC 6.10 (rule 34): weight " + r.weights()[k].name + " has " +
                                        std::to_string(weights[k]->data.rows() * weights[k]->data.cols()) +
                                        " elements, its declaration " + std::to_string(r.weights()[k].numel));
    const size_t ni = r.inputs().size(), nw = r.weights().size(), ns = r.state().size();
    std::vector<Var> env(r.slots());
    std::vector<bool> broadcast(r.slots(), false);
    for (size_t k = 0; k < nw; ++k) env[ni + k] = weights[k], broadcast[ni + k] = true;
    Evaluator ev(env, broadcast);
    // The initial state: the init bodies (one row, broadcast to the B sequences), +0 elsewhere.
    std::vector<Var> state(ns);
    for (size_t k = 0; k < ns; ++k) {
        const size_t n = r.state()[k].numel;
        if (state_in) {
            if (state_in->size() != ns || (*state_in)[k].rows() != B || (*state_in)[k].cols() != n)
                throw std::invalid_argument("recurrence: state_in must hold [B, numel] per state tensor");
            state[k] = make_var((*state_in)[k], false);
            continue;
        }
        state[k] = make_var(Matrix(B, n), false);
        for (const Recurrence::Def& d : r.init_defs())
            if (d.name == r.state()[k].name) {
                std::vector<int> id(n);
                for (size_t p = 0; p < n; ++p) id[p] = static_cast<int>(p);
                state[k] = gather_cols(ev.define(d, 1), id, B, true);
            }
    }
    std::vector<Var> ys;
    std::vector<std::vector<int>> where;
    for (size_t t = 0; t < sl; ++t) {
        std::vector<int> rows(B);
        for (size_t b = 0; b < B; ++b) rows[b] = static_cast<int>(b * sl + t);
        for (size_t k = 0; k < ni; ++k) env[k] = gather_rows(ins[k], rows);
        for (size_t k = 0; k < ns; ++k) env[ni + nw + k] = state[k];
        for (const Recurrence::Def& d : r.let_defs()) env[static_cast<size_t>(d.slot)] = ev.define(d, B);
        std::vector<Var> next(ns);  // every update reads the state before the token
        for (size_t k = 0; k < ns; ++k) next[k] = ev.define(r.update_defs()[k], B);
        state = std::move(next);
        for (size_t k = 0; k < ns; ++k) env[ni + nw + k] = state[k];
        ys.push_back(ev.define(r.readout_def(), B));
        where.push_back(std::move(rows));
    }
    if (state_out) {
        state_out->clear();
        for (const Var& s : state) {
            device::materialize(s->data);
            state_out->push_back(s->data);
        }
    }
    return scatter_add_rows(ys, where, T);
}

}  // namespace ops
}  // namespace microtorch
