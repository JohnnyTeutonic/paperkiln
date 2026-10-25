#pragma once
// Runtime-defined ops (the IR's `expr` op, microtorch/ir.hpp): a new
// elementwise operation from a paper, say a ReLU variant, written as a
// string in a closed grammar and run without recompiling. Expr parses the
// string once; ops::expr builds the result from differentiable primitives
// (ops.hpp) on the tape, so the backward comes from the primitives' own
// backwards. spec/tools/ir_expr.py is the numpy reference of the same grammar
// and semantics.
//
// Grammar (nothing else is accepted):
//   expr    := term (('+' | '-') term)*
//   term    := unary (('*' | '/') unary)*
//   unary   := '-' unary | power
//   power   := primary ('^' unary)?          right-associative; -x^2 = -(x^2)
//   primary := number | name | 'pi' | '(' expr ')' | fn '(' expr (',' expr)* ')'
//   number  := decimal or scientific (1, 0.5, .5, 2., 1e-5, 1.5E+3), the
//              nearest float to the literal (strtof); must be finite
//   name    := the inputs in order: x, then a, b, c, ... (x the first, a
//              the second; papers/llm_extract.py names them the same way)
//   fn      := exp log sqrt tanh sigmoid erf abs relu (one argument),
//              max min (two), where (three), rowsum rowmean rowmax (one)
// Semantics, every input [T, w] (one w) and the result [T, w]: + - * /
// elementwise; ^ is pow; sigmoid(e) = 1 / (1 + exp(-e)); relu(e) =
// max(0, e); max(a, b) = a where a >= b, else b; min(a, b) = a where
// a <= b, else b; where(c, a, b) = a where c > 0, else b; rowsum(e) sums
// each row's w columns in ascending column order, rowmean(e) = rowsum(e) /
// w, rowmax(e) is the row's largest entry, each broadcast back to [T, w].
// Every primitive is computed in f32 and rounded to f32 after it (no fused
// multiply-add); a constant subexpression is folded in f32 with the same
// operations. Gradients follow the chosen operand of max, min and where
// (none reaches where's condition), the first largest entry of rowmax,
// and are 0 for abs and relu at 0.
#include <memory>
#include <string>
#include <vector>

#include "microtorch/autograd.hpp"

namespace microtorch {

class Expr {
public:
    enum class Kind {
        NUMBER, INPUT, NEG, ADD, SUB, MUL, DIV, POW,
        EXP, LOG, SQRT, TANH, SIGMOID, ERF, ABS, RELU,  // one argument
        MAX, MIN,                                       // two
        WHERE,                                          // three
        ROWSUM, ROWMEAN, ROWMAX,                        // one, over the row
    };
    struct Node {
        Kind kind = Kind::NUMBER;
        float value = 0.0f;       // NUMBER
        size_t input = 0;         // INPUT: 0 for x, 1 for a, 2 for b, ...
        std::vector<Node> args;   // operands, in order
        size_t column = 0;        // where the node starts in the text (1-based)
    };
    static constexpr size_t kMaxInputs = 23;  // x, then a .. v
    static constexpr size_t kMaxDepth = 200;  // nesting, so a hostile string cannot exhaust the stack

    // Throws std::invalid_argument naming the problem and its column, e.g.
    //   expr: expected an operand at column 5 in "x + * 2".
    static Expr parse(const std::string& text, size_t n_inputs = 1);

    const std::string& text() const { return text_; }
    size_t n_inputs() const { return n_inputs_; }
    const Node& root() const { return root_; }

private:
    std::string text_;
    size_t n_inputs_ = 1;
    Node root_;
};

namespace ops {

// The expression over ins (each [T, w], one w; ins.size() inputs), built
// from tape ops; [T, w]. A result that reads no input is that constant at
// every entry, with no gradient.
Var expr(const std::vector<Var>& ins, const Expr& e);
Var expr(const std::vector<Var>& ins, const std::string& text);  // parses with ins.size() inputs

}  // namespace ops
}  // namespace microtorch
