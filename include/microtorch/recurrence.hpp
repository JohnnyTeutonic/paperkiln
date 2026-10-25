#pragma once
// The IR's recurrence op (spec/IR_SPEC.md section 6.10, profile recurrence):
// a per-token state update declared in the document, with init, let, update
// and readout bodies in an indexed extension of the expr grammar. Recurrence
// parses and checks a node (the same rules as tools/ir_recurrence.py, the
// numpy reference); ops::recurrence runs it on the tape, token by token, so
// its backward is the autograd of the unrolled bodies (backpropagation
// through time over the sequential fold). Every primitive is computed in f32
// and rounded, with no fused multiply-add; a sum adds its terms in ascending
// index order from +0; a read outside a tensor is +0.
//
// The four hand-written stateful ops have recurrence forms (IR_SPEC.md
// 6.10.5, tools/ir_recurrence.py's rewrites); on the CPU, ops::recurrence over
// the linear_attention body equals ops::kimi_attention bit for bit, and over
// the ssm_scan body ops::ssm_scan (tests/test_recurrence_op.cpp).
#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "microtorch/autograd.hpp"

namespace microtorch {

class Recurrence {
public:
    struct Expr {
        enum Kind {
            NUM, PI, READ, INDEX, NEG, ADD, SUB, MUL, DIV, POW,
            EXP, LOG, SQRT, TANH, SIGMOID, ERF, ABS, RELU, MAX, MIN, WHERE, SUM
        } kind = NUM;
        float value = 0.0f;                          // NUM
        std::string name;                            // READ: the tensor; INDEX, SUM: the index
        std::vector<std::pair<std::string, int>> ix;  // READ: (index or "", offset or constant)
        std::vector<Expr> args;
        size_t column = 0;
        // Compiled (by from_json): the points of this node's scope, a READ's
        // source slot and flat offsets (-1 = outside: +0), an INDEX's values,
        // a SUM's term count.
        size_t points = 0;
        int slot = -1;
        std::vector<int> gather;
        std::vector<float> values;
        size_t terms = 0;
    };
    struct Def {
        std::string name;
        std::vector<std::string> index;
        Expr e;
        std::string where;
        int slot = -1;  // the slot it writes
    };
    struct Tensor {
        std::string name, tensor;          // tensor: a weight's GGUF name ("" otherwise)
        std::vector<std::string> index;
        std::vector<size_t> shape;
        size_t numel = 1;
    };

    // Parses and checks a recurrence node (IR_SPEC.md 10.1 rules 31-33);
    // throws std::invalid_argument naming the rule, the body and the column.
    static Recurrence from_json(const nlohmann::ordered_json& node);

    const std::vector<Tensor>& inputs() const { return inputs_; }
    const std::vector<Tensor>& weights() const { return weights_; }
    const std::vector<Tensor>& state() const { return state_; }  // in update order
    const Tensor& output() const { return out_; }
    bool initialised(size_t k) const;  // state tensor k has an init body
    // EXACT when no body uses exp, log, tanh, erf, sigmoid or ^ (IR_SPEC.md 6.10.4).
    bool exact() const { return exact_; }

    // Internal (ops::recurrence).
    const std::vector<Def>& init_defs() const { return init_; }
    const std::vector<Def>& let_defs() const { return let_; }
    const std::vector<Def>& update_defs() const { return update_; }
    const Def& readout_def() const { return readout_; }
    size_t slots() const { return n_slots_; }

private:
    std::vector<Tensor> inputs_, weights_, state_, lets_;
    Tensor out_;
    std::vector<Def> init_, let_, update_;
    Def readout_;
    bool exact_ = true;
    size_t n_slots_ = 0;
};

namespace ops {

// The recurrence over ins (one [T, width] per input, T = B * seq_len rows:
// B sequences of seq_len rows each; seq_len 0 = one sequence of T rows), its
// weights in Recurrence::weights() order (any shape with the declared element
// count; gradients reach them), from the initial state or from *state_in
// (one [B, numel] per state tensor). *state_out, when given, receives the
// state after each sequence's last row. [T, output width].
Var recurrence(const Recurrence& r, const std::vector<Var>& ins, const std::vector<Var>& weights,
               size_t seq_len = 0, const std::vector<Matrix>* state_in = nullptr,
               std::vector<Matrix>* state_out = nullptr);

}  // namespace ops
}  // namespace microtorch
