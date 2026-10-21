// expr_eval: ops::expr (microtorch/expr.hpp) on inputs from stdin, for
// tests/test_ir_expr.py's comparison with the numpy reference
// (tools/ir_expr.py).
//
//   stdin:  [{"expr": "x*sigmoid(1.702*x)", "inputs": [[[row], ...], ...]},
//            {"expr": "max(a,", "n_inputs": 2}, ...]
//   stdout: [{"out": [[row], ...]}, {"error": "expr: ... at column 7 ..."}, ...]
//
// A case without "inputs" is only parsed (with "n_inputs" inputs, default
// 1). Values travel as JSON doubles, exact for floats both ways; a
// non-finite output is the string "nan", "inf" or "-inf".
#include <cmath>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "microtorch/expr.hpp"

using json = nlohmann::json;
using namespace microtorch;

namespace {

json number(float v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    return static_cast<double>(v);
}

json run(const json& c) {
    const std::string text = c.at("expr").get<std::string>();
    if (!c.contains("inputs")) {
        Expr::parse(text, c.value("n_inputs", size_t(1)));
        return {{"ok", true}};
    }
    std::vector<Var> ins;
    for (const json& m : c.at("inputs")) {
        const size_t R = m.size(), C = R ? m[0].size() : 0;
        Matrix x(R, C);
        for (size_t i = 0; i < R; ++i)
            for (size_t j = 0; j < C; ++j) x(i, j) = static_cast<float>(m[i][j].get<double>());
        ins.push_back(make_var(std::move(x), false));
    }
    NoGrad ng;
    const Var y = ops::expr(ins, text);
    json rows = json::array();
    for (size_t i = 0; i < y->data.rows(); ++i) {
        json row = json::array();
        for (size_t j = 0; j < y->data.cols(); ++j) row.push_back(number(y->data(i, j)));
        rows.push_back(std::move(row));
    }
    return {{"out", std::move(rows)}};
}

}  // namespace

int main() {
    const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    const json cases = json::parse(in);
    json out = json::array();
    for (const json& c : cases) {
        try {
            out.push_back(run(c));
        } catch (const std::exception& e) {
            out.push_back({{"error", e.what()}});
        }
    }
    std::cout << out.dump() << "\n";
    return 0;
}
