// ir_eval: paperkiln's C++ answers to the IR conformance suite's node-level requests (spec/IR_SPEC.md
// section 11.3; tools/ir_conformance_mtstudio.py, mtstudio's adapter). Reads a JSON array of requests on stdin and
// writes a JSON array of results, one per request, in order:
//   {"method": "digest", "text": str, "prose": bool}
//       -> {"sha256", "canonical"} | {"refused": message}          (section 3.10; ir::digest)
//   {"method": "run_node", "node", "inputs": {value: tensor}, "weights": {tensor name: tensor}}
//       -> {"outputs": {out: tensor}, "state": {slot: {tensor: tensor}}}   (a recurrence node, 6.10)
//   {"method": "run_node_grad", "node", "inputs", "weights", "out_grad": tensor}
//       -> {"grads": {value or tensor name: tensor}}   d sum(out * out_grad) by paperkiln's autograd
// A tensor is {"shape": [...], "f32": hex} (section 11.2). A failure is {"error": message}; a request this
// tool has no path for is {"skip": reason}.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "microtorch/ir.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/recurrence.hpp"

using namespace microtorch;
using json = nlohmann::ordered_json;

namespace {

Matrix decode(const json& t) {
    const std::string hex = t.at("f32").get<std::string>();
    std::vector<size_t> shape = t.at("shape").get<std::vector<size_t>>();
    size_t n = 1;
    for (size_t s : shape) n *= s;
    if (hex.size() != 8 * n) throw std::runtime_error("tensor: hex length does not match its shape");
    const size_t rows = shape.size() == 2 ? shape[0] : 1, cols = shape.size() == 2 ? shape[1] : n;
    Matrix m(rows, cols);
    for (size_t k = 0; k < n; ++k) {
        const uint32_t bits = static_cast<uint32_t>(std::stoul(hex.substr(8 * k, 8), nullptr, 16));
        float f;
        std::memcpy(&f, &bits, 4);
        m(k / cols, k % cols) = f;
    }
    return m;
}

json encode(const Matrix& m, std::vector<size_t> shape = {}) {
    static const char* hx = "0123456789abcdef";
    std::string out;
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j) {
            uint32_t bits;
            const float f = m(i, j);
            std::memcpy(&bits, &f, 4);
            for (int s = 28; s >= 0; s -= 4) out += hx[(bits >> s) & 15];
        }
    if (shape.empty()) shape = {m.rows(), m.cols()};
    return {{"shape", shape}, {"f32", out}};
}

struct Run {
    Recurrence r;
    std::vector<Var> ins, ws;
    std::vector<std::string> in_names, w_names;
    std::vector<std::vector<size_t>> in_shapes, w_shapes;
};

Run prepare(const json& req, bool grad) {
    const json& node = req.at("node");
    if (node.value("op", "") != "recurrence") throw std::domain_error("ir_eval runs recurrence nodes only");
    Run run{Recurrence::from_json(node), {}, {}, {}, {}, {}, {}};
    for (const auto& n : node.at("in")) {
        const json& t = req.at("inputs").at(n.get<std::string>());
        run.ins.push_back(make_var(decode(t), grad));
        run.in_names.push_back(n.get<std::string>());
        run.in_shapes.push_back(t.at("shape").get<std::vector<size_t>>());
    }
    for (const auto& w : run.r.weights()) {
        const json& t = req.at("weights").at(w.tensor);
        run.ws.push_back(make_var(decode(t), grad));
        run.w_names.push_back(w.tensor);
        run.w_shapes.push_back(t.at("shape").get<std::vector<size_t>>());
    }
    return run;
}

json handle(const json& req) {
    const std::string m = req.value("method", "");
    if (m == "digest") {
        const std::string text = req.at("text").get<std::string>();
        try {
            const json v = ir::parse_ijson(text);
            const std::string c = ir::canonical(v, req.value("prose", false));
            return {{"sha256", ir::sha256_hex(c)}, {"canonical", c}};
        } catch (const ir::DigestError& e) {
            return {{"refused", e.what()}};
        }
    }
    if (m == "run_node") {
        NoGrad ng;
        Run run = prepare(req, false);
        std::vector<Matrix> st;
        Var y = ops::recurrence(run.r, run.ins, run.ws, 0, nullptr, &st);
        json state = json::object();
        for (size_t k = 0; k < st.size(); ++k) state[run.r.state()[k].name] = encode(st[k], run.r.state()[k].shape);
        return {{"outputs", {{req["node"]["out"].get<std::string>(), encode(y->data)}}},
                {"state", {{req["node"]["state"].get<std::string>(), state}}}};
    }
    if (m == "run_node_grad") {
        Run run = prepare(req, true);
        Var y = ops::recurrence(run.r, run.ins, run.ws, 0);
        const Var g = make_var(decode(req.at("out_grad")));
        backward(ops::scale(ops::mean(ops::mul(y, g)), static_cast<float>(y->data.rows() * y->data.cols())));
        json grads = json::object();
        for (size_t k = 0; k < run.ins.size(); ++k)
            grads[run.in_names[k]] = encode(run.ins[k]->grad.rows() ? run.ins[k]->grad
                                                                    : Matrix(run.ins[k]->data.rows(), run.ins[k]->data.cols()),
                                            run.in_shapes[k]);
        for (size_t k = 0; k < run.ws.size(); ++k)
            grads[run.w_names[k]] = encode(run.ws[k]->grad.rows() ? run.ws[k]->grad
                                                                  : Matrix(run.ws[k]->data.rows(), run.ws[k]->data.cols()),
                                           run.w_shapes[k]);
        return {{"grads", grads}};
    }
    return {{"skip", "ir_eval has no method " + m}};
}

}  // namespace

int main() {
    const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    const json reqs = json::parse(in);
    json out = json::array();
    for (const auto& req : reqs) {
        try {
            out.push_back(handle(req));
        } catch (const std::domain_error& e) {
            out.push_back({{"skip", e.what()}});
        } catch (const std::exception& e) {
            out.push_back({{"error", e.what()}});
        }
    }
    std::cout << out.dump() << "\n";
    return 0;
}
