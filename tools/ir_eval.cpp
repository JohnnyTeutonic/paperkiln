// ir_eval: paperkiln's C++ answers to the IR conformance suite's node-level requests (spec/IR_SPEC.md
// section 11.3; tools/ir_conformance_mtstudio.py, mtstudio's adapter). Reads a JSON array of requests on stdin and
// writes a JSON array of results, one per request, in order:
//   {"method": "digest", "text": str, "prose": bool}
//       -> {"sha256", "canonical"} | {"refused": message}          (section 3.10; ir::digest)
//   {"method": "run_node", "node", "inputs": {value: tensor}, "weights": {tensor name: tensor}}
//       -> {"outputs": {out: tensor}, "state": {slot: {tensor: tensor}}}   (a recurrence node, 6.10; and
//          rope, rmsnorm, attention, conv1d_causal, selective_scan through paperkiln's training ops)
//   {"method": "run_node_grad", "node", "inputs", "weights", "out_grad": tensor}
//       -> {"grads": {value or tensor name: tensor}}   d sum(out * out_grad) by paperkiln's autograd
// A tensor is {"shape": [...], "f32": hex} (section 11.2). A failure is {"error": message}; a request this
// tool has no path for is {"skip": reason}.
#include <cmath>
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

// The thirdparty-profile single ops through the ops paperkiln trains with (LOCAL_LLM_STACK.md section 11
// item 4.4): rope (pairing, n_rot, kv_heads; freq_scale 1 only), rmsnorm (group; weight_offset 0 only),
// causal attention (kv_heads <= heads, no window or sinks; T5's position_bias through ops::relpos_attention,
// Shaw's relative_kv through ops::shaw_attention),
// conv1d_causal, selective_scan. Anything else
// these ops do not express is a skip naming the attribute.
json run_op(const json& req) {
    NoGrad ng;
    const json& nd = req.at("node");
    const std::string op = nd.at("op").get<std::string>();
    auto in = [&](size_t k) { return decode(req.at("inputs").at(nd.at("in").at(k).get<std::string>())); };
    auto weight = [&](const char* key) -> Var {
        if (!nd.contains(key) || nd[key].is_null()) return nullptr;
        return make_var(decode(req.at("weights").at(nd[key].get<std::string>())));
    };
    if (op == "rope") {
        // This adapter exercises training ops, not arbitrary IR RoPE variants.
        // Refuse untrained modifiers instead of silently returning plain RoPE.
        if (nd.contains("freq_factors") && !nd.at("freq_factors").is_null())
            throw std::domain_error(
                "ir_eval: rope freq_factors (paperkiln training op has no factors)");
        if (nd.value("attn_factor", 1.0) != 1.0)
            throw std::domain_error(
                "ir_eval: rope attn_factor != 1 (paperkiln training op has no amplitude scale)");
        if (nd.value("freq_scale", 1.0) != 1.0) throw std::domain_error("ir_eval: rope freq_scale != 1 (paperkiln trains none)");
        if (nd.at("rope_heads") != "all") throw std::domain_error("ir_eval: rope_heads first");
        const size_t H = nd.at("heads"), hd = nd.at("head_dim"), KV = nd.value("kv_heads", H);
        const size_t n_rot = nd.value("n_rot", hd);
        const std::string pairing = nd.at("pairing");
        const auto pr = pairing == "neox" ? ops::RopePairing::Neox : ops::RopePairing::Interleaved;
        const std::vector<int> pos = req.at("inputs").at(nd.at("in").at(2).get<std::string>()).get<std::vector<int>>();
        json outs = json::object();
        const Matrix q = in(0), k = in(1);
        size_t j = 0;
        for (const Matrix* t : {&q, &k}) {
            if (t->cols() != (j == 0 ? H : KV) * hd) throw std::runtime_error("ir_eval: rope input width");
            // The fused [T, 3w] layout apply_rope takes, the tensor in its q part.
            Var v = make_var(*t);
            Var y = ops::apply_rope(ops::concat_cols({v, v, v}), pos, nd.at("theta").get<float>(), hd, true, pr,
                                    n_rot == hd ? 0 : n_rot);
            outs[nd.at("out").at(j).get<std::string>()] = encode(ops::slice_cols(y, 0, t->cols())->data);
            ++j;
        }
        return {{"outputs", outs}, {"state", json::object()}};
    }
    if (op == "rmsnorm") {
        if (nd.value("weight_offset", 0.0) != 0.0) throw std::domain_error("ir_eval: rmsnorm weight_offset (paperkiln trains none)");
        const Matrix x = in(0);
        const size_t g = nd.value("group", x.cols());
        Var y = ops::rmsnorm_group(make_var(x), weight("weight"), g, nd.at("eps").get<float>());
        return {{"outputs", {{nd.at("out").get<std::string>(), encode(y->data)}}}, {"state", json::object()}};
    }
    if (op == "attention") {
        if (nd.value("softcap", 0.0) != 0.0)
            throw std::domain_error(
                "ir_eval: attention softcap (paperkiln training op has no softcap)");
        if (!nd.at("causal").get<bool>() || nd.value("window", 0) != 0 || nd.value("sinks", 0) != 0 ||
            nd.value("mask", std::string("causal")) != "causal")
            throw std::domain_error("ir_eval: attention other than plain causal (the llama block's)");
        const size_t H = nd.at("heads"), KV = nd.at("kv_heads"), hd = nd.at("head_dim");
        if (KV == 0 || H % KV != 0) throw std::runtime_error("ir_eval: kv_heads must divide heads");
        Var q = make_var(in(0)), k = make_var(in(1)), v = make_var(in(2));
        k = ops::repeat_heads(k, KV, H / KV, hd);
        v = ops::repeat_heads(v, KV, H / KV, hd);
        if (nd.contains("position_bias")) {
            // keramos 1.3 (IR_SPEC 5.15): T5's bias through the training op, ops::relpos_attention per query head;
            // the table [B, heads] is the weight, the buckets the node's own.
            const json& pb = nd.at("position_bias");
            Var table = make_var(decode(req.at("weights").at(pb.at("weight").get<std::string>())));
            const auto buckets = pb.at("buckets").get<std::vector<int>>();
            const float sc = nd.at("scale").get<float>();
            std::vector<Var> heads;
            for (size_t h = 0; h < H; ++h)
                heads.push_back(ops::relpos_attention(ops::slice_cols(q, h * hd, (h + 1) * hd),
                                                      ops::slice_cols(k, h * hd, (h + 1) * hd),
                                                      ops::slice_cols(v, h * hd, (h + 1) * hd), table, h, buckets,
                                                      sc, 0));
            Var y = ops::concat_cols(heads);
            return {{"outputs", {{nd.at("out").get<std::string>(), encode(y->data)}}}, {"state", json::object()}};
        }
        if (nd.contains("relative_kv")) {
            // keramos 1.4 (IR_SPEC 5.15): Shaw's relative key and value vectors through the training op,
            // ops::shaw_attention per query head; a table of width head_dim is shared by the heads, one of
            // width heads * head_dim holds head h's vectors at columns h * head_dim.
            const json& rk = nd.at("relative_kv");
            Var tk = make_var(decode(req.at("weights").at(rk.at("key").get<std::string>())));
            Var tv = rk.contains("value") && !rk["value"].is_null()
                         ? make_var(decode(req.at("weights").at(rk["value"].get<std::string>())))
                         : nullptr;
            const size_t K = rk.at("max_distance").get<size_t>();
            const float sc = nd.at("scale").get<float>();
            std::vector<Var> heads;
            for (size_t h = 0; h < H; ++h)
                heads.push_back(ops::shaw_attention(ops::slice_cols(q, h * hd, (h + 1) * hd),
                                                    ops::slice_cols(k, h * hd, (h + 1) * hd),
                                                    ops::slice_cols(v, h * hd, (h + 1) * hd), tk, tv,
                                                    tk->data.cols() == hd ? 0 : h * hd, K, sc, 0));
            Var y = ops::concat_cols(heads);
            return {{"outputs", {{nd.at("out").get<std::string>(), encode(y->data)}}}, {"state", json::object()}};
        }
        Var y = ops::attention_heads(ops::concat_cols({q, k, v}), H, nd.at("scale").get<float>(), 0, true);
        return {{"outputs", {{nd.at("out").get<std::string>(), encode(y->data)}}}, {"state", json::object()}};
    }
    if (op == "conv1d_causal") {
        const Matrix x = in(0);
        const size_t K = nd.at("kernel"), C = nd.at("channels");
        Var w = weight("weight");
        if (!w || w->data.rows() * w->data.cols() != C * K) throw std::runtime_error("ir_eval: conv1d weight");
        Matrix wk(C, K);
        for (size_t i = 0; i < C * K; ++i) wk(i / K, i % K) = w->data(i / w->data.cols(), i % w->data.cols());
        Var y = ops::conv1d_causal(make_var(x), make_var(wk), weight("bias"));
        // The state: the last K - 1 rows of (K - 1 zero rows, then x).
        Matrix st(K - 1, C);
        for (size_t r = 0; r + 1 < K; ++r) {
            const long src = static_cast<long>(x.rows()) - static_cast<long>(K - 1) + static_cast<long>(r);
            for (size_t c = 0; c < C; ++c) st(r, c) = src >= 0 ? x(static_cast<size_t>(src), c) : 0.0f;
        }
        return {{"outputs", {{nd.at("out").get<std::string>(), encode(y->data)}}},
                {"state", {{nd.at("state").get<std::string>(), {{"x", encode(st)}}}}}};
    }
    if (op == "selective_scan") {
        const size_t D = nd.at("channels"), N = nd.at("d_state");
        Var A = weight("a");
        if (!A || A->data.rows() * A->data.cols() != D * N) throw std::runtime_error("ir_eval: selective_scan a");
        Matrix a(D, N);
        for (size_t i = 0; i < D * N; ++i) a(i / N, i % N) = A->data(i / A->data.cols(), i % A->data.cols());
        Var y = ops::selective_scan(make_var(in(0)), make_var(in(1)), make_var(in(2)), make_var(in(3)), make_var(a),
                                    weight("d"), nd.value("dt_softplus", true));
        // The state after the last row: rerun the recurrence's last state from the op's own floats.
        // selective_scan keeps it internally; recompute here with the same expressions.
        const Matrix x = in(0), dt = in(1), Bm = in(2);
        Matrix s(D, N);
        for (size_t t = 0; t < x.rows(); ++t)
            for (size_t c = 0; c < D; ++c) {
                const float v = dt(t, c);
                const float dl = nd.value("dt_softplus", true) ? (v <= 20.0f ? std::log1p(std::exp(v)) : v) : v;
                const float xd = x(t, c) * dl;
                for (size_t n = 0; n < N; ++n) s(c, n) = s(c, n) * std::exp(dl * a(c, n)) + Bm(t, n) * xd;
            }
        return {{"outputs", {{nd.at("out").get<std::string>(), encode(y->data)}}},
                {"state", {{nd.at("state").get<std::string>(), {{"s", encode(s)}}}}}};
    }
    throw std::domain_error("ir_eval runs recurrence nodes and the thirdparty ops (rope, rmsnorm, attention, "
                            "conv1d_causal, selective_scan) only, not " + op);
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
    if (m == "run_node" && req.at("node").value("op", "") != "recurrence") return run_op(req);
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
    if (m == "run_node_grad" && req.at("node").value("op", "") == "attention") {
        // keramos 1.4 (IR_SPEC 5.15): attention with relative_kv, d sum(out * out_grad) through
        // ops::shaw_attention's backward (causal, no window or sinks), per query head over repeated kv heads.
        const json& nd = req.at("node");
        if (!nd.contains("relative_kv")) throw std::domain_error("ir_eval differentiates attention with relative_kv only");
        if (nd.value("window", 0) != 0 || nd.value("sinks", 0) != 0 || nd.value("mask", std::string("causal")) != "causal")
            throw std::domain_error("ir_eval: attention gradient other than plain causal");
        const size_t H = nd.at("heads"), KV = nd.at("kv_heads"), hd = nd.at("head_dim");
        const json& rk = nd.at("relative_kv");
        auto leaf = [&](const json& t) { return make_var(decode(t), true); };
        std::vector<Var> ins;
        std::vector<std::string> in_names;
        for (const auto& n : nd.at("in")) {
            ins.push_back(leaf(req.at("inputs").at(n.get<std::string>())));
            in_names.push_back(n.get<std::string>());
        }
        std::vector<Var> ws;
        std::vector<std::string> w_names;
        for (const char* key : {"key", "value"})
            if (rk.contains(key) && !rk[key].is_null()) {
                w_names.push_back(rk[key].get<std::string>());
                ws.push_back(leaf(req.at("weights").at(w_names.back())));
            }
        Var kk = ops::repeat_heads(ins[1], KV, H / KV, hd), vv = ops::repeat_heads(ins[2], KV, H / KV, hd);
        Var tk = ws[0], tv = ws.size() > 1 ? ws[1] : nullptr;
        std::vector<Var> heads;
        for (size_t h = 0; h < H; ++h)
            heads.push_back(ops::shaw_attention(ops::slice_cols(ins[0], h * hd, (h + 1) * hd),
                                                ops::slice_cols(kk, h * hd, (h + 1) * hd),
                                                ops::slice_cols(vv, h * hd, (h + 1) * hd), tk, tv,
                                                tk->data.cols() == hd ? 0 : h * hd, rk.at("max_distance").get<size_t>(),
                                                nd.at("scale").get<float>(), 0));
        Var y = ops::concat_cols(heads);
        const Var g = make_var(decode(req.at("out_grad")));
        backward(ops::scale(ops::mean(ops::mul(y, g)), static_cast<float>(y->data.rows() * y->data.cols())));
        json grads = json::object();
        auto put = [&](const std::string& name, const Var& v) {
            grads[name] = encode(v->grad.rows() ? v->grad : Matrix(v->data.rows(), v->data.cols()));
        };
        for (size_t i = 0; i < ins.size(); ++i) put(in_names[i], ins[i]);
        for (size_t i = 0; i < ws.size(); ++i) put(w_names[i], ws[i]);
        return {{"grads", grads}};
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
