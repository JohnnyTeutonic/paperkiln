// The IR's recurrence op (spec/IR_SPEC.md 6.10; microtorch/recurrence.hpp), CPU:
//   1. the linear_attention body equals ops::kimi_attention bit for bit (two sequences, two heads), and its
//      derived backward agrees with kimi's hand-written one;
//   2. the ssm_scan body equals ops::ssm_scan bit for bit, gradients agree with its hand-written BPTT;
//   3. prefill equals token-by-token decode (state_out -> state_in) bit for bit, outputs and state;
//   4. central-difference gradchecks of every body's derived backward (inputs and weights), an init body
//      reading a learned weight included;
//   5. refusals (rules 31-34);
//   6. the mtstudio lane (tools/parity_model.hpp, attention "recurrence"): the model with the default
//      body equals the kimi model bit for bit (logits), its parameter gradients agree with kimi's, an FD
//      gradcheck through the whole model, and its IR export carries the recurrence node and slot.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../tools/parity_model.hpp"
#include "check.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/recurrence.hpp"

using namespace microtorch;
using json = nlohmann::ordered_json;

namespace {

Matrix randn(size_t r, size_t c, unsigned seed, float s = 1.0f) {
    std::mt19937 g(seed);
    std::normal_distribution<float> n(0.0f, s);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = n(g);
    return m;
}

bool bits_equal(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j) {
            const float x = a(i, j), y = b(i, j);
            if (std::memcmp(&x, &y, 4) != 0 && !(std::isnan(x) && std::isnan(y))) return false;
        }
    return true;
}

double rel_diff(const Matrix& a, const Matrix& b) {
    double w = 0, m = 1;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j) {
            w = std::max(w, static_cast<double>(std::fabs(a(i, j) - b(i, j))));
            m = std::max(m, static_cast<double>(std::fabs(b(i, j))));
        }
    return w / m;
}

const char* kLinearAttention = R"J({"op": "recurrence", "in": ["q", "k", "v"], "state": "lin",
  "index": {"h": 2, "e": 3}, "inputs": ["q[h,e]", "k[h,e]", "v[h,e]"],
  "let": ["pk[h,e] = where(k[h,e], k[h,e] + 1, (exp(min(0, k[h,e])) - 1) + 1)"],
  "update": ["num[h,e] = num[h,e] + pk[h,e] * v[h,e]", "den[h,e] = den[h,e] + pk[h,e]"],
  "readout": "y[h,e] = (where(q[h,e], q[h,e] + 1, (exp(min(0, q[h,e])) - 1) + 1) * num[h,e]) / max(den[h,e], 1e-08)",
  "out": "y"})J";
const char* kSsmScan = R"J({"op": "recurrence", "in": ["u"], "state": "ssm", "index": {"r": 4, "j": 4, "o": 1},
  "inputs": ["u[r]"], "weights": {"A[r,j]": "A", "B[r]": "B", "C[r]": "C", "D[o]": "D"},
  "update": ["s[r] = sum(j, A[r,j] * s[j]) + B[r] * u[r]"], "readout": "y[r] = C[r] * s[r] + D[0] * u[r]",
  "out": "y"})J";
const char* kConv = R"J({"op": "recurrence", "in": ["x"], "state": "conv", "index": {"c": 3, "k": 4, "r": 3},
  "inputs": ["u[c]"], "weights": {"W[c,k]": "w", "Bias[c]": "b"},
  "let": ["xx[k,c] = where(k - 2.5, u[c], x[k,c])"], "update": ["x[r,c] = xx[r + 1,c]"],
  "readout": "y[c] = sum(k, W[c,k] * xx[k,c]) + Bias[c]", "out": "y"})J";
const char* kSelective = R"J({"op": "recurrence", "in": ["x", "dt", "B", "C"], "state": "ssm",
  "index": {"c": 3, "n": 2}, "inputs": ["x[c]", "dt[c]", "Bm[n]", "Cm[n]"], "weights": {"A[c,n]": "A", "Dk[c]": "Dk"},
  "let": ["delta[c] = where(dt[c] - 20, dt[c], log(1 + exp(min(20, dt[c]))))", "xd[c] = x[c] * delta[c]"],
  "update": ["s[c,n] = s[c,n] * exp(delta[c] * A[c,n]) + Bm[n] * xd[c]"],
  "readout": "y[c] = sum(n, s[c,n] * Cm[n]) + Dk[c] * x[c]", "out": "y"})J";
// A tanh RNN whose initial state is a learned weight (an init body), read through an index offset.
const char* kRnnInit = R"J({"op": "recurrence", "in": ["x"], "state": "rnn", "index": {"i": 3, "j": 3},
  "inputs": ["x[i]"], "weights": {"W[i,j]": "W", "U[i,j]": "U", "h0[i]": "h0"},
  "init": ["h[i] = h0[i]"],
  "update": ["h[i] = tanh(sum(j, W[i,j] * h[j]) + sum(j, U[i,j] * x[j]))"],
  "readout": "y[i] = h[i] + 0.5 * h[i - 1]", "out": "y"})J";

Recurrence parse(const char* text) { return Recurrence::from_json(json::parse(text)); }

// loss = sum(recurrence(ins, ws) .* R); FD over every element of ins and ws.
double gradcheck(const Recurrence& r, std::vector<Var> ins, std::vector<Var> ws, size_t seq_len, unsigned seed) {
    const Matrix R = randn(ins[0]->data.rows(), r.output().numel, seed);
    const Var rv = make_var(R);
    auto loss_of = [&]() {
        Var y = ops::recurrence(r, ins, ws, seq_len);
        const float n = static_cast<float>(y->data.rows() * y->data.cols());
        return ops::scale(ops::mean(ops::mul(y, rv)), n);
    };
    Var loss = loss_of();
    backward(loss);
    double worst = 0;
    const float h = 1e-2f;
    std::vector<Var> all = ins;
    all.insert(all.end(), ws.begin(), ws.end());
    for (const Var& p : all) {
        CHECK(p->grad.rows() == p->data.rows());
        for (size_t i = 0; i < p->data.rows(); ++i)
            for (size_t j = 0; j < p->data.cols(); ++j) {
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
    return worst;
}

Var leaf(Matrix m) { return make_var(std::move(m), true); }

}  // namespace

int main() {
    // 1. linear_attention ------------------------------------------------------
    {
        const size_t B = 2, sl = 7, H = 2, E = 3, T = B * sl;
        const Recurrence r = parse(kLinearAttention);
        CHECK(!r.exact());
        Var q = leaf(randn(T, H * E, 1)), k = leaf(randn(T, H * E, 2)), v = leaf(randn(T, H * E, 3));
        Var y = ops::recurrence(r, {q, k, v}, {}, sl);
        std::vector<Var> heads;
        Var q2 = leaf(q->data), k2 = leaf(k->data), v2 = leaf(v->data);
        for (size_t h = 0; h < H; ++h)
            heads.push_back(ops::kimi_attention(ops::slice_cols(q2, h * E, (h + 1) * E),
                                                ops::slice_cols(k2, h * E, (h + 1) * E),
                                                ops::slice_cols(v2, h * E, (h + 1) * E), true, sl));
        Var ref = ops::concat_cols(heads);
        CHECK(bits_equal(y->data, ref->data));
        const Var g = make_var(randn(T, H * E, 4));
        backward(ops::mean(ops::mul(y, g)));
        backward(ops::mean(ops::mul(ref, g)));
        const double dq = rel_diff(q->grad, q2->grad), dk = rel_diff(k->grad, k2->grad), dv = rel_diff(v->grad, v2->grad);
        std::printf("1. linear_attention body == ops::kimi_attention bit for bit ([%zu, %zu], %zu sequences); "
                    "derived vs hand-written backward: dq %.1e dk %.1e dv %.1e\n", T, H * E, B, dq, dk, dv);
        CHECK(dq < 1e-5 && dk < 1e-5 && dv < 1e-5);
    }
    // 2. ssm_scan ----------------------------------------------------------------
    {
        const size_t T = 9, n = 4;
        const Recurrence r = parse(kSsmScan);
        CHECK(r.exact());
        Var u = leaf(randn(T, n, 5)), A = leaf(randn(n, n, 6, 0.3f)), Bv = leaf(randn(n, 1, 7)),
            Cv = leaf(randn(1, n, 8)), Dv = leaf(randn(1, 1, 9));
        Var y = ops::recurrence(r, {u}, {A, Bv, Cv, Dv});
        Var u2 = leaf(u->data), A2 = leaf(A->data), B2 = leaf(Bv->data), C2 = leaf(Cv->data), D2 = leaf(Dv->data);
        Var ref = ops::ssm_scan(u2, A2, B2, C2, D2);
        CHECK(bits_equal(y->data, ref->data));
        const Var g = make_var(randn(T, n, 10));
        backward(ops::mean(ops::mul(y, g)));
        backward(ops::mean(ops::mul(ref, g)));
        double worst = 0;
        for (auto [a, b] : {std::pair{u, u2}, {A, A2}, {Bv, B2}, {Cv, C2}, {Dv, D2}})
            worst = std::max(worst, rel_diff(a->grad, b->grad));
        std::printf("2. ssm_scan body == ops::ssm_scan bit for bit ([%zu, %zu]); derived vs hand-written backward "
                    "worst %.1e\n", T, n, worst);
        CHECK(worst < 1e-5);
    }
    // 3. prefill = token-by-token decode --------------------------------------------
    {
        NoGrad ng;
        const size_t T = 6;
        struct Case {
            const char* text;
            std::vector<size_t> widths;
            std::vector<std::pair<size_t, size_t>> wshapes;
        };
        const std::vector<Case> cases = {{kLinearAttention, {6, 6, 6}, {}},
                                         {kSsmScan, {4}, {{4, 4}, {4, 1}, {1, 4}, {1, 1}}},
                                         {kConv, {3}, {{3, 4}, {1, 3}}},
                                         {kSelective, {3, 3, 2, 2}, {{3, 2}, {1, 3}}},
                                         {kRnnInit, {3}, {{3, 3}, {3, 3}, {1, 3}}}};
        unsigned seed = 20;
        for (const Case& c : cases) {
            const Recurrence r = parse(c.text);
            std::vector<Var> ins, ws;
            for (size_t w : c.widths) ins.push_back(make_var(randn(T, w, seed++)));
            for (auto [a, b] : c.wshapes) ws.push_back(make_var(randn(a, b, seed++, 0.4f)));
            std::vector<Matrix> st_full;
            Var y = ops::recurrence(r, ins, ws, 0, nullptr, &st_full);
            std::vector<Matrix> st;
            bool same = true;
            for (size_t t = 0; t < T; ++t) {
                std::vector<Var> row;
                for (const Var& x : ins) row.push_back(ops::gather_rows(x, {static_cast<int>(t)}));
                std::vector<Matrix> next;
                Var yt = ops::recurrence(r, row, ws, 0, t == 0 ? nullptr : &st, &next);
                same = same && bits_equal(yt->data, ops::gather_rows(y, {static_cast<int>(t)})->data);
                st = next;
            }
            for (size_t k = 0; k < st.size(); ++k) same = same && bits_equal(st[k], st_full[k]);
            CHECK(same);
        }
        std::printf("3. prefill == token-by-token decode bit for bit (outputs and state) for %zu bodies\n",
                    cases.size());
    }
    // 4. gradchecks ------------------------------------------------------------------
    {
        double worst_all = 0;
        auto report = [&](const char* name, double w) {
            std::printf("   %-34s FD rel err %.2e\n", name, w);
            CHECK(w < 2e-3);
            worst_all = std::max(worst_all, w);
        };
        report("linear_attention (q, k, v)",
               gradcheck(parse(kLinearAttention), {leaf(randn(8, 6, 31)), leaf(randn(8, 6, 32)), leaf(randn(8, 6, 33))},
                         {}, 4, 34));
        report("ssm_scan (u, A, B, C, D)",
               gradcheck(parse(kSsmScan), {leaf(randn(6, 4, 35))},
                         {leaf(randn(4, 4, 36, 0.3f)), leaf(randn(4, 1, 37)), leaf(randn(1, 4, 38)), leaf(randn(1, 1, 39))},
                         0, 40));
        report("conv1d_causal (x, W, bias)",
               gradcheck(parse(kConv), {leaf(randn(10, 3, 41))}, {leaf(randn(3, 4, 42)), leaf(randn(1, 3, 43))}, 5, 44));
        report("selective_scan (x, dt, B, C, A, D)",
               gradcheck(parse(kSelective),
                         {leaf(randn(6, 3, 45)), leaf(randn(6, 3, 46)), leaf(randn(6, 2, 47)), leaf(randn(6, 2, 48))},
                         {leaf(randn(3, 2, 49, 0.5f)), leaf(randn(1, 3, 50))}, 0, 51));
        report("tanh RNN with a learned h0 (init)",
               gradcheck(parse(kRnnInit), {leaf(randn(8, 3, 52))},
                         {leaf(randn(3, 3, 53, 0.5f)), leaf(randn(3, 3, 54, 0.5f)), leaf(randn(1, 3, 55))}, 4, 56));
        std::printf("4. gradients: 5 bodies FD-checked, worst %.2e\n", worst_all);
    }
    // 5. refusals ----------------------------------------------------------------------
    {
        struct Bad {
            const char* text;
            int rule;
        };
        const std::vector<Bad> bad = {
            {R"J({"op":"recurrence","in":["x"],"index":{"i":2},"inputs":["x[i]"],"update":["s[i] = s[i] +"],"readout":"y[i] = s[i]"})J", 31},
            {R"J({"op":"recurrence","in":["x"],"index":{"i":2},"inputs":["x[i]"],"update":["x[i] = x[i]"],"readout":"y[i] = x[i]"})J", 31},
            {R"J({"op":"recurrence","in":["x"],"index":{"i":0},"inputs":["x[i]"],"update":["s[i] = s[i]"],"readout":"y[i] = s[i]"})J", 31},
            {R"J({"op":"recurrence","in":["x"],"index":{"i":2,"j":2},"inputs":["x[i]"],"update":["s[i] = s[j]"],"readout":"y[i] = s[i]"})J", 32},
            {R"J({"op":"recurrence","in":["x"],"index":{"i":2},"inputs":["x[i]"],"update":["s[i] = sum(i, s[i])"],"readout":"y[i] = s[i]"})J", 32},
            {R"J({"op":"recurrence","in":["x"],"index":{"i":2},"inputs":["x[i]"],"update":["s[i] = s[i,i]"],"readout":"y[i] = s[i]"})J", 32},
            {R"J({"op":"recurrence","in":["x"],"index":{"i":2},"inputs":["x[i]"],"update":["s[i] = rowsum(s[i])"],"readout":"y[i] = s[i]"})J", 31},
            {R"J({"op":"recurrence","in":["x"],"index":{"i":2},"inputs":["x[i]"],"init":["s[i] = x[i]"],"update":["s[i] = s[i]"],"readout":"y[i] = s[i]"})J", 32},
            {R"J({"op":"recurrence","in":["x"],"index":{"i":2},"inputs":["x[i]"],"init":["t[i] = 1"],"update":["s[i] = s[i]"],"readout":"y[i] = s[i]"})J", 33},
        };
        for (const Bad& b : bad) {
            bool refused = false;
            try {
                Recurrence::from_json(json::parse(b.text));
            } catch (const std::invalid_argument& e) {
                refused = std::string(e.what()).find("(rule " + std::to_string(b.rule) + ")") != std::string::npos;
                if (!refused) std::printf("   unexpected message: %s\n", e.what());
            }
            CHECK(refused);
        }
        // Rule 34: an input whose width differs from its declaration, at run time.
        bool refused = false;
        try {
            NoGrad ng;
            ops::recurrence(parse(kConv), {make_var(Matrix(4, 5))}, {make_var(Matrix(3, 4)), make_var(Matrix(1, 3))});
        } catch (const std::invalid_argument& e) {
            refused = std::string(e.what()).find("rule 34") != std::string::npos;
        }
        CHECK(refused);
        std::printf("5. refusals: %zu malformed nodes and a mis-sized input refused, each under its rule\n",
                    bad.size());
    }
    // 6. the mtstudio lane ---------------------------------------------------------------
    {
        const size_t V = 20, d = 8, H = 2, sl = 6;
        parity::ParityLM km(parity::AttnKind::KIMI, V, d, H, sl, 7);
        parity::ParityLM rm(parity::AttnKind::RECURRENCE, V, d, H, sl, 7);
        std::mt19937 g(11);
        std::vector<int> ids(2 * sl), tgt(2 * sl);
        for (auto& t : ids) t = static_cast<int>(g() % V);
        for (auto& t : tgt) t = static_cast<int>(g() % V);
        Var lk = km.forward(ids, sl), lr = rm.forward(ids, sl);
        CHECK(bits_equal(lk->data, lr->data));
        backward(ops::cross_entropy(lk, tgt));
        Var loss = ops::cross_entropy(lr, tgt);
        backward(loss);
        const auto pk = km.named_parameters(), pr = rm.named_parameters();
        CHECK(pk.size() == pr.size());
        double worst = 0;
        for (size_t k = 0; k < pk.size(); ++k) {
            CHECK(pk[k].first == pr[k].first);
            worst = std::max(worst, rel_diff(pr[k].second->grad, pk[k].second->grad));
        }
        CHECK(worst < 1e-5);
        // FD through the whole model, on the parameters upstream and downstream of the recurrence (the
        // attention projections of both blocks): 4 entries of each.
        const float h = 2e-3f;
        double fd_worst = 0;
        size_t n_fd = 0;
        for (const auto& [name, p] : pr) {
            if (name.rfind("attn_", 0) != 0) continue;
            for (size_t c = 0; c < 4; ++c) {
                const size_t i = (c * 7919u) % p->data.rows(), j = (c * 104729u) % p->data.cols();
                const float keep = p->data(i, j);
                NoGrad ng;
                p->data(i, j) = keep + h;
                const double up = ops::cross_entropy(rm.forward(ids, sl), tgt)->data(0, 0);
                p->data(i, j) = keep - h;
                const double dn = ops::cross_entropy(rm.forward(ids, sl), tgt)->data(0, 0);
                p->data(i, j) = keep;
                const double fd = (up - dn) / (2.0 * h), a = p->grad(i, j);
                fd_worst = std::max(fd_worst, std::abs(a - fd) / (1.0 + std::max(std::abs(a), std::abs(fd))));
                ++n_fd;
            }
        }
        std::printf("   whole-model FD over the attention projections: %zu entries, worst %.2e\n", n_fd, fd_worst);
        CHECK(n_fd >= 16 && fd_worst < 1e-2);
        const auto fx = rm.family_export();
        CHECK(fx.has_value() && fx->arch == "gpt2" && fx->ir["family_hint"] == "recurrence");
        bool node = false, slot = false;
        for (const auto& n : fx->ir["block"]["nodes"]) node = node || n["op"] == "recurrence";
        for (const auto& st : fx->ir["block"]["state"])
            slot = slot || (st["op"] == "recurrence" && st["tensors"].size() == 2);
        CHECK(node && slot);
        std::printf("6. mtstudio lane: recurrence model == kimi model bit for bit (logits, 2 sequences); parameter "
                    "gradients within %.1e; FD through the model, %zu entries, worst %.2e; IR export carries the "
                    "recurrence node and slot\n", worst, n_fd, fd_worst);
    }
    std::printf("recurrence: all checks passed\n");
    return 0;
}
