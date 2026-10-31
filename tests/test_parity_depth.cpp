// Configurable parity depth: independent composition, gradients and default pins.
#include <cmath>
#include <cstdio>
#include <cstring>
#include "../tools/parity_model.hpp"
#include "check.hpp"
using namespace microtorch;

bool same(const Matrix& a, const Matrix& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() &&
           std::memcmp(a.data(), b.data(), a.rows() * a.cols() * sizeof(float)) == 0;
}
Var composed(parity::ParityLM& m, const std::vector<int>& ids, size_t sl) {
    std::vector<int> positions(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) positions[i] = static_cast<int>(i % sl);
    Var h = ops::add(m.wte->forward(ids), m.wpe->forward(positions));
    for (size_t b = 0; b < m.ln1.size(); ++b) {
        Var n = m.ln1[b]->forward(h), a;
        if (m.kind_ == parity::AttnKind::SRD)
            a = m.srd[b]->forward(n, sl);
        else if (m.kind_ == parity::AttnKind::KIMI)
            a = m.kimi[b]->forward(n, sl);
        else {
            const size_t d = n->data.cols();
            Var qkv = m.kimi[b]->c_attn->forward(n);
            a = m.kimi[b]->c_proj->forward(
                ops::recurrence(*m.rec,
                                {ops::slice_cols(qkv, 0, d), ops::slice_cols(qkv, d, 2 * d),
                                 ops::slice_cols(qkv, 2 * d, 3 * d)},
                                {}, sl));
        }
        h = ops::add(h, a);
        h = ops::add(h, m.mlp[b]->forward(m.ln2[b]->forward(h)));
    }
    return m.head->forward(m.ln_f->forward(h));
}
int main() {
    size_t fd_checks = 0;
    for (auto kind :
         {parity::AttnKind::KIMI, parity::AttnKind::SRD, parity::AttnKind::RECURRENCE}) {
        bool refused = false;
        try {
            parity::ParityLM bad(kind, 7, 4, 2, 8, 41, 4, 1, nullptr, 0);
        } catch (const std::runtime_error&) {
            refused = true;
        }
        CHECK(refused);
        // The historical two-block constructor and explicit depth have identical
        // names, seeds, parameter inventory, forwards and backward accumulation.
        parity::ParityLM old(kind, 7, 4, 2, 8, 41),
            explicit_two(kind, 7, 4, 2, 8, 41, 64, 1, nullptr, 2);
        auto oldp = old.named_parameters(), newp = explicit_two.named_parameters();
        CHECK(oldp.size() == newp.size());
        for (size_t i = 0; i < oldp.size(); ++i) {
            CHECK(oldp[i].first == newp[i].first);
            CHECK(same(oldp[i].second->data, newp[i].second->data));
        }
        std::vector<int> ids = {1, 2, 3, 4};
        Var y = old.forward(ids), manual = composed(explicit_two, ids, ids.size());
        CHECK(same(y->data, manual->data));
        if (kind == parity::AttnKind::SRD) {
            Var legacy = ops::scale(
                ops::add(ops::mean(old.srd[0]->gate()), ops::mean(old.srd[1]->gate())), 0.5f);
            CHECK(same(old.mean_gate()->data, legacy->data));
        }
        backward(ops::cross_entropy(y, {2, 3, 4, 5}));
        backward(ops::cross_entropy(manual, {2, 3, 4, 5}));
        for (size_t i = 0; i < oldp.size(); ++i)
            CHECK(same(oldp[i].second->grad, newp[i].second->grad));
        for (size_t depth : {1u, 3u, 4u}) {
            parity::ParityLM m(kind, 7, 4, 2, 8, 41, 64, 1, nullptr, depth);
            CHECK(m.ln1.size() == depth && m.ln2.size() == depth && m.mlp.size() == depth);
            Var batched = m.forward({1, 2, 3, 4}, 2);
            Var singles = composed(m, {1, 2, 3, 4}, 2);
            CHECK(same(batched->data, singles->data));
            // Stacked sequences retain independent positions and mixer state.
            Var first = m.forward({1, 2});
            Var second = m.forward({3, 4});
            for (size_t i = 0; i < 2; ++i)
                for (size_t j = 0; j < 7; ++j) {
                    CHECK(std::fabs(batched->data(i, j) - first->data(i, j)) < 1e-6f);
                    CHECK(std::fabs(batched->data(i + 2, j) - second->data(i, j)) < 1e-6f);
                }
            auto exported = m.family_export();
            CHECK(exported.has_value());
            CHECK(exported->ir.at("model").at("n_layers").get<size_t>() == depth);
            // The new last block participates in the whole-model derivative.
            zero_grad(m.parameters());
            Var loss = ops::cross_entropy(m.forward(ids), {2, 3, 4, 5});
            backward(loss);
            std::vector<Var> selected = {
                m.wte->weight, m.mlp.back()->c_fc->W, m.head->W,
                kind == parity::AttnKind::SRD ? m.srd.back()->c_attn->W : m.kimi.back()->c_attn->W};
            for (const Var& p : selected) {
                const size_t row = p == m.wte->weight ? 1 : 0;
                const float analytic = p->grad(row, 0), value = p->data(row, 0), h = 0.002f;
                double up, dn;
                {
                    NoGrad ng;
                    p->data(row, 0) = value + h;
                    up = ops::cross_entropy(m.forward(ids), {2, 3, 4, 5})->data(0, 0);
                    p->data(row, 0) = value - h;
                    dn = ops::cross_entropy(m.forward(ids), {2, 3, 4, 5})->data(0, 0);
                }
                p->data(row, 0) = value;
                const double fd = (up - dn) / (2 * h);
                CHECK(std::fabs(analytic - fd) /
                          (1 + std::max(std::fabs(fd), std::fabs(double(analytic)))) <
                      0.002);
                ++fd_checks;
            }
        }
    }
    std::printf(
        "parity depths 1/2/3/4: independent composition/default-bit/gradient/state isolation PASS; "
        "%zu FD checks\n",
        fd_checks);
}
