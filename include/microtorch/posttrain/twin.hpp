#pragma once
// Cue-selective twin loss (the root workspace's AI_ML/chat_screens_claude/cue_selective/PLAN.md;
// lane amendment A31). Claude Opus 5.5, 13 Nov 2026.
//
// A twin group holds one scenario told three ways: A and B differ in the one decisive detail and
// have different rule-checked replies yA and yB; N (A') differs from A in one detail that does not
// change the answer and keeps yA. The group's rows, each a context followed by a reply:
//   0: A + yA    1: B + yA    2: B + yB    3: A + yB    4: N + yA (only when mu > 0)
// lp(row) is the reply's log-probability summed over its target tokens (its close included).
// Per reply token (n_A, n_B: the replies' target counts):
//   dA = (lp(0) - lp(1)) / n_A     dB = (lp(2) - lp(3)) / n_B     dN = (lp(0) - lp(4)) / n_A
//   loss = lambda * [mean_g relu(m - dA) + mean_g relu(m - dB)] + mu * mean_g |dN|
// No row carries next-token loss here: the anchor is the ordinary SFT loss, whose packet holds the
// same twins as ordinary rows. The loss is composed on the tape from ops that are each
// finite-difference checked (gather_rows, logsumexp_row, gather_elems, scatter_add_rows, sub, mul,
// add_scalar, relu, abs, mean, scale), so it needs no new gradient code; tests/test_twin_loss.cpp
// checks the composition against a direct computation and against finite differences.
//
// The answer-token margin (lane amendment A32, arm NA; Claude Opus 5.5, 15 Nov 2026): with per-target weights w (0 or 1; a
// reply's answer-deciding tokens D are 1), rows 0 to 3 are also summed through w, and the margin reads only D, over the whole
// reply's count, against a target scaled by |D| / n:
//   dA^D = (lpw(0) - lpw(1)) / n_A     dB^D = (lpw(2) - lpw(3)) / n_B
//   loss = lambda * [mean_g relu(m |D_A| / n_A - dA^D) + mean_g relu(m |D_B| / n_B - dB^D)] + mu * mean_g |dN|
// i.e. relu((|D|/n) (m - mean over D of the contrast)): each answer token gets the coefficient 1/n that M or N gives it while its
// hinge is active, and every other token none from the margin. dN is unchanged: the whole reply, unweighted. Empty w is A31's loss.
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "microtorch/device_cache.hpp"
#include "microtorch/ops.hpp"

namespace microtorch {
namespace posttrain {

// The reply targets of one batch of twin rows, laid out as the logits' rows.
struct TwinTargets {
    std::vector<int> pos;      // logits row of each reply target (row r's targets are at r * L + t)
    std::vector<int> tgt;      // the target id there
    std::vector<int> pos_row;  // which twin row (0 .. rows-1) the target belongs to
    size_t rows = 0;           // twin rows in the batch = groups * per_group
    size_t groups = 0;
    int per_group = 4;         // 4 (no nuisance row) or 5
    std::vector<float> n_a, n_b;  // reply target counts of yA and yB, per group
    // A32's answer-token margin: per-target weights (empty = the whole reply), and |D| of yA and yB per group
    std::vector<float> w;
    std::vector<float> d_a, d_b;
};

struct TwinStats {
    double loss = 0.0, d_a = 0.0, d_b = 0.0, d_n = 0.0;  // means over groups, per token
    double active = 0.0;  // share of the 2 * groups hinges with m - d > 0
    bool has_n = false;
};

inline void check_twin_targets(const TwinTargets& t, size_t logit_rows) {
    if (t.per_group != 4 && t.per_group != 5) throw std::runtime_error("twin loss: per_group must be 4 or 5");
    if (t.groups == 0 || t.rows != t.groups * static_cast<size_t>(t.per_group))
        throw std::runtime_error("twin loss: rows must be groups * per_group");
    if (t.pos.empty() || t.pos.size() != t.tgt.size() || t.pos.size() != t.pos_row.size())
        throw std::runtime_error("twin loss: one row, target and twin row per reply target");
    if (t.n_a.size() != t.groups || t.n_b.size() != t.groups)
        throw std::runtime_error("twin loss: one reply length per group and reply");
    for (size_t g = 0; g < t.groups; ++g)
        if (!(std::isfinite(t.n_a[g]) && t.n_a[g] >= 1.0f && std::isfinite(t.n_b[g]) && t.n_b[g] >= 1.0f))
            throw std::runtime_error("twin loss: reply lengths must be finite and at least 1");
    // every twin row owns one block of L = logit_rows / rows consecutive logits rows, and its targets lie in it
    if (logit_rows % t.rows != 0) throw std::runtime_error("twin loss: logits rows are not rows * L");
    const size_t L = logit_rows / t.rows;
    std::vector<size_t> seen(t.rows, 0);
    for (size_t i = 0; i < t.pos.size(); ++i) {
        if (t.pos[i] < 0 || static_cast<size_t>(t.pos[i]) >= logit_rows)
            throw std::runtime_error("twin loss: target position out of range");
        if (t.pos_row[i] < 0 || static_cast<size_t>(t.pos_row[i]) >= t.rows)
            throw std::runtime_error("twin loss: twin row out of range");
        if (static_cast<size_t>(t.pos[i]) / L != static_cast<size_t>(t.pos_row[i]))
            throw std::runtime_error("twin loss: a target position outside its twin row's block");
        if (t.tgt[i] < 0) throw std::runtime_error("twin loss: negative target id");
        ++seen[t.pos_row[i]];
    }
    if (!t.w.empty()) {
        if (t.w.size() != t.pos.size()) throw std::runtime_error("twin loss: one weight per reply target");
        if (t.d_a.size() != t.groups || t.d_b.size() != t.groups)
            throw std::runtime_error("twin loss: one answer-token count per group and reply");
        std::vector<double> wsum(t.rows, 0.0);
        for (size_t i = 0; i < t.w.size(); ++i) {
            if (!(t.w[i] == 0.0f || t.w[i] == 1.0f)) throw std::runtime_error("twin loss: answer-token weights must be 0 or 1");
            wsum[t.pos_row[i]] += t.w[i];
        }
        for (size_t g = 0; g < t.groups; ++g) {
            const size_t k = g * t.per_group;
            // a reply's answer tokens are the same in each context it is read in, and at least one
            if (wsum[k] < 1.0 || wsum[k] != wsum[k + 1] || wsum[k] != static_cast<double>(t.d_a[g]) || wsum[k + 2] < 1.0 ||
                wsum[k + 2] != wsum[k + 3] || wsum[k + 2] != static_cast<double>(t.d_b[g]) || t.d_a[g] > t.n_a[g] ||
                t.d_b[g] > t.n_b[g])
                throw std::runtime_error("twin loss: group " + std::to_string(g) + " answer-token weights do not pair up");
        }
        if (t.per_group == 5)
            for (size_t i = 0; i < t.w.size(); ++i)
                if (static_cast<size_t>(t.pos_row[i]) % 5 == 4 && t.w[i] != 0.0f)
                    throw std::runtime_error("twin loss: the nuisance row carries no answer-token weight");
    }
    for (size_t g = 0; g < t.groups; ++g) {
        const size_t k = g * t.per_group;
        // a reply is scored by the same number of targets in each context it is read in
        if (seen[k] == 0 || seen[k] != seen[k + 1] || seen[k] != static_cast<size_t>(t.n_a[g]) ||
            seen[k + 2] == 0 || seen[k + 2] != seen[k + 3] || seen[k + 2] != static_cast<size_t>(t.n_b[g]) ||
            (t.per_group == 5 && seen[k + 4] != seen[k]))
            throw std::runtime_error("twin loss: group " + std::to_string(g) + " reply targets do not pair up");
    }
}

// The reply log-probability of every twin row, [rows, 1], on the tape; with weighted != nullptr and t.w set, also the
// answer-token sums (each target's log-probability times its weight), [rows, 1].
inline Var twin_row_logp(const Var& logits, const TwinTargets& t, Var* weighted = nullptr) {
    check_twin_targets(t, logits->data.rows());
    for (int id : t.tgt)
        if (static_cast<size_t>(id) >= logits->data.cols()) throw std::runtime_error("twin loss: target id past the vocabulary");
    const size_t n = t.pos.size();
    const Var lg = ops::gather_rows(logits, t.pos);  // [n, C]
    const Var lse = ops::logsumexp_row(lg);          // [n, 1]
    std::vector<int> iota(n);
    for (size_t i = 0; i < n; ++i) iota[i] = static_cast<int>(i);
    const Var tg = ops::gather_elems(lg, iota, t.tgt, 1);  // [n, 1]
    const Var lp = ops::sub(tg, lse);
    if (weighted && !t.w.empty()) {
        Matrix wm(n, 1);
        for (size_t i = 0; i < n; ++i) wm(i, 0) = t.w[i];
        *weighted = ops::scatter_add_rows(ops::mul(lp, make_var(wm)), t.pos_row, t.rows);
    }
    return ops::scatter_add_rows(lp, t.pos_row, t.rows);  // [rows, 1]
}

inline Var twin_loss(const Var& logits, const TwinTargets& t, float margin, float lambda, float mu,
                     TwinStats* st = nullptr) {
    if (!std::isfinite(lambda) || !std::isfinite(mu) || !std::isfinite(margin) || !(lambda >= 0.0f) || !(mu >= 0.0f) ||
        !(lambda > 0.0f || mu > 0.0f))
        throw std::runtime_error("twin loss: lambda, mu and the margin must be finite, lambda and mu >= 0, not both 0");
    if (mu > 0.0f && t.per_group != 5) throw std::runtime_error("twin loss: mu > 0 needs the nuisance row");
    Var SW;
    const Var S = twin_row_logp(logits, t, &SW);
    const bool answer = !t.w.empty();
    const size_t G = t.groups, k = static_cast<size_t>(t.per_group);
    auto pick_from = [&](const Var& V, size_t j) {
        std::vector<int> idx(G);
        for (size_t g = 0; g < G; ++g) idx[g] = static_cast<int>(g * k + j);
        return ops::gather_rows(V, idx);
    };
    auto pick = [&](size_t j) { return pick_from(S, j); };
    auto mpick = [&](size_t j) { return pick_from(answer ? SW : S, j); };
    Matrix inv_a(G, 1), inv_b(G, 1);
    for (size_t g = 0; g < G; ++g) {
        inv_a(g, 0) = 1.0f / t.n_a[g];
        inv_b(g, 0) = 1.0f / t.n_b[g];
    }
    const Var ia = make_var(inv_a), ib = make_var(inv_b);
    const Var s0 = pick(0);
    const Var dA = ops::mul(ops::sub(mpick(0), mpick(1)), ia);
    const Var dB = ops::mul(ops::sub(mpick(2), mpick(3)), ib);
    // the hinge targets: m, or m |D| / n per group for the answer-token margin
    std::vector<float> ta(G, margin), tb(G, margin);
    if (answer)
        for (size_t g = 0; g < G; ++g) {
            ta[g] = margin * t.d_a[g] / t.n_a[g];
            tb[g] = margin * t.d_b[g] / t.n_b[g];
        }
    Var loss;
    if (lambda > 0.0f) {
        Var hA, hB;
        if (answer) {
            Matrix ma(G, 1), mb(G, 1);
            for (size_t g = 0; g < G; ++g) {
                ma(g, 0) = ta[g];
                mb(g, 0) = tb[g];
            }
            hA = ops::relu(ops::add(ops::scale(dA, -1.0f), make_var(ma)));
            hB = ops::relu(ops::add(ops::scale(dB, -1.0f), make_var(mb)));
        } else {
            hA = ops::relu(ops::add_scalar(ops::scale(dA, -1.0f), margin));
            hB = ops::relu(ops::add_scalar(ops::scale(dB, -1.0f), margin));
        }
        loss = ops::scale(ops::add(ops::mean(hA), ops::mean(hB)), lambda);
    }
    Var dN;
    if (t.per_group == 5) dN = ops::mul(ops::sub(s0, pick(4)), ia);
    if (mu > 0.0f) {
        const Var inv = ops::scale(ops::mean(ops::abs(dN)), mu);
        loss = loss ? ops::add(loss, inv) : inv;
    }
    if (st) {
        // results may be device-resident inside a step window: read them on the host first
        device::materialize(loss->data);
        device::materialize(dA->data);
        device::materialize(dB->data);
        if (dN) device::materialize(dN->data);
        double a = 0, b = 0, nn = 0, act = 0;
        for (size_t g = 0; g < G; ++g) {
            a += dA->data(g, 0);
            b += dB->data(g, 0);
            act += (ta[g] - dA->data(g, 0) > 0.0f) + (tb[g] - dB->data(g, 0) > 0.0f);
            if (dN) nn += std::fabs(dN->data(g, 0));
        }
        st->loss = loss->data(0, 0);
        st->d_a = a / G;
        st->d_b = b / G;
        st->d_n = dN ? nn / G : 0.0;
        st->has_n = static_cast<bool>(dN);
        st->active = act / (2.0 * G);
    }
    return loss;
}

}  // namespace posttrain
}  // namespace microtorch
