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
// Own-context floor (lane amendment A34; Claude Opus 5.5, 15 Nov 2026). With floor mu_f > 0 and a
// frozen per-token reference for each group's own-context terms (ref_a for lp(yA|A)/n_A, ref_b for
// lp(yB|B)/n_B, set by the caller for the step), the loss adds
//   mu_f * [mean_g relu(ref_a - lp(0)/n_A) + mean_g relu(ref_b - lp(2)/n_B)]
// built from the same checked ops (sub, mul, relu, mean, scale, add). mu_f = 0 adds nothing to the tape,
// so the loss and its gradient are A31's bit for bit.
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
    std::vector<float> ref_a, ref_b;  // A34's floor: the per-token own-context references, per group (floor > 0 only)
};

struct TwinStats {
    double loss = 0.0, d_a = 0.0, d_b = 0.0, d_n = 0.0;  // means over groups, per token
    double active = 0.0;  // share of the 2 * groups hinges with m - d > 0
    bool has_n = false;
    // A34's floor (has_floor only): its term's value, the share of the 2 * groups own-context sides below their
    // reference, the mean signed gap ref - own over those sides, and the largest |ref - own|
    bool has_floor = false;
    double floor_loss = 0.0, floor_active = 0.0, floor_gap = 0.0, floor_gap_absmax = 0.0;
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
    for (size_t g = 0; g < t.groups; ++g) {
        const size_t k = g * t.per_group;
        // a reply is scored by the same number of targets in each context it is read in
        if (seen[k] == 0 || seen[k] != seen[k + 1] || seen[k] != static_cast<size_t>(t.n_a[g]) ||
            seen[k + 2] == 0 || seen[k + 2] != seen[k + 3] || seen[k + 2] != static_cast<size_t>(t.n_b[g]) ||
            (t.per_group == 5 && seen[k + 4] != seen[k]))
            throw std::runtime_error("twin loss: group " + std::to_string(g) + " reply targets do not pair up");
    }
}

// The reply log-probability of every twin row, [rows, 1], on the tape.
inline Var twin_row_logp(const Var& logits, const TwinTargets& t) {
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
    return ops::scatter_add_rows(lp, t.pos_row, t.rows);  // [rows, 1]
}

inline Var twin_loss(const Var& logits, const TwinTargets& t, float margin, float lambda, float mu,
                     TwinStats* st = nullptr, float floor = 0.0f) {
    if (!std::isfinite(lambda) || !std::isfinite(mu) || !std::isfinite(margin) || !(lambda >= 0.0f) || !(mu >= 0.0f) ||
        !(lambda > 0.0f || mu > 0.0f))
        throw std::runtime_error("twin loss: lambda, mu and the margin must be finite, lambda and mu >= 0, not both 0");
    if (mu > 0.0f && t.per_group != 5) throw std::runtime_error("twin loss: mu > 0 needs the nuisance row");
    if (!std::isfinite(floor) || !(floor >= 0.0f)) throw std::runtime_error("twin loss: the floor weight must be finite and >= 0");
    if (floor > 0.0f) {
        if (t.ref_a.size() != t.groups || t.ref_b.size() != t.groups)
            throw std::runtime_error("twin loss: the floor needs one reference per group and reply");
        for (size_t g = 0; g < t.groups; ++g)
            if (!std::isfinite(t.ref_a[g]) || !std::isfinite(t.ref_b[g]))
                throw std::runtime_error("twin loss: floor references must be finite");
    }
    const Var S = twin_row_logp(logits, t);
    const size_t G = t.groups, k = static_cast<size_t>(t.per_group);
    auto pick = [&](size_t j) {
        std::vector<int> idx(G);
        for (size_t g = 0; g < G; ++g) idx[g] = static_cast<int>(g * k + j);
        return ops::gather_rows(S, idx);
    };
    Matrix inv_a(G, 1), inv_b(G, 1);
    for (size_t g = 0; g < G; ++g) {
        inv_a(g, 0) = 1.0f / t.n_a[g];
        inv_b(g, 0) = 1.0f / t.n_b[g];
    }
    const Var ia = make_var(inv_a), ib = make_var(inv_b);
    const Var s0 = pick(0);
    const Var dA = ops::mul(ops::sub(s0, pick(1)), ia);
    const Var dB = ops::mul(ops::sub(pick(2), pick(3)), ib);
    Var loss;
    if (lambda > 0.0f) {
        const Var hA = ops::relu(ops::add_scalar(ops::scale(dA, -1.0f), margin));
        const Var hB = ops::relu(ops::add_scalar(ops::scale(dB, -1.0f), margin));
        loss = ops::scale(ops::add(ops::mean(hA), ops::mean(hB)), lambda);
    }
    Var dN;
    if (t.per_group == 5) dN = ops::mul(ops::sub(s0, pick(4)), ia);
    if (mu > 0.0f) {
        const Var inv = ops::scale(ops::mean(ops::abs(dN)), mu);
        loss = loss ? ops::add(loss, inv) : inv;
    }
    Var ownA, ownB;  // A34's floor: per-token own-context terms; gaps ref - own
    Matrix gapA(G, 1), gapB(G, 1);
    if (floor > 0.0f) {
        Matrix ra(G, 1), rb(G, 1);
        for (size_t g = 0; g < G; ++g) {
            ra(g, 0) = t.ref_a[g];
            rb(g, 0) = t.ref_b[g];
        }
        ownA = ops::mul(s0, ia);
        ownB = ops::mul(pick(2), ib);
        const Var fA = ops::relu(ops::sub(make_var(ra), ownA));
        const Var fB = ops::relu(ops::sub(make_var(rb), ownB));
        const Var fl = ops::scale(ops::add(ops::mean(fA), ops::mean(fB)), floor);
        loss = ops::add(loss, fl);  // loss exists: lambda > 0 or mu > 0
        if (st) {
            device::materialize(fl->data);
            device::materialize(ownA->data);
            device::materialize(ownB->data);
            st->floor_loss = fl->data(0, 0);
            for (size_t g = 0; g < G; ++g) {
                gapA(g, 0) = ra(g, 0) - ownA->data(g, 0);
                gapB(g, 0) = rb(g, 0) - ownB->data(g, 0);
            }
        }
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
            act += (margin - dA->data(g, 0) > 0.0f) + (margin - dB->data(g, 0) > 0.0f);
            if (dN) nn += std::fabs(dN->data(g, 0));
        }
        st->loss = loss->data(0, 0);
        st->d_a = a / G;
        st->d_b = b / G;
        st->d_n = dN ? nn / G : 0.0;
        st->has_n = static_cast<bool>(dN);
        st->active = act / (2.0 * G);
        st->has_floor = floor > 0.0f;
        if (st->has_floor) {
            double fa = 0, gs = 0, gm = 0;
            for (size_t g = 0; g < G; ++g)
                for (double x : {static_cast<double>(gapA(g, 0)), static_cast<double>(gapB(g, 0))}) {
                    fa += x > 0.0;
                    gs += x;
                    gm = std::max(gm, std::fabs(x));
                }
            st->floor_active = fa / (2.0 * G);
            st->floor_gap = gs / (2.0 * G);
            st->floor_gap_absmax = gm;
        }
    }
    return loss;
}

}  // namespace posttrain
}  // namespace microtorch
