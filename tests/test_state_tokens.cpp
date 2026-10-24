// Per-turn state tokens in the llama family (nn::LlamaConfig state_k /
// state_mask; ir.hpp embedding state_weight, attention mask "turn_state",
// kv slot evict "turn_state"; experiments/tiny_chat_long_2026-10/
// IR_SPEC_DRAFT.md sections 1 and 5).
//   1. The codes (ops::turn_state_codes): a worked example, the incremental
//      tracker equal to the batch form, restarts per stacked sequence, and
//      partial or out-of-order blocks.
//   2. The template contract (nn::insert_state_ids, nn::state_ids_error):
//      blocks after every reply's <|endoftext|>, none after other ones, the
//      early-stop rule (<|endoftext|> and a block before a <|user|> that
//      follows an unfinished reply), and raw-ids validation.
//   3. k > 0 but no state id: logits and every gradient bit for bit those of
//      the plain model (both masks), and the IR's attention keeps no mask.
//   4. The mask's meaning: in a one-layer model the logits of a later turn do
//      not depend on an earlier turn's text at all (bit for bit), but do under
//      the causal mask; with two layers the earlier turn reaches them through
//      the state rows.
//   5. Finite-difference gradchecks: ops::attention_heads_turn_state on
//      random q, k, v over two stacked sequences, and a whole state-token
//      model (state embedding included), plain and recursed.
//   6. Prefill equals decode: forward_cached token by token and chunked
//      against forward (rounding), turn_state and causal masks, plain and
//      1 + [2] x 2 + 1 recursion.
//   7. Eviction: a cache that drops finished turns' text rows gives bit for
//      bit the logits of one that keeps every row, token by token and
//      chunked, holding n k state rows plus the current turn; an early-
//      stopped reply closed by the template; a rewind into an evicted turn
//      (truncate returns that segment's first position, and re-prefilling
//      equals a run that never evicted); a rewind inside the current turn.
//   8. Refusals: schedule mor, ids past the state ids, evict without
//      turn_state.
//   9. Training steps: on a CUDA build with MICROTORCH_DEVICE=cuda
//      MICROTORCH_DEVICE_OPS=1 the device op set (the turn_state softmax
//      kernel) with step residency and deferred downloads against the host
//      path; on a CPU build host against host, bit for bit.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/ir.hpp"
#include "microtorch/llama.hpp"

using namespace microtorch;

namespace {

bool on_device() { return device::device_ops_enabled() && device::get() == device::Device::CUDA; }

// Vocabulary 0 .. 36; <|endoftext|> 1, <|user|> 2, <|assistant|> 3 (the BPE
// specials' order); k = 3 state ids 37, 38, 39.
constexpr size_t kV = 37, kK = 3;
constexpr int EOT = 1, USR = 2, AST = 3, S0 = 37, S1 = 38, S2 = 39;

nn::StateTemplate tmpl() {
    nn::StateTemplate t;
    t.vocab = kV;
    t.k = kK;
    t.eot = EOT;
    t.user = USR;
    t.assistant = AST;
    return t;
}

nn::LlamaConfig tiny(size_t state_k, const std::string& mask = "turn_state", size_t layers = 2) {
    nn::LlamaConfig c;
    c.vocab = kV;
    c.d = 16;
    c.n_heads = 2;
    c.d_ff = 24;
    c.n_ctx = 48;
    c.n_layers = layers;
    c.state_k = state_k;
    c.state_mask = mask;
    return c;
}

nn::LlamaConfig recursed(size_t state_k) {
    nn::LlamaConfig c = tiny(state_k, "turn_state", 4);
    c.recursion.prelude = 1;
    c.recursion.block = 2;
    c.recursion.repeats = 2;
    c.recursion.coda = 1;
    return c;
}

// A three-turn conversation (text ids), and the same with the template's
// state blocks: 2 U U 3 A A 1 [S] 2 U 3 A A A 1 [S] 2 U U 3 A.
const std::vector<int> kText = {2, 9, 11, 3, 14, 15, 1, 2, 20, 3, 21, 22, 23, 1, 2, 30, 31, 3, 33, 34};

std::vector<int> conversation() { return nn::insert_state_ids(kText, tmpl()); }

size_t bits_differ(const Matrix& a, const Matrix& b) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols());
    size_t n = 0;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j) {
            const float x = a(i, j), y = b(i, j);
            n += std::memcmp(&x, &y, sizeof x) != 0;
        }
    return n;
}

double rel(const Matrix& a, const Matrix& b) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols());
    double s = 0, w = 0;
    for (size_t i = 0; i < b.rows(); ++i)
        for (size_t j = 0; j < b.cols(); ++j) {
            s = std::max(s, (double)std::fabs(b(i, j)));
            w = std::max(w, std::fabs((double)a(i, j) - b(i, j)));
        }
    return w / std::max(s, 1e-30);
}

void randomise(nn::Module& m, unsigned seed, float scale = 0.3f) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto& [name, p] : m.named_parameters()) {
        const bool norm = name.find("norm") != std::string::npos;
        for (size_t i = 0; i < p->data.rows(); ++i)
            for (size_t j = 0; j < p->data.cols(); ++j) p->data(i, j) = norm ? 1.0f + 0.2f * nd(g) : nd(g);
    }
}

void clear_grads(nn::Module& m) {
    for (auto& p : m.parameters()) p->grad = Matrix();
}

std::map<std::string, Matrix> grads(const nn::Module& m) {
    std::map<std::string, Matrix> g;
    for (const auto& [n, p] : m.named_parameters()) g.emplace(n, p->grad);
    return g;
}

// Targets: the next id, a state target (and the last state position) mapped
// to 0 (the masked rows of mtstudio's loss); a model test needs only a loss.
std::vector<int> targets_of(const std::vector<int>& ids) {
    std::vector<int> y(ids.size(), 0);
    for (size_t i = 0; i + 1 < ids.size(); ++i) y[i] = ids[i + 1] >= static_cast<int>(kV) ? 0 : ids[i + 1];
    return y;
}

// ---- 1. codes ----
void test_codes() {
    const std::vector<int> ids = conversation();
    // 2 9 11 3 14 15 1 S S S | 2 20 3 21 22 23 1 S S S | 2 30 31 3 33 34
    const std::vector<int> want_ids = {2, 9, 11, 3, 14, 15, 1, S0, S1, S2, 2, 20, 3, 21, 22, 23, 1, S0, S1, S2,
                                       2, 30, 31, 3, 33, 34};
    CHECK(ids == want_ids);
    const std::vector<int> c = ops::turn_state_codes(ids, kV, kK);
    const std::vector<int> want = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4};
    CHECK(c == want);
    int seg = 0, run = 0;
    for (size_t p = 0; p < ids.size(); ++p) CHECK(ops::turn_state_step(ids[p], kV, kK, seg, run) == c[p]);
    CHECK(seg == 2 && run == 0);
    // Two stacked sequences restart.
    std::vector<int> two = ids;
    two.insert(two.end(), ids.begin(), ids.end());
    const std::vector<int> c2 = ops::turn_state_codes(two, kV, kK, ids.size());
    for (size_t p = 0; p < ids.size(); ++p) CHECK(c2[p] == c[p] && c2[ids.size() + p] == c[p]);
    // A window that starts inside a block: S1 S2 are state positions of
    // segment 0 and complete nothing; S0 S2 S1 is no block; S0 S0 S1 S2 is one.
    CHECK(ops::turn_state_codes({S1, S2, 5, 1, S0, S1, S2, 7}, kV, kK) ==
          (std::vector<int>{1, 1, 0, 0, 1, 1, 1, 2}));
    CHECK(ops::turn_state_codes({S0, S2, S1, 5}, kV, kK) == (std::vector<int>{1, 1, 1, 0}));
    CHECK(ops::turn_state_codes({S0, S0, S1, S2, 5}, kV, kK) == (std::vector<int>{1, 1, 1, 1, 2}));
    std::printf("  codes: worked example, incremental = batch, stacked restart, partial blocks\n");
}

// ---- 2. template ----
void test_template() {
    const nn::StateTemplate t = tmpl();
    nn::StateInsertCount n;
    // Every reply's <|endoftext|> gets a block; a document's does not.
    const std::vector<int> doc = {5, 6, 1, 2, 7, 3, 8, 1, 9, 1};
    CHECK(nn::insert_state_ids(doc, t, &n) == (std::vector<int>{5, 6, 1, 2, 7, 3, 8, 1, S0, S1, S2, 9, 1}));
    CHECK(n.blocks == 1 && n.closed == 0);
    // A user turn without a reply is no exchange.
    CHECK(nn::insert_state_ids({2, 7, 1, 5}, t) == (std::vector<int>{2, 7, 1, 5}));
    // Early stop: the reply 8 8 has no <|endoftext|> before the next <|user|>.
    const std::vector<int> cut = {2, 7, 3, 8, 8, 2, 9, 3, 10, 1};
    CHECK(nn::insert_state_ids(cut, t, &n) ==
          (std::vector<int>{2, 7, 3, 8, 8, 1, S0, S1, S2, 2, 9, 3, 10, 1, S0, S1, S2}));
    CHECK(n.blocks == 2 && n.closed == 1);
    bool threw = false;
    try {
        nn::insert_state_ids({2, S0}, t);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
    // Raw ids.
    CHECK(nn::state_ids_error(conversation(), t).empty());
    CHECK(nn::state_ids_error({2, 7, 3, 8, 1}, t).empty());              // a block may be absent
    CHECK(!nn::state_ids_error({2, 7, 3, 8, 1, S0, S1}, t).empty());     // incomplete
    CHECK(!nn::state_ids_error({2, 7, 3, 8, 1, S1, S0, S2}, t).empty()); // out of order
    CHECK(!nn::state_ids_error({5, 1, S0, S1, S2}, t).empty());          // not after a reply
    CHECK(!nn::state_ids_error({2, 7, 3, 8, S0, S1, S2}, t).empty());    // no <|endoftext|>
    CHECK(!nn::state_ids_error({2, 7, 3, 8, 1, S0, S1, S2, S0, S1, S2}, t).empty());  // two blocks
    CHECK(!nn::state_ids_error({2, 40}, t).empty());
    std::printf("  template: reply blocks only, early-stop rule (%zu closed), raw-ids validation\n", n.closed);
}

// ---- 3. no state id ----
void test_no_state_is_plain() {
    for (const char* mask : {"turn_state", "causal"}) {
        nn::Llama plain(tiny(0), 42), st(tiny(kK, mask), 42);
        CHECK(st.state_embd != nullptr && plain.state_embd == nullptr);
        Var a = plain.forward(kText), b = st.forward(kText);
        const size_t fwd = bits_differ(a->data, b->data);
        const std::vector<int> y = targets_of(kText);
        backward(ops::cross_entropy(a, y));
        backward(ops::cross_entropy(b, y));
        const auto ga = grads(plain), gb = grads(st);
        size_t diff = 0;
        for (const auto& [n, g] : ga) diff += bits_differ(g, gb.at(n));
        CHECK(gb.at("state_embd.weight").rows() == 0);  // never read
        // Two stacked sequences too.
        std::vector<int> two = kText;
        two.insert(two.end(), kText.begin(), kText.end());
        const size_t fwd2 = bits_differ(plain.forward(two, kText.size())->data, st.forward(two, kText.size())->data);
        std::printf("  k = %zu, mask %s, no state id: logits %zu + %zu bits differ, %zu gradients %zu bits differ\n",
                    kK, mask, fwd, fwd2, ga.size(), diff);
        CHECK(fwd == 0 && fwd2 == 0 && diff == 0);
    }
    // The IR: the attention node carries mask turn_state only under that mask.
    const ir::json ts = nn::Llama(tiny(kK), 1).export_ir(), cs = nn::Llama(tiny(kK, "causal"), 1).export_ir(),
                   pl = nn::Llama(tiny(0), 1).export_ir();
    auto att = [](const ir::json& d) {
        for (const auto& n : d["block"]["nodes"])
            if (n["op"] == "attention") return n;
        return ir::json();
    };
    CHECK(att(ts)["mask"] == "turn_state" && !att(cs).contains("mask") && !att(pl).contains("mask"));
    CHECK(ts["block"]["state"][0]["evict"] == "turn_state" && !cs["block"]["state"][0].contains("evict"));
    CHECK(ts["embed"]["nodes"][0]["state_weight"] == "state_embd.weight" &&
          cs["embed"]["nodes"][0]["state_weight"] == "state_embd.weight" &&
          !pl["embed"]["nodes"][0].contains("state_weight"));
    CHECK(ts["family_hint"] == "llama_state" && cs["family_hint"] == "llama_state" && pl["family_hint"] == "llama");
    std::printf("  IR: mask and evict under turn_state only, state_weight under both masks, family llama_state\n");
}

// ---- 4. the mask's meaning ----
void test_mask_meaning() {
    const std::vector<int> ids = conversation();
    std::vector<int> alt = ids;
    alt[1] = 25;  // turn 0's user text
    alt[4] = 26;  // and its reply
    const size_t turn2 = 20;  // segment 2 starts here
    auto later_bits = [&](const nn::LlamaConfig& cfg) {
        nn::Llama m(cfg, 9);
        randomise(m, 4);
        NoGrad ng;
        const Matrix a = m.forward(ids)->data, b = m.forward(alt)->data;
        size_t d = 0;
        for (size_t i = turn2; i < ids.size(); ++i)
            for (size_t j = 0; j < a.cols(); ++j) {
                const float x = a(i, j), y = b(i, j);
                d += std::memcmp(&x, &y, sizeof x) != 0;
            }
        return d;
    };
    const size_t one_ts = later_bits(tiny(kK, "turn_state", 1)), one_c = later_bits(tiny(kK, "causal", 1)),
                 two_ts = later_bits(tiny(kK, "turn_state", 2));
    std::printf("  mask: turn 2's logits after editing turn 0's text: 1 layer turn_state %zu bits differ (must be 0), "
                "causal %zu; 2 layers turn_state %zu (through the state rows)\n",
                one_ts, one_c, two_ts);
    CHECK(one_ts == 0 && one_c > 0 && two_ts > 0);
}

// ---- 5. gradchecks ----
void fd_op() {
    const size_t H = 2, dk = 3, d = H * dk, sl = 13;
    // Two sequences: turns with complete blocks and a window that starts
    // inside one (codes from ids, k = 3).
    const std::vector<int> ids = {2, 9, 3, 14, 1, S0, S1, S2, 2, 20, 3, 21, 22,
                                  S1, S2, 5, 6, 1, S0, S1, S2, 2, 7, 3, 8, 9};
    const std::vector<int> code = ops::turn_state_codes(ids, kV, kK, sl);
    std::mt19937 g(5);
    std::normal_distribution<float> nd(0.0f, 0.7f);
    Var qkv = make_var(Matrix(ids.size(), 3 * d), true);
    for (size_t i = 0; i < qkv->data.rows(); ++i)
        for (size_t j = 0; j < qkv->data.cols(); ++j) qkv->data(i, j) = nd(g);
    Matrix R(ids.size(), d);
    for (size_t i = 0; i < R.rows(); ++i)
        for (size_t j = 0; j < R.cols(); ++j) R(i, j) = nd(g);
    auto loss = [&] {
        Var y = ops::attention_heads_turn_state(qkv, H, 0.6f, sl, code);
        return ops::mean(ops::mul(y, make_var(R)));
    };
    qkv->grad = Matrix();
    backward(loss());
    const Matrix an = qkv->grad;
    double worst = 0;
    const float eps = 1e-2f;
    for (size_t i = 0; i < qkv->data.rows(); ++i)
        for (size_t j = 0; j < qkv->data.cols(); ++j) {
            double fp, fm;
            {
                NoGrad ng;
                const float keep = qkv->data(i, j);
                qkv->data(i, j) = keep + eps;
                fp = loss()->data(0, 0);
                qkv->data(i, j) = keep - eps;
                fm = loss()->data(0, 0);
                qkv->data(i, j) = keep;
            }
            const double fd = (fp - fm) / (2.0 * eps), a = an(i, j);
            worst = std::max(worst, std::fabs(fd - a) / std::max({std::fabs(fd), std::fabs(a), 1e-3}));
        }
    // Every entry of q, k and v of a row is checked; the forward against a
    // dense masked reference (scores of invisible keys never enter).
    Matrix ref(ids.size(), d);
    for (size_t i = 0; i < ids.size(); ++i) {
        const size_t b0 = (i / sl) * sl;
        for (size_t h = 0; h < H; ++h) {
            std::vector<double> s;
            std::vector<size_t> js;
            for (size_t j = b0; j <= i; ++j) {
                if (!ops::turn_state_visible(code[i], code[j])) continue;
                double dot = 0;
                for (size_t c = 0; c < dk; ++c) dot += (double)qkv->data(i, h * dk + c) * qkv->data(j, d + h * dk + c);
                s.push_back(dot * 0.6);
                js.push_back(j);
            }
            const double mx = *std::max_element(s.begin(), s.end());
            double z = 0;
            for (double& x : s) z += (x = std::exp(x - mx));
            for (size_t c = 0; c < dk; ++c) {
                double acc = 0;
                for (size_t t = 0; t < js.size(); ++t) acc += s[t] / z * qkv->data(js[t], 2 * d + h * dk + c);
                ref(i, h * dk + c) = static_cast<float>(acc);
            }
        }
    }
    Matrix y;
    {
        NoGrad ng;
        y = ops::attention_heads_turn_state(qkv, H, 0.6f, sl, code)->data;
    }
    const double fr = rel(y, ref);
    std::printf("  FD gradcheck attention_heads_turn_state (2 sequences of %zu, %zu entries): worst rel %.2e; forward "
                "vs a double-precision masked reference rel %.2e\n",
                sl, qkv->data.rows() * qkv->data.cols(), worst, fr);
    CHECK(worst < 2e-2 && fr < 1e-5);
}

void fd_model(const nn::LlamaConfig& cfg, const char* label) {
    nn::Llama m(cfg, 5);
    randomise(m, 9);
    const std::vector<int> ids = conversation(), y = targets_of(ids);
    auto loss = [&] { return ops::cross_entropy(m.forward(ids), y); };
    clear_grads(m);
    backward(loss());
    std::mt19937 rng(17);
    double worst = 0, wfd = 0, wan = 0;
    std::string wname;
    size_t probes = 0;
    for (const auto& [n, p] : m.named_parameters()) {
        const int k = n == "state_embd.weight" ? 9 : 2;
        for (int t = 0; t < k; ++t) {
            const size_t i = n == "state_embd.weight" ? t % kK : rng() % p->data.rows(),
                         j = rng() % p->data.cols();
            const double an = p->grad(i, j);
            double best = 1e30, bfd = 0;
            for (const float eps : {1e-2f, 3e-3f}) {
                double fp, fm;
                {
                    NoGrad ng;
                    const float keep = p->data(i, j);
                    p->data(i, j) = keep + eps;
                    fp = loss()->data(0, 0);
                    p->data(i, j) = keep - eps;
                    fm = loss()->data(0, 0);
                    p->data(i, j) = keep;
                }
                const double fd = (fp - fm) / (2.0 * eps);
                const double r = std::fabs(fd - an) / std::max({std::fabs(fd), std::fabs(an), 1e-2});
                if (r < best) best = r, bfd = fd;
            }
            if (best > worst) worst = best, wfd = bfd, wan = an, wname = n;
            ++probes;
        }
    }
    std::printf("  FD gradcheck %s: %zu entries (9 of state_embd), worst rel %.2e (%s: fd %.5f, analytic %.5f)\n",
                label, probes, worst, wname.c_str(), wfd, wan);
    CHECK(worst < 2e-2);
}

// ---- 6./7. cache ----
// Logits of `ids` through a cache fed in chunks of `chunk` tokens (0 = one
// call), optionally evicting.
Matrix cached(const nn::Llama& m, const std::vector<int>& ids, size_t chunk, bool evict, nn::LlamaKV* out = nullptr,
              std::vector<size_t>* rows_per_turn = nullptr) {
    nn::LlamaKV kv;
    kv.evict = evict;
    Matrix all(ids.size(), m.cfg.vocab);
    size_t at = 0;
    const size_t step = chunk == 0 ? ids.size() : chunk;
    while (at < ids.size()) {
        const size_t n = std::min(step, ids.size() - at);
        const Matrix l = m.forward_cached(std::vector<int>(ids.begin() + at, ids.begin() + at + n), kv)->data;
        for (size_t r = 0; r < n; ++r)
            for (size_t c = 0; c < l.cols(); ++c) all(at + r, c) = l(r, c);
        at += n;
        if (rows_per_turn && at < ids.size() && ids[at] == USR) rows_per_turn->push_back(kv.rows());
    }
    if (out) *out = kv;
    return all;
}

void test_prefill_decode(const nn::LlamaConfig& cfg, const char* label) {
    nn::Llama m(cfg, 21);
    randomise(m, 6);
    NoGrad ng;
    const std::vector<int> ids = conversation();
    const Matrix full = m.forward(ids)->data;
    const double r1 = rel(cached(m, ids, 1, false), full), r4 = rel(cached(m, ids, 4, false), full),
                 r0 = rel(cached(m, ids, 0, false), full);
    std::printf("  prefill = decode %s: token by token %.2e, chunks of 4 %.2e, one call %.2e (rel to forward)\n",
                label, r1, r4, r0);
    const double tol = on_device() ? 1e-4 : 1e-5;
    CHECK(r1 < tol && r4 < tol && r0 < tol);
}

void test_eviction(const nn::LlamaConfig& cfg, const char* label) {
    nn::Llama m(cfg, 23);
    randomise(m, 7);
    NoGrad ng;
    const std::vector<int> ids = conversation();
    size_t worst_bits = 0;
    double worst_rel = 0;
    for (size_t chunk : {1, 3, 0}) {
        nn::LlamaKV a, b;
        std::vector<size_t> ra, rb;
        const Matrix keep = cached(m, ids, chunk, false, &a, &ra), ev = cached(m, ids, chunk, true, &b, &rb);
        worst_bits = std::max(worst_bits, bits_differ(keep, ev));
        worst_rel = std::max(worst_rel, rel(ev, keep));
        CHECK(a.rows() == ids.size() && a.len == ids.size() && b.len == ids.size());
        // After the two finished turns: 2 k state rows plus the current turn's 6.
        CHECK(b.rows() == 2 * kK + 6 && b.evicted == 2);
        if (chunk == 1) {
            // Before each new turn: n k state rows (and nothing of the text).
            CHECK(rb == (std::vector<size_t>{kK, 2 * kK}));
            CHECK(ra == (std::vector<size_t>{10, 20}));
        }
    }
    std::printf("  eviction %s: keep every row vs evict (token by token, chunks of 3, one call): %zu bits differ "
                "(rel %.2e); %zu rows kept of %zu after two turns\n",
                label, worst_bits, worst_rel, 2 * kK + 6, ids.size());
    // EXACT on the host (the cache attention sums the visible rows in
    // ascending order either way); ROUNDING allowed with the device op set.
    if (on_device()) CHECK(worst_rel < 1e-5);
    else CHECK(worst_bits == 0);

    // An early-stopped reply (max_new_tokens): the template closes it, so the
    // next turn is a new segment and its rows are evicted like any other.
    const std::vector<int> cut_text = {2, 9, 11, 3, 14, 15, 15, 2, 20, 3, 21, 1, 2, 30, 3, 33};
    nn::StateInsertCount n;
    const std::vector<int> cut = nn::insert_state_ids(cut_text, tmpl(), &n);
    CHECK(n.closed == 1 && n.blocks == 2);
    CHECK(std::vector<int>(cut.begin() + 7, cut.begin() + 11) == (std::vector<int>{EOT, S0, S1, S2}));
    nn::LlamaKV c1;
    const Matrix ck = cached(m, cut, 1, false), ce = cached(m, cut, 1, true, &c1);
    const size_t cb = bits_differ(ck, ce);
    CHECK((on_device() ? rel(ce, ck) < 1e-5 : cb == 0) && c1.rows() == 2 * kK + 4 && c1.evicted == 2);

    // Rewind into an evicted turn: decode everything with eviction, then go
    // back to inside turn 1 (positions 10 .. 19) and continue differently.
    {
        nn::LlamaKV kv;
        cached(m, ids, 1, true, &kv);
        std::vector<int> edit(ids.begin(), ids.begin() + 13);  // keep turn 1's first three ids
        const std::vector<int> tail = {24, 25, 1, S0, S1, S2, 2, 26, 3, 27};
        edit.insert(edit.end(), tail.begin(), tail.end());
        const size_t kept = kv.truncate(13);
        CHECK(kept == 10);  // segment 1's first position: its text rows were gone
        CHECK(kv.rows() == kK && kv.evicted == 1 && kv.len == 10 && kv.seg == 1);
        Matrix got(edit.size() - kept, m.cfg.vocab);
        for (size_t t = kept; t < edit.size(); ++t) {
            const Matrix l = m.forward_cached({edit[t]}, kv)->data;
            for (size_t c = 0; c < l.cols(); ++c) got(t - kept, c) = l(0, c);
        }
        const Matrix ref = cached(m, edit, 1, false);
        Matrix tailref(edit.size() - kept, m.cfg.vocab);
        for (size_t t = kept; t < edit.size(); ++t)
            for (size_t c = 0; c < ref.cols(); ++c) tailref(t - kept, c) = ref(t, c);
        const size_t rb = bits_differ(got, tailref);
        // Inside the current turn: an ordinary truncation.
        nn::LlamaKV kv2;
        cached(m, ids, 1, true, &kv2);
        const size_t kept2 = kv2.truncate(23);
        CHECK(kept2 == 23 && kv2.rows() == 2 * kK + 3 && kv2.evicted == 2);
        // Rewind into the middle of block 1 (segment 1 again, evicted).
        nn::LlamaKV kv3;
        cached(m, ids, 1, true, &kv3);
        CHECK(kv3.truncate(18) == 10 && kv3.rows() == kK);
        std::printf("  rewind into evicted turn 1: truncate(13) -> %zu, re-prefill vs a run that never evicted: %zu "
                    "bits differ; truncate inside the current turn keeps %zu; into block 1 -> 10\n",
                    kept, rb, kept2);
        CHECK(on_device() ? rel(got, tailref) < 1e-5 : rb == 0);
    }
}

void test_refusals() {
    auto throws = [](const std::function<void()>& f) {
        try {
            f();
        } catch (const std::exception&) {
            return true;
        }
        return false;
    };
    nn::LlamaConfig mor = recursed(kK);
    mor.recursion.schedule = "mor";
    CHECK(throws([&] { nn::Llama m(mor, 1); }));
    nn::LlamaConfig ar = tiny(kK);
    ar.attn_res.on = true;
    CHECK(throws([&] { nn::Llama m(ar, 1); }));
    nn::LlamaConfig bad = tiny(kK, "window");
    CHECK(throws([&] { nn::Llama m(bad, 1); }));
    nn::Llama m(tiny(kK), 1);
    CHECK(throws([&] { NoGrad ng; m.forward({2, 3, 40}); }));
    nn::Llama c(tiny(kK, "causal"), 1);
    CHECK(throws([&] {
        NoGrad ng;
        nn::LlamaKV kv;
        kv.evict = true;
        c.forward_cached({2, 3}, kv);
    }));
    std::printf("  refusals: schedule mor, attention residuals, unknown mask, ids past the state ids, evict under the causal mask\n");
}

struct Run {
    std::vector<float> losses;
    std::map<std::string, Matrix> g1, params;
};

// mtstudio's step (as tests/test_recursion.cpp): two stacked conversations per
// micro-batch, two micro-batches, clip, AdamW.
Run train(bool dev, size_t steps, const nn::LlamaConfig& cfg) {
    Run r;
    device::set_device_ops(dev);
    if (dev) {
        device::set_step_residency(true);
        device::set_defer_downloads(true);
    }
    nn::Llama m(cfg, 41);
    randomise(m, 12, 0.2f);
    m.train();
    nn::AdamW opt(m.parameters(), 3e-3f);
    std::mt19937 data(5);
    const std::vector<int> base = conversation();
    const size_t T = base.size(), B = 2, accum = 2;
    for (size_t step = 1; step <= steps; ++step) {
        device::step_begin();
        opt.zero_grad();
        float lsum = 0;
        for (size_t k = 0; k < accum; ++k) {
            std::vector<int> x;
            for (size_t b = 0; b < B; ++b)
                for (int t : base) {
                    const bool special = t == EOT || t == USR || t == AST || t >= static_cast<int>(kV);
                    x.push_back(special ? t : 4 + static_cast<int>(data() % (kV - 4)));
                }
            std::vector<int> y(x.size(), 0);
            for (size_t i = 0; i + 1 < x.size(); ++i) y[i] = x[i + 1] >= static_cast<int>(kV) ? 0 : x[i + 1];
            Var loss = ops::cross_entropy(m.forward(x, T), y);
            backward(ops::scale(loss, 1.0f / accum));
            lsum += loss->data(0, 0) / accum;
        }
        if (!device::deferring()) device::step_end();
        if (step == 1) {
            device::materialize_all();
            for (const auto& [n, p] : m.named_parameters()) r.g1.emplace(n, p->grad);
        }
        ops::clip_grad_norm(m.parameters(), 1.0f);
        opt.step();
        if (device::deferring()) {
            const auto ps = m.parameters();
            for (const auto& p : ps) device::discard(p->grad);
            if (step < steps) {
                std::vector<const Matrix*> keep;
                for (const auto& p : ps) keep.push_back(&p->data);
                device::step_end_keeping(keep);
            } else {
                device::step_end();
            }
        }
        r.losses.push_back(lsum);
    }
    device::materialize_all();
    r.params = m.state_dict();
    if (dev) {
        device::set_defer_downloads(false);
        device::set_step_residency(false);
    }
    device::set_device_ops(true);
    return r;
}

void test_device_steps(const nn::LlamaConfig& cfg, const char* label) {
    const size_t steps = 4;
    const Run a = train(true, steps, cfg), b = train(false, steps, cfg);
    double lw = 0, gw = 0, pw = 0;
    size_t pbits = 0, ptotal = 0;
    for (size_t i = 0; i < steps; ++i)
        lw = std::max(lw, std::fabs((double)a.losses[i] - b.losses[i]) / std::fabs((double)b.losses[i]));
    for (const auto& [n, g] : b.g1) gw = std::max(gw, rel(a.g1.at(n), g));
    for (const auto& [n, p] : b.params) {
        for (size_t i = 0; i < p.rows(); ++i)
            for (size_t j = 0; j < p.cols(); ++j)
                pw = std::max(pw, std::fabs((double)a.params.at(n)(i, j) - p(i, j)));
        pbits += bits_differ(a.params.at(n), p);
        ptotal += p.rows() * p.cols();
    }
    std::printf("  %s, %zu steps, accum 2 (%s): loss worst rel %.2e, step-1 gradients worst rel %.2e, final "
                "parameters max |diff| %.2e (%zu/%zu bits differ); losses %.4f -> %.4f\n",
                label, steps, on_device() ? "device op set + residency + deferred downloads vs host" : "host vs host",
                lw, gw, pw, pbits, ptotal, b.losses.front(), b.losses.back());
    if (on_device()) {
        CHECK(lw < 1e-4 && gw < 1e-3 && pw < 4 * 3e-3);
    } else {
        CHECK(pbits == 0 && lw == 0 && gw == 0);
    }
}

// The device kernel against the host loops, forward and backward, on one op.
void test_device_op() {
    if (!on_device()) {
        std::printf("  device op: skipped (CPU build or op set off)\n");
        return;
    }
    const size_t H = 2, d = 8, sl = 26;
    const std::vector<int> ids = conversation();
    const std::vector<int> code = ops::turn_state_codes(ids, kV, kK, sl);
    std::mt19937 g(8);
    std::normal_distribution<float> nd(0.0f, 0.7f);
    Matrix q0(ids.size(), 3 * d), R(ids.size(), d);
    for (size_t i = 0; i < q0.rows(); ++i)
        for (size_t j = 0; j < q0.cols(); ++j) q0(i, j) = nd(g);
    for (size_t i = 0; i < R.rows(); ++i)
        for (size_t j = 0; j < R.cols(); ++j) R(i, j) = nd(g);
    auto run = [&](bool dev, Matrix& y, Matrix& gq) {
        device::set_device_ops(dev);
        Var qkv = make_var(q0, true);
        Var out = ops::attention_heads_turn_state(qkv, H, 0.5f, sl, code);
        backward(ops::mean(ops::mul(out, make_var(R))));
        device::materialize(out->data);
        device::materialize(qkv->grad);
        y = out->data;
        gq = qkv->grad;
        device::set_device_ops(true);
    };
    Matrix yd, gd, yh, gh;
    run(true, yd, gd);
    run(false, yh, gh);
    const double ry = rel(yd, yh), rg = rel(gd, gh);
    std::printf("  device op (turn_state softmax kernel) vs host: forward rel %.2e, backward rel %.2e\n", ry, rg);
    CHECK(ry < 1e-5 && rg < 1e-4);
}

}  // namespace

int main() {
    device::set_from_env();
    std::printf("per-turn state tokens (llama family, k = %zu)\n", kK);
    test_codes();
    test_template();
    test_no_state_is_plain();
    test_mask_meaning();
    fd_op();
    fd_model(tiny(kK), "turn_state, 2 layers");
    fd_model(tiny(kK, "causal"), "causal mask with state rows");
    fd_model(recursed(kK), "turn_state, 1 + [2] x 2 + 1");
    test_prefill_decode(tiny(kK), "turn_state");
    test_prefill_decode(tiny(kK, "causal"), "causal");
    test_prefill_decode(recursed(kK), "turn_state, 1 + [2] x 2 + 1");
    test_eviction(tiny(kK), "turn_state");
    test_eviction(recursed(kK), "turn_state, 1 + [2] x 2 + 1");
    test_refusals();
    std::printf("device (%s)\n", on_device() ? "device op set ON" : "CPU build or op set off");
    test_device_op();
    test_device_steps(tiny(kK), "turn_state");
    test_device_steps(recursed(kK), "turn_state, 1 + [2] x 2 + 1");
    std::printf("\n[PASS] state tokens\n");
    return 0;
}
