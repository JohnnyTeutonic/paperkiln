// Packed scoring (mtstudio score --batch-tokens; LOCAL_LLM_WORK E5.2, 28 Oct 2026):
// nn::Llama::forward_packed_hidden + head over several sequences against forward()
// on each sequence alone.
//
// Host legs (every build):
//   1. Plain, untied, the third-party options (GQA, qk_norm, NeoX partial rotary),
//      a fixed recursion with injection, and a causal state-token model (state ids
//      in the prompts): every row's logits bit for bit, all rows and a subset of
//      rows (the head on scored rows only), where every GEMM is in
//      ops::host_gemm_rows_free's regime (rows computed as they would be alone).
//   2. A batch whose rows force the host GEMM's M split (host_gemm_rows_free false):
//      float rounding (|d| <= 1e-5 max(1, |ref|), IR_SPEC 4.3's CPU tolerance),
//      argmax equal.
//   3. Refusals: MoE, Attention Residuals, a routed recursion, and turn_state over
//      ids holding a state id are not packable (and the call throws); turn_state
//      over text ids is packable and bit for bit.
//   4. Bad ranges (a gap, an overlap, a short tiling) and rows out of range throw.
// Device leg (CUDA build with MICROTORCH_DEVICE=cuda, run on a GPU only):
//   5. Inside a step window with the tiled GEMM (MICROTORCH_CUBLAS unset), where every
//      GEMM gives a row the summation order it gets alone: packed = each sequence
//      packed alone, bit for bit; against forward() written through (Phase A cuBLAS),
//      float rounding (5e-5 max(1, |ref|)).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/llama.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

namespace {

bool bits_row(const Matrix& a, size_t ra, const Matrix& b, size_t rb) {
    return a.cols() == b.cols() && std::memcmp(a.get_data() + ra * a.cols(), b.get_data() + rb * b.cols(),
                                               a.cols() * sizeof(float)) == 0;
}

double rel_row(const Matrix& a, size_t ra, const Matrix& ref, size_t rr) {
    double mx = 1.0, d = 0.0;
    for (size_t k = 0; k < ref.cols(); ++k) mx = std::max(mx, std::fabs(static_cast<double>(ref(rr, k))));
    for (size_t k = 0; k < ref.cols(); ++k)
        d = std::max(d, std::fabs(static_cast<double>(a(ra, k)) - static_cast<double>(ref(rr, k))));
    return d / mx;
}

size_t argmax_row(const Matrix& m, size_t r) {
    size_t b = 0;
    for (size_t k = 1; k < m.cols(); ++k)
        if (m(r, k) > m(r, b)) b = k;
    return b;
}

nn::LlamaConfig base(size_t vocab = 300, size_t d = 32) {
    nn::LlamaConfig c;
    c.vocab = vocab;
    c.d = d;
    c.n_layers = 2;
    c.n_heads = 4;
    c.d_ff = 64;
    c.n_ctx = 64;
    return c;
}

std::vector<std::vector<int>> sequences(const std::vector<size_t>& lens, size_t vocab, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<std::vector<int>> s;
    for (size_t n : lens) {
        std::vector<int> v(n);
        for (auto& x : v) x = static_cast<int>(g() % vocab);
        s.push_back(v);
    }
    return s;
}

struct Packed {
    std::vector<int> ids, rows;
    std::vector<std::pair<size_t, size_t>> segs;
    std::vector<std::vector<size_t>> local;  // per sequence: the local rows asked for
};

// tail = 0: every row of every sequence; otherwise its last `tail` rows (the scored rows).
Packed pack(const std::vector<std::vector<int>>& seqs, size_t tail) {
    Packed p;
    for (const auto& s : seqs) {
        p.segs.emplace_back(p.ids.size(), s.size());
        std::vector<size_t> loc;
        const size_t from = tail == 0 || tail >= s.size() ? 0 : s.size() - tail;
        for (size_t i = from; i < s.size(); ++i) {
            loc.push_back(i);
            p.rows.push_back(static_cast<int>(p.ids.size() + i));
        }
        p.local.push_back(loc);
        p.ids.insert(p.ids.end(), s.begin(), s.end());
    }
    return p;
}

// Packed logits against forward() alone: {all rows bit for bit, worst relative difference, argmax equal}.
struct Cmp {
    bool exact = true;
    double worst = 0.0;
    bool argmax = true;
};
Cmp compare(const nn::Llama& m, const std::vector<std::vector<int>>& seqs, size_t tail) {
    NoGrad ng;
    const Packed p = pack(seqs, tail);
    const Var lg = m.head(m.forward_packed_hidden(p.ids, p.segs, p.rows));
    CHECK(lg->data.rows() == p.rows.size());
    Cmp c;
    size_t r = 0;
    for (size_t s = 0; s < seqs.size(); ++s) {
        const Var alone = m.forward(seqs[s]);
        for (size_t i : p.local[s]) {
            c.exact = c.exact && bits_row(lg->data, r, alone->data, i);
            c.worst = std::max(c.worst, rel_row(lg->data, r, alone->data, i));
            c.argmax = c.argmax && argmax_row(lg->data, r) == argmax_row(alone->data, i);
            ++r;
        }
    }
    return c;
}

bool all_free(const nn::LlamaConfig& c, size_t rows) {
    const size_t dk = c.d / c.n_heads, kv = c.kv_heads() * dk;
    for (auto [n, k] : std::vector<std::pair<size_t, size_t>>{
             {c.d, c.d}, {kv, c.d}, {c.d_ff, c.d}, {c.d, c.d_ff}, {c.vocab, c.d}})
        if (!ops::host_gemm_rows_free(rows, n, k)) return false;
    return true;
}

void leg1() {
    std::printf("-- 1. packed = alone, bit for bit, where the host GEMM's rows are free --\n");
    const std::vector<size_t> lens{1, 5, 16, 23, 40, 7};  // 92 rows
    struct Case {
        const char* name;
        nn::LlamaConfig cfg;
    };
    std::vector<Case> cases;
    cases.push_back({"plain tied", base()});
    {
        auto c = base();
        c.tie_embeddings = false;
        cases.push_back({"untied", c});
    }
    {
        auto c = base();
        c.n_kv_heads = 2;
        c.qk_norm = true;
        c.rope_pairing = "neox";
        c.rope_dims = 4;
        cases.push_back({"gqa + qk_norm + neox partial", c});
    }
    {
        auto c = base();
        c.n_layers = 3;
        c.recursion.prelude = 1;
        c.recursion.block = 1;
        c.recursion.repeats = 3;
        c.recursion.coda = 1;
        c.recursion.inject = "add";
        cases.push_back({"fixed recursion, inject add", c});
    }
    for (const auto& cs : cases) {
        nn::Llama m(cs.cfg, 7);
        const auto seqs = sequences(lens, cs.cfg.vocab, 11);
        CHECK(all_free(cs.cfg, 92));
        for (size_t tail : {size_t(0), size_t(3)}) {
            const Cmp c = compare(m, seqs, tail);
            std::printf("  %-30s tail %zu: exact %d worst %.2e argmax %d\n", cs.name, tail, c.exact, c.worst, c.argmax);
            CHECK(c.exact);
        }
    }
    // A causal state-token model: state ids in the prompts take the state embedding, attention stays causal.
    {
        auto c = base();
        c.state_k = 4;
        c.state_mask = "causal";
        nn::Llama m(c, 9);
        auto seqs = sequences(lens, c.vocab, 13);
        for (auto& s : seqs)
            if (s.size() > 3) s[s.size() / 2] = static_cast<int>(c.vocab + 1);
        CHECK(m.packable(seqs[3]));
        const Cmp r = compare(m, seqs, 0);
        std::printf("  %-30s: exact %d\n", "state tokens, causal mask", r.exact);
        CHECK(r.exact);
    }
}

void leg2() {
    std::printf("-- 2. rows the host GEMM splits by M: float rounding --\n");
    const auto cfg = base();
    nn::Llama m(cfg, 7);
    const auto seqs = sequences({40, 33, 64, 50, 17, 60, 9}, cfg.vocab, 21);  // 273 rows
    CHECK(!all_free(cfg, 273));
    const Cmp c = compare(m, seqs, 0);
    std::printf("  273 rows: exact %d worst %.2e argmax %d\n", c.exact, c.worst, c.argmax);
    CHECK(c.worst <= 1e-5);
    CHECK(c.argmax);
}

template <class F>
bool throws(F f) {
    try {
        f();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void leg3() {
    std::printf("-- 3. what is not packable --\n");
    const std::vector<int> text{1, 2, 3, 4};
    const std::vector<std::pair<size_t, size_t>> one{{0, 4}};
    const std::vector<int> rows{3};
    {
        auto c = base();
        c.moe.experts = 4;
        c.moe.top_k = 2;
        nn::Llama m(c, 3);
        CHECK(!m.packable(text));
        CHECK(throws([&] { NoGrad ng; m.forward_packed_hidden(text, one, rows); }));
    }
    {
        auto c = base();
        c.attn_res.on = true;
        nn::Llama m(c, 3);
        CHECK(!m.packable(text));
    }
    {
        auto c = base();
        c.n_layers = 3;
        c.recursion.prelude = 1;
        c.recursion.block = 1;
        c.recursion.repeats = 2;
        c.recursion.coda = 1;
        c.recursion.schedule = "mor";
        nn::Llama m(c, 3);
        CHECK(!m.packable(text));
    }
    {
        auto c = base();
        c.state_k = 4;  // turn_state mask (the default)
        nn::Llama m(c, 3);
        CHECK(m.packable(text));  // no state id: plain causal attention, as forward() runs it
        const std::vector<int> st{1, 2, static_cast<int>(c.vocab), 4};
        CHECK(!m.packable(st));
        CHECK(throws([&] { NoGrad ng; m.forward_packed_hidden(st, one, rows); }));
        const Cmp r = compare(m, sequences({5, 12, 30}, c.vocab, 5), 0);
        CHECK(r.exact);
    }
}

void leg4() {
    std::printf("-- 4. bad ranges and rows --\n");
    nn::Llama m(base(), 3);
    const std::vector<int> ids{1, 2, 3, 4, 5};
    NoGrad ng;
    CHECK(throws([&] { m.forward_packed_hidden(ids, {{0, 2}, {3, 2}}, {0}); }));  // a gap
    CHECK(throws([&] { m.forward_packed_hidden(ids, {{0, 3}, {2, 3}}, {0}); }));  // an overlap
    CHECK(throws([&] { m.forward_packed_hidden(ids, {{0, 4}}, {0}); }));          // short
    CHECK(throws([&] { m.forward_packed_hidden(ids, {{0, 5}}, {5}); }));          // row out of range
    CHECK(!throws([&] { m.forward_packed_hidden(ids, {{0, 2}, {2, 3}}, {1, 4}); }));
}

void leg5_device() {
#ifdef MICROTORCH_CUDA
    const char* dev = std::getenv("MICROTORCH_DEVICE");
    if (!dev || std::string(dev) != "cuda") {
        std::printf("-- 5. device: skipped (MICROTORCH_DEVICE is not cuda) --\n");
        return;
    }
    std::printf("-- 5. device: packed in a step window (tiled GEMM) --\n");
    const char* cub = std::getenv("MICROTORCH_CUBLAS");
    const bool tiled = !(cub && std::string(cub) == "1");
    const auto cfg = base(2048, 64);
    nn::Llama m(cfg, 7);
    const auto seqs = sequences({40, 33, 64, 50, 17, 60, 9}, cfg.vocab, 21);
    auto packed = [&](const std::vector<std::vector<int>>& ss) {
        NoGrad ng;
        const Packed p = pack(ss, 0);
        device::step_begin();
        Var lg = m.head(m.forward_packed_hidden(p.ids, p.segs, p.rows));
        device::step_end();
        return lg->data;
    };
    const Matrix all = packed(seqs);
    double worst = 0.0;
    bool exact = true;
    size_t r = 0;
    for (const auto& s : seqs) {
        const Matrix alone_packed = packed({s});
        Matrix wt;
        {
            NoGrad ng;
            wt = m.forward(s)->data;  // written through: the registered unbatched path's kernels
        }
        for (size_t i = 0; i < s.size(); ++i, ++r) {
            exact = exact && bits_row(all, r, alone_packed, i);
            worst = std::max(worst, rel_row(all, r, wt, i));
        }
    }
    std::printf("  tiled %d: packed = packed alone bit for bit %d; vs written-through forward worst %.2e\n", tiled,
                exact, worst);
    if (tiled) CHECK(exact);
    CHECK(worst <= 5e-5);
#endif
}

}  // namespace

int main() {
    device::set_from_env();
    leg1();
    leg2();
    leg3();
    leg4();
    leg5_device();
    std::printf("test_score_batch: all checks passed\n");
    return 0;
}
