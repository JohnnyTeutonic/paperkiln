// Llama KV cache (nn::Llama::forward_cached, mtstudio chat --kv-cache):
// cached decoding gives the full recompute's logits to rounding, row for
// row, and the same greedy tokens, for
//   1. prefill then one token at a time, to the context length;
//   2. chunked prefill (several tokens per call after the first);
//   3. a truncated cache continued with different tokens (a chat turn
//      that re-sends its history, then diverges);
// over dense, MoE (top-2 renorm), legacy first-head RoPE and untied-head
// models. Serving parity (chat with and without --kv-cache, the same
// greedy replies) is checked against a trained model separately.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/llama.hpp"

using namespace microtorch;

namespace {

constexpr float kTol = 2e-4f;  // max |dlogit| over the logit range (rounding, not a model difference)

Matrix full_logits(const nn::Llama& m, const std::vector<int>& ids) { return m.forward(ids)->data; }

// Compare cached rows [from, from + n) of `ids` against the full forward.
float worst(const nn::Llama& m, const std::vector<int>& ids, const Matrix& cached, size_t from) {
    const Matrix full = full_logits(m, std::vector<int>(ids.begin(), ids.begin() + static_cast<long>(from + cached.rows())));
    float lo = 1e30f, hi = -1e30f, w = 0.0f;
    for (size_t i = 0; i < cached.rows(); ++i) {
        size_t a1 = 0, a2 = 0;
        for (size_t j = 0; j < cached.cols(); ++j) {
            const float f = full(from + i, j), c = cached(i, j);
            lo = std::min(lo, f);
            hi = std::max(hi, f);
            w = std::max(w, std::fabs(f - c));
            if (f > full(from + i, a1)) a1 = j;
            if (c > cached(i, a2)) a2 = j;
        }
        // The same greedy token, unless the full forward itself ties them to rounding.
        CHECK(a1 == a2 || std::fabs(full(from + i, a1) - full(from + i, a2)) < 1e-5f);
    }
    return w / std::max(hi - lo, 1e-6f);
}

void run(const char* name, nn::LlamaConfig cfg) {
    nn::Llama m(cfg, 7);
    m.eval();
    NoGrad ng;
    std::mt19937 rng(3);
    std::vector<int> ids(cfg.n_ctx);
    for (int& t : ids) t = static_cast<int>(rng() % cfg.vocab);

    // 1. prefill 5, then one token at a time.
    float w1 = 0.0f;
    {
        nn::LlamaKV kv;
        Matrix pre = m.forward_cached(std::vector<int>(ids.begin(), ids.begin() + 5), kv)->data;
        w1 = std::max(w1, worst(m, ids, pre, 0));
        for (size_t t = 5; t < ids.size(); ++t) {
            Matrix one = m.forward_cached({ids[t]}, kv)->data;
            CHECK(one.rows() == 1 && kv.len == t + 1);
            w1 = std::max(w1, worst(m, ids, one, t));
        }
    }
    // 2. chunked prefill: 3, then chunks of 4.
    float w2 = 0.0f;
    {
        nn::LlamaKV kv;
        size_t at = 0;
        for (size_t n : {3, 4, 4, 4}) {
            if (at + n > ids.size()) break;
            Matrix c = m.forward_cached(std::vector<int>(ids.begin() + static_cast<long>(at),
                                                         ids.begin() + static_cast<long>(at + n)), kv)->data;
            w2 = std::max(w2, worst(m, ids, c, at));
            at += n;
        }
    }
    // 3. decode 12, truncate to 7, continue with other tokens.
    float w3 = 0.0f;
    {
        nn::LlamaKV kv;
        m.forward_cached(std::vector<int>(ids.begin(), ids.begin() + 12), kv);
        kv.truncate(7);
        CHECK(kv.len == 7 && kv.k[0].size() == 7 * cfg.d);
        std::vector<int> alt(ids.begin(), ids.begin() + 7);
        for (size_t t = 7; t < ids.size(); ++t) alt.push_back(static_cast<int>((ids[t] + 1) % cfg.vocab));
        for (size_t t = 7; t < alt.size(); ++t) w3 = std::max(w3, worst(m, alt, m.forward_cached({alt[t]}, kv)->data, t));
    }
    printf("  %-12s worst |dlogit| / range: decode %.2e, chunked %.2e, truncated %.2e\n", name, w1, w2, w3);
    CHECK(w1 < kTol && w2 < kTol && w3 < kTol);
}

}  // namespace

int main() {
    device::set_from_env();
    nn::LlamaConfig cfg;
    cfg.vocab = 97;
    cfg.d = 32;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.d_ff = 64;
    cfg.n_ctx = 24;
    printf("=== KV cache vs full recompute ===\n");
    run("dense", cfg);
    nn::LlamaConfig moe = cfg;
    moe.moe.experts = 4;
    moe.moe.top_k = 2;
    run("moe top-2", moe);
    nn::LlamaConfig first = cfg;
    first.rope_all_heads = false;
    run("rope first", first);
    nn::LlamaConfig untied = cfg;
    untied.tie_embeddings = false;
    run("untied head", untied);
    printf("KV cache tests passed\n");
    return 0;
}
