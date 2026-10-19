#include "microtorch/ir.hpp"

#include <charconv>
#include <cstdlib>

namespace microtorch {
namespace ir {

double f32(float v) {
    char buf[32];
    const auto r = std::to_chars(buf, buf + sizeof buf - 1, v);  // shortest round-trip for float
    *r.ptr = '\0';
    return std::strtod(buf, nullptr);
}

json rmsnorm(const std::string& in, const std::string& weight, float eps, const std::string& out) {
    return {{"op", "rmsnorm"}, {"in", json::array({in})}, {"weight", weight}, {"eps", f32(eps)}, {"out", out}};
}

json rope(const std::string& q, const std::string& k, const std::string& pos, size_t heads, size_t head_dim,
          float theta, bool all_heads) {
    return {{"op", "rope"},
            {"in", json::array({q, k, pos})},
            {"heads", heads},
            {"head_dim", head_dim},
            {"theta", f32(theta)},
            {"rope_heads", all_heads ? "all" : "first"},
            {"pairing", "interleaved"},
            {"_note",
             "adjacent pairs (x[2j], x[2j+1]) within each head, angle pos * theta^(-2j/head_dim); "
             "products rounded separately (no FMA)"},
            {"out", json::array({q, k})}};
}

json attention(const std::string& q, const std::string& k, const std::string& v, const std::string& state,
               size_t heads, size_t kv_heads, size_t head_dim, float scale, bool causal, size_t window,
               size_t sinks, const std::string& out) {
    return {{"op", "attention"},
            {"in", json::array({q, k, v})},
            {"state", state},
            {"heads", heads},
            {"kv_heads", kv_heads},
            {"head_dim", head_dim},
            {"causal", causal},
            {"scale", f32(scale)},
            {"window", window},
            {"sinks", sinks},
            {"softmax", "max-shifted, masked entries exactly zero"},
            {"out", out}};
}

json kv_state(const std::string& name, size_t heads, size_t kv_heads, size_t head_dim, size_t max_len,
              size_t window, size_t sinks) {
    return {{"name", name},       {"kind", "kv"},     {"heads", heads}, {"kv_heads", kv_heads},
            {"head_dim", head_dim}, {"max_len", max_len}, {"window", window}, {"sinks", sinks},
            {"dtype", "f32"}};
}

json binary(const std::string& op, const std::string& a, const std::string& b, const std::string& out) {
    return {{"op", op}, {"in", json::array({a, b})}, {"out", out}};
}

json unary(const std::string& op, const std::string& in, const std::string& out) {
    return {{"op", op}, {"in", json::array({in})}, {"out", out}};
}

}  // namespace ir
}  // namespace microtorch
