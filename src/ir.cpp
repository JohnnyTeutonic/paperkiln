#include "microtorch/ir.hpp"

#include <charconv>
#include <cstdlib>
#include <stdexcept>

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

json layernorm(const std::string& in, const std::string& weight, const std::string& bias, float eps,
               const std::string& out) {
    return {{"op", "layernorm"}, {"in", json::array({in})}, {"weight", weight},
            {"bias", bias},      {"eps", f32(eps)},         {"out", out}};
}

json gelu(const std::string& in, const std::string& approx, const std::string& out) {
    if (approx != "tanh" && approx != "none")
        throw std::runtime_error("ir::gelu: approx must be \"tanh\" or \"none\", got \"" + approx + "\"");
    return {{"op", "gelu"}, {"in", json::array({in})}, {"approx", approx}, {"out", out}};
}

json split(const std::string& in, const std::vector<size_t>& sizes, const std::vector<std::string>& outs) {
    if (sizes.size() != outs.size() || sizes.empty())
        throw std::runtime_error("ir::split: one size per output, at least one");
    return {{"op", "split"}, {"in", json::array({in})}, {"sizes", sizes}, {"out", outs}};
}

json pos_embedding(const std::string& x, const std::string& pos, const std::string& weight, const std::string& out) {
    return {{"op", "pos_embedding"}, {"in", json::array({x, pos})}, {"weight", weight}, {"out", out}};
}

json scale(const std::string& in, const std::string& weight, const std::string& bias, const std::string& out) {
    return {{"op", "scale"},
            {"in", json::array({in})},
            {"weight", weight},
            {"bias", bias.empty() ? json(nullptr) : json(bias)},
            {"out", out}};
}

json affine(const std::string& in, float scale, float shift, const std::string& out) {
    return {{"op", "affine"}, {"in", json::array({in})}, {"scale", f32(scale)}, {"shift", f32(shift)}, {"out", out}};
}

json rms_row(const std::string& in, float eps, const std::string& out) {
    return {{"op", "rms_row"}, {"in", json::array({in})}, {"eps", f32(eps)}, {"out", out}};
}

json linear_attention(const std::string& q, const std::string& k, const std::string& v, const std::string& state,
                      size_t heads, size_t head_dim, const std::string& out) {
    return {{"op", "linear_attention"},
            {"in", json::array({q, k, v})},
            {"state", state},
            {"heads", heads},
            {"head_dim", head_dim},
            {"causal", true},
            {"feature_map", "elu_plus_one"},
            {"den_floor", f32(1e-8f)},
            {"out", out}};
}

namespace {
json recurrent_state(const std::string& name, const std::string& op, const json& tensors) {
    return {{"name", name}, {"kind", "recurrent"}, {"op", op}, {"dtype", "f32"}, {"tensors", tensors}};
}
}  // namespace

json linear_attention_state(const std::string& name, size_t heads, size_t head_dim) {
    const json shape = json::array({heads, head_dim});
    return recurrent_state(name, "linear_attention",
                           json::array({{{"name", "num"}, {"shape", shape}, {"init", 0}},
                                        {{"name", "den"}, {"shape", shape}, {"init", 0}}}));
}

json depth_attend(const std::vector<std::string>& ins, const std::string& weight, float eps, const std::string& out) {
    if (ins.empty()) throw std::runtime_error("ir::depth_attend: needs a list input");
    return {{"op", "depth_attend"}, {"in", ins}, {"weight", weight}, {"eps", f32(eps)}, {"out", out}};
}

json list(const std::vector<std::string>& ins, const std::string& out) {
    return {{"op", "list"}, {"in", ins}, {"out", out}};
}

json append(const std::string& list, const std::string& x, const std::string& out) {
    return {{"op", "append"}, {"in", json::array({list, x})}, {"out", out}};
}

json expr(const std::vector<std::string>& ins, const std::string& text, const std::string& out) {
    return {{"op", "expr"}, {"in", ins}, {"expr", text}, {"out", out}};
}

json ssm_scan(const std::string& in, const std::string& state, const std::string& prefix, size_t n,
              const std::string& out) {
    return {{"op", "ssm_scan"},       {"in", json::array({in})},  {"state", state},
            {"n", n},                 {"a", prefix + "ssm_a.weight"}, {"b", prefix + "ssm_b.weight"},
            {"c", prefix + "ssm_c.weight"}, {"d", prefix + "ssm_d.weight"}, {"out", out}};
}

json ssm_state(const std::string& name, size_t n) {
    return recurrent_state(name, "ssm_scan",
                           json::array({{{"name", "s"}, {"shape", json::array({n})}, {"init", 0}}}));
}

}  // namespace ir
}  // namespace microtorch
