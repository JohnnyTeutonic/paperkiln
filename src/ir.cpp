#include "microtorch/ir.hpp"

#include "microtorch/recurrence.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
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

// ---- the digest (IR_SPEC.md section 3.10) ----

namespace {

// ECMAScript Number::toString of a finite double (RFC 8785 section 3.2.2.3).
std::string es_number(double x) {
    if (!std::isfinite(x)) throw DigestError("digest: a number is not finite");
    if (x == 0) {
        if (std::signbit(x)) throw DigestError("digest: a negative zero (RFC 8785 writes it as 0)");
        return "0";
    }
    if (x < 0) return "-" + es_number(-x);
    char buf[64];
    // Shortest round-trip digits, the closest of them (as ECMAScript requires).
    const auto r = std::to_chars(buf, buf + sizeof buf, x, std::chars_format::scientific);
    const std::string s(buf, r.ptr);
    const size_t epos = s.find('e');
    std::string digits;
    for (size_t i = 0; i < epos; ++i)
        if (s[i] != '.') digits += s[i];
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int k = static_cast<int>(digits.size());
    const int n = std::stoi(s.substr(epos + 1)) + 1;  // x = 0.digits * 10^n
    if (k <= n && n <= 21) return digits + std::string(static_cast<size_t>(n - k), '0');
    if (0 < n && n <= 21) return digits.substr(0, static_cast<size_t>(n)) + "." + digits.substr(static_cast<size_t>(n));
    if (-6 < n && n <= 0) return "0." + std::string(static_cast<size_t>(-n), '0') + digits;
    const int e = n - 1;
    return (k == 1 ? digits : digits.substr(0, 1) + "." + digits.substr(1)) + "e" + (e >= 0 ? "+" : "-") +
           std::to_string(e >= 0 ? e : -e);
}

// The UTF-16 code units of a UTF-8 string (the RFC 8785 key order); throws on malformed UTF-8.
std::u16string utf16(const std::string& s) {
    std::u16string out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0;
        size_t len = 0;
        if (c < 0x80) cp = c, len = 1;
        else if ((c & 0xE0) == 0xC0) cp = c & 0x1F, len = 2;
        else if ((c & 0xF0) == 0xE0) cp = c & 0x0F, len = 3;
        else if ((c & 0xF8) == 0xF0) cp = c & 0x07, len = 4;
        else throw DigestError("digest: malformed UTF-8");
        if (i + len > s.size()) throw DigestError("digest: malformed UTF-8");
        for (size_t j = 1; j < len; ++j) {
            const unsigned char d = static_cast<unsigned char>(s[i + j]);
            if ((d & 0xC0) != 0x80) throw DigestError("digest: malformed UTF-8");
            cp = (cp << 6) | (d & 0x3F);
        }
        static const uint32_t kMin[] = {0, 0, 0x80, 0x800, 0x10000};
        if (cp < kMin[len] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
            throw DigestError("digest: malformed UTF-8");
        if (cp >= 0x10000) {
            cp -= 0x10000;
            out.push_back(static_cast<char16_t>(0xD800 + (cp >> 10)));
            out.push_back(static_cast<char16_t>(0xDC00 + (cp & 0x3FF)));
        } else {
            out.push_back(static_cast<char16_t>(cp));
        }
        i += len;
    }
    return out;
}

// JSON.stringify of a string (RFC 8785 section 3.2.2.2).
std::string es_string(const std::string& s) {
    utf16(s);  // well-formed or throws
    std::string out = "\"";
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\t': out += "\\t"; break;
            case '\n': out += "\\n"; break;
            case '\f': out += "\\f"; break;
            case '\r': out += "\\r"; break;
            default:
                if (c < 0x20) {
                    static const char* hex = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 15];
                } else {
                    out += ch;
                }
        }
    }
    return out + "\"";
}

void canon(const json& v, std::string& out, bool keep_prose) {
    if (v.is_null()) {
        out += "null";
    } else if (v.is_boolean()) {
        out += v.get<bool>() ? "true" : "false";
    } else if (v.is_number_unsigned() || v.is_number_integer()) {
        constexpr double kMax = 9007199254740992.0;  // 2^53
        const double d = v.is_number_unsigned() ? static_cast<double>(v.get<uint64_t>())
                                                : static_cast<double>(v.get<int64_t>());
        const bool in_range = v.is_number_unsigned() ? v.get<uint64_t>() <= 9007199254740992ull
                                                     : (v.get<int64_t>() <= 9007199254740992ll &&
                                                        v.get<int64_t>() >= -9007199254740992ll);
        if (!in_range || std::fabs(d) > kMax) throw DigestError("digest: an integer beyond 2^53");
        out += es_number(d);
    } else if (v.is_number_float()) {
        out += es_number(v.get<double>());
    } else if (v.is_string()) {
        out += es_string(v.get<std::string>());
    } else if (v.is_array()) {
        out += '[';
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) out += ',';
            canon(v[i], out, keep_prose);
        }
        out += ']';
    } else if (v.is_object()) {
        std::vector<std::pair<std::u16string, std::string>> keys;
        for (const auto& [k, x] : v.items())
            if (keep_prose || k.empty() || k[0] != '_') keys.emplace_back(utf16(k), k);
        std::sort(keys.begin(), keys.end());
        out += '{';
        for (size_t i = 0; i < keys.size(); ++i) {
            if (i) out += ',';
            out += es_string(keys[i].second);
            out += ':';
            canon(v.at(keys[i].second), out, keep_prose);
        }
        out += '}';
    } else {
        throw DigestError("digest: not a JSON value");
    }
}

}  // namespace

std::string canonical(const json& doc, bool keep_prose) {
    std::string out;
    if (!keep_prose) {
        // IR_SPEC 3.10 step 1 runs on the whole document, its "_" members included: a value there that
        // has no digest (non-finite, beyond 2^53, malformed UTF-8, a floating -0) takes the digest away.
        canon(doc, out, true);
        out.clear();
    }
    canon(doc, out, keep_prose);
    return out;
}

// FIPS 180-4 SHA-256.
std::string sha256_hex(const std::string& bytes) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::string m = bytes;
    const uint64_t bits = static_cast<uint64_t>(bytes.size()) * 8;
    m += static_cast<char>(0x80);
    while (m.size() % 64 != 56) m += '\0';
    for (int i = 7; i >= 0; --i) m += static_cast<char>((bits >> (8 * i)) & 0xff);
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (static_cast<uint32_t>(static_cast<unsigned char>(m[off + 4 * i])) << 24) |
                   (static_cast<uint32_t>(static_cast<unsigned char>(m[off + 4 * i + 1])) << 16) |
                   (static_cast<uint32_t>(static_cast<unsigned char>(m[off + 4 * i + 2])) << 8) |
                   static_cast<uint32_t>(static_cast<unsigned char>(m[off + 4 * i + 3]));
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
    }
    static const char* hex = "0123456789abcdef";
    std::string out;
    for (uint32_t x : h)
        for (int i = 7; i >= 0; --i) out += hex[(x >> (4 * i)) & 15];
    return out;
}

std::string digest(const json& doc) { return sha256_hex(canonical(doc)); }

json parse_ijson(const std::string& text) {
    std::vector<std::vector<std::string>> keys;  // the member names seen, per open object
    bool dup = false;
    json::parser_callback_t cb = [&](int, json::parse_event_t ev, json& parsed) {
        if (ev == json::parse_event_t::object_start) {
            keys.emplace_back();
        } else if (ev == json::parse_event_t::object_end) {
            if (!keys.empty()) keys.pop_back();
        } else if (ev == json::parse_event_t::key && !keys.empty()) {
            const std::string k = parsed.get<std::string>();
            auto& seen = keys.back();
            if (std::find(seen.begin(), seen.end(), k) != seen.end()) dup = true;
            seen.push_back(k);
        }
        return true;
    };
    json v;
    try {
        v = json::parse(text, cb);
    } catch (const json::exception& e) {
        throw DigestError(std::string("digest: not an I-JSON text (") + e.what() + ")");
    }
    if (dup) throw DigestError("digest: a duplicate member name");
    return v;
}

std::string digest_text(const std::string& text) { return digest(parse_ijson(text)); }

json parse_document(const std::string& text) {
    std::vector<std::vector<std::string>> keys;  // the member names seen, per open object
    std::string dup;
    json::parser_callback_t cb = [&](int, json::parse_event_t ev, json& parsed) {
        if (ev == json::parse_event_t::object_start) {
            keys.emplace_back();
        } else if (ev == json::parse_event_t::object_end) {
            if (!keys.empty()) keys.pop_back();
        } else if (ev == json::parse_event_t::key && !keys.empty()) {
            const std::string k = parsed.get<std::string>();
            auto& seen = keys.back();
            if (dup.empty() && std::find(seen.begin(), seen.end(), k) != seen.end()) dup = k;
            seen.push_back(k);
        }
        return true;
    };
    json v;
    try {
        v = json::parse(text, cb);
    } catch (const json::exception& e) {
        throw DocumentError(std::string("IR_SPEC 3.2 (rule 1): not a JSON text (") + e.what() + ")");
    }
    if (!dup.empty())
        throw DocumentError("IR_SPEC 3.2 (rule 35): the member name \"" + dup +
                            "\" occurs twice in one object (refused at any depth, '_' members included)");
    return v;
}

namespace {

std::vector<std::string> node_outs(const json& n) {
    std::vector<std::string> out;
    if (!n.contains("out")) return out;
    const json& o = n["out"];
    if (o.is_string()) out.push_back(o.get<std::string>());
    if (o.is_array())
        for (const auto& x : o)
            if (x.is_string()) out.push_back(x.get<std::string>());
    return out;
}

const json* section_nodes(const json& doc, const char* sec) {
    if (!doc.contains(sec) || !doc[sec].is_object() || !doc[sec].contains("nodes") || !doc[sec]["nodes"].is_array())
        return nullptr;
    return &doc[sec]["nodes"];
}

}  // namespace

std::vector<std::string> read_errors(const json& doc) {
    std::vector<std::string> errs;
    const json* embed = section_nodes(doc, "embed");
    const json* block = section_nodes(doc, "block");
    const json* head = section_nodes(doc, "head");
    if (!embed || !block || !head) return errs;  // rule 1
    const bool routed = !block->empty() && (*block)[0].is_object() && (*block)[0].value("op", "") == "depth_route";
    using Set = std::vector<std::string>;
    auto has = [](const Set& s, const std::string& x) { return std::find(s.begin(), s.end(), x) != s.end(); };
    auto walk = [&](const char* sec, const json& nodes, size_t from, size_t to, Set& have, int64_t application = -1) {
        for (size_t j = from; j < to && j < nodes.size(); ++j) {
            const json& n = nodes[j];
            if (!n.is_object()) continue;
            if (application >= 0 && n.contains("apply")) {
                const json* active = &n["apply"];
                if (active->is_object() && active->contains("per_layer") && (*active)["per_layer"].is_array() &&
                    static_cast<size_t>(application) < (*active)["per_layer"].size())
                    active = &(*active)["per_layer"][static_cast<size_t>(application)];
                // Schema/refusal rules check malformed apply values separately. As the pinned Python
                // reader does, only a literal false removes this application's node and its writes.
                if (active->is_boolean() && !active->get<bool>()) continue;
            }
            if (n.contains("in") && n["in"].is_array())
                for (size_t k = 0; k < n["in"].size(); ++k) {
                    if (!n["in"][k].is_string()) continue;
                    const std::string x = n["in"][k].get<std::string>();
                    if (!has(have, x))
                        errs.push_back(std::string("IR_SPEC ") + (routed && std::string(sec) != "embed" ? "7.4" : "3.5") +
                                       " (rule 36): " + sec + "[" + std::to_string(j) + "] " + n.value("op", "?") +
                                       " reads '" + x + "' (its input " + std::to_string(k) +
                                       "), which no earlier node writes" +
                                       (application >= 0 ? " (application " + std::to_string(application) + ")" : ""));
                }
            for (const auto& o : node_outs(n))
                if (!has(have, o)) have.push_back(o);
        }
    };
    Set have = {"tokens", "pos"};
    walk("embed", *embed, 0, embed->size(), have);
    if (!has(have, "x")) errs.push_back("IR_SPEC 3.5 (rule 36): the embed section writes no x");
    int64_t applications = 0;
    if (doc["block"].contains("repeat") && doc["block"]["repeat"].is_number_integer())
        applications = doc["block"]["repeat"].get<int64_t>();
    bool conditional = false;
    for (const auto& n : *block) conditional |= n.is_object() && n.contains("apply");
    if (!routed && applications > 0 && conditional) {
        // Revision 1.6: inactive nodes do not define values; active writes remain available in
        // later applications. Keep original node indices rather than filtering a temporary list.
        for (int64_t i = 0; i < applications; ++i)
            walk("block", *block, 0, block->size(), have, i);
        walk("head", *head, 0, head->size(), have);
    } else if (routed) {
        Set embed_have = have;
        walk("block", *block, 0, 1, have);
        Set step = {"x", "pos"};
        walk("block", *block, 1, block->size(), step);
        Set h = embed_have;
        if (!has(h, "x")) h.push_back("x");
        walk("head", *head, 0, head->size(), h);
    } else {
        walk("block", *block, 0, block->size(), have);
        walk("head", *head, 0, head->size(), have);
    }
    bool logits = false;
    for (const auto& n : *head)
        if (n.is_object())
            for (const auto& o : node_outs(n)) logits |= o == "logits";
    if (!logits) errs.push_back("IR_SPEC 3.4 (rule 36): the head writes no logits");
    return errs;
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

json recurrence(const std::vector<std::string>& ins, const std::string& state, const json& body,
                const std::string& out) {
    json n = {{"op", "recurrence"}, {"in", ins}, {"state", state}};
    for (const char* k : {"index", "inputs", "weights", "init", "let", "update", "readout"})
        if (body.contains(k)) n[k] = body[k];
    n["out"] = out;
    return n;
}

json recurrence_state(const std::string& name, const json& node) {
    const Recurrence r = Recurrence::from_json(node);
    json tensors = json::array();
    for (size_t k = 0; k < r.state().size(); ++k)
        tensors.push_back({{"name", r.state()[k].name},
                           {"shape", r.state()[k].shape},
                           {"init", r.initialised(k) ? json("body") : json(0)}});
    return {{"name", name}, {"kind", "recurrent"}, {"op", "recurrence"}, {"dtype", "f32"}, {"tensors", tensors}};
}

json linear_attention_body(size_t heads, size_t head_dim) {
    auto phi = [](const std::string& x) {
        return "where(" + x + ", " + x + " + 1, (exp(min(0, " + x + ")) - 1) + 1)";
    };
    return {{"index", {{"h", heads}, {"e", head_dim}}},
            {"inputs", json::array({"q[h,e]", "k[h,e]", "v[h,e]"})},
            {"let", json::array({"pk[h,e] = " + phi("k[h,e]")})},
            {"update", json::array({"num[h,e] = num[h,e] + pk[h,e] * v[h,e]", "den[h,e] = den[h,e] + pk[h,e]"})},
            {"readout", "y[h,e] = (" + phi("q[h,e]") + " * num[h,e]) / max(den[h,e], 1e-08)"}};
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
