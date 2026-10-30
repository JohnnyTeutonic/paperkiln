#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace studio_probes {
using nlohmann::json;
// Streamed form of the existing src/ir.cpp FIPS180-4 compression; fixed64-byte workspace.
inline std::string file_sha256(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot hash " + path);
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    auto compress = [&](const unsigned char* m) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (static_cast<uint32_t>(static_cast<unsigned char>(m[4 * i])) << 24) |
                   (static_cast<uint32_t>(static_cast<unsigned char>(m[4 * i + 1])) << 16) |
                   (static_cast<uint32_t>(static_cast<unsigned char>(m[4 * i + 2])) << 8) |
                   static_cast<uint32_t>(static_cast<unsigned char>(m[4 * i + 3]));
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 =
                hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 =
                (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
    };
    std::array<unsigned char, 128> block{};
    uint64_t total = 0;
    size_t got = 0;
    while (true) {
        input.read(reinterpret_cast<char*>(block.data()), 64);
        got = static_cast<size_t>(input.gcount());
        total += got;
        if (got < 64) break;
        compress(block.data());
    }
    if (!input.eof()) throw std::runtime_error("hash read failed");
    std::fill(block.begin() + got, block.end(), 0);
    block[got] = 0x80;
    size_t padded = got < 56 ? 64 : 128;
    uint64_t bits = total * 8;
    for (int i = 0; i < 8; ++i) block[padded - 1 - i] = static_cast<unsigned char>(bits >> (8 * i));
    compress(block.data());
    if (padded == 128) compress(block.data() + 64);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint32_t v : h) out << std::setw(8) << v;
    return out.str();
}
inline void validate(const json& config) {
    if (!config.is_object()) throw std::runtime_error("evaluation_probes must be an object");
    for (auto it = config.begin(); it != config.end(); ++it)
        if (it.key() != "cases" && it.key() != "max_new_tokens")
            throw std::runtime_error("unknown evaluation_probes key");
    if (!config.contains("max_new_tokens") || !config["max_new_tokens"].is_number_integer())
        throw std::runtime_error("probe max_new_tokens must be integer");
    double bound = config["max_new_tokens"].get<double>();
    if (bound < 1 || bound > 128) throw std::runtime_error("probe token budget outside1..128");
    int n = config["max_new_tokens"].get<int>();
    if (n < 1 || n > 128) throw std::runtime_error("probe max_new_tokens outside1..128");
    if (!config.contains("cases") || !config["cases"].is_array() || config["cases"].empty() ||
        config["cases"].size() > 32 || config["cases"].size() * n > 1024)
        throw std::runtime_error("probe case/token budget invalid");
    std::set<std::string> ids;
    for (const auto& c : config["cases"]) {
        if (!c.is_object() || c.size() != 5)
            throw std::runtime_error("probe case needs exactly5 fields");
        for (const char* k : {"id", "kind", "prompt", "contract", "expected"})
            if (!c.contains(k) || !c[k].is_string())
                throw std::runtime_error("probe field must be string");
        auto id = c["id"].get<std::string>();
        if (id.empty() || id.size() > 64 || !ids.insert(id).second)
            throw std::runtime_error("probe id invalid/duplicate");
        for (unsigned char v : id)
            if (!((v >= 'a' && v <= 'z') || (v >= 'A' && v <= 'Z') || (v >= '0' && v <= '9') ||
                  v == '_' || v == '-'))
                throw std::runtime_error("unsafe probe id");
        if (c["kind"] != "needle" && c["kind"] != "behaviour")
            throw std::runtime_error("unknown probe kind");
        if (c["contract"] != "exact" && c["contract"] != "contains" && c["contract"] != "prefix" &&
            c["contract"] != "not_contains")
            throw std::runtime_error("unknown probe contract");
        if (c["kind"] == "needle" && (c["contract"] == "not_contains" ||
                                      c["prompt"].get<std::string>().find(
                                          c["expected"].get<std::string>()) == std::string::npos))
            throw std::runtime_error(
                "needle expected literal must occur in prompt; not_contains is behaviour only");
        if (c["prompt"].get<std::string>().empty() ||
            c["prompt"].get<std::string>().size() > 8192 ||
            c["expected"].get<std::string>().empty() ||
            c["expected"].get<std::string>().size() > 256)
            throw std::runtime_error("probe text byte bound exceeded");
    }
}
inline void validate_spec(const json& specification) {
    if (!specification.contains("evaluation_probes")) return;
    validate(specification.at("evaluation_probes"));
    const auto formats = specification.value("export", json::object())
                             .value("formats", std::vector<std::string>{"safetensors"});
    if (std::find(formats.begin(), formats.end(), "safetensors") == formats.end())
        throw std::runtime_error("evaluation_probes requires declared final safetensors export");
}
inline json error_report(const json& config, const std::exception& error);
struct ArtifactGuard {
    std::string output, lock;
    bool owned = false;
    explicit ArtifactGuard(std::string path) : output(std::move(path)), lock(output + ".lock") {
        auto parent = std::filesystem::path(output).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        if (!std::filesystem::create_directory(lock))
            throw std::runtime_error("probe artifact already locked");
        owned = true;
        try {
            for (const auto& p : {output, output + ".tmp", output + ".cases.jsonl"})
                if (std::filesystem::exists(p))
                    throw std::runtime_error("probe artifact exists; refusing overwrite");
        } catch (...) {
            std::filesystem::remove(lock);
            owned = false;
            throw;
        }
    }
    ~ArtifactGuard() {
        if (owned) {
            std::error_code e;
            std::filesystem::remove(lock, e);
        }
    }
};
inline void publish(const std::string& path, const json& value) {
    auto temp = path + ".tmp";
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write probe receipt");
        out << value.dump(2) << "\n";
        out.flush();
        if (!out) throw std::runtime_error("probe receipt write failed");
    }
    // Atomic no-replace publication, including a foreign writer racing the initial preflight.
    std::filesystem::create_hard_link(temp, path);
    std::filesystem::remove(temp);
}
inline bool valid_utf8(const std::string& text) {
    try {
        (void)json(text).dump();
        return true;
    } catch (...) {
        return false;
    }
}
inline std::string safe_error(const std::exception& error) {
    std::string result;
    for (unsigned char c : std::string(error.what())) {
        if (result.size() == 512) break;
        result += c < 128 ? char(c) : '?';
    }
    return result;
}
inline json error_report(const json& config, const std::exception& error) {
    return {{"schema_version", 1},
            {"status", "ERROR"},
            {"configuration", config},
            {"checkpoint", nullptr},
            {"cases", json::array()},
            {"summary", {{"total", 0}, {"passed", 0}, {"failed", 0}, {"errors", 0}}},
            {"global_errors", 1},
            {"error", safe_error(error)}};
}
inline std::string hex_bytes(const std::string& data) {
    static const char h[] = "0123456789abcdef";
    std::string out;
    for (unsigned char c : data) {
        out += h[c >> 4];
        out += h[c & 15];
    }
    return out;
}
inline bool matches(const std::string& text, const std::string& rule, const std::string& expected) {
    return rule == "exact"      ? text == expected
           : rule == "prefix"   ? text.rfind(expected, 0) == 0
           : rule == "contains" ? text.find(expected) != std::string::npos
                                : text.find(expected) == std::string::npos;
}
}  // namespace studio_probes
