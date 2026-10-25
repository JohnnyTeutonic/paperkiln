// The IR digest in C++ (spec/IR_SPEC.md 3.10; microtorch/ir.hpp): SHA-256 known answers, then every vector
// of spec/conformance/digest/vectors.json (the path is argv[1]): the RFC 8785 serialisation and the sha256
// equal the reference's (tools/ir_validate.py), and the texts that have no digest are refused.
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "check.hpp"
#include "microtorch/ir.hpp"

using namespace microtorch;

int main(int argc, char** argv) {
    CHECK(ir::sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(ir::sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(ir::sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(ir::sha256_hex(std::string(1000000, 'a')) ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    std::printf("1. SHA-256: 4 FIPS 180-4 known answers\n");
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_ir_digest spec/conformance/digest/vectors.json\n");
        return 2;
    }
    std::ifstream f(argv[1]);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const auto vec = ir::json::parse(text)["vectors"];
    size_t ok = 0, refused = 0;
    for (const auto& v : vec) {
        const std::string t = v["text"].get<std::string>();
        if (v.contains("refused")) {
            bool threw = false;
            try {
                ir::canonical(ir::parse_ijson(t));
            } catch (const ir::DigestError&) {
                threw = true;
            }
            if (!threw) std::fprintf(stderr, "vector %s: has no digest, but C++ gave one\n", v["name"].get<std::string>().c_str());
            CHECK(threw);
            ++refused;
            continue;
        }
        const std::string c = ir::canonical(ir::parse_ijson(t), v.value("prose", false));
        if (c != v["canonical"].get<std::string>())
            std::fprintf(stderr, "vector %s:\n  C++  %s\n  want %s\n", v["name"].get<std::string>().c_str(),
                         c.substr(0, 200).c_str(), v["canonical"].get<std::string>().substr(0, 200).c_str());
        CHECK(c == v["canonical"].get<std::string>());
        CHECK(ir::sha256_hex(c) == v["sha256"].get<std::string>());
        if (!v.value("prose", false)) CHECK(ir::digest_text(t) == v["sha256"].get<std::string>());
        ++ok;
    }
    std::printf("2. digest vectors: %zu serialised and hashed as the reference does, %zu refused\n", ok, refused);
    std::printf("ir_digest: all checks passed\n");
    return 0;
}
