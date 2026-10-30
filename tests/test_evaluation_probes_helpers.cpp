#include <chrono>
#include <iostream>
#include "../tools/evaluation_probes.hpp"
#include "check.hpp"
using namespace studio_probes;
int main() {
    auto base =
        std::filesystem::temp_directory_path() /
        ("pk_probe_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    CHECK(std::filesystem::create_directory(base));
    {
        auto p = base / "hash";
        std::ofstream(p, std::ios::binary) << "";
        CHECK(file_sha256(p.string()) ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }
    {
        auto p = base / "hash";
        std::ofstream(p, std::ios::binary) << "abc";
        CHECK(file_sha256(p.string()) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
    {
        auto p = base / "hash";
        std::ofstream(p, std::ios::binary) << std::string(55, 'a');
        CHECK(file_sha256(p.string()) ==
              "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
    }
    {
        auto p = base / "hash";
        std::ofstream(p, std::ios::binary) << std::string(56, 'a');
        CHECK(file_sha256(p.string()) ==
              "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
    }
    {
        auto p = base / "hash";
        std::ofstream(p, std::ios::binary) << std::string(63, 'a');
        CHECK(file_sha256(p.string()) ==
              "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34");
    }
    {
        auto p = base / "hash";
        std::ofstream(p, std::ios::binary) << std::string(64, 'a');
        CHECK(file_sha256(p.string()) ==
              "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
    }
    {
        auto p = base / "hash";
        std::ofstream(p, std::ios::binary) << std::string(65, 'a');
        CHECK(file_sha256(p.string()) ==
              "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0");
    }
    {
        auto p = base / "hash";
        std::ofstream(p, std::ios::binary) << std::string(1000000, 'a');
        CHECK(file_sha256(p.string()) ==
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }
    CHECK(!valid_utf8(std::string("\xc3", 1)));
    CHECK(valid_utf8(std::string("\xc3\xa9", 2)));
    CHECK(hex_bytes(std::string("\xc3", 1)) == "c3");
    json c = {{"max_new_tokens", 2},
              {"cases", json::array({{{"id", "a"},
                                      {"kind", "needle"},
                                      {"prompt", "the key amber"},
                                      {"expected", "amber"},
                                      {"contract", "contains"}}})}};
    validate(c);
    CHECK(matches("amber", "exact", "amber"));
    CHECK(!matches("", "exact", "amber"));
    CHECK(matches("", "not_contains", "amber"));
    {
        ArtifactGuard guard((base / "out.json").string());
        bool refused = false;
        try {
            ArtifactGuard second((base / "out.json").string());
        } catch (...) {
            refused = true;
        }
        CHECK(refused);
        publish((base / "out.json").string(), c);
    }
    {
        bool refused = false;
        try {
            ArtifactGuard again((base / "out.json").string());
        } catch (...) {
            refused = true;
        }
        CHECK(refused);
    }
    {
        auto path = (base / "race.json").string();
        ArtifactGuard guard(path);
        std::ofstream(path) << "historical";
        bool refused = false;
        try {
            publish(path, c);
        } catch (...) {
            refused = true;
        }
        CHECK(refused);
        std::ifstream in(path);
        std::string text;
        in >> text;
        CHECK(text == "historical");
    }
    std::filesystem::remove_all(base);
    std::cout << "PASS8SHA256knownvectors, UTF8split-prefix/final-invalid, literalcontracts, "
                 "immutableartifact/lock/TOCTOU gates\n";
}
