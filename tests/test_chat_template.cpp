// include/microtorch/chat_template.hpp against its Python twin (tools/chat_template.py), through
// tests/chat_template_cases.json (written by tests/test_chat_template.py): the hand-written
// conversations render to the same text or are refused alike, the registry's records are the
// same JSON, the generated Jinja and Modelfile are byte-identical, resolution picks the same
// record, and the v1 rendering of (user, assistant) pairs equals the string mtstudio's
// chat_prompt_bpe built before it rendered through this header.
//
//   test_chat_template tests/chat_template_cases.json
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "check.hpp"
#include "microtorch/chat_template.hpp"

using namespace microtorch;
using nlohmann::json;

// mtstudio's chat_prompt_bpe as it was (a frozen copy): past turns, then the new message.
static std::string legacy_prompt(const std::vector<std::pair<std::string, std::string>>& turns, const std::string& user) {
    std::string p;
    for (const auto& t : turns) p += "<|user|>" + t.first + "<|assistant|>" + t.second + "<|endoftext|>";
    return p + "<|user|>" + user + "<|assistant|>";
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "tests/chat_template_cases.json";
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return 1;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const json fx = json::parse(ss.str());

    // Edge cases: same text, or refused alike.
    size_t rendered = 0, refused = 0;
    for (const auto& c : fx["cases"]) {
        const chat::Template t = c["template"].is_string() ? chat::get(c["template"].get<std::string>())
                                                            : chat::Template::from_json(c["template"]);
        const int thinking = c.contains("thinking") ? (c["thinking"].get<bool>() ? 1 : 0) : -1;
        std::string got;
        bool threw = false;
        try {
            got = chat::render(chat::messages_from_json(c["messages"]), t, c["generation_prompt"].get<bool>(), thinking);
        } catch (const chat::Error&) {
            threw = true;
        }
        if (c["expect"].is_null()) {
            if (!threw) std::fprintf(stderr, "case '%s' rendered, the Python twin refuses it\n", c["name"].get<std::string>().c_str());
            CHECK(threw);
            ++refused;
        } else {
            if (threw || got != c["expect"].get<std::string>())
                std::fprintf(stderr, "case '%s': got [%s]\n", c["name"].get<std::string>().c_str(), got.c_str());
            CHECK(!threw && got == c["expect"].get<std::string>());
            ++rendered;
        }
    }

    // The registry: the same records (compared as JSON values), and from_json(to_json) is the identity.
    size_t records = 0;
    for (const auto& t : chat::registry()) {
        CHECK(fx["records"].contains(t.id()));
        CHECK(json::parse(t.to_json().dump()) == fx["records"][t.id()]);
        CHECK(chat::Template::from_json(t.to_json()).to_json() == t.to_json());
        ++records;
    }
    CHECK(records == fx["records"].size());
    bool unknown = false;
    try {
        json bad = chat::v1().to_json();
        bad["content"] = "lower";
        chat::Template::from_json(bad);
    } catch (const chat::Error&) {
        unknown = true;
    }
    CHECK(unknown);

    // Jinja and the Modelfile: byte-identical.
    chat::Template none = chat::v1();
    none.system_mode = "none";
    CHECK(chat::jinja(chat::v1()) == fx["jinja"]["paperkiln/1"].get<std::string>());
    CHECK(chat::jinja(none) == fx["jinja"]["paperkiln/1 none"].get<std::string>());
    CHECK(chat::jinja(chat::legacy_word()) == fx["jinja"]["paperkiln/legacy-word"].get<std::string>());
    CHECK(chat::modelfile("model.gguf", chat::v1()) == fx["modelfile"]["paperkiln/1"].get<std::string>());

    // Resolution and its sha256.
    for (const auto& [k, v] : fx["sha256"].items()) CHECK(chat::detail::sha256_hex(k) == v.get<std::string>());
    size_t resolved = 0;
    for (const auto& r : fx["resolve"]) {
        const auto got = chat::resolve(r["declared"].get<std::string>(), r["tokens"].get<std::vector<std::string>>(),
                                       r["model"].get<std::string>(), r["chat_template"].get<std::string>(),
                                       r["arch"].get<std::string>());
        if (r["expect"].is_null()) {
            CHECK(!got.chat);
        } else {
            CHECK(got.chat && got.tmpl.id() == r["expect"].get<std::string>());
        }
        ++resolved;
    }

    // mtstudio's old string builder: identical on (user, assistant) pairs, empty and newline-only
    // contents included.
    const std::vector<std::string> words = {"", "\n", "hi", "naïve café", " spaced ", "x y z"};
    size_t legacy = 0;
    for (size_t n = 0; n < 5; ++n)
        for (size_t w = 0; w < words.size(); ++w) {
            std::vector<std::pair<std::string, std::string>> turns;
            for (size_t i = 0; i < n; ++i) turns.push_back({words[(w + i) % words.size()], words[(w + 2 * i + 1) % words.size()]});
            const std::string user = words[(w + n) % words.size()];
            CHECK(chat::render(chat::from_pairs(turns, &user), chat::v1(), true) == legacy_prompt(turns, user));
            ++legacy;
        }

    std::printf("chat_template: %zu cases rendered and %zu refused as in Python, %zu records, Jinja and Modelfile "
                "identical, %zu resolutions, %zu legacy prompts identical\n",
                rendered, refused, records, resolved, legacy);
    return 0;
}
