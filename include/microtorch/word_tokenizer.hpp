#pragma once
// The word-level tokenizer mtstudio trains, samples and chats with (the
// srd_parity path), header-only so tests can pin it. A token is a run of
// letters, digits and apostrophes (lower-cased), or one non-space
// character. tools/get_tinystories_data.py tokenize_like_mtstudio() is the
// Python mirror; change both together.
//
// End of text: when the vocabulary contains the literal "<|endoftext|>"
// (TinyStories' story separator; chat7b.gguf carries it at id 25), that
// string is ONE token rather than the five punctuation/word pieces
// "< | endoftext | >". A vocabulary without it tokenizes exactly as
// before, so corpora and vocabs from earlier runs are unaffected.
#include <cctype>
#include <map>
#include <string>
#include <vector>

namespace microtorch {
namespace wordtok {

inline const std::string& eos_literal() {
    static const std::string s = "<|endoftext|>";
    return s;
}

// Id of the end-of-text token, or -1 when the vocabulary has none.
inline int eos_id(const std::map<std::string, int>& vocab) {
    auto it = vocab.find(eos_literal());
    return it == vocab.end() ? -1 : it->second;
}

inline std::vector<int> tokenize(const std::string& text, const std::map<std::string, int>& vocab,
                                 size_t max_tokens) {
    std::vector<int> ids;
    std::string cur;
    const int eos = eos_id(vocab);
    const std::string& lit = eos_literal();
    auto flush = [&]() {
        if (cur.empty()) return;
        auto it = vocab.find(cur);
        ids.push_back(it == vocab.end() ? 0 : it->second);
        cur.clear();
    };
    for (size_t i = 0; i < text.size(); ++i) {
        if (ids.size() >= max_tokens) break;
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == '<' && eos >= 0 && text.compare(i, lit.size(), lit) == 0) {
            flush();
            ids.push_back(eos);
            i += lit.size() - 1;
        } else if (std::isalpha(c) || c == '\'' || std::isdigit(c)) {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else {
            flush();
            if (!std::isspace(c)) {
                std::string pch(1, static_cast<char>(c));
                auto it = vocab.find(pch);
                ids.push_back(it == vocab.end() ? 0 : it->second);
            }
        }
    }
    flush();
    return ids;
}

// Readable text from word-level ids: words space-separated, closing
// punctuation attached to the word before it ("hello , world ." ->
// "hello, world."). For people reading a reply, not for round trips.
inline std::string detokenize(const std::vector<int>& ids, const std::vector<std::string>& tokens) {
    std::string out;
    for (int id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= tokens.size()) continue;
        const std::string& t = tokens[id];
        const bool attach = t.size() == 1 && std::string(".,!?;:)").find(t[0]) != std::string::npos;
        if (!out.empty() && !attach && out.back() != '(') out += ' ';
        out += t;
    }
    return out;
}

// True when `ids` ends with the non-empty sequence `stop`.
inline bool ends_with(const std::vector<int>& ids, const std::vector<int>& stop) {
    if (stop.empty() || ids.size() < stop.size()) return false;
    for (size_t k = 0; k < stop.size(); ++k)
        if (ids[ids.size() - stop.size() + k] != stop[k]) return false;
    return true;
}

}  // namespace wordtok
}  // namespace microtorch
