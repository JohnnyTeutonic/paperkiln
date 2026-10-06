// Word tokenizer pins (include/microtorch/word_tokenizer.hpp): the literal
// "<|endoftext|>" is ONE token when the vocabulary has it, and every
// vocabulary without it tokenizes exactly as before (the pre-registered
// sweeps' corpora and vocabs must not move).
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "check.hpp"
#include "microtorch/word_tokenizer.hpp"

using namespace microtorch;

namespace {
std::map<std::string, int> make_vocab(const std::vector<std::string>& toks) {
    std::map<std::string, int> v;
    for (size_t i = 0; i < toks.size(); ++i) v.emplace(toks[i], static_cast<int>(i));
    return v;
}
}  // namespace

int main() {
    // Legacy vocabulary: no end-of-text entry -> five pieces, as always.
    const std::vector<std::string> legacy = {"<unk>", "the", "end", ".", "<",
                                             "|",     ">",   "endoftext", "user", ":"};
    const auto lv = make_vocab(legacy);
    CHECK(wordtok::eos_id(lv) == -1);
    {
        const auto ids = wordtok::tokenize("The end.<|endoftext|>the", lv, 1000);
        const std::vector<int> want = {1, 2, 3, 4, 5, 7, 5, 6, 1};
        CHECK(ids == want);
    }

    // With the entry: one token, flanking text unaffected, case-sensitive
    // literal (a lower-cased or partial marker is ordinary text).
    std::vector<std::string> with_eos = legacy;
    with_eos.push_back("<|endoftext|>");  // id 10
    const auto ev = make_vocab(with_eos);
    CHECK(wordtok::eos_id(ev) == 10);
    {
        const auto ids = wordtok::tokenize("The end.<|endoftext|>the", ev, 1000);
        const std::vector<int> want = {1, 2, 3, 10, 1};
        CHECK(ids == want);
    }
    {   // word run right before the marker is flushed first
        const auto ids = wordtok::tokenize("end<|endoftext|><|endoftext|>", ev, 1000);
        const std::vector<int> want = {2, 10, 10};
        CHECK(ids == want);
    }
    {   // truncated marker at end of text: plain punctuation
        const auto ids = wordtok::tokenize("end <|endof", ev, 1000);
        const std::vector<int> want = {2, 4, 5, 0};
        CHECK(ids == want);
    }
    {   // the cap still bounds the output
        const auto ids = wordtok::tokenize("<|endoftext|> the end", ev, 2);
        CHECK(ids.size() == 2 && ids[0] == 10 && ids[1] == 1);
    }

    // Stop-sequence helper and the reader-facing detokenizer.
    const auto user_turn = wordtok::tokenize("user:", ev, 16);
    CHECK((user_turn == std::vector<int>{8, 9}));
    CHECK(wordtok::ends_with({1, 2, 8, 9}, user_turn));
    CHECK(!wordtok::ends_with({1, 8}, user_turn));
    CHECK(!wordtok::ends_with({1, 2}, {}));
    CHECK(wordtok::detokenize({1, 2, 3, 1}, with_eos) == "the end. the");
    CHECK(wordtok::detokenize({}, with_eos).empty());

    std::printf("word_tokenizer: all checks passed\n");
    return 0;
}
