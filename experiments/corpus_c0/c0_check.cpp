// Cross-check: count <unk> with the trainer's own tokenizer header.
#include "microtorch/word_tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
int main(int, char** argv) {
    std::ifstream vf(argv[1]); std::string line; std::vector<std::string> toks;
    while (std::getline(vf, line)) toks.push_back(line);
    std::map<std::string, int> vocab;
    for (size_t i = 0; i < toks.size(); ++i) vocab.emplace(toks[i], (int)i);
    std::ifstream cf(argv[2]);
    std::string text((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
    auto ids = microtorch::wordtok::tokenize(text, vocab, 400000);
    size_t unk = 0, vunk = 0; const size_t vs = ids.size() - ids.size() / 20;
    for (size_t i = 0; i < ids.size(); ++i) if (ids[i] == 0) { ++unk; if (i >= vs) ++vunk; }
    std::cout << "tokens " << ids.size() << " vocab " << vocab.size() << " unk " << unk << " val_unk " << vunk << "\n";
}
