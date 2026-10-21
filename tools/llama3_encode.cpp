// llama3_encode --split < records       ->  per record, its pre-tokeniser words as hex, space-separated
// llama3_encode VOCAB.gguf < records    ->  per record, its token ids, space-separated
// Records are separated by NUL bytes, so they can hold any newlines. The C++
// half of tests/test_llama3_tokenizer.py (parity with the llama3 regex and
// with HF tokenizers).
#include <iostream>
#include <iterator>
#include <string>

#include "microtorch/llama3_tokenizer.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: llama3_encode --split|VOCAB.gguf < NUL-separated records\n";
        return 2;
    }
    try {
        const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
        const bool split = std::string(argv[1]) == "--split";
        microtorch::llama3::Tokenizer tok({}, {}, {});
        if (!split) tok = microtorch::llama3::load_gguf(argv[1]);
        size_t at = 0;
        while (at <= in.size()) {
            size_t end = in.find('\0', at);
            if (end == std::string::npos) end = in.size();
            const std::string rec = in.substr(at, end - at);
            if (split) {
                const auto words = microtorch::llama3::split_words(rec);
                for (size_t i = 0; i < words.size(); ++i) {
                    if (i) std::cout << ' ';
                    for (unsigned char c : words[i]) std::cout << "0123456789abcdef"[c >> 4] << "0123456789abcdef"[c & 15];
                }
            } else {
                const auto ids = tok.encode(rec);
                for (size_t i = 0; i < ids.size(); ++i) std::cout << (i ? " " : "") << ids[i];
            }
            std::cout << "\n";
            if (end == in.size()) break;
            at = end + 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "llama3_encode: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
