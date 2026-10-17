// bpe_encode VOCAB.gguf < text  ->  one line of space-separated token ids per
// input line (the trailing newline of each line is encoded too, so the ids
// cover the input exactly). The C++ half of the tools/bpe.py parity test.
#include <iostream>
#include <string>

#include "microtorch/bpe_tokenizer.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: bpe_encode VOCAB.gguf < text\n";
        return 2;
    }
    try {
        const auto tok = microtorch::bpe::load_gguf(argv[1]);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!std::cin.eof()) line += '\n';
            const auto ids = tok.encode(line);
            for (size_t i = 0; i < ids.size(); ++i) std::cout << (i ? " " : "") << ids[i];
            std::cout << "\n";
            if (tok.decode(ids) != line) {
                std::cerr << "ROUNDTRIP FAILED\n";
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "bpe_encode: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
