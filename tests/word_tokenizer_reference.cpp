// Hex-framed CPU reference for HF legacy-tokenizer parity tests. No model/GPU.
#include "microtorch/word_tokenizer.hpp"
#include <clocale>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

std::string unhex(const std::string& s) {
  if (s.size() % 2) throw std::runtime_error("odd hex");
  std::string out;
  for (size_t i = 0; i < s.size(); i += 2)
    out.push_back(static_cast<char>(std::stoul(s.substr(i, 2), nullptr, 16)));
  return out;
}
std::string hex(const std::string& s) {
  const char* digits = "0123456789abcdef";
  std::string out;
  for (unsigned char c : s) { out += digits[c >> 4]; out += digits[c & 15]; }
  return out;
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::setlocale(LC_ALL, "C");
  std::ifstream input(argv[1]);
  if (!input) return 2;
  std::vector<std::string> tokens;
  std::map<std::string, int> vocab;
  std::string line;
  while (std::getline(input, line)) {
    std::string token = unhex(line);
    vocab.emplace(token, static_cast<int>(tokens.size()));
    tokens.push_back(token);
  }
  while (std::getline(std::cin, line)) {
    std::vector<int> ids;
    if (line.rfind("d:", 0) == 0) {
      std::istringstream values(line.substr(2));
      std::string item;
      while (std::getline(values, item, ',')) ids.push_back(std::stoi(item));
    } else {
      ids = microtorch::wordtok::tokenize(unhex(line), vocab, 1000000);
    }
    for (size_t i = 0; i < ids.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << ids[i];
    }
    std::cout << '\t' << hex(microtorch::wordtok::detokenize(ids, tokens)) << '\n';
  }
}
