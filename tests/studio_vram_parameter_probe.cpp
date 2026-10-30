#include <iostream>
#include "microtorch/llama.hpp"
#include "microtorch/nn.hpp"
int main() {
    microtorch::nn::LlamaConfig l;
    l.vocab = 64;
    l.d = 16;
    l.n_layers = 2;
    l.n_heads = 2;
    l.n_ctx = 8;
    l.d_ff = 48;
    microtorch::nn::Llama a(l, 1);
    std::cout << "llama " << a.parameter_count() << "\n";
    l.d_ff = 64;
    microtorch::nn::Llama b(l, 1);
    std::cout << "llama_ff64 " << b.parameter_count() << "\n";
    microtorch::nn::GPT2Config g;
    g.vocab = 64;
    g.d = 16;
    g.n_layers = 2;
    g.n_heads = 2;
    g.n_ctx = 8;
    microtorch::nn::GPT2 c(g, 1);
    std::cout << "gpt2 " << c.parameter_count() << "\n";
}
