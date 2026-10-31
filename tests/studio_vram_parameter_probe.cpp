#include <iostream>
#include "microtorch/llama.hpp"
#include "parity_model.hpp"

// Mirror only the CLI constructor dispatch, then enumerate actual parameter storage.
// The JavaScript test compares these independent inventories with its estimate.
int main() {
    using namespace microtorch;
    nlohmann::json rows = nlohmann::json::array();
    for (const auto dims : {std::array<size_t, 5>{7, 8, 4, 2, 2},
                            std::array<size_t, 5>{64, 16, 8, 2, 1},
                            std::array<size_t, 5>{64, 16, 8, 2, 2},
                            std::array<size_t, 5>{64, 16, 8, 2, 3},
                            std::array<size_t, 5>{65, 16, 9, 4, 4},
                            std::array<size_t, 5>{19, 12, 7, 3, 5}}) {
        const auto [vocab, d, T, heads, layers] = dims;
        size_t count = 0;
        if (layers == 2) {
            parity::ParityLM model(parity::AttnKind::EXACT, vocab, d, heads, T, 1);
            count = model.parameter_count();
        } else {
            parity::FlexConfig cfg;
            cfg.vocab = vocab; cfg.d = d; cfg.n_ctx = T; cfg.n_heads = heads;
            cfg.n_layers = layers; cfg.d_ff = 4 * d;
            parity::FlexLM model(cfg, 1);
            count = model.parameter_count();
        }
        rows.push_back({{"preset", "gpt2-nano"}, {"vocab", vocab}, {"d", d}, {"T", T},
                        {"heads", heads}, {"layers", layers}, {"params", count}});
    }
    for (size_t ff : {48, 64}) {
        nn::LlamaConfig cfg;
        cfg.vocab = 64; cfg.d = 16; cfg.n_layers = 2; cfg.n_heads = 2; cfg.n_ctx = 8;
        cfg.d_ff = ff;
        nn::Llama model(cfg, 1);
        rows.push_back({{"preset", "llama-tiny"}, {"vocab", cfg.vocab}, {"d", cfg.d},
                        {"T", cfg.n_ctx}, {"heads", cfg.n_heads}, {"layers", cfg.n_layers},
                        {"d_ff", ff}, {"params", model.parameter_count()}});
    }
    std::cout << rows.dump() << "\n";
}
