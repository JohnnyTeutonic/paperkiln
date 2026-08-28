// Phase-1c success test, exactly as docs/DESIGN.md section 3 defines it: load a
// small pretrained HF model from safetensors, reproduce its logits within
// tolerance, then fine-tune it through the new autograd.
//
//   gpt2_parity <model.safetensors> <reference.json>
//
// reference.json comes from tools/make_gpt2_reference.py (HF transformers,
// float32, eval mode): {"tokens": [...], "rows": T, "cols": V,
// "logits": [T*V floats, row-major]}.
#include <cmath>
#include <cstdio>
#include <fstream>

#include <nlohmann/json.hpp>

#include "microtorch/nn.hpp"
#include "microtorch/safetensors.hpp"

using microtorch::Var;
namespace ops = microtorch::ops;
namespace nn = microtorch::nn;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: gpt2_parity <model.safetensors> <ref.json>\n");
        return 2;
    }

    // ---- load the checkpoint onto the module tree, strictly ----
    std::map<std::string, std::string> skipped;
    auto sd = microtorch::load_safetensors(argv[1], &skipped);
    std::printf("loaded %zu tensors (%zu skipped by rank)\n", sd.size(), skipped.size());
    nn::GPT2 model{nn::GPT2Config{}};
    model.load_state_dict(sd, /*strict=*/true);
    model.eval();

    // ---- reference ----
    std::ifstream rf(argv[2]);
    if (!rf) {
        std::fprintf(stderr, "cannot open %s\n", argv[2]);
        return 2;
    }
    const auto ref = nlohmann::json::parse(rf);
    const std::vector<int> tokens = ref.at("tokens").get<std::vector<int>>();
    const size_t R = ref.at("rows").get<size_t>();
    const size_t C = ref.at("cols").get<size_t>();
    const auto rl = ref.at("logits").get<std::vector<float>>();

    // ---- our logits, no tape ----
    Var logits;
    {
        microtorch::NoGrad ng;
        logits = model.forward(tokens);
    }
    if (logits->data.rows() != R || logits->data.cols() != C) {
        std::fprintf(stderr, "shape mismatch vs reference\n");
        return 1;
    }
    double worst = 0.0, worst_rel = 0.0;
    size_t argmax_hits = 0;
    for (size_t i = 0; i < R; ++i) {
        size_t am_ref = 0, am_our = 0;
        for (size_t j = 0; j < C; ++j) {
            const double a = logits->data(i, j), b = rl[i * C + j];
            worst = std::max(worst, std::abs(a - b));
            worst_rel = std::max(worst_rel, std::abs(a - b) / (1.0 + std::abs(b)));
            if (rl[i * C + j] > rl[i * C + am_ref]) am_ref = j;
            if (logits->data(i, j) > logits->data(i, am_our)) am_our = j;
        }
        argmax_hits += (am_ref == am_our);
    }
    std::printf("logits: max abs diff %.3e, max rel diff %.3e, argmax %zu/%zu\n", worst, worst_rel,
                argmax_hits, R);
    const bool parity = worst < 5e-2 && argmax_hits == R;
    std::printf("[%s] HF logit parity\n", parity ? "ok" : "FAIL");

    // ---- fine-tune through the tape: next-token CE on this sequence ----
    // Plain SGD (no optimizer state): the demonstration is that 124M real
    // parameters move through OUR backward and the loss obeys them.
    // lr note: 3e-3 diverged on the first try (3.49 -> 2.53 -> 11.0, the
    // classic overshoot); 1e-4 is in the regime GPT-2 fine-tuning actually
    // uses. The first step's drop is the gradient-correctness signal.
    model.train();
    std::vector<int> in(tokens.begin(), tokens.end() - 1);
    std::vector<int> tgt(tokens.begin() + 1, tokens.end());
    nn::SGD sgd(model.parameters(), /*lr=*/1e-4f);
    float first = 0.0f, last = 0.0f;
    for (int it = 0; it < 3; ++it) {
        Var loss = ops::cross_entropy(model.forward(in), tgt);
        last = loss->data(0, 0);
        if (it == 0) first = last;
        std::printf("  fine-tune step %d: loss %.4f\n", it, last);
        sgd.zero_grad();
        microtorch::backward(loss);
        sgd.step();
    }
    {  // the held-out step: did the last update help too?
        microtorch::NoGrad ng;
        Var loss = ops::cross_entropy(model.forward(in), tgt);
        std::printf("  after step 2:     loss %.4f\n", loss->data(0, 0));
        last = loss->data(0, 0);
    }
    const bool learns = last < first;
    std::printf("[%s] fine-tune: loss %.4f -> %.4f through the tape\n", learns ? "ok" : "FAIL",
                first, last);

    if (parity && learns) {
        std::printf("\nPHASE 1C SUCCESS TEST PASSED\n");
        return 0;
    }
    std::printf("\nFAILED\n");
    return 1;
}
