// e9_quality: registered criterion 3 of AI_ML/e9_f16kv_claude/REGISTRATION.md, on the device.
// CUDA-f16 against CUDA-f32 on ir_device_gate's prompts (--n 12 --len 24 --seed 0): prefill rows at every
// position and the device's token-by-token decode against the f32 prefill. Pass iff the maximum rel
// (|dlogit| / f32 row range) <= --limit (1.5 x the CPU's f16-vs-f32 maximum, measured beforehand).
//
//   e9_quality MODEL.gguf --limit L [--n 12] [--len 24] [--seed 0] > out.json
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <random>
#include <string>
#include <vector>

#ifndef E9_QUALITY_CHECKS_ONLY
#include <nlohmann/json.hpp>

#include "gguf_parser.h"
#include "graph_ir.h"

using json = nlohmann::json;
#endif

static void require_logits_rows(const std::vector<float>& values, size_t rows, size_t vocab) {
  if (rows == 0 || vocab == 0 || rows > std::numeric_limits<size_t>::max() / vocab ||
      values.size() != rows * vocab)
    throw std::runtime_error("e9_quality: incomplete or invalid logit shape");
  for (float value : values)
    if (!std::isfinite(value))
      throw std::runtime_error("e9_quality: nonfinite logits");
}

static double rel_row(const float* a, const float* ref, int n) {
  if (a == nullptr || ref == nullptr || n <= 0)
    throw std::runtime_error("e9_quality: empty logit row");
  float lo = ref[0], hi = ref[0];
  double d = 0.0;
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(ref[i]))
      throw std::runtime_error("e9_quality: nonfinite logit row");
    lo = std::min(lo, ref[i]);
    hi = std::max(hi, ref[i]);
    d = std::max(d, std::fabs(static_cast<double>(a[i]) - ref[i]));
  }
  const float range = hi - lo;  // retain the registered ordinary finite arithmetic
  if (!std::isfinite(range))
    throw std::runtime_error("e9_quality: nonfinite derived row range");
  const double error = hi > lo ? d / range : d;
  if (!std::isfinite(error))
    throw std::runtime_error("e9_quality: nonfinite relative row error");
  return error;
}

#ifndef E9_QUALITY_CHECKS_ONLY
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: e9_quality MODEL.gguf --limit L [--n 12] [--len 24] [--seed 0]\n";
    return 2;
  }
  int n = 12, len = 24;
  unsigned seed = 0;
  double limit = -1.0;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--n") n = std::atoi(argv[i + 1]);
    else if (k == "--len") len = std::atoi(argv[i + 1]);
    else if (k == "--seed") seed = static_cast<unsigned>(std::atoi(argv[i + 1]));
    else if (k == "--limit") limit = std::atof(argv[i + 1]);
  }
  if (!(limit > 0)) {
    std::cerr << "--limit is required\n";
    return 2;
  }
  const GGUFData gguf = load_gguf_meta(argv[1], true);
  const std::string ir = ember_ir::find_ir(gguf);
  setenv("EMBER_KV", "f16", 1);
  ember_ir::Interpreter f16(ir, gguf, true);
  setenv("EMBER_KV", "f32", 1);
  ember_ir::Interpreter f32(ir, gguf, true);
  unsetenv("EMBER_KV");
  if (!f16.on_gpu() || !f32.on_gpu()) {
    std::cout << json{{"on_gpu", false}, {"pass", false}}.dump(1) << "\n";
    return 1;
  }
  const int V = f16.vocab();
  if (n <= 0 || len <= 0 || V <= 0 || f32.vocab() != V)
    throw std::runtime_error("e9_quality: invalid request or vocabulary geometry");
  const int hi = std::min<int>(V, static_cast<int>(gguf.tokenizer_tokens.empty() ? V : gguf.tokenizer_tokens.size()));
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> pick(0, hi - 1);
  double worst = 0.0, worst_decode = 0.0;
  json rows = json::array();
  for (int s = 0; s < n; ++s) {
    std::vector<int> ids(len);
    for (int& t : ids) t = pick(rng);
    const std::vector<float> a = f16.forward(ids, 0, true), b = f32.forward(ids, 0, true);
    require_logits_rows(a, static_cast<size_t>(len), static_cast<size_t>(V));
    require_logits_rows(b, static_cast<size_t>(len), static_cast<size_t>(V));
    double w = 0.0, wd = 0.0;
    for (int t = 0; t < len; ++t) w = std::max(w, rel_row(a.data() + size_t(t) * V, b.data() + size_t(t) * V, V));
    for (int t = 0; t < len; ++t) {
      const std::vector<float> d = f16.forward({ids[t]}, t, false);
      require_logits_rows(d, 1, static_cast<size_t>(V));
      wd = std::max(wd, rel_row(d.data() + d.size() - V, b.data() + size_t(t) * V, V));
    }
    worst = std::max(worst, w);
    worst_decode = std::max(worst_decode, wd);
    rows.push_back({{"i", s}, {"prefill_rel", w}, {"decode_rel", wd}});
  }
  if (!std::isfinite(worst) || !std::isfinite(worst_decode))
    throw std::runtime_error("e9_quality: nonfinite final worst error");
  const bool pass = worst <= limit && worst_decode <= limit;
  std::cout << json{{"gguf", argv[1]}, {"n", n}, {"len", len}, {"seed", seed}, {"limit", limit},
                    {"worst_prefill_rel", worst}, {"worst_decode_rel", worst_decode}, {"pass", pass}, {"rows", rows}}
                   .dump(1)
            << "\n";
  return pass ? 0 : 1;
}

#endif
