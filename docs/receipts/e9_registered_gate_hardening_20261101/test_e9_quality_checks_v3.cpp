#define E9_QUALITY_CHECKS_ONLY
#include "e9_quality.v3.cpp"
#include <cassert>
#include <functional>
#include <limits>
static double rel_row_original(const float* a, const float* ref, int n) {
  float lo = ref[0], hi = ref[0];
  double d = 0.0;
  for (int i = 0; i < n; ++i) {
    lo = std::min(lo, ref[i]);
    hi = std::max(hi, ref[i]);
    d = std::max(d, std::fabs(static_cast<double>(a[i]) - ref[i]));
  }
  return hi > lo ? d / (hi - lo) : d;
}


static int rejects = 0;
static void reject(const std::function<void()>& f) {
  bool failed = false;
  try { f(); } catch (const std::runtime_error&) { failed = true; }
  assert(failed); ++rejects;
}
int main() {
  const std::vector<float> ref{1.0f, 2.0f, 3.0f};
  auto actual = ref; actual[1] = 2.25f;
  require_logits_rows(ref, 1, 3); require_logits_rows(actual, 1, 3);
  assert(rel_row(actual.data(), ref.data(), 3) == 0.125);
  for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity()}) {
    auto corrupt = ref; corrupt[1] = bad;
    reject([&] { rel_row(corrupt.data(), ref.data(), 3); });
    reject([&] { rel_row(ref.data(), corrupt.data(), 3); });
    reject([&] { require_logits_rows(corrupt, 1, 3); });
  }
  reject([&] { require_logits_rows({}, 1, 3); });
  reject([&] { require_logits_rows({1.0f, 2.0f}, 1, 3); });
  reject([&] { require_logits_rows({1.0f, 2.0f, 3.0f, 4.0f}, 1, 3); });
  reject([&] { require_logits_rows(ref, 0, 3); });
  reject([&] { require_logits_rows(ref, 1, 0); });
  reject([&] { require_logits_rows(ref, std::numeric_limits<size_t>::max(), 2); });
  reject([&] { rel_row(ref.data(), ref.data(), 0); });
  reject([&] { rel_row(nullptr, ref.data(), 3); });
  reject([&] { rel_row(ref.data(), nullptr, 3); });
  std::mt19937 rng(71);
  std::uniform_real_distribution<float> draw(-1000.0f, 1000.0f);
  for (int trial = 0; trial < 256; ++trial) {
    std::vector<float> a(17), b(17);
    for (int i = 0; i < 17; ++i) { a[i] = draw(rng); b[i] = draw(rng); }
    assert(rel_row(a.data(), b.data(), 17) == rel_row_original(a.data(), b.data(), 17));
  }
  const std::vector<float> flat{1.0f,1.0f,1.0f};
  assert(rel_row(ref.data(),flat.data(),3)==rel_row_original(ref.data(),flat.data(),3));
  const float largest = std::numeric_limits<float>::max();
  const std::vector<float> extreme{-largest, 0.0f, largest};
  reject([&] { rel_row(extreme.data(), extreme.data(), 3); });
  reject([&] { rel_row(ref.data(), extreme.data(), 3); });
  std::cout << "{\"pass\":true,\"negative_cases\":" << rejects
            << ",\"finite_reference_error\":0.125}\n";
}
