// Synthetic proof-parser format probe only: no CUDA claim or numerical-op result.
#include "../../../tests/check.hpp"
int main() {
    std::puts(R"({"case":"gelu_forward_x","actual_cuda":true,"stale_operand_proved":true,"host_poison":true,"ops_disabled_before_fallback":true})");
    std::puts(R"({"case":"gelu_forward_x","quantity":"forward","normalized_max_error":0.1,"numeric_reference_pass":false})");
    double maximum = .1;
#line 77 "test_fallback_materialization.cpp"
    CHECK(maximum < 2e-5);
}
