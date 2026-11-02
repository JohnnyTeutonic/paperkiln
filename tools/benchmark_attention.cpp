// Benchmark: Kimi Linear attention vs standard scaled-dot-product attention
// Measures forward time, backward time, and output accuracy

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <vector>

#include "microtorch/autograd.hpp"
#include "microtorch/nn.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

struct BenchmarkResult {
    std::string name;
    double forward_ms;
    double backward_ms;
    double total_ms;
    float output_norm;  // For accuracy comparison
};

// Benchmark a single forward+backward pass
template <typename AttentionModule>
BenchmarkResult benchmark_attention(const std::string& name, size_t seq_len, size_t d_model,
                                    size_t n_heads, int iterations = 5) {
    AttentionModule attn(d_model, n_heads, 42);
    attn.train();

    // Create input
    Matrix x_data(seq_len, d_model);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (size_t i = 0; i < x_data.rows(); ++i) {
        for (size_t j = 0; j < x_data.cols(); ++j) {
            x_data(i, j) = dist(rng);
        }
    }

    BenchmarkResult result{};
    result.name = name;
    result.forward_ms = 0;
    result.backward_ms = 0;

    for (int iter = 0; iter < iterations; ++iter) {
        Var x = make_var(x_data, true);

        // Forward
        auto t_start = std::chrono::high_resolution_clock::now();
        Var y = attn.forward(x);
        auto t_forward = std::chrono::high_resolution_clock::now();

        // Compute scalar loss (mean output)
        float sum = 0;
        for (size_t i = 0; i < y->data.rows(); ++i) {
            for (size_t j = 0; j < y->data.cols(); ++j) {
                sum += y->data(i, j);
            }
        }
        Var loss = make_var(Matrix(1, 1, sum / y->data.size()));
        result.output_norm = std::abs(loss->data(0, 0));

        // Backward (gradients enabled by default when requires_grad=true)
        backward(loss);
        auto t_backward = std::chrono::high_resolution_clock::now();

        result.forward_ms += std::chrono::duration<double, std::milli>(t_forward - t_start).count();
        result.backward_ms +=
            std::chrono::duration<double, std::milli>(t_backward - t_forward).count();
    }

    result.forward_ms /= iterations;
    result.backward_ms /= iterations;
    result.total_ms = result.forward_ms + result.backward_ms;

    return result;
}

int main() {
    printf("=== Attention Mechanism Benchmark ===\n");
    printf("Comparing Kimi Linear (O(n*d²)) vs Standard Attention (O(n²*d))\n\n");

    // Test configurations
    std::vector<std::tuple<size_t, size_t, size_t>> configs = {
        {16, 256, 4},   // seq_len=16, d=256, heads=4
        {32, 512, 8},   // seq_len=32, d=512, heads=8
        {64, 768, 12},  // seq_len=64, d=768, heads=12
    };

    printf("%10s | %6s | %7s | %10s | %10s | %10s | %s\n", "Config", "Seq", "Dim", "Fwd (ms)",
           "Bwd (ms)", "Total (ms)", "Speedup");
    printf("%s\n", std::string(85, '-').c_str());

    for (const auto& [seq, d, heads] : configs) {
        // Benchmark standard attention
        auto standard = benchmark_attention<nn::CausalSelfAttention>("Standard", seq, d, heads, 3);

        // Benchmark Kimi Linear
        auto kimi = benchmark_attention<nn::KimiLinearAttention>("KimiLinear", seq, d, heads, 3);

        printf("Config [%2zu, %3zu, %2zu]:\n", seq, d, heads);
        printf("  Standard Attention: Forward %6.2fms | Backward %6.2fms | Total %6.2fms\n",
               standard.forward_ms, standard.backward_ms, standard.total_ms);
        printf("  Kimi Linear:        Forward %6.2fms | Backward %6.2fms | Total %6.2fms\n",
               kimi.forward_ms, kimi.backward_ms, kimi.total_ms);

        double speedup = standard.total_ms / kimi.total_ms;
        printf("  Speedup: %.2fx", speedup);

        if (kimi.total_ms < standard.total_ms) {
            printf(" ✓ FASTER\n\n");
        } else if (kimi.total_ms > standard.total_ms * 1.1) {
            printf(" ✗ SLOWER\n\n");
        } else {
            printf(" ~ COMPARABLE\n\n");
        }
    }

    printf("\n=== Accuracy Check ===\n");
    printf("Output norms (for sanity check):\n");

    for (const auto& [seq, d, heads] : configs) {
        auto standard = benchmark_attention<nn::CausalSelfAttention>("Standard", seq, d, heads, 1);
        auto kimi = benchmark_attention<nn::KimiLinearAttention>("Kimi", seq, d, heads, 1);

        printf("  [%zu, %zu, %zu] Standard: %.6f  Kimi: %.6f\n", seq, d, heads,
               standard.output_norm, kimi.output_norm);
    }

    printf("\nNotes:\n");
    printf("- Speedup > 1.0 means Kimi Linear is faster\n");
    printf("- Measurements on CPU; GPU results may differ significantly\n");
    printf("- Theory: Kimi should dominate at longer sequences (n >> d)\n");
    printf("- Current test is n ~ d regime; advantage grows with seq_len\n");

    return 0;
}
