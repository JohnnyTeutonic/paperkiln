// Exercise the actual private CLI helpers, without extracting or rewriting them.
#define main embedded_mtstudio_main
#include "../tools/mtstudio.cpp"
#undef main
#include "check.hpp"

struct ReferenceSample {
    int id;
    double logp, logp_sample;
    std::mt19937 generator;
};

static Matrix fixture(size_t rows) {
    Matrix m(rows, 4);
    const float values[8] = {-4.0f, -1.0f, 1.0f, 2.0f, 2.0f, 0.0f, -2.0f, 1.0f};
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < 4; ++j) m(i, j) = values[i * 4 + j];
    return m;
}

static std::vector<double> reference_logsoftmax(const Matrix& m, double temperature = 1.0) {
    std::vector<double> result(m.rows() * 4);
    for (size_t i = 0; i < m.rows(); ++i) {
        double maximum = -INFINITY;
        for (size_t j = 0; j < 4; ++j) maximum = std::max(maximum, double(m(i, j)) / temperature);
        double denominator = 0.0;
        for (size_t j = 0; j < 4; ++j)
            denominator += std::exp(double(m(i, j)) / temperature - maximum);
        for (size_t j = 0; j < 4; ++j)
            result[i * 4 + j] = double(m(i, j)) / temperature - maximum - std::log(denominator);
    }
    return result;
}

static ReferenceSample reference_sample() {
    const Matrix m = fixture(1);
    const auto lp = reference_logsoftmax(m), lpt = reference_logsoftmax(m, 0.75);
    std::mt19937 gen(0);
    double draw = std::uniform_real_distribution<double>(0.0, 1.0)(gen);
    int chosen = 3;
    for (size_t j = 0; j < 4; ++j)
        if ((draw -= std::exp(lpt[j])) <= 0.0) {
            chosen = static_cast<int>(j);
            break;
        }
    return {chosen, lp[chosen], lpt[chosen], gen};
}

static Var producer(size_t rows, bool gpu) {
    auto value = make_var(fixture(rows));
    if (!gpu) return value;
    CHECK(!grad_enabled());
    CHECK(device::deferring());
    auto copied = ops::slice_cols(value, 0, 4);
    CHECK(device::host_stale(copied->data));
    // Only the stale host bytes are poisoned; the real GPU copy is untouched.
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < 4; ++j) copied->data(i, j) = 0.0f;
    CHECK(device::host_stale(copied->data));
    std::printf(
        "{\"actual_cuda\":true,\"stale_producer\":true,\"host_poison\":true,\"no_grad_context\":"
        "true,\"deferred_window\":true}\n");
    std::fflush(stdout);
    return copied;
}

static void sample_case(bool gpu) {
    const auto expected = reference_sample();
    std::mt19937 gen(0);
    size_t calls = 0;
    auto actual = sample_pure(
        [&](const std::vector<int>& context) {
            CHECK(context == std::vector<int>{0});
            ++calls;
            return producer(1, gpu);
        },
        {0}, 1, 4, 0.75f, 1, -1, gen);
    CHECK(std::fabs(actual.logp - expected.logp) < 1e-12);
    CHECK(std::fabs(actual.logp_sample - expected.logp_sample) < 1e-12);
    CHECK(actual.ids == std::vector<int>{expected.id});
    CHECK(actual.stop == "length");
    CHECK(gen == expected.generator);
    CHECK(calls == 1);
}

static void logsoftmax_case(bool gpu) {
    const auto expected = reference_logsoftmax(fixture(2));
    auto value = producer(2, gpu);
    Matrix actual = log_softmax_rows(value->data);
    CHECK(actual.rows() == 2 && actual.cols() == 4);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j) CHECK(std::fabs(actual(i, j) - expected[i * 4 + j]) < 1e-6);
}

int main(int argc, char** argv) {
    bool gpu = false;
    std::string which = "all";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--require-cuda")
            gpu = true;
        else if (arg == "--case" && i + 1 < argc)
            which = argv[++i];
        else
            return 2;
    }
    if (which != "all" && which != "sample" && which != "logsoftmax") return 2;
#ifdef MICROTORCH_DEVCHECK
    constexpr bool checks = true;
#else
    constexpr bool checks = false;
#endif
    if (gpu && (!device::cuda_compiled() || !checks)) {
        std::fprintf(stderr, "requires actual CUDA and DEVCHECK\n");
        return 2;
    }
    device::set(device::Device::CPU);
    device::set_device_ops(false);
    // Retain the exact CPU helper IDs, probabilities and RNG advancement first.
    {
        NoGrad ng;
        sample_case(false);
        logsoftmax_case(false);
    }
    if (gpu) {
        device::set(device::Device::CUDA);
        device::set_device_ops(true);
        device::set_step_residency(true);
        device::set_defer_downloads(true);
        device::step_begin();
        CHECK(device::deferring());
    }
    {
        NoGrad ng;  // The real GRPO training sampling/reference context.
        if (which == "all" || which == "sample") sample_case(gpu);
        if (which == "all" || which == "logsoftmax") logsoftmax_case(gpu);
    }
    if (gpu) {
        device::step_end();
        device::set_defer_downloads(false);
        device::set_step_residency(false);
        device::set_device_ops(false);
        device::evict_all();
        device::set(device::Device::CPU);
    }
    std::printf(
        "{\"pass\":true,\"case\":\"%s\",\"actual_cuda\":%s,\"devcheck_enabled\":%s,"
        "\"no_grad_context\":true,\"cpu_reference\":true,\"stale_producer\":%s,\"host_poison\":%s,"
        "\"sampler_ids_logp_rng\":%s,\"all_softmax_rows\":%s}\n",
        which.c_str(), gpu ? "true" : "false", checks ? "true" : "false", gpu ? "true" : "false",
        gpu ? "true" : "false", which != "logsoftmax" ? "true" : "false",
        which != "sample" ? "true" : "false");
}
