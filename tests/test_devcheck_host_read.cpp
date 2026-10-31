// Real host-read boundaries: CPU predicate checks plus opt-in CUDA integration.
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include "../tools/parity_model.hpp"
#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/nn.hpp"

using namespace microtorch;

bool throws_at(const std::function<void()>& fn, const std::string& message) {
    try {
        fn();
    } catch (const std::runtime_error& e) {
        return std::string(e.what()).find(message) != std::string::npos;
    }
    return false;
}

int main(int argc, char** argv) {
    const bool require_cuda = argc == 2 && std::string(argv[1]) == "--require-cuda";
    if (argc > 2 || (argc == 2 && !require_cuda)) return 2;
#ifdef MICROTORCH_DEVCHECK
    constexpr bool checks = true;
#else
    constexpr bool checks = false;
#endif
    device::detail::check_host_read_state(false, false, "host fresh");
    CHECK(throws_at(
        [] { device::detail::check_host_read_state(true, false, "missing materialization"); },
        "device-stale tensor at missing materialization"));
    CHECK(throws_at([] { device::detail::check_host_read_state(false, true, "released value"); },
                    "released activation at released value"));
    Matrix host(2, 2, 1.0f);
    int evaluations = 0;
    MT_DEVCHECK_HOST_READ((++evaluations, host), "macro argument evaluation");
    CHECK(evaluations == (checks ? 1 : 0));
    nn::Linear layer(2, 2, false, 7);
    const auto p = layer.parameters().at(0);
    CHECK(layer.state_dict().size() == 1);
    std::vector<gguf::NamedTensor> exported;
    parity::put_linear(exported, "linear", layer);
    CHECK(exported.size() == 1);
    CHECK(exported.front().m(0, 0) == p->data(0, 0));
    nn::Embedding tokens(3, 2, 11), positions(4, 2, 12);
    nn::LayerNorm norm(2);
    std::map<std::string, Matrix> shell;
    parity::put_gpt2_shell(shell, tokens, positions, norm, layer);
    CHECK(shell.size() == 5);
    CHECK(shell.at("wte.weight")(0, 0) == tokens.weight->data(0, 0));
    if (require_cuda) {
        if (!device::cuda_compiled() || !checks) {
            std::fprintf(stderr, "requires actual CUDA build and MICROTORCH_DEVCHECK=ON\n");
            return 2;
        }
        device::set(device::Device::CUDA);
        device::set_device_ops(true);
        device::set_step_residency(true);
        device::set_defer_downloads(true);
        device::step_begin();
        CHECK(device::get() == device::Device::CUDA && device::deferring());
        CHECK(device::devops::fill(p->data, 3.25f));
        CHECK(device::host_stale(p->data));
        // This is the real Module::state_dict host copy, not a mocked cache.
        CHECK(throws_at([&] { (void)layer.state_dict(); },
                        "device-stale tensor at Module::state_dict"));
        exported.clear();
        CHECK(throws_at([&] { parity::put_linear(exported, "linear", layer); },
                        "device-stale tensor at parity family tensor export copy"));
        CHECK(exported.empty());
        device::materialize(p->data);
        MT_DEVCHECK_HOST_READ(p->data, "materialized parameter");
        CHECK(!device::host_stale(p->data));
        const auto sd = layer.state_dict();
        CHECK(sd.at("weight")(0, 0) == 3.25f);
        parity::put_linear(exported, "linear", layer);
        CHECK(exported.size() == 1);
        CHECK(exported.front().m(0, 0) == 3.25f);
        CHECK(device::devops::fill(p->data, 4.5f));
        CHECK(device::host_stale(p->data));
        device::release(p->data);
        // release clears stale: a stale-only guard incorrectly accepts this.
        CHECK(!device::host_stale(p->data));
        CHECK(throws_at([&] { (void)layer.state_dict(); },
                        "released activation at Module::state_dict"));
        exported.clear();
        CHECK(throws_at([&] { parity::put_linear(exported, "linear", layer); },
                        "released activation at parity family tensor export copy"));
        CHECK(exported.empty());
        CHECK(throws_at([&] { device::materialize(p->data); }, "released activation"));
        device::discard(p->data);
        shell.clear();
        CHECK(device::devops::fill(tokens.weight->data, 2.75f));
        CHECK(device::host_stale(tokens.weight->data));
        CHECK(throws_at([&] { parity::put_gpt2_shell(shell, tokens, positions, norm, layer); },
                        "device-stale tensor at parity GPT2 export copy"));
        CHECK(shell.empty());
        device::materialize(tokens.weight->data);
        parity::put_gpt2_shell(shell, tokens, positions, norm, layer);
        CHECK(shell.at("wte.weight")(0, 0) == 2.75f);
        CHECK(device::devops::fill(tokens.weight->data, 1.5f));
        device::release(tokens.weight->data);
        shell.clear();
        CHECK(throws_at([&] { parity::put_gpt2_shell(shell, tokens, positions, norm, layer); },
                        "released activation at parity GPT2 export copy"));
        CHECK(shell.empty());
        device::discard(tokens.weight->data);
        device::step_end();
        device::set_defer_downloads(false);
        device::set_step_residency(false);
        device::set_device_ops(false);
        device::evict_all();
        device::set(device::Device::CPU);
    }
    std::printf(
        "{\"pass\":true,\"devcheck_enabled\":%s,\"cuda_integration\":%s,"
        "\"stale_and_released_predicate_negatives\":true,\"macro_evaluation_checked\":true}\n",
        checks ? "true" : "false", require_cuda ? "true" : "false");
    return 0;
}
