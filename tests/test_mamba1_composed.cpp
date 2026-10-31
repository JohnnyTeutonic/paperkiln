// Full raw parameter/gradient dump for an independent binary64 finite-difference oracle.
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include "../tools/parity_model.hpp"
#include "microtorch/device.hpp"
using namespace microtorch;
using json = nlohmann::json;
void need(bool x, const char* msg) {
    if (!x) throw std::runtime_error(msg);
}
std::vector<float> values(const Matrix& m) {
    std::vector<float> v;
    for (size_t i = 0; i < m.rows(); i++)
        for (size_t j = 0; j < m.cols(); j++) {
            need(std::isfinite(m(i, j)), "nonfinite raw tensor");
            v.push_back(m(i, j));
        }
    return v;
}
int main(int argc, char** argv) {
    try {
        bool gpu = false;
        json fixture;
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--require-cuda")
                gpu = true;
            else if (a == "--weights" && i + 1 < argc) {
                std::ifstream f(argv[++i]);
                f >> fixture;
            } else
                return 2;
        }
#ifdef MICROTORCH_DEVCHECK
        const bool checks = true;
#else
        const bool checks = false;
#endif
        if (gpu) {
            need(device::cuda_compiled() && checks, "requires actual CUDA and DEVCHECK");
            need(device::gpu_profile_on(), "requires GPU_PROFILE");
            need(!fixture.is_null(), "requires frozen weights for CUDA");
        }
        const std::vector<int> ids = {0, 1, 2, 3, 4, 5, 6, 0}, targets = {1, 2, 3, 4, 5, 6, 0, 1};
        for (size_t depth : {1, 2}) {
            device::set(device::Device::CPU);
            device::set_device_ops(false);
            parity::Mamba1Config config;
            config.on = true;
            config.d_conv = 3;
            config.expand = 1;
            config.dt_rank = 2;
            parity::MambaLM model(7, 8, depth, 2, 37, config);
            auto params = model.named_parameters();
            if (!fixture.is_null()) {
                const auto& fp = fixture.at(depth - 1).at("parameters");
                need(fp.size() == params.size(), "frozen parameter inventory");
                for (auto& [name, p] : params) {
                    const auto& v = fp.at(name);
                    need(v.at("shape") == json::array({p->data.rows(), p->data.cols()}),
                         "frozen parameter shape");
                    auto data = v.at("values").get<std::vector<float>>();
                    need(data.size() == p->data.rows() * p->data.cols(),
                         "frozen parameter data count");
                    for (size_t i = 0; i < p->data.rows(); i++)
                        for (size_t j = 0; j < p->data.cols(); j++)
                            p->data(i, j) = data[i * p->data.cols() + j];
                }
            }
            json parameters = json::object();
            for (const auto& [name, p] : params)
                parameters[name] = {{"shape", {p->data.rows(), p->data.cols()}},
                                    {"values", values(p->data)}};
            if (gpu) {
                device::set(device::Device::CUDA);
                device::set_device_ops(true);
                device::set_step_residency(true);
                device::set_defer_downloads(true);
                device::step_begin();
                device::gpu_profile_take();
            }
            auto logits = model.forward(ids, 4);
            auto loss = ops::cross_entropy(logits, targets);
            backward(loss);
            device::materialize(logits->data);
            device::materialize(loss->data);
            for (const auto& [name, p] : params) {
                device::materialize(p->grad);
                parameters[name]["gradient"] = values(p->grad);
                need(p->grad.rows() == p->data.rows() && p->grad.cols() == p->data.cols(),
                     "gradient shape");
            }
            json counts = json::object(), ms = json::object();
            if (gpu) {
                for (auto& [name, t] : device::gpu_profile_take()) ms[name] = t;
                for (auto& [name, n] : device::gpu_profile_counts()) counts[name] = n;
            }
            json row = {{"depth", depth},
                        {"actual_cuda", gpu},
                        {"devcheck", checks},
                        {"seed", 37},
                        {"d", 8},
                        {"vocab", 7},
                        {"state", 2},
                        {"expand", 1},
                        {"dt_rank", 2},
                        {"conv", 3},
                        {"batch", 2},
                        {"seq_len", 4},
                        {"ids", ids},
                        {"targets", targets},
                        {"rms_eps", ops::kRmsNormEps},
                        {"logits_shape", {8, 7}},
                        {"logits", values(logits->data)},
                        {"loss", loss->data(0, 0)},
                        {"parameters", parameters},
                        {"gpu_n", counts},
                        {"gpu_ms", ms}};
            puts(row.dump().c_str());
            fflush(stdout);
            if (gpu) {
                device::step_end();
                device::set_defer_downloads(false);
                device::set_step_residency(false);
                device::evict_all();
                device::set_device_ops(false);
                device::set(device::Device::CPU);
            }
        }
        puts(json({{"summary", true}, {"cases", 2}, {"actual_cuda", gpu}, {"pass", true}})
                 .dump()
                 .c_str());
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "Mamba1 composed: %s\n", e.what());
        return 2;
    }
}
