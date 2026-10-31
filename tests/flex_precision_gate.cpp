// CPU semantic proof and prospective CUDA harness; no paid execution is required.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <vector>
#include "../tools/parity_model.hpp"
#include "check.hpp"
#include "microtorch/device.hpp"
using namespace microtorch;
using J = nlohmann::json;
using device::Precision;
J dump(const Matrix& x) {
    device::materialize(x);
    J values = J::array();
    for (size_t i = 0; i < x.rows(); ++i)
        for (size_t j = 0; j < x.cols(); ++j) {
            CHECK(std::isfinite(x(i, j)));
            values.push_back(x(i, j));
        }
    return {{"shape", {x.rows(), x.cols()}}, {"values", values}};
}
bool bits(const Matrix& a, const Matrix& b) {
    device::materialize(a);
    device::materialize(b);
    return a.rows() == b.rows() && a.cols() == b.cols() &&
           std::memcmp(a.get_data(), b.get_data(), a.rows() * a.cols() * sizeof(float)) == 0;
}
void fixture(nn::Module& model) {
    for (auto& item : model.named_parameters()) {
        uint32_t h = 2166136261u;
        for (unsigned char c : item.first) h = (h ^ c) * 16777619u;
        const bool gain = item.first.find("ln_") != std::string::npos && item.first.size() >= 7 &&
                          item.first.substr(item.first.size() - 7) == ".weight";
        Matrix& x = item.second->data;
        for (size_t i = 0; i < x.rows() * x.cols(); ++i) {
            const int b = int(uint32_t(h + uint32_t(17 * i)) % 37u) - 18;
            x(i / x.cols(), i % x.cols()) = gain ? 1.f + float(b) / 512.f : float(b) / 64.f;
        }
    }
}
J counts() {
    device::gpu_profile_take();
    J c = J::object();
    for (const auto& x : device::gpu_profile_counts()) c[x.first] = x.second;
    return c;
}
// Preserved canonical Flex composition: explicitly invokes the FP32 legacy op.
Var legacy(parity::FlexLM& m, const std::vector<int>& ids, size_t sl) {
    std::vector<int> pos(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) pos[i] = int(i % sl);
    Var x = ops::add(m.wte->forward(ids), m.wpe->forward(pos));
    for (auto& b : m.blocks) {
        auto a = b->attn;
        const size_t d = a->H * a->dk;
        Var qkv = a->c_attn->forward(b->ns[0].ln->forward(x));
        std::vector<Var> hs;
        for (size_t h = 0; h < a->H; ++h)
            hs.push_back(ops::fused_attention(
                ops::slice_cols(qkv, h * a->dk, (h + 1) * a->dk),
                ops::slice_cols(qkv, d + h * a->dk, d + (h + 1) * a->dk),
                ops::slice_cols(qkv, 2 * d + h * a->dk, 2 * d + (h + 1) * a->dk),
                1.f / std::sqrt(float(a->dk)), sl, true));
        x = ops::add(x, a->c_proj->forward(ops::concat_cols(hs)));
        x = ops::add(x, b->mlp->forward(b->ns[1].ln->forward(x)));
    }
    return m.head->forward(m.ln_f_ln->forward(x));
}
J probe(size_t layers, size_t hidden, Precision precision, bool gpu) {
    parity::FlexConfig c;
    c.vocab = 7;
    c.d = 8;
    c.n_heads = 2;
    c.n_ctx = 4;
    c.n_layers = layers;
    c.d_ff = hidden;
    parity::FlexLM model(c, 37, precision);
    fixture(model);
    CHECK(device::matmul_precision() == Precision::FP32);
    for (auto& b : model.blocks) CHECK(b->attn->attention_precision == precision);
    const std::vector<int> ids = {1, 2, 3, 4, 6, 5, 2, 1}, targets = {2, 3, 4, 1, 5, 2, 1, 6};
    J parameters = J::object();
    for (auto& p : model.named_parameters()) parameters[p.first] = dump(p.second->data);
    counts();
    device::step_begin();
    Var y = model.forward(ids, 4);
    J logits = dump(y->data);
    auto loss = ops::cross_entropy(y, targets);
    J l = dump(loss->data);
    backward(loss);
    J gradients = J::object();
    for (auto& p : model.named_parameters()) gradients[p.first] = dump(p.second->grad);
    device::step_end();
    J profile = counts();
    if (gpu)
        CHECK(profile.value("gemm_bf16", 0) ==
              (precision == Precision::BF16 ? int(layers * 2 * 6) : 0));
    return {{"precision", precision == Precision::BF16 ? "bf16" : "fp32"},
            {"layers", layers},
            {"d_ff", hidden},
            {"d", 8},
            {"heads", 2},
            {"vocab", 7},
            {"seq_len", 4},
            {"ids", ids},
            {"targets", targets},
            {"parameters", parameters},
            {"logits", logits},
            {"loss", l},
            {"gradients", gradients},
            {"profile", profile}};
}
J default_proof(size_t layers, size_t hidden) {
    parity::FlexConfig c;
    c.vocab = 7;
    c.d = 8;
    c.n_heads = 2;
    c.n_ctx = 4;
    c.n_layers = layers;
    c.d_ff = hidden;
    parity::FlexLM old(c, 37), now(c, 37, Precision::FP32), mixed(c, 37, Precision::BF16);
    fixture(old);
    fixture(now);
    fixture(mixed);
    CHECK(old.export_ir() == now.export_ir() && old.export_ir() == mixed.export_ir());
    std::vector<int> ids = {1, 2, 3, 4, 6, 5, 2, 1}, labels = {2, 3, 4, 1, 5, 2, 1, 6};
    device::step_begin();
    auto a = legacy(old, ids, 4), b = now.forward(ids, 4);
    CHECK(bits(a->data, b->data));
    backward(ops::cross_entropy(a, labels));
    backward(ops::cross_entropy(b, labels));
    auto pa = old.named_parameters(), pb = now.named_parameters();
    CHECK(pa.size() == pb.size());
    for (size_t i = 0; i < pa.size(); ++i) {
        CHECK(pa[i].first == pb[i].first);
        CHECK(bits(pa[i].second->grad, pb[i].second->grad));
    }
    device::step_end();
    // Training mode is per-instance, outside cfg/state/IR. Reload is explicit FP32.
    parity::FlexLM loaded(c, 7);
    loaded.load_state_dict(mixed.state_dict());
    for (auto& x : loaded.blocks) CHECK(x->attn->attention_precision == Precision::FP32);
    CHECK(bits(loaded.forward(ids, 4)->data, now.forward(ids, 4)->data));
    return {{"legacy_fp32_logits_and_all_gradients_bit_equal", true},
            {"IR_precision_independent", true},
            {"reload_fp32_bit_equal", true}};
}
void contract_controls() {
    parity::FlexConfig good;
    good.vocab = 7;
    good.d = 8;
    good.n_heads = 2;
    good.n_ctx = 4;
    good.d_ff = 12;
    std::vector<parity::FlexConfig> bad;
    auto change = [&](const std::function<void(parity::FlexConfig&)>& f) {
        auto c = good;
        f(c);
        bad.push_back(c);
    };
    change([](auto& c) { c.attention = "swa"; });
    change([](auto& c) { c.pos = "rope"; });
    change([](auto& c) { c.pos = "t5_relative"; });
    change([](auto& c) { c.pos = "shaw_relative"; });
    change([](auto& c) { c.qkv_conv = 3; });
    change([](auto& c) { c.qk_norm = "per_head"; });
    change([](auto& c) { c.act = "relu"; });
    change([](auto& c) { c.act = "swiglu"; });
    change([](auto& c) { c.norm = "rmsnorm"; });
    change([](auto& c) { c.residual = "highway"; });
    change([](auto& c) { c.place_attn = "post"; });
    change([](auto& c) { c.place_ffn = "sandwich"; });
    change([](auto& c) { c.final_norm = 0; });
    change([](auto& c) { c.tie_head = true; });
    change([](auto& c) { c.embed_expr = "expr:x"; });
    change([](auto& c) { c.logits_expr = "expr:x"; });
    change([](auto& c) { c.n_layers = 0; });
    change([](auto& c) { c.n_heads = 0; });
    for (const auto& c : bad) {
        bool refused = false;
        try {
            parity::FlexLM m(c, 37, Precision::BF16);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        CHECK(refused);
        refused = false;
        try {
            parity::FlexBlock b(c, 37, Precision::BF16);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        CHECK(refused);
    }
    bool refused = false;
    try {
        parity::FlexLM m(good, 37, static_cast<Precision>(99));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    CHECK(refused);
}
int main(int argc, char** argv) {
    try {
        bool gpu = false;
        std::string output;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--require-cuda")
                gpu = true;
            else if (a == "--out" && i + 1 < argc)
                output = argv[++i];
            else
                throw std::runtime_error("unknown argument");
        }
        if (gpu) {
            CHECK(device::cuda_compiled() && device::gpu_profile_on());
#ifndef MICROTORCH_DEVCHECK
            throw std::runtime_error("DEVCHECK required");
#endif
        }
        device::set(gpu ? device::Device::CUDA : device::Device::CPU);
        device::set_device_ops(gpu);
        device::set_step_residency(gpu);
        device::set_defer_downloads(gpu);
        device::set_matmul_precision(Precision::FP32);
        ops::set_attention_kernel(ops::AttnKernel::Tiled);
        contract_controls();
        J cases = J::array();
        for (auto shape : std::vector<std::pair<size_t, size_t>>{{1, 12}, {3, 20}}) {
            J proof = default_proof(shape.first, shape.second);
            for (auto p : {Precision::FP32, Precision::BF16}) {
                J row = probe(shape.first, shape.second, p, gpu);
                row["default_proof"] = proof;
                cases.push_back(row);
            }
        }
        const J result = {{"pass", true}, {"actual_cuda", gpu}, {"cases", cases}};
        if (output.empty())
            std::puts(result.dump().c_str());
        else {
            CHECK(!std::ifstream(output).good());
            std::ofstream f(output);
            CHECK(f.good());
            f << result.dump() << "\n";
            f.flush();
            CHECK(f.good());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
