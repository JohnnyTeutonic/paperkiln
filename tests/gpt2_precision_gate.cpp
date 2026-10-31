// Test-only access to production CLI constructors. Never changes the CLI protocol.
#define main precision_gate_embedded_cli_main
#include "../tools/mtstudio.cpp"
#undef main
#include <cstring>

namespace {
void require_gate(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
json values(const Matrix& m) {
    device::materialize(m);
    json a = json::array();
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j) {
            require_gate(std::isfinite(m(i, j)), "nonfinite gate tensor");
            a.push_back(m(i, j));
        }
    return {{"rows", m.rows()}, {"cols", m.cols()}, {"values", a}};
}
bool exact_bits(const Matrix& a, const Matrix& b) {
    device::materialize(a);
    device::materialize(b);
    return a.rows() == b.rows() && a.cols() == b.cols() &&
           std::memcmp(a.get_data(), b.get_data(), a.rows() * a.cols() * sizeof(float)) == 0;
}
json counts() {
    device::gpu_profile_take();
    json c = json::object();
    for (auto& item : device::gpu_profile_counts()) c[item.first] = item.second;
    return c;
}
void setup(bool gpu) {
#ifdef MICROTORCH_DEVCHECK
    const bool checks = true;
#else
    const bool checks = false;
#endif
    require_gate(!gpu || (checks && device::cuda_compiled() && device::gpu_profile_on()),
                 "actual CUDA+DEVCHECK+profiler required");
    device::set(gpu ? device::Device::CUDA : device::Device::CPU);
    device::set_device_ops(gpu);
    device::set_step_residency(gpu);
    device::set_defer_downloads(gpu);
    device::set_matmul_precision(device::Precision::FP32);
    ops::set_attention_kernel(ops::AttnKernel::Tiled);
}
// Preserved pre-change exact ParityLM forward and CausalSelfAttention composition.
Var legacy_forward(parity::ParityLM& m, const std::vector<int>& ids, size_t sl) {
    std::vector<int> pos(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) pos[i] = int(i % sl);
    Var h = ops::add(m.wte->forward(ids), m.wpe->forward(pos));
    for (size_t b = 0; b < m.ln1.size(); ++b) {
        auto a = m.exact[b];
        Var qkv = a->c_attn->forward(m.ln1[b]->forward(h));
        const size_t d = a->H * a->dk;
        const float sc = 1.f / std::sqrt(float(a->dk));
        std::vector<Var> heads;
        for (size_t j = 0; j < a->H; ++j) {
            Var q = ops::slice_cols(qkv, j * a->dk, (j + 1) * a->dk);
            Var k = ops::slice_cols(qkv, d + j * a->dk, d + (j + 1) * a->dk);
            Var v = ops::slice_cols(qkv, 2 * d + j * a->dk, 2 * d + (j + 1) * a->dk);
            heads.push_back(ops::fused_attention(q, k, v, sc, sl, true));
        }
        h = ops::add(h, a->c_proj->forward(ops::concat_cols(heads)));
        h = ops::add(h, m.mlp[b]->forward(m.ln2[b]->forward(h)));
    }
    return m.head->forward(m.ln_f->forward(h));
}
json model_probe(bool gpu) {
    setup(gpu);
    const std::vector<int> ids = {1, 2, 3, 4, 1, 2, 3, 4}, target = {2, 3, 4, 1, 2, 3, 4, 1};
    parity::ParityLM old(parity::AttnKind::EXACT, 16, 8, 2, 4, 31);
    parity::ParityLM current(parity::AttnKind::EXACT, 16, 8, 2, 4, 31);
    device::step_begin();
    Var a = legacy_forward(old, ids, 4), b = current.forward(ids, 4);
    require_gate(exact_bits(a->data, b->data),
                 "default model logits differ from preserved composition");
    backward(ops::cross_entropy(a, target));
    backward(ops::cross_entropy(b, target));
    auto pa = old.named_parameters(), pb = current.named_parameters();
    for (size_t i = 0; i < pa.size(); ++i)
        require_gate(exact_bits(pa[i].second->grad, pb[i].second->grad),
                     "default model gradient differs from preserved composition");
    device::step_end();
    json default_counts = counts();
    parity::ParityLM model(parity::AttnKind::EXACT, 16, 8, 2, 4, 31, 64, 1, nullptr, 2,
                           device::Precision::BF16);
    json params = json::object();
    for (auto& item : model.named_parameters()) params[item.first] = values(item.second->data);
    device::step_begin();
    Var z = model.forward(ids, 4);
    json logits = values(z->data);
    Var loss = ops::cross_entropy(z, target);
    json initial_loss = values(loss->data);
    backward(loss);
    json grads = json::object();
    for (auto& item : model.named_parameters()) grads[item.first] = values(item.second->grad);
    device::step_end();
    json bf16_counts = counts();
    nn::AdamW optimizer(model.parameters(), .01f);
    json losses = json::array();
    for (int step = 0; step < 16; ++step) {
        device::step_begin();
        optimizer.zero_grad();
        Var l = ops::cross_entropy(model.forward(ids, 4), target);
        device::materialize(l->data);
        require_gate(std::isfinite(l->data(0, 0)), "nonfinite model loss");
        losses.push_back(l->data(0, 0));
        backward(l);
        for (auto& item : model.parameters()) values(item->grad);
        optimizer.step();
        device::step_end();
    }
    json training_counts = counts();
    if (gpu) {
        require_gate(bf16_counts.value("gemm_bf16", 0) == 24,
                     "missing six BF16 GEMMs per layer/head");
        require_gate(training_counts.value("gemm_bf16", 0) == 16 * 24,
                     "missing training BF16 GEMMs");
        require_gate(default_counts.value("gemm_bf16", 0) == 0, "BF16 contaminated default model");
    }
    return {{"pass", true},
            {"actual_cuda", gpu},
            {"default_preserved_logits_gradients_bit_equal", true},
            {"vocab", 16},
            {"d", 8},
            {"heads", 2},
            {"layers", 2},
            {"seq_len", 4},
            {"ids", ids},
            {"targets", target},
            {"parameters", params},
            {"logits", logits},
            {"initial_loss", initial_loss},
            {"gradients", grads},
            {"losses", losses},
            {"default_profile_counts", default_counts},
            {"bf16_profile_counts", bf16_counts},
            {"training_profile_counts", training_counts},
            {"steps", 16}};
}
json inference_probe(const std::string& spec, bool gpu) {
    setup(gpu);
    Spec s = parse_spec(spec);
    require_gate(s.attention_precision == "bf16", "saved BF16 recipe required");
    LoadedLM lm(s);
    require_gate(bool(lm.gpt), "plain GPT2 loaded model required");
    for (auto& a : lm.gpt->exact)
        require_gate(a->attention_precision == device::Precision::FP32,
                     "loaded inference is not FP32");
    counts();
    json result;
    {
        NoGrad ng;
        device::step_begin();
        auto y = lm.forward({1, 2, 3});
        result = values(y->data);
        device::step_end();
    }
    json c = counts();
    if (gpu) {
        require_gate(c.value("gemm_bf16", 0) == 0 && c.value("gemm_head_bf16", 0) == 0,
                     "BF16 inference GEMM");
        require_gate(c.value("gemm", 0) + c.value("gemm_head", 0) > 0,
                     "missing FP32 inference GEMM");
    }
    return {{"pass", true},
            {"actual_cuda", gpu},
            {"loaded_attention_precision", "fp32"},
            {"profile_counts", c},
            {"logits", result}};
}
}  // namespace
int main(int argc, char** argv) {
    if (argc >= 2 &&
        (std::string(argv[1]) == "--gate-model" || std::string(argv[1]) == "--gate-inference")) {
        try {
            bool model = std::string(argv[1]) == "--gate-model";
            require_gate(argc == (model ? 4 : 5), "gate argument count");
            const std::string mode = argv[2];
            require_gate(mode == "cpu" || mode == "cuda", "gate backend");
            json result =
                model ? model_probe(mode == "cuda") : inference_probe(argv[3], mode == "cuda");
            const char* path = argv[model ? 3 : 4];
            std::ofstream f(path);
            require_gate(bool(f), "gate output");
            f << result.dump() << "\n";
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "precision gate: %s\n", e.what());
            return 1;
        }
    }
    return precision_gate_embedded_cli_main(argc, argv);
}
