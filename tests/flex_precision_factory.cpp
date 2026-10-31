// Test-only access to the exact CLI training and saved-inference constructors.
#define main flex_precision_embedded_cli_main
#include "../tools/mtstudio.cpp"
#undef main
namespace {
void require_factory(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
json inspect(const Spec& spec, bool saved) {
    require_factory(spec.family == "flex", "Flex recipe required");
    auto training = make_flex(spec, 7, true);
    std::shared_ptr<parity::FlexLM> inference;
    if (saved) {
        LoadedLM loaded(spec);
        inference = loaded.flex;
    } else
        inference = make_flex(spec, 7, false);
    require_factory(bool(inference), "actual saved Flex model required");
    const auto wanted =
        spec.attention_precision == "bf16" ? device::Precision::BF16 : device::Precision::FP32;
    for (auto& b : training->blocks)
        require_factory(b->attn->attention_precision == wanted,
                        "training factory ignored precision");
    for (auto& b : inference->blocks)
        require_factory(b->attn->attention_precision == device::Precision::FP32,
                        "saved inference inherited BF16");
    require_factory(training->export_ir() == inference->export_ir(),
                    "training precision changed inference IR");
    require_factory(device::matmul_precision() == device::Precision::FP32,
                    "global projection precision changed");
    return {{"pass", true},
            {"actual_saved_model_loaded", saved},
            {"training_attention_precision", spec.attention_precision},
            {"saved_attention_precision", "fp32"},
            {"layers", spec.layers},
            {"global_projection_precision", "fp32"},
            {"inference_IR_equal", true}};
}
}  // namespace
int main(int argc, char** argv) {
    try {
        device::set_matmul_precision(device::Precision::FP32);
        if (argc == 3 && std::string(argv[1]) == "--spec") {
            auto spec = parse_spec(argv[2]);
            std::puts(inspect(spec, true).dump().c_str());
            return 0;
        }
        require_factory(argc == 1, "factory probe takes --spec path or no arguments");
        for (auto depth : {size_t(1), size_t(3)})
            for (const std::string mode : {"fp32", "bf16"}) {
                Spec s;
                s.family = "flex";
                s.d = 8;
                s.heads = 2;
                s.T = 4;
                s.layers = depth;
                s.d_ff = 12;
                s.norm = "layernorm";
                s.attention_precision = mode;
                inspect(s, false);
            }
        std::puts("{\"pass\":true,\"four_constructor_modes\":true,\"no_inference_BF16\":true}");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
