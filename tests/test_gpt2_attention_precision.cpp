// Experimental train-only GPT2 integration. CPU semantic gate, not GPU/quality evidence.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>
#include "check.hpp"
#include "microtorch/safetensors.hpp"
#include "../tools/parity_model.hpp"
using namespace microtorch;
using device::Precision;

bool bits(const Matrix& a, const Matrix& b) {
    return a.rows()==b.rows() && a.cols()==b.cols() &&
           std::memcmp(&a(0,0), &b(0,0), a.rows()*a.cols()*sizeof(float))==0;
}
void finite(const Matrix& a) {
    for(size_t i=0;i<a.rows();++i) for(size_t j=0;j<a.cols();++j) CHECK(std::isfinite(a(i,j)));
}
parity::ParityLM make(Precision p) {
    return parity::ParityLM(parity::AttnKind::EXACT,16,8,2,4,31,64,1,nullptr,2,p);
}
int main() {
    gguf::Gpt2ExportConfig legacy{"legacy-name"};
    CHECK(legacy.name=="legacy-name" && legacy.train_attention_precision.empty());
    device::set_matmul_precision(Precision::FP32);
    const std::vector<int> ids={1,2,3,4,1,2,3,4}, labels={2,3,4,1,2,3,4,1};
    parity::ParityLM original(parity::AttnKind::EXACT,16,8,2,4,31);
    auto explicit_fp=make(Precision::FP32), bf=make(Precision::BF16);
    CHECK(bf.exact[0]->attention_precision==Precision::BF16);
    CHECK(device::matmul_precision()==Precision::FP32);
    CHECK(original.export_ir()==explicit_fp.export_ir());
    CHECK(original.export_ir()==bf.export_ir()); // training recipe is not inference IR
    Var a=original.forward(ids,4), b=explicit_fp.forward(ids,4);
    CHECK(bits(a->data,b->data));
    backward(ops::cross_entropy(a,labels));backward(ops::cross_entropy(b,labels));
    auto pa=original.named_parameters(), pb=explicit_fp.named_parameters();
    CHECK(pa.size()==pb.size());
    for(size_t i=0;i<pa.size();++i) {
        CHECK(pa[i].first==pb[i].first);
        CHECK(bits(pa[i].second->data,pb[i].second->data));
        CHECK(bits(pa[i].second->grad,pb[i].second->grad));
    }
    // Per-instance mode cannot contaminate another model or change a live tape.
    auto bf_copy=make(Precision::BF16);
    Var c=bf.forward(ids,4), d=bf_copy.forward(ids,4);
    explicit_fp.forward(ids,4);
    backward(ops::cross_entropy(c,labels));backward(ops::cross_entropy(d,labels));
    auto pc=bf.named_parameters(), pd=bf_copy.named_parameters();
    CHECK(bits(c->data,d->data));
    bool nonzero=false;
    for(size_t i=0;i<pc.size();++i) {
        finite(pc[i].second->grad);CHECK(bits(pc[i].second->grad,pd[i].second->grad));
        for(size_t r=0;r<pc[i].second->grad.rows();++r) for(size_t col=0;col<pc[i].second->grad.cols();++col)
            nonzero |= pc[i].second->grad(r,col)!=0;
    }
    CHECK(nonzero);
    nn::AdamW opt(bf.parameters(),.01f);
    float first=0,last=0;
    for(int step=0;step<16;++step) {
        opt.zero_grad();Var loss=ops::cross_entropy(bf.forward(ids,4),labels);
        const float value=loss->data(0,0);CHECK(std::isfinite(value));
        if(step==0)first=value;last=value;
        backward(loss);for(auto& p:bf.parameters())finite(p->grad);
        opt.step();
    }
    CHECK(last<first);
    const auto path=std::filesystem::temp_directory_path()/"paperkiln_gpt2_precision_test.safetensors";
    save_safetensors(path.string(),bf.state_dict());
    auto inference=make(Precision::FP32);
    inference.load_state_dict(load_safetensors(path.string()));
    CHECK(inference.exact[0]->attention_precision==Precision::FP32);
    auto same=make(Precision::FP32);same.load_state_dict(bf.state_dict());
    CHECK(bits(inference.forward(ids,4)->data,same.forward(ids,4)->data));
    std::filesystem::remove(path);
    bool refused=false;
    try {parity::ParityLM bad(parity::AttnKind::SWA,16,8,2,4,31,64,1,nullptr,2,Precision::BF16);}
    catch(const std::invalid_argument&){refused=true;}CHECK(refused);
    std::printf("{\"pass\":true,\"default_logits_gradients_bit_equal\":true,\"bf16_finite_gradients\":true,"
                "\"fp32_reload_bit_equal\":true,\"steps\":16,\"first_loss\":%.9g,\"last_loss\":%.9g}\n",first,last);
}
