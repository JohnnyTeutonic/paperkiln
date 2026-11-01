// CPU authority emulation is explicitly separate from real CUDA validation.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "microtorch/device.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
namespace {
std::map<const float*, std::vector<float>> pending;
int restores = 0;
void need(bool okay, const char* why) { if (!okay) throw std::runtime_error(why); }
Matrix fixture(size_t rows, size_t cols, int shift) {
    Matrix m(rows,cols);
    for(size_t i=0;i<rows;++i) for(size_t j=0;j<cols;++j)
        m(i,j) = float(int((i*cols+j+shift)%9)-4)*.125f;
    return m;
}
std::vector<float> values(const Matrix& m) {
    return std::vector<float>(m.get_data(),m.get_data()+m.rows()*m.cols());
}
void poison(Matrix& m, bool gpu) {
    if(gpu) need(device::host_stale(m), "real producer must be device-authoritative");
    else pending[m.get_data()] = values(m);
    std::fill(m.get_data(),m.get_data()+m.rows()*m.cols(),13.f);
    if(gpu) need(device::host_stale(m),"host poison must not replace device authority");
}
void compare(const Matrix& actual,const std::vector<double>& truth) {
    device::materialize(actual);
    need(actual.rows()*actual.cols()==truth.size(),"output inventory");
    for(size_t i=0;i<truth.size();++i)
        need(std::isfinite(actual.get_data()[i]) && std::abs(double(actual.get_data()[i])-truth[i])<1e-6,"independent numerical reference differs");
}
void start(bool gpu) {
    pending.clear();restores=0;
    device::set(gpu?device::Device::CUDA:device::Device::CPU);
    device::set_device_ops(gpu);device::set_step_residency(gpu);device::set_defer_downloads(gpu);
    device::set_residency(false);  // the actual-L4 stepoff leg isolates Phase-A fallback
    device::set_matmul_precision(device::Precision::FP32);
    if(gpu) {device::step_begin();need(device::deferring(),"deferred window required");}
}
void finish(bool gpu) {
    if(gpu) {device::step_end();device::set_defer_downloads(false);device::set_step_residency(false);device::set_device_ops(false);device::evict_all();device::set(device::Device::CPU);}
    pending.clear();
}
void transition(bool gpu,const std::string& which) {
    if(!gpu) return;
    if(which=="cpu") device::set(device::Device::CPU);
    else if(which=="stepoff") device::set_step_residency(false);
    else need(which=="active","unknown transition");
}
void forward(bool gpu,bool emulate,const std::string& which,bool ta,bool tb) {
    const size_t M=2,K=3,N=2;
    Matrix a=fixture(ta?K:M,ta?M:K,1),b=fixture(tb?N:K,tb?K:N,3);
    auto av=values(a),bv=values(b);std::vector<double> expected(M*N,0);
    for(size_t i=0;i<M;++i) for(size_t j=0;j<N;++j) for(size_t k=0;k<K;++k)
        expected[i*N+j]+=double(av[ta?k*M+i:i*K+k])*bv[tb?j*K+k:k*N+j];
    auto va=make_var(std::move(a)),vb=make_var(std::move(b));
    auto aa=gpu?ops::slice_cols(va,0,va->data.cols()):va;
    auto bb=gpu?ops::slice_cols(vb,0,vb->data.cols()):vb;
    if(gpu||emulate) {poison(aa->data,gpu);poison(bb->data,gpu);}
    transition(gpu,which);
    Matrix got=device::gemm(aa->data,nullptr,ta?device::Trans::T:device::Trans::N,bb->data,nullptr,tb?device::Trans::T:device::Trans::N,device::Precision::FP32);
    compare(got,expected);
    if(emulate) need(restores==2,"both simulated device operands must materialize");
}
void backward_case(bool gpu,bool emulate,const std::string& which) {
    Matrix a=fixture(2,3,1),b=fixture(3,2,3),weights(2,2);
    weights(0,0)=.5f;weights(0,1)=-1.f;weights(1,0)=1.5f;weights(1,1)=2.f;
    auto av=values(a),bv=values(b),uv=values(weights);
    auto va=make_var(std::move(a),true),vb=make_var(std::move(b),true);
    auto aa=gpu?ops::slice_cols(va,0,3):va;auto bb=gpu?ops::slice_cols(vb,0,2):vb;
    auto y=ops::matmul(aa,bb);
    auto loss=ops::mean(ops::mul(y,make_var(std::move(weights))));
    // CPU-only: transfer the SAME mathematical values into a synthetic
    // authority store after the forward. Actual CUDA leaves these saved
    // operands deferred naturally. Only non-authoritative host bytes change.
    if(gpu||emulate) {poison(aa->data,gpu);poison(bb->data,gpu);}
    transition(gpu,which);backward(loss);
    std::vector<double> da(6,0),db(6,0);
    for(size_t i=0;i<2;++i) for(size_t k=0;k<3;++k) for(size_t j=0;j<2;++j)
        da[i*3+k]+=double(uv[i*2+j])*.25*bv[k*2+j];
    for(size_t k=0;k<3;++k) for(size_t j=0;j<2;++j) for(size_t i=0;i<2;++i)
        db[k*2+j]+=double(av[i*3+k])*uv[i*2+j]*.25;
    compare(aa->grad,da);compare(bb->grad,db);
    if(emulate) need(restores==2,"saved operands must materialize in real backward closures");
}
}
#ifdef GEMM_FALLBACK_WRAP
extern "C" void __real__ZN10microtorch6device11materializeERK6Matrix(const Matrix&);
extern "C" void __wrap__ZN10microtorch6device11materializeERK6Matrix(const Matrix& m) {
    auto it=pending.find(m.get_data());
    if(it!=pending.end()) {std::copy(it->second.begin(),it->second.end(),const_cast<float*>(m.get_data()));pending.erase(it);++restores;}
    __real__ZN10microtorch6device11materializeERK6Matrix(m);
}
#endif
int main(int argc,char** argv) {
    bool gpu=false,emulate=false;std::string which="cpu";
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg=="--synthetic") emulate=true;
        else if(arg=="--require-cuda") gpu=true;
        else if(arg=="--transition"&&i+1<argc) which=argv[++i];
        else return 2;
    }
    if(gpu&&emulate) return 2;
#ifndef GEMM_FALLBACK_WRAP
    if(emulate) return 2;
#endif
    if(gpu&&!device::cuda_compiled()) return 2;
    if(which!="cpu"&&which!="stepoff"&&which!="active") return 2;
    int failures=0;
    for(int c=0;c<5;++c) {
        const char* names[]={"NN","NT","TN","TT","saved_parent_backward"};
        try {
            start(gpu);
            if(c<4)forward(gpu,emulate,which,c>=2,c%2);
            else backward_case(gpu,emulate,which);
            std::printf("GEMM_FALLBACK_JSON {\"case\":\"%s\",\"pass\":true,\"actual_cuda\":%s,\"synthetic_authority\":%s,\"restores\":%d}\n",names[c],gpu?"true":"false",emulate?"true":"false",restores);
        } catch(const std::exception& e) {
            ++failures;std::printf("GEMM_FALLBACK_JSON {\"case\":\"%s\",\"pass\":false,\"actual_cuda\":%s,\"synthetic_authority\":%s,\"error\":\"%s\"}\n",names[c],gpu?"true":"false",emulate?"true":"false",e.what());
        }
        finish(gpu);
    }
    return failures?1:0;
}
