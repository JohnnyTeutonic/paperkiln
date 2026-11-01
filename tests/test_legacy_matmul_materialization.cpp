// Synthetic CPU authority and actual CUDA are deliberately separate modes.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
namespace {
std::map<const float*,std::vector<float>> pending;
int restores=0;
void need(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
Matrix fixture(size_t r,size_t c,int shift) {
    Matrix m(r,c);
    for(size_t i=0;i<r*c;++i)m.get_data()[i]=float(int((i+shift)%11)-5)*.125f;
    return m;
}
void poison(Matrix& m,bool gpu) {
    if(gpu)need(device::host_stale(m),"CUDA producer is not device-authoritative");
    else pending[m.get_data()]=std::vector<float>(m.get_data(),m.get_data()+m.rows()*m.cols());
    std::fill(m.get_data(),m.get_data()+m.rows()*m.cols(),13.f);
    if(gpu)need(device::host_stale(m),"host poisoning changed CUDA authority");
}
void run_case(int index,bool gpu,bool emulate,const std::string& transition,bool b1) {
    const size_t ms[]={2,2,2,1,5},ks[]={3,3,3,7,7},ns[]={4,4,4,1,3};
    const size_t M=ms[index],K=ks[index],N=ns[index];
    Matrix a=fixture(M,K,1),b=fixture(K,N,4);
    std::vector<double> truth(M*N,0);
    for(size_t i=0;i<M;++i)for(size_t j=0;j<N;++j)for(size_t k=0;k<K;++k)
        truth[i*N+j]+=double(a(i,k))*b(k,j);
    const bool stale_a=index!=2,stale_b=index!=1;
    auto av=make_var(std::move(a)),bv=make_var(std::move(b));
    auto aa=gpu&&stale_a?ops::slice_cols(av,0,K):av;
    auto bb=gpu&&stale_b?ops::slice_cols(bv,0,N):bv;
    if(gpu||emulate){if(stale_a)poison(aa->data,gpu);if(stale_b)poison(bb->data,gpu);}
    if(gpu&&transition=="cpu")device::set(device::Device::CPU);
    if(gpu&&transition=="stepoff")device::set_step_residency(false);
    device::set_residency(b1);
    Matrix result=device::matmul(aa->data,bb->data);
    device::materialize(result);
    need(result.rows()==M&&result.cols()==N,"result shape differs");
    for(size_t i=0;i<M*N;++i)
        need(std::isfinite(result.get_data()[i])&&std::abs(double(result.get_data()[i])-truth[i])<1e-6,"independent numerical reference differs");
    if(emulate)need(restores==int(stale_a)+int(stale_b),"operand materialization count differs");
}
}
#ifdef LEGACY_MATMUL_WRAP
extern "C" void __real__ZN10microtorch6device11materializeERK6Matrix(const Matrix&);
extern "C" void __wrap__ZN10microtorch6device11materializeERK6Matrix(const Matrix& m) {
    auto it=pending.find(m.get_data());
    if(it!=pending.end()){std::copy(it->second.begin(),it->second.end(),const_cast<float*>(m.get_data()));pending.erase(it);++restores;}
    __real__ZN10microtorch6device11materializeERK6Matrix(m);
}
#endif
int main(int argc,char** argv) {
    bool gpu=false,emulate=false,b1=false;std::string transition="active";
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="--require-cuda")gpu=true;
        else if(a=="--synthetic")emulate=true;
        else if(a=="--b1")b1=true;
        else if(a=="--transition"&&i+1<argc)transition=argv[++i];
        else return 2;
    }
    if((gpu&&emulate)||(gpu&&!device::cuda_compiled()))return 2;
#ifndef LEGACY_MATMUL_WRAP
    if(emulate)return 2;
#endif
    if(transition!="active"&&transition!="cpu"&&transition!="stepoff")return 2;
    int failures=0;
    const char* names[]={"both_rectangular","left_only","right_only","dot","odd_rectangular"};
    for(int c=0;c<5;++c){
        pending.clear();restores=0;
        device::set(gpu?device::Device::CUDA:device::Device::CPU);
        device::set_device_ops(gpu);device::set_step_residency(gpu);device::set_defer_downloads(gpu);device::set_residency(false);
        if(gpu)device::step_begin();
        try{
            run_case(c,gpu,emulate,transition,b1);
            std::printf("LEGACY_MATMUL_JSON {\"case\":\"%s\",\"pass\":true,\"actual_cuda\":%s,\"synthetic_authority\":%s}\n",names[c],gpu?"true":"false",emulate?"true":"false");
        }catch(const std::exception& e){
            ++failures;std::printf("LEGACY_MATMUL_JSON {\"case\":\"%s\",\"pass\":false,\"actual_cuda\":%s,\"synthetic_authority\":%s,\"error\":\"%s\"}\n",names[c],gpu?"true":"false",emulate?"true":"false",e.what());
        }
        if(gpu){device::step_end();device::set_defer_downloads(false);device::set_step_residency(false);device::set_device_ops(false);device::set_residency(false);device::evict_all();device::set(device::Device::CPU);}
    }
    return failures?1:0;
}
