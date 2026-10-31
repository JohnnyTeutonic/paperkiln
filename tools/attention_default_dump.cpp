// Portable raw-bit fixture for comparing the preserved FP32 library with a candidate.
#include <cstdio>
#include <cstring>
#include "microtorch/device.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
Matrix values(size_t r,size_t c,int seed){Matrix m(r,c);for(size_t i=0;i<r;++i)for(size_t j=0;j<c;++j)m(i,j)=float(int((i*13+j*7+seed)%29)-14)/17;return m;}
void dump(const Matrix&m){for(size_t i=0;i<m.rows();++i)for(size_t j=0;j<m.cols();++j){unsigned u;float f=m(i,j);std::memcpy(&u,&f,4);std::printf("%08x\n",u);}}
int main(){device::set(device::Device::CPU);for(bool causal:{false,true})for(size_t batch:{1,2})for(size_t width:{3,8}){
 auto q=make_var(values(7*batch,width,1),true),k=make_var(values(7*batch,width,2),true),v=make_var(values(7*batch,5,3),true);
 auto y=ops::fused_attention(q,k,v,.5f,7,causal);backward(ops::mean(ops::mul(y,make_var(values(7*batch,5,4)))));
 dump(y->data);dump(q->grad);dump(k->grad);dump(v->grad);
}}
