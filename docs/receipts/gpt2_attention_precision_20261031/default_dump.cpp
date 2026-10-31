#include <cstdio>
#include <cstdint>
#include <cstring>
#include "parity_model.hpp"
using namespace microtorch;
void dump(const Matrix& m) { for(size_t i=0;i<m.rows();++i) for(size_t j=0;j<m.cols();++j) {uint32_t u;float f=m(i,j);std::memcpy(&u,&f,4);std::printf("%08x\n",u);} }
int main(){ device::set_matmul_precision(device::Precision::FP32);parity::ParityLM m(parity::AttnKind::EXACT,16,8,2,4,31);std::vector<int> x={1,2,3,4,1,2,3,4},y={2,3,4,1,2,3,4,1};auto z=m.forward(x,4);dump(z->data);backward(ops::cross_entropy(z,y));for(auto& p:m.named_parameters()){std::printf("%s\n",p.first.c_str());dump(p.second->data);dump(p.second->grad);} }
