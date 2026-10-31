#include <vector>
#include <utility>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "check.hpp"
#include "microtorch/ops.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
using namespace microtorch;
static void empty_fallback() {
    // External-storage constructors can represent empty dimensions, but the
    // legacy output Matrix constructor rejects them. Keep that exact behavior
    // rather than inventing a stricter convolution-specific shape rule.
    float dummy=0;
    for(auto shape:std::vector<std::pair<size_t,size_t>>{{0,2},{4,0}}) {
        Matrix xm(shape.first,shape.second,&dummy);
        Matrix wm=shape.second?Matrix(shape.second,3):Matrix(0,3,&dummy);
        Matrix bm=shape.second?Matrix(1,shape.second):Matrix(1,0,&dummy);
        auto x=make_var(std::move(xm)),w=make_var(std::move(wm)),b=make_var(std::move(bm));
        bool threw=false;
        try {ops::conv1d_causal(x,w,b,2);}
        catch(const std::runtime_error& e) {threw=std::string(e.what())=="Matrix dimensions cannot be zero";}
        CHECK(threw);
    }
}
int main(){device::set(device::Device::CPU);device::set_device_ops(false);empty_fallback();std::puts("legacy empty shape rejection pass");}
