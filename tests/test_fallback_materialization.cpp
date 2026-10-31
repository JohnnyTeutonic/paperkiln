// Conditional device-to-host fallback coherence, with independent binary64 references.
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
constexpr size_t R = 2, C = 4;
const std::array<float, 8> input{.2f, -.6f, 1.2f, 2.f, -.7f, .8f, 1.7f, -1.2f};
const std::array<float, 4> weights{.7f, -1.1f, .4f, 1.8f};
const std::array<float, 8> upstream{.2f, -.4f, .7f, 1.3f, -.9f, .6f, .1f, -.3f};
struct Cell { const char* op; const char* phase; const char* operand; bool old_failure; };
const std::vector<Cell> cells{
    {"gelu", "forward", "x", true}, {"gelu", "backward", "x", true},
    {"silu", "forward", "x", false}, {"silu", "backward", "x", true},
    {"softmax", "forward", "x", true}, {"softmax", "backward", "y", true},
    {"sigmoid", "forward", "x", true}, {"sigmoid", "backward", "y", true},
    {"rmsnorm", "forward", "x", true}, {"rmsnorm", "forward", "w", true},
    {"rmsnorm", "backward", "x", true}, {"rmsnorm", "backward", "w", true},
    {"group", "forward", "x", false}, {"group", "forward", "w", true},
    {"group", "backward", "x", false}, {"group", "backward", "w", true}
};
std::string name(const Cell& c) { return std::string(c.op) + "_" + c.phase + "_" + c.operand; }
Matrix matrix(size_t rows, size_t cols, const float* data) {
    Matrix m(rows, cols);
    for (size_t i=0; i<rows; ++i) for (size_t j=0; j<cols; ++j) m(i,j)=data[i*cols+j];
    return m;
}
void poison(const Var& value) {
    CHECK(device::host_stale(value->data));
    // Cache remains device-authoritative. Alter only the stale, non-authoritative host bytes.
    for (size_t i=0; i<value->data.rows(); ++i)
        for (size_t j=0; j<value->data.cols(); ++j) value->data(i,j)=0.f;
    CHECK(device::host_stale(value->data));
}
void reference(const Cell& c, std::array<double,8>& y, std::array<double,8>& dx,
               std::array<double,4>& dw) {
    const std::string op=c.op;
    if (op=="rmsnorm" || op=="group") {
        const size_t g=op=="group" ? 2 : C;
        for (size_t i=0; i<R; ++i) for (size_t o=0; o<C; o+=g) {
            double ss=0, term=0;
            for (size_t j=0; j<g; ++j) ss+=double(input[i*C+o+j])*input[i*C+o+j];
            const double ri=1/std::sqrt(ss/g+double(ops::kRmsNormEps));
            for (size_t j=0; j<g; ++j) {
                const size_t k=i*C+o+j;
                const double dy=double(upstream[k])/(R*C);
                y[k]=input[k]*ri*weights[j];
                dw[j]+=dy*input[k]*ri;
                term+=dy*weights[j]*input[k];
            }
            for (size_t j=0; j<g; ++j) {
                const size_t k=i*C+o+j;
                dx[k]=ri*(double(upstream[k])/(R*C)*weights[j]-input[k]*ri*ri*term/g);
            }
        }
    } else if (op=="softmax") {
        for (size_t i=0; i<R; ++i) {
            double maximum=-INFINITY, sum=0, dot=0;
            for (size_t j=0; j<C; ++j) maximum=std::max(maximum,double(input[i*C+j]));
            for (size_t j=0; j<C; ++j) sum+=std::exp(double(input[i*C+j])-maximum);
            for (size_t j=0; j<C; ++j) {size_t k=i*C+j; y[k]=std::exp(input[k]-maximum)/sum; dot+=double(upstream[k])/(R*C)*y[k];}
            for (size_t j=0; j<C; ++j) {size_t k=i*C+j; dx[k]=y[k]*(double(upstream[k])/(R*C)-dot);}
        }
    } else {
        for (size_t k=0; k<R*C; ++k) {
            const double x=input[k], sig=1/(1+std::exp(-x));
            double derivative=0;
            if (op=="gelu") {
                const double coefficient=std::sqrt(2/std::acos(-1.0));
                const double t=std::tanh(coefficient*(x+.044715*x*x*x));
                y[k]=.5*x*(1+t);
                derivative=.5*(1+t)+.5*x*(1-t*t)*coefficient*(1+3*.044715*x*x);
            } else if (op=="silu") {y[k]=x*sig; derivative=sig*(1+x*(1-sig));}
            else {y[k]=sig; derivative=sig*(1-sig);}
            dx[k]=double(upstream[k])/(R*C)*derivative;
        }
    }
}
void check_values(const Matrix& actual, const double* expected, const std::string& cell,
                  const char* quantity) {
    double maximum=0;
    for (size_t i=0; i<actual.rows(); ++i) for (size_t j=0; j<actual.cols(); ++j) {
        const double a=actual(i,j), e=expected[i*actual.cols()+j];
        const double error=std::fabs(a-e)/(1+std::fabs(e));
        if (!std::isfinite(a) || !std::isfinite(error)) maximum=INFINITY;
        else maximum=std::max(maximum,error);
    }
    std::printf("{\"case\":\"%s\",\"quantity\":\"%s\",\"normalized_max_error\":%.17g,\"numeric_reference_pass\":%s}\n",
                cell.c_str(),quantity,maximum,maximum<2e-5 ? "true":"false");
    std::fflush(stdout);
    CHECK(maximum < 2e-5);
}
void run(const Cell& c, bool gpu) {
    const std::string op=c.op, phase=c.phase, operand=c.operand, cell=name(c);
    const size_t wc=op=="group" ? 2 : C;
    auto x0=make_var(matrix(R,C,input.data()),true), w0=make_var(matrix(1,wc,weights.data()),true);
    Var x=x0,w=w0,stale;
    if (gpu) {
        device::set(device::Device::CUDA); device::set_device_ops(true);
        device::set_step_residency(true); device::set_defer_downloads(true); device::step_begin();
        CHECK(device::deferring());
        if (operand=="x") x=stale=ops::slice_cols(x0,0,C);
        else if (operand=="w") w=stale=ops::slice_cols(w0,0,wc);
        if (stale) poison(stale);
        if (phase=="forward") device::set_device_ops(false);
    }
    Var y;
    if (op=="gelu") y=ops::gelu(x);
    else if (op=="silu") y=ops::silu(x);
    else if (op=="softmax") y=ops::softmax_row(x);
    else if (op=="sigmoid") y=ops::sigmoid(x);
    else if (op=="rmsnorm") y=ops::rmsnorm(x,w);
    else y=ops::rmsnorm_group(x,w,wc,ops::kRmsNormEps);
    auto weighted=ops::mul(y,make_var(matrix(R,C,upstream.data())));
    auto loss=ops::mean(weighted);
    if (gpu && phase=="backward") {
        if (operand=="y") {stale=y; poison(stale);}
        CHECK(device::host_stale(stale->data));
        device::set_device_ops(false);
    }
    if (gpu) {
        CHECK(!device::device_ops_enabled());
        std::printf("{\"case\":\"%s\",\"actual_cuda\":true,\"stale_operand_proved\":true,\"host_poison\":true,\"ops_disabled_before_fallback\":true}\n",cell.c_str());
        std::fflush(stdout);
    }
    // Backward first: materializing y for a forward check would hide saved-output gaps.
    backward(loss);
    device::materialize(y->data); device::materialize(x0->grad);
    if (op=="rmsnorm" || op=="group") device::materialize(w0->grad);
    std::array<double,8> expected{},dx{}; std::array<double,4> dw{};
    reference(c,expected,dx,dw);
    check_values(y->data,expected.data(),cell,"forward");
    check_values(x0->grad,dx.data(),cell,"input_gradient");
    if (op=="rmsnorm" || op=="group") check_values(w0->grad,dw.data(),cell,"weight_gradient");
    if (gpu) {
        device::step_end(); device::set_defer_downloads(false); device::set_step_residency(false);
        device::set_device_ops(false); device::evict_all(); device::set(device::Device::CPU);
    }
    std::printf("{\"pass\":true,\"case\":\"%s\",\"actual_cuda\":%s,\"independent_forward_backward\":true}\n",cell.c_str(),gpu?"true":"false");
}
int main(int argc,char**argv) {
    bool gpu=false; std::string which="all";
    for (int i=1;i<argc;++i) {std::string a=argv[i]; if(a=="--require-cuda")gpu=true; else if(a=="--case" && i+1<argc)which=argv[++i]; else return 2;}
#ifdef MICROTORCH_DEVCHECK
    constexpr bool checks=true;
#else
    constexpr bool checks=false;
#endif
    if (gpu && (!device::cuda_compiled() || !checks)) {std::fprintf(stderr,"requires actual CUDA plus DEVCHECK build\n");return 2;}
    device::set(device::Device::CPU); device::set_device_ops(false);
    device::set_defer_downloads(false); device::set_step_residency(false);
    size_t count=0;
    for(const auto& c:cells) if(which=="all" || which==name(c)) {run(c,gpu);++count;}
    if(!count)return 2;
    std::printf("{\"pass\":true,\"actual_cuda\":%s,\"devcheck_enabled\":%s,\"cases\":%zu}\n",gpu?"true":"false",checks?"true":"false",count);
}
