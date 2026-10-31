// CPU oracle and strict, opt-in actual-device integration gate.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <string>
#include <vector>
#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"
#include "microtorch/ops.hpp"
using namespace microtorch;
static Matrix values(size_t r,size_t c,int salt) {
    Matrix m(r,c);
    for(size_t i=0;i<r;++i) for(size_t j=0;j<c;++j)
        m(i,j)=float(int((i*17+j*11+salt)%29)-14)/13.0f;
    return m;
}
static void close(const Matrix& m,const std::vector<double>& ref) {
    CHECK(m.rows()*m.cols()==ref.size());
    for(size_t i=0;i<ref.size();++i) {
        const double v=m(i/m.cols(),i%m.cols());
        CHECK(std::isfinite(v));
        CHECK(std::fabs(v-ref[i])<2e-5*(1+std::fabs(ref[i])));
    }
}
struct Ref { std::vector<double> y,dx,dw,db; };
// Dense linear map A[(t,c),(u,d)], built using sequence/channel membership.
// This formulation is deliberately independent of the implementation's tap loops.
static Ref dense(const Matrix& x,const Matrix& w,const Matrix* b,const Matrix& dy,size_t seq) {
    const size_t T=x.rows(),C=x.cols(),K=w.cols(),sl=seq?seq:T,N=T*C;
    Ref z{std::vector<double>(N),std::vector<double>(N),std::vector<double>(C*K),std::vector<double>(C)};
    for(size_t o=0;o<N;++o) {
        size_t t=o/C,c=o%C;
        z.y[o]=b?(*b)(0,c):0;
        z.db[c]+=dy(t,c);
        for(size_t i=0;i<N;++i) {
            size_t u=i/C,d=i%C;
            if(c!=d || t/sl!=u/sl || u>t || t-u>=K) continue;
            size_t k=K-1-(t-u);
            z.y[o]+=double(x(u,d))*w(c,k);
            z.dx[i]+=double(dy(t,c))*w(c,k);
            z.dw[c*K+k]+=double(dy(t,c))*x(u,d);
        }
    }
    return z;
}
static float madd(float a,float b,float c) { volatile float p=b*c; volatile float z=a+p; return z; }
static Ref rounded(const Matrix& x,const Matrix& w,const Matrix* b,const Matrix& dy,size_t seq) {
    const size_t T=x.rows(),C=x.cols(),K=w.cols(),sl=seq?seq:T;
    Ref z{std::vector<double>(T*C),std::vector<double>(T*C),std::vector<double>(C*K),std::vector<double>(C)};
    for(size_t t=0;t<T;++t)for(size_t c=0;c<C;++c) {
        float a=0;
        for(size_t u=t/sl*sl;u<=t;++u)if(t-u<K)a=madd(a,w(c,K-1-t+u),x(u,c));
        z.y[t*C+c]=b?float(a+(*b)(0,c)):a;
        a=0;
        for(size_t v=(t/sl+1)*sl;v-->t;)if(v-t<K)a=madd(a,dy(v,c),w(c,K-1-v+t));
        z.dx[t*C+c]=a;
    }
    for(size_t c=0;c<C;++c) {
        float a=0;for(size_t t=0;t<T;++t)a=float(a+dy(t,c));z.db[c]=a;
        for(size_t k=0;k<K;++k) {
            a=0;for(size_t t=0;t<T;++t)if(t%sl>=K-1-k)a=madd(a,dy(t,c),x(t-(K-1-k),c));
            z.dw[c*K+k]=a;
        }
    }
    return z;
}
static void exact(const Matrix& m,const std::vector<double>& ref) {
    for(size_t i=0;i<ref.size();++i) {
        float expected=float(ref[i]),actual=m(i/m.cols(),i%m.cols());
        CHECK(std::memcmp(&actual,&expected,sizeof(float))==0);
    }
}
static double objective(const Matrix& x,const Matrix& w,const Matrix& b,const Matrix& dy,size_t sl) {
    auto y=ops::conv1d_causal(make_var(x),make_var(w),make_var(b),sl);double v=0;
    device::materialize(y->data);
    for(size_t i=0;i<dy.rows();++i)for(size_t j=0;j<dy.cols();++j)v+=double(y->data(i,j))*dy(i,j);
    return v;
}
static void finite_difference() {
    Matrix x=values(6,2,1),w=values(2,5,2),b=values(1,2,3),dy=values(6,2,4);
    const auto z=dense(x,w,&b,dy,3); const float h=1.0f/256;
    for(int which=0;which<3;++which) {
        Matrix* m=which==0?&x:which==1?&w:&b;
        const auto& ref=which==0?z.dx:which==1?z.dw:z.db;
        for(size_t i=0;i<ref.size();++i) {
            float& v=(*m)(i/m->cols(),i%m->cols());float old=v;
            v=old+h;double plus=objective(x,w,b,dy,3);
            v=old-h;double minus=objective(x,w,b,dy,3);v=old;
            CHECK(std::fabs((plus-minus)/(2*h)-ref[i])<2e-3);
        }
    }
}
static void run_case(bool gpu,size_t T,size_t C,size_t K,size_t sl,bool bias,int mask,bool fallback) {
    Matrix xm=values(T,C,1),wm=values(C,K,2),bm=values(1,C,3),dm=values(T,C,4);
    const auto ref=dense(xm,wm,bias?&bm:nullptr,dm,sl);
    const auto bits=rounded(xm,wm,bias?&bm:nullptr,dm,sl);
    Var x=make_var(std::move(xm),mask&1),w=make_var(std::move(wm),mask&2);
    Var b=bias?make_var(std::move(bm),mask&4):nullptr;
    // Device-produced identities leave authoritative values in cache. Poison
    // only stale host backing buffers to prove that the CUDA body reads cache.
    auto computed=[&](Var v) {
        if(!gpu)return v;
        Var q=ops::scale(v,1.0f);
        CHECK(device::host_stale(q->data));
        for(size_t i=0;i<q->data.rows();++i)for(size_t j=0;j<q->data.cols();++j)q->data(i,j)=-7777;
        return q;
    };
    Var cx=computed(x),cw=computed(w),cb=b?computed(b):nullptr;
    Var y=ops::conv1d_causal(cx,cw,cb,sl);
    if(gpu) CHECK(device::host_stale(y->data));
    device::materialize(y->data);close(y->data,ref.y);exact(y->data,bits.y);
    if(mask) {
        Var d=make_var(std::move(dm));
        if(gpu)d=computed(d);
        // Invoke the recorded local backward with an arbitrary vector seed;
        // this isolates convolution's VJP from unrelated loss reductions.
        y->grad=std::move(d->data);
        if(fallback)device::set_device_ops(false);
        y->backward_fn();
        size_t gradient_index=0;
        for(auto entry:std::vector<std::pair<Var,const std::vector<double>*>>{{cx,&ref.dx},{cw,&ref.dw},{cb,&ref.db}}) {
            if(!entry.first)continue;
            if(entry.first->requires_grad) {
                if(gpu&&!fallback)CHECK(device::host_stale(entry.first->grad));
                device::materialize(entry.first->grad);close(entry.first->grad,*entry.second);
                exact(entry.first->grad,gradient_index==0?bits.dx:gradient_index==1?bits.dw:bits.db);
            } else CHECK(entry.first->grad.rows()*entry.first->grad.cols()==0);
            ++gradient_index;
        }
        if(fallback)device::set_device_ops(true);
    }
}
static void nonfinite_padding(bool gpu) {
    Matrix x(3,1),w(1,4);for(size_t i=0;i<3;++i)x(i,0)=1;
    w(0,0)=std::numeric_limits<float>::infinity();
    auto y=ops::conv1d_causal(make_var(std::move(x)),make_var(std::move(w)),nullptr);
    if(gpu)CHECK(device::host_stale(y->data));
    device::materialize(y->data);
    for(size_t i=0;i<3;++i)CHECK(std::isnan(y->data(i,0)));
}
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
        Matrix sentinel(1,1);sentinel(0,0)=123;
        CHECK(!device::devops::conv1d_causal_fwd(x->data,w->data,&b->data,2,sentinel));
        CHECK(!device::devops::conv1d_causal_bwd(x->data,w->data,x->data,2,&sentinel,nullptr,nullptr));
        CHECK(sentinel(0,0)==123);
        bool threw=false;
        try {ops::conv1d_causal(x,w,b,2);}
        catch(const std::runtime_error& e) {threw=std::string(e.what())=="Matrix dimensions cannot be zero";}
        CHECK(threw);
    }
}
static void invalid() {
    auto bad=[](size_t T,size_t C,size_t W,size_t K,size_t B,size_t sl) {
        bool threw=false;try {ops::conv1d_causal(make_var(Matrix(T,C)),make_var(Matrix(W,K)),make_var(Matrix(1,B)),sl);}
        catch(const std::runtime_error&){threw=true;} CHECK(threw);
    };
    bad(0,2,2,3,2,0);bad(4,2,3,3,2,0);
    bad(4,2,2,1,2,0);bad(4,2,2,3,3,0);bad(4,2,2,3,2,3);bad(4,2,2,3,2,8);
    Matrix x(4,2),w(2,3),out(4,2);out(0,0)=123;
    CHECK(!device::devops::conv1d_causal_fwd(x,w,nullptr,3,out));CHECK(out(0,0)==123);
    Matrix dw(2,3);dw(0,0)=456;
    CHECK(!device::devops::conv1d_causal_bwd(x,w,x,3,nullptr,&dw,nullptr));CHECK(dw(0,0)==456);
}
int main(int argc,char** argv) {
    const bool gpu=argc==2&&std::string(argv[1])=="--require-cuda";
    if(argc>2||(argc==2&&!gpu))return 2;
#ifdef MICROTORCH_DEVCHECK
    constexpr bool devcheck=true;
#else
    constexpr bool devcheck=false;
    if(gpu){std::fprintf(stderr,"requires MICROTORCH_DEVCHECK=ON\n");return 2;}
#endif
    if(gpu&&!device::cuda_compiled()){std::fprintf(stderr,"requires real CUDA\n");return 2;}
    device::set(device::Device::CPU);device::set_device_ops(false);
    finite_difference();invalid();nonfinite_padding(false);empty_fallback();
    if(gpu) {
        device::set(device::Device::CUDA);device::set_device_ops(true);
        device::set_step_residency(true);device::set_defer_downloads(true);
    }
    size_t cases=0;
    for(int repeat=0;repeat<2;++repeat)for(auto shape:std::vector<std::vector<size_t>>{{6,3,2,3},{6,5,9,2},{3,1,4,1},{7,3,4,0},{66,5,4,11}})
        for(bool b:{false,true})for(int mask=0;mask<(b?8:4);++mask) {
            if(gpu){device::step_begin();CHECK(device::deferring());}
            run_case(gpu,shape[0],shape[1],shape[2],shape[3],b,mask,false);
            if(gpu)device::step_end();++cases;
        }
    if(gpu) {
        device::step_begin();invalid();nonfinite_padding(true);empty_fallback();run_case(true,6,3,7,2,true,7,true);device::step_end();

    }
    // Exercise the public autograd traversal as well as the isolated VJP.
    auto x=make_var(values(6,3,1),true),w=make_var(values(3,5,2),true),b=make_var(values(1,3,3),true);
    Matrix dy(6,3);for(size_t i=0;i<6;++i)for(size_t j=0;j<3;++j)dy(i,j)=1.0f/18;
    const auto ref=dense(x->data,w->data,&b->data,dy,2);
    if(gpu)device::step_begin();
    auto y=ops::conv1d_causal(x,w,b,2);
    if(gpu)CHECK(device::host_stale(y->data));
    backward(ops::mean(y));
    if(gpu) {
        CHECK(device::host_stale(x->grad));CHECK(device::host_stale(w->grad));CHECK(device::host_stale(b->grad));
    }
    device::materialize(x->grad);device::materialize(w->grad);device::materialize(b->grad);
    close(x->grad,ref.dx);close(w->grad,ref.dw);close(b->grad,ref.db);
    if(gpu) {
        device::step_end();device::set_defer_downloads(false);device::set_step_residency(false);
        // The same seam must also work synchronously, outside a step window.
        Matrix xx=values(6,3,1),ww=values(3,9,2),bb=values(1,3,3),dd=values(6,3,4);
        auto rz=dense(xx,ww,&bb,dd,2);Matrix yy(6,3),dx(6,3),dw(3,9),db(1,3);
        CHECK(device::devops::conv1d_causal_fwd(xx,ww,&bb,2,yy));CHECK(!device::host_stale(yy));close(yy,rz.y);
        CHECK(device::devops::conv1d_causal_bwd(xx,ww,dd,2,&dx,&dw,&db));
        CHECK(!device::host_stale(dx));CHECK(!device::host_stale(dw));CHECK(!device::host_stale(db));
        close(dx,rz.dx);close(dw,rz.dw);close(db,rz.db);
        device::set_device_ops(false);device::evict_all();device::set(device::Device::CPU);
    }
    std::printf("{\"pass\":true,\"actual_cuda\":%s,\"devcheck_enabled\":%s,\"cases\":%zu,"
                "\"dense_binary64\":true,\"finite_difference\":true,\"f32_order_bitwise\":true,\"all_gradient_masks\":true,"
                "\"poisoned_stale_inputs\":%s,\"deferred_outputs\":%s,\"deferred_gradients\":%s,"
                "\"device_to_host_backward_fallback\":%s,\"public_autograd_traversal\":true,\"write_through\":%s}\n",
                gpu?"true":"false",devcheck?"true":"false",cases,
                gpu?"true":"false",gpu?"true":"false",gpu?"true":"false",gpu?"true":"false",gpu?"true":"false");
}
