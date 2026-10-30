// Experimental dense MHA primitive/probe. Production training dispatch is untouched.
// Build: nvcc -O3 -std=c++17 -arch=sm_89 -lcublas tools/bf16_attention_probe.cu -o PROBE
// Contract/oracle: tools/bf16_attention_reference.py. BF16 round casts use a
// straight-through surrogate backward; this is not a derivative of rounding.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void cb(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(e));
}
template <class T>
struct Buffer {
    T* p = nullptr;
    size_t n;
    explicit Buffer(size_t count) : n(count) {
        ck(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
    }
    ~Buffer() {
        if (p) cudaFree(p);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const float* host) {
        ck(cudaMemcpy(p, host, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<float> download() const {
        static_assert(sizeof(T) == sizeof(float), "download floats only");
        std::vector<float> out(n);
        ck(cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
};
struct Handle {
    cublasHandle_t h;
    Handle() { cb(cublasCreate(&h)); }
    ~Handle() { cublasDestroy(h); }
};
struct Event {
    cudaEvent_t e;
    Event() { ck(cudaEventCreate(&e)); }
    ~Event() { cudaEventDestroy(e); }
};
__global__ void round_bf16(const float* src, __nv_bfloat16* dst, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x)
        dst[i] = __float2bfloat16_rn(src[i]);
}
void pack(const Buffer<float>& src, Buffer<__nv_bfloat16>& dst) {
    round_bf16<<<std::min<size_t>((src.n + 255) / 256, 65535), 256>>>(src.p, dst.p, src.n);
    ck(cudaGetLastError());
}
__global__ void softmax(const float* score, float* prob, int seq, bool causal) {
    const int row = blockIdx.x, t = threadIdx.x, i = row % seq;
    __shared__ float reduction[256];
    float mx = -CUDART_INF_F;
    for (int j = t; j < seq; j += 256)
        if (!causal || j <= i) mx = fmaxf(mx, score[size_t(row) * seq + j]);
    reduction[t] = mx;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride) reduction[t] = fmaxf(reduction[t], reduction[t + stride]);
        __syncthreads();
    }
    mx = reduction[0];
    float sum = 0;
    for (int j = t; j < seq; j += 256) {
        const float e = (!causal || j <= i) ? expf(score[size_t(row) * seq + j] - mx) : 0.f;
        prob[size_t(row) * seq + j] = e;
        sum += e;
    }
    reduction[t] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride) reduction[t] += reduction[t + stride];
        __syncthreads();
    }
    for (int j = t; j < seq; j += 256) prob[size_t(row) * seq + j] /= reduction[0];
}
__global__ void softback(const float* p, const float* dp, float* ds, int seq) {
    const int row = blockIdx.x, t = threadIdx.x;
    __shared__ float reduction[256];
    float dot = 0;
    for (int j = t; j < seq; j += 256) dot += p[size_t(row) * seq + j] * dp[size_t(row) * seq + j];
    reduction[t] = dot;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride) reduction[t] += reduction[t + stride];
        __syncthreads();
    }
    for (int j = t; j < seq; j += 256) {
        size_t k = size_t(row) * seq + j;
        ds[k] = p[k] * (dp[k] - reduction[0]);
    }
}

// Owns reusable scratch; no allocation inside forward/backward or timed steps.
class Attention {
    int batches_, s_, d_;
    bool causal_, low_;
    float scale_;
    Handle handle_;
    Buffer<float> q_, k_, v_, dy_, score_, p_, dp_, ds_, y_, dq_, dk_, dv_;
    Buffer<__nv_bfloat16> q16_, k16_, v16_, dy16_, p16_, ds16_;
    const void* operand(const Buffer<float>& f, const Buffer<__nv_bfloat16>& h) const {
        return low_ ? static_cast<const void*>(h.p) : static_cast<const void*>(f.p);
    }
    void gemm(bool ta, bool tb, int m, int n, int k, const void* a, const void* b, float* c,
              float alpha = 1.f) {
        const float beta = 0;
        const cudaDataType_t type = low_ ? CUDA_R_16BF : CUDA_R_32F;
        cb(cublasGemmStridedBatchedEx(
            handle_.h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k,
            &alpha, b, type, tb ? k : n, static_cast<long long>(k) * n, a, type, ta ? m : k,
            static_cast<long long>(m) * k, &beta, c, CUDA_R_32F, n, static_cast<long long>(m) * n,
            batches_, low_ ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_PEDANTIC,
            low_ ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
    }

public:
    Attention(int b, int h, int s, int d, bool causal, bool low, const std::vector<float>& in)
        : batches_(b * h),
          s_(s),
          d_(d),
          causal_(causal),
          low_(low),
          scale_(1.f / std::sqrt(float(d))),
          q_(size_t(b) * h * s * d),
          k_(q_.n),
          v_(q_.n),
          dy_(q_.n),
          score_(size_t(b) * h * s * s),
          p_(score_.n),
          dp_(score_.n),
          ds_(score_.n),
          y_(q_.n),
          dq_(q_.n),
          dk_(q_.n),
          dv_(q_.n),
          q16_(q_.n),
          k16_(q_.n),
          v16_(q_.n),
          dy16_(q_.n),
          p16_(p_.n),
          ds16_(ds_.n) {
        q_.upload(in.data());
        k_.upload(in.data() + q_.n);
        v_.upload(in.data() + 2 * q_.n);
        dy_.upload(in.data() + 3 * q_.n);
    }
    void forward() {
        if (low_) {
            pack(q_, q16_);
            pack(k_, k16_);
            pack(v_, v16_);
        }
        gemm(false, true, s_, s_, d_, operand(q_, q16_), operand(k_, k16_), score_.p, scale_);
        softmax<<<batches_ * s_, 256>>>(score_.p, p_.p, s_, causal_);
        ck(cudaGetLastError());
        if (low_) pack(p_, p16_);
        gemm(false, false, s_, d_, s_, operand(p_, p16_), operand(v_, v16_), y_.p);
    }
    void backward() {
        if (low_) pack(dy_, dy16_);
        gemm(false, true, s_, s_, d_, operand(dy_, dy16_), operand(v_, v16_), dp_.p);
        softback<<<batches_ * s_, 256>>>(p_.p, dp_.p, ds_.p, s_);
        ck(cudaGetLastError());
        if (low_) pack(ds_, ds16_);
        gemm(false, false, s_, d_, s_, operand(ds_, ds16_), operand(k_, k16_), dq_.p, scale_);
        gemm(true, false, s_, d_, s_, operand(ds_, ds16_), operand(q_, q16_), dk_.p, scale_);
        gemm(true, false, s_, d_, s_, operand(p_, p16_), operand(dy_, dy16_), dv_.p);
    }
    std::array<std::vector<float>, 8> results() const {
        return {score_.download(), p_.download(),  dp_.download(), ds_.download(),
                y_.download(),     dq_.download(), dk_.download(), dv_.download()};
    }
};
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: bf16_attention_probe INPUT.bin OUTPUT.bin");
        static_assert(sizeof(float) == 4 && sizeof(uint32_t) == 4, "binary sizes");
        std::ifstream existing(argv[2], std::ios::binary);
        if (existing.good()) throw std::runtime_error("output exists");
        existing.close();
        std::ifstream f(argv[1], std::ios::binary);
        std::array<uint32_t, 8> header{};
        if (!f.read(reinterpret_cast<char*>(header.data()), 32))
            throw std::runtime_error("short header");
        const uint32_t b = header[2], h = header[3], s = header[4], d = header[5];
        if (header[0] != 0x54414642 || header[1] != 1 || !b || !h || !s || !d ||
            uint64_t(b) * h > 256 || s > 2048 || d > 256 || d % 8 ||
            uint64_t(b) * h * s * s > 16777216 || header[6] > 1 || header[7] > 1)
            throw std::runtime_error("unsupported shape/header");
        const size_t n = size_t(b) * h * s * d;
        std::vector<float> input(4 * n);
        if (!f.read(reinterpret_cast<char*>(input.data()), std::streamsize(input.size() * 4)))
            throw std::runtime_error("short input");
        if (f.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("trailing input bytes");
        for (float x : input)
            if (!std::isfinite(x)) throw std::runtime_error("finite inputs required");
        int device = 0;
        ck(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, device));
        if (header[7] && prop.major < 8) throw std::runtime_error("BF16 requires SM80 or newer");
        std::ostringstream uuid;
        for (unsigned char x : prop.uuid.bytes)
            uuid << std::hex << std::setw(2) << std::setfill('0') << unsigned(x);
        Attention op(b, h, s, d, header[6], header[7], input);
        for (int i = 0; i < 5; i++) {
            op.forward();
            op.backward();
        }
        ck(cudaDeviceSynchronize());
        Event start, end;
        ck(cudaEventRecord(start.e));
        for (int i = 0; i < 20; i++) {
            op.forward();
            op.backward();
        }
        ck(cudaEventRecord(end.e));
        ck(cudaEventSynchronize(end.e));
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, start.e, end.e));
        auto output = op.results();
        for (const auto& a : output)
            for (float x : a)
                if (!std::isfinite(x)) throw std::runtime_error("nonfinite output");
        const std::string tmp = std::string(argv[2]) + ".tmp";
        std::ifstream oldtmp(tmp);
        if (oldtmp.good()) throw std::runtime_error("temporary output exists");
        oldtmp.close();
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<char*>(header.data()), 32);
        for (const auto& a : output)
            out.write(reinterpret_cast<const char*>(a.data()), std::streamsize(a.size() * 4));
        out.close();
        if (!out) throw std::runtime_error("output write failed");
        std::filesystem::create_hard_link(tmp, argv[2]);
        std::filesystem::remove(tmp);
        std::cout << "{\"device\":" << std::quoted(prop.name)
                  << ",\"device_uuid\":" << std::quoted(uuid.str())
                  << ",\"sm\":" << prop.major * 10 + prop.minor
                  << ",\"iterations\":20,\"forward_backward_ms\":" << std::setprecision(9)
                  << ms / 20 << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}