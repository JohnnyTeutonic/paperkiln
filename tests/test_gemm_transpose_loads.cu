// Standalone harness: candidate header is extracted from the production source.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
constexpr int TILE = 32;
#include "fixtures/gemm_transpose_old.cuh"
#include "gemm_candidate.cuh"
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
float* alloc(const std::vector<float>& v) {
    float* p;
    ck(cudaMalloc(&p, (v.empty() ? 1 : v.size()) * 4));
    if (!v.empty()) ck(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    return p;
}
void dump(const char* n, const std::vector<float>& v) {
    printf("%s", n);
    for (float x : v) printf(" %a", double(x));
    puts("");
}
template <bool A, bool B>
void launch(bool old, bool acc, float* a, float* b, float* c, int m, int n, int k) {
    dim3 block(32, 32), grid((n + 31) / 32, (m + 31) / 32);
    if (old) {
        if (acc)
            old_gemm_tiled_acc<A, B><<<grid, block>>>(a, b, c, m, n, k);
        else
            old_gemm_tiled_ops<A, B><<<grid, block>>>(a, b, c, m, n, k);
    } else {
        if (acc)
            gemm_tiled_acc<A, B><<<grid, block>>>(a, b, c, m, n, k);
        else
            gemm_tiled_ops<A, B><<<grid, block>>>(a, b, c, m, n, k);
    }
    ck(cudaGetLastError());
}
void run(int flags, int m, int n, int k, int mode) {
    bool ta = flags & 2, tb = flags & 1;
    std::vector<float> a(m * k), b(k * n), init(m * n), x(m * n), y(m * n);
    for (size_t i = 0; i < a.size(); i++) a[i] = float(int((i * 17 + 3) % 41) - 20) / 31.0f;
    for (size_t i = 0; i < b.size(); i++) b[i] = float(int((i * 13 + 7) % 43) - 21) / 29.0f;
    for (size_t i = 0; i < init.size(); i++) init[i] = float(int(i % 11) - 5) / 17.0f;
    float *da = alloc(a), *db = alloc(b), *dx = alloc(init), *dy = alloc(init);
    auto go = [&](bool old, float* dc, float* aa, float* bb, int kk, bool acc) {
        switch (flags) {
            case 0:
                launch<false, false>(old, acc, aa, bb, dc, m, n, kk);
                break;
            case 1:
                launch<false, true>(old, acc, aa, bb, dc, m, n, kk);
                break;
            case 2:
                launch<true, false>(old, acc, aa, bb, dc, m, n, kk);
                break;
            case 3:
                launch<true, true>(old, acc, aa, bb, dc, m, n, kk);
                break;
        }
    };
    // Split-K inputs are repacked so each kernel has its documented contiguous layout.
    for (int old = 0; old < 2; old++) {
        float* dc = old ? dy : dx;
        if (mode < 2)
            go(old, dc, da, db, k, mode == 1);
        else {
            int cut = k / 2;
            for (int part = 0; part < 2; part++) {
                int off = part ? cut : 0, kk = part ? k - cut : cut;
                std::vector<float> ap(m * kk), bp(kk * n);
                for (int i = 0; i < m; i++)
                    for (int q = 0; q < kk; q++)
                        ap[ta ? q * m + i : i * kk + q] =
                            a[ta ? (q + off) * m + i : i * k + q + off];
                for (int q = 0; q < kk; q++)
                    for (int j = 0; j < n; j++)
                        bp[tb ? j * kk + q : q * n + j] =
                            b[tb ? j * k + q + off : (q + off) * n + j];
                float *pa = alloc(ap), *pb = alloc(bp);
                go(old, dc, pa, pb, kk, true);
                ck(cudaDeviceSynchronize());
                ck(cudaFree(pa));
                ck(cudaFree(pb));
            }
        }
    }
    ck(cudaDeviceSynchronize());
    ck(cudaMemcpy(x.data(), dx, x.size() * 4, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    if (memcmp(x.data(), y.data(), x.size() * 4))
        throw std::runtime_error("old/candidate bit mismatch");
    double err = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double ref = mode ? init[i * n + j] : 0;
            for (int q = 0; q < k; q++)
                ref += double(a[ta ? q * m + i : i * k + q]) * b[tb ? j * k + q : q * n + j];
            err = fmax(err, fabs(x[i * n + j] - ref) / (1 + fabs(ref)));
        }
    if (!std::isfinite(err) || err > 3e-5) throw std::runtime_error("double mismatch");
    printf("CASE %d %d %d %d %d %.17g\n", flags, m, n, k, mode, err);
    dump("A", a);
    dump("B", b);
    dump("I", init);
    dump("C", x);
    dump("OLD", y);
    ck(cudaFree(da));
    ck(cudaFree(db));
    ck(cudaFree(dx));
    ck(cudaFree(dy));
}
template <bool A, bool B>
void timing() {
    const int n = 256;
    std::vector<float> v(n * n, 0.125f);
    float *a = alloc(v), *b = alloc(v), *c = alloc(v);
    cudaEvent_t start, end;
    ck(cudaEventCreate(&start));
    ck(cudaEventCreate(&end));
    for (int pair = 0; pair < 7; pair++)
        for (int leg = 0; leg < 2; leg++) {
            bool old = (pair % 2) ? !leg : leg;
            for (int w = 0; w < 3; w++) launch<A, B>(old, false, a, b, c, n, n, n);
            ck(cudaEventRecord(start));
            for (int rep = 0; rep < 10; rep++) launch<A, B>(old, false, a, b, c, n, n, n);
            ck(cudaEventRecord(end));
            ck(cudaEventSynchronize(end));
            float ms;
            ck(cudaEventElapsedTime(&ms, start, end));
            printf("TIMING %d %d %d %.9g\n", int(A) * 2 + int(B), pair, int(old), ms / 10);
        }
    ck(cudaEventDestroy(start));
    ck(cudaEventDestroy(end));
    ck(cudaFree(a));
    ck(cudaFree(b));
    ck(cudaFree(c));
}
int main(int argc, char** argv) {
    try {
#ifndef MICROTORCH_DEVCHECK
        throw std::runtime_error("DEVCHECK build required");
#endif
        const char* required = getenv("MICROTORCH_DEVICE");
        if (!required || strcmp(required, "cuda"))
            throw std::runtime_error("explicit CUDA required");
        int count = 0;
        ck(cudaGetDeviceCount(&count));
        if (!count) throw std::runtime_error("CUDA required");
        cudaDeviceProp p;
        ck(cudaGetDeviceProperties(&p, 0));
        printf("DEVICE %s\n", p.name);
        if (argc == 2 && !strcmp(argv[1], "--timing")) {
            timing<false, false>();
            timing<false, true>();
            timing<true, false>();
            timing<true, true>();
            puts("TIMING_PASS 56");
            return 0;
        }
        const int shapes[][3] = {{1, 1, 0},    {1, 1, 1},    {2, 3, 7},    {31, 33, 35},
                                 {32, 32, 32}, {33, 31, 65}, {64, 96, 64}, {65, 67, 63}};
        for (int f = 0; f < 4; f++)
            for (auto& s : shapes)
                for (int mode = 0; mode < 3; mode++) run(f, s[0], s[1], s[2], mode);
        puts("PASS 96");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
