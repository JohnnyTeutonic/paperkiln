// Same fixture source, linked separately to preserved and candidate CPU libraries.
#define main embedded_selective_test_main
#include "../tests/test_selective_scan_cuda.cpp"
#undef main
#include <cstdint>
#include <cstring>
void bits(const Matrix& m) {
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j) {
            float x = m(i, j);
            uint32_t u;
            std::memcpy(&u, &x, 4);
            std::printf("%08x\n", u);
        }
}
int main() {
    device::set(device::Device::CPU);
    device::set_device_ops(false);
    for (bool sp : {false, true})
        for (bool skip : {false, true})
            for (auto shape :
                 {std::array<size_t, 4>{1, 1, 1, 1}, {3, 2, 3, 2}, {9, 2, 5, 3}, {4, 2, 2, 5}}) {
                Problem p = fixture(shape[0], shape[1], shape[2], shape[3], sp, skip);
                size_t T = p.sl * p.batch, D = p.d, N = p.n;
                size_t rs[] = {T, T, T, T, D, 1}, cs[] = {D, D, N, N, N, D};
                std::array<Var, 6> a;
                for (size_t v = 0; v < 6; ++v) a[v] = make_var(matrix(rs[v], cs[v], p.a[v]), true);
                auto y = ops::selective_scan(a[0], a[1], a[2], a[3], a[4], skip ? a[5] : Var{}, sp,
                                             p.sl);
                backward(ops::mean(ops::mul(y, make_var(matrix(T, D, upstream(p))))));
                bits(y->data);
                for (size_t v = 0; v < (skip ? 6 : 5); ++v) bits(a[v]->grad);
            }
}
