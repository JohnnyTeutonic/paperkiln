#include "microtorch/gguf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "gguf_export.hpp"  // transformer_cpp's verified writer
#include "microtorch/ir.hpp"

namespace microtorch {
namespace gguf {

namespace {

using gguf_export::GGMLType;
using gguf_export::GGUFWriter;
using gguf_export::TensorExportInfo;

// Look up `key` allowing an optional "model." prefix on the dict side.
const Matrix* find(const std::map<std::string, Matrix>& sd, const std::string& key) {
    auto it = sd.find(key);
    if (it != sd.end()) return &it->second;
    it = sd.find("model." + key);
    if (it != sd.end()) return &it->second;
    return nullptr;
}

// float32 -> IEEE fp16 bits, round to nearest even (numpy's astype(float16),
// which the reference quantiser uses for the block scale), with subnormals,
// overflow to inf and NaN handled.
uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) return static_cast<uint16_t>(sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0u));
    if (ax >= 0x477ff000u) return static_cast<uint16_t>(sign | 0x7c00u);  // >= 65520 rounds to inf
    if (ax < 0x38800000u) {  // below 2^-14: fp16 subnormal or zero
        if (ax <= 0x33000000u) return static_cast<uint16_t>(sign);  // <= 2^-25 ties to even 0
        const uint32_t e = ax >> 23, m = (ax & 0x7fffffu) | 0x800000u;
        const uint32_t shift = 126 - e;  // value in units of 2^-24 is m >> shift
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;  // may carry into the smallest normal
        return static_cast<uint16_t>(sign | h);
    }
    const uint32_t m = ax & 0x7fffffu;
    uint32_t h = (((ax >> 23) - 112) << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;  // a carry steps the exponent, as it should
    return static_cast<uint16_t>(sign | h);
}

}  // namespace

void quantize_q8_0(const float* x, size_t n, uint8_t* out) {
    if (n % 32 != 0) throw std::runtime_error("quantize_q8_0: n must be a multiple of 32");
    for (size_t b = 0; b < n / 32; ++b, x += 32, out += 34) {
        // gguf-py quants.Q8_0.quantize_blocks, operation for operation in
        // float32: d = amax / 127, id = 1 / d (0 when d is 0), q = roundf(x * id)
        // where its np_roundf rounds half away from zero, as std::round does.
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            if (!std::isfinite(x[i])) throw std::runtime_error("quantize_q8_0: non-finite input");
            amax = std::max(amax, std::fabs(x[i]));
        }
        const float d = amax / 127.0f;
        const float id = d == 0.0f ? 0.0f : 1.0f / d;
        const uint16_t dh = fp32_to_fp16(d);
        std::memcpy(out, &dh, 2);  // little-endian, as GGUF stores it
        for (int i = 0; i < 32; ++i) {
            const auto q = static_cast<int8_t>(std::round(x[i] * id));
            std::memcpy(out + 2 + i, &q, 1);
        }
    }
}

// The 4- and 5-bit blocks follow ggml-quants.c's quantize_row_*_ref, the
// path llama.cpp's quantiser takes without an importance matrix. Every
// product is rounded to float before the following add, as there: GCC
// would otherwise be free to fuse x * id + 8.5f into one FMA on a target
// that has it, and the truncation below would then differ in the last place.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

namespace {

// The first value of largest magnitude, with its sign (ggml: amax < |v|).
float signed_absmax(const float* x, const char* who) {
    float amax = 0.0f, max = 0.0f;
    for (int i = 0; i < 32; ++i) {
        if (!std::isfinite(x[i])) throw std::runtime_error(std::string(who) + ": non-finite input");
        if (amax < std::fabs(x[i])) {
            amax = std::fabs(x[i]);
            max = x[i];
        }
    }
    return max;
}

// Smallest and largest value of a block (ggml: min starts at FLT_MAX).
void min_max(const float* x, const char* who, float& lo, float& hi) {
    lo = std::numeric_limits<float>::max();
    hi = -std::numeric_limits<float>::max();
    for (int i = 0; i < 32; ++i) {
        if (!std::isfinite(x[i])) throw std::runtime_error(std::string(who) + ": non-finite input");
        lo = std::min(lo, x[i]);
        hi = std::max(hi, x[i]);
    }
}

// ggml's (int8_t)(v) cast: truncation toward zero; v stays well inside int8.
int trunc_int(float v) { return static_cast<int>(v); }

void put_fp16(uint8_t* out, float v) {
    const uint16_t h = fp32_to_fp16(v);
    std::memcpy(out, &h, 2);  // little-endian, as GGUF stores it
}

}  // namespace

void quantize_q4_0(const float* x, size_t n, uint8_t* out) {
    if (n % 32 != 0) throw std::runtime_error("quantize_q4_0: n must be a multiple of 32");
    for (size_t b = 0; b < n / 32; ++b, x += 32, out += 18) {
        const float d = signed_absmax(x, "quantize_q4_0") / -8;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        put_fp16(out, d);
        for (int j = 0; j < 16; ++j) {
            const float x0 = x[j] * id;
            const float x1 = x[16 + j] * id;
            const int q0 = std::min(15, trunc_int(x0 + 8.5f));
            const int q1 = std::min(15, trunc_int(x1 + 8.5f));
            out[2 + j] = static_cast<uint8_t>(q0 | (q1 << 4));
        }
    }
}

void quantize_q5_0(const float* x, size_t n, uint8_t* out) {
    if (n % 32 != 0) throw std::runtime_error("quantize_q5_0: n must be a multiple of 32");
    for (size_t b = 0; b < n / 32; ++b, x += 32, out += 22) {
        const float d = signed_absmax(x, "quantize_q5_0") / -16;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        put_fp16(out, d);
        uint32_t qh = 0;
        for (int j = 0; j < 16; ++j) {
            const float x0 = x[j] * id;
            const float x1 = x[16 + j] * id;
            const int q0 = std::min(31, trunc_int(x0 + 16.5f));
            const int q1 = std::min(31, trunc_int(x1 + 16.5f));
            out[6 + j] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
            qh |= static_cast<uint32_t>((q0 & 0x10) >> 4) << j;
            qh |= static_cast<uint32_t>((q1 & 0x10) >> 4) << (j + 16);
        }
        std::memcpy(out + 2, &qh, 4);
    }
}

void quantize_q5_1(const float* x, size_t n, uint8_t* out) {
    if (n % 32 != 0) throw std::runtime_error("quantize_q5_1: n must be a multiple of 32");
    for (size_t b = 0; b < n / 32; ++b, x += 32, out += 24) {
        float lo, hi;
        min_max(x, "quantize_q5_1", lo, hi);
        const float d = (hi - lo) / 31;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        put_fp16(out, d);
        put_fp16(out + 2, lo);
        uint32_t qh = 0;
        for (int j = 0; j < 16; ++j) {
            const float x0 = (x[j] - lo) * id;
            const float x1 = (x[16 + j] - lo) * id;
            const int q0 = std::min(31, trunc_int(x0 + 0.5f));
            const int q1 = std::min(31, trunc_int(x1 + 0.5f));
            out[8 + j] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
            qh |= static_cast<uint32_t>((q0 & 0x10) >> 4) << j;
            qh |= static_cast<uint32_t>((q1 & 0x10) >> 4) << (j + 16);
        }
        std::memcpy(out + 4, &qh, 4);
    }
}

// ---- k-quants: 256-value super-blocks (ggml-quants.c, llama.cpp b6000 = 4762ad73) ----
// quantize_row_q4_K_ref / _q5_K_ref / _q6_K_ref and their helpers nearest_int,
// make_qkx2_quants and make_qx_quants, statement for statement: the code
// llama.cpp's quantiser runs for these types without an importance matrix
// (quantize_q4_K & co. call the _ref rows when quant_weights is null). ggml
// builds ggml-quants.c in ggml-base without -march flags, so its float
// operations are these, unfused; the pragma above keeps them unfused here.
namespace {

constexpr int kQK_K = 256;

// ggml's nearest_int: round half to even through the float mantissa.
int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t x;
    if (e == 0) {
        if (m == 0) {
            x = sign;
        } else {  // subnormal: normalise
            int k = -1;
            uint32_t mm = m;
            do {
                ++k;
                mm <<= 1;
            } while ((mm & 0x400u) == 0);
            x = sign | ((112u - static_cast<uint32_t>(k)) << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31) {
        x = sign | 0x7f800000u | (m << 13);
    } else {
        x = sign | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

float make_qkx2_quants(int n, int nmax, const float* x, const float* weights, uint8_t* L, float* the_min,
                       uint8_t* Laux, float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * (x[i] - min));
        L[i] = static_cast<uint8_t>(std::max(0, std::min(nmax, l)));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? std::fabs(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * (x[i] - min));
            l = std::max(0, std::min(nmax, l));
            Laux[i] = static_cast<uint8_t>(l);
            float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? std::fabs(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

// make_qx_quants with rmse_type 1 and no importance weights (the only call q6_K makes).
float make_qx_quants_rmse1(int n, int nmax, const float* x, int8_t* L) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = std::fabs(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < 1e-15f) {  // GROUP_MAX_EPS: all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax - 1, l));
        L[i] = static_cast<int8_t>(l + nmax);
        float w = x[i] * x[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax - 1, l));
            float w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = static_cast<int8_t>(nmax + std::max(-nmax, std::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void check_k(const float* x, size_t n, const char* who) {
    if (n % kQK_K != 0) throw std::runtime_error(std::string(who) + ": n must be a multiple of 256");
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(x[i])) throw std::runtime_error(std::string(who) + ": non-finite input");
}

// The shared part of q4_K and q5_K: 8 sub-blocks of 32 with a 6-bit scale and
// min each, packed into 12 bytes, the fp16 super-scales d and dmin, and the
// final levels L (0 .. nmax) against the quantised scales.
void kx_superblock(const float* x, int nmax, float rmin, int nstep, uint8_t* dm4, uint8_t* scales12, uint8_t* L) {
    uint8_t Laux[32];
    float weights[32];
    float mins[kQK_K / 32];
    float scales[kQK_K / 32];
    float max_scale = 0;  // as we are deducting the min, scales are always positive
    float max_min = 0;
    for (int j = 0; j < kQK_K / 32; ++j) {
        float sum_x2 = 0;
        for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
        float av_x = std::sqrt(sum_x2 / 32);
        for (int l = 0; l < 32; ++l) weights[l] = av_x + std::fabs(x[32 * j + l]);
        scales[j] = make_qkx2_quants(32, nmax, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, rmin, 0.1f, nstep,
                                     false);
        float scale = scales[j];
        if (scale > max_scale) max_scale = scale;
        float min = mins[j];
        if (min > max_min) max_min = min;
    }
    float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
    float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
    for (int j = 0; j < kQK_K / 32; ++j) {
        uint8_t ls = static_cast<uint8_t>(nearest_int(inv_scale * scales[j]));
        uint8_t lm = static_cast<uint8_t>(nearest_int(inv_min * mins[j]));
        ls = std::min<uint8_t>(63, ls);
        lm = std::min<uint8_t>(63, lm);
        if (j < 4) {
            scales12[j] = ls;
            scales12[j + 4] = lm;
        } else {
            scales12[j + 4] = static_cast<uint8_t>((ls & 0xF) | ((lm & 0xF) << 4));
            scales12[j - 4] |= static_cast<uint8_t>((ls >> 4) << 6);
            scales12[j - 0] |= static_cast<uint8_t>((lm >> 4) << 6);
        }
    }
    const uint16_t d16 = fp32_to_fp16(max_scale / 63.f), dmin16 = fp32_to_fp16(max_min / 63.f);
    std::memcpy(dm4, &d16, 2);
    std::memcpy(dm4 + 2, &dmin16, 2);
    for (int j = 0; j < kQK_K / 32; ++j) {
        uint8_t sc, m;
        get_scale_min_k4(j, scales12, &sc, &m);
        const float d = fp16_to_fp32(d16) * sc;
        if (!d) continue;
        const float dm = fp16_to_fp32(dmin16) * m;
        for (int ii = 0; ii < 32; ++ii) {
            int l = nearest_int((x[32 * j + ii] + dm) / d);
            L[32 * j + ii] = static_cast<uint8_t>(std::max(0, std::min(nmax, l)));
        }
    }
}

}  // namespace

void quantize_q4_k(const float* x, size_t n, uint8_t* out) {
    check_k(x, n, "quantize_q4_k");
    uint8_t L[kQK_K];
    for (size_t b = 0; b < n / kQK_K; ++b, x += kQK_K, out += 144) {
        // block_q4_K: d, dmin (fp16), scales[12], qs[128]
        kx_superblock(x, 15, -1.f, 20, out, out + 4, L);
        uint8_t* q = out + 16;
        for (int j = 0; j < kQK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = static_cast<uint8_t>(L[j + l] | (L[j + l + 32] << 4));
            q += 32;
        }
    }
}

void quantize_q5_k(const float* x, size_t n, uint8_t* out) {
    check_k(x, n, "quantize_q5_k");
    uint8_t L[kQK_K];
    for (size_t b = 0; b < n / kQK_K; ++b, x += kQK_K, out += 176) {
        // block_q5_K: d, dmin (fp16), scales[12], qh[32], qs[128]
        kx_superblock(x, 31, -0.5f, 15, out, out + 4, L);
        uint8_t* qh = out + 16;
        uint8_t* ql = out + 48;
        std::memset(qh, 0, kQK_K / 8);
        uint8_t m1 = 1, m2 = 2;
        for (int nn = 0; nn < kQK_K; nn += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[nn + j];
                if (l1 > 15) {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                int l2 = L[nn + j + 32];
                if (l2 > 15) {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                ql[j] = static_cast<uint8_t>(l1 | (l2 << 4));
            }
            m1 = static_cast<uint8_t>(m1 << 2);
            m2 = static_cast<uint8_t>(m2 << 2);
            ql += 32;
        }
    }
}

void quantize_q6_k(const float* x, size_t n, uint8_t* out) {
    check_k(x, n, "quantize_q6_k");
    int8_t L[kQK_K];
    float scales[kQK_K / 16];
    for (size_t b = 0; b < n / kQK_K; ++b, x += kQK_K, out += 210) {
        // block_q6_K: ql[128], qh[64], scales[16] (int8), d (fp16)
        uint8_t* ql = out;
        uint8_t* qh = out + 128;
        int8_t* sc = reinterpret_cast<int8_t*>(out + 192);
        float max_scale = 0;
        float max_abs_scale = 0;
        for (int ib = 0; ib < kQK_K / 16; ++ib) {
            const float scale = make_qx_quants_rmse1(16, 32, x + 16 * ib, L + 16 * ib);
            scales[ib] = scale;
            const float abs_scale = std::fabs(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }
        if (max_abs_scale < 1e-15f) {  // GROUP_MAX_EPS
            std::memset(out, 0, 210);
            continue;  // d = fp16(0) = 0x0000
        }
        float iscale = -128.f / max_scale;
        const uint16_t d16 = fp32_to_fp16(1 / iscale);
        std::memcpy(out + 208, &d16, 2);
        for (int ib = 0; ib < kQK_K / 16; ++ib)
            sc[ib] = static_cast<int8_t>(std::min(127, nearest_int(iscale * scales[ib])));
        for (int j = 0; j < kQK_K / 16; ++j) {
            float d = fp16_to_fp32(d16) * sc[j];
            if (!d) continue;
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16 * j + ii] / d);
                l = std::max(-32, std::min(31, l));
                L[16 * j + ii] = static_cast<int8_t>(l + 32);
            }
        }
        for (int j = 0; j < kQK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l + 0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l + 0] = static_cast<uint8_t>(q1 | (q3 << 4));
                ql[l + 32] = static_cast<uint8_t>(q2 | (q4 << 4));
                qh[l] = static_cast<uint8_t>((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) |
                                             ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6));
            }
            ql += 64;
            qh += 32;
        }
    }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

namespace {

// ggml type ids of the k-quants llama.cpp's mixes ask for (quantize_q4_k & co.).
constexpr uint32_t kQ4_K = 12, kQ5_K = 13, kQ6_K = 14;

// llama.cpp's use_more_bits (llama-quant.cpp): the first and last eighth of
// the layers, and every third layer between.
bool use_more_bits(int i, int n) { return i < n / 8 || i >= 7 * n / 8 || (i - n / 8) % 3 == 2; }

// llama.cpp's tensor order in a quantised file: its weight_name_comparer
// (llama-model-loader) puts non-layer tensors first, then layers by number,
// by name within each.
bool llamacpp_before(const std::string& a, const std::string& b) {
    int la = -1, lb = -1;
    std::sscanf(a.c_str(), "blk.%d.", &la);
    std::sscanf(b.c_str(), "blk.%d.", &lb);
    if (la != lb) return la < lb;
    return a < b;
}

// llama.cpp's llama_tensor_get_type for a dense llama model without an
// importance matrix, for the ftypes the exporter offers, followed by its
// fallback for k-quants whose 256-value super-block does not divide ne0.
// The counters advance as llama.cpp's do (in file order).
uint32_t llamacpp_type(const std::string& quant, const std::string& name, uint64_t ne0, bool has_output,
                       int& i_attn_v, int n_attn_v, int& i_ffn_down, int n_ffn_down) {
    uint32_t t = quant == "q4_0" ? static_cast<uint32_t>(GGMLType::Q4_0) : kQ4_K;
    if (name == "output.weight" || (!has_output && name == "token_embd.weight")) {
        t = kQ6_K;
    } else if (name == "token_embd.weight") {
        // the ftype's own type
    } else if (name.find("attn_v.weight") != std::string::npos) {
        if (quant == "q4_k_m" && use_more_bits(i_attn_v, n_attn_v))
            t = kQ6_K;
        else if (quant == "q4_k_s" && i_attn_v < 4)
            t = kQ5_K;
        ++i_attn_v;
    } else if (name.find("ffn_down") != std::string::npos) {
        if (quant == "q4_k_m" && use_more_bits(i_ffn_down, n_ffn_down))
            t = kQ6_K;
        else if (quant == "q4_k_s" && i_ffn_down < n_ffn_down / 8)
            t = kQ5_K;
        ++i_ffn_down;
    }
    if (ne0 % 256 != 0) {
        if (t == kQ4_K) t = static_cast<uint32_t>(GGMLType::Q5_0);
        if (t == kQ5_K) t = static_cast<uint32_t>(GGMLType::Q5_1);
        if (t == kQ6_K) t = static_cast<uint32_t>(GGMLType::Q8_0);
    }
    return t;
}

// The tokenizer block, shared by the llama and gpt2 exports (an empty token
// list writes none).
template <class Cfg>
void write_tokenizer(GGUFWriter& writer, const Cfg& cfg) {
    if (cfg.tokens.empty()) return;
    writer.write_metadata_string("tokenizer.ggml.model", cfg.tokenizer_model);
    if (!cfg.tokenizer_pre.empty()) writer.write_metadata_string("tokenizer.ggml.pre", cfg.tokenizer_pre);
    writer.write_metadata_string_array("tokenizer.ggml.tokens", cfg.tokens);
    if (!cfg.token_type.empty()) writer.write_metadata_int32_array("tokenizer.ggml.token_type", cfg.token_type);
    if (!cfg.merges.empty()) writer.write_metadata_string_array("tokenizer.ggml.merges", cfg.merges);
    if (!cfg.tokenizer_pre.empty()) {
        // llama.cpp turns add_bos on for pre "llama-bpe" unless these keys say
        // otherwise, and paperkiln models have no BOS.
        writer.write_metadata_bool("tokenizer.ggml.add_bos_token", false);
        writer.write_metadata_bool("tokenizer.ggml.add_eos_token", false);
    }
    writer.write_metadata_float_array("tokenizer.ggml.scores", std::vector<float>(cfg.tokens.size(), 0.0f));
    writer.write_metadata_uint32("tokenizer.ggml.bos_token_id", cfg.bos_token_id);
    writer.write_metadata_uint32("tokenizer.ggml.eos_token_id", cfg.eos_token_id);
    writer.write_metadata_uint32("tokenizer.ggml.unknown_token_id", cfg.unk_token_id);
    writer.write_metadata_uint32("tokenizer.ggml.padding_token_id", cfg.pad_token_id);
    if (!cfg.chat_template.empty()) writer.write_metadata_string("paperkiln.chat_template", cfg.chat_template);
    if (!cfg.chat_template_jinja.empty())
        writer.write_metadata_string("tokenizer.chat_template", cfg.chat_template_jinja);
}

// The IR keys, written last so every key before them sits where it did
// without the IR (an empty `ir` writes none): the document, its version and
// its digest (IR_SPEC.md section 3.10; a document with no digest, which no
// exporter writes, gets no digest key).
void write_ir(GGUFWriter& writer, const std::string& ir, const char* who) {
    if (ir.empty()) return;
    const auto doc = ir::json::parse(ir);
    if (!doc.contains("ir_version") || !doc["ir_version"].is_number_unsigned())
        throw std::runtime_error(std::string(who) + ": the IR has no unsigned ir_version");
    const std::string dump = doc.dump();
    writer.write_metadata_string(ir::kGGUFKey, dump);
    writer.write_metadata_uint32(ir::kGGUFVersionKey, doc["ir_version"].get<uint32_t>());
    std::string sha;
    try {
        sha = ir::digest(ir::json::parse(dump));  // the embedded text, as a reader parses it
    } catch (const ir::DigestError&) {
    }
    if (!sha.empty()) writer.write_metadata_string(ir::kGGUFDigestKey, sha);
}

}  // namespace

void export_gguf_llama(const std::string& path, const std::map<std::string, Matrix>& state_dict,
                       const LlamaExportConfig& cfg, std::vector<std::string>* unmapped) {
    if (cfg.embedding_length == 0 || cfg.block_count == 0 || cfg.feed_forward_length == 0 ||
        cfg.head_count == 0 || cfg.vocab_size == 0) {
        throw std::runtime_error(
            "export_gguf_llama: embedding_length, block_count, feed_forward_length, "
            "head_count and vocab_size must all be set");
    }
    const uint32_t kv_heads = cfg.head_count_kv ? cfg.head_count_kv : cfg.head_count;
    const uint32_t head_dim = cfg.embedding_length / cfg.head_count;
    const bool q8 = cfg.quant.rfind("q8_0", 0) == 0;
    // llama.cpp's own mixes: its tensor order, types and two extra keys.
    const bool lcpp = cfg.quant == "q4_0" || cfg.quant == "q4_k_m" || cfg.quant == "q4_k_s";
    if (cfg.quant != "f32" && cfg.quant != "q8_0" && cfg.quant != "q8_0_out" && cfg.quant != "q8_0_all" && !lcpp)
        throw std::runtime_error(
            "export_gguf_llama: quant must be f32, q8_0, q8_0_out, q8_0_all, q4_0, q4_k_m or q4_k_s, got " +
            cfg.quant);
    if (lcpp && cfg.state_tokens_k > 0)
        throw std::runtime_error("export_gguf_llama: " + cfg.quant +
                                 " follows llama.cpp's type choice, which has no state embedding; state-token models "
                                 "export f32 and q8_0*");
    if (lcpp && cfg.expert_count > 0)
        throw std::runtime_error("export_gguf_llama: " + cfg.quant +
                                 " follows llama.cpp's dense-model type choice; MoE models are not supported");

    GGUFWriter writer(path);

    // ---- metadata: the exact llama.* block tinyllama.cpp reads ----
    writer.write_metadata_string("general.architecture", cfg.architecture);
    writer.write_metadata_string("general.name", cfg.name);
    writer.write_metadata_uint32("general.alignment", gguf_export::GGUF_DEFAULT_ALIGNMENT);
    const std::string& a = cfg.architecture;
    writer.write_metadata_uint32(a + ".context_length", cfg.context_length);
    writer.write_metadata_uint32(a + ".embedding_length", cfg.embedding_length);
    writer.write_metadata_uint32(a + ".block_count", cfg.block_count);
    writer.write_metadata_uint32(a + ".feed_forward_length", cfg.feed_forward_length);
    writer.write_metadata_uint32(a + ".attention.head_count", cfg.head_count);
    writer.write_metadata_uint32(a + ".attention.head_count_kv", kv_heads);
    writer.write_metadata_float32(a + ".attention.layer_norm_rms_epsilon", cfg.rms_eps);
    writer.write_metadata_uint32(a + ".vocab_size", cfg.vocab_size);
    writer.write_metadata_float32(a + ".rope.freq_base", cfg.rope_freq_base);
    // Partial rotary writes the rotated dims (llama.cpp's n_rot); 0 = the head width.
    writer.write_metadata_uint32(a + ".rope.dimension_count",
                                 cfg.rope_dimension_count ? cfg.rope_dimension_count : head_dim);
    if (cfg.expert_count > 0) {
        if (cfg.expert_used_count == 0 || cfg.expert_used_count > cfg.expert_count)
            throw std::runtime_error("export_gguf_llama: expert_used_count must be in [1, expert_count]");
        writer.write_metadata_uint32(a + ".expert_count", cfg.expert_count);
        writer.write_metadata_uint32(a + ".expert_used_count", cfg.expert_used_count);
    }
    if (cfg.state_tokens_k > 0) writer.write_metadata_uint32("paperkiln.state_tokens.k", cfg.state_tokens_k);
    // Training provenance (mixed precision, 24 Oct 2026): written only when set, and never into the
    // 4-bit variants, which are llama.cpp's quantiser output byte for byte.
    if (!cfg.train_precision.empty() && !lcpp)
        writer.write_metadata_string("paperkiln.train.precision", cfg.train_precision);

    write_tokenizer(writer, cfg);
    if (lcpp) {
        // llama.cpp's quantiser copies every source key (the IR keys of the F32
        // file included) and then appends these two (GGML_QNT_VERSION, and the
        // llama_ftype: MOSTLY_Q4_0 = 2, MOSTLY_Q4_K_S = 14, MOSTLY_Q4_K_M = 15),
        // so here they follow the IR keys.
        write_ir(writer, cfg.ir, "export_gguf_llama");
        writer.write_metadata_uint32("general.quantization_version", 2);
        writer.write_metadata_uint32("general.file_type",
                                     cfg.quant == "q4_0" ? 2u : cfg.quant == "q4_k_s" ? 14u : 15u);
    } else {
        write_ir(writer, cfg.ir, "export_gguf_llama");
    }

    // ---- tensors ----
    // Keep-alive for transposed copies: GGUFWriter buffers raw pointers
    // until finalize(), so every temporary must outlive it (the same
    // `owned` idiom as transformer_cpp's exporter).
    std::vector<std::unique_ptr<std::vector<float>>> owned;
    std::vector<std::string> mapped_keys;  // dict keys we consumed

    // Every tensor is collected here as F32 and handed to the writer at the
    // end, quantised as LlamaExportConfig::quant says (see `emit` below).
    std::vector<TensorExportInfo> pending;
    auto add = [&](TensorExportInfo info) { pending.push_back(std::move(info)); };

    // NOTE: GGUFWriter takes `shape` in row-major (numpy) order and writes it
    // REVERSED, so ne0 (the contiguous axis) is the LAST entry passed. An
    // [out][in] buffer is passed as {out, in} and lands as ne = {in, out}, the
    // order llama.cpp checks. (Exports before 19 Oct 2026 passed {in, out} and
    // recorded the dims reversed; the bytes were right and ember, which reads
    // by element count, loaded them unchanged.)
    //
    // 2-D projection in llama layout [out][in] row-major, ne = {in, out}.
    auto add_proj = [&](const Matrix& w, const std::string& gguf_name) {
        TensorExportInfo info;
        info.type = GGMLType::F32;
        info.name = gguf_name;
        if (cfg.weights_in_out) {
            // microtorch layout [in][out] -> transpose to [out][in]
            const size_t R = w.rows(), C = w.cols();  // R=in, C=out
            auto buf = std::make_unique<std::vector<float>>(R * C);
            for (size_t i = 0; i < R; ++i)
                for (size_t j = 0; j < C; ++j) (*buf)[j * R + i] = w(i, j);
            info.shape = {static_cast<uint64_t>(C), static_cast<uint64_t>(R)};  // {out,in} -> ne={in,out}
            info.data = buf->data();
            info.num_elements = R * C;
            owned.push_back(std::move(buf));
        } else {
            // HF layout rows=[out], cols=[in] is already llama's byte order.
            info.shape = {static_cast<uint64_t>(w.rows()),
                          static_cast<uint64_t>(w.cols())};  // {out,in} -> ne={in,out}
            info.data = &w(0, 0);
            info.num_elements = w.rows() * w.cols();
        }
        add(info);
    };

    // RMSNorm gamma stored [1, d] -> 1-D tensor {d}.
    auto add_norm = [&](const Matrix& g, const std::string& gguf_name) {
        TensorExportInfo info;
        info.name = gguf_name;
        info.shape = {static_cast<uint64_t>(g.rows() * g.cols())};
        info.type = GGMLType::F32;
        info.data = &g(0, 0);
        info.num_elements = g.rows() * g.cols();
        add(info);
    };

    auto require = [&](const std::string& key) -> const Matrix& {
        const Matrix* m = find(state_dict, key);
        if (!m) throw std::runtime_error("export_gguf_llama: missing tensor " + key);
        mapped_keys.push_back(key);
        return *m;
    };

    // MoE tensors in llama.cpp's dimension order, by the same convention:
    // passing {E, out, in} for a buffer laid out [E][out][in] gives the file
    // ne = {in, out, E}, which is what llama.cpp's create_tensor checks.
    //
    // Stack E projections as [E][out][in] row-major (transposed per matrix
    // under weights_in_out). Returns the buffer; in/out via the references.
    auto pack = [&](const std::vector<const Matrix*>& ws, const std::string& gguf_name, size_t& in, size_t& out) {
        in = cfg.weights_in_out ? ws[0]->rows() : ws[0]->cols();
        out = cfg.weights_in_out ? ws[0]->cols() : ws[0]->rows();
        auto buf = std::make_unique<std::vector<float>>(ws.size() * in * out);
        for (size_t e = 0; e < ws.size(); ++e) {
            const Matrix& w = *ws[e];
            if (w.rows() != ws[0]->rows() || w.cols() != ws[0]->cols())
                throw std::runtime_error("export_gguf_llama: expert shapes differ in " + gguf_name);
            float* dst = buf->data() + e * in * out;
            for (size_t o = 0; o < out; ++o)
                for (size_t i = 0; i < in; ++i) dst[o * in + i] = cfg.weights_in_out ? w(i, o) : w(o, i);
        }
        return buf;
    };
    auto add_packed = [&](std::unique_ptr<std::vector<float>> buf, std::vector<uint64_t> shape,
                          const std::string& gguf_name) {
        TensorExportInfo info;
        info.type = GGMLType::F32;
        info.name = gguf_name;
        info.shape = std::move(shape);
        info.data = buf->data();
        info.num_elements = buf->size();
        owned.push_back(std::move(buf));
        add(info);
    };
    // Router: microtorch [d, E] (or HF [E, d]) -> [E][d] bytes, ne = {d, E}.
    auto add_router = [&](const std::string& hf_prefix, const std::string& gguf_name) {
        size_t in = 0, out = 0;
        auto buf = pack({&require(hf_prefix + "mlp.router.weight")}, gguf_name, in, out);
        if (out != cfg.expert_count) throw std::runtime_error("export_gguf_llama: router width != expert_count");
        add_packed(std::move(buf), {static_cast<uint64_t>(out), static_cast<uint64_t>(in)}, gguf_name);
    };
    // Experts: E projections -> one 3-D tensor, ne = {in, out, E}; expert e's
    // [out][in] block at element offset e * in * out (llama.cpp *_exps).
    auto add_experts = [&](const std::string& hf_prefix, const std::string& proj, const std::string& gguf_name) {
        std::vector<const Matrix*> ws;
        for (uint32_t e = 0; e < cfg.expert_count; ++e)
            ws.push_back(&require(hf_prefix + "mlp.experts." + std::to_string(e) + "." + proj));
        size_t in = 0, out = 0;
        auto buf = pack(ws, gguf_name, in, out);
        add_packed(std::move(buf),
                   {static_cast<uint64_t>(ws.size()), static_cast<uint64_t>(out), static_cast<uint64_t>(in)},
                   gguf_name);
    };

    // Embeddings: [vocab, hidden] row-major == llama layout, ne={hidden,vocab}.
    {
        const Matrix& emb = require("embed_tokens.weight");
        TensorExportInfo info;
        info.name = "token_embd.weight";
        info.shape = {static_cast<uint64_t>(emb.rows()), static_cast<uint64_t>(emb.cols())};  // {vocab,hidden} -> ne={hidden,vocab}
        info.type = GGMLType::F32;
        info.data = &emb(0, 0);
        info.num_elements = emb.rows() * emb.cols();
        add(info);
    }
    if (cfg.state_tokens_k > 0) {
        // The state rows, laid out as token_embd: [k, d] row-major, ne = {d, k}.
        const Matrix& st = require("state_embd.weight");
        if (st.rows() != cfg.state_tokens_k || st.cols() != cfg.embedding_length)
            throw std::runtime_error("export_gguf_llama: state_embd.weight is not [state_tokens_k, d]");
        TensorExportInfo info;
        info.name = "state_embd.weight";
        info.shape = {static_cast<uint64_t>(st.rows()), static_cast<uint64_t>(st.cols())};
        info.type = GGMLType::F32;
        info.data = &st(0, 0);
        info.num_elements = st.rows() * st.cols();
        add(info);
    }

    for (uint32_t l = 0; l < cfg.block_count; ++l) {
        const std::string hf = "layers." + std::to_string(l) + ".";
        const std::string gg = "blk." + std::to_string(l) + ".";
        add_proj(require(hf + "self_attn.q_proj.weight"), gg + "attn_q.weight");
        add_proj(require(hf + "self_attn.k_proj.weight"), gg + "attn_k.weight");
        add_proj(require(hf + "self_attn.v_proj.weight"), gg + "attn_v.weight");
        add_proj(require(hf + "self_attn.o_proj.weight"), gg + "attn_output.weight");
        // Per-head q/k RMSNorm weights (Qwen3 / Gemma 3; llama.cpp's names), when the model has them.
        if (find(state_dict, hf + "self_attn.q_norm.weight")) {
            add_norm(require(hf + "self_attn.q_norm.weight"), gg + "attn_q_norm.weight");
            add_norm(require(hf + "self_attn.k_norm.weight"), gg + "attn_k_norm.weight");
        }
        add_norm(require(hf + "input_layernorm.weight"), gg + "attn_norm.weight");
        if (cfg.expert_count > 0) {
            add_router(hf, gg + "ffn_gate_inp.weight");
            add_experts(hf, "gate_proj.weight", gg + "ffn_gate_exps.weight");
            add_experts(hf, "up_proj.weight", gg + "ffn_up_exps.weight");
            add_experts(hf, "down_proj.weight", gg + "ffn_down_exps.weight");
        } else {
            add_proj(require(hf + "mlp.gate_proj.weight"), gg + "ffn_gate.weight");
            add_proj(require(hf + "mlp.up_proj.weight"), gg + "ffn_up.weight");
            add_proj(require(hf + "mlp.down_proj.weight"), gg + "ffn_down.weight");
        }
        add_norm(require(hf + "post_attention_layernorm.weight"), gg + "ffn_norm.weight");
    }

    add_norm(require("norm.weight"), "output_norm.weight");
    for (const auto& [key, name] : cfg.extra_proj) add_proj(require(key), name);
    for (const auto& [key, name] : cfg.extra_vec) {
        const Matrix& v = require(key);
        if (v.rows() != 1) throw std::runtime_error("export_gguf_llama: extra_vec " + key + " is not a row");
        add_norm(v, name);
    }

    // lm_head is optional: tied-embedding models reuse token_embd.
    if (const Matrix* head = find(state_dict, "lm_head.weight")) {
        mapped_keys.push_back("lm_head.weight");
        add_proj(*head, "output.weight");
    }

    // Report anything in the dict we did not export (biases, rotary
    // caches...) so callers can fail loudly instead of silently shipping a
    // model missing weights it needs.
    if (unmapped) {
        for (const auto& [key, m] : state_dict) {
            std::string bare = key.rfind("model.", 0) == 0 ? key.substr(6) : key;
            bool used = false;
            for (const auto& mk : mapped_keys) used = used || mk == bare;
            if (!used) unmapped->push_back(key);
        }
    }

    // ---- quantise and hand the tensors to the writer ----
    // GGUFWriter sizes a tensor as num_elements * ggml_type_size(type) and
    // knows no block sizes, so a quantised tensor is handed over as its byte
    // count in that unit, the kept-alive buffer zero-padded to a whole unit.
    // The writer zero-pads every tensor to 32 bytes anyway, so the file is
    // unchanged while the extra bytes stay inside that padding (checked).
    std::vector<std::unique_ptr<std::vector<uint8_t>>> owned_q;
    auto store = [&](TensorExportInfo info, GGMLType type) {
        size_t block = 0, per = 32;  // bytes per block of `per` values
        void (*quantize)(const float*, size_t, uint8_t*) = nullptr;
        const char* tname = "";
        switch (static_cast<uint32_t>(type)) {
            case static_cast<uint32_t>(GGMLType::Q4_0): block = 18, quantize = quantize_q4_0, tname = "Q4_0"; break;
            case static_cast<uint32_t>(GGMLType::Q5_0): block = 22, quantize = quantize_q5_0, tname = "Q5_0"; break;
            case static_cast<uint32_t>(GGMLType::Q5_1): block = 24, quantize = quantize_q5_1, tname = "Q5_1"; break;
            case static_cast<uint32_t>(GGMLType::Q8_0): block = 34, quantize = quantize_q8_0, tname = "Q8_0"; break;
            case kQ4_K: block = 144, per = 256, quantize = quantize_q4_k, tname = "Q4_K"; break;
            case kQ5_K: block = 176, per = 256, quantize = quantize_q5_k, tname = "Q5_K"; break;
            case kQ6_K: block = 210, per = 256, quantize = quantize_q6_k, tname = "Q6_K"; break;
            default:
                throw std::runtime_error("export_gguf_llama: " + info.name + ": no quantiser for ggml type " +
                                         std::to_string(static_cast<uint32_t>(type)));
        }
        const uint64_t ne0 = info.shape.back();  // the writer reverses: last entry is ne0
        if (ne0 % per != 0)
            throw std::runtime_error("export_gguf_llama: " + info.name + " has ne0 = " + std::to_string(ne0) +
                                     ", not a multiple of " + std::to_string(per) + ", so it cannot be stored as " +
                                     tname);
        const size_t bytes = info.num_elements / per * block;
        const size_t unit = gguf_export::ggml_type_size(type);
        const size_t units = (bytes + unit - 1) / unit;
        if (((bytes + 31) & ~size_t(31)) != ((units * unit + 31) & ~size_t(31)))
            throw std::runtime_error("export_gguf_llama: GGUFWriter's size unit for " + std::string(tname) +
                                     " cannot carry the " + std::to_string(bytes) + " bytes of " + info.name);
        auto buf = std::make_unique<std::vector<uint8_t>>(units * unit, 0);
        quantize(info.data, info.num_elements, buf->data());
        info.type = type;
        info.data = reinterpret_cast<const float*>(buf->data());  // written as raw bytes
        info.num_elements = units;
        owned_q.push_back(std::move(buf));
        writer.add_tensor(info);
    };

    if (lcpp) {
        // llama.cpp's quantiser: its tensor order, then its type for every
        // tensor of two or more dimensions (1-D norms stay F32).
        std::stable_sort(pending.begin(), pending.end(),
                         [](const TensorExportInfo& x, const TensorExportInfo& y) {
                             return llamacpp_before(x.name, y.name);
                         });
        bool has_output = false;
        int n_attn_v = 0;
        for (const auto& t : pending) {
            has_output = has_output || t.name == "output.weight";
            n_attn_v += t.name.find("attn_v.weight") != std::string::npos;
        }
        int i_attn_v = 0, i_ffn_down = 0;
        for (const auto& t : pending) {
            if (t.shape.size() < 2) {
                writer.add_tensor(t);
                continue;
            }
            const uint32_t type = llamacpp_type(cfg.quant, t.name, t.shape.back(), has_output, i_attn_v, n_attn_v,
                                                i_ffn_down, static_cast<int>(cfg.block_count));
            store(t, static_cast<GGMLType>(type));
        }
    } else {
        // The Q8_0 variants' selection (see LlamaExportConfig::quant), in
        // the exporter's own order.
        for (const auto& t : pending) {
            const bool pick = q8 && t.shape.size() == 2 && t.type == GGMLType::F32 &&
                              (t.name.rfind("blk.", 0) == 0 ||
                               (cfg.quant != "q8_0" && t.name == "output.weight") ||
                               (cfg.quant == "q8_0_all" && t.name == "token_embd.weight"));
            if (pick)
                store(t, GGMLType::Q8_0);
            else
                writer.add_tensor(t);
        }
    }

    if (!writer.finalize()) {
        throw std::runtime_error("export_gguf_llama: finalize failed for " + path);
    }
    if (lcpp) {
        // llama.cpp pads the last tensor to the alignment as well.
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        const auto size = static_cast<uint64_t>(in.tellg());
        in.close();
        const uint64_t pad = (gguf_export::GGUF_DEFAULT_ALIGNMENT - size % gguf_export::GGUF_DEFAULT_ALIGNMENT) %
                             gguf_export::GGUF_DEFAULT_ALIGNMENT;
        std::ofstream out(path, std::ios::binary | std::ios::app);
        for (uint64_t i = 0; i < pad; ++i) out.put('\0');
        if (!out) throw std::runtime_error("export_gguf_llama: could not pad " + path);
    }
}

void export_gguf_gpt2(const std::string& path, const std::map<std::string, Matrix>& state_dict,
                      const Gpt2ExportConfig& cfg, std::vector<std::string>* unmapped) {
    const char* who = "export_gguf_gpt2";
    if (cfg.context_length == 0 || cfg.embedding_length == 0 || cfg.block_count == 0 ||
        cfg.feed_forward_length == 0 || cfg.head_count == 0 || cfg.vocab_size == 0)
        throw std::runtime_error(std::string(who) +
                                 ": context_length, embedding_length, block_count, feed_forward_length, "
                                 "head_count and vocab_size must all be set");
    if (cfg.embedding_length % cfg.head_count != 0)
        throw std::runtime_error(std::string(who) + ": embedding_length must divide by head_count");
    if (cfg.architecture != "gpt2" && cfg.architecture != "paperkiln")
        throw std::runtime_error(std::string(who) + ": architecture must be gpt2 or paperkiln, not " + cfg.architecture);
    if (cfg.architecture == "paperkiln" && cfg.ir.empty())
        throw std::runtime_error(std::string(who) + ": architecture paperkiln needs the IR (only it states the model)");
    const uint64_t d = cfg.embedding_length, ff = cfg.feed_forward_length;

    GGUFWriter writer(path);
    const std::string a = cfg.architecture;
    writer.write_metadata_string("general.architecture", a);
    writer.write_metadata_string("general.name", cfg.name);
    writer.write_metadata_uint32("general.alignment", gguf_export::GGUF_DEFAULT_ALIGNMENT);
    writer.write_metadata_uint32(a + ".context_length", cfg.context_length);
    writer.write_metadata_uint32(a + ".embedding_length", cfg.embedding_length);
    writer.write_metadata_uint32(a + ".block_count", cfg.block_count);
    writer.write_metadata_uint32(a + ".feed_forward_length", cfg.feed_forward_length);
    writer.write_metadata_uint32(a + ".attention.head_count", cfg.head_count);
    writer.write_metadata_float32(a + ".attention.layer_norm_epsilon", cfg.layer_norm_eps);
    writer.write_metadata_uint32(a + ".vocab_size", cfg.vocab_size);
    write_tokenizer(writer, cfg);
    write_ir(writer, cfg.ir, who);

    // GGUFWriter buffers raw pointers until finalize(): transposed copies
    // live in `owned` until then. Shapes are passed row-major and written
    // reversed (ne0 = the last entry), as in export_gguf_llama.
    std::vector<std::unique_ptr<std::vector<float>>> owned;
    std::vector<std::string> mapped_keys;
    auto require = [&](const std::string& key, uint64_t rows, uint64_t cols) -> const Matrix& {
        auto it = state_dict.find(key);
        if (it == state_dict.end()) it = state_dict.find("transformer." + key);
        if (it == state_dict.end()) throw std::runtime_error(std::string(who) + ": missing tensor " + key);
        if (it->second.rows() != rows || it->second.cols() != cols)
            throw std::runtime_error(std::string(who) + ": " + key + " is [" + std::to_string(it->second.rows()) +
                                     ", " + std::to_string(it->second.cols()) + "], expected [" +
                                     std::to_string(rows) + ", " + std::to_string(cols) + "]");
        mapped_keys.push_back(key);
        return it->second;
    };
    auto add = [&](const std::string& name, std::vector<uint64_t> shape, const float* data, size_t n) {
        TensorExportInfo info;
        info.type = GGMLType::F32;
        info.name = name;
        info.shape = std::move(shape);
        info.data = data;
        info.num_elements = n;
        writer.add_tensor(info);
    };
    // A [rows, cols] table already in llama.cpp's byte order (embeddings).
    auto add_rows = [&](const Matrix& m, const std::string& name) {
        add(name, {static_cast<uint64_t>(m.rows()), static_cast<uint64_t>(m.cols())}, &m(0, 0), m.rows() * m.cols());
    };
    // [in][out] -> [out][in] bytes, ne = {in, out}.
    auto add_proj = [&](const Matrix& w, const std::string& name) {
        const size_t R = w.rows(), C = w.cols();
        auto buf = std::make_unique<std::vector<float>>(R * C);
        for (size_t i = 0; i < R; ++i)
            for (size_t j = 0; j < C; ++j) (*buf)[j * R + i] = w(i, j);
        add(name, {static_cast<uint64_t>(C), static_cast<uint64_t>(R)}, buf->data(), R * C);
        owned.push_back(std::move(buf));
    };
    // [1, n] -> 1-D {n} (norm weights and biases).
    auto add_vec = [&](const Matrix& v, const std::string& name) {
        add(name, {static_cast<uint64_t>(v.cols())}, &v(0, 0), v.cols());
    };
    // A LayerNorm or a biased projection: hf.weight / hf.bias -> gg.weight / gg.bias.
    auto add_norm = [&](const std::string& hf, const std::string& gg) {
        add_vec(require(hf + ".weight", 1, d), gg + ".weight");
        add_vec(require(hf + ".bias", 1, d), gg + ".bias");
    };
    auto add_linear = [&](const std::string& hf, const std::string& gg, uint64_t in, uint64_t out) {
        add_proj(require(hf + ".weight", in, out), gg + ".weight");
        add_vec(require(hf + ".bias", 1, out), gg + ".bias");
    };

    add_rows(require("wte.weight", cfg.vocab_size, d), "token_embd.weight");
    add_rows(require("wpe.weight", cfg.context_length, d), "position_embd.weight");
    for (uint32_t l = 0; l < cfg.block_count; ++l) {
        const std::string hf = "h." + std::to_string(l) + ".";
        const std::string gg = "blk." + std::to_string(l) + ".";
        add_norm(hf + "ln_1", gg + "attn_norm");
        add_linear(hf + "attn.c_attn", gg + "attn_qkv", d, 3 * d);
        add_linear(hf + "attn.c_proj", gg + "attn_output", d, d);
        add_norm(hf + "ln_2", gg + "ffn_norm");
        add_linear(hf + "mlp.c_fc", gg + "ffn_up", d, ff);
        add_linear(hf + "mlp.c_proj", gg + "ffn_down", ff, d);
    }
    add_norm("ln_f", "output_norm");
    if (state_dict.count("lm_head.weight") || state_dict.count("transformer.lm_head.weight"))
        add_proj(require("lm_head.weight", d, cfg.vocab_size), "output.weight");

    if (unmapped) {
        for (const auto& [key, m] : state_dict) {
            const std::string bare = key.rfind("transformer.", 0) == 0 ? key.substr(12) : key;
            if (std::find(mapped_keys.begin(), mapped_keys.end(), bare) == mapped_keys.end())
                unmapped->push_back(key);
        }
    }
    if (!writer.finalize()) throw std::runtime_error(std::string(who) + ": finalize failed for " + path);
}

void export_gguf_paperkiln(const std::string& path, const std::vector<NamedTensor>& tensors,
                           const PaperkilnExportConfig& cfg) {
    const char* who = "export_gguf_paperkiln";
    if (cfg.context_length == 0 || cfg.embedding_length == 0 || cfg.block_count == 0 || cfg.vocab_size == 0)
        throw std::runtime_error(std::string(who) +
                                 ": context_length, embedding_length, block_count and vocab_size must all be set");
    if (cfg.ir.empty()) throw std::runtime_error(std::string(who) + ": a paperkiln file needs its IR");
    for (size_t i = 0; i < tensors.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (tensors[i].name == tensors[j].name)
                throw std::runtime_error(std::string(who) + ": tensor " + tensors[i].name + " given twice");

    GGUFWriter writer(path);
    const std::string a = "paperkiln";
    writer.write_metadata_string("general.architecture", a);
    writer.write_metadata_string("general.name", cfg.name);
    writer.write_metadata_uint32("general.alignment", gguf_export::GGUF_DEFAULT_ALIGNMENT);
    writer.write_metadata_uint32(a + ".context_length", cfg.context_length);
    writer.write_metadata_uint32(a + ".embedding_length", cfg.embedding_length);
    writer.write_metadata_uint32(a + ".block_count", cfg.block_count);
    if (cfg.feed_forward_length) writer.write_metadata_uint32(a + ".feed_forward_length", cfg.feed_forward_length);
    if (cfg.head_count) writer.write_metadata_uint32(a + ".attention.head_count", cfg.head_count);
    writer.write_metadata_uint32(a + ".vocab_size", cfg.vocab_size);
    write_tokenizer(writer, cfg);
    write_ir(writer, cfg.ir, who);

    // As export_gguf_gpt2: shapes row-major, written reversed; the writer
    // holds raw pointers until finalize(), so every buffer is owned here.
    std::vector<std::unique_ptr<std::vector<float>>> owned;
    for (const auto& t : tensors) {
        const Matrix& m = t.m;
        const size_t R = m.rows(), C = m.cols();
        auto buf = std::make_unique<std::vector<float>>(R * C);
        std::vector<uint64_t> shape;
        if (t.layout == NamedTensor::Layout::PROJ) {
            for (size_t i = 0; i < R; ++i)
                for (size_t j = 0; j < C; ++j) (*buf)[j * R + i] = m(i, j);
            shape = {static_cast<uint64_t>(C), static_cast<uint64_t>(R)};
        } else {
            for (size_t i = 0; i < R; ++i)
                for (size_t j = 0; j < C; ++j) (*buf)[i * C + j] = m(i, j);
            if (t.layout == NamedTensor::Layout::ROWS) {
                shape = {static_cast<uint64_t>(R), static_cast<uint64_t>(C)};
            } else {
                if (R != 1 && C != 1)
                    throw std::runtime_error(std::string(who) + ": " + t.name + " is not a vector");
                shape = {static_cast<uint64_t>(R * C)};
            }
        }
        TensorExportInfo info;
        info.type = GGMLType::F32;
        info.name = t.name;
        info.shape = std::move(shape);
        info.data = buf->data();
        info.num_elements = R * C;
        writer.add_tensor(info);
        owned.push_back(std::move(buf));
    }
    if (!writer.finalize()) throw std::runtime_error(std::string(who) + ": finalize failed for " + path);
}

}  // namespace gguf
}  // namespace microtorch
