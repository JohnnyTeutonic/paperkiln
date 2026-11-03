#pragma once
// MICROTORCH_SCAN_CHUNK, the opt-in switch for the chunked (parallel-in-time) selective scan on
// CUDA (src/cuda_ops.cu reads it on every scan call; the CPU path never reads it):
//   unset, empty or "0"  the ordered scan (the default);
//   a decimal count L    the chunked scan with chunks of L rows (L >= 1);
//   "auto"               the chunked scan with kScanChunkAuto rows per chunk.
// Anything else (signs, spaces, other characters, other spellings of auto, a value past 2^31) is
// refused, so a typo cannot silently select a path. Host-only, so the CPU tests can check it.
//
// Why auto is a constant: on the L4 the kernel timings of runs r1 (forward) and b1 (backward), over
// six (sequence length, batch, width) cells from 64 x 16 to 4096 x 1 and L = 32-256, put a fixed
// L = 128 within 0.2% (forward) and 2% (backward) of each cell's best L, and the end-to-end prefill
// run p2 found L = 64 and 128 within 1.5% of each other everywhere
// (experiments/mamba_parallel_scan_2026-11-03/). A shape-dependent rule would be fitting noise.
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

namespace microtorch {

constexpr std::size_t kScanChunkAuto = 128;

// 0 for the ordered scan, else the chunk length; throws std::runtime_error on a malformed value.
inline std::size_t parse_scan_chunk(const char* v) {
    if (!v || !*v) return 0;
    if (std::strcmp(v, "auto") == 0) return kScanChunkAuto;
    std::size_t L = 0;
    for (const char* p = v; *p; ++p) {
        if (*p < '0' || *p > '9' || L > (std::size_t(1) << 31))
            throw std::runtime_error(
                std::string("MICROTORCH_SCAN_CHUNK must be a decimal count up to 2^31, or auto: ") +
                v);
        L = L * 10 + std::size_t(*p - '0');
    }
    if (L > (std::size_t(1) << 31))
        throw std::runtime_error(
            std::string("MICROTORCH_SCAN_CHUNK must be a decimal count up to 2^31, or auto: ") + v);
    return L;
}

}  // namespace microtorch
