// Caching device allocator shared by cuda_resident.cu and cuda_ops.cu
// (18 Oct 2026). Why: every device op allocated and freed its operands and
// outputs, and gdb samples of a chat-shape training step on an L4 put
// cudaMalloc/cudaFree near 12% of the step on their own (cudaFree syncs the
// device). Shapes repeat every step, so a freed block is kept on a free list
// keyed by its rounded size and handed to the next request of that size.
//
// Safety: every kernel and copy in these TUs runs on the default stream,
// so work using a block is ordered before any later work on the block that
// reuses it, the property cudaFree's implicit sync provided. Cached bytes
// are capped (MICROTORCH_POOL_MB, default 1024), and the cache is returned
// to the driver if a cudaMalloc fails, before one retry. Pointers the pool
// did not hand out are freed with cudaFree as before.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace microtorch {
namespace device {
namespace pool {

struct Pool {
    std::mutex mu;
    std::unordered_map<void*, size_t> live;                 // handed out: pointer -> rounded bytes
    std::unordered_map<size_t, std::vector<void*>> freel;   // rounded bytes -> cached blocks
    size_t cached = 0;
    size_t cap = 0;
};

inline Pool& get() {
    // Leaked on purpose: static destructors run after the CUDA context is
    // gone, and freeing then would fail.
    static Pool* p = [] {
        auto* q = new Pool;
        const char* e = std::getenv("MICROTORCH_POOL_MB");
        q->cap = static_cast<size_t>(e ? std::atol(e) : 1024) << 20;
        return q;
    }();
    return *p;
}

inline size_t round_bytes(size_t b) { return b == 0 ? 512 : (b + 511) / 512 * 512; }

// Return every cached block to the driver. Caller holds the lock.
inline void release_cached_locked(Pool& p) {
    for (auto& kv : p.freel)
        for (void* q : kv.second) cudaFree(q);
    p.freel.clear();
    p.cached = 0;
}

inline void release_cached() {
    Pool& p = get();
    std::lock_guard<std::mutex> lk(p.mu);
    release_cached_locked(p);
}

inline size_t cached_bytes() {
    Pool& p = get();
    std::lock_guard<std::mutex> lk(p.mu);
    return p.cached;
}

template <class T>
inline cudaError_t dmalloc(T** ptr, size_t bytes) {
    Pool& p = get();
    const size_t b = round_bytes(bytes);
    std::lock_guard<std::mutex> lk(p.mu);
    auto it = p.freel.find(b);
    if (it != p.freel.end() && !it->second.empty()) {
        void* q = it->second.back();
        it->second.pop_back();
        p.cached -= b;
        p.live[q] = b;
        *ptr = static_cast<T*>(q);
        return cudaSuccess;
    }
    void* q = nullptr;
    cudaError_t e = cudaMalloc(&q, b);
    if (e == cudaErrorMemoryAllocation) {
        cudaGetLastError();  // clear the sticky-free error before retrying
        release_cached_locked(p);
        e = cudaMalloc(&q, b);
    }
    if (e != cudaSuccess) return e;
    p.live[q] = b;
    *ptr = static_cast<T*>(q);
    return cudaSuccess;
}

inline cudaError_t dfree(void* q) {
    if (q == nullptr) return cudaSuccess;
    Pool& p = get();
    std::lock_guard<std::mutex> lk(p.mu);
    auto it = p.live.find(q);
    if (it == p.live.end()) return cudaFree(q);
    const size_t b = it->second;
    p.live.erase(it);
    if (p.cached + b > p.cap) return cudaFree(q);
    p.freel[b].push_back(q);
    p.cached += b;
    return cudaSuccess;
}

}  // namespace pool
}  // namespace device
}  // namespace microtorch
