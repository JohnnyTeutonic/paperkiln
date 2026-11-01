#pragma once
// Diagnostic only. Never advances checkpoint/cadence state or synchronizes CUDA.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace microtorch {
class DpoStartupPhases {
    using Clock = std::chrono::steady_clock;
    bool enabled_ = false;
    Clock::time_point start_{}, progress_{};
public:
    explicit DpoStartupPhases(bool dpo) noexcept {
        const char* value = std::getenv("MICROTORCH_DPO_PHASES");
        enabled_ = dpo && value && std::strcmp(value, "1") == 0;
        if (enabled_) start_ = progress_ = Clock::now();
    }
    void mark(const char* phase, size_t completed, size_t total, int step = 0) noexcept {
        if (!enabled_) return;
        const double seconds = std::chrono::duration<double>(Clock::now() - start_).count();
        // Constant phase labels and counts only; no prompts, tokens or paths.
        // fprintf/fflush failures are diagnostic failures, never training errors.
        std::fprintf(stderr, "DPO_STARTUP_PHASE {\"phase\":\"%s\",\"elapsed_seconds\":%.6f,\"pairs_completed\":%zu,\"pairs_total\":%zu,\"step\":%d,\"diagnostic_only\":true}\n", phase, seconds, completed, total, step);
        std::fflush(stderr);  // no fsync, new file or manager-clock update
    }
    void progress(size_t completed, size_t total) noexcept {
        if (!enabled_) return;
        const auto now = Clock::now();
        if (completed == 1 || completed % 32 == 0 || completed == total ||
            std::chrono::duration<double>(now - progress_).count() >= 5.0) {
            mark("reference_cache_progress", completed, total);
            progress_ = now;
        }
    }
};
}  // namespace microtorch
