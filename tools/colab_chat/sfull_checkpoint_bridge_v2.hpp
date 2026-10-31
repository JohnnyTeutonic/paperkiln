#pragma once
// Checkpoint I/O amendment only. The save call blocks until publication completes.
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace sfull_checkpoint_v2 {
inline void publish(const std::string& out, int step, int seed, int batch, int accum) {
    const char* script = std::getenv("MICROTORCH_CHECKPOINT_PUBLISH_V2");
    if (!script) return;  // opt-in source; old default checkpoint behavior is preserved
    const char* source = std::getenv("S_FULL_NUMERICAL_COMMIT");
    if (!source) throw std::runtime_error("checkpoint V2 source binding missing");
    const auto a = std::to_string(step), b = std::to_string(seed), c = std::to_string(batch),
               d = std::to_string(accum);
    const pid_t pid = fork();
    if (pid < 0) throw std::runtime_error("checkpoint V2 fork failed");
    if (pid == 0) {
        execlp("python3", "python3", script, "publish", "--out", out.c_str(), "--step", a.c_str(),
               "--seed", b.c_str(), "--batch", c.c_str(), "--accum", d.c_str(), "--source", source,
               static_cast<char*>(nullptr));
        _exit(127);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) throw std::runtime_error("checkpoint V2 wait failed");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("checkpoint V2 publication failed; stop before next update");
}
}  // namespace sfull_checkpoint_v2
