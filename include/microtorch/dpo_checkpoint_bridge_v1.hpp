#pragma once
#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <process.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
namespace dpo_checkpoint_v1 {
inline void invoke(const char* action, const std::string& out, int step, int seed, int batch,
                   int accum, const std::string& spec = "") {
    const char* script = std::getenv("MICROTORCH_DPO_CHECKPOINT_PUBLISH_V1");
    if (!script) return;
    const char* source = std::getenv("DPO_NUMERICAL_SOURCE_COMMIT");
    if (!source) throw std::runtime_error("DPO checkpoint source binding missing");
    const auto a = std::to_string(step), b = std::to_string(seed), c = std::to_string(batch),
               d = std::to_string(accum);
    const char* argv[] = {"python3", script,     action,    "--out",   out.c_str(),  "--step",
                          a.c_str(), "--seed",   b.c_str(), "--batch", c.c_str(),    "--accum",
                          d.c_str(), "--source", source,    "--spec",  spec.c_str(), nullptr};
#ifdef _WIN32
    if (_spawnvp(_P_WAIT, "python3", argv) != 0)
        throw std::runtime_error("DPO checkpoint I/O failed");
#else
    const pid_t pid = fork();
    if (pid < 0) throw std::runtime_error("DPO checkpoint fork failed");
    if (pid == 0) {
        execvp("python3", const_cast<char* const*>(argv));
        _exit(127);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) throw std::runtime_error("DPO checkpoint wait failed");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("DPO checkpoint I/O failed; stop before next update");
#endif
}
}  // namespace dpo_checkpoint_v1
