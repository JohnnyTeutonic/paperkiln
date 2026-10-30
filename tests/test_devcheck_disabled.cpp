// Compile without microtorch's public DEVCHECK definition: arguments disappear.
#include <cstdio>
#include "check.hpp"
#include "microtorch/device_cache.hpp"
#ifdef MICROTORCH_DEVCHECK
#error This target must test the default disabled macro.
#endif
int main() {
    int evaluations = 0;
    MT_DEVCHECK_HOST_READ(++evaluations, "disabled macro");
    CHECK(evaluations == 0);
    std::puts("{\"pass\":true,\"devcheck_enabled\":false,\"arguments_evaluated\":false}");
}
