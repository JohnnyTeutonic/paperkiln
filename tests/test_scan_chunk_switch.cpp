// MICROTORCH_SCAN_CHUNK's parser (src/scan_chunk_switch.hpp), which the CUDA scan path calls on every
// scan: the ordered default, explicit chunk lengths, auto, and the refusals.
#include <cstddef>
#include <stdexcept>
#include <string>

#include "../src/scan_chunk_switch.hpp"
#include "check.hpp"

using microtorch::kScanChunkAuto;
using microtorch::parse_scan_chunk;

static bool refused(const char* v) {
    try {
        parse_scan_chunk(v);
    } catch (const std::runtime_error& e) {
        // the CUDA protocols' gate and the prefill bench match this part of the message
        return std::string(e.what()).find("MICROTORCH_SCAN_CHUNK must be a decimal count") !=
               std::string::npos;
    }
    return false;
}

int main() {
    // the ordered scan
    CHECK(parse_scan_chunk(nullptr) == 0);
    CHECK(parse_scan_chunk("") == 0);
    CHECK(parse_scan_chunk("0") == 0);
    // explicit chunk lengths
    CHECK(parse_scan_chunk("1") == 1);
    CHECK(parse_scan_chunk("2") == 2);
    CHECK(parse_scan_chunk("64") == 64);
    CHECK(parse_scan_chunk("128") == 128);
    CHECK(parse_scan_chunk("0128") == 128);
    CHECK(parse_scan_chunk("2147483648") == (std::size_t(1) << 31));
    // auto is exactly the measured choice, 128, so it runs the code the L = 128 gates passed
    CHECK(kScanChunkAuto == 128);
    CHECK(parse_scan_chunk("auto") == parse_scan_chunk("128"));
    // refusals: a typo never selects a path
    for (const char* bad : {" 2", "2 ", "+2", "-2", "2x", "0x80", "1e3", "Auto", "AUTO", "auto ", " auto",
                            "automatic", "2147483649", "99999999999999999999"})
        CHECK(refused(bad));
    return 0;
}
