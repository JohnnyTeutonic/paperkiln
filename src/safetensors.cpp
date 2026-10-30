#include "microtorch/safetensors.hpp"
#include "microtorch/device_cache.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace microtorch {

std::map<std::string, Matrix> load_safetensors(const std::string& path,
                                               std::map<std::string, std::string>* skipped) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("safetensors: cannot open " + path);

    std::uint64_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 8);  // little-endian, like x86
    if (!f || hlen == 0 || hlen > (1u << 26)) {
        throw std::runtime_error("safetensors: implausible header length");
    }
    std::string htext(hlen, '\0');
    f.read(htext.data(), static_cast<std::streamsize>(hlen));
    const auto header = nlohmann::json::parse(htext);
    const std::streamoff base = 8 + static_cast<std::streamoff>(hlen);

    std::map<std::string, Matrix> out;
    for (auto it = header.begin(); it != header.end(); ++it) {
        std::string name = it.key();
        if (name == "__metadata__") continue;
        if (name.rfind("transformer.", 0) == 0) name = name.substr(12);
        const auto& meta = it.value();
        const auto shape = meta.at("shape").get<std::vector<std::uint64_t>>();
        const auto offs = meta.at("data_offsets").get<std::vector<std::uint64_t>>();

        if (shape.size() > 2) {  // rank-skip, see header
            if (skipped) (*skipped)[name] = "rank " + std::to_string(shape.size());
            continue;
        }
        const std::string dtype = meta.at("dtype").get<std::string>();
        if (dtype != "F32") {
            throw std::runtime_error("safetensors: " + name + " is " + dtype +
                                     "; this loader is F32-only (phase 1c)");
        }
        const size_t rows = shape.size() == 2 ? shape[0] : 1;
        const size_t cols = shape.size() == 2 ? shape[1] : (shape.empty() ? 1 : shape[0]);
        Matrix m(rows, cols);
        const std::uint64_t nbytes = offs.at(1) - offs.at(0);
        if (nbytes != rows * cols * 4) {
            throw std::runtime_error("safetensors: size mismatch at " + name);
        }
        f.seekg(base + static_cast<std::streamoff>(offs.at(0)));
        f.read(reinterpret_cast<char*>(m.data()), static_cast<std::streamsize>(nbytes));
        if (!f) throw std::runtime_error("safetensors: short read at " + name);
        out.emplace(std::move(name), std::move(m));
    }
    return out;
}

void save_safetensors(const std::string& path, const std::map<std::string, Matrix>& tensors) {
    // Build the JSON header first; offsets are relative to the byte
    // region after the header, per the format spec.
    nlohmann::json header;
    std::uint64_t offset = 0;
    for (const auto& [name, m] : tensors) {
        MT_DEVCHECK_HOST_READ(m, "save_safetensors tensor serialization");
        const std::uint64_t nbytes =
            static_cast<std::uint64_t>(m.rows()) * m.cols() * sizeof(float);
        header[name] = {
            {"dtype", "F32"},
            {"shape", {m.rows(), m.cols()}},
            {"data_offsets", {offset, offset + nbytes}},
        };
        offset += nbytes;
    }
    std::string htext = header.dump();
    // Pad header to 8-byte alignment with spaces -- the reference
    // implementation does this, and tinyllama.cpp's GGUF-adjacent readers
    // assume aligned tensor data (the repair_gguf.py lesson).
    while (htext.size() % 8 != 0) htext.push_back(' ');

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("safetensors: cannot write " + path);
    const std::uint64_t hlen = htext.size();
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(htext.data(), static_cast<std::streamsize>(hlen));
    for (const auto& [name, m] : tensors) {
        for (size_t i = 0; i < m.rows(); ++i)
            for (size_t j = 0; j < m.cols(); ++j) {
                const float v = m(i, j);
                f.write(reinterpret_cast<const char*>(&v), sizeof(float));
            }
    }
    if (!f) throw std::runtime_error("safetensors: write failed " + path);
}

}  // namespace microtorch
