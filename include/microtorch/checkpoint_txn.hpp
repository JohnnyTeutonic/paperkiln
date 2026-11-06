#pragma once
// Transactional checkpoints for `mtstudio run`, opt-in with MICROTORCH_CHECKPOINT_TXN=1 (stronger base, 7 Nov 2026).
//
// The plain save writes model.safetensors, optim.safetensors and state.txt in place, so a job lost in the write leaves a
// torn file, and separate renames could leave weights, optimiser and step from different steps. Here one step's files are
// one unit:
//
//   OUT/ckpt_gen/g<step>.tmp/   the payloads and MANIFEST.json (sizes and sha256), each fsynced, then the directory
//   OUT/ckpt_gen/g<step>/       renamed whole; nothing reads it yet
//   OUT/ckpt_gen/live           a symlink to the generation, replaced by one rename: the commit point
//
// OUT/model.safetensors, optim.safetensors and state.txt become fixed symlinks through `live`, so everything that reads the
// flat names (jobs_vm.py's snapshots and restore, the runner's relay) sees one committed step at each open. recover()
// runs before a resume: it validates the live generation by manifest and hashes, falls back to the previous valid one if
// that fails, refuses if a pointer exists and no generation is whole, and drops what was never committed. commit() keeps the
// previous generation, so a reader or a fallback always has one. Bytes are those of save_safetensors. Nothing here touches
// arithmetic: without the variable the trainer takes the old path.
//
// MICROTORCH_CKPT_CRASH=stage[:step] ends the process at a publication boundary, for the crash tests.
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "microtorch/safetensors.hpp"

namespace microtorch::ckpt_txn {

namespace fs = std::filesystem;

class Sha256 {
public:
    void update(const void* data, size_t n) {
        const auto* p = static_cast<const unsigned char*>(data);
        total_ += n;
        if (fill_) {
            const size_t take = std::min(n, sizeof(buf_) - fill_);
            std::memcpy(buf_ + fill_, p, take);
            fill_ += take;
            p += take;
            n -= take;
            if (fill_ == sizeof(buf_)) {
                block(buf_);
                fill_ = 0;
            }
        }
        for (; n >= sizeof(buf_); n -= sizeof(buf_), p += sizeof(buf_)) block(p);
        if (n) {
            std::memcpy(buf_, p, n);
            fill_ = n;
        }
    }
    std::string hex() {
        const uint64_t bits = total_ * 8;
        const unsigned char pad = 0x80, zero = 0;
        update(&pad, 1);
        while (fill_ != 56) update(&zero, 1);
        unsigned char len[8];
        for (int i = 0; i < 8; ++i) len[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
        update(len, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h_[i]);
        return std::string(out, 64);
    }

private:
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const unsigned char* p) {
        static constexpr uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) | (uint32_t(p[4 * i + 2]) << 8) |
                   uint32_t(p[4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
        h_[5] += f;
        h_[6] += g;
        h_[7] += h;
    }
    uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    unsigned char buf_[64];
    size_t fill_ = 0;
    uint64_t total_ = 0;
};

inline void crash_point(const char* stage, int step) {
    static const char* want = std::getenv("MICROTORCH_CKPT_CRASH");
    if (!want) return;
    const std::string w(want);
    const auto colon = w.find(':');
    if (w.substr(0, colon) != stage) return;
    if (colon != std::string::npos && std::atoi(w.c_str() + colon + 1) != step) return;
    std::fprintf(stderr, "checkpoint crash test: %s at step %d\n", stage, step);
    _exit(137);
}

inline void fsync_dir(const fs::path& d) {
    const int fd = ::open(d.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) throw std::runtime_error("checkpoint: cannot open directory " + d.string());
    const int rc = ::fsync(fd);
    const int err = errno;
    ::close(fd);
    if (rc != 0 && err != EINVAL) throw std::runtime_error("checkpoint: fsync failed on " + d.string());
}

// A file written once: buffered, hashed as it goes (a read-back from a network volume costs far more), fsynced at close.
class Writer {
public:
    explicit Writer(const fs::path& p) : path_(p), buf_(1 << 20) {
        fd_ = ::open(p.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (fd_ < 0) throw std::runtime_error("checkpoint: cannot create " + p.string());
    }
    ~Writer() {
        if (fd_ >= 0) ::close(fd_);
    }
    void write(const void* data, size_t n) {
        const auto* p = static_cast<const char*>(data);
        while (n) {
            const size_t take = std::min(n, buf_.size() - fill_);
            std::memcpy(buf_.data() + fill_, p, take);
            fill_ += take;
            p += take;
            n -= take;
            if (fill_ == buf_.size()) flush();
        }
    }
    nlohmann::json finish() {
        flush();
        if (::fsync(fd_) != 0) throw std::runtime_error("checkpoint: fsync failed on " + path_.string());
        if (::close(fd_) != 0) throw std::runtime_error("checkpoint: close failed on " + path_.string());
        fd_ = -1;
        return {{"bytes", bytes_}, {"sha256", sha_.hex()}};
    }

private:
    void flush() {
        sha_.update(buf_.data(), fill_);
        size_t off = 0;
        while (off < fill_) {
            const ssize_t w = ::write(fd_, buf_.data() + off, fill_ - off);
            if (w < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("checkpoint: write failed on " + path_.string());
            }
            off += static_cast<size_t>(w);
        }
        bytes_ += fill_;
        fill_ = 0;
    }
    fs::path path_;
    int fd_ = -1;
    std::vector<char> buf_;
    size_t fill_ = 0;
    uint64_t bytes_ = 0;
    Sha256 sha_;
};

inline const char* const kFiles[3] = {"model.safetensors", "optim.safetensors", "state.txt"};

inline int gen_step(const std::string& name) {
    if (name.size() < 2 || name[0] != 'g') return -1;
    for (size_t i = 1; i < name.size(); ++i)
        if (name[i] < '0' || name[i] > '9') return -1;
    return std::atoi(name.c_str() + 1);
}

// Why the generation is not whole, or "" when it is.
inline std::string invalid(const fs::path& d, int step) {
    std::ifstream mf(d / "MANIFEST.json");
    if (!mf) return "no manifest";
    const auto m = nlohmann::json::parse(mf, nullptr, false);
    if (m.is_discarded() || !m.is_object() || m.value("format", "") != "paperkiln.ckpt_gen/1" || m.value("step", -1) != step)
        return "bad manifest";
    for (const char* name : kFiles) {
        const fs::path p = d / name;
        if (!m.contains("files") || !m["files"].contains(name)) return std::string("manifest lacks ") + name;
        std::ifstream f(p, std::ios::binary);
        if (!f) return std::string("missing ") + name;
        Sha256 h;
        std::vector<char> b(1 << 20);
        uint64_t n = 0;
        while (f) {
            f.read(b.data(), static_cast<std::streamsize>(b.size()));
            h.update(b.data(), static_cast<size_t>(f.gcount()));
            n += static_cast<uint64_t>(f.gcount());
        }
        const auto& e = m["files"][name];
        if (e.value("bytes", uint64_t(0)) != n) return std::string("size differs: ") + name;
        if (e.value("sha256", "") != h.hex()) return std::string("hash differs: ") + name;
    }
    std::ifstream st(d / "state.txt");
    std::string line;
    if (!std::getline(st, line) || std::atoi(line.c_str()) != step) return "state.txt names another step";
    return "";
}

struct Recovered {
    std::string live;      // the generation resumed from; "" for none (fresh, or the flat files of an old checkpoint)
    int step = -1;
    bool fallback = false;  // the pointer's generation failed validation and an older one is live now
    std::vector<std::string> rejected;     // kept on disk as rejected_<name>, with why
    std::vector<std::string> uncommitted;  // removed: written and never published
};

class Store {
public:
    explicit Store(const std::string& out_dir) : out_(out_dir), gen_(fs::path(out_dir) / "ckpt_gen") {}

    Recovered recover() {
        Recovered r;
        if (!fs::is_directory(gen_)) return r;
        std::vector<int> steps;
        for (const auto& e : fs::directory_iterator(gen_)) {
            const std::string n = e.path().filename().string();
            if (n.size() > 4 && n.compare(n.size() - 4, 4, ".tmp") == 0) {
                fs::remove_all(e.path());
                r.uncommitted.push_back(n);
            } else if (gen_step(n) >= 0) {
                steps.push_back(gen_step(n));
            }
        }
        std::sort(steps.rbegin(), steps.rend());
        const fs::path link = gen_ / "live";
        std::error_code ec;
        const fs::path target = fs::read_symlink(link, ec);
        if (ec) {  // no commit ever finished: nothing here was published
            for (int s : steps) {
                fs::remove_all(gen_ / ("g" + std::to_string(s)));
                r.uncommitted.push_back("g" + std::to_string(s));
            }
            return r;
        }
        const int live = gen_step(target.string());
        if (live < 0) throw std::runtime_error("checkpoint: live names no generation: " + target.string());
        std::string chosen;
        for (int s : steps) {
            const std::string name = "g" + std::to_string(s);
            if (s > live) {  // renamed into place, pointer never moved
                fs::remove_all(gen_ / name);
                r.uncommitted.push_back(name);
                continue;
            }
            const std::string why = invalid(gen_ / name, s);
            if (why.empty()) {
                chosen = name;
                r.step = s;
                break;
            }
            std::string kept = "rejected_" + name;  // never over an earlier rejection of the same step
            for (int i = 1; fs::exists(gen_ / kept); ++i) kept = "rejected_" + name + "." + std::to_string(i);
            fs::rename(gen_ / name, gen_ / kept);
            r.rejected.push_back(name + ": " + why);
        }
        if (chosen.empty())
            throw std::runtime_error("checkpoint: no whole generation under " + gen_.string() + "; refusing to resume");
        if (chosen != target.string()) {
            point(chosen);
            r.fallback = true;
        }
        links();
        r.live = chosen;
        return r;
    }

    struct Committed {
        std::string generation;
        uint64_t bytes = 0;
    };

    Committed commit(int step, const std::map<std::string, Matrix>& model, const std::map<std::string, Matrix>& optim,
                     const std::string& state_text) {
        fs::create_directories(gen_);
        const std::string name = "g" + std::to_string(step);
        const fs::path tmp = gen_ / (name + ".tmp"), dst = gen_ / name, link = gen_ / "live";
        std::error_code ec;
        const fs::path prev = fs::read_symlink(link, ec);
        if (!ec && prev == name) throw std::runtime_error("checkpoint: generation " + name + " is already live");
        fs::remove_all(tmp);
        fs::remove_all(dst);  // uncommitted by the check above
        fs::create_directory(tmp);
        crash_point("mkdir", step);
        nlohmann::json files;
        files["model.safetensors"] = tensors(tmp / "model.safetensors", model);
        crash_point("model", step);
        files["optim.safetensors"] = tensors(tmp / "optim.safetensors", optim);
        crash_point("optim", step);
        {
            Writer w(tmp / "state.txt");
            w.write(state_text.data(), state_text.size());
            files["state.txt"] = w.finish();
        }
        crash_point("state", step);
        {
            const std::string text = nlohmann::json{{"format", "paperkiln.ckpt_gen/1"}, {"step", step}, {"files", files}}.dump(1);
            Writer w(tmp / "MANIFEST.json");
            w.write(text.data(), text.size());
            w.finish();
        }
        crash_point("manifest", step);
        fsync_dir(tmp);
        crash_point("dirsync", step);
        fs::rename(tmp, dst);
        fsync_dir(gen_);
        crash_point("rename", step);
        point(name);  // the commit
        crash_point("commit", step);
        links();
        crash_point("links", step);
        const int keep_prev = ec ? -1 : gen_step(prev.string());
        for (const auto& e : fs::directory_iterator(gen_)) {
            const int s = gen_step(e.path().filename().string());
            if (s >= 0 && s != step && s != keep_prev) fs::remove_all(e.path());
        }
        crash_point("prune", step);
        Committed c;
        c.generation = name;
        for (const char* n : kFiles) c.bytes += files[n].value("bytes", uint64_t(0));
        return c;
    }

private:
    nlohmann::json tensors(const fs::path& p, const std::map<std::string, Matrix>& t) {
        Writer w(p);
        write_safetensors(t, [&](const void* d, size_t n) { w.write(d, n); });
        return w.finish();
    }

    // The commit point: a new symlink renamed over the old one, then the directory fsynced.
    void point(const std::string& name) {
        const fs::path tmp = gen_ / "live.tmp";
        fs::remove(tmp);
        fs::create_symlink(name, tmp);
        fs::rename(tmp, gen_ / "live");
        fsync_dir(gen_);
    }

    // The flat names, once. An old checkpoint's regular files are replaced only after a generation is live.
    void links() {
        for (const char* n : kFiles) {
            const fs::path p = out_ / n;
            std::error_code ec;
            if (fs::is_symlink(fs::symlink_status(p, ec))) continue;
            const fs::path tmp = out_ / (std::string(n) + ".link.tmp");
            fs::remove(tmp);
            fs::create_symlink(fs::path("ckpt_gen") / "live" / n, tmp);
            fs::rename(tmp, p);
        }
        fsync_dir(out_);
    }

    fs::path out_, gen_;
};

}  // namespace microtorch::ckpt_txn
