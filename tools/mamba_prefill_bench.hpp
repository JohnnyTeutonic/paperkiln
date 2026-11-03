#pragma once
// End-to-end Mamba-1 prefill measurement
// (experiments/mamba_parallel_scan_2026-11-03/PROTOCOL_PREFILL.md): the pieces
// tools/mamba_prefill_bench.cpp runs and tests/test_mamba_prefill_bench.cpp checks on the host.
//
// The model is parity::MambaLM with Mamba1Config on (tools/parity_model.hpp), the composed Mamba-1
// the CUDA lifecycle receipts ran (docs/receipts/mamba1_cli_cuda_20261031/). prefill() is
// MambaLM::forward's Mamba-1 branch line for line, with the head's GEMM under a profile tag; the
// host test checks that it equals MambaLM::forward bit for bit. block_forward() is
// Mamba1Block::forward line for line with the scan passed in, so the CPU calibration can swap in
// the chunked scan's host emulation; the host test checks it, with the op's own scan, against
// Mamba1Block::forward bit for bit.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/selective_scan_chunked.cuh"
#include "parity_model.hpp"

namespace prefill {

using namespace microtorch;

// The gate's tolerances under err() (below): chunked against ordered prefill, and either against
// the CPU path, for the logits and for every block's scan output. Rounding differences grow with
// depth, so the bound is set per shape: 1e-4 for the 2-layer shape, 1e-3 for the 12- and 24-layer
// ones. PROTOCOL_PREFILL.md gives the argument; receipts/prefill_calibration_local.json and the
// host test the host evidence.
constexpr double kBoundShallow = 1e-4, kBoundDeep = 1e-3;

// Mamba-1 shapes: d_state 16, expand 2, d_conv 4, dt_rank ceil(d / 16), and Mamba's GPT-NeoX
// vocabulary padded to 50280 (state-spaces/mamba-130m's config). "tiny" is the host tests' and dry
// runs' shape.
struct Shape {
    std::string name;
    size_t d, layers, vocab;
    double bound;
};
inline const std::vector<Shape>& shapes() {
    static const std::vector<Shape> s = {
        {"d768x24", 768, 24, 50280, kBoundDeep},  // Mamba-130m's width, depth and vocabulary
        {"d384x12", 384, 12, 50280, kBoundDeep},  // half the width, half the depth
        {"tiny", 64, 2, 512, kBoundShallow},
    };
    return s;
}
inline const Shape& shape(const std::string& name) {
    for (const auto& s : shapes())
        if (s.name == name) return s;
    throw std::invalid_argument("unknown shape " + name);
}
constexpr unsigned kModelSeed = 1103;
constexpr unsigned kIdsSeed = 7;

struct Case {
    size_t sl, batch;
};
// The timed cases: single sequences of 512-4096 tokens and one batched case.
inline std::vector<Case> timed_cases() {
    return {{512, 1}, {1024, 1}, {2048, 1}, {4096, 1}, {2048, 4}};
}

// A scan variant: 0 is the ordered scan (MICROTORCH_SCAN_CHUNK unset), L the chunked scan at chunk
// length L.
inline std::vector<size_t> timed_variants() {
    return {0, 64, 128};
}
inline std::string variant_name(size_t L) {
    return L == 0 ? "ordered" : "L" + std::to_string(L);
}

// Repetition r runs the variants starting at r mod n, so over n repetitions each variant takes each
// position once (drift and warm-up order effects are spread evenly).
inline std::vector<size_t> rotation(const std::vector<size_t>& variants, size_t rep) {
    std::vector<size_t> out;
    for (size_t i = 0; i < variants.size(); ++i)
        out.push_back(variants[(rep + i) % variants.size()]);
    return out;
}

inline double median(std::vector<double> v) {
    if (v.empty()) throw std::invalid_argument("median of nothing");
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---- comparison: err(a, ref) = max_i |a_i - ref_i| / rms(ref)
struct Cmp {
    size_t n = 0;
    bool same_size = false, finite = false, identical = false;
    double max_abs = 0, rms_ref = 0, err = 0;
    bool pass(double bound) const { return same_size && finite && err <= bound; }
};
inline Cmp compare(const std::vector<float>& a, const std::vector<float>& ref) {
    Cmp c;
    c.n = ref.size();
    c.same_size = a.size() == ref.size() && !ref.empty();
    if (!c.same_size) {
        c.err = std::numeric_limits<double>::infinity();
        return c;
    }
    c.finite = true;
    double ss = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(ref[i])) c.finite = false;
        const double d = std::fabs(double(a[i]) - double(ref[i]));
        if (d > c.max_abs || std::isnan(d)) c.max_abs = d;
        ss += double(ref[i]) * double(ref[i]);
    }
    c.identical = std::memcmp(a.data(), ref.data(), a.size() * sizeof(float)) == 0;
    c.rms_ref = std::sqrt(ss / double(ref.size()));
    if (!c.finite)
        c.err = std::numeric_limits<double>::infinity();
    else if (c.rms_ref > 0)
        c.err = c.max_abs / c.rms_ref;
    else
        c.err = c.max_abs == 0 ? 0 : std::numeric_limits<double>::infinity();
    return c;
}

// ---- the scan switch: the device path reads MICROTORCH_SCAN_CHUNK on every call (src/cuda_ops.cu)
inline void set_chunk(size_t L) {
#ifdef _WIN32
    _putenv_s("MICROTORCH_SCAN_CHUNK", L ? std::to_string(L).c_str() : "");
#else
    if (L == 0)
        unsetenv("MICROTORCH_SCAN_CHUNK");
    else
        setenv("MICROTORCH_SCAN_CHUNK", std::to_string(L).c_str(), 1);
#endif
}
inline void set_chunk_raw(const char* v) {  // the malformed-value check
#ifdef _WIN32
    _putenv_s("MICROTORCH_SCAN_CHUNK", v);
#else
    setenv("MICROTORCH_SCAN_CHUNK", v, 1);
#endif
}
inline bool chunk_env_set() {
    const char* v = std::getenv("MICROTORCH_SCAN_CHUNK");
    return v && *v;
}

// ---- arguments
struct Args {
    std::string mode;  // gate | time | profile | calibrate
    std::vector<std::string> shapes;
    size_t reps = 0, warmup = 0;
    bool require_cuda = false, host_dry_run = false;
};
inline size_t parse_count(const std::string& flag, const std::string& v, size_t lo, size_t hi) {
    if (v.empty() || v.size() > 6 || v.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(flag + " wants a decimal count");
    const size_t n = std::stoul(v);
    if (n < lo || n > hi)
        throw std::invalid_argument(flag + " must be in " + std::to_string(lo) + ".." +
                                    std::to_string(hi));
    return n;
}
inline Args parse_args(const std::vector<std::string>& argv) {
    Args a;
    bool have_reps = false, have_warmup = false, have_shapes = false;
    for (size_t i = 0; i < argv.size(); ++i) {
        const std::string& k = argv[i];
        auto value = [&]() -> const std::string& {
            if (i + 1 >= argv.size()) throw std::invalid_argument(k + " needs a value");
            return argv[++i];
        };
        if (k == "--mode") {
            if (!a.mode.empty()) throw std::invalid_argument("--mode given twice");
            a.mode = value();
            if (a.mode != "gate" && a.mode != "time" && a.mode != "profile" &&
                a.mode != "calibrate")
                throw std::invalid_argument("--mode must be gate, time, profile or calibrate");
        } else if (k == "--reps") {
            if (have_reps) throw std::invalid_argument("--reps given twice");
            a.reps = parse_count(k, value(), 1, 1000);
            have_reps = true;
        } else if (k == "--warmup") {
            if (have_warmup) throw std::invalid_argument("--warmup given twice");
            a.warmup = parse_count(k, value(), 0, 100);
            have_warmup = true;
        } else if (k == "--shapes") {
            if (have_shapes) throw std::invalid_argument("--shapes given twice");
            const std::string v = value();
            size_t p = 0;
            while (p <= v.size()) {
                const size_t q = std::min(v.find(',', p), v.size());
                const std::string s = v.substr(p, q - p);
                if (s.empty() || s == "tiny")
                    throw std::invalid_argument("--shapes: d768x24 and/or d384x12");
                shape(s);  // throws on an unknown name
                if (std::find(a.shapes.begin(), a.shapes.end(), s) != a.shapes.end())
                    throw std::invalid_argument("--shapes: " + s + " twice");
                a.shapes.push_back(s);
                p = q + 1;
            }
            have_shapes = true;
        } else if (k == "--require-cuda") {
            a.require_cuda = true;
        } else if (k == "--host-dry-run") {
            a.host_dry_run = true;
        } else {
            throw std::invalid_argument("unknown argument " + k);
        }
    }
    if (a.mode.empty()) throw std::invalid_argument("--mode is required");
    if (a.require_cuda && a.host_dry_run)
        throw std::invalid_argument("--require-cuda and --host-dry-run conflict");
    if (a.mode == "calibrate" && a.require_cuda)
        throw std::invalid_argument("calibrate runs on the CPU");
    if (a.host_dry_run && have_shapes)
        throw std::invalid_argument("--host-dry-run uses the tiny shape only");
    if ((have_reps || have_warmup) && a.mode != "time" && a.mode != "profile")
        throw std::invalid_argument("--reps and --warmup are for time and profile");
    if (!have_shapes)
        a.shapes = a.host_dry_run ? std::vector<std::string>{"tiny"}
                                  : std::vector<std::string>{"d768x24", "d384x12"};
    if (!have_reps) a.reps = a.mode == "profile" ? 5 : 15;
    if (!have_warmup) a.warmup = 2;
    return a;
}

// ---- the model and its inputs, deterministic from the seeds
inline std::unique_ptr<parity::MambaLM> make_model(const Shape& s) {
    parity::Mamba1Config c;
    c.on = true;
    c.d_conv = 4;
    c.expand = 2;
    c.dt_rank = 0;  // ceil(d / 16)
    return std::make_unique<parity::MambaLM>(s.vocab, s.d, s.layers, 16, kModelSeed, c);
}
inline std::vector<int> make_ids(size_t vocab, size_t rows, unsigned seed = kIdsSeed) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> u(0, static_cast<int>(vocab) - 1);
    std::vector<int> ids(rows);
    for (auto& v : ids) v = u(g);
    return ids;
}

// MambaLM::forward's Mamba-1 branch, with the head's GEMM reported as "gemm_head" under
// MICROTORCH_GPU_PROFILE (names only; numerics untouched).
inline Var prefill(const parity::MambaLM& m, const std::vector<int>& ids, size_t sl) {
    if (!m.m1cfg.on) throw std::invalid_argument("prefill: a Mamba-1 MambaLM");
    if (sl == 0 || ids.size() % sl != 0)
        throw std::invalid_argument("prefill: rows not a multiple of seq_len");
    Var h = m.embed->forward(ids);
    for (const auto& b : m.m1blocks) h = b->forward(h, sl);
    Var hn = ops::rmsnorm(h, m.norm_f);
    device::ProfileTag tag("gemm_head");
    return m.lm_head->forward(hn);
}

// Mamba1Block::forward with the scan passed in: scan(xs, dt, B, C, A, D, seq_len).
template <class Scan>
Var block_forward(const parity::Mamba1Block& b, const Var& x, size_t sl, const Scan& scan) {
    Var h = ops::rmsnorm(x, b.norm_w);
    Var xz = b.in_proj->forward(h);
    Var xs = ops::slice_cols(xz, 0, b.Di), z = ops::slice_cols(xz, b.Di, 2 * b.Di);
    xs = ops::silu(ops::conv1d_causal(xs, b.conv_w, b.conv_b, sl));
    Var xdb = b.x_proj->forward(xs);
    Var dt = b.dt_proj->forward(ops::slice_cols(xdb, 0, b.R));
    Var Bm = ops::slice_cols(xdb, b.R, b.R + b.N),
        Cm = ops::slice_cols(xdb, b.R + b.N, b.R + 2 * b.N);
    Var y = scan(xs, dt, Bm, Cm, b.A(), b.Dskip, sl);
    y = ops::mul(y, ops::silu(z));
    return ops::add(x, b.out_proj->forward(y));
}
template <class Scan>
Var prefill_with(const parity::MambaLM& m, const std::vector<int>& ids, size_t sl,
                 const Scan& scan) {
    if (!m.m1cfg.on) throw std::invalid_argument("prefill_with: a Mamba-1 MambaLM");
    if (sl == 0 || ids.size() % sl != 0)
        throw std::invalid_argument("prefill_with: rows not a multiple of seq_len");
    Var h = m.embed->forward(ids);
    for (const auto& b : m.m1blocks) h = block_forward(*b, h, sl, scan);
    return m.lm_head->forward(ops::rmsnorm(h, m.norm_f));
}

// The scans the CPU calibration and host tests swap in. ordered: the op itself.
struct OrderedScan {
    Var operator()(const Var& x, const Var& dt, const Var& B, const Var& C, const Var& A,
                   const Var& D, size_t sl) const {
        return ops::selective_scan(x, dt, B, C, A, D, /*dt_softplus=*/true, sl);
    }
};
// The chunked forward's host emulation (src/selective_scan_chunked.cuh: the functions the device
// kernels call, in the same phases), at chunk length L. Host data only (the CPU path).
struct ChunkedHostScan {
    size_t L;
    Var operator()(const Var& x, const Var& dt, const Var& B, const Var& C, const Var& A,
                   const Var& D, size_t sl) const {
        const size_t T = x->data.rows(), Dc = x->data.cols(), N = A->data.cols();
        Matrix y(T, Dc);
        std::vector<float> S((T + T / sl) * Dc * N), delta(T * Dc);
        ss_chunked::selective_scan_chunked_host(
            x->data.get_data(), dt->data.get_data(), B->data.get_data(), C->data.get_data(),
            A->data.get_data(), D->data.get_data(), y.get_data(), S.data(), delta.data(), T, Dc, N,
            sl, L, true);
        return make_var(std::move(y));
    }
};
// A deliberately broken scan: the ordered op restarted from zero every L rows, which is what a
// chunked scan that dropped its carried chunk starts would compute. The calibration and host test
// use it to show that the bound detects that class of error. L must divide the rows.
struct DroppedCarryScan {
    size_t L;
    Var operator()(const Var& x, const Var& dt, const Var& B, const Var& C, const Var& A,
                   const Var& D, size_t) const {
        return ops::selective_scan(x, dt, B, C, A, D, /*dt_softplus=*/true, L);
    }
};

inline std::vector<float> host_values(const Var& v) {
    device::materialize(v->data);
    const float* p = v->data.get_data();
    return std::vector<float>(p, p + v->data.rows() * v->data.cols());
}

// A prefill's logits and every block's scan output y. The logits alone are a weak witness at these
// random weights: the scan's state term is a fraction of y (the D skip carries the rest), and each
// block adds a small update to the residual stream, so a scan that dropped its chunk starts moves
// the logits by only ~5e-4 of their rms at the tiny shape. Every layer's y is compared too.
struct Run {
    std::vector<float> logits;
    std::vector<std::vector<float>> scan_y;  // per block, in order
};
// A 64-bit fingerprint of the exact bits (FNV-1a's xor-multiply over each float's bit pattern, then
// the length): the repeat checks compare these instead of holding a second copy (a d768x24 batched
// run is ~2.9 GB on the host). Equal fingerprints of runs that differ are a ~2^-64 event.
inline void fnv(uint64_t& h, const std::vector<float>& v) {
    for (float x : v) {
        uint32_t u;
        std::memcpy(&u, &x, sizeof u);
        h = (h ^ u) * 1099511628211ULL;
    }
    h = (h ^ v.size()) * 1099511628211ULL;
}
inline uint64_t fingerprint(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ULL;
    fnv(h, v);
    return h;
}
inline uint64_t fingerprint(const Run& r) {
    uint64_t h = 1469598103934665603ULL;
    fnv(h, r.logits);
    for (const auto& y : r.scan_y) fnv(h, y);
    return h;
}
template <class Inner>
struct RecordingScan {
    Inner inner;
    std::vector<std::vector<float>>* ys;
    Var operator()(const Var& x, const Var& dt, const Var& B, const Var& C, const Var& A,
                   const Var& D, size_t sl) const {
        Var y = inner(x, dt, B, C, A, D, sl);
        ys->push_back(host_values(y));  // a download on the device path; the values are unchanged
        return y;
    }
};
template <class Scan>
Run record_run(const parity::MambaLM& m, const std::vector<int>& ids, size_t sl, const Scan& scan) {
    Run r;
    NoGrad ng;
    r.logits = host_values(prefill_with(m, ids, sl, RecordingScan<Scan>{scan, &r.scan_y}));
    return r;
}
struct RunCmp {
    Cmp logits, scan;  // scan: the block with the largest err
    size_t scan_layer = 0;
    bool identical = false;  // logits and every scan output bit for bit
    bool pass(double bound) const { return logits.pass(bound) && scan.pass(bound); }
};
inline RunCmp compare(const Run& a, const Run& ref) {
    RunCmp c;
    c.logits = compare(a.logits, ref.logits);
    c.identical = c.logits.identical && a.scan_y.size() == ref.scan_y.size();
    if (a.scan_y.size() != ref.scan_y.size() || ref.scan_y.empty()) {
        c.scan.err = std::numeric_limits<double>::infinity();
        return c;
    }
    for (size_t l = 0; l < ref.scan_y.size(); ++l) {
        const Cmp k = compare(a.scan_y[l], ref.scan_y[l]);
        c.identical = c.identical && k.identical;
        if (l == 0 || k.err > c.scan.err) {  // a nonfinite or missized layer has err = inf
            c.scan = k;
            c.scan_layer = l;
        }
    }
    return c;
}

}  // namespace prefill
