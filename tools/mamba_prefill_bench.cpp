// End-to-end Mamba-1 prefill: ordered against chunked selective scan inside a whole model
// (experiments/mamba_parallel_scan_2026-11-03/PROTOCOL_PREFILL.md). One JSON object per line on
// stdout.
//
// --mode gate [--require-cuda]: correctness. Chunked and ordered prefill agree (logits and every
//     block's scan output), and both match the CPU path; exit 0 only if every check passes.
// --mode time [--require-cuda]: whole-prefill wall time, ordered against L = 64 and 128,
//     alternating, median of --reps (15) after --warmup (2) rounds.
// --mode profile [--require-cuda]: with MICROTORCH_GPU_PROFILE=1, per-op device time from CUDA
//     events, so the scan's share of the prefill.
// --mode calibrate: on the CPU, the chunked scan's host emulation inside the model against the
//     ordered op, and a scan that drops its chunk starts (the tolerance's evidence).
// --shapes d768x24,d384x12 (default both). --host-dry-run runs the same code on the CPU at the
//     tiny shape: bookkeeping only, since the CPU path never reads MICROTORCH_SCAN_CHUNK and its
//     "chunked" arms are ordered.
//
// The device configuration is mtstudio's on-device evaluation (25 Oct 2026): device ops, step
// residency and deferred downloads, one step window per case, so the parameters upload once per
// window and activations stay on the device. GEMMs are the engine's default tiled kernel, or cuBLAS
// FP32 with MICROTORCH_CUBLAS=1. Exit: 0 pass, 1 a check failed, 2 usage or environment.
#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>

#include "mamba_prefill_bench.hpp"
#include "microtorch/device.hpp"
#include "microtorch/device_cache.hpp"

using namespace microtorch;
using json = nlohmann::json;

namespace {

bool g_gpu = false;  // false: the host dry run (and calibrate)

void emit(const json& j) {
    std::cout << j.dump() << "\n";
    std::cout.flush();
}

void use_device() {
    if (!g_gpu) return;
    device::set(device::Device::CUDA);
    device::set_device_ops(true);
    device::set_step_residency(true);
    device::set_defer_downloads(true);
}
void use_cpu() {
    if (g_gpu) {
        device::set_defer_downloads(false);
        device::set_step_residency(false);
        device::set_device_ops(false);
        device::evict_all();
    }
    device::set(device::Device::CPU);
}

// One step window: the parameters upload once, activations stay on the device; nothing is stale
// after it.
struct Window {
    Window() {
        if (g_gpu) device::step_begin();
    }
    ~Window() {
        if (g_gpu) device::step_end();
    }
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
};

// The timed function's logits (prefill(), Mamba1Block::forward inside).
std::vector<float> logits(const parity::MambaLM& m, const std::vector<int>& ids, size_t sl,
                          size_t L) {
    prefill::set_chunk(L);
    NoGrad ng;
    Var lg = prefill::prefill(m, ids, sl);
    std::vector<float> v = prefill::host_values(lg);
    prefill::set_chunk(0);
    return v;
}
// Logits and every block's scan output (the replica, with the op's own scan reading the switch).
prefill::Run recorded(const parity::MambaLM& m, const std::vector<int>& ids, size_t sl, size_t L) {
    prefill::set_chunk(L);
    prefill::Run r = prefill::record_run(m, ids, sl, prefill::OrderedScan{});
    prefill::set_chunk(0);
    return r;
}

double wall_ms(const parity::MambaLM& m, const std::vector<int>& ids, size_t sl, size_t L) {
    prefill::set_chunk(L);
    Var lg;
    device::synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    {
        NoGrad ng;
        lg = prefill::prefill(m, ids, sl);
    }
    device::synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    lg.reset();  // its device buffers go back to the pool outside the timed span
    prefill::set_chunk(0);
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

json cmp_json(const prefill::Cmp& c, double bound) {
    return {{"n", c.n},
            {"same_size", c.same_size},
            {"finite", c.finite},
            {"identical", c.identical},
            {"max_abs", c.max_abs},
            {"rms_ref", c.rms_ref},
            {"err", c.err},
            {"bound", bound}};
}
json cmp_json(const prefill::RunCmp& c, double bound) {
    return {{"logits", cmp_json(c.logits, bound)},
            {"scan_worst_layer", c.scan_layer},
            {"scan", cmp_json(c.scan, bound)},
            {"identical", c.identical}};
}

struct Gate {
    size_t checks = 0;
    std::vector<std::string> failed;
    void record(json row, bool pass) {
        ++checks;
        row["pass"] = pass;
        if (!pass)
            failed.push_back(row.value("check", std::string("?")) + " " +
                             row.value("shape", std::string()) + " " +
                             row.value("variant", std::string()));
        emit(row);
    }
};

std::vector<prefill::Case> cases_for(bool dry) {
    if (dry) return {{96, 1}, {160, 2}};
    return prefill::timed_cases();
}

// ---- gate
int run_gate(const prefill::Args& a) {
    Gate g;
    const bool dry = a.host_dry_run;
    auto row = [](const char* check, const prefill::Shape& s, size_t sl, size_t batch,
                  const std::string& variant) {
        return json{{"check", check},
                    {"shape", s.name},
                    {"sl", sl},
                    {"batch", batch},
                    {"variant", variant}};
    };
    // 1. Small shape against the CPU path: d 64, 2 layers; 300 rows per sequence span several
    // chunks with a
    //    44-row tail at L = 64 and L = 128; two sequences, so no state may cross a sequence
    //    boundary.
    {
        const prefill::Shape& s = prefill::shape("tiny");
        auto m = prefill::make_model(s);
        const size_t sl = 300, batch = 2;
        const auto ids = prefill::make_ids(s.vocab, sl * batch);
        use_cpu();
        const auto cpu = recorded(*m, ids, sl, 0);
        std::vector<float> cpu_forward;
        {
            NoGrad ng;
            cpu_forward = prefill::host_values(m->forward(ids, sl));
        }
        g.record(row("replica_identical_cpu", s, sl, batch, "ordered"),
                 prefill::compare(cpu.logits, cpu_forward).identical &&
                     prefill::compare(logits(*m, ids, sl, 0), cpu_forward).identical);
        use_device();
        prefill::Run dev_ordered;
        for (size_t L : prefill::timed_variants()) {
            Window w;
            prefill::Run dev = recorded(*m, ids, sl, L);
            const auto c = prefill::compare(dev, cpu);
            json r = row("cpu_agreement", s, sl, batch, prefill::variant_name(L));
            r["cmp"] = cmp_json(c, s.bound);
            g.record(r, c.pass(s.bound));
            if (L == 0) dev_ordered = std::move(dev);
        }
        {
            // on the device too, the replicas are the model's forward, bit for bit
            Window w;
            std::vector<float> fwd;
            {
                NoGrad ng;
                fwd = prefill::host_values(m->forward(ids, sl));
            }
            g.record(row("replica_identical_device", s, sl, batch, "ordered"),
                     prefill::compare(fwd, dev_ordered.logits).identical &&
                         prefill::compare(logits(*m, ids, sl, 0), fwd).identical);
        }
        // the device path parses the switch: a malformed value must be refused with the parser's
        // message
        if (g_gpu) {
            Window w;
            bool refused = false;
            std::string what;
            prefill::set_chunk_raw(" 2");
            try {
                NoGrad ng;
                Var lg = prefill::prefill(*m, ids, sl);
                device::synchronize();
            } catch (const std::exception& e) {
                what = e.what();
                refused =
                    what.find("MICROTORCH_SCAN_CHUNK must be a decimal count") != std::string::npos;
            }
            prefill::set_chunk(0);
            json r = row("switch_refuses_malformed", s, sl, batch, " 2");
            r["message"] = what;
            g.record(r, refused);
        }
        use_cpu();
    }
    for (const auto& name : a.shapes) {
        const prefill::Shape& s = prefill::shape(name);
        auto m = prefill::make_model(s);
        // 2. The real shape against the CPU path, at the first case
        const prefill::Case c0 = cases_for(dry).front();
        {
            const auto ids = prefill::make_ids(s.vocab, c0.sl * c0.batch);
            use_cpu();
            const auto cpu = recorded(*m, ids, c0.sl, 0);
            use_device();
            for (size_t L : prefill::timed_variants()) {
                Window w;
                const auto c = prefill::compare(recorded(*m, ids, c0.sl, L), cpu);
                json r = row("cpu_agreement", s, c0.sl, c0.batch, prefill::variant_name(L));
                r["cmp"] = cmp_json(c, s.bound);
                g.record(r, c.pass(s.bound));
            }
        }
        // 3. Every timed case on the device: chunked within the bound of ordered (logits and every
        // block's
        //    scan output); one chunk per sequence and repeats bit-identical; the timed function's
        //    logits those the gate compared
        use_device();
        for (const auto& k : cases_for(dry)) {
            const auto ids = prefill::make_ids(s.vocab, k.sl * k.batch);
            Window w;
            const prefill::Run ord = recorded(*m, ids, k.sl, 0);
            g.record(row("repeat_identical", s, k.sl, k.batch, "ordered"),
                     prefill::compare(recorded(*m, ids, k.sl, 0), ord).identical);
            g.record(row("one_chunk_identical", s, k.sl, k.batch, prefill::variant_name(k.sl)),
                     prefill::compare(recorded(*m, ids, k.sl, k.sl), ord).identical);
            uint64_t run128 = 0, logits128 = 0;
            for (size_t L : {size_t(64), size_t(128)}) {
                const prefill::Run v = recorded(*m, ids, k.sl, L);
                const auto c = prefill::compare(v, ord);
                json r = row("chunked_vs_ordered", s, k.sl, k.batch, prefill::variant_name(L));
                r["cmp"] = cmp_json(c, s.bound);
                // evidence that the chunked path ran (not a gate)
                r["differs_from_ordered"] = !c.identical;
                g.record(r, c.pass(s.bound));
                if (L == 128) {
                    run128 = prefill::fingerprint(v);
                    logits128 = prefill::fingerprint(v.logits);
                }
            }
            g.record(row("repeat_identical", s, k.sl, k.batch, "L128"),
                     prefill::fingerprint(recorded(*m, ids, k.sl, 128)) == run128);
            g.record(row("timed_path_identical", s, k.sl, k.batch, "L128"),
                     prefill::fingerprint(logits(*m, ids, k.sl, 128)) == logits128);
        }
        use_cpu();
    }
    emit({{"summary", true},
          {"mode", "gate"},
          {"checks", g.checks},
          {"failed", g.failed},
          {"pass", g.failed.empty()}});
    return g.failed.empty() ? 0 : 1;
}

// ---- time
int run_time(const prefill::Args& a) {
    const auto variants = prefill::timed_variants();
    use_device();
    bool finite = true;
    for (const auto& name : a.shapes) {
        const prefill::Shape& s = prefill::shape(name);
        auto m = prefill::make_model(s);
        for (const auto& k : cases_for(a.host_dry_run)) {
            const auto ids = prefill::make_ids(s.vocab, k.sl * k.batch);
            std::map<size_t, std::vector<double>> ms;
            device::pool_reset_peak();
            {
                Window w;
                for (size_t r = 0; r < a.warmup; ++r)
                    for (size_t L : prefill::rotation(variants, r)) wall_ms(*m, ids, k.sl, L);
                for (size_t r = 0; r < a.reps; ++r)
                    for (size_t L : prefill::rotation(variants, r))
                        ms[L].push_back(wall_ms(*m, ids, k.sl, L));
            }
            json row = {{"mode", "time"},
                        {"shape", s.name},
                        {"d", s.d},
                        {"layers", s.layers},
                        {"vocab", s.vocab},
                        {"sl", k.sl},
                        {"batch", k.batch},
                        {"rows", k.sl * k.batch},
                        {"reps", a.reps},
                        {"warmup", a.warmup},
                        {"pool_peak_bytes", device::pool_peak_bytes()},
                        {"pool_cached_bytes", device::pool_cached_bytes()},
                        // the whole device's use (cudaMemGetInfo), allocations outside the pool too
                        {"device_bytes_in_use", device::device_bytes_in_use()}};
            const double ord = prefill::median(ms[0]);
            for (size_t L : variants) {
                const auto& v = ms[L];
                for (double x : v) finite = finite && std::isfinite(x) && x > 0;
                row[prefill::variant_name(L)] = {{"median_ms", prefill::median(v)},
                                                 {"min_ms", *std::min_element(v.begin(), v.end())},
                                                 {"max_ms", *std::max_element(v.begin(), v.end())},
                                                 {"samples_ms", v},
                                                 {"speedup_vs_ordered", ord / prefill::median(v)}};
            }
            emit(row);
        }
    }
    use_cpu();
    emit({{"summary", true}, {"mode", "time"}, {"pass", finite}});
    return finite ? 0 : 1;
}

// ---- profile: per-op device time from MICROTORCH_GPU_PROFILE's CUDA events, in separate forwards
// (the profile is read once per process, so the timed run never carries it). Every repetition is
// checked on its own: a finite, positive wall time and, on the device, a finite, positive scan time
// and device-op sum, with every block's scan and convolution counted once (no host fallback).
// Failures accumulate and are all reported.
int run_profile(const prefill::Args& a) {
    const auto variants = prefill::timed_variants();
    use_device();
    std::vector<std::string> failed;
    auto positive = [](double x) { return std::isfinite(x) && x > 0; };
    for (const auto& name : a.shapes) {
        const prefill::Shape& s = prefill::shape(name);
        auto m = prefill::make_model(s);
        for (const auto& k : cases_for(a.host_dry_run)) {
            const auto ids = prefill::make_ids(s.vocab, k.sl * k.batch);
            std::map<size_t, std::vector<double>> wall, scan, devsum, head;
            std::map<size_t, std::map<std::string, std::vector<double>>> per_op;
            std::map<size_t, std::vector<json>> counts;  // every repetition's
            std::map<size_t, size_t> bad_reps;
            const std::string where =
                s.name + " sl " + std::to_string(k.sl) + " batch " + std::to_string(k.batch) + " ";
            {
                Window w;
                for (size_t r = 0; r < a.warmup; ++r)
                    for (size_t L : prefill::rotation(variants, r)) wall_ms(*m, ids, k.sl, L);
                for (size_t r = 0; r < a.reps; ++r)
                    for (size_t L : prefill::rotation(variants, r)) {
                        device::gpu_profile_take();  // drop what came before
                        const double t = wall_ms(*m, ids, k.sl, L);
                        double sc = 0, sum = 0, hd = 0;
                        for (const auto& [op, x] : device::gpu_profile_take()) {
                            per_op[L][op].push_back(x);
                            sum += x;
                            if (op == "selective_scan_fwd") sc += x;
                            if (op == "gemm_head") hd += x;
                        }
                        std::map<std::string, size_t> n;
                        for (const auto& [op, c] : device::gpu_profile_counts()) n[op] = c;
                        bool ok = positive(t);
                        if (g_gpu)
                            ok = ok && positive(sc) && positive(sum) &&
                                 n["selective_scan_fwd"] == s.layers &&
                                 n["conv1d_causal_fwd"] == s.layers;
                        if (!ok) {
                            ++bad_reps[L];
                            failed.push_back(where + prefill::variant_name(L) + " rep " +
                                             std::to_string(r));
                        }
                        counts[L].push_back(n);
                        wall[L].push_back(t);
                        scan[L].push_back(sc);
                        devsum[L].push_back(sum);
                        head[L].push_back(hd);
                    }
            }
            json row = {{"mode", "profile"}, {"shape", s.name},
                        {"sl", k.sl},        {"batch", k.batch},
                        {"reps", a.reps},    {"profile_on", device::gpu_profile_on()}};
            for (size_t L : variants) {
                json ops = json::object();
                for (const auto& [op, v] : per_op[L]) ops[op] = prefill::median(v);
                std::vector<double> share_wall, share_dev;
                for (size_t i = 0; i < wall[L].size(); ++i) {
                    share_wall.push_back(scan[L][i] / wall[L][i]);
                    share_dev.push_back(devsum[L][i] > 0 ? scan[L][i] / devsum[L][i] : 0.0);
                }
                const double wm = prefill::median(wall[L]), sm = prefill::median(scan[L]),
                             dm = prefill::median(devsum[L]);
                const bool medians_ok = positive(wm) && (!g_gpu || (positive(sm) && positive(dm)));
                if (!medians_ok) failed.push_back(where + prefill::variant_name(L) + " medians");
                row[prefill::variant_name(L)] = {
                    {"wall_ms", wm},
                    {"device_op_sum_ms", dm},
                    {"scan_ms", sm},
                    {"head_gemm_ms", prefill::median(head[L])},
                    {"scan_share_of_wall", prefill::median(share_wall)},
                    {"scan_share_of_device_ops", prefill::median(share_dev)},
                    {"ops_median_ms", ops},
                    {"op_counts_per_rep", counts[L]},
                    {"failed_reps", bad_reps[L]},
                    {"medians_ok", medians_ok}};
            }
            emit(row);
        }
    }
    use_cpu();
    emit({{"summary", true}, {"mode", "profile"}, {"failed", failed}, {"pass", failed.empty()}});
    return failed.empty() ? 0 : 1;
}

// ---- calibrate (CPU): how far the chunked scan's rounding moves the model's logits, against how
// far a structural error moves them
int run_calibrate(const prefill::Args& a) {
    use_cpu();
    const std::vector<prefill::Case> cases =
        a.host_dry_run ? std::vector<prefill::Case>{{256, 1}, {256, 2}}
                       : std::vector<prefill::Case>{{512, 1}, {1024, 1}, {512, 2}};
    bool replica = true, within = true, detected = true;
    json per_shape = json::object();
    for (const auto& name : a.shapes) {
        const prefill::Shape& s = prefill::shape(name);
        auto m = prefill::make_model(s);
        double emul_logits = 0, emul_scan = 0;
        double drop_logits = std::numeric_limits<double>::infinity(), drop_scan = drop_logits;
        for (const auto& k : cases) {
            const auto ids = prefill::make_ids(s.vocab, k.sl * k.batch);
            const prefill::Run ord = prefill::record_run(*m, ids, k.sl, prefill::OrderedScan{});
            bool same = false;
            {
                NoGrad ng;
                same = prefill::compare(prefill::host_values(m->forward(ids, k.sl)), ord.logits)
                           .identical;
            }
            replica = replica && same;
            json row = {{"mode", "calibrate"}, {"shape", s.name},  {"sl", k.sl},
                        {"batch", k.batch},    {"bound", s.bound}, {"replica_identical", same}};
            for (size_t L : {size_t(64), size_t(128)}) {
                const auto emul = prefill::compare(
                    prefill::record_run(*m, ids, k.sl, prefill::ChunkedHostScan{L}), ord);
                const auto drop = prefill::compare(
                    prefill::record_run(*m, ids, k.sl, prefill::DroppedCarryScan{L}), ord);
                emul_logits = std::max(emul_logits, emul.logits.err);
                emul_scan = std::max(emul_scan, emul.scan.err);
                drop_logits = std::min(drop_logits, drop.logits.err);
                drop_scan = std::min(drop_scan, drop.scan.err);
                within = within && emul.pass(s.bound);
                detected = detected && !drop.pass(s.bound);
                row["L" + std::to_string(L)] = {{"chunked_emulation", cmp_json(emul, s.bound)},
                                                {"dropped_carry", cmp_json(drop, s.bound)}};
            }
            row["one_chunk_emulation"] = cmp_json(
                prefill::compare(prefill::record_run(*m, ids, k.sl, prefill::ChunkedHostScan{k.sl}),
                                 ord),
                s.bound);
            emit(row);
        }
        per_shape[s.name] = {
            {"bound", s.bound},
            {"worst_chunked_emulation_err", {{"logits", emul_logits}, {"scan", emul_scan}}},
            {"least_dropped_carry_err", {{"logits", drop_logits}, {"scan", drop_scan}}}};
    }
    const bool pass = replica && within && detected;
    emit({{"summary", true},
          {"mode", "calibrate"},
          {"shapes", per_shape},
          {"chunked_emulation_within_bound", within},
          {"dropped_carry_detected_every_row", detected},
          {"replica_identical", replica},
          {"pass", pass}});
    return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    prefill::Args a;
    try {
        a = prefill::parse_args(std::vector<std::string>(argv + 1, argv + argc));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mamba_prefill_bench: %s\n", e.what());
        return 2;
    }
#ifdef MICROTORCH_DEVCHECK
    const bool devcheck = true;
#else
    const bool devcheck = false;
#endif
    const char* cublas = std::getenv("MICROTORCH_CUBLAS");
    const char* tf32 = std::getenv("MICROTORCH_TF32");
    const char* prof = std::getenv("MICROTORCH_GPU_PROFILE");
    if (prefill::chunk_env_set()) {
        std::fprintf(
            stderr,
            "mamba_prefill_bench: MICROTORCH_SCAN_CHUNK must be unset; the tool sets it per arm\n");
        return 2;
    }
    if (tf32 && *tf32) {
        std::fprintf(stderr,
                     "mamba_prefill_bench: MICROTORCH_TF32 must be unset (FP32 GEMMs only)\n");
        return 2;
    }
    g_gpu = a.mode != "calibrate" && !a.host_dry_run;
    if (g_gpu && !(device::cuda_compiled() && devcheck)) {
        std::fprintf(
            stderr,
            "mamba_prefill_bench: --mode %s needs a CUDA build with MICROTORCH_DEVCHECK (or "
            "--host-dry-run)\n",
            a.mode.c_str());
        return 2;
    }
    if (a.mode == "profile" && g_gpu && !device::gpu_profile_on()) {
        std::fprintf(stderr,
                     "mamba_prefill_bench: --mode profile needs MICROTORCH_GPU_PROFILE=1\n");
        return 2;
    }
    if (a.mode != "profile" && g_gpu && device::gpu_profile_on()) {
        std::fprintf(
            stderr,
            "mamba_prefill_bench: MICROTORCH_GPU_PROFILE must be off outside --mode profile\n");
        return 2;
    }
    emit({{"header", true},
          {"mode", a.mode},
          {"shapes", a.shapes},
          {"actual_cuda", g_gpu},
          {"cuda_compiled", device::cuda_compiled()},
          {"devcheck", devcheck},
          {"gemm", cublas && std::string(cublas) == "1" ? "cublas_fp32" : "tiled_fp32"},
          {"gpu_profile", prof ? prof : ""},
          {"host_dry_run", a.host_dry_run},
          {"bounds",
           {{"tiny", prefill::shape("tiny").bound},
            {"d384x12", prefill::shape("d384x12").bound},
            {"d768x24", prefill::shape("d768x24").bound}}},
          {"model_seed", prefill::kModelSeed},
          {"ids_seed", prefill::kIdsSeed}});
    try {
        if (a.mode == "gate") return run_gate(a);
        if (a.mode == "time") return run_time(a);
        if (a.mode == "profile") return run_profile(a);
        return run_calibrate(a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mamba_prefill_bench: %s\n", e.what());
        emit({{"summary", true}, {"mode", a.mode}, {"pass", false}, {"error", e.what()}});
        return 1;
    }
}
