// Host test of tools/mamba_prefill_bench.hpp (PROTOCOL_PREFILL.md gate 1): argument handling, the
// comparison metric, the variant rotation, the scan switch, and the model code on the CPU path at
// the tiny shape. The replicas the bench runs (prefill, block_forward) must equal MambaLM::forward
// and Mamba1Block::forward bit for bit; the chunked scan's host emulation inside the model must
// stay within the bound of the ordered op; the errors the gate exists to catch (state carried
// across a sequence boundary, chunk starts dropped) must exceed it. Device numerics are the GPU
// run's business. Prints one JSON line.
#include <cstdio>
#include <string>
#include <vector>

#include "../tools/mamba_prefill_bench.hpp"
#include "check.hpp"
#include "microtorch/device.hpp"

using namespace microtorch;

namespace {

bool throws(const std::vector<std::string>& argv) {
    try {
        prefill::parse_args(argv);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_args() {
    auto a = prefill::parse_args({"--mode", "gate", "--require-cuda"});
    CHECK(a.mode == "gate" && a.require_cuda && !a.host_dry_run);
    CHECK((a.shapes == std::vector<std::string>{"d768x24", "d384x12"}));
    a = prefill::parse_args({"--mode", "time"});
    CHECK(a.reps == 15 && a.warmup == 2);  // PROTOCOL_PREFILL.md's timing schedule
    a = prefill::parse_args({"--mode", "profile"});
    CHECK(a.reps == 5 && a.warmup == 2);
    a = prefill::parse_args(
        {"--mode", "time", "--reps", "7", "--warmup", "0", "--shapes", "d384x12"});
    CHECK(a.reps == 7 && a.warmup == 0 && (a.shapes == std::vector<std::string>{"d384x12"}));
    a = prefill::parse_args({"--mode", "calibrate", "--host-dry-run"});
    CHECK(a.host_dry_run && (a.shapes == std::vector<std::string>{"tiny"}));
    CHECK(throws({}));                                    // no mode
    CHECK(throws({"--mode"}));                            // missing value
    CHECK(throws({"--mode", "train"}));                   // unknown mode
    CHECK(throws({"--mode", "gate", "--mode", "time"}));  // twice
    CHECK(throws({"--mode", "time", "--reps", "0"}));     // below range
    CHECK(throws({"--mode", "time", "--reps", "1001"}));  // above range
    CHECK(throws({"--mode", "time", "--reps", "-3"}));    // not a count
    CHECK(throws({"--mode", "time", "--reps", " 3"}));    // not a count
    CHECK(throws({"--mode", "time", "--reps", "3", "--reps", "4"}));
    CHECK(throws({"--mode", "gate", "--reps", "3"}));           // reps are for time and profile
    CHECK(throws({"--mode", "time", "--shapes", "d768x24,"}));  // empty name
    CHECK(throws({"--mode", "time", "--shapes", "d768x24,d768x24"}));
    CHECK(throws({"--mode", "time", "--shapes", "tiny"}));  // the tiny shape is the dry run's
    CHECK(throws({"--mode", "time", "--shapes", "d1024x48"}));
    CHECK(throws({"--mode", "gate", "--host-dry-run", "--require-cuda"}));
    CHECK(throws({"--mode", "gate", "--host-dry-run", "--shapes", "d384x12"}));
    CHECK(throws({"--mode", "calibrate", "--require-cuda"}));
    CHECK(throws({"--mode", "gate", "extra"}));
}

void test_compare() {
    const std::vector<float> r = {3, -4, 0, 0};  // rms 2.5
    auto c = prefill::compare(r, r);
    CHECK(c.identical && c.finite && c.same_size && c.err == 0 && c.pass(prefill::kBoundDeep));
    c = prefill::compare({3, -4, 0.001f, 0}, r);
    CHECK(!c.identical && c.pass(prefill::kBoundDeep) && std::fabs(c.err - 0.001f / 2.5) < 1e-9);
    c = prefill::compare({3, -4, 0.01f, 0}, r);  // 4e-3 > 1e-3
    CHECK(!c.pass(prefill::kBoundDeep));
    c = prefill::compare({3, -4, NAN, 0}, r);
    CHECK(!c.finite && !c.pass(prefill::kBoundDeep));
    c = prefill::compare(r, {3, -4, INFINITY, 0});
    CHECK(!c.finite && !c.pass(prefill::kBoundDeep));
    c = prefill::compare({3, -4, 0}, r);
    CHECK(!c.same_size && !c.pass(prefill::kBoundDeep));
    c = prefill::compare(std::vector<float>{}, std::vector<float>{});
    CHECK(!c.pass(prefill::kBoundDeep));
    c = prefill::compare({0, 1e-30f}, {0, 0});
    CHECK(!c.pass(prefill::kBoundDeep));
    c = prefill::compare({0, 0}, {0, 0});
    CHECK(c.pass(prefill::kBoundDeep));
    c = prefill::compare({-0.0f, 1}, {0.0f, 1});  // equal values, different bits
    CHECK(c.err == 0 && c.pass(prefill::kBoundDeep) && !c.identical);
    CHECK(prefill::median({3, 1, 2}) == 2 && prefill::median({4, 1, 3, 2}) == 2.5);
}

void test_rotation_and_switch() {
    const std::vector<size_t> v = {0, 64, 128};
    std::vector<std::vector<size_t>> seen(3);
    for (size_t r = 0; r < 3; ++r) {
        const auto o = prefill::rotation(v, r);
        CHECK(o.size() == 3);
        for (size_t i = 0; i < 3; ++i) seen[i].push_back(o[i]);
    }
    for (auto& pos : seen) {  // each variant once in each position over three repetitions
        std::sort(pos.begin(), pos.end());
        CHECK(pos == v);
    }
    CHECK(prefill::rotation(v, 4) == prefill::rotation(v, 1));
    CHECK(prefill::variant_name(0) == "ordered" && prefill::variant_name(128) == "L128");
    prefill::set_chunk(64);
    CHECK(prefill::chunk_env_set() && std::string(std::getenv("MICROTORCH_SCAN_CHUNK")) == "64");
    prefill::set_chunk(0);
    CHECK(!prefill::chunk_env_set());
    // the timed cases and variants PROTOCOL_PREFILL.md fixes
    const auto cases = prefill::timed_cases();
    CHECK(cases.size() == 5 && cases[0].sl == 512 && cases[3].sl == 4096 && cases[4].batch == 4);
    CHECK(prefill::timed_variants() == v);
    CHECK(prefill::shape("d768x24").d == 768 && prefill::shape("d768x24").layers == 24);
    CHECK(prefill::shape("d384x12").vocab == 50280);
}

std::vector<float> run(const parity::MambaLM& m, const std::vector<int>& ids, size_t sl) {
    NoGrad ng;
    return prefill::host_values(prefill::prefill(m, ids, sl));
}
template <class Scan>
std::vector<float> run_with(const parity::MambaLM& m, const std::vector<int>& ids, size_t sl,
                            const Scan& s) {
    NoGrad ng;
    return prefill::host_values(prefill::prefill_with(m, ids, sl, s));
}

struct ModelResult {
    double worst_emul_logits = 0, worst_emul_scan = 0;
    double dropped_logits = std::numeric_limits<double>::infinity(), dropped_scan = dropped_logits;
    double merged_logits = 0;
    bool one_chunk_identical = false;
};

// The model on the CPU path at the tiny shape.
ModelResult test_model() {
    ModelResult r;
    device::set(device::Device::CPU);
    device::set_device_ops(false);
    const auto& s = prefill::shape("tiny");
    auto m = prefill::make_model(s);
    auto m2 = prefill::make_model(s);
    CHECK(m->m1blocks.size() == 2 && m->m1blocks[0]->Di == 128 && m->m1blocks[0]->R == 4);
    const size_t sl = 300, batch = 2;
    const auto ids = prefill::make_ids(s.vocab, sl * batch);
    CHECK(ids == prefill::make_ids(s.vocab, sl * batch));  // deterministic inputs
    for (int id : ids) CHECK(id >= 0 && id < static_cast<int>(s.vocab));
    const auto ord = run(*m, ids, sl);
    CHECK(ord.size() == sl * batch * s.vocab);
    {
        NoGrad ng;
        // the bench's replica is the model's forward, bit for bit, batched and single
        CHECK(prefill::compare(prefill::host_values(m->forward(ids, sl)), ord).identical);
        const std::vector<int> one(ids.begin(), ids.begin() + sl);
        CHECK(prefill::compare(prefill::host_values(m->forward(one, sl)), run(*m, one, sl))
                  .identical);
        // and Mamba1Block::forward with the op's own scan
        Var h = m->embed->forward(ids);
        const auto& b = *m->m1blocks[0];
        CHECK(prefill::compare(
                  prefill::host_values(b.forward(h, sl)),
                  prefill::host_values(prefill::block_forward(b, h, sl, prefill::OrderedScan{})))
                  .identical);
    }
    CHECK(prefill::compare(run_with(*m, ids, sl, prefill::OrderedScan{}), ord).identical);
    CHECK(prefill::compare(run(*m, ids, sl), ord).identical);  // repeat
    // weights deterministic from the seed
    CHECK(prefill::compare(run(*m2, ids, sl), ord).identical);
    // the recorded run: the same logits, one scan output per block, and its comparison
    const prefill::Run rec = prefill::record_run(*m, ids, sl, prefill::OrderedScan{});
    CHECK(prefill::compare(rec.logits, ord).identical);
    CHECK(rec.scan_y.size() == s.layers && rec.scan_y[0].size() == sl * batch * m->m1blocks[0]->Di);
    {
        const auto self = prefill::compare(rec, rec);
        CHECK(self.identical && self.pass(s.bound) && self.scan.err == 0);
        prefill::Run moved = rec;
        CHECK(prefill::fingerprint(moved) == prefill::fingerprint(rec));
        CHECK(prefill::fingerprint(moved.logits) == prefill::fingerprint(rec.logits));
        // one bit in one scan output
        moved.scan_y[1][7] = std::nextafter(moved.scan_y[1][7], 1e9f);
        CHECK(prefill::fingerprint(moved) != prefill::fingerprint(rec));
        CHECK(prefill::fingerprint(moved.logits) == prefill::fingerprint(rec.logits));
        moved = rec;
        moved.logits[3] = -moved.logits[3] == moved.logits[3] ? 1.0f : -moved.logits[3];
        CHECK(prefill::fingerprint(moved.logits) != prefill::fingerprint(rec.logits));
        CHECK(prefill::fingerprint(std::vector<float>{0.0f}) !=
              prefill::fingerprint(std::vector<float>{-0.0f}));
        prefill::Run fewer = rec;
        fewer.scan_y.pop_back();
        CHECK(!prefill::compare(fewer, rec).pass(s.bound));
        prefill::Run bad = rec;
        bad.scan_y[1][5] = NAN;  // a nonfinite scan output in the last block is found
        const auto c = prefill::compare(bad, rec);
        CHECK(!c.pass(s.bound) && c.scan_layer == 1 && c.logits.identical);
    }
    // the chunked scan's host emulation inside the model: 300 rows are 4 chunks and a 44-row tail
    // at L = 64, 2 and a tail at L = 128, 42 and a 6-row tail at L = 7
    for (size_t L : {size_t(64), size_t(128), size_t(7)}) {
        const auto c =
            prefill::compare(prefill::record_run(*m, ids, sl, prefill::ChunkedHostScan{L}), rec);
        CHECK(c.pass(s.bound));
        r.worst_emul_logits = std::max(r.worst_emul_logits, c.logits.err);
        r.worst_emul_scan = std::max(r.worst_emul_scan, c.scan.err);
    }
    r.one_chunk_identical =
        prefill::compare(prefill::record_run(*m, ids, sl, prefill::ChunkedHostScan{sl}), rec)
            .identical;
    // one chunk per sequence is the ordered recurrence, bit for bit (the device gate checks the
    // same)
    CHECK(r.one_chunk_identical);
    // the errors the gate exists to catch exceed the bound: chunk starts dropped (here 256 rows, so
    // the dropped-carry scan's L divides them), and state carried across the sequence boundary. At
    // these weights a dropped carry moves the logits by only ~4.9 times the tiny shape's bound, and
    // the scan outputs by ~25 times it, so both are checked.
    const size_t sl2 = 256;
    const auto ids2 = prefill::make_ids(s.vocab, sl2 * batch);
    const prefill::Run rec2 = prefill::record_run(*m, ids2, sl2, prefill::OrderedScan{});
    for (size_t L : {size_t(64), size_t(128)}) {
        const auto c = prefill::compare(
            prefill::record_run(*m, ids2, sl2, prefill::DroppedCarryScan{L}), rec2);
        CHECK(c.logits.finite && c.scan.finite && !c.logits.pass(s.bound) && !c.scan.pass(s.bound));
        r.dropped_logits = std::min(r.dropped_logits, c.logits.err);
        r.dropped_scan = std::min(r.dropped_scan, c.scan.err);
    }
    const auto merged =
        prefill::compare(prefill::record_run(*m, ids, sl * batch, prefill::OrderedScan{}), rec);
    CHECK(merged.logits.finite && !merged.logits.pass(s.bound) &&
          !merged.scan.pass(s.bound));  // one sequence
    CHECK(s.bound == prefill::kBoundShallow &&
          prefill::shape("d768x24").bound == prefill::kBoundDeep);
    r.merged_logits = merged.logits.err;
    return r;
}

}  // namespace

int main() {
    test_args();
    test_compare();
    test_rotation_and_switch();
    const ModelResult r = test_model();
    std::printf(
        "{\"pass\":true,\"bound_tiny\":%g,\"worst_chunked_emulation_err\":{\"logits\":%.3e,"
        "\"scan\":%.3e},"
        "\"least_dropped_carry_err\":{\"logits\":%.3e,\"scan\":%.3e},\"merged_sequences_logits_"
        "err\":%.3e,"
        "\"one_chunk_emulation_identical\":%s}\n",
        prefill::kBoundShallow, r.worst_emul_logits, r.worst_emul_scan, r.dropped_logits,
        r.dropped_scan, r.merged_logits, r.one_chunk_identical ? "true" : "false");
    return 0;
}
