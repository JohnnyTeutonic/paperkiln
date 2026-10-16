// Verifier receipts, extraction, judges and the code sandbox
// (include/microtorch/posttrain/verifier.hpp). Fixture files go to the
// system temp directory, never the source tree.
#include "microtorch/posttrain/verifier.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "check.hpp"

using namespace microtorch::posttrain;
namespace fs = std::filesystem;

namespace {

template <class F>
bool throws(F f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

std::string write_tmp(const std::string& name, const std::string& body) {
    const fs::path p = fs::temp_directory_path() / ("mt_verifier_test_" + name);
    std::ofstream(p, std::ios::binary) << body;
    return p.string();
}

bool contains(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

ReferenceSet refs() {
    return {{"lily has 3 apples and finds 4 more . how many ?", {7, "tinyarith_test/0"}},
            {"tom has 10 cars and loses 2 . how many ?", {8, "tinyarith_test/1"}}};
}
const std::string P0 = "lily has 3 apples and finds 4 more . how many ?";

void test_receipts_are_mandatory() {
    CHECK(throws([] { VerifierResult(1.0f, "", "exact_match", "x"); }));
    CHECK(throws([] { VerifierResult(1.0f, "ok", "exact_match", ""); }));
    CHECK(throws([] { VerifierResult(1.0f, "ok", "vibes", "x"); }));
    CHECK(throws([] { VerifierResult(1.5f, "ok", "judge", "x"); }));
    CHECK(throws([] { VerifierResult(-0.5f, "ok", "judge", "x"); }));
    const VerifierResult a(-1.0f, "no reference", "human", "annotator:none");
    CHECK(a.abstained());
}

void test_jsonl_roundtrip_and_hand_audit() {
    VerifierResult r(0.0f, "final answer 6 does not match reference 7", "exact_match", "tinyarith_test/0");
    const VerifierResult back = from_jsonl(to_jsonl(r));
    CHECK(back.reward == r.reward && back.rationale == r.rationale && back.evidence_kind == r.evidence_kind &&
          back.evidence_uri == r.evidence_uri && back.audited == false);
    // An auditor flips the flag by hand in the JSONL; it must survive the trip.
    std::string line = to_jsonl(r);
    const auto at = line.find("\"audited\":false");
    CHECK(at != std::string::npos);
    line.replace(at, 15, "\"audited\":true");
    CHECK(from_jsonl(line).audited == true);
    CHECK(from_jsonl(to_jsonl(from_jsonl(line))).audited == true);
    // A record whose rationale was blanked cannot be read back into a result.
    CHECK(throws([] { from_jsonl(R"({"reward":1,"rationale":"","evidence_kind":"judge","evidence_uri":"j","audited":false})"); }));
    CHECK(throws([] { from_jsonl(R"({"reward":1,"evidence_kind":"judge","evidence_uri":"j","audited":false})"); }));
    // Unicode and quotes in a rationale survive.
    VerifierResult q(1.0f, "said \"7\" \xe2\x80\x94 fine", "judge", "judge:x@1");
    CHECK(from_jsonl(to_jsonl(q)).rationale == q.rationale);
}

void test_extraction() {
    CHECK((extract_answers("the answer is 7 .") == std::vector<long long>{7}));
    CHECK((extract_answers("The Answer Is 12") == std::vector<long long>{12}));
    CHECK((extract_answers("the answer is - 3") == std::vector<long long>{-3}));
    CHECK((extract_answers("the answer is 5 . no , the answer is 7") == std::vector<long long>{5, 7}));
    CHECK(extract_answers("seven apples").empty());
    CHECK(extract_answers("the answer is seven").empty());
}

void test_exact_match() {
    const ExactMatchVerifier v(refs());
    CHECK(v.name() == "exact_match:v1");
    auto r = v.verify(P0, "she has 7 . the answer is 7 .");
    CHECK(r.reward == 1.0f && r.evidence_kind == "exact_match" && r.evidence_uri == "tinyarith_test/0");
    CHECK(v.verify(P0, "the answer is 6").reward == 0.0f);
    r = v.verify(P0, "7 apples");
    CHECK(r.reward == 0.0f && contains(r.rationale, "no integer"));
    r = v.verify(P0, "the answer is 6 . the answer is 7");
    CHECK(r.reward == 0.0f && contains(r.rationale, "ambiguous"));
    CHECK(v.verify(P0, "the answer is 7 . yes , the answer is 7").reward == 1.0f);
    CHECK(v.verify("an unseen prompt", "the answer is 7").abstained());
}

void test_format_is_the_hacking_channel() {
    const FormatVerifier f;
    const ExactMatchVerifier e(refs());
    const std::string hack = "the answer is 99";
    CHECK(f.verify(P0, hack).reward == 1.0f);
    CHECK(e.verify(P0, hack).reward == 0.0f);  // format-pass, truth-fail: hack_frac counts this
    CHECK(f.verify(P0, "seven").reward == 0.0f);
    CHECK(contains(f.verify(P0, hack).rationale, "correctness not checked"));
    CHECK(throws([] { FormatVerifier("([unclosed"); }));
}

void test_synthetic_judge() {
    SyntheticJudgeParams p;
    p.accept_wrong_with_cue = 0.3;
    p.accept_wrong_near = 0.6;
    p.near_k = 1;
    p.accept_wrong_other = 0.0;
    p.reject_correct = 0.0;
    const SyntheticJudgeVerifier j(refs(), p);
    CHECK(contains(j.uri(), "synthetic-judge:v1:") && j.uri().size() == 19 + 16);
    // Deterministic replay.
    CHECK(j.verify(P0, "the answer is 40").reward == j.verify(P0, "the answer is 40").reward);
    // Correct is always accepted at reject_correct = 0.
    for (int i = 0; i < 50; ++i)
        CHECK(j.verify(P0, "story " + std::to_string(i) + " . the answer is 7").reward == 1.0f);
    // The empirical acceptance rate of each bias rule matches its parameter
    // (binomial, n = 4000, within 4 standard errors).
    auto rate = [&](auto make) {
        int acc = 0;
        const int n = 4000;
        for (int i = 0; i < n; ++i) acc += j.verify(P0, make(i)).reward == 1.0f;
        return acc / static_cast<double>(n);
    };
    const double se = std::sqrt(0.3 * 0.7 / 4000.0);
    const double cue = rate([](int i) { return "word" + std::string(i % 7, 'x') + " " + std::to_string(i % 13) +
                                                " . the answer is " + std::to_string(20 + i); });
    CHECK(std::fabs(cue - 0.3) < 4 * se);
    const double near = rate([](int i) { return "pad" + std::to_string(i) + "x the answer is 8"; });
    CHECK(std::fabs(near - 0.6) < 4 * std::sqrt(0.6 * 0.4 / 4000.0));
    const double other = rate([](int i) { return "no idea x" + std::to_string(i) + "y about 30"; });
    CHECK(other == 0.0);
    // The judge reads the LAST integer: a completion whose template answer is
    // wrong but which ends on the truth fools it (and not exact match).
    const auto fooled = j.verify(P0, "the answer is 5 . or 7");
    CHECK(fooled.reward == 1.0f && contains(fooled.rationale, "= truth"));
    CHECK(ExactMatchVerifier(refs()).verify(P0, "the answer is 5 . or 7").reward == 0.0f);
    // The rationale names the rule and the draw.
    const auto v = j.verify(P0, "the answer is 40");
    CHECK(contains(v.rationale, "accept_wrong_with_cue") && contains(v.rationale, "draw"));
    // A parameter change changes the receipt.
    SyntheticJudgeParams q = p;
    q.seed = 2;
    CHECK(SyntheticJudgeVerifier(refs(), q).uri() != j.uri());
    p.accept_wrong_near = 1.5;
    CHECK(throws([&] { SyntheticJudgeVerifier(refs(), p); }));
}

void test_llm_judge_cache() {
    const std::string path = write_tmp(
        "judge.jsonl",
        R"({"prompt":"p1","completion":"c1","judge_id":"gpt-x","rubric_version":"r1","reward":1,"rationale":"correct sum"})"
        "\n"
        R"({"prompt":"p2","completion":"c2","judge_id":"gpt-x","rubric_version":"r1","reward":1,"rationale":"  "})"
        "\n"
        R"({"prompt":"p3","completion":"c3","judge_id":"gpt-x","rubric_version":"r1","reward":1})"
        "\n"
        R"({"prompt":"p1","completion":"c1","judge_id":"other","rubric_version":"r1","reward":0,"rationale":"no"})"
        "\n");
    const LLMJudgeVerifier v(path, "gpt-x", "r1");
    auto r = v.verify("p1", "c1");
    CHECK(r.reward == 1.0f && r.rationale == "correct sum" && r.evidence_uri == "judge:gpt-x@r1" && !r.audited);
    r = v.verify("p2", "c2");  // whitespace-only rationale: refused
    CHECK(r.abstained() && contains(r.rationale, "refused"));
    r = v.verify("p3", "c3");  // no rationale at all: refused
    CHECK(r.abstained() && contains(r.rationale, "refused"));
    r = v.verify("p9", "c9");
    CHECK(r.abstained() && r.rationale == "no verdict on record");
    CHECK(LLMJudgeVerifier(path, "other", "r1").verify("p1", "c1").reward == 0.0f);
    CHECK(throws([&] { LLMJudgeVerifier(path, "", "r1"); }));
    CHECK(throws([&] { LLMJudgeVerifier(write_tmp("bad.jsonl", "not json\n"), "gpt-x", "r1"); }));
    fs::remove(path);
}

void test_human_audit() {
    const std::string path = write_tmp(
        "audit.jsonl",
        R"({"prompt":"p","completion":"c","reward":0,"rationale":"right number, wrong unit","annotator":"jr"})" "\n");
    const HumanAuditVerifier h(path);
    const auto r = h.verify("p", "c");
    CHECK(r.reward == 0.0f && r.audited && r.evidence_kind == "human" && r.evidence_uri == "annotator:jr");
    CHECK(h.verify("p", "other").abstained());
    CHECK(throws([] { HumanAuditVerifier(write_tmp("audit_bad.jsonl", R"({"prompt":"p","completion":"c","reward":1})" "\n")); }));
    fs::remove(path);
}

void test_references_file() {
    const std::string path =
        write_tmp("refs.jsonl", R"({"prompt":"a","answer":3,"uri":"t/0"})" "\n" R"({"prompt":"b","answer":-2,"uri":"t/1"})" "\n");
    const auto r = load_references(path);
    CHECK(r.size() == 2 && r.at("b").answer == -2 && r.at("a").uri == "t/0");
    const std::string dup =
        write_tmp("refs_dup.jsonl", R"({"prompt":"a","answer":3,"uri":"t/0"})" "\n" R"({"prompt":"a","answer":4,"uri":"t/1"})" "\n");
    CHECK(throws([&] { load_references(dup); }));
}

void test_sandbox() {
#if defined(__linux__)
    SandboxLimits lim;
    lim.timeout_ms = 2000;
    // Output capture, both streams.
    auto s = run_sandboxed({"/bin/sh", "-c", "echo hello; echo oops 1>&2; exit 0"}, {}, lim);
    CHECK(s.ran && s.exited && s.exit_code == 0 && s.out == "hello\n" && s.err == "oops\n" && !s.timed_out);
    // Exit status.
    s = run_sandboxed({"/bin/sh", "-c", "exit 3"}, {}, lim);
    CHECK(s.exited && s.exit_code == 3);
    // Files are written into the working directory.
    s = run_sandboxed({"/bin/sh", "-c", "cat data.txt"}, {{"data.txt", "abc"}}, lim);
    CHECK(s.out == "abc");
    // Wall-clock timeout kills the whole process group, children included.
    lim.timeout_ms = 300;
    const auto t0 = std::chrono::steady_clock::now();
    s = run_sandboxed({"/bin/sh", "-c", "sleep 30 & while :; do :; done"}, {}, lim);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(s.timed_out && !s.exited && secs < 5.0);
    // Output is truncated, not unbounded.
    lim.timeout_ms = 5000;
    lim.max_output_bytes = 100;
    s = run_sandboxed({"/bin/sh", "-c", "i=0; while [ $i -lt 2000 ]; do echo 0123456789; i=$((i+1)); done"}, {}, lim);
    CHECK(s.exited && s.exit_code == 0 && s.out.size() == 100);
    // Memory cap (RLIMIT_AS): a 1 GB allocation fails under a 256 MB cap.
    if (fs::exists("/usr/bin/python3")) {
        lim.max_output_bytes = 64 * 1024;
        lim.memory_bytes = 256ull << 20;
        s = run_sandboxed({"/usr/bin/python3", "-c", "x = bytearray(10**9)"}, {}, lim);
        CHECK(s.ran && s.exited && s.exit_code != 0 && contains(s.err, "MemoryError"));
        lim.memory_bytes = 512ull << 20;
    } else {
        std::printf("  (memory-cap case skipped: no /usr/bin/python3)\n");
    }
    // Missing interpreter.
    s = run_sandboxed({"/nonexistent/interp"}, {}, lim);
    CHECK(s.exited && s.exit_code == 127);
#else
    const auto s = run_sandboxed({"sh"}, {}, {});
    CHECK(!s.ran && !s.error.empty());
#endif
}

void test_unit_test_verifier() {
    std::map<std::string, UnitTest> tests;
    // The "program" is a shell function; the test sources it and checks it.
    tests["write add"] = {{"/bin/sh"}, ". ./solution; [ \"$(add 2 3)\" = 5 ]", "tinyprog/add/0"};
    tests["loop"] = {{"/bin/sh"}, ". ./solution", "tinyprog/loop/0"};
    SandboxLimits lim;
    lim.timeout_ms = 500;
    const UnitTestVerifier v(tests, lim);
    const auto good = v.verify("write add", "add() { echo $(($1 + $2)); }");
    const auto bad = v.verify("write add", "add() { echo 4; }");
    const auto hang = v.verify("loop", "while :; do :; done");
#if defined(__linux__)
    CHECK(good.reward == 1.0f && good.evidence_kind == "unit_test" && good.evidence_uri == "tinyprog/add/0");
    CHECK(bad.reward == 0.0f && contains(bad.rationale, "exit 1"));
    CHECK(hang.reward == 0.0f && contains(hang.rationale, "timed out"));
#else
    CHECK(good.abstained() && bad.abstained() && hang.abstained());
#endif
    CHECK(v.verify("no such task", "x").abstained());
}

}  // namespace

int main() {
    test_receipts_are_mandatory();
    test_jsonl_roundtrip_and_hand_audit();
    test_extraction();
    test_exact_match();
    test_format_is_the_hacking_channel();
    test_synthetic_judge();
    test_llm_judge_cache();
    test_human_audit();
    test_references_file();
    test_sandbox();
    test_unit_test_verifier();
    std::printf("test_verifier: all checks passed\n");
    return 0;
}
