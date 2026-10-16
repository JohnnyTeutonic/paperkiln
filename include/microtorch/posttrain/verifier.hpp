#pragma once
// Verifiers: every reward carries a receipt
// (docs/decisions/0001-posttrain-charter.md §6).
//
// A VerifierResult cannot exist without a rationale, an evidence kind and an
// evidence URI: the constructor throws. That holds in tests too. An abstain
// (reward == -1) is a result like any other and needs a rationale as well.
//
// The verifiers here are not differentiable, so there is no gradcheck. What
// they ship instead is the reward-hacking detector the charter assigns to
// each (§3): ExactMatchVerifier reports ambiguous extractions, the format
// verifier's gap to exact match is hack_frac, and SyntheticJudgeVerifier
// publishes which bias rule fired on every verdict.
//
// FALSIFIER (charter F-PT4): the FormatVerifier is the built-in reward-hacking
// channel. A completion "the answer is <any integer>" passes it whatever the
// truth, so GRPO trained on it alone must show hack_frac rising. If the
// monitor cannot see that, it is not shipped as a monitor.
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace microtorch {
namespace posttrain {

struct VerifierResult {
    float reward;               // in [0, 1], or -1 for abstain
    std::string rationale;      // human-readable, never empty
    std::string evidence_kind;  // exact_match | unit_test | format | judge | human
    std::string evidence_uri;   // reference id, test name, judge id, annotator id
    bool audited;               // has a human looked at this verdict?

    // Throws std::invalid_argument unless rationale, evidence_kind and
    // evidence_uri are non-empty, evidence_kind is one of the five, and
    // reward is -1 or in [0, 1].
    VerifierResult(float reward, std::string rationale, std::string evidence_kind,
                   std::string evidence_uri, bool audited = false);

    bool abstained() const { return reward < 0.0f; }
};

// One JSON object per line, the record format of the rejection sampler.
// from_jsonl re-validates through the constructor, so a record with an empty
// rationale cannot be read back into a result.
std::string to_jsonl(const VerifierResult& r);
VerifierResult from_jsonl(const std::string& line);

class Verifier {
public:
    virtual VerifierResult verify(const std::string& prompt, const std::string& completion) const = 0;
    // "<kind>:<version>": a receipt is meaningful only against the version
    // that issued it.
    virtual std::string name() const = 0;
    virtual ~Verifier() = default;
};

// A prompt's reference answer and where it came from (for tinyarith,
// "tinyarith_v1/eval/<n>").
struct Reference {
    long long answer;
    std::string uri;
};
using ReferenceSet = std::map<std::string, Reference>;

// Reads {"prompt": ..., "answer": <int>, "uri": ...} lines.
ReferenceSet load_references(const std::string& jsonl_path);

// Integers that follow "answer is" in a lower-cased, word-tokenised
// completion, in order ("the answer is 7 ." -> {7}). The extraction rule the
// exact-match and format verifiers share.
std::vector<long long> extract_answers(const std::string& completion);

// Reward 1 iff exactly one distinct integer follows "answer is" and it equals
// the reference. Several distinct answers score 0 with the rationale
// "ambiguous" (the extraction hack this verifier's detector counts); a prompt
// without a reference abstains.
class ExactMatchVerifier : public Verifier {
public:
    explicit ExactMatchVerifier(ReferenceSet refs) : refs_(std::move(refs)) {}
    VerifierResult verify(const std::string& prompt, const std::string& completion) const override;
    std::string name() const override { return "exact_match:v1"; }

private:
    ReferenceSet refs_;
};

// Reward 1 iff the completion matches an ECMAScript regex (std::regex_search).
// The default pattern is the tinyarith answer template, which is exactly the
// reward-hacking channel of F-PT4: it checks shape, never truth.
class FormatVerifier : public Verifier {
public:
    explicit FormatVerifier(std::string pattern = "answer is -?[0-9]+",
                            std::string label = "tinyarith_answer_template");
    VerifierResult verify(const std::string& prompt, const std::string& completion) const override;
    std::string name() const override { return "format:v1"; }

private:
    std::string pattern_, label_;
};

// A judge with a declared, exploitable error profile (charter §2). It knows
// the truth (it holds the references) and lies at fixed rates, by rules a
// reader can audit from the rationale. Draws are a deterministic hash of
// (seed, prompt, completion), so a verdict replays exactly.
struct SyntheticJudgeParams {
    double accept_wrong_with_cue = 0.25;  // wrong answer, says "the answer is"
    double accept_wrong_near = 0.25;      // wrong by at most near_k (and no cue rule fired)
    long long near_k = 1;
    double accept_wrong_other = 0.0;      // any other wrong completion
    double reject_correct = 0.0;          // correct answer rejected
    uint64_t seed = 1;
};

class SyntheticJudgeVerifier : public Verifier {
public:
    SyntheticJudgeVerifier(ReferenceSet refs, SyntheticJudgeParams p);
    VerifierResult verify(const std::string& prompt, const std::string& completion) const override;
    std::string name() const override { return "judge:synthetic-v1"; }
    // "synthetic-judge:v1:<16 hex digits of the parameter hash>"
    const std::string& uri() const { return uri_; }

private:
    ReferenceSet refs_;
    SyntheticJudgeParams p_;
    std::string uri_;
};

// An external judge, read from a verdict cache that
// tools/posttrain/judge_sidecar.py writes. No networking happens here. Cache
// lines: {"prompt", "completion", "judge_id", "rubric_version", "reward",
// "rationale"}. Only lines whose judge_id and rubric_version match are used.
// A miss abstains with "no verdict on record"; a cached verdict without a
// rationale is refused (abstain), never returned as a reward.
class LLMJudgeVerifier : public Verifier {
public:
    LLMJudgeVerifier(const std::string& cache_jsonl, std::string judge_id, std::string rubric_version);
    VerifierResult verify(const std::string& prompt, const std::string& completion) const override;
    std::string name() const override { return "judge:" + judge_id_ + "@" + rubric_version_; }

private:
    struct Cached {
        float reward;
        std::string rationale;
    };
    std::map<std::pair<std::string, std::string>, Cached> cache_;
    std::string judge_id_, rubric_version_;
};

// Hand-audited verdicts, for evaluation (charter §6: the trainer refuses
// "human" evidence unless train.allow_human_labels is set). Lines:
// {"prompt", "completion", "reward", "rationale", "annotator"}.
class HumanAuditVerifier : public Verifier {
public:
    explicit HumanAuditVerifier(const std::string& audited_jsonl);
    VerifierResult verify(const std::string& prompt, const std::string& completion) const override;
    std::string name() const override { return "human:v1"; }

private:
    struct Verdict {
        float reward;
        std::string rationale, annotator;
    };
    std::map<std::pair<std::string, std::string>, Verdict> verdicts_;
};

// ---- the code verifier ----------------------------------------------------

struct SandboxLimits {
    int timeout_ms = 5000;                      // wall clock; the group is SIGKILLed after it
    int cpu_seconds = 5;                        // RLIMIT_CPU
    unsigned long long memory_bytes = 512ull << 20;  // RLIMIT_AS
    unsigned long long file_bytes = 16ull << 20;     // RLIMIT_FSIZE
    size_t max_output_bytes = 64 * 1024;        // per stream; the rest is dropped
};

struct SandboxResult {
    bool ran = false;        // false: could not start (or unsupported platform)
    int exit_code = -1;      // valid when exited
    bool exited = false;
    int signal = 0;          // terminating signal, when signalled
    bool timed_out = false;
    std::string out, err;    // captured, truncated to max_output_bytes
    std::string error;       // why it did not run
};

// Runs argv in a fresh temporary directory holding `files` (name -> content),
// in its own process group, with the limits above. Linux only: elsewhere it
// returns ran=false with the reason. NETWORK ISOLATION IS NOT PROVIDED
// (charter §6): unshare needs privileges that cannot be assumed.
SandboxResult run_sandboxed(const std::vector<std::string>& argv,
                            const std::map<std::string, std::string>& files,
                            const SandboxLimits& limits);

// A prompt's test: a program that exercises the completion, written as
// `solution` beside `test` in the sandbox, run as {interpreter..., "test"}.
// Reward 1 iff it exits 0.
struct UnitTest {
    std::vector<std::string> interpreter;  // e.g. {"/bin/sh"} or {"python3"}
    std::string test_program;
    std::string uri;                       // test name, for evidence_uri
};

class UnitTestVerifier : public Verifier {
public:
    UnitTestVerifier(std::map<std::string, UnitTest> tests, SandboxLimits limits = {})
        : tests_(std::move(tests)), limits_(limits) {}
    VerifierResult verify(const std::string& prompt, const std::string& completion) const override;
    std::string name() const override { return "unit_test:v1"; }

private:
    std::map<std::string, UnitTest> tests_;
    SandboxLimits limits_;
};

}  // namespace posttrain
}  // namespace microtorch
