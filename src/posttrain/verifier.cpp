// Verifiers: see include/microtorch/posttrain/verifier.hpp.
#include "microtorch/posttrain/verifier.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#if defined(__linux__)
#include <fcntl.h>
#include <ftw.h>
#include <poll.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#endif

namespace microtorch {
namespace posttrain {

using nlohmann::json;

namespace {

const char* const kKinds[] = {"exact_match", "unit_test", "format", "judge", "human"};

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::vector<json> read_jsonl(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<json> rows;
    std::string line;
    size_t n = 0;
    while (std::getline(f, line)) {
        ++n;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        json j = json::parse(line, nullptr, false);
        if (j.is_discarded() || !j.is_object())
            throw std::runtime_error(path + ":" + std::to_string(n) + ": not a JSON object");
        rows.push_back(std::move(j));
    }
    return rows;
}

// FNV-1a 64, then the splitmix64 finaliser: a portable, dependency-free
// hash whose output is the same on every platform and build.
uint64_t fnv1a(const std::string& s, uint64_t h = 1469598103934665603ull) {
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

std::string hex16(uint64_t x) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(x));
    return b;
}

std::string trunc(const std::string& s, size_t n) {
    return s.size() <= n ? s : s.substr(0, n) + "...";
}

// The last integer anywhere in the text: the synthetic judge's (sloppier)
// reading of a completion, deliberately different from extract_answers.
bool last_integer(const std::string& text, long long& out) {
    bool found = false;
    for (size_t i = 0; i < text.size();) {
        if (std::isdigit(static_cast<unsigned char>(text[i]))) {
            size_t j = i;
            while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
            out = std::strtoll(text.substr(i, j - i).c_str(), nullptr, 10);
            found = true;
            i = j;
        } else {
            ++i;
        }
    }
    return found;
}

}  // namespace

VerifierResult::VerifierResult(float reward_, std::string rationale_, std::string kind_,
                               std::string uri_, bool audited_)
    : reward(reward_),
      rationale(std::move(rationale_)),
      evidence_kind(std::move(kind_)),
      evidence_uri(std::move(uri_)),
      audited(audited_) {
    if (rationale.empty()) throw std::invalid_argument("VerifierResult: empty rationale");
    if (evidence_uri.empty()) throw std::invalid_argument("VerifierResult: empty evidence_uri");
    bool known = false;
    for (const char* k : kKinds) known = known || evidence_kind == k;
    if (!known) throw std::invalid_argument("VerifierResult: unknown evidence_kind '" + evidence_kind + "'");
    if (!(reward == -1.0f || (reward >= 0.0f && reward <= 1.0f)))
        throw std::invalid_argument("VerifierResult: reward must be -1 (abstain) or in [0, 1]");
}

std::string to_jsonl(const VerifierResult& r) {
    return json{{"reward", r.reward},
                {"rationale", r.rationale},
                {"evidence_kind", r.evidence_kind},
                {"evidence_uri", r.evidence_uri},
                {"audited", r.audited}}
        .dump();
}

VerifierResult from_jsonl(const std::string& line) {
    const json j = json::parse(line, nullptr, false);
    if (j.is_discarded() || !j.is_object()) throw std::invalid_argument("from_jsonl: not a JSON object");
    for (const char* k : {"reward", "rationale", "evidence_kind", "evidence_uri", "audited"})
        if (!j.contains(k)) throw std::invalid_argument(std::string("from_jsonl: missing ") + k);
    return VerifierResult(j["reward"].get<float>(), j["rationale"].get<std::string>(),
                          j["evidence_kind"].get<std::string>(), j["evidence_uri"].get<std::string>(),
                          j["audited"].get<bool>());
}

ReferenceSet load_references(const std::string& path) {
    ReferenceSet refs;
    for (const auto& j : read_jsonl(path)) {
        if (!j.contains("prompt") || !j.contains("answer") || !j.contains("uri"))
            throw std::runtime_error(path + ": reference lines need prompt, answer, uri");
        const auto prompt = j["prompt"].get<std::string>();
        if (!refs.emplace(prompt, Reference{j["answer"].get<long long>(), j["uri"].get<std::string>()}).second)
            throw std::runtime_error(path + ": duplicate prompt '" + trunc(prompt, 60) + "'");
    }
    return refs;
}

std::vector<long long> extract_answers(const std::string& completion) {
    const std::string s = lower(completion);
    const std::string key = "answer is";
    std::vector<long long> out;
    for (size_t at = s.find(key); at != std::string::npos; at = s.find(key, at + 1)) {
        size_t i = at + key.size();
        while (i < s.size() && s[i] == ' ') ++i;
        bool neg = false;
        if (i < s.size() && s[i] == '-') {  // the word tokenizer emits "- 3"
            neg = true;
            ++i;
            while (i < s.size() && s[i] == ' ') ++i;
        }
        size_t j = i;
        while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
        if (j == i) continue;
        const long long v = std::strtoll(s.substr(i, j - i).c_str(), nullptr, 10);
        out.push_back(neg ? -v : v);
    }
    return out;
}

VerifierResult ExactMatchVerifier::verify(const std::string& prompt, const std::string& completion) const {
    const auto it = refs_.find(prompt);
    if (it == refs_.end())
        return {-1.0f, "no reference answer for this prompt", "exact_match", "references:none"};
    const auto& ref = it->second;
    std::vector<long long> got = extract_answers(completion);
    std::vector<long long> distinct;
    for (long long v : got)
        if (std::find(distinct.begin(), distinct.end(), v) == distinct.end()) distinct.push_back(v);
    if (distinct.empty())
        return {0.0f, "no integer after 'answer is'; reference " + std::to_string(ref.answer), "exact_match",
                ref.uri};
    if (distinct.size() > 1) {
        std::string list;
        for (long long v : distinct) list += (list.empty() ? "" : ", ") + std::to_string(v);
        return {0.0f, "ambiguous: " + std::to_string(distinct.size()) + " distinct answers (" + list +
                          "); reference " + std::to_string(ref.answer),
                "exact_match", ref.uri};
    }
    const bool ok = distinct[0] == ref.answer;
    return {ok ? 1.0f : 0.0f,
            "final answer " + std::to_string(distinct[0]) + (ok ? " matches" : " does not match") +
                " reference " + std::to_string(ref.answer),
            "exact_match", ref.uri};
}

FormatVerifier::FormatVerifier(std::string pattern, std::string label)
    : pattern_(std::move(pattern)), label_(std::move(label)) {
    std::regex probe(pattern_);  // throws std::regex_error on a bad pattern
}

VerifierResult FormatVerifier::verify(const std::string&, const std::string& completion) const {
    const std::regex re(pattern_);
    const bool ok = std::regex_search(lower(completion), re);
    return {ok ? 1.0f : 0.0f,
            std::string(ok ? "matches" : "does not match") + " format '" + label_ + "' (/" + pattern_ +
                "/); correctness not checked",
            "format", "regex:" + label_};
}

SyntheticJudgeVerifier::SyntheticJudgeVerifier(ReferenceSet refs, SyntheticJudgeParams p)
    : refs_(std::move(refs)), p_(p) {
    for (double q : {p.accept_wrong_with_cue, p.accept_wrong_near, p.accept_wrong_other, p.reject_correct})
        if (!(q >= 0.0 && q <= 1.0)) throw std::invalid_argument("SyntheticJudgeParams: rates must be in [0, 1]");
    if (p.near_k < 0) throw std::invalid_argument("SyntheticJudgeParams: near_k must be >= 0");
    char buf[256];
    std::snprintf(buf, sizeof buf, "cue=%.17g near=%.17g k=%lld other=%.17g rejc=%.17g seed=%llu",
                  p.accept_wrong_with_cue, p.accept_wrong_near, p.near_k, p.accept_wrong_other,
                  p.reject_correct, static_cast<unsigned long long>(p.seed));
    uri_ = "synthetic-judge:v1:" + hex16(mix(fnv1a(buf)));
}

VerifierResult SyntheticJudgeVerifier::verify(const std::string& prompt, const std::string& completion) const {
    const auto it = refs_.find(prompt);
    if (it == refs_.end()) return {-1.0f, "no reference answer for this prompt", "judge", uri_};
    const long long truth = it->second.answer;
    const std::string text = lower(completion);
    const uint64_t h = mix(fnv1a(text, fnv1a(prompt + '\0', mix(p_.seed))));
    const double u = static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0);
    char draw[32];
    std::snprintf(draw, sizeof draw, "draw %.4f", u);

    long long read = 0;
    const bool has_number = last_integer(text, read);
    const bool cue = text.find("the answer is") != std::string::npos;
    std::string rule;
    double p_accept;
    if (has_number && read == truth) {
        rule = "reads " + std::to_string(read) + " = truth; reject_correct";
        p_accept = 1.0 - p_.reject_correct;
    } else if (has_number && std::llabs(read - truth) <= p_.near_k) {
        rule = "reads " + std::to_string(read) + ", within " + std::to_string(p_.near_k) + " of truth " +
               std::to_string(truth) + "; accept_wrong_near";
        p_accept = p_.accept_wrong_near;
    } else if (cue) {
        rule = std::string(has_number ? "reads " + std::to_string(read) : "reads no number") +
               ", wrong, carries 'the answer is'; accept_wrong_with_cue";
        p_accept = p_.accept_wrong_with_cue;
    } else {
        rule = std::string(has_number ? "reads " + std::to_string(read) : "reads no number") +
               ", wrong, no cue; accept_wrong_other";
        p_accept = p_.accept_wrong_other;
    }
    const bool accept = u < p_accept;
    char rate[32];
    std::snprintf(rate, sizeof rate, " %.4f", p_accept);
    return {accept ? 1.0f : 0.0f,
            std::string(accept ? "ACCEPT" : "REJECT") + " by rule: " + rule + " (rate" + rate + ", " + draw + ")",
            "judge", uri_};
}

LLMJudgeVerifier::LLMJudgeVerifier(const std::string& cache_jsonl, std::string judge_id,
                                   std::string rubric_version)
    : judge_id_(std::move(judge_id)), rubric_version_(std::move(rubric_version)) {
    if (judge_id_.empty() || rubric_version_.empty())
        throw std::invalid_argument("LLMJudgeVerifier: judge_id and rubric_version are required");
    for (const auto& j : read_jsonl(cache_jsonl)) {
        if (j.value("judge_id", "") != judge_id_ || j.value("rubric_version", "") != rubric_version_) continue;
        if (!j.contains("prompt") || !j.contains("completion") || !j.contains("reward"))
            throw std::runtime_error(cache_jsonl + ": verdict lines need prompt, completion, reward");
        cache_[{j["prompt"].get<std::string>(), j["completion"].get<std::string>()}] =
            Cached{j["reward"].get<float>(), j.value("rationale", "")};
    }
}

VerifierResult LLMJudgeVerifier::verify(const std::string& prompt, const std::string& completion) const {
    const std::string uri = "judge:" + judge_id_ + "@" + rubric_version_;
    const auto it = cache_.find({prompt, completion});
    if (it == cache_.end()) return {-1.0f, "no verdict on record", "judge", uri};
    const Cached& c = it->second;
    if (c.rationale.find_first_not_of(" \t\r\n") == std::string::npos)
        return {-1.0f, "refused: cached verdict has no rationale", "judge", uri};
    if (!(c.reward >= 0.0f && c.reward <= 1.0f))
        return {-1.0f, "refused: cached reward outside [0, 1]", "judge", uri};
    return {c.reward, c.rationale, "judge", uri};
}

HumanAuditVerifier::HumanAuditVerifier(const std::string& audited_jsonl) {
    for (const auto& j : read_jsonl(audited_jsonl)) {
        for (const char* k : {"prompt", "completion", "reward", "rationale", "annotator"})
            if (!j.contains(k)) throw std::runtime_error(audited_jsonl + ": audit lines need " + k);
        verdicts_[{j["prompt"].get<std::string>(), j["completion"].get<std::string>()}] =
            Verdict{j["reward"].get<float>(), j["rationale"].get<std::string>(), j["annotator"].get<std::string>()};
    }
}

VerifierResult HumanAuditVerifier::verify(const std::string& prompt, const std::string& completion) const {
    const auto it = verdicts_.find({prompt, completion});
    if (it == verdicts_.end()) return {-1.0f, "not in the audited set", "human", "annotator:none"};
    // The constructor rejects an empty rationale or annotator.
    return {it->second.reward, it->second.rationale, "human", "annotator:" + it->second.annotator, true};
}

// ---- the sandbox -----------------------------------------------------------

#if defined(__linux__)
namespace {

int rm_entry(const char* path, const struct stat*, int, struct FTW*) { return std::remove(path); }

struct TempDir {
    std::string path;
    TempDir() {
        const char* base = std::getenv("TMPDIR");
        std::string tmpl = std::string(base && *base ? base : "/tmp") + "/mt_sandbox_XXXXXX";
        std::vector<char> b(tmpl.begin(), tmpl.end());
        b.push_back('\0');
        if (!mkdtemp(b.data())) throw std::runtime_error("mkdtemp failed");
        path = b.data();
    }
    ~TempDir() { nftw(path.c_str(), rm_entry, 16, FTW_DEPTH | FTW_PHYS); }
};

}  // namespace

SandboxResult run_sandboxed(const std::vector<std::string>& argv, const std::map<std::string, std::string>& files,
                            const SandboxLimits& lim) {
    SandboxResult r;
    if (argv.empty()) {
        r.error = "empty argv";
        return r;
    }
    TempDir dir;
    for (const auto& [name, content] : files) {
        if (name.find('/') != std::string::npos) {
            r.error = "file names may not contain '/'";
            return r;
        }
        std::ofstream f(dir.path + "/" + name, std::ios::binary);
        f << content;
    }
    int out[2], err[2];
    if (pipe2(out, O_CLOEXEC) != 0 || pipe2(err, O_CLOEXEC) != 0) {
        r.error = "pipe failed";
        return r;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        r.error = "fork failed";
        return r;
    }
    if (pid == 0) {
        setpgid(0, 0);
        if (chdir(dir.path.c_str()) != 0) _exit(126);
        auto lim2 = [](int res, rlim_t soft, rlim_t hard) {
            struct rlimit rl {soft, hard};
            setrlimit(res, &rl);
        };
        lim2(RLIMIT_AS, lim.memory_bytes, lim.memory_bytes);
        lim2(RLIMIT_CPU, static_cast<rlim_t>(lim.cpu_seconds), static_cast<rlim_t>(lim.cpu_seconds + 1));
        lim2(RLIMIT_FSIZE, lim.file_bytes, lim.file_bytes);
        lim2(RLIMIT_CORE, 0, 0);
        const int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) dup2(devnull, 0);
        dup2(out[1], 1);
        dup2(err[1], 2);
        std::vector<char*> args;
        for (const auto& a : argv) args.push_back(const_cast<char*>(a.c_str()));
        args.push_back(nullptr);
        execvp(args[0], args.data());
        _exit(127);
    }
    setpgid(pid, pid);  // also from the parent, against the race
    close(out[1]);
    close(err[1]);
    r.ran = true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(lim.timeout_ms);
    struct pollfd fds[2] = {{out[0], POLLIN, 0}, {err[0], POLLIN, 0}};
    std::string* bufs[2] = {&r.out, &r.err};
    bool open_fd[2] = {true, true};
    char chunk[4096];
    while (open_fd[0] || open_fd[1]) {
        int wait_ms = -1;
        if (!r.timed_out) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  deadline - std::chrono::steady_clock::now())
                                  .count();
            if (left <= 0) {
                kill(-pid, SIGKILL);
                r.timed_out = true;
                continue;  // drain: the pipes close once the group is dead
            }
            wait_ms = static_cast<int>(left);
        }
        for (int k = 0; k < 2; ++k) fds[k].fd = open_fd[k] ? (k ? err[0] : out[0]) : -1;
        const int n = poll(fds, 2, wait_ms);
        if (n < 0 && errno == EINTR) continue;
        for (int k = 0; k < 2; ++k) {
            if (!open_fd[k] || !(fds[k].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            const ssize_t got = read(fds[k].fd, chunk, sizeof chunk);
            if (got <= 0) {
                open_fd[k] = false;
                continue;
            }
            const size_t room = lim.max_output_bytes > bufs[k]->size() ? lim.max_output_bytes - bufs[k]->size() : 0;
            bufs[k]->append(chunk, std::min(room, static_cast<size_t>(got)));
        }
    }
    close(out[0]);
    close(err[0]);
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {
    }
    kill(-pid, SIGKILL);  // stragglers the program left in its group
    if (WIFEXITED(st)) {
        r.exited = true;
        r.exit_code = WEXITSTATUS(st);
        if (r.exit_code == 127 && r.err.empty()) r.error = "exec failed: " + argv[0];
    } else if (WIFSIGNALED(st)) {
        r.signal = WTERMSIG(st);
    }
    return r;
}
#else
SandboxResult run_sandboxed(const std::vector<std::string>&, const std::map<std::string, std::string>&,
                            const SandboxLimits&) {
    SandboxResult r;
    r.error = "the code sandbox is Linux-only (fork, setrlimit, process groups)";
    return r;
}
#endif

VerifierResult UnitTestVerifier::verify(const std::string& prompt, const std::string& completion) const {
    const auto it = tests_.find(prompt);
    if (it == tests_.end()) return {-1.0f, "no unit test for this prompt", "unit_test", "tests:none"};
    const UnitTest& t = it->second;
    std::vector<std::string> argv = t.interpreter;
    argv.push_back("test");
    const SandboxResult s = run_sandboxed(argv, {{"solution", completion}, {"test", t.test_program}}, limits_);
    if (!s.ran) return {-1.0f, "abstain: " + s.error, "unit_test", t.uri};
    std::string why;
    if (s.timed_out)
        why = "timed out after " + std::to_string(limits_.timeout_ms) + " ms";
    else if (s.exited)
        why = s.exit_code == 0 ? "tests passed (exit 0)" : "tests failed (exit " + std::to_string(s.exit_code) + ")";
    else
        why = "killed by signal " + std::to_string(s.signal);
    if (!s.error.empty()) why += "; " + s.error;
    const std::string tail = trunc(s.err.empty() ? s.out : s.err, 200);
    if (!tail.empty()) why += "; output: " + tail;
    const bool pass = s.exited && s.exit_code == 0 && !s.timed_out;
    return {pass ? 1.0f : 0.0f, why, "unit_test", t.uri};
}

}  // namespace posttrain
}  // namespace microtorch
