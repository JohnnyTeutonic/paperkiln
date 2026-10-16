// posttrain_verify: score {"prompt", "completion", ...} JSONL with one verifier
// and write each record back with its receipt attached
// (include/microtorch/posttrain/verifier.hpp).
//
//   posttrain_verify exact_match     --refs R.jsonl            < in.jsonl > out.jsonl
//   posttrain_verify format          [--pattern RE --label L]  < in.jsonl > out.jsonl
//   posttrain_verify synthetic_judge --refs R.jsonl [--cue P --near P --near-k K
//                                     --other P --reject-correct P --seed S] < in > out
//   posttrain_verify llm_judge       --cache C.jsonl --judge-id ID --rubric-version V < in > out
//   posttrain_verify human           --audit A.jsonl            < in.jsonl > out.jsonl
//
// Every input field is kept; the tool adds reward, rationale, evidence_kind,
// evidence_uri, audited and "verifier" (name and version). This is the record
// format of the rejection sampler (charter 0001, Phase 2). Exit 2 on a usage
// error, 1 on a malformed input line (with its line number).
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "microtorch/posttrain/verifier.hpp"

using namespace microtorch::posttrain;
using nlohmann::json;

namespace {

int usage(const char* msg) {
    std::fprintf(stderr, "posttrain_verify: %s\n(see the header of tools/posttrain_verify.cpp)\n", msg);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage("missing verifier kind");
    const std::string kind = argv[1];
    std::map<std::string, std::string> opt;
    for (int i = 2; i < argc; i += 2) {
        const std::string k = argv[i];
        if (k.rfind("--", 0) != 0 || i + 1 >= argc) return usage(("bad option " + k).c_str());
        opt[k.substr(2)] = argv[i + 1];
    }
    auto need = [&](const char* k) -> const std::string& {
        auto it = opt.find(k);
        if (it == opt.end()) throw std::invalid_argument(std::string("--") + k + " is required");
        return it->second;
    };
    std::unique_ptr<Verifier> v;
    try {
        if (kind == "exact_match") {
            v = std::make_unique<ExactMatchVerifier>(load_references(need("refs")));
        } else if (kind == "format") {
            v = opt.count("pattern") ? std::make_unique<FormatVerifier>(opt["pattern"], need("label"))
                                     : std::make_unique<FormatVerifier>();
        } else if (kind == "synthetic_judge") {
            SyntheticJudgeParams p;
            if (opt.count("cue")) p.accept_wrong_with_cue = std::stod(opt["cue"]);
            if (opt.count("near")) p.accept_wrong_near = std::stod(opt["near"]);
            if (opt.count("near-k")) p.near_k = std::stoll(opt["near-k"]);
            if (opt.count("other")) p.accept_wrong_other = std::stod(opt["other"]);
            if (opt.count("reject-correct")) p.reject_correct = std::stod(opt["reject-correct"]);
            if (opt.count("seed")) p.seed = std::stoull(opt["seed"]);
            v = std::make_unique<SyntheticJudgeVerifier>(load_references(need("refs")), p);
        } else if (kind == "llm_judge") {
            v = std::make_unique<LLMJudgeVerifier>(need("cache"), need("judge-id"), need("rubric-version"));
        } else if (kind == "human") {
            v = std::make_unique<HumanAuditVerifier>(need("audit"));
        } else {
            return usage(("unknown verifier kind " + kind).c_str());
        }
    } catch (const std::exception& e) {
        return usage(e.what());
    }
    std::string line;
    size_t n = 0;
    while (std::getline(std::cin, line)) {
        ++n;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        json rec = json::parse(line, nullptr, false);
        if (rec.is_discarded() || !rec.is_object() || !rec.contains("prompt") || !rec.contains("completion") ||
            !rec["prompt"].is_string() || !rec["completion"].is_string()) {
            std::fprintf(stderr, "posttrain_verify: line %zu: need a JSON object with string prompt and completion\n", n);
            return 1;
        }
        const VerifierResult r = v->verify(rec["prompt"].get<std::string>(), rec["completion"].get<std::string>());
        const json receipt = json::parse(to_jsonl(r));
        for (auto it = receipt.begin(); it != receipt.end(); ++it) rec[it.key()] = it.value();
        rec["verifier"] = v->name();
        std::cout << rec.dump() << "\n";
    }
    return 0;
}
