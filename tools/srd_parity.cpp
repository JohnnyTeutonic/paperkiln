// SRD graduation gate (docs/SPARSE_ATTENTION.md protocol): TinyStories
// training parity.
//
//   srd_parity <vocab.gguf> <corpus.txt> [steps=300] [T=64] [d=128] [csv]
//
// Trains FOUR parameter-matched 2-layer language models on IDENTICAL
// batches (same seeds, same windows):
//
//   exact  -- pre-LN blocks with CausalSelfAttention
//   kimi   -- same blocks with KimiLinearAttention
//   srd    -- same blocks with SurpriseRoutedAttention
//   srd-f  -- SRD with the shuffled-predictor FALSIFIER: gate keeps its
//             distribution but carries no query-aligned information
//
// Graduation reads: srd should track exact within noise while kimi trails
// (or all match -- also informative); srd-f materially worse than srd
// means the surprise signal is doing real work. If srd-f == srd, the gate
// is not using surprise and V1 is falsified.
//
// Vocabulary comes from a trained transformer_cpp GGUF (word-level
// lowercased whole-word lookup, <unk>=0), so tokenization matches the
// pipeline the repo already serves.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "microtorch/nn.hpp"
#include "microtorch/safetensors.hpp"
#include "microtorch/srd.hpp"

using namespace microtorch;

namespace {

// ---- minimal GGUF metadata reader (vocab only; tensors skipped) ----
template <typename T>
T rd(const std::vector<uint8_t>& b, size_t& p) {
    T v;
    std::memcpy(&v, b.data() + p, sizeof(T));
    p += sizeof(T);
    return v;
}
std::string rd_str(const std::vector<uint8_t>& b, size_t& p) {
    const uint64_t n = rd<uint64_t>(b, p);
    std::string s(reinterpret_cast<const char*>(b.data() + p), n);
    p += n;
    return s;
}

std::vector<std::string> read_gguf_vocab(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> b(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(b.size()));
    size_t p = 0;
    if (rd<uint32_t>(b, p) != 0x46554747u) throw std::runtime_error("not GGUF");
    rd<uint32_t>(b, p);  // version
    rd<uint64_t>(b, p);  // tensor count
    const uint64_t n_meta = rd<uint64_t>(b, p);
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < n_meta; ++i) {
        const std::string key = rd_str(b, p);
        const uint32_t vt = rd<uint32_t>(b, p);
        switch (vt) {
            case 4:
                rd<uint32_t>(b, p);
                break;
            case 5:
                rd<int32_t>(b, p);
                break;
            case 6:
                rd<float>(b, p);
                break;
            case 8:
                rd_str(b, p);
                break;
            case 9: {
                const uint32_t et = rd<uint32_t>(b, p);
                const uint64_t n = rd<uint64_t>(b, p);
                for (uint64_t k = 0; k < n; ++k) {
                    if (et == 8) {
                        std::string t = rd_str(b, p);
                        if (key == "tokenizer.ggml.tokens") tokens.push_back(std::move(t));
                    } else if (et == 6)
                        rd<float>(b, p);
                    else
                        throw std::runtime_error("bad array");
                }
                break;
            }
            default:
                throw std::runtime_error("bad meta type");
        }
    }
    if (tokens.empty()) throw std::runtime_error("no vocab in " + path);
    return tokens;
}

// Word-level tokenizer matching transformer_cpp: lowercase words, single
// punctuation marks as their own tokens, <unk>=0 fallback.
std::vector<int> tokenize(const std::string& text, const std::map<std::string, int>& vocab,
                          size_t max_tokens) {
    std::vector<int> ids;
    std::string cur;
    auto flush = [&]() {
        if (cur.empty()) return;
        auto it = vocab.find(cur);
        ids.push_back(it == vocab.end() ? 0 : it->second);
        cur.clear();
    };
    for (char ch : text) {
        if (ids.size() >= max_tokens) break;
        const unsigned char c = static_cast<unsigned char>(ch);
        if (std::isalpha(c) || c == '\'') {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else if (std::isdigit(c)) {
            cur.push_back(static_cast<char>(c));
        } else {
            flush();
            if (!std::isspace(c)) {
                std::string p(1, static_cast<char>(c));
                auto it = vocab.find(p);
                ids.push_back(it == vocab.end() ? 0 : it->second);
            }
        }
    }
    flush();
    return ids;
}

// ---- the parity model: 2 pre-LN blocks, attention type pluggable ----
enum class AttnKind { EXACT, KIMI, SRD };

class ParityLM : public nn::Module {
public:
    ParityLM(AttnKind kind, size_t vocab, size_t d, size_t n_heads, size_t n_ctx, unsigned seed)
        : kind_(kind) {
        wte = mod<nn::Embedding>("wte", vocab, d, seed + 1);
        wpe = mod<nn::Embedding>("wpe", n_ctx, d, seed + 2);
        for (int b = 0; b < 2; ++b) {
            const unsigned s = seed + 10 * (b + 1);
            ln1.push_back(mod<nn::LayerNorm>("ln1_" + std::to_string(b), d));
            ln2.push_back(mod<nn::LayerNorm>("ln2_" + std::to_string(b), d));
            mlp.push_back(mod<nn::MLP>("mlp_" + std::to_string(b), d, 4 * d, s + 3));
            switch (kind) {
                case AttnKind::EXACT:
                    exact.push_back(
                        mod<nn::CausalSelfAttention>("attn_" + std::to_string(b), d, n_heads, s));
                    break;
                case AttnKind::KIMI:
                    kimi.push_back(
                        mod<nn::KimiLinearAttention>("attn_" + std::to_string(b), d, n_heads, s));
                    break;
                case AttnKind::SRD:
                    srd.push_back(mod<nn::SurpriseRoutedAttention>("attn_" + std::to_string(b), d,
                                                                   n_heads, s));
                    break;
            }
        }
        ln_f = mod<nn::LayerNorm>("ln_f", d);
        head = mod<nn::Linear>("head", d, vocab, false, seed + 99);
    }

    Var forward(const std::vector<int>& ids) const {
        std::vector<int> pos(ids.size());
        for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i);
        Var h = ops::add(wte->forward(ids), wpe->forward(pos));
        for (int b = 0; b < 2; ++b) {
            Var a;
            Var n1 = ln1[b]->forward(h);
            switch (kind_) {
                case AttnKind::EXACT:
                    a = exact[b]->forward(n1);
                    break;
                case AttnKind::KIMI:
                    a = kimi[b]->forward(n1);
                    break;
                case AttnKind::SRD:
                    a = srd[b]->forward(n1);
                    break;
            }
            h = ops::add(h, a);
            h = ops::add(h, mlp[b]->forward(ln2[b]->forward(h)));
        }
        return head->forward(ln_f->forward(h));
    }

    // SRD-only: mean gate over both blocks (tape Var for the aux loss).
    Var mean_gate() const {
        return ops::scale(ops::add(ops::mean(srd[0]->gate()), ops::mean(srd[1]->gate())), 0.5f);
    }
    void set_falsifier(bool on) {
        for (auto& s : srd) s->shuffle_predictor = on;
    }

    AttnKind kind_;
    std::shared_ptr<nn::Embedding> wte, wpe;
    std::vector<std::shared_ptr<nn::LayerNorm>> ln1, ln2;
    std::vector<std::shared_ptr<nn::MLP>> mlp;
    std::vector<std::shared_ptr<nn::CausalSelfAttention>> exact;
    std::vector<std::shared_ptr<nn::KimiLinearAttention>> kimi;
    std::vector<std::shared_ptr<nn::SurpriseRoutedAttention>> srd;
    std::shared_ptr<nn::LayerNorm> ln_f;
    std::shared_ptr<nn::Linear> head;
};

struct Lane {
    const char* name;
    ParityLM model;
    nn::AdamW opt;
    bool is_srd;
    Lane(const char* n, AttnKind k, size_t vocab, size_t d, size_t heads, size_t n_ctx,
         unsigned seed, float lr)
        : name(n),
          model(k, vocab, d, heads, n_ctx, seed),
          opt(model.parameters(), lr),
          is_srd(k == AttnKind::SRD) {}
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: srd_parity <vocab.gguf> <corpus.txt> [steps] [T] "
                     "[d] [csv]\n");
        return 2;
    }
    const std::string gguf_path = argv[1], corpus_path = argv[2];
    const int steps = argc > 3 ? std::atoi(argv[3]) : 300;
    const size_t T = argc > 4 ? static_cast<size_t>(std::atoi(argv[4])) : 64;
    const size_t d = argc > 5 ? static_cast<size_t>(std::atoi(argv[5])) : 128;
    const std::string csv_path = argc > 6 ? argv[6] : "/tmp/srd_parity.csv";
    // vocab_cap: keep only the first N vocab ids (word-level GGUF vocabs are
    // frequency-ordered); the rest map to <unk>. The softmax head is the
    // dominant FLOP at long T, so capping makes T=256+ runs tractable while
    // every lane still sees identical data.
    const size_t vocab_cap = argc > 7 ? static_cast<size_t>(std::atoi(argv[7])) : 0;
    const size_t heads = 4;
    const float lr = 3e-3f, lambda_gate = 0.05f;

    // Vocabulary + corpus.
    auto tokens = read_gguf_vocab(gguf_path);
    if (vocab_cap > 0 && vocab_cap < tokens.size()) tokens.resize(vocab_cap);
    std::map<std::string, int> vocab;
    for (size_t i = 0; i < tokens.size(); ++i) vocab.emplace(tokens[i], static_cast<int>(i));
    std::ifstream cf(corpus_path);
    if (!cf) {
        std::fprintf(stderr, "no corpus\n");
        return 2;
    }
    std::string text((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
    auto ids = tokenize(text, vocab, 120000);
    size_t unk = 0;
    for (int t : ids) unk += (t == 0);
    std::printf("vocab %zu | corpus %zu tokens | unk rate %.2f%%\n", tokens.size(), ids.size(),
                100.0 * static_cast<double>(unk) / ids.size());
    if (ids.size() < 10 * T) {
        std::fprintf(stderr, "corpus too small\n");
        return 2;
    }

    // Four lanes, identical init seeds for the shared components.
    std::vector<Lane> lanes;
    lanes.emplace_back("exact", AttnKind::EXACT, tokens.size(), d, heads, T, 7, lr);
    lanes.emplace_back("kimi", AttnKind::KIMI, tokens.size(), d, heads, T, 7, lr);
    lanes.emplace_back("srd", AttnKind::SRD, tokens.size(), d, heads, T, 7, lr);
    lanes.emplace_back("srd_f", AttnKind::SRD, tokens.size(), d, heads, T, 7, lr);
    lanes[3].model.set_falsifier(true);
    for (auto& l : lanes) l.model.train();

    // Chunked execution with resume: WSL kills detached jobs, so long runs
    // go as N foreground chunks. Model weights checkpoint per lane;
    // optimizer moments deliberately restart each chunk (identical
    // treatment for every lane, so the comparison stays fair -- noted in
    // the results). The batch RNG fast-forwards so windows continue the
    // same stream.
    const char* ckpt_env = std::getenv("SRD_CKPT_DIR");
    const std::string ckpt_dir = ckpt_env ? ckpt_env : "/tmp/srd_ckpt";
    int start_step = 0;
    {
        std::ifstream st(ckpt_dir + "/state.txt");
        if (st >> start_step && start_step > 0) {
            std::printf("resuming from step %d\n", start_step);
            for (auto& lane : lanes) {
                lane.model.load_state_dict(
                    load_safetensors(ckpt_dir + "/" + lane.name + ".safetensors"));
            }
        } else {
            start_step = 0;
        }
    }

    std::ofstream csv(csv_path, start_step > 0 ? std::ios::app : std::ios::out);
    if (start_step == 0) csv << "step,exact,kimi,srd,srd_f,srd_gate,srdf_gate\n";

    std::mt19937 batch_rng(123);                       // ONE stream: identical windows per lane
    for (int s = 0; s < start_step; ++s) batch_rng();  // fast-forward

    auto save_ckpt = [&](int step_done) {
        std::system(("mkdir -p " + ckpt_dir).c_str());
        for (auto& lane : lanes) {
            save_safetensors(ckpt_dir + "/" + lane.name + ".safetensors", lane.model.state_dict());
        }
        std::ofstream st(ckpt_dir + "/state.txt");
        st << step_done << "\n";
    };

    std::printf("%5s %9s %9s %9s %9s %7s %7s\n", "step", "exact", "kimi", "srd", "srd_f", "gate",
                "gate_f");
    for (int step = start_step + 1; step <= steps; ++step) {
        const size_t start = batch_rng() % (ids.size() - T - 1);
        std::vector<int> x(ids.begin() + start, ids.begin() + start + T);
        std::vector<int> y(ids.begin() + start + 1, ids.begin() + start + T + 1);

        float losses[4] = {0, 0, 0, 0}, gates[2] = {0, 0};
        for (size_t li = 0; li < lanes.size(); ++li) {
            Lane& lane = lanes[li];
            Var logits = lane.model.forward(x);
            Var task = ops::cross_entropy(logits, y);
            Var loss = task;
            if (lane.is_srd) {
                loss = ops::add(task, ops::scale(lane.model.mean_gate(), lambda_gate));
                gates[li - 2] = lane.model.mean_gate()->data(0, 0);
            }
            lane.opt.zero_grad();
            backward(loss);
            ops::clip_grad_norm(lane.model.parameters(), 1.0f);
            lane.opt.step();
            losses[li] = task->data(0, 0);  // task loss only, comparable
        }
        csv << step << ',' << losses[0] << ',' << losses[1] << ',' << losses[2] << ',' << losses[3]
            << ',' << gates[0] << ',' << gates[1] << '\n';
        if (step % 10 == 0 || step == 1) {
            std::printf("%5d %9.4f %9.4f %9.4f %9.4f %7.3f %7.3f\n", step, losses[0], losses[1],
                        losses[2], losses[3], gates[0], gates[1]);
            std::fflush(stdout);
        }
    }
    csv.close();
    save_ckpt(steps);
    std::printf("wrote %s (through step %d)\n", csv_path.c_str(), steps);
    return 0;
}
