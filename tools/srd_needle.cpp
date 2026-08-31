// Needle-in-haystack associative recall: the sharpest test of V1's claim
// (docs/SPARSE_ATTENTION.md). Linear attention's compressed state should fail
// precise KV recall; exact attention should succeed; SRD graduates only if
// it matches exact AND its gate concentrates on retrieval-critical
// positions.
//
//   srd_needle [steps=600] [T=256] [d=128] [csv_prefix=/tmp/srd_needle]
//              [batch=1] [npairs=8] [nkeys=64] [seed=7]
//
// Task (synthetic vocab of 256 symbols):
//   [filler | 8 x (key value) pairs | random filler ... | QUERY key_j]
//   next-token target at the final position is val_j.
// Fresh random sequences every step (infinite data); a FIXED 32-sequence
// probe set is evaluated every 25 steps under NoGrad: answer CE, answer
// argmax accuracy, and the SRD gate profile (gate at the query/key tail
// vs gate over filler) per lane.
//
// Pre-registered (before the first run):
//   exact answer-accuracy high (>0.8 by step 600); kimi materially lower;
//   srd close to exact; srd_f below srd; SRD tail-gate > filler-gate.
//
// Four lanes on identical sequences, checkpoint/resume identical to
// srd_parity (chunked execution; optimizer moments reset per chunk,
// identically for every lane).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "microtorch/safetensors.hpp"
#include "parity_model.hpp"

using namespace microtorch;
using parity::AttnKind;
using parity::ParityLM;

namespace {

constexpr int VOCAB = 256, QUERY = 1;
constexpr int KEY0 = 2, NKEYS = 64;      // keys  [2, 66)
constexpr int VAL0 = 66;                 // vals  [66, 130)
constexpr int FILL0 = 130, NFILL = 120;  // fill  [130, 250)

// Difficulty knobs (control-first calibration: three runs where exact
// never left the log-nkeys plateau mean the comparison needs a config
// the control can pass first). Defaults reproduce the original task.
// Vocab layout is unchanged; smaller nkeys/npairs just restrict draws.
int g_npairs = 8, g_nkeys = 64;

// ---- rung 2 (experiments/srd_r2/PREREGISTRATION_R2.md): separating H_retrieval from H_novelty ----
// The status-quo task partitions the vocabulary BY ROLE (keys [2,66),
// values [66,130), filler [130,250)), so needle tokens are
// distributionally distinct by construction and a prediction-residual
// gate must fire on them whether or not they are retrieval-critical.
// Two knobs remove that confound:
//   g_indist  keys/values drawn from the FILLER range — retrieval
//             structure identical, distributional signature gone
//   g_decoys  pairs of DISTINCT-range tokens in pair layout that are
//             never queried: maximally novel, zero retrieval value
bool g_indist = false;
int g_decoys = 0;

// Region labels for the four-region gate profile (the rung-2 primary
// metric). Written into `regions` alongside the sequence.
enum Region : unsigned char { R_FILLER = 0, R_TARGET, R_NONTARGET, R_DECOY, R_TAIL };

// One sequence of length T+1; x = seq[0..T-1], y = seq[1..T].
// regions (size T) labels each INPUT position for the gate profile.
std::vector<int> make_seq(size_t T, std::mt19937& rng,
                          std::vector<unsigned char>* regions = nullptr) {
    std::uniform_int_distribution<int> fill(FILL0, FILL0 + NFILL - 1);
    std::vector<int> keys(g_nkeys);
    for (int i = 0; i < g_nkeys; ++i) keys[i] = i;
    std::shuffle(keys.begin(), keys.end(), rng);  // distinct keys per seq

    // In-distribution mode: keys and values become filler-range symbols.
    // Distinct symbols per sequence keep the retrieval task well posed.
    std::vector<int> key_sym(g_nkeys), val_sym(g_nkeys);
    if (g_indist) {
        std::vector<int> pool(NFILL);
        for (int i = 0; i < NFILL; ++i) pool[i] = FILL0 + i;
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int i = 0; i < g_nkeys; ++i) {
            key_sym[i] = pool[i % NFILL];
            val_sym[i] = pool[(i + g_nkeys) % NFILL];
        }
    }
    auto key_tok = [&](int k) { return g_indist ? key_sym[k] : KEY0 + k; };
    auto val_tok = [&](int v) { return g_indist ? val_sym[v] : VAL0 + v; };

    std::vector<int> seq(T + 1);
    if (regions) regions->assign(T, R_FILLER);
    seq[0] = fill(rng);
    std::vector<int> val_of(g_npairs);
    for (int p = 0; p < g_npairs; ++p) {
        val_of[p] = rng() % g_nkeys;
        seq[1 + 2 * p] = key_tok(keys[p]);
        seq[2 + 2 * p] = val_tok(val_of[p]);
    }
    for (size_t i = 1 + 2 * g_npairs; i + 2 < T; ++i) seq[i] = fill(rng);

    // Decoy pairs: distinct-range tokens in pair layout, never queried.
    // Placed inside the filler span, clear of the pair block and tail.
    const size_t dec_lo = 1 + 2 * g_npairs, dec_hi = T > 12 ? T - 12 : dec_lo;
    for (int dcount = 0; dcount < g_decoys && dec_hi > dec_lo + 2; ++dcount) {
        const size_t at = dec_lo + (rng() % (dec_hi - dec_lo - 1));
        seq[at] = KEY0 + (rng() % g_nkeys);
        seq[at + 1] = VAL0 + (rng() % g_nkeys);
        if (regions) {
            (*regions)[at] = R_DECOY;
            (*regions)[at + 1] = R_DECOY;
        }
    }

    const int j = rng() % g_npairs;
    seq[T - 2] = QUERY;
    seq[T - 1] = key_tok(keys[j]);
    seq[T] = val_tok(val_of[j]);  // the answer
    if (regions) {
        for (int p = 0; p < g_npairs; ++p) {
            const unsigned char r = (p == j) ? R_TARGET : R_NONTARGET;
            (*regions)[1 + 2 * p] = r;
            (*regions)[2 + 2 * p] = r;
        }
        (*regions)[T - 2] = R_TAIL;
        (*regions)[T - 1] = R_TAIL;
    }
    return seq;
}

struct Lane {
    const char* name;
    ParityLM model;
    nn::AdamW opt;
    bool is_srd;
    Lane(const char* n, AttnKind k, size_t T, size_t d, unsigned seed, float lr)
        : name(n),
          model(k, VOCAB, d, /*heads=*/4, T, seed),
          opt(model.parameters(), lr),
          is_srd(k == AttnKind::SRD) {}
};

}  // namespace

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 600;
    const size_t T = argc > 2 ? static_cast<size_t>(std::atoi(argv[2])) : 256;
    const size_t d = argc > 3 ? static_cast<size_t>(std::atoi(argv[3])) : 128;
    const std::string prefix = argc > 4 ? argv[4] : "/tmp/srd_needle";
    // Batch>1 is the pre-registered escalation after two runs where no lane
    // (incl. exact) escaped the log-64 plateau at 1 seq/step: B sequences
    // per optimizer step, gradients accumulated, identical batch per lane.
    const int B = argc > 5 ? std::max(1, std::atoi(argv[5])) : 1;
    if (argc > 6) g_npairs = std::max(1, std::min(8, std::atoi(argv[6])));
    if (argc > 7) g_nkeys = std::max(g_npairs, std::min(64, std::atoi(argv[7])));
    // Model-init seed (shared by all four lanes, as always). CLI-exposed
    // for the rung-1 breakthrough replication; 7 reproduces every prior run.
    const unsigned seed = argc > 8 ? static_cast<unsigned>(std::atoi(argv[8])) : 7;
    // Rung 2 (experiments/srd_r2/PREREGISTRATION_R2.md): argv[9] = needle distribution
    // (distinct|indist), argv[10] = number of decoy pairs. Defaults
    // reproduce every prior run exactly.
    if (argc > 9) g_indist = std::string(argv[9]) == "indist";
    if (argc > 10) g_decoys = std::max(0, std::atoi(argv[10]));
    // argv[11]: learning rate. Registry finding S3-lrxopt says 3e-3 (the
    // historical hardcode) is the WORST setting under AdamW — the 2b
    // calibration tests whether that finding explains the task's
    // universal non-learning. Default preserves every prior run.
    const float lr = argc > 11 ? static_cast<float>(std::atof(argv[11])) : 3e-3f;
    const float lambda_gate = 0.05f;
    const int PROBE_EVERY = 25, NPROBE = 32;

    std::vector<Lane> lanes;
    lanes.emplace_back("exact", AttnKind::EXACT, T, d, seed, lr);
    lanes.emplace_back("kimi", AttnKind::KIMI, T, d, seed, lr);
    lanes.emplace_back("srd", AttnKind::SRD, T, d, seed, lr);
    lanes.emplace_back("srd_f", AttnKind::SRD, T, d, seed, lr);
    lanes[3].model.set_falsifier(true);
    for (auto& l : lanes) l.model.train();

    // Fixed held-out probe set.
    std::mt19937 probe_rng(9999);
    std::vector<std::vector<int>> probes;
    std::vector<std::vector<unsigned char>> probe_regions;
    for (int i = 0; i < NPROBE; ++i) {
        std::vector<unsigned char> reg;
        probes.push_back(make_seq(T, probe_rng, &reg));
        probe_regions.push_back(std::move(reg));
    }

    const char* ck = std::getenv("SRD_CKPT_DIR");
    const std::string ckpt_dir = ck ? ck : "/tmp/srd_needle_ckpt";
    int start_step = 0;
    {
        std::ifstream st(ckpt_dir + "/state.txt");
        if (st >> start_step && start_step > 0) {
            std::printf("resuming from step %d\n", start_step);
            for (auto& lane : lanes)
                lane.model.load_state_dict(
                    load_safetensors(ckpt_dir + "/" + lane.name + ".safetensors"));
        } else {
            start_step = 0;
        }
    }

    std::ofstream train_csv(prefix + "_train.csv", start_step ? std::ios::app : std::ios::out);
    std::ofstream probe_csv(prefix + "_probe.csv", start_step ? std::ios::app : std::ios::out);
    if (!start_step) {
        train_csv << "step,exact,kimi,srd,srd_f\n";
        probe_csv << "step,lane,answer_ce,answer_acc,tail_gate,fill_gate,"
                     "target_gate,nontarget_gate,decoy_gate\n";
    }

    std::mt19937 batch_rng(123);
    for (int s = 0; s < start_step * B; ++s) make_seq(T, batch_rng);  // fast-forward

    auto save_ckpt = [&](int step_done) {
        std::system(("mkdir -p " + ckpt_dir).c_str());
        for (auto& lane : lanes)
            save_safetensors(ckpt_dir + "/" + lane.name + ".safetensors", lane.model.state_dict());
        std::ofstream st(ckpt_dir + "/state.txt");
        st << step_done << "\n";
    };

    auto probe_eval = [&](int step) {
        NoGrad ng;
        for (auto& lane : lanes) {
            lane.model.eval();
            double ce = 0, acc = 0, tail_g = 0, fill_g = 0;
            // Rung-2 four-region profile (experiments/srd_r2/PREREGISTRATION_R2.md): target vs
            // non-target is the discrimination only H_retrieval predicts
            // (identical distribution, differing only in being asked for).
            double tgt_g = 0, non_g = 0, dec_g = 0;
            size_t n_probe = 0;
            for (const auto& seq : probes) {
                const std::vector<unsigned char>& reg = probe_regions[n_probe++];
                std::vector<int> x(seq.begin(), seq.end() - 1);
                Var logits = lane.model.forward(x);
                // Softmax CE at the final position only.
                const size_t last = T - 1;
                float mx = -1e30f;
                for (int v = 0; v < VOCAB; ++v) mx = std::max(mx, logits->data(last, v));
                double z = 0;
                for (int v = 0; v < VOCAB; ++v) z += std::exp(logits->data(last, v) - mx);
                const int ans = seq[T];
                ce += -(logits->data(last, ans) - mx - std::log(z));
                int arg = 0;
                for (int v = 1; v < VOCAB; ++v)
                    if (logits->data(last, v) > logits->data(last, arg)) arg = v;
                acc += (arg == ans) ? 1.0 : 0.0;
                if (lane.is_srd) {
                    // Gate profile from block 0 of the LAST forward.
                    const Var g = lane.model.srd[0]->gate();
                    tail_g += 0.5 * (g->data(T - 2, 0) + g->data(T - 1, 0));
                    double fg = 0;
                    size_t n = 0;
                    for (size_t t = 20; t < T - 10; ++t, ++n) fg += g->data(t, 0);
                    fill_g += fg / static_cast<double>(n);
                    // Region means for this sequence, then accumulate.
                    double s[5] = {0, 0, 0, 0, 0};
                    size_t c[5] = {0, 0, 0, 0, 0};
                    for (size_t t = 0; t < T; ++t) {
                        s[reg[t]] += g->data(t, 0);
                        ++c[reg[t]];
                    }
                    if (c[R_TARGET]) tgt_g += s[R_TARGET] / c[R_TARGET];
                    if (c[R_NONTARGET]) non_g += s[R_NONTARGET] / c[R_NONTARGET];
                    if (c[R_DECOY]) dec_g += s[R_DECOY] / c[R_DECOY];
                }
            }
            const double N = probes.size();
            probe_csv << step << ',' << lane.name << ',' << ce / N << ',' << acc / N << ','
                      << tail_g / N << ',' << fill_g / N << ',' << tgt_g / N << ',' << non_g / N
                      << ',' << dec_g / N << '\n';
            std::printf("  probe %-5s: answer_ce %.4f acc %.3f", lane.name, ce / N, acc / N);
            if (lane.is_srd) std::printf("  tail_gate %.3f fill_gate %.3f", tail_g / N, fill_g / N);
            std::printf("\n");
            lane.model.train();
        }
        probe_csv.flush();
    };

    std::printf(
        "batch=%d npairs=%d nkeys=%d seed=%u needle=%s decoys=%d "
        "(uniform-CE floor %.4f)\n",
        B, g_npairs, g_nkeys, seed, g_indist ? "indist" : "distinct", g_decoys,
        std::log(static_cast<double>(g_nkeys)));
    std::printf("%5s %9s %9s %9s %9s\n", "step", "exact", "kimi", "srd", "srd_f");
    for (int step = start_step + 1; step <= steps; ++step) {
        std::vector<std::vector<int>> batch;
        for (int b = 0; b < B; ++b) batch.push_back(make_seq(T, batch_rng));

        float losses[4];
        for (size_t li = 0; li < lanes.size(); ++li) {
            Lane& lane = lanes[li];
            lane.opt.zero_grad();
            double task_sum = 0;
            for (const auto& seq : batch) {
                std::vector<int> x(seq.begin(), seq.end() - 1);
                std::vector<int> y(seq.begin() + 1, seq.end());
                Var logits = lane.model.forward(x);
                Var task = ops::cross_entropy(logits, y);
                Var loss = lane.is_srd
                               ? ops::add(task, ops::scale(lane.model.mean_gate(), lambda_gate))
                               : task;
                backward(ops::scale(loss, 1.0f / static_cast<float>(B)));
                task_sum += task->data(0, 0);
            }
            ops::clip_grad_norm(lane.model.parameters(), 1.0f);
            lane.opt.step();
            losses[li] = static_cast<float>(task_sum / B);
        }
        train_csv << step << ',' << losses[0] << ',' << losses[1] << ',' << losses[2] << ','
                  << losses[3] << '\n';
        if (step % 10 == 0)
            std::printf("%5d %9.4f %9.4f %9.4f %9.4f\n", step, losses[0], losses[1], losses[2],
                        losses[3]);
        if (step % PROBE_EVERY == 0) {
            std::printf("-- probe @ %d --\n", step);
            probe_eval(step);
        }
        std::fflush(stdout);
    }
    save_ckpt(steps);
    std::printf("done through %d; wrote %s_{train,probe}.csv\n", steps, prefix.c_str());

    // ---- P5: matched-density controls (experiments/srd_r2/PREREGISTRATION_R2.md, env-gated) ----
    // On the SAME trained srd weights, evaluate the probe set with the
    // gate REPLACED by three policies at equal density rho: the SRD
    // gate's own top-rho queries, a random rho subset, and a positional
    // baseline (the LAST rho*T queries — recency, the field's default).
    // exact_ref (rho=1) and linear_ref (rho=0) bound the range.
    if (std::getenv("SRD_DENSITY_EVAL")) {
        NoGrad ng;
        Lane& srd_lane = lanes[2];
        srd_lane.model.eval();
        std::ofstream dcsv(prefix + "_density.csv", std::ios::trunc);
        dcsv << "policy,rho,answer_ce,answer_acc\n";
        auto eval_forced = [&](const std::vector<std::vector<float>>& masks, const char* name,
                               double rho) {
            double ce = 0, acc = 0;
            for (size_t pi = 0; pi < probes.size(); ++pi) {
                for (auto& blk : srd_lane.model.srd) blk->forced_gate = masks[pi];
                const auto& seq = probes[pi];
                std::vector<int> x(seq.begin(), seq.end() - 1);
                Var logits = srd_lane.model.forward(x);
                const size_t last = T - 1;
                float mx = -1e30f;
                for (int v = 0; v < VOCAB; ++v) mx = std::max(mx, logits->data(last, v));
                double z = 0;
                for (int v = 0; v < VOCAB; ++v) z += std::exp(logits->data(last, v) - mx);
                const int ans = seq[T];
                ce += -(logits->data(last, ans) - mx - std::log(z));
                int arg = 0;
                for (int v = 1; v < VOCAB; ++v)
                    if (logits->data(last, v) > logits->data(last, arg)) arg = v;
                acc += (arg == ans) ? 1.0 : 0.0;
            }
            for (auto& blk : srd_lane.model.srd) blk->forced_gate.clear();
            const double N = probes.size();
            dcsv << name << ',' << rho << ',' << ce / N << ',' << acc / N << '\n';
            std::printf("  density %-10s rho=%.2f: ce %.4f acc %.3f\n", name, rho, ce / N, acc / N);
        };
        // Natural gate values per probe (block 0, the profiled block).
        std::vector<std::vector<float>> nat(probes.size());
        for (size_t pi = 0; pi < probes.size(); ++pi) {
            std::vector<int> x(probes[pi].begin(), probes[pi].end() - 1);
            (void)srd_lane.model.forward(x);
            const Var g = srd_lane.model.srd[0]->gate();
            nat[pi].resize(T);
            for (size_t t = 0; t < T; ++t) nat[pi][t] = g->data(t, 0);
        }
        std::mt19937 drng(4242);
        for (double rho : {0.10, 0.25}) {
            const size_t k = std::max<size_t>(1, static_cast<size_t>(rho * T));
            std::vector<std::vector<float>> m_srd(probes.size()), m_rnd(probes.size()),
                m_pos(probes.size());
            for (size_t pi = 0; pi < probes.size(); ++pi) {
                // SRD policy: top-k by the natural gate.
                std::vector<size_t> idx(T);
                for (size_t t = 0; t < T; ++t) idx[t] = t;
                std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                                  [&](size_t a, size_t b) { return nat[pi][a] > nat[pi][b]; });
                m_srd[pi].assign(T, 0.0f);
                for (size_t j = 0; j < k; ++j) m_srd[pi][idx[j]] = 1.0f;
                // Random policy at the same k.
                std::shuffle(idx.begin(), idx.end(), drng);
                m_rnd[pi].assign(T, 0.0f);
                for (size_t j = 0; j < k; ++j) m_rnd[pi][idx[j]] = 1.0f;
                // Positional: the last k queries.
                m_pos[pi].assign(T, 0.0f);
                for (size_t t = T - k; t < T; ++t) m_pos[pi][t] = 1.0f;
            }
            eval_forced(m_srd, "srd_top", rho);
            eval_forced(m_rnd, "random", rho);
            eval_forced(m_pos, "positional", rho);
        }
        std::vector<std::vector<float>> ones(probes.size(), std::vector<float>(T, 1.0f));
        std::vector<std::vector<float>> zeros(probes.size(), std::vector<float>(T, 0.0f));
        eval_forced(ones, "exact_ref", 1.0);
        eval_forced(zeros, "linear_ref", 0.0);
        std::printf("wrote %s_density.csv\n", prefix.c_str());
    }
    return 0;
}
