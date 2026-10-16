#pragma once
// GRPO, minimal (docs/decisions/0001-posttrain-charter.md §4, Phase 4).
//
// Group-relative advantage: G completions per prompt, A_i = (r_i - mean_G) /
// (std_G + eps) with std-normalisation a switch (it biases the gradient;
// Liu et al. 2025, "Dr. GRPO"). A group whose rewards are all equal has
// A = 0 and is counted as flat. One gradient step per sampled batch, so the
// importance ratio is exactly 1 where it is differentiated and PPO clipping
// would do nothing; it is omitted. The loss itself is ops::policy_gradient_kl.
//
// FALSIFIER (charter §4): AdvantageMode::Shuffled permutes advantages within
// each group (the same values, attached to the wrong completions). If reward
// still climbs under it, the improvement did not come from the advantage.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace microtorch {
namespace posttrain {

enum class AdvantageMode { Normal, Shuffled };

struct GroupStats {
    size_t groups = 0;
    size_t flat_groups = 0;  // every reward equal: no policy gradient from it
};

// rewards: P groups of G, laid out group-major. Abstentions (reward < 0)
// are not allowed here; the caller decides what an abstain is worth.
inline std::vector<float> group_advantages(const std::vector<float>& rewards, size_t G, bool std_norm,
                                           GroupStats* stats = nullptr, float eps = 1e-6f,
                                           AdvantageMode mode = AdvantageMode::Normal, unsigned shuffle_seed = 0) {
    if (G == 0 || rewards.size() % G != 0) throw std::invalid_argument("group_advantages: size not a multiple of G");
    for (float r : rewards)
        if (r < 0.0f) throw std::invalid_argument("group_advantages: abstained reward in a group");
    std::vector<float> A(rewards.size(), 0.0f);
    GroupStats st;
    std::mt19937 gen(shuffle_seed);
    for (size_t g0 = 0; g0 < rewards.size(); g0 += G) {
        ++st.groups;
        double mean = 0.0;
        for (size_t i = 0; i < G; ++i) mean += rewards[g0 + i];
        mean /= static_cast<double>(G);
        double var = 0.0;
        for (size_t i = 0; i < G; ++i) var += (rewards[g0 + i] - mean) * (rewards[g0 + i] - mean);
        const double sd = G > 1 ? std::sqrt(var / static_cast<double>(G)) : 0.0;
        bool flat = true;
        for (size_t i = 1; i < G; ++i) flat = flat && rewards[g0 + i] == rewards[g0];
        if (flat) {
            ++st.flat_groups;
            continue;  // A stays 0
        }
        for (size_t i = 0; i < G; ++i) {
            const double a = rewards[g0 + i] - mean;
            A[g0 + i] = static_cast<float>(std_norm ? a / (sd + eps) : a);
        }
        if (mode == AdvantageMode::Shuffled) std::shuffle(A.begin() + g0, A.begin() + g0 + G, gen);
    }
    if (stats) *stats = st;
    return A;
}

}  // namespace posttrain
}  // namespace microtorch
