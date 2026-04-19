#include "training_controller.h"

#include <algorithm>
#include <numeric>
#include <utility>

namespace {

const char* conditionName(TrainingCondition condition) {
    switch (condition) {
        case TrainingCondition::Null:
            return "null";
        case TrainingCondition::Random:
            return "random";
        case TrainingCondition::Adaptive:
            return "adaptive";
    }
    return "null";
}

} // namespace

TrainingController::TrainingController(std::vector<std::string> pattern_names,
                                       double value_alpha,
                                       double eligibility_gamma,
                                       double min_reward,
                                       std::uint32_t random_seed)
    : pattern_names_(std::move(pattern_names)),
      values_(pattern_names_.size(), min_reward),
      eligibility_(pattern_names_.size(), 0.0),
      value_alpha_(value_alpha),
      eligibility_gamma_(eligibility_gamma),
      min_reward_(min_reward),
      rng_(random_seed) {}

TrainingDecision TrainingController::onEpisodeEnd(double reward_seconds,
                                                  TrainingCondition condition,
                                                  bool allow_delivery) {
    updateValues(reward_seconds);

    recent_rewards_.push_back(reward_seconds);
    if (recent_rewards_.size() > 20) {
        recent_rewards_.pop_front();
    }

    TrainingDecision decision;
    decision.condition_name = conditionName(condition);
    decision.mean_5 = movingAverage(5);
    decision.mean_20 = movingAverage(20);

    decayEligibility();

    if (recent_rewards_.size() < 5) {
        return decision;
    }

    // STAR methods condition: only stimulate when short-term performance drops.
    if (decision.mean_5 >= decision.mean_20) {
        return decision;
    }

    decision.eligible = true;
    if (!allow_delivery) {
        return decision;
    }

    if (condition == TrainingCondition::Null) {
        return decision;
    }

    if (condition == TrainingCondition::Random) {
        decision.delivered = true;
        decision.sequence_type = "random5";
        return decision;
    }

    if (pattern_names_.empty()) {
        return decision;
    }

    const int pattern_index = samplePatternIndex();
    if (pattern_index < 0) {
        return decision;
    }

    eligibility_[static_cast<std::size_t>(pattern_index)] += 1.0;
    decision.delivered = true;
    decision.pattern_index = pattern_index;
    decision.sequence_name = pattern_names_[static_cast<std::size_t>(pattern_index)];
    decision.sequence_type = "adaptive_pair";
    return decision;
}

double TrainingController::movingAverage(std::size_t count) const {
    if (recent_rewards_.empty()) {
        return 0.0;
    }

    const std::size_t actual = (std::min)(count, recent_rewards_.size());
    double total = 0.0;
    for (std::size_t i = recent_rewards_.size() - actual; i < recent_rewards_.size(); ++i) {
        total += recent_rewards_[i];
    }
    return total / static_cast<double>(actual);
}

void TrainingController::updateValues(double reward_seconds) {
    for (std::size_t i = 0; i < values_.size(); ++i) {
        values_[i] += value_alpha_ * (reward_seconds - values_[i]) * eligibility_[i];
        values_[i] = (std::max)(values_[i], min_reward_);
    }
}

void TrainingController::decayEligibility() {
    for (double& value : eligibility_) {
        value *= eligibility_gamma_;
    }
}

int TrainingController::samplePatternIndex() {
    const double total = std::accumulate(values_.begin(), values_.end(), 0.0);
    if (total <= 0.0) {
        return -1;
    }

    std::uniform_real_distribution<double> dist(0.0, total);
    double draw = dist(rng_);
    for (std::size_t i = 0; i < values_.size(); ++i) {
        draw -= values_[i];
        if (draw <= 0.0) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(values_.size() - 1);
}
