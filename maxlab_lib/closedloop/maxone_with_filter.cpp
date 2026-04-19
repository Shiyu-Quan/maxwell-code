#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cartpole_task.h"
#include "gamewindow.h"
#include "maxlab/maxlab.h"
#include "spike_detection.h"
#include "training_controller.h"

#ifdef USE_QT
#include <QApplication>
#include <QMetaObject>
#endif

using SteadyClock = std::chrono::steady_clock;

namespace {
std::atomic<bool> g_running{true};

struct RunConfig {
    std::string config_path;
    uint8_t target_well = 0;
    int read_window_ms = 200;
    int training_window_ms = 300;
    bool show_gui = true;
    bool wait_for_sync = true;
    std::size_t channel_count = 1024;
    double experiment_duration_s = 900.0;
    int num_cycles = 1;
    double active_cycle_duration_s = 900.0;
    double rest_duration_s = 2700.0;
    double first_cycle_pass_threshold_s = 10.0;
    double encoding_scale_a = 7.0;
    double encoding_scale_b = 0.15;
    double ema_alpha = 0.2;
    double force_scale_n = 10.0;
    double timestep_seconds = 0.2;
    ExperimentMode mode = ExperimentMode::Cycled;
    SpikeDetectorConfig detector;
    std::vector<int> decoding_left_channels;
    std::vector<int> decoding_right_channels;
    std::string encoding_left_sequence;
    std::string encoding_right_sequence;
    std::vector<std::string> training_pattern_names;
    std::vector<std::string> cycle_condition_schedule;
    std::vector<int> encoding_stim_electrodes;
    std::vector<int> training_stim_electrodes;
    std::vector<int> decoding_left_electrodes;
    std::vector<int> decoding_right_electrodes;
    double stim_pulse_amplitude_mV = 0.0;
    int stim_phase_us = 0;
    double stim_inter_pulse_interval_ms = 0.0;
    double stim_training_frequency_hz = 0.0;
    std::string artifact_filter_mode = "iir";
    std::string log_path;
    std::uint32_t random_seed = 12345;
};

struct JsonParser {
    explicit JsonParser(std::string text) : text_(std::move(text)) {}

    bool hasKey(const std::string& key) const {
        const std::string pattern = "\"" + key + "\"";
        return text_.find(pattern) != std::string::npos;
    }

    std::string stringValue(const std::string& key) const {
        const std::size_t value_start = valueStart(key);
        if (text_[value_start] != '"') {
            throw std::runtime_error("Expected string for key: " + key);
        }
        return parseString(value_start);
    }

    double numberValue(const std::string& key) const {
        const std::size_t value_start = valueStart(key);
        std::size_t end = value_start;
        while (end < text_.size() &&
               (std::isdigit(static_cast<unsigned char>(text_[end])) || text_[end] == '-' ||
                text_[end] == '+' || text_[end] == '.' || text_[end] == 'e' || text_[end] == 'E')) {
            ++end;
        }
        return std::stod(text_.substr(value_start, end - value_start));
    }

    bool boolValue(const std::string& key) const {
        const std::size_t value_start = valueStart(key);
        if (text_.compare(value_start, 4, "true") == 0) return true;
        if (text_.compare(value_start, 5, "false") == 0) return false;
        throw std::runtime_error("Expected bool for key: " + key);
    }

    std::vector<int> intArrayValue(const std::string& key) const {
        return splitNumericArray<int>(key);
    }

    std::vector<std::string> stringArrayValue(const std::string& key) const {
        const std::size_t value_start = valueStart(key);
        if (text_[value_start] != '[') {
            throw std::runtime_error("Expected string array for key: " + key);
        }
        std::vector<std::string> result;
        std::size_t pos = value_start + 1;
        while (pos < text_.size()) {
            pos = skipWhitespace(pos);
            if (text_[pos] == ']') break;
            if (text_[pos] != '"') {
                throw std::runtime_error("Expected string element in array for key: " + key);
            }
            result.push_back(parseString(pos));
            pos = nextAfterString(pos);
            pos = skipWhitespace(pos);
            if (text_[pos] == ',') ++pos;
        }
        return result;
    }

    std::vector<int> intArrayValueOr(const std::string& key, const std::vector<int>& fallback) const {
        return hasKey(key) ? intArrayValue(key) : fallback;
    }

    std::vector<std::string> stringArrayValueOr(const std::string& key,
                                                const std::vector<std::string>& fallback) const {
        return hasKey(key) ? stringArrayValue(key) : fallback;
    }

    double numberValueOr(const std::string& key, double fallback) const {
        return hasKey(key) ? numberValue(key) : fallback;
    }

private:
    template <typename T>
    std::vector<T> splitNumericArray(const std::string& key) const {
        const std::size_t value_start = valueStart(key);
        if (text_[value_start] != '[') {
            throw std::runtime_error("Expected numeric array for key: " + key);
        }
        std::vector<T> result;
        std::size_t pos = value_start + 1;
        while (pos < text_.size()) {
            pos = skipWhitespace(pos);
            if (text_[pos] == ']') break;
            std::size_t end = pos;
            while (end < text_.size() &&
                   (std::isdigit(static_cast<unsigned char>(text_[end])) || text_[end] == '-' ||
                    text_[end] == '+' || text_[end] == '.')) {
                ++end;
            }
            result.push_back(static_cast<T>(std::stod(text_.substr(pos, end - pos))));
            pos = skipWhitespace(end);
            if (text_[pos] == ',') ++pos;
        }
        return result;
    }

    std::size_t valueStart(const std::string& key) const {
        const std::string pattern = "\"" + key + "\"";
        const std::size_t key_pos = text_.find(pattern);
        if (key_pos == std::string::npos) {
            throw std::runtime_error("Missing key in config: " + key);
        }
        const std::size_t colon = text_.find(':', key_pos + pattern.size());
        if (colon == std::string::npos) {
            throw std::runtime_error("Missing colon in config for key: " + key);
        }
        return skipWhitespace(colon + 1);
    }

    std::size_t skipWhitespace(std::size_t pos) const {
        while (pos < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos]))) {
            ++pos;
        }
        return pos;
    }

    std::string parseString(std::size_t quote_pos) const {
        std::string out;
        for (std::size_t pos = quote_pos + 1; pos < text_.size(); ++pos) {
            const char ch = text_[pos];
            if (ch == '\\') {
                ++pos;
                if (pos >= text_.size()) break;
                out.push_back(text_[pos]);
                continue;
            }
            if (ch == '"') {
                return out;
            }
            out.push_back(ch);
        }
        throw std::runtime_error("Unterminated string in config");
    }

    std::size_t nextAfterString(std::size_t quote_pos) const {
        for (std::size_t pos = quote_pos + 1; pos < text_.size(); ++pos) {
            if (text_[pos] == '\\') {
                ++pos;
                continue;
            }
            if (text_[pos] == '"') {
                return pos + 1;
            }
        }
        return text_.size();
    }

    std::string text_;
};

struct EpisodeLogger {
    explicit EpisodeLogger(const std::string& path) : stream(path, std::ios::out | std::ios::trunc) {
        if (!path.empty() && !stream.is_open()) {
            throw std::runtime_error("Failed to open log file: " + path);
        }
    }

    void writeEpisode(int episode_index,
                      double duration_s,
                      double mean_5,
                      double mean_20,
                      int cycle_index,
                      const std::string& cycle_condition,
                      bool training_eligible,
                      bool training_delivered,
                      const std::string& training_sequence_type,
                      const std::string& training_sequence,
                      const std::string& training_electrodes,
                      const std::string& termination_reason,
                      double theta_rad) {
        if (!stream.is_open()) return;
        stream << std::fixed << std::setprecision(6)
               << "{\"episode_index\":" << episode_index
               << ",\"time_balanced_s\":" << duration_s
               << ",\"mean_5_s\":" << mean_5
               << ",\"mean_20_s\":" << mean_20
               << ",\"cycle_index\":" << cycle_index
               << ",\"cycle_condition\":\"" << cycle_condition << "\""
               << ",\"training_eligible\":" << (training_eligible ? "true" : "false")
               << ",\"training_delivered\":" << (training_delivered ? "true" : "false")
               << ",\"training_sequence_type\":\"" << training_sequence_type << "\""
               << ",\"training_sequence\":\"" << training_sequence << "\""
               << ",\"training_electrodes\":\"" << training_electrodes << "\""
               << ",\"termination_reason\":\"" << termination_reason << "\""
               << ",\"terminal_theta_rad\":" << theta_rad
               << "}\n";
        stream.flush();
    }

    void writeSummary(const std::string& failure_reason) {
        if (!stream.is_open()) return;
        stream << "{\"event\":\"experiment_summary\",\"failure_reason\":\""
               << failure_reason
               << "\"}\n";
        stream.flush();
    }

    std::ofstream stream;
};

struct AppState {
    AppState(const RunConfig& config, GameWindow* ui_window)
        : window_start(SteadyClock::now()),
          experiment_start(window_start),
          training_until(window_start),
          task(config.timestep_seconds),
          detector(config.channel_count, config.detector),
          trainer(config.training_pattern_names, 0.3, 0.3, 10.0, config.random_seed),
          logger(config.log_path),
          window(ui_window) {
        spike_counts.resize(config.channel_count, 0);
        if (!config.cycle_condition_schedule.empty()) {
            current_cycle_condition = config.cycle_condition_schedule.front();
        }
    }

    SteadyClock::time_point window_start;
    SteadyClock::time_point experiment_start;
    SteadyClock::time_point training_until;
    CartpoleTask task;
    SpikeDetector detector;
    TrainingController trainer;
    EpisodeLogger logger;
    GameWindow* window;
    std::vector<std::uint32_t> spike_counts;
    double left_rate = 0.0;
    double right_rate = 0.0;
    double left_phase = 0.0;
    double right_phase = 0.0;
    bool was_active_phase = false;
    int episode_index = 0;
    int current_cycle_index = 0;
    bool first_cycle_passed = false;
    std::string last_training_sequence = "none";
    std::string last_training_sequence_type = "none";
    std::string last_training_electrodes = "[]";
    std::string current_cycle_condition = "null";
};

void on_sigint(int) {
    g_running = false;
}

void resetWindow(AppState& state, SteadyClock::time_point now) {
    state.window_start = now;
    state.detector.resetCounts();
}

void resetEpisodeState(AppState& state) {
    state.task.reset();
    state.left_rate = 0.0;
    state.right_rate = 0.0;
    state.left_phase = 0.0;
    state.right_phase = 0.0;
}

double sumCounts(const std::vector<std::uint32_t>& spike_counts, const std::vector<int>& channels) {
    double total = 0.0;
    for (int channel : channels) {
        if (channel >= 0 && static_cast<std::size_t>(channel) < spike_counts.size()) {
            total += spike_counts[static_cast<std::size_t>(channel)];
        }
    }
    return total;
}

TrainingCondition parseTrainingCondition(const std::string& value) {
    if (value == "random") return TrainingCondition::Random;
    if (value == "adaptive") return TrainingCondition::Adaptive;
    return TrainingCondition::Null;
}

const char* trainingConditionName(TrainingCondition condition) {
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

const char* experimentModeName(ExperimentMode mode) {
    switch (mode) {
        case ExperimentMode::Cycled:
            return "cycled";
        case ExperimentMode::ContinuousAdaptive:
            return "continuous_adaptive";
        case ExperimentMode::NullCondition:
            return "null";
        case ExperimentMode::RandomCondition:
            return "random";
        case ExperimentMode::AdaptiveCondition:
            return "adaptive";
    }
    return "cycled";
}

int currentCycleIndex(const RunConfig& config, double elapsed_seconds) {
    const double cycle_span = config.active_cycle_duration_s + config.rest_duration_s;
    if (cycle_span <= 0.0) {
        return 0;
    }
    const int raw_index = static_cast<int>(elapsed_seconds / cycle_span);
    return std::clamp(raw_index, 0, (std::max)(config.num_cycles - 1, 0));
}

TrainingCondition currentTrainingCondition(const RunConfig& config, int cycle_index) {
    if (config.cycle_condition_schedule.empty()) {
        return TrainingCondition::Adaptive;
    }
    const int clamped_index = std::clamp(
        cycle_index,
        0,
        static_cast<int>(config.cycle_condition_schedule.size() - 1));
    return parseTrainingCondition(config.cycle_condition_schedule[static_cast<std::size_t>(clamped_index)]);
}

bool isActivePhase(const RunConfig& config, double elapsed_seconds) {
    const double cycle_span = config.active_cycle_duration_s + config.rest_duration_s;
    if (cycle_span <= 0.0) return true;
    const double offset = std::fmod(elapsed_seconds, cycle_span);
    return offset < config.active_cycle_duration_s;
}

int trainingWindowForSequenceType(const std::string& sequence_type) {
    if (sequence_type == "random5") {
        return 200;
    }
    if (sequence_type == "adaptive_pair") {
        return 300;
    }
    return 0;
}

std::vector<std::pair<int, int>> buildAdaptivePatternElectrodePairs(const std::vector<int>& electrodes) {
    std::vector<std::pair<int, int>> result;
    for (std::size_t first = 0; first < electrodes.size(); ++first) {
        for (std::size_t second = first + 1; second < electrodes.size(); ++second) {
            result.emplace_back(electrodes[first], electrodes[second]);
        }
    }
    return result;
}

std::vector<int> parsePipeSeparatedIntList(const std::string& text) {
    std::vector<int> values;
    if (text.empty()) return values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stoi(token));
        }
    }
    return values;
}

struct RequestedSequence {
    std::string sequence_name;
    std::string sequence_type;
    std::vector<int> training_electrodes;
};

RequestedSequence requestRandomTrainingSequenceFromPython() {
    std::cout << "[TRAINING_REQUEST] random5" << std::endl;
    std::string response;
    if (!std::getline(std::cin, response)) {
        throw std::runtime_error("Failed to receive random training sequence from Python");
    }
    const std::string ok_prefix = "training_sequence|";
    const std::string error_prefix = "training_sequence_error|";
    if (response.rfind(error_prefix, 0) == 0) {
        throw std::runtime_error(response.substr(error_prefix.size()));
    }
    if (response.rfind(ok_prefix, 0) != 0) {
        throw std::runtime_error("Unexpected training sequence response: " + response);
    }
    std::vector<std::string> parts;
    std::stringstream stream(response);
    std::string token;
    while (std::getline(stream, token, '|')) {
        parts.push_back(token);
    }
    if (parts.size() != 4) {
        throw std::runtime_error("Malformed training sequence response: " + response);
    }
    RequestedSequence requested;
    requested.sequence_name = parts[1];
    requested.sequence_type = parts[2];
    requested.training_electrodes = parsePipeSeparatedIntList(parts[3]);
    return requested;
}

double clampUnitForce(double force_scale_n, double left_rate, double right_rate) {
    const double unit_force = std::clamp(left_rate - right_rate, -1.0, 1.0);
    return unit_force * force_scale_n;
}

void emitRatePulse(const std::string& sequence_name, double frequency_hz, double dt_seconds, double* phase) {
    if (sequence_name.empty() || phase == nullptr) return;
    *phase += frequency_hz * dt_seconds;
    while (*phase >= 1.0) {
        maxlab::verifyStatus(maxlab::sendSequence(sequence_name.c_str()));
        *phase -= 1.0;
    }
}

void waitForStartSignal(bool enable_sync) {
    if (!enable_sync) {
        std::cout << "[SYNC] Sync disabled, starting immediately" << std::endl;
        return;
    }

    std::cout << "[SYNC] Waiting for start signal from Python via stdin..." << std::endl;
    std::string signal;
    if (std::getline(std::cin, signal)) {
        std::cout << "[SYNC] Start signal received, beginning cartpole loop" << std::endl;
    } else {
        std::cerr << "[SYNC] Warning: failed to read from stdin, starting immediately" << std::endl;
    }
}

std::string joinIntVector(const std::vector<int>& values) {
    if (values.empty()) return "[]";
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) oss << ",";
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

void updateWindow(AppState& state, const RunConfig& config, SteadyClock::time_point now) {
    state.detector.getCounts(&state.spike_counts);

    const double elapsed_seconds =
        std::chrono::duration<double>(now - state.experiment_start).count();
    const int cycle_index = currentCycleIndex(config, elapsed_seconds);
    const TrainingCondition cycle_condition = currentTrainingCondition(config, cycle_index);
    const bool active_phase = isActivePhase(config, elapsed_seconds);
    state.current_cycle_index = cycle_index;
    state.current_cycle_condition = trainingConditionName(cycle_condition);

    const auto finalize_episode = [&](bool allow_training_delivery, const std::string& termination_reason) {
        const double reward_seconds = state.task.getTimeBalanced();
        if (reward_seconds <= 0.0) {
            resetEpisodeState(state);
            return;
        }

        const TrainingDecision decision =
            state.trainer.onEpisodeEnd(reward_seconds, cycle_condition, allow_training_delivery);
        ++state.episode_index;

        std::string training_sequence = decision.sequence_name.empty() ? "none" : decision.sequence_name;
        std::string training_sequence_type = decision.sequence_type.empty() ? "none" : decision.sequence_type;
        std::vector<int> training_electrodes;

        if (decision.delivered) {
            if (training_sequence_type == "adaptive_pair" && decision.pattern_index >= 0) {
                const auto adaptive_pairs = buildAdaptivePatternElectrodePairs(config.training_stim_electrodes);
                const std::size_t pair_index = static_cast<std::size_t>(decision.pattern_index);
                if (pair_index < adaptive_pairs.size()) {
                    training_electrodes = {
                        adaptive_pairs[pair_index].first,
                        adaptive_pairs[pair_index].second,
                    };
                }
            } else if (training_sequence_type == "random5") {
                const RequestedSequence requested = requestRandomTrainingSequenceFromPython();
                training_sequence = requested.sequence_name;
                training_sequence_type = requested.sequence_type;
                training_electrodes = requested.training_electrodes;
            }
        }

        if (cycle_index == 0 && reward_seconds > config.first_cycle_pass_threshold_s) {
            state.first_cycle_passed = true;
        }

        const std::string training_electrodes_text = joinIntVector(training_electrodes);
        state.logger.writeEpisode(
            state.episode_index,
            reward_seconds,
            decision.mean_5,
            decision.mean_20,
            cycle_index,
            trainingConditionName(cycle_condition),
            decision.eligible,
            decision.delivered,
            training_sequence_type,
            training_sequence,
            training_electrodes_text,
            termination_reason,
            state.task.getPoleAngleRad());

        std::cout << std::fixed << std::setprecision(2)
                  << "[EPISODE] index=" << state.episode_index
                  << " cycle=" << cycle_index
                  << " condition=" << trainingConditionName(cycle_condition)
                  << " duration_s=" << reward_seconds
                  << " mean5=" << decision.mean_5
                  << " mean20=" << decision.mean_20
                  << " training=" << (decision.delivered ? training_sequence : "none")
                  << std::endl;

        if (decision.delivered) {
            maxlab::verifyStatus(maxlab::sendSequence(training_sequence.c_str()));
            state.training_until = now + std::chrono::milliseconds(trainingWindowForSequenceType(training_sequence_type));
            state.last_training_sequence = training_sequence;
            state.last_training_sequence_type = training_sequence_type;
            state.last_training_electrodes = training_electrodes_text;
        } else {
            state.last_training_sequence = "none";
            state.last_training_sequence_type = "none";
            state.last_training_electrodes = "[]";
        }

        resetEpisodeState(state);
    };

    if (!active_phase && state.was_active_phase) {
        finalize_episode(false, "cycle_end");
        if (cycle_index == 0 && !state.first_cycle_passed) {
            state.logger.writeSummary("first_cycle_under_10s_requires_reselection");
            throw std::runtime_error("first_cycle_under_10s_requires_reselection");
        }
    }
    state.was_active_phase = active_phase;

    if (elapsed_seconds >= config.experiment_duration_s) {
        finalize_episode(false, "experiment_end");
        g_running = false;
        return;
    }

    if (!active_phase) {
        if (state.window != nullptr) {
            state.window->setState(0.0f, 0.0f, 0.0f, 0.0f);
            state.window->setTelemetry(
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                state.episode_index,
                false,
                state.current_cycle_index,
                false,
                state.current_cycle_condition,
                state.last_training_sequence_type,
                state.last_training_sequence,
                state.last_training_electrodes);
        }
        resetWindow(state, now);
        return;
    }

    if (now < state.training_until) {
        resetWindow(state, now);
        return;
    }

    const double left_count = sumCounts(state.spike_counts, config.decoding_left_channels);
    const double right_count = sumCounts(state.spike_counts, config.decoding_right_channels);
    state.left_rate = (1.0 - config.ema_alpha) * state.left_rate + config.ema_alpha * left_count;
    state.right_rate = (1.0 - config.ema_alpha) * state.right_rate + config.ema_alpha * right_count;

    const double force_newtons = clampUnitForce(config.force_scale_n, state.left_rate, state.right_rate);
    const bool terminal = state.task.step(force_newtons);

    const double theta = state.task.getPoleAngleRad();
    const double frequency_left =
        config.encoding_scale_a * std::pow(config.encoding_scale_b - std::sin(theta), 2.0);
    const double frequency_right =
        config.encoding_scale_a * std::pow(config.encoding_scale_b + std::sin(theta), 2.0);

    emitRatePulse(config.encoding_left_sequence, frequency_left, config.timestep_seconds, &state.left_phase);
    emitRatePulse(config.encoding_right_sequence, frequency_right, config.timestep_seconds, &state.right_phase);

    if (state.window != nullptr) {
        state.window->setState(
            static_cast<float>(state.task.getCartPosition()),
            static_cast<float>(theta),
            static_cast<float>(state.task.getTimeBalanced()),
            static_cast<float>(force_newtons));
        state.window->setTelemetry(
            static_cast<float>(state.left_rate),
            static_cast<float>(state.right_rate),
            static_cast<float>(frequency_left),
            static_cast<float>(frequency_right),
            state.episode_index,
            now < state.training_until,
            state.current_cycle_index,
            true,
            state.current_cycle_condition,
            state.last_training_sequence_type,
            state.last_training_sequence,
            state.last_training_electrodes);
    }

    if (terminal) {
        (void)theta;
        finalize_episode(true, "terminal");
    }

    resetWindow(state, now);
}

struct FrameSamplesView {
    const float* samples = nullptr;
    std::size_t channel_count = 0;
};

bool extractFrameSamples(const maxlab::FilteredFrameData& frame_data, FrameSamplesView* out) {
#ifdef MAXONE_USE_RAW_SAMPLES
    out->samples = frame_data.analogSamples;
    out->channel_count = frame_data.frameInfo.channelCount;
    return (out->samples != nullptr && out->channel_count > 0);
#else
    (void)frame_data;
    (void)out;
    return false;
#endif
}

RunConfig loadConfig(const std::string& config_path) {
    std::ifstream stream(config_path);
    if (!stream.is_open()) {
        throw std::runtime_error("Unable to open config file: " + config_path);
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    JsonParser parser(buffer.str());

    RunConfig config;
    config.config_path = config_path;
    config.target_well = static_cast<uint8_t>(parser.numberValue("target_well"));
    config.read_window_ms = static_cast<int>(parser.numberValue("read_window_ms"));
    config.training_window_ms = static_cast<int>(parser.numberValue("training_window_ms"));
    config.show_gui = parser.boolValue("show_gui");
    config.wait_for_sync = parser.boolValue("wait_for_sync");
    config.channel_count = static_cast<std::size_t>(parser.numberValue("channel_count"));
    config.experiment_duration_s = parser.numberValue("experiment_duration_s");
    config.num_cycles = static_cast<int>(parser.numberValue("num_cycles"));
    config.active_cycle_duration_s = parser.numberValue("active_cycle_duration_s");
    config.rest_duration_s = parser.numberValue("rest_duration_s");
    config.first_cycle_pass_threshold_s = parser.numberValue("first_cycle_pass_threshold_s");
    config.encoding_scale_a = parser.numberValue("encoding_scale_a");
    config.encoding_scale_b = parser.numberValue("encoding_scale_b");
    config.ema_alpha = parser.numberValue("ema_alpha");
    config.force_scale_n = parser.numberValue("force_scale_n");
    config.detector.sample_rate_hz = parser.numberValue("sample_rate_hz");
    config.detector.threshold_multiplier = static_cast<float>(parser.numberValue("threshold_multiplier"));
    config.detector.min_threshold = static_cast<float>(parser.numberValue("min_threshold"));
    config.detector.refractory_samples = static_cast<int>(parser.numberValue("refractory_samples"));
    config.decoding_left_channels = parser.intArrayValue("decoding_left_channels");
    config.decoding_right_channels = parser.intArrayValue("decoding_right_channels");
    config.encoding_left_sequence = parser.stringValue("encoding_left_sequence");
    config.encoding_right_sequence = parser.stringValue("encoding_right_sequence");
    config.cycle_condition_schedule = parser.stringArrayValue("cycle_condition_schedule");
    config.training_pattern_names =
        parser.stringArrayValueOr("adaptive_training_pattern_names",
                                  parser.stringArrayValueOr("training_pattern_names", {}));
    config.encoding_stim_electrodes = parser.intArrayValueOr("encoding_stim_electrodes", {});
    config.training_stim_electrodes = parser.intArrayValueOr("training_stim_electrodes", {});
    config.decoding_left_electrodes = parser.intArrayValueOr("decoding_left_electrodes", {});
    config.decoding_right_electrodes = parser.intArrayValueOr("decoding_right_electrodes", {});
    config.stim_pulse_amplitude_mV = parser.numberValueOr("stim_pulse_amplitude_mV", 0.0);
    config.stim_phase_us = static_cast<int>(parser.numberValueOr("stim_phase_us", 0.0));
    config.stim_inter_pulse_interval_ms = parser.numberValueOr("stim_inter_pulse_interval_ms", 0.0);
    config.stim_training_frequency_hz = parser.numberValueOr("stim_training_frequency_hz", 0.0);
    if (parser.hasKey("artifact_filter_mode")) {
        config.artifact_filter_mode = parser.stringValue("artifact_filter_mode");
    }
    config.log_path = parser.stringValue("log_path");
    config.random_seed = static_cast<std::uint32_t>(parser.numberValue("random_seed"));

    const std::string mode = parser.stringValue("mode");
    if (mode == "cycled") {
        config.mode = ExperimentMode::Cycled;
    } else if (mode == "continuous_adaptive") {
        config.mode = ExperimentMode::ContinuousAdaptive;
    } else if (mode == "null") {
        config.mode = ExperimentMode::NullCondition;
    } else if (mode == "random") {
        config.mode = ExperimentMode::RandomCondition;
    } else if (mode == "adaptive") {
        config.mode = ExperimentMode::AdaptiveCondition;
    } else {
        throw std::runtime_error("Unsupported mode in config: " + mode);
    }
    config.timestep_seconds = static_cast<double>(config.read_window_ms) / 1000.0;
    return config;
}

int runGameLoop(const RunConfig& config, GameWindow* window) {
    try {
        maxlab::checkVersions();
        waitForStartSignal(config.wait_for_sync);
        if (config.artifact_filter_mode == "salpa") {
            std::cout
                << "[WARN] artifact_filter_mode=salpa requested; SALPA runtime path is not yet implemented, "
                << "falling back to IIR stream filter."
                << std::endl;
        }
        maxlab::verifyStatus(maxlab::DataStreamerFiltered_open(maxlab::FilterType::IIR));

        AppState state(config, window);

        while (g_running.load()) {
            maxlab::FilteredFrameData frame_data;
            const maxlab::Status status = maxlab::DataStreamerFiltered_receiveNextFrame(&frame_data);

            if (status == maxlab::Status::MAXLAB_NO_FRAME) {
                const auto now = SteadyClock::now();
                if (now - state.window_start >= std::chrono::milliseconds(config.read_window_ms)) {
                    updateWindow(state, config, now);
                }
                continue;
            }

            if (status != maxlab::Status::MAXLAB_OK) {
                continue;
            }
            if (frame_data.frameInfo.corrupted) continue;
            if (frame_data.frameInfo.well_id != config.target_well) continue;

            FrameSamplesView samples;
            if (extractFrameSamples(frame_data, &samples)) {
                state.detector.processFrame(samples.samples, samples.channel_count);
            } else {
                for (uint64_t i = 0; i < frame_data.spikeCount; ++i) {
                    const maxlab::SpikeEvent& sp = frame_data.spikeEvents[i];
                    state.detector.addSpikeEvent(sp.channel);
                }
            }

            const auto now = SteadyClock::now();
            if (now - state.window_start >= std::chrono::milliseconds(config.read_window_ms)) {
                updateWindow(state, config, now);
            }
        }

        maxlab::verifyStatus(maxlab::DataStreamerFiltered_close());
        return 0;
    } catch (const std::exception& e) {
        (void)maxlab::DataStreamerFiltered_close();
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: maxone_with_filter <config.json>" << std::endl;
        return 2;
    }

    const RunConfig config = loadConfig(argv[1]);
    std::signal(SIGINT, on_sigint);

    std::cout << "[INFO] cartpole loop config=" << config.config_path
              << " target_well=" << static_cast<int>(config.target_well)
              << " read_window_ms=" << config.read_window_ms
              << " training_window_ms=" << config.training_window_ms
              << " artifact_filter_mode=" << config.artifact_filter_mode
              << " mode=" << experimentModeName(config.mode)
              << " duration_s=" << config.experiment_duration_s
              << std::endl;

#ifdef USE_QT
    if (config.show_gui) {
        QApplication app(argc, argv);
        GameWindow window;
        window.setRuntimeInfo(
            joinIntVector(config.encoding_stim_electrodes),
            joinIntVector(config.training_stim_electrodes),
            joinIntVector(config.decoding_left_electrodes),
            joinIntVector(config.decoding_right_electrodes),
            static_cast<float>(config.stim_pulse_amplitude_mV * 1000.0),
            static_cast<int>(config.training_pattern_names.size()),
            config.read_window_ms,
            config.training_window_ms);
        window.show();

        std::thread worker([&]() {
            runGameLoop(config, &window);
            QMetaObject::invokeMethod(&app, "quit", Qt::QueuedConnection);
        });

        const int rc = app.exec();
        g_running = false;
        if (worker.joinable()) worker.join();
        return rc;
    }
#else
    if (config.show_gui) {
        std::cerr << "Qt UI is disabled (build without USE_QT); running headless." << std::endl;
    }
#endif

    return runGameLoop(config, nullptr);
}
