#ifndef GAMEWINDOW_H
#define GAMEWINDOW_H

#include <string>

#ifdef USE_QT
#include <atomic>
#include <mutex>
#include <QWidget>

class GameWindow : public QWidget {
public:
    explicit GameWindow(QWidget* parent = nullptr);
    void setState(float cartX, float poleAngleRad, float timeBalancedSeconds, float forceNewtons);
    void setTelemetry(
        float leftRate,
        float rightRate,
        float leftFrequencyHz,
        float rightFrequencyHz,
        int episodeIndex,
        bool trainingActive,
        const std::string& lastTrainingSequence);
    void setRuntimeInfo(
        const std::string& encodingElectrodes,
        const std::string& trainingElectrodes,
        const std::string& decodingLeftElectrodes,
        const std::string& decodingRightElectrodes,
        float pulseAmplitudeMicroV,
        int trainingPatterns,
        int readWindowMs,
        int trainingWindowMs);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    std::atomic<float> cart_x_;
    std::atomic<float> pole_angle_rad_;
    std::atomic<float> time_balanced_seconds_;
    std::atomic<float> force_newtons_;
    std::atomic<float> left_rate_;
    std::atomic<float> right_rate_;
    std::atomic<float> left_frequency_hz_;
    std::atomic<float> right_frequency_hz_;
    std::atomic<int> episode_index_;
    std::atomic<int> training_active_;
    std::atomic<float> pulse_amplitude_micro_v_;
    std::atomic<int> training_patterns_;
    std::atomic<int> read_window_ms_;
    std::atomic<int> training_window_ms_;
    std::mutex text_mutex_;
    std::string encoding_electrodes_text_;
    std::string training_electrodes_text_;
    std::string decoding_left_electrodes_text_;
    std::string decoding_right_electrodes_text_;
    std::string last_training_sequence_;
};
#else
class GameWindow {
public:
    explicit GameWindow(void* parent = nullptr) { (void)parent; }
    void setState(float, float, float, float) {}
    void setTelemetry(float, float, float, float, int, bool, const std::string&) {}
    void setRuntimeInfo(const std::string&, const std::string&, const std::string&, const std::string&,
                        float, int, int, int) {}
};
#endif

#endif // GAMEWINDOW_H
