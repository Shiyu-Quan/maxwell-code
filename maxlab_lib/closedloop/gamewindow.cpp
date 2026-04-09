#ifdef USE_QT
#include "gamewindow.h"

#include <QPainter>
#include <QString>
#include <QTimer>

GameWindow::GameWindow(QWidget* parent)
    : QWidget(parent),
      cart_x_(0.0f),
      pole_angle_rad_(0.0f),
      time_balanced_seconds_(0.0f),
      force_newtons_(0.0f),
      left_rate_(0.0f),
      right_rate_(0.0f),
      left_frequency_hz_(0.0f),
      right_frequency_hz_(0.0f),
      episode_index_(0),
      training_active_(0),
      pulse_amplitude_micro_v_(0.0f),
      training_patterns_(0),
      read_window_ms_(0),
      training_window_ms_(0) {
    setWindowTitle("Cartpole Viewer");
    resize(920, 620);

    auto* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, QOverload<>::of(&QWidget::update));
    timer->start(16); // ~60 FPS
}

void GameWindow::setState(float cartX, float poleAngleRad, float timeBalancedSeconds, float forceNewtons) {
    cart_x_.store(cartX, std::memory_order_relaxed);
    pole_angle_rad_.store(poleAngleRad, std::memory_order_relaxed);
    time_balanced_seconds_.store(timeBalancedSeconds, std::memory_order_relaxed);
    force_newtons_.store(forceNewtons, std::memory_order_relaxed);
}

void GameWindow::setTelemetry(
    float leftRate,
    float rightRate,
    float leftFrequencyHz,
    float rightFrequencyHz,
    int episodeIndex,
    bool trainingActive,
    const std::string& lastTrainingSequence) {
    left_rate_.store(leftRate, std::memory_order_relaxed);
    right_rate_.store(rightRate, std::memory_order_relaxed);
    left_frequency_hz_.store(leftFrequencyHz, std::memory_order_relaxed);
    right_frequency_hz_.store(rightFrequencyHz, std::memory_order_relaxed);
    episode_index_.store(episodeIndex, std::memory_order_relaxed);
    training_active_.store(trainingActive ? 1 : 0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(text_mutex_);
    last_training_sequence_ = lastTrainingSequence;
}

void GameWindow::setRuntimeInfo(
    const std::string& encodingElectrodes,
    const std::string& trainingElectrodes,
    const std::string& decodingLeftElectrodes,
    const std::string& decodingRightElectrodes,
    float pulseAmplitudeMicroV,
    int trainingPatterns,
    int readWindowMs,
    int trainingWindowMs) {
    pulse_amplitude_micro_v_.store(pulseAmplitudeMicroV, std::memory_order_relaxed);
    training_patterns_.store(trainingPatterns, std::memory_order_relaxed);
    read_window_ms_.store(readWindowMs, std::memory_order_relaxed);
    training_window_ms_.store(trainingWindowMs, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(text_mutex_);
    encoding_electrodes_text_ = encodingElectrodes;
    training_electrodes_text_ = trainingElectrodes;
    decoding_left_electrodes_text_ = decodingLeftElectrodes;
    decoding_right_electrodes_text_ = decodingRightElectrodes;
}

void GameWindow::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.fillRect(rect(), QColor(18, 20, 24));

    const int w = width();
    const int h = height();
    const int ground_y = h - 70;
    const float cart_x = cart_x_.load(std::memory_order_relaxed);
    const float pole_angle = pole_angle_rad_.load(std::memory_order_relaxed);
    const float time_balanced = time_balanced_seconds_.load(std::memory_order_relaxed);
    const float force_newtons = force_newtons_.load(std::memory_order_relaxed);
    const float left_rate = left_rate_.load(std::memory_order_relaxed);
    const float right_rate = right_rate_.load(std::memory_order_relaxed);
    const float left_freq = left_frequency_hz_.load(std::memory_order_relaxed);
    const float right_freq = right_frequency_hz_.load(std::memory_order_relaxed);
    const int episode_index = episode_index_.load(std::memory_order_relaxed);
    const bool training_active = training_active_.load(std::memory_order_relaxed) != 0;
    const float pulse_amplitude_uv = pulse_amplitude_micro_v_.load(std::memory_order_relaxed);
    const int training_patterns = training_patterns_.load(std::memory_order_relaxed);
    const int read_window_ms = read_window_ms_.load(std::memory_order_relaxed);
    const int training_window_ms = training_window_ms_.load(std::memory_order_relaxed);
    std::string encoding_text;
    std::string training_text;
    std::string decoding_left_text;
    std::string decoding_right_text;
    std::string last_training_sequence;
    {
        std::lock_guard<std::mutex> lock(text_mutex_);
        encoding_text = encoding_electrodes_text_;
        training_text = training_electrodes_text_;
        decoding_left_text = decoding_left_electrodes_text_;
        decoding_right_text = decoding_right_electrodes_text_;
        last_training_sequence = last_training_sequence_;
    }

    painter.setPen(QPen(QColor(90, 95, 105), 2));
    painter.drawLine(0, ground_y, w, ground_y);

    const float cart_range = static_cast<float>(w) * 0.35f;
    const int cart_width = 90;
    const int cart_height = 28;
    const int cart_center_x = static_cast<int>(w / 2.0f + cart_x * cart_range);
    const QRect cart_rect(cart_center_x - cart_width / 2, ground_y - cart_height, cart_width, cart_height);

    painter.setBrush(QColor(84, 161, 255));
    painter.setPen(Qt::NoPen);
    painter.drawRoundedRect(cart_rect, 6, 6);

    const QPoint pivot(cart_center_x, ground_y - cart_height);
    const int pole_length = 170;
    const QPoint pole_tip(
        static_cast<int>(pivot.x() + std::sin(pole_angle) * pole_length),
        static_cast<int>(pivot.y() - std::cos(pole_angle) * pole_length));

    painter.setPen(QPen(QColor(240, 200, 80), 8, Qt::SolidLine, Qt::RoundCap));
    painter.drawLine(pivot, pole_tip);

    painter.setBrush(QColor(255, 244, 214));
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(pivot, 8, 8);

    const QRect panel_rect(16, 16, w - 32, 208);
    painter.setBrush(QColor(30, 34, 42, 210));
    painter.setPen(Qt::NoPen);
    painter.drawRoundedRect(panel_rect, 10, 10);

    painter.setPen(QColor(235, 238, 245));
    painter.drawText(28, 42, QString("Episode: %1").arg(episode_index));
    painter.drawText(170, 42, QString("Training: %1").arg(training_active ? "ACTIVE" : "idle"));
    painter.drawText(340, 42, QString("Read/Train Window: %1 / %2 ms").arg(read_window_ms).arg(training_window_ms));
    painter.drawText(620, 42, QString("Stim Amp: %1 uV/phase").arg(pulse_amplitude_uv, 0, 'f', 1));

    painter.drawText(28, 70, QString("Time Balanced: %1 s").arg(time_balanced, 0, 'f', 2));
    painter.drawText(240, 70, QString("Force: %1 N").arg(force_newtons, 0, 'f', 2));
    painter.drawText(380, 70, QString("Theta: %1 rad").arg(pole_angle, 0, 'f', 3));

    painter.drawText(28, 98, QString("Decoded Rate L/R: %1 / %2")
                                  .arg(left_rate, 0, 'f', 2)
                                  .arg(right_rate, 0, 'f', 2));
    painter.drawText(340, 98, QString("Encode Freq L/R: %1 / %2 Hz")
                                   .arg(left_freq, 0, 'f', 2)
                                   .arg(right_freq, 0, 'f', 2));
    painter.drawText(620, 98, QString("Training Patterns: %1").arg(training_patterns));

    painter.drawText(28, 126, QString("Encoding Electrodes: %1").arg(QString::fromStdString(encoding_text)));
    painter.drawText(28, 152, QString("Decoding Left Electrodes: %1").arg(QString::fromStdString(decoding_left_text)));
    painter.drawText(28, 178, QString("Decoding Right Electrodes: %1").arg(QString::fromStdString(decoding_right_text)));
    painter.drawText(28, 204, QString("Training Electrodes: %1").arg(QString::fromStdString(training_text)));
    painter.drawText(620, 204, QString("Last Training Seq: %1").arg(QString::fromStdString(last_training_sequence)));
}
#endif
