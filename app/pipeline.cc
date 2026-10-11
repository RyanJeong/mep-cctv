// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file pipeline.cc
 * @brief Pipeline implementation.
 */

#include "app/pipeline.hpp"

#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace mep_cctv {
namespace app {

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

// Smoothing factor for the reported frame rate; a raw per-frame reciprocal
// jitters too much to read.
constexpr double kFpsSmoothing = 0.9;

/** @brief Monotonic seconds since an arbitrary epoch. */
double NowSeconds() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

}  // namespace

Pipeline::Pipeline(const PipelineConfig& config)
    : config_(config),
      detector_(config.yolo_model, Detector::kDefaultConfThreshold,
                Detector::kDefaultIouThreshold),
      encoder_(config.jpeg_quality),
      running_(false),
      sound_detected_(false),
      audio_score_(0.0f),
      sequence_(0),
      last_frame_seconds_(0.0),
      fps_(0.0),
      last_metrics_seconds_(0.0) {
  video_ = OpenVideoCapture(config.video_device, config.width, config.height);
  if (video_ == nullptr)
    video_.reset(new SyntheticVideoSource(config.width, config.height));

  audio_ = OpenAudioCapture(config.audio_device, config.sample_rate,
                            config.audio_chunk);

  if (audio_ == nullptr)
    audio_.reset(new SilentAudioSource(config.sample_rate, config.audio_chunk));
}

Pipeline::~Pipeline() { Stop(); }

bool Pipeline::Start() {
  if (!detector_.IsReady() || running_.load()) return false;

  running_.store(true);
  video_worker_ = std::thread([this]() { return this->RunVideoLoop(); });
  audio_worker_ = std::thread([this]() { return this->RunAudioLoop(); });

  return true;
}

bool Pipeline::Stop() {
  if (!running_.load()) return false;

  running_.store(false);
  if (video_worker_.joinable()) video_worker_.join();

  // The audio thread may be inside a blocking capture read; it returns within
  // one chunk (100 ms by default), so this join is bounded.
  if (audio_worker_.joinable()) audio_worker_.join();
  return true;
}

int Pipeline::RunVideoLoop() {
  int frames = 0;
  while (running_.load()) frames += ProcessFrame();
  return frames;
}

int Pipeline::RunAudioLoop() {
  int chunks = 0;
  while (running_.load()) chunks += ProcessAudioChunk();
  return chunks;
}

int Pipeline::ProcessAudioChunk() {
  // Blocking capture -- deliberately off the video thread.
  const std::vector<float> samples = audio_->Read();
  if (samples.empty()) return 0;

  const cctv_audio_result result =
      detector_.DetectAudio(samples, audio_->SampleRate());
  sound_detected_.store(result.person_present != 0);
  audio_score_.store(result.score);
  return 1;
}

int Pipeline::ProcessFrame() {
  Frame frame = video_->Read();
  if (!frame.IsValid()) return 0;

  const std::vector<cctv_box> boxes = detector_.DetectObjects(frame);
  annotator_.Draw(&frame, boxes.data(), static_cast<int>(boxes.size()));

  Snapshot snapshot;
  snapshot.jpeg = encoder_.Encode(frame);
  snapshot.detection_count = static_cast<int>(boxes.size());
  snapshot.fps = UpdateFps();
  // Read whatever the audio thread published last; never wait for it.
  snapshot.person_present = sound_detected_.load();
  snapshot.audio_score = audio_score_.load();

  const HostSample host = UpdateHostMetrics();
  snapshot.cpu_percent = host.cpu_percent;
  snapshot.rss_mb = host.rss_mb;
  snapshot.temp_c = host.temp_c;

  {
    // Publish the completed snapshot without holding the lock during device
    // I/O.
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot.sequence = ++sequence_;
    latest_ = std::move(snapshot);
  }
  return 1;
}

HostSample Pipeline::UpdateHostMetrics() {
  // Reuse metrics for one second; check time without rereading /proc or /sys.
  const double now = NowSeconds();
  if (last_metrics_seconds_ > 0.0 && (now - last_metrics_seconds_) < 1.0)
    return host_sample_;

  host_sample_ = host_metrics_.Sample(host_sample_);
  last_metrics_seconds_ = now;
  return host_sample_;
}

double Pipeline::UpdateFps() {
  const double now = NowSeconds();
  if (last_frame_seconds_ > 0.0) {
    const double delta = now - last_frame_seconds_;
    if (delta > 0.0) {
      const double instant = 1.0 / delta;
      fps_ = (fps_ <= 0.0)
                 ? instant
                 : (kFpsSmoothing * fps_ + (1.0 - kFpsSmoothing) * instant);
    }
  }
  last_frame_seconds_ = now;
  return fps_;
}

Snapshot Pipeline::Latest() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return latest_;
}

std::string Pipeline::VideoSourceName() const { return video_->Name(); }

std::string Pipeline::AudioSourceName() const { return audio_->Name(); }

const Detector& Pipeline::GetDetector() const { return detector_; }

}  // namespace app
}  // namespace mep_cctv
