// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef APP_PIPELINE_HPP_
#define APP_PIPELINE_HPP_

/**
 * @file pipeline.hpp
 * @brief The CCTV capture/inference loop and the snapshot it publishes.
 *
 * Two background threads run independently:
 *
 *   video: capture -> detect -> draw boxes -> JPEG encode -> publish Snapshot
 *   audio: capture -> acoustic detection -> publish the latest verdict
 *
 * Blocking audio reads run independently of video. HTTP handlers read the
 * latest completed snapshot rather than capturing per request.
 */

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "app/annotator.hpp"
#include "app/detector.hpp"
#include "app/host_metrics.hpp"
#include "app/jpeg_encoder.hpp"
#include "app/sources.hpp"

namespace mep_cctv {
namespace app {

/** @brief Value snapshot of the latest processed frame. */
struct Snapshot {
  std::vector<uint8_t> jpeg;
  int detection_count = 0;
  bool person_present = false;
  float audio_score = 0.0f;
  double fps = 0.0;
  uint64_t sequence = 0;
  // What the detection above cost the machine. -1 means this platform cannot
  // report the value; the page hides it rather than drawing a zero.
  double cpu_percent = -1.0;
  double rss_mb = -1.0;
  double temp_c = -1.0;
};

/** @brief How the pipeline should be configured at startup. */
struct PipelineConfig {
  std::string video_device = "/dev/video0";
  std::string audio_device = "default";
  std::string yolo_model;
  int width = 640;
  int height = 480;
  int sample_rate = 16000;
  // Samples per blocking microphone read; the default chunk spans 100 ms.
  int audio_chunk = 1600;
  int jpeg_quality = 0;  // 0 selects JpegEncoder::kDefaultQuality
};

/** @brief Runs the capture loop and publishes snapshots. */
class Pipeline {
 public:
  explicit Pipeline(const PipelineConfig& config);
  ~Pipeline();

  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  /**
   * @brief Start the video and audio worker threads.
   *
   * @return True on start; false when already running or the engine is
   * unusable.
   *
   * @side_effects VIOLATION: Non-referentially transparent
   * @side_effects_reason Spawns workers that update shared capture results.
   * @side_effects_what   Thread creation; continuous device I/O.
   * @side_effects_impact Runs until Stop() or destruction.
   * @side_effects_alternatives Capturing on the request thread would tie the
   *                      frame rate to browser polling and re-open the camera
   *                      per viewer.
   */
  bool Start();

  /**
   * @brief Ask both workers to finish and join them.
   *
   * @return True when a running pipeline was stopped.
   *
   * @side_effects VIOLATION: Non-referentially transparent
   * @side_effects_reason Signals and joins both worker threads.
   * @side_effects_what   Thread lifetime management.
   * @side_effects_impact Blocks until the in-flight frame completes.
   * @side_effects_alternatives None -- shutdown is inherently effectful.
   */
  bool Stop();

  /** @brief Copy of the most recent snapshot; empty jpeg before the first. */
  Snapshot Latest() const;

  /** @brief Name of the active video source, e.g. "v4l2:/dev/video0". */
  std::string VideoSourceName() const;

  /** @brief Name of the active audio source. */
  std::string AudioSourceName() const;

  /** @brief The detector, for reporting backend/version on /status. */
  const Detector& GetDetector() const;

 private:
  int RunVideoLoop();
  int RunAudioLoop();
  int ProcessFrame();
  int ProcessAudioChunk();
  double UpdateFps();

  /** @brief Refresh host metrics at most once a second; returns the current
   *         reading, which may be the one taken on an earlier frame. */
  HostSample UpdateHostMetrics();

  const PipelineConfig config_;
  Detector detector_;
  Annotator annotator_;
  JpegEncoder encoder_;
  std::unique_ptr<VideoSource> video_;
  std::unique_ptr<AudioSource> audio_;

  mutable std::mutex mutex_;
  Snapshot latest_;
  std::atomic<bool> running_;
  std::thread video_worker_;
  std::thread audio_worker_;
  // Audio publishes scalar results without acquiring the snapshot mutex.
  std::atomic<bool> sound_detected_;
  std::atomic<float> audio_score_;
  uint64_t sequence_;
  double last_frame_seconds_;
  double fps_;
  // Metrics are refreshed on the video thread at most once per second.
  HostMetrics host_metrics_;
  HostSample host_sample_;
  double last_metrics_seconds_;
};

}  // namespace app
}  // namespace mep_cctv

#endif  // APP_PIPELINE_HPP_
