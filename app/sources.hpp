// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef APP_SOURCES_HPP_
#define APP_SOURCES_HPP_

/**
 * @file sources.hpp
 * @brief Camera and microphone abstractions, plus their synthetic fallbacks.
 *
 * This header is platform-independent by design: the real capture code lives
 * in per-platform translation units picked by CMake
 * (video_capture_v4l2.cc on Linux, video_capture_null.cc elsewhere), so no
 * #ifdef ever reaches a header.
 *
 * OpenVideoCapture()/OpenAudioCapture() return nullptr when the platform has
 * no real device support; callers fall back to the synthetic sources so the
 * app runs anywhere, which is what makes it demonstrable on a laptop.
 */

#include <memory>
#include <string>
#include <vector>

#include "app/frame.hpp"

namespace mep_cctv {
namespace app {

/** @brief A source of video frames. */
class VideoSource {
 public:
  virtual ~VideoSource() = default;

  /**
   * @brief Return the next frame, or an invalid Frame when none is available.
   *
   * @side_effects VIOLATION: Non-referentially transparent
   * @side_effects_reason Reads from a capture device or advances an internal
   *                      animation clock; successive calls differ.
   * @side_effects_what   Device I/O (V4L2 dequeue) or counter increment.
   * @side_effects_impact Blocks until a frame is ready, bounded by the device
   *                      frame interval.
   * @side_effects_alternatives A pure interface cannot model a live camera.
   */
  virtual Frame Read() = 0;

  /** @brief Human-readable source name, for the status endpoint. */
  virtual std::string Name() const = 0;
};

/** @brief A source of mono PCM audio. */
class AudioSource {
 public:
  virtual ~AudioSource() = default;

  /**
   * @brief Return the next chunk of mono float samples in [-1, 1].
   *
   * @side_effects VIOLATION: Non-referentially transparent
   * @side_effects_reason Drains a live capture buffer.
   * @side_effects_what   Device I/O (ALSA read).
   * @side_effects_impact Blocks until the chunk is filled.
   * @side_effects_alternatives A pure interface cannot model a live mic.
   */
  virtual std::vector<float> Read() = 0;

  /** @brief Sample rate in Hz. */
  virtual int SampleRate() const = 0;

  /** @brief Human-readable source name, for the status endpoint. */
  virtual std::string Name() const = 0;
};

/**
 * @brief Deterministic moving-pattern video, used when no camera is present.
 *
 * Generates a drifting gradient with a moving bright block so the web page
 * visibly updates and the pipeline can be exercised without hardware.
 */
class SyntheticVideoSource : public VideoSource {
 public:
  SyntheticVideoSource(int width, int height) noexcept;

  Frame Read() override;
  std::string Name() const override;

 private:
  const int width_;
  const int height_;
  int tick_;
};

/** @brief Silent audio, used when no microphone is present. */
class SilentAudioSource : public AudioSource {
 public:
  SilentAudioSource(int sample_rate, int chunk_samples) noexcept;

  std::vector<float> Read() override;
  int SampleRate() const override;
  std::string Name() const override;

 private:
  const int sample_rate_;
  const int chunk_samples_;
};

/**
 * @brief Open a real camera, or return nullptr when unavailable.
 *
 * @param device  Device path (e.g. "/dev/video0"); ignored off Linux.
 * @param width   Requested capture width in pixels.
 * @param height  Requested capture height in pixels.
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason Opens a device file and negotiates a capture format.
 * @side_effects_what   open(2)/ioctl(2) on a V4L2 device; mmap of buffers.
 * @side_effects_impact Holds the camera until the returned object is
 *                      destroyed; a second opener will fail.
 * @side_effects_alternatives None -- device acquisition is inherently
 *                      effectful.
 */
std::unique_ptr<VideoSource> OpenVideoCapture(const std::string& device,
                                              int width, int height);

/**
 * @brief Open a real microphone, or return nullptr when unavailable.
 *
 * @param device       ALSA device name (e.g. "default"); ignored off Linux.
 * @param sample_rate  Requested sample rate in Hz.
 * @param chunk        Samples returned per Read() call.
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason Opens and configures an ALSA capture handle.
 * @side_effects_what   snd_pcm_open and friends.
 * @side_effects_impact Holds the device until destroyed.
 * @side_effects_alternatives None -- device acquisition is inherently
 *                      effectful.
 */
std::unique_ptr<AudioSource> OpenAudioCapture(const std::string& device,
                                              int sample_rate, int chunk);

}  // namespace app
}  // namespace mep_cctv

#endif  // APP_SOURCES_HPP_
