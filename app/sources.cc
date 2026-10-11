// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file sources.cc
 * @brief Synthetic video and silent audio fallbacks.
 *
 * Platform-independent: the real capture implementations live in
 * video_capture_*.cc and audio_capture_*.cc, selected by CMake.
 */

#include "app/sources.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace mep_cctv {
namespace app {

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

// Size of the moving block drawn into the synthetic frame, in pixels.
constexpr int kBlockSize = 96;
// How far the block advances per frame.
constexpr int kBlockStep = 7;
// Full-scale 8-bit value.
constexpr int kMaxByte = 255;

/**
 * @brief Paint a solid white square that slides across the frame.
 *
 * @param frame   Frame to modify in place; must be valid.
 * @param offset  Horizontal position of the block's left edge.
 * @return Number of pixels written.
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason Writes into the caller's buffer.
 * @side_effects_what   Mutates frame->rgb.
 * @side_effects_impact Bounded by kBlockSize squared.
 * @side_effects_alternatives Returning a fresh buffer would copy the whole
 *                      frame for a small local edit.
 */
int DrawMovingBlock(Frame* frame, int offset) {
  int written = 0;
  const int y0 = (frame->height - kBlockSize) / 2;
  for (int y = y0; y < y0 + kBlockSize && y < frame->height; ++y) {
    if (y < 0) continue;

    for (int x = offset; x < offset + kBlockSize && x < frame->width; ++x) {
      if (x < 0) continue;

      const std::size_t i = (static_cast<std::size_t>(y) *
                                 static_cast<std::size_t>(frame->width) +
                             static_cast<std::size_t>(x)) *
                            3u;
      frame->rgb[i] = static_cast<uint8_t>(kMaxByte);
      frame->rgb[i + 1] = static_cast<uint8_t>(kMaxByte);
      frame->rgb[i + 2] = static_cast<uint8_t>(kMaxByte);
      ++written;
    }
  }
  return written;
}

}  // namespace

// ---------------------------------------------------------------------------
// SyntheticVideoSource
// ---------------------------------------------------------------------------

SyntheticVideoSource::SyntheticVideoSource(int width, int height) noexcept
    : width_(width), height_(height), tick_(0) {}

Frame SyntheticVideoSource::Read() {
  Frame frame;
  frame.width = width_;
  frame.height = height_;
  frame.rgb.assign(
      static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * 3u,
      0u);

  // A drifting diagonal gradient makes motion obvious in the browser.
  const int offset = (tick_ * kBlockStep) % (width_ > 0 ? width_ : 1);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const std::size_t i =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) +
           static_cast<std::size_t>(x)) *
          3u;
      frame.rgb[i] = static_cast<uint8_t>((x + offset) % kMaxByte);
      frame.rgb[i + 1] = static_cast<uint8_t>((y + offset) % kMaxByte);
      frame.rgb[i + 2] = static_cast<uint8_t>((x + y) % kMaxByte);
    }
  }

  DrawMovingBlock(&frame, offset);
  ++tick_;
  return frame;
}

std::string SyntheticVideoSource::Name() const { return "synthetic"; }

// ---------------------------------------------------------------------------
// SilentAudioSource
// ---------------------------------------------------------------------------

SilentAudioSource::SilentAudioSource(int sample_rate,
                                     int chunk_samples) noexcept
    : sample_rate_(sample_rate), chunk_samples_(chunk_samples) {}

std::vector<float> SilentAudioSource::Read() {
  return std::vector<float>(static_cast<std::size_t>(chunk_samples_), 0.0f);
}

int SilentAudioSource::SampleRate() const { return sample_rate_; }

std::string SilentAudioSource::Name() const { return "silent"; }

}  // namespace app
}  // namespace mep_cctv
