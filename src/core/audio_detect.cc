// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file audio_detect.cc
 * @brief Code-based acoustic activity detection.
 */

#include "src/core/audio_detect.hpp"

#include <cmath>

namespace mep_cctv {
namespace core {

// Chosen empirically: normal room tone on a USB microphone sits well under
// this, while speech or footsteps nearby clear it comfortably.
const float kAudioRmsThreshold = 0.05f;

cctv_audio_result DetectAcousticActivity(const float* samples,
                                         int num_samples) noexcept {
  cctv_audio_result result;
  result.person_present = 0;
  result.score = 0.0f;

  if (samples == nullptr || num_samples <= 0) return result;

  // Remove the DC offset first. Cheap microphones frequently sit on a
  // constant bias; measuring raw RMS would report that steady bias as a
  // roomful of noise and latch the indicator permanently on.
  double sum = 0.0;
  for (int i = 0; i < num_samples; ++i) sum += static_cast<double>(samples[i]);
  const double mean = sum / static_cast<double>(num_samples);

  double sum_sq = 0.0;
  for (int i = 0; i < num_samples; ++i) {
    const double centred = static_cast<double>(samples[i]) - mean;
    sum_sq += centred * centred;
  }

  const double rms = std::sqrt(sum_sq / static_cast<double>(num_samples));
  // Clamp so the reported score stays a sane 0..1 even if the caller hands us
  // samples outside the nominal [-1, 1] range.
  const auto clamped = static_cast<float>(rms > 1.0 ? 1.0 : rms);

  result.score = clamped;
  result.person_present = (clamped >= kAudioRmsThreshold) ? 1 : 0;
  return result;
}

}  // namespace core
}  // namespace mep_cctv
