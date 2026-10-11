// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file audio_capture_null.cc
 * @brief No-microphone stub, used when ALSA is unavailable.
 *
 * Selected by CMake for non-Linux targets, and also for Linux builds where
 * libasound was not found. Callers fall back to SilentAudioSource.
 */

#include <memory>
#include <string>

#include "app/sources.hpp"

namespace mep_cctv {
namespace app {

std::unique_ptr<AudioSource> OpenAudioCapture(const std::string& device,
                                              int sample_rate, int chunk) {
  (void) device;
  (void) sample_rate;
  (void) chunk;
  return nullptr;
}

}  // namespace app
}  // namespace mep_cctv
