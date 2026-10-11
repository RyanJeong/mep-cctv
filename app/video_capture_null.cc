// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file video_capture_null.cc
 * @brief No-camera stub for non-Linux targets (macOS, Windows dev hosts).
 *
 * The deployment target is a Raspberry Pi 4 running Linux, where
 * video_capture_v4l2.cc provides real USB webcam capture. On a developer
 * laptop there is no V4L2, so OpenVideoCapture reports "no device" and the
 * caller falls back to SyntheticVideoSource -- the app still runs and the
 * web page still updates, just with a generated pattern instead of a camera.
 */

#include <memory>
#include <string>

#include "app/sources.hpp"

namespace mep_cctv {
namespace app {

std::unique_ptr<VideoSource> OpenVideoCapture(const std::string& device,
                                              int width, int height) {
  (void) device;
  (void) width;
  (void) height;
  return nullptr;
}

}  // namespace app
}  // namespace mep_cctv
