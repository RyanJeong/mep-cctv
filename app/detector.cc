// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file detector.cc
 * @brief Detector implementation.
 */

#include "app/detector.hpp"

#include <string>
#include <vector>

namespace mep_cctv {
namespace app {

const float Detector::kDefaultConfThreshold = 0.25f;
const float Detector::kDefaultIouThreshold = 0.45f;

Detector::Detector(const std::string& yolo_model, float conf, float iou)
    : engine_(nullptr) {
  cctv_config cfg;
  // The native backend ignores the path; passing NULL rather than an empty
  // string is what its contract expects when no model file is involved.
  cfg.yolo_model_path = yolo_model.empty() ? nullptr : yolo_model.c_str();
  cfg.conf_threshold = conf;
  cfg.iou_threshold = iou;
  engine_ = cctv_engine_create(&cfg);
}

Detector::~Detector() {
  if (engine_ != nullptr) cctv_engine_destroy(engine_);
}

bool Detector::IsReady() const { return engine_ != nullptr; }

std::vector<cctv_box> Detector::DetectObjects(const Frame& frame) const {
  std::vector<cctv_box> result;
  if (engine_ == nullptr || !frame.IsValid()) return result;

  cctv_detections out;
  out.boxes = nullptr;
  out.count = 0;
  const cctv_status status = cctv_detect_objects(
      engine_, frame.rgb.data(), frame.width, frame.height, &out);
  if (status != CCTV_OK) return result;

  result.assign(out.boxes, out.boxes + out.count);
  // Ownership of out.boxes transferred to us by the C ABI.
  cctv_free_detections(&out);
  return result;
}

cctv_audio_result Detector::DetectAudio(const std::vector<float>& samples,
                                        int sample_rate) const {
  cctv_audio_result result;
  result.person_present = 0;
  result.score = 0.0f;
  if (engine_ == nullptr || samples.empty() || sample_rate <= 0) return result;

  cctv_detect_audio(engine_, samples.data(), static_cast<int>(samples.size()),
                    sample_rate, &result);

  return result;
}

std::string Detector::BackendName() const {
  const char* name = cctv_backend();
  return name != nullptr ? std::string(name) : std::string();
}

std::string Detector::Version() const {
  const char* version = cctv_version();
  return version != nullptr ? std::string(version) : std::string();
}

bool Detector::SimdEnabled() const { return cctv_simd_enabled() != 0; }

}  // namespace app
}  // namespace mep_cctv
