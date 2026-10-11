// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef APP_DETECTOR_HPP_
#define APP_DETECTOR_HPP_

/**
 * @file detector.hpp
 * @brief RAII wrapper over the libcctv C ABI.
 *
 * The application uses the public C API, not backend implementation headers.
 */

#include <string>
#include <vector>

#include "app/frame.hpp"
#include "cctv/cctv.h"

namespace mep_cctv {
namespace app {

/** @brief Owns a cctv_engine and exposes safe C++ calls over it. */
class Detector {
 public:
  /** @brief Default detection confidence threshold. */
  static const float kDefaultConfThreshold;
  /** @brief Default NMS IoU threshold. */
  static const float kDefaultIouThreshold;

  /**
   * @brief Create an engine.
   *
   * @param yolo_model    Path to the YOLO .onnx; ignored by the native
   *                      backend, required by onnx mode.
   * @param conf          Confidence threshold in (0, 1).
   * @param iou           NMS IoU threshold in (0, 1).
   *
   * @side_effects VIOLATION: Non-referentially transparent
   * @side_effects_reason Allocates the engine and, for onnx mode, reads
   *                      model files from disk.
   * @side_effects_what   Heap allocation; file I/O for onnx.
   * @side_effects_impact O(model size) once at startup.
   * @side_effects_alternatives None -- engine creation is inherently
   *                      effectful.
   */
  Detector(const std::string& yolo_model, float conf, float iou);

  ~Detector();

  Detector(const Detector&) = delete;
  Detector& operator=(const Detector&) = delete;

  /** @brief True when the engine was created successfully. */
  bool IsReady() const;

  /**
   * @brief Run object detection on a frame.
   *
   * @param frame  Frame to analyse.
   * @return Detected boxes in the frame's pixel space; empty on failure.
   *
   * @side_effects VIOLATION: Non-referentially transparent
   * @side_effects_reason Calls into the engine, which allocates a detection
   *                      array that this method takes ownership of and frees.
   * @side_effects_what   Heap allocation inside libcctv, released here.
   * @side_effects_impact Proportional to the detection count.
   * @side_effects_alternatives The C ABI transfers ownership by design.
   */
  std::vector<cctv_box> DetectObjects(const Frame& frame) const;

  /**
   * @brief Detect acoustic activity ("is something making noise nearby?").
   *
   * @param samples      Mono PCM in [-1, 1].
   * @param sample_rate  Sample rate in Hz.
   * @return Result with person_present zeroed on failure.
   *
   * @side_effects None -- referentially transparent for a fixed engine.
   */
  cctv_audio_result DetectAudio(const std::vector<float>& samples,
                                int sample_rate) const;

  /** @brief Mode reported by the library ("native" or "onnx"). */
  std::string BackendName() const;

  /** @brief Library version string. */
  std::string Version() const;

  /** @brief True when SIMD kernels were compiled in. */
  bool SimdEnabled() const;

 private:
  cctv_engine* engine_;
};

}  // namespace app
}  // namespace mep_cctv

#endif  // APP_DETECTOR_HPP_
