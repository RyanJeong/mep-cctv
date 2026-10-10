// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef SRC_BACKEND_ONNX_ONNX_BACKEND_HPP_
#define SRC_BACKEND_ONNX_ONNX_BACKEND_HPP_

/**
 * @file onnx_backend.hpp
 *
 * Model paths and thresholds are fixed at construction. Inference allocates
 * outputs and updates ONNX Runtime's internal session state.
 *
 * Lifetime contract:
 *   - The ORT environment and session are created at construction; if the
 *     model path is NULL or the file is unreadable, the constructor throws
 *     std::runtime_error and the engine pointer returned by
 *     cctv_engine_create will be NULL.
 *
 * ONNX Runtime types remain behind a pimpl to isolate dependency headers.
 */

#include <memory>
#include <stdexcept>
#include <vector>

#include "cctv/cctv.h"
#include "src/core/image_ops.hpp"

namespace mep_cctv {
namespace backend_onnx {

/**
 * @brief Raised when the ONNX Runtime backend cannot be constructed.
 *
 * The C API converts construction failures to a null engine handle.
 */
class OnnxBackendError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/** @brief Loads an ONNX object detector; audio uses the shared RMS detector. */
class OnnxBackend {
 public:
  /**
   * @brief Construct and load the YOLO ONNX model.
   *
   * @param yolo_path   Path to the YOLOv8 ONNX model file.
   * @param conf_threshold Confidence threshold for YOLO detections.
   * @param iou_threshold  IoU threshold for NMS.
   *
   * @side_effects VIOLATION: Non-referentially transparent
   * @side_effects_reason       Loads model files from disk; allocates ORT
   *                            sessions.
   * @side_effects_what         File I/O; ORT heap allocation for model weights.
   * @side_effects_impact       O(model-size) startup cost; one-time.
   * @side_effects_alternatives A lazy-load strategy is possible but adds
   *                            thread-safety complexity for no benefit here.
   */
  OnnxBackend(const char* yolo_path, float conf_threshold, float iou_threshold);

  /**
   * @brief Releases the ONNX Runtime session.
   * @note Declared here and defined in the .cc because the pimpl type is
   *       incomplete in this header.
   */
  ~OnnxBackend();

  OnnxBackend(const OnnxBackend&) = delete;
  OnnxBackend& operator=(const OnnxBackend&) = delete;

  /**
   * @brief Run YOLO inference on a preprocessed float NCHW tensor.
   *
   * @param tensor     Float array of length core::kInputElements in NCHW order.
   * @param params     Letterbox parameters used during preprocessing.
   * @param orig_w     Original image width in pixels.
   * @param orig_h     Original image height in pixels.
   * @return Decoded, NMS-filtered detections in original image space.
   *
   * @side_effects VIOLATION: Non-referentially transparent
   * @side_effects_reason       ORT Run() allocates output tensors and mutates
   *                            internal session arena state
   *                            (non-const by ORT API).
   * @side_effects_what         Heap allocation inside ORT session; arena
   *                            mutation.
   * @side_effects_impact       Proportional to output tensor size.
   * @side_effects_alternatives Output pre-allocation via IoBinding would
   *                            avoid this but increases API complexity.
   */
  // NOTE: Detect is non-const because Ort::Session::Run is non-const in
  // ORT 1.27 (it mutates internal arena allocation state).
  std::vector<cctv_box> Detect(const float* tensor,
                               const core::LetterboxParams& params, int orig_w,
                               int orig_h) noexcept;

 private:
  /** @brief Holds the ONNX Runtime environment, session and cached names. */
  class Impl;

  const float conf_threshold_;
  const float iou_threshold_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace backend_onnx
}  // namespace mep_cctv

#endif  // SRC_BACKEND_ONNX_ONNX_BACKEND_HPP_
