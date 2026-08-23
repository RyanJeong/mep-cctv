// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef SRC_BACKEND_ONNX_ONNX_BACKEND_HPP_
#define SRC_BACKEND_ONNX_ONNX_BACKEND_HPP_

/**
 * @file onnx_backend.hpp
 *
 * OnnxBackend is immutable after construction: model paths and thresholds
 * are set once and never changed.  Every inference call is logically pure
 * (the ORT session state is read-only from this class's perspective; ORT
 * itself maintains internal allocation pools that are an implementation
 * detail).
 *
 * Lifetime contract:
 *   - The ORT environment and session are created at construction; if the
 *     model path is NULL or the file is unreadable, the constructor throws
 *     std::runtime_error and the engine pointer returned by
 *     cctv_engine_create will be NULL.
 *
 * This header deliberately does NOT include any ONNX Runtime header. ORT's
 * C API declares exception specifications in typedefs, which is a hard error
 * under C++14, so every translation unit that sees those headers must be
 * compiled as C++17. Hiding the ORT types behind a pimpl confines that to
 * onnx_backend.cc alone and keeps the rest of the project uniformly C++14.
 */

#include <memory>
#include <stdexcept>
#include <vector>

#include "cctv/cctv.h"
#include "src/core/image_ops.hpp"

namespace mep_cctv {
namespace backend_onnx {

/**
 * @brief ONNX Runtime inference backend (object detection only).
 *
 * The YOLO session is loaded at construction time from disk. Audio is handled
 * by core::DetectAcousticActivity and never reaches this class.
 */
/**
 * @brief Raised when the ONNX Runtime backend cannot be constructed.
 *
 * A dedicated type rather than std::runtime_error so a caller can distinguish
 * "this model or path is unusable" from any other runtime failure that
 * happens to pass through. The C API converts it to a null handle, but the
 * C++ backend is also used directly by the tests.
 */
class OnnxBackendError : public std::runtime_error {
 public:
  // Inherit the base constructors rather than redeclaring one: the class adds
  // no state, so a hand-written forwarder would only be a copy of what
  // runtime_error already provides. This also keeps the const char* overload,
  // which the throw sites use -- forcing them through std::string would build
  // a temporary for a literal that outlives the process.
  using std::runtime_error::runtime_error;
};

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
   * @param tensor     Float array of length kInputElements (1*3*640*640).
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
   * @side_effects_impact       Proportional to output tensor size; ~2.7 MB.
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
